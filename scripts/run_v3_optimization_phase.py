"""Resume the manuscript-v3 preflight at trust calibration and optimization."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from closed_loop.manuscript_v3 import (
    TEST_500,
    clarifier_for,
    create_design,
    violation_record,
)
from closed_loop.model import (
    ArticleOperatingPoint,
    INVARIANT_MATRIX,
    TSS_VECTOR,
    assemble_target,
    solve_steady_state,
)
from closed_loop.projection import NetworkLayout, QuadraticSurrogate
from closed_loop.v3_smooth import (
    DirectCase,
    compare_smooth_reference,
    fit_direct_assets,
    solve_direct_multistart,
)
from closed_loop.v3_surrogate_nlp import (
    SurrogateCase,
    build_surrogate_assets,
    solve_surrogate_multistart,
)
from closed_loop.v3_trust import calibrate_trust_diagnostics


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "article_v3" / "test_500_l5_revision_001"
NOMINAL = (
    np.asarray([0.0, 20.0, 5.0, 12.0, 0.0, 0.0, 0.0, 2.0, 10.0, 1.6,
                20.0, 60.0, 15.0, 5.0, 2.0, 1.0, 0.5, 0.5, 0.0, 0.0])
    + np.asarray([0.5, 180.0, 80.0, 55.0, 3.0, 8.0, 2.0, 18.0, 90.0, 5.2,
                  120.0, 280.0, 100.0, 60.0, 20.0, 30.0, 8.0, 8.0, 12.0, 12.0])
) / 2.0


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _load_inputs() -> tuple[dict[str, object], np.ndarray, np.ndarray, QuadraticSurrogate, np.ndarray]:
    design = create_design(TEST_500)
    with np.load(RUN / "datasets/development/mechanistic_rows_v3.npz", allow_pickle=False) as data:
        targets = np.asarray(data["targets"], dtype=float)
    scores = pd.read_csv(RUN / "metrics/ridge_cross_validation.csv")
    gamma = float(scores.loc[scores["selected"].astype(bool), "gamma"].iloc[0])
    model = QuadraticSurrogate.fit_ridge(
        design["development_decisions"], design["development_influents"], targets,
        ridge_penalty=gamma,
    )
    with np.load(RUN / "models/ridge_surrogate.npz", allow_pickle=False) as data:
        oof_raw = np.asarray(data["out_of_fold_raw"], dtype=float)
    return design, np.asarray(targets), oof_raw, model, np.asarray(gamma)


def _direct_summary(result: Any) -> dict[str, Any]:
    selected = result.selected
    return {
        "status": result.status,
        "selected_start": None if selected is None else selected.start_index,
        "starts": [
            {
                "start_index": item.start_index,
                "nearest_development_row": item.nearest_development_row,
                "status": item.status,
                "objective": item.objective,
                "feasible": item.feasible,
                "stationary": item.stationary,
                "theta": item.theta,
                "stages": [asdict(stage) for stage in item.stages],
                "kkt": None if item.kkt is None else asdict(item.kkt),
            }
            for item in result.starts
        ],
    }


def main(case_limit: int) -> None:
    design, targets, oof_raw, model, _ = _load_inputs()
    layout = NetworkLayout(layer_count=TEST_500.layer_count)
    direct_assets = fit_direct_assets(
        design["development_decisions"], design["development_influents"], targets,
        clarifier=clarifier_for(TEST_500),
    )
    trust = calibrate_trust_diagnostics(
        model, design["development_decisions"], design["development_influents"],
        targets, oof_raw, direct_assets, layout=layout,
    )
    surrogate_assets = build_surrogate_assets(
        model, design["development_decisions"], design["development_influents"], targets,
        layout=layout, invariant_operator=INVARIANT_MATRIX, tss_weights=TSS_VECTOR,
        correction_rms_threshold=trust.correction_limit,
        trust_callbacks=trust.callbacks,
        split_rms_threshold=trust.split_limit,
        reactor_rms_threshold=trust.reactor_limit,
        flux_rms_threshold=trust.flux_limit,
    )
    trust_table = pd.DataFrame(
        trust.development_values,
        columns=["correction", "particulate_split", "reactor_residual", "clarifier_flux"],
    )
    trust_table.to_csv(RUN / "metrics/trust_development_oof.csv", index=False)
    _write_json(RUN / "metrics/trust_limits.json", {
        "correction": trust.correction_limit,
        "regularized_leverage": surrogate_assets.trust_thresholds.regularized_leverage,
        "particulate_split": trust.split_limit,
        "reactor_residual": trust.reactor_limit,
        "clarifier_flux": trust.flux_limit,
        "correction_gate_at_most_0_50": trust.correction_limit <= 0.50,
    })
    print("trust limits", (trust.correction_limit, trust.split_limit,
                            trust.reactor_limit, trust.flux_limit))

    case_inputs = [("nominal", NOMINAL)] + [
        (f"robustness_{index + 1:02d}", row)
        for index, row in enumerate(design["robustness_influents"])
    ]
    case_root = RUN / "optimization"
    case_root.mkdir(parents=True, exist_ok=True)
    violations: list[dict[str, object]] = []
    for case_id, influent in case_inputs[:case_limit]:
        case_directory = case_root / case_id
        case_directory.mkdir(parents=True, exist_ok=True)
        surrogate_path = case_directory / "surrogate.json"
        direct_path = case_directory / "direct.json"
        print(f"[{case_id}] surrogate route", flush=True)
        started = perf_counter()
        surrogate = solve_surrogate_multistart(
            surrogate_assets, SurrogateCase(influent=np.asarray(influent), case_id=case_id),
            name="preflight_surrogate",
        )
        surrogate_payload = surrogate.as_dict()
        surrogate_payload["elapsed_seconds"] = perf_counter() - started
        _write_json(surrogate_path, surrogate_payload)
        if surrogate.selected is not None and surrogate.selected.final is not None:
            final = surrogate.selected.final
            np.savez_compressed(
                case_directory / "surrogate_selected.npz", theta=final.theta,
                raw=final.raw, projected=final.projected,
            )
            for method, response in (("raw", final.raw), ("projected", final.projected)):
                violations.append(violation_record(
                    method, f"{case_id}:surrogate", response, final.theta, influent,
                    layout, surrogate_assets.row_scales.equality,
                    surrogate_assets.row_scales.inequality, model.response_scale,
                ))

        print(f"[{case_id}] direct route", flush=True)
        started = perf_counter()
        direct = solve_direct_multistart(
            direct_assets, DirectCase(influent=np.asarray(influent), case_id=case_id),
            design["development_decisions"], design["development_influents"], targets,
        )
        direct_payload = _direct_summary(direct)
        direct_payload["elapsed_seconds"] = perf_counter() - started
        _write_json(direct_path, direct_payload)
        if direct.selected is not None:
            selected = direct.selected
            np.savez_compressed(
                case_directory / "direct_selected.npz", theta=selected.theta,
                response=selected.response, state=selected.state,
            )
            violations.append(violation_record(
                "smooth", f"{case_id}:direct", selected.response, selected.theta,
                influent, layout, surrogate_assets.row_scales.equality,
                surrogate_assets.row_scales.inequality, model.response_scale,
                nonlinear_state=selected.state,
            ))
            equivalence = compare_smooth_reference(
                selected.theta, influent, direct_assets,
            )
            _write_json(case_directory / "direct_equivalence.json", asdict(equivalence))
            operating = ArticleOperatingPoint(*map(float, selected.theta))
            reference = solve_steady_state(
                operating, influent, starts=(1,), clarifier=direct_assets.clarifier,
                logarithmic_only=True, strict_v3=True,
            )
            response = assemble_target(reference.state, operating, influent, direct_assets.clarifier)
            np.savez_compressed(
                case_directory / "direct_reference.npz", theta=selected.theta,
                response=response, state=reference.state,
            )
            violations.append(violation_record(
                "reference", f"{case_id}:direct", response, selected.theta,
                influent, layout, surrogate_assets.row_scales.equality,
                surrogate_assets.row_scales.inequality, model.response_scale,
                nonlinear_state=reference.state,
            ))
        pd.DataFrame(violations).to_csv(
            RUN / "metrics/physical_violations_selected_cases.csv", index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-limit", type=int, default=6, choices=range(1, 7))
    arguments = parser.parse_args()
    main(arguments.case_limit)
