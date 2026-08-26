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
    reduce_mechanistic_responses,
    violation_record,
)
from closed_loop.model import (
    ArticleOperatingPoint,
    INVARIANT_MATRIX,
    TSS_VECTOR,
    assemble_target,
    solve_steady_state,
)
from closed_loop.projection import (
    LogOverflowTSSClosure,
    NetworkLayout,
    QuadraticSurrogate,
)
from closed_loop.v3_reporting import write_reporting_tables
from closed_loop.v3_smooth import (
    DirectCase,
    SolverSettings,
    compare_smooth_reference,
    fit_direct_assets,
    solve_direct_multistart,
)
from closed_loop.v3_surrogate_nlp import (
    SurrogateCase,
    SurrogateSolverSettings,
    build_surrogate_assets,
    ordered_normalized_starts as surrogate_normalized_starts,
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
SMOKE_START_COUNT = 1


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


def _load_inputs() -> tuple[
    dict[str, object], np.ndarray, np.ndarray, np.ndarray,
    QuadraticSurrogate, np.ndarray, LogOverflowTSSClosure, np.ndarray,
]:
    design = create_design(TEST_500)
    with np.load(RUN / "datasets/development/mechanistic_rows_v3.npz", allow_pickle=False) as data:
        mechanistic_targets = np.asarray(data["targets"], dtype=float)
    targets = reduce_mechanistic_responses(
        mechanistic_targets, TEST_500.layer_count,
    )
    scores = pd.read_csv(RUN / "metrics/ridge_cross_validation.csv")
    gamma = float(scores.loc[scores["selected"].astype(bool), "gamma"].iloc[0])
    model = QuadraticSurrogate.fit_ridge(
        design["development_decisions"], design["development_influents"], targets,
        ridge_penalty=gamma,
    )
    with np.load(RUN / "models/ridge_surrogate.npz", allow_pickle=False) as data:
        oof_raw = np.asarray(data["out_of_fold_raw"], dtype=float)
        stored_scale = np.asarray(data["response_scale"], dtype=float)
    if oof_raw.shape != targets.shape or stored_scale.shape != (TEST_500.surrogate_response_count,):
        raise RuntimeError(
            "the ridge checkpoint uses the superseded layer-output response; "
            "rerun reduced-response ridge selection before optimization"
        )
    closure_path = RUN / "models/log_overflow_closure.npz"
    if not closure_path.is_file():
        raise RuntimeError(
            "the required log-overflow-TSS closure checkpoint is unavailable; "
            "rerun the assessment phase before optimization"
        )
    with np.load(closure_path, allow_pickle=False) as data:
        closure_arrays = {name: np.asarray(data[name]) for name in data.files}
    overflow_closure = LogOverflowTSSClosure.from_serialized_arrays(closure_arrays)
    oof_overflow_tss = np.asarray(closure_arrays["out_of_fold_tss"], dtype=float)
    if oof_overflow_tss.shape != (len(targets),) or np.any(oof_overflow_tss <= 0.0):
        raise RuntimeError("the log-overflow-TSS closure checkpoint is inconsistent")
    return (
        design, mechanistic_targets, targets, oof_raw, model, np.asarray(gamma),
        overflow_closure, oof_overflow_tss,
    )


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


def _shared_response(response: np.ndarray) -> np.ndarray:
    """Map a saved mechanistic response to the common reduced contract."""

    values = np.asarray(response, dtype=float)
    if values.shape == (TEST_500.surrogate_response_count,):
        return values
    return reduce_mechanistic_responses(values, TEST_500.layer_count)


def main(case_limit: int) -> None:
    (
        design, mechanistic_targets, targets, oof_raw, model, _,
        overflow_closure, oof_overflow_tss,
    ) = _load_inputs()
    layout = NetworkLayout(layer_count=TEST_500.layer_count)
    direct_assets = fit_direct_assets(
        design["development_decisions"], design["development_influents"],
        mechanistic_targets,
        clarifier=clarifier_for(TEST_500),
    )
    trust = calibrate_trust_diagnostics(
        model, design["development_decisions"], design["development_influents"],
        targets, oof_raw, direct_assets, layout=layout,
        overflow_closure=overflow_closure,
        out_of_fold_overflow_tss=oof_overflow_tss,
    )
    surrogate_assets = build_surrogate_assets(
        model, design["development_decisions"], design["development_influents"], targets,
        layout=layout, invariant_operator=INVARIANT_MATRIX, tss_weights=TSS_VECTOR,
        correction_rms_threshold=trust.correction_limit,
        trust_callbacks=trust.callbacks,
        split_rms_threshold=trust.split_limit,
        reactor_rms_threshold=trust.reactor_limit,
        overflow_closure=overflow_closure,
        development_overflow_tss_closure=oof_overflow_tss,
    )
    features = model.feature_map.transform(
        design["development_decisions"], design["development_influents"],
    )
    leverage = np.einsum(
        "ij,jk,ik->i", features, surrogate_assets.leverage_precision, features,
    )
    trust_values = np.column_stack((
        trust.development_values[:, 0], leverage, trust.development_values[:, 1:],
    ))
    trust_table = pd.DataFrame(
        trust_values,
        columns=[
            "correction", "regularized_leverage", "particulate_split",
            "reactor_residual",
        ],
    )
    trust_table.to_csv(RUN / "metrics/trust_development_oof.csv", index=False)
    _write_json(RUN / "metrics/trust_limits.json", {
        "correction": trust.correction_limit,
        "regularized_leverage": surrogate_assets.trust_thresholds.regularized_leverage,
        "particulate_split": trust.split_limit,
        "reactor_residual": trust.reactor_limit,
        "correction_gate_at_most_0_50": trust.correction_limit <= 0.50,
    })
    print("trust limits", (
        trust.correction_limit,
        surrogate_assets.trust_thresholds.regularized_leverage,
        trust.split_limit,
        trust.reactor_limit,
    ))

    case_inputs = [("nominal", NOMINAL)] + [
        (f"robustness_{index + 1:02d}", row)
        for index, row in enumerate(design["robustness_influents"])
    ]
    case_root = RUN / "optimization"
    case_root.mkdir(parents=True, exist_ok=True)
    smoke_starts = surrogate_normalized_starts()[:SMOKE_START_COUNT]
    violations: list[dict[str, object]] = []
    for case_id, influent in case_inputs[:case_limit]:
        case_directory = case_root / case_id
        case_directory.mkdir(parents=True, exist_ok=True)
        surrogate_path = case_directory / "surrogate.json"
        direct_path = case_directory / "direct.json"
        print(f"[{case_id}] surrogate route", flush=True)
        started = perf_counter()
        partial_surrogate: list[dict[str, Any]] = []

        def checkpoint_surrogate(item: Any) -> None:
            partial_surrogate.append(item.as_dict())
            _write_json(case_directory / "surrogate_starts.partial.json", partial_surrogate)
            print(
                f"[{case_id}] surrogate start {item.start_index + 1}/{SMOKE_START_COUNT}: {item.status}",
                flush=True,
            )

        cached_surrogate = None
        if surrogate_path.is_file():
            candidate = json.loads(surrogate_path.read_text(encoding="utf-8"))
            if candidate.get("preflight_stage_wall_time_seconds") == 600.0:
                cached_surrogate = candidate
        if cached_surrogate is None:
            surrogate = solve_surrogate_multistart(
                surrogate_assets, SurrogateCase(influent=np.asarray(influent), case_id=case_id),
                settings=SurrogateSolverSettings(maximum_wall_time=600.0),
                starts=smoke_starts,
                name="preflight_surrogate",
                progress_callback=checkpoint_surrogate,
                allow_reduced_starts=True,
            )
            surrogate_payload = surrogate.as_dict()
            surrogate_payload["elapsed_seconds"] = perf_counter() - started
            surrogate_payload["preflight_stage_wall_time_seconds"] = 600.0
            surrogate_payload["smoke_test_start_count"] = SMOKE_START_COUNT
            _write_json(surrogate_path, surrogate_payload)
            surrogate_selected = (
                surrogate.selected.final
                if surrogate.selected is not None and surrogate.selected.final is not None
                else None
            )
        else:
            surrogate_payload = cached_surrogate
            surrogate_selected = None
            selected_file = case_directory / "surrogate_selected.npz"
            if selected_file.is_file():
                with np.load(selected_file, allow_pickle=False) as stored:
                    selected_theta = np.asarray(stored["theta"], dtype=float)
                    selected_raw = np.asarray(stored["raw"], dtype=float)
                    selected_projected = np.asarray(stored["projected"], dtype=float)
                for method, response in (("raw", selected_raw), ("projected", selected_projected)):
                    violations.append(violation_record(
                        method, f"{case_id}:surrogate", response, selected_theta, influent,
                        layout, surrogate_assets.row_scales.equality,
                        surrogate_assets.row_scales.inequality, model.response_scale,
                        overflow_tss_closure=float(
                            overflow_closure.predict(selected_theta, influent)
                        ),
                    ))
        if cached_surrogate is None and surrogate_selected is not None:
            final = surrogate_selected
            np.savez_compressed(
                case_directory / "surrogate_selected.npz", theta=final.theta,
                raw=final.raw, projected=final.projected,
            )
            for method, response in (("raw", final.raw), ("projected", final.projected)):
                violations.append(violation_record(
                    method, f"{case_id}:surrogate", response, final.theta, influent,
                    layout, surrogate_assets.row_scales.equality,
                    surrogate_assets.row_scales.inequality, model.response_scale,
                    overflow_tss_closure=float(
                        overflow_closure.predict(final.theta, influent)
                    ),
                ))

        print(f"[{case_id}] direct route", flush=True)
        started = perf_counter()
        direct_selected_path = case_directory / "direct_selected.npz"
        if direct_path.is_file() and direct_selected_path.is_file():
            direct_payload = json.loads(direct_path.read_text(encoding="utf-8"))
            with np.load(direct_selected_path, allow_pickle=False) as stored:
                direct_theta = np.asarray(stored["theta"], dtype=float)
                direct_response = np.asarray(stored["response"], dtype=float)
                direct_state = np.asarray(stored["state"], dtype=float)
        else:
            direct = solve_direct_multistart(
                direct_assets, DirectCase(influent=np.asarray(influent), case_id=case_id),
                design["development_decisions"], design["development_influents"],
                mechanistic_targets,
                settings=SolverSettings(maximum_wall_time=600.0),
                starts=smoke_starts,
                allow_reduced_starts=True,
            )
            direct_payload = _direct_summary(direct)
            direct_payload["elapsed_seconds"] = perf_counter() - started
            direct_payload["preflight_stage_wall_time_seconds"] = 600.0
            direct_payload["smoke_test_start_count"] = SMOKE_START_COUNT
            _write_json(direct_path, direct_payload)
            if direct.selected is None:
                direct_theta = direct_response = direct_state = None
            else:
                direct_theta = direct.selected.theta
                direct_response = direct.selected.response
                direct_state = direct.selected.state
        if direct_theta is not None and direct_response is not None and direct_state is not None:
            np.savez_compressed(
                direct_selected_path, theta=direct_theta,
                response=direct_response, state=direct_state,
            )
            violations.append(violation_record(
                "smooth", f"{case_id}:direct", _shared_response(direct_response), direct_theta,
                influent, layout, surrogate_assets.row_scales.equality,
                surrogate_assets.row_scales.inequality, model.response_scale,
                overflow_tss_closure=float(
                    overflow_closure.predict(direct_theta, influent)
                ),
            ))
            equivalence = compare_smooth_reference(
                direct_theta, influent, direct_assets,
            )
            _write_json(case_directory / "direct_equivalence.json", asdict(equivalence))
            operating = ArticleOperatingPoint(*map(float, direct_theta))
            reference = solve_steady_state(
                operating, influent, starts=(1,), clarifier=direct_assets.clarifier,
                logarithmic_only=True, strict_v3=True,
            )
            response = assemble_target(reference.state, operating, influent, direct_assets.clarifier)
            np.savez_compressed(
                case_directory / "direct_reference.npz", theta=direct_theta,
                response=response, state=reference.state,
            )
            violations.append(violation_record(
                "reference", f"{case_id}:direct", _shared_response(response), direct_theta,
                influent, layout, surrogate_assets.row_scales.equality,
                surrogate_assets.row_scales.inequality, model.response_scale,
                overflow_tss_closure=float(
                    overflow_closure.predict(direct_theta, influent)
                ),
            ))
        pd.DataFrame(violations).to_csv(
            RUN / "metrics/physical_violations_selected_cases.csv", index=False,
        )
        write_reporting_tables(RUN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-limit", type=int, default=6, choices=range(1, 7))
    arguments = parser.parse_args()
    main(arguments.case_limit)
