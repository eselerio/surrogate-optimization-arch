"""Re-adjudicate a completed article-v3 run without a minimum-SRT gate.

The source run is read-only. The script writes a small, provenance-linked
post-processing run containing revised comparison rows for chart generation.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from closed_loop.model import (
    ClarifierParameters,
    INFLUENT_LOWER,
    INFLUENT_UPPER,
    N_COMPONENTS,
    N_STAGES,
)
from closed_loop.v3_smooth import (
    DECISION_LOWER,
    DECISION_UPPER,
    DirectAssets,
    SmoothScales,
    engineering_feasible,
    engineering_quantities,
)
from scripts.run_article_v3_5000 import (
    COMPARISON_PROTOCOL,
    OPTIMIZATION_PROTOCOL,
    _casewise_comparison_row,
)


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_assets(layer_count: int) -> DirectAssets:
    state_count = N_STAGES * N_COMPONENTS + layer_count
    return DirectAssets(
        clarifier=ClarifierParameters(
            layer_count=layer_count,
            feed_layer=(layer_count - 1) // 2,
            layer_volume=6_000.0 / layer_count,
        ),
        smoothing=SmoothScales(
            10.0, 100.0, 100.0, 100.0, 100.0,
            100.0, 100.0, 250.0, 10_000.0,
        ),
        state_center=np.ones(state_count),
        state_scale=np.ones(state_count),
        feed_scale=100.0,
        balance_scale=np.ones(state_count),
        quality_scale=np.ones(4),
        envelope_scale=np.ones(2 * (layer_count - 2)),
        engineering_scale=np.ones(4),
        decision_center=(DECISION_LOWER + DECISION_UPPER) / 2.0,
        decision_scale=(DECISION_UPPER - DECISION_LOWER) / np.sqrt(12.0),
        influent_center=(INFLUENT_LOWER + INFLUENT_UPPER) / 2.0,
        influent_scale=(INFLUENT_UPPER - INFLUENT_LOWER) / np.sqrt(12.0),
    )


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_run", type=Path)
    parser.add_argument("output_run", type=Path)
    args = parser.parse_args()
    source = args.source_run.resolve()
    output = args.output_run.resolve()
    if source == output:
        raise ValueError("the post-processing output must not overwrite its source run")
    if not (source / "inputs/contract.json").is_file():
        raise FileNotFoundError("source run has no source-bound contract")

    cases = ["nominal", *[f"robustness_{index:02d}" for index in range(1, 11)]]
    assets = _audit_assets(layer_count=10)
    rows: list[dict[str, Any]] = []
    route_audit_rows: list[dict[str, Any]] = []
    source_artifacts: dict[str, str] = {}
    for case in cases:
        payloads: list[dict[str, Any]] = []
        for route in ("surrogate", "direct"):
            json_path = source / "optimization" / case / f"{route}_casewise_reference.json"
            arrays_path = source / "optimization" / case / f"{route}_casewise_reference.npz"
            payload = _json(json_path)
            with np.load(arrays_path, allow_pickle=False) as stored:
                theta = np.asarray(stored["theta"], dtype=float)
                exact = np.asarray(stored["exact_reference_full"], dtype=float)
            retained_feasible = engineering_feasible(theta, exact, assets)
            quantities = engineering_quantities(theta, exact, assets)
            payload["comparison_valid"] = bool(
                payload.get("exact_replay_valid") is True and retained_feasible
            )
            reference = payload.get("reference")
            if isinstance(reference, dict):
                reference["engineering_feasible"] = retained_feasible
            payloads.append(payload)
            route_audit_rows.append({
                "case": case,
                "route": route,
                "srt_d": quantities["srt_d"],
                "retained_engineering_feasible": retained_feasible,
                "minimum_srt_eligibility_gate": False,
            })
            for path in (json_path, arrays_path):
                source_artifacts[path.relative_to(source).as_posix()] = _digest(path)
        rows.append(_casewise_comparison_row(case, *payloads))

    comparison = pd.DataFrame(rows)
    if not bool(comparison["comparison_eligible"].all()):
        failures = comparison.loc[
            ~comparison["comparison_eligible"], ["case", "ineligibility_reasons"]
        ]
        raise RuntimeError(f"retained checks still exclude cases:\n{failures}")

    metrics = output / "metrics"
    inputs = output / "inputs"
    figures = output / "report" / "figures" / "system_surrogate_vs_smooth_nlp"
    for directory in (metrics, inputs, figures):
        directory.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(metrics / "case_common_reference_comparison.csv", index=False)
    pd.DataFrame(route_audit_rows).to_csv(
        metrics / "retained_engineering_replay_audit.csv", index=False,
    )
    source_comparison = source / "metrics" / "case_common_reference_comparison.csv"
    source_artifacts[source_comparison.relative_to(source).as_posix()] = _digest(
        source_comparison
    )
    manifest = {
        "schema": "article-v3-no-minimum-srt-postprocessing-v1",
        "source_run_id": source.name,
        "output_run_id": output.name,
        "source_run_unchanged": True,
        "minimum_srt_eligibility_gate": False,
        "srt_remains_reported": True,
        "optimization_protocol_for_future_runs": OPTIMIZATION_PROTOCOL,
        "comparison_protocol": COMPARISON_PROTOCOL,
        "comparison_case_count": len(comparison),
        "comparison_eligible_count": int(comparison["comparison_eligible"].sum()),
        "postprocessor_sha256": _digest(Path(__file__).resolve()),
        "source_artifacts": dict(sorted(source_artifacts.items())),
    }
    (inputs / "postprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "run_state.json").write_text(
        json.dumps(
            {"stage": "no_minimum_srt_postprocessing", "status": "complete"},
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(metrics / "case_common_reference_comparison.csv")


if __name__ == "__main__":
    main()
