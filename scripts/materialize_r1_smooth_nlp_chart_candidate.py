"""Materialize the archived R1 Smooth NLP endpoint for reporting charts only.

The R1 direct solve completed its final NLP stage but did not retain a selected
candidate after a separate multiplier-reconstruction audit exception.  This
utility evaluates that archived final-stage primal point and its exact replay,
then writes a chart-only casewise-reference input.  It never changes the run's
scientific optimization artifacts, contracts, or validation status.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from closed_loop.v3_smooth import (
    DirectCase,
    SolverSettings,
    build_direct_nlp,
    evaluate_direct,
    fit_direct_assets,
)
from scripts.run_article_v3_5000 import (
    CONTINUATION_SCHEDULE,
    _casewise_exact_reference,
    reduce_mechanistic_responses,
)


CASE = "robustness_01"
ROUTE = "direct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()

    with np.load(run / "datasets" / "effective_design.npz", allow_pickle=False) as stored:
        development_decisions = np.asarray(stored["development_decisions"], dtype=float)
        development_influents = np.asarray(stored["development_influents"], dtype=float)
        influent = np.asarray(stored["robustness_influents"][0], dtype=float)
    with np.load(
        run / "datasets" / "development" / "mechanistic_accepted_v3.npz",
        allow_pickle=False,
    ) as stored:
        development_targets = np.asarray(stored["targets"], dtype=float)

    archived_path = run / "optimization" / CASE / "direct.json"
    archived = json.loads(archived_path.read_text(encoding="utf-8"))
    starts = archived.get("starts", [])
    if len(starts) != 1:
        raise RuntimeError("R1 direct archive must contain exactly one center start")
    stages = starts[0].get("stages", [])
    if not stages:
        raise RuntimeError("R1 direct archive does not contain continuation stages")
    primal = np.asarray(stages[-1]["primal"], dtype=float)
    epsilon, receiver_half_width = CONTINUATION_SCHEDULE[-1]

    assets = fit_direct_assets(
        development_decisions,
        development_influents,
        development_targets,
    )
    problem = build_direct_nlp(
        assets,
        epsilon=epsilon,
        receiver_half_width=receiver_half_width,
        settings=SolverSettings(maximum_wall_time=None),
        name="reporting_r1_direct_archived_endpoint",
        compile_solver=False,
    )
    evaluated = evaluate_direct(problem, primal, DirectCase(influent=influent, case_id=CASE))
    theta = np.asarray(evaluated["theta"], dtype=float)
    native_full = np.asarray(evaluated["response"], dtype=float)
    native = reduce_mechanistic_responses(native_full, assets.clarifier.layer_count)

    analysis = SimpleNamespace(direct_assets=assets)
    reference_full, state_1, state_2, reference = _casewise_exact_reference(
        theta,
        influent,
        analysis,
    )
    reference_reduced = reduce_mechanistic_responses(
        reference_full,
        assets.clarifier.layer_count,
    )
    if not (
        np.all(np.isfinite(native))
        and np.all(np.isfinite(reference_reduced))
        and np.all(np.isfinite(reference_full))
    ):
        raise RuntimeError("R1 reporting candidate produced a non-finite response")

    output = run / "report" / "chart_overrides"
    output.mkdir(parents=True, exist_ok=True)
    arrays_path = output / f"{CASE}_{ROUTE}_casewise_reference.npz"
    np.savez_compressed(
        arrays_path,
        theta=theta,
        normalized_controls=np.asarray(evaluated["normalized_controls"], dtype=float),
        optimizer_native=native,
        optimizer_native_full=native_full,
        exact_reference=reference_reduced,
        exact_reference_full=np.asarray(reference_full, dtype=float),
        exact_state_start_1=np.asarray(state_1, dtype=float),
        exact_state_start_2=np.asarray(state_2, dtype=float),
    )
    metadata = {
        "purpose": "chart_reporting_override",
        "case": CASE,
        "route": ROUTE,
        "source": "archived final continuation-stage primal in optimization/robustness_01/direct.json",
        "native_objective": float(evaluated["objective"]),
        "exact_reference": reference,
        "note": "Materialized for the user-requested normal chart comparison; scientific run validation artifacts are unchanged.",
    }
    arrays_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(arrays_path)


if __name__ == "__main__":
    main()
