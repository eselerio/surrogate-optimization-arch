"""Build the canonical, resumable manuscript-v3 execution notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "main_closed_loop.ipynb"


def code(source: str):
    """Return an unexecuted code cell with normalized trailing whitespace."""

    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str):
    """Return a markdown cell with normalized trailing whitespace."""

    return nbf.v4.new_markdown_cell(source.strip() + "\n")


cells = [
    markdown(r"""
# Manuscript-v3 closed-loop article run

This notebook is the canonical, resumable interface to
`scripts/run_article_v3_5000.py`. The production driver, rather than duplicated
notebook code, owns candidate generation and deterministic replacement, two-start
mechanistic acceptance, ridge selection,
untouched-test surrogate assessment, scientific admission gates, paired optimization,
casewise exact-reference replay, convergence and physical audits, and
Results/Discussion tables.

The full model-function workload is fixed at **5,000 accepted datasets**: 4,000
development inputs and 1,000 untouched test inputs. Rejected mechanistic
candidates remain fully audited but are excluded and deterministically
replaced. It uses ten robustness cases plus the nominal case and a ten-layer
Clarifier. Each route uses one deterministic box-center start per case and
searches only that basin without claiming global optimality. The surrogate
route primarily runs the seven-variable exact-QP active-set optimizer. If
exact active-set derivatives are unavailable, deterministic value-only COBYQA
cold-solves the unchanged projection QP at every distinct fallback trial and
retains the best feasible visited incumbent. A budget-limited or
stationarity-unresolved incumbent is not called a local optimum. Each selected
surrogate endpoint then receives a bounded two-scale exact-QP feasible poll;
this can support a finite-direction, finite-resolution convergence
qualification without claiming differentiable KKT stationarity. The surrogate
route does not execute the retired embedded-KKT IPOPT problem or any of its
seven gap-continuation stages. The direct smooth-mechanistic route retains its
separate three-stage smoothing continuation. There is no wall-time ceiling in
the full article run. The scientific admission thresholds remain unchanged and still
determine whether results are article-eligible. For this model-function
exercise they are advisory for execution: failures are recorded and propagated
while later stages are attempted without refitting. Non-finite or incomplete
objects needed by a later stage and run-integrity failures remain fatal.

Every projection retains the same strictly convex QP. Its independent dual
audit reconstructs multipliers with deterministic bounded-variable least
squares (BVLS); a failed cold numerical attempt may use the two declared cold
OSQP retry settings, without regularizing or otherwise changing the problem.

The separate, already-completed 500-input preflight record used 400 development
inputs, 100 untouched test inputs, five robustness cases, and five Clarifier
layers. Under the then-current protocol, its limited optimization smoke used
one center start and a 600-second (10-minute) ceiling for each embedded-surrogate
IPOPT continuation stage. That historical solver path and every preflight
artifact are excluded from the revised full article optimization.
"""),
    markdown(r"""
## Immutable-source execution

The production contract hashes this notebook byte-for-byte. For a robust
article execution, keep `main_closed_loop.ipynb` unmodified while the run is in
progress. The safest command is to execute it into a different output file:

```powershell
uv run jupyter nbconvert --to notebook --execute main_closed_loop.ipynb `
  --output main_closed_loop.executed.ipynb --ExecutePreprocessor.timeout=-1
```

For interactive work, open a copy of the notebook or disable autosave; do not
save execution counts or outputs back into this source notebook until the run
has finished. All scientific checkpoints live under the selected result
directory, so rerunning a stage resumes verified work rather than starting
over.
"""),
    code(r"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from IPython.display import display

from closed_loop.manuscript_v3 import ARTICLE_FULL
from scripts.run_article_v3_5000 import (
    DEFAULT_RUN_ID,
    LEGACY_RUN_ID,
    OPTIMIZATION_PROTOCOL,
    RUN_ID_PATTERN,
    main as run_article,
    resolve_run_directory,
)

ROOT = Path.cwd().resolve()
if not (ROOT / "scripts" / "run_article_v3_5000.py").is_file():
    raise RuntimeError(
        "Run this notebook from the surrogate-optimization-arch repository root."
    )

# Every rerun gets its own directory. Configure the target and optional source
# with environment variables before starting, or edit these assignments.
RUN_ID = os.environ.get("ARTICLE_V3_RUN_ID", DEFAULT_RUN_ID)
if RUN_ID_PATTERN.fullmatch(RUN_ID) is None or ".." in RUN_ID:
    raise ValueError(
        "ARTICLE_V3_RUN_ID must match article_full_5000_<identifier>."
    )
RUN_ROOT = resolve_run_directory(RUN_ID)
REUSE_FROM_RUN_ID = os.environ.get("ARTICLE_V3_REUSE_FROM_RUN_ID", LEGACY_RUN_ID)
if REUSE_FROM_RUN_ID == RUN_ID:
    raise ValueError("The rerun source and target run IDs must differ.")

profile = asdict(ARTICLE_FULL)
expected_profile = {
    "name": "article_full",
    "development_count": 4_000,
    "test_count": 1_000,
    "robustness_count": 10,
    "layer_count": 10,
    "article_eligible": True,
    "enforce_admission_gate": True,
}
for field, expected in expected_profile.items():
    if profile.get(field) != expected:
        raise RuntimeError(
            f"ARTICLE_FULL contract mismatch for {field}: "
            f"{profile.get(field)!r} != {expected!r}"
        )
if profile["development_count"] + profile["test_count"] != 5_000:
    raise RuntimeError("The article profile must require exactly 5,000 accepted inputs.")

contract_config = json.loads(
    (ROOT / "config" / "params_manuscript_v3.json").read_text(encoding="utf-8")
)
optimization = contract_config["optimization"]
article_config = contract_config["profiles"]["article_full"]
required_optimization = {
    "protocol": "single_start_exact_qp_active_set",
    "runner_protocol": "single_center_local_exact_qp_v1",
    "surrogate_protocol": "seven_variable_exact_qp_single_start_v1",
    "direct_protocol": "smooth_direct_single_center_v1",
    "start_count": 1,
    "direct_start_count": 1,
    "surrogate_embedded_kkt_ipopt_enabled": False,
    "direct_smoothing_continuation_retained": True,
    "optimization_case_count": 11,
    "nominal_case_count": 1,
    "robustness_case_count": 10,
    "optimization_case_failure_stops_workflow": False,
}
for field, expected in required_optimization.items():
    if optimization.get(field) != expected:
        raise RuntimeError(
            f"Optimization contract mismatch for {field}: "
            f"{optimization.get(field)!r} != {expected!r}"
        )
if article_config.get("optimization_start_count") != 1:
    raise RuntimeError("The article profile requires one start per route and case.")
if len(optimization.get("smooth_sequence", [])) != 3:
    raise RuntimeError("The direct route must retain its three smoothing stages.")
if optimization.get("surrogate_gap_sequence") != []:
    raise RuntimeError("The surrogate route must not execute gap-continuation stages.")
if OPTIMIZATION_PROTOCOL != optimization["runner_protocol"]:
    raise RuntimeError("The runner and configuration optimization protocols differ.")
fallback = optimization.get("surrogate_active_set_derivative_fallback", {})
required_fallback = {
    "enabled": True,
    "method": "COBYQA",
    "deterministic": True,
    "value_only": True,
    "finite_difference_derivatives_used": False,
    "maximum_iterations": 250,
    "maximum_function_evaluations": 250,
    "selected_point_cold_replayed": True,
    "endpoint_active_set_and_upper_kkt_reaudited": True,
    "fallback_failure_stops_other_cases": False,
}
for field, expected in required_fallback.items():
    if fallback.get(field) != expected:
        raise RuntimeError(
            f"Surrogate fallback contract mismatch for {field}: "
            f"{fallback.get(field)!r} != {expected!r}"
        )
certification = optimization.get("surrogate_local_convergence_certification", {})
required_certification = {
    "protocol": "exact_qp_two_scale_accelerated_feasible_poll_v3",
    "poll_radii_normalized": [1e-3, 1e-4],
    "maximum_exact_qp_evaluations": 10_000,
    "acceleration_growth_factor": 2.0,
    "maximum_acceleration_probes_per_winning_ray": 16,
    "final_certificate_requires_fresh_unaccelerated_poll": True,
    "finite_poll_claims_classical_stationarity": False,
}
for field, expected in required_certification.items():
    if certification.get(field) != expected:
        raise RuntimeError(
            f"Surrogate certification contract mismatch for {field}: "
            f"{certification.get(field)!r} != {expected!r}"
        )
reporting_contract = contract_config["reporting"]
if reporting_contract.get("validation_protocol") != "casewise_exact_common_reference_v3":
    raise RuntimeError("The casewise exact common-reference protocol is required.")
if reporting_contract.get("untouched_test_smooth_reference_equivalence_executed") is not False:
    raise RuntimeError("Test-set-wide smooth/reference equivalence must remain retired.")
if reporting_contract.get("common_reference_start_count_per_decision") != 2:
    raise RuntimeError("Every selected decision requires a two-start exact replay.")
if reporting_contract.get("timing_protocol") != "robustness_casewise_aggregate_v1":
    raise RuntimeError("Timing must be aggregated from the ten robustness cases.")
if reporting_contract.get("untouched_test_repeated_inference_benchmark_executed") is not False:
    raise RuntimeError("Repeated untouched-test timing must remain retired.")

display(pd.Series({
    "run_id": RUN_ID,
    "run_directory": str(RUN_ROOT),
    "reused_from_run_id": REUSE_FROM_RUN_ID,
    "accepted_dataset_target": 5_000,
    "accepted_development_target": profile["development_count"],
    "accepted_untouched_test_target": profile["test_count"],
    "candidate_attempt_count": "reported after generation; may exceed 5,000",
    "candidate_replacement_policy": "audit, exclude, deterministically replace",
    "robustness_cases": profile["robustness_count"],
    "clarifier_layers": profile["layer_count"],
    "starts_per_route_and_case": 1,
    "surrogate_optimization": optimization["surrogate_protocol"],
    "surrogate_derivative_fallback": fallback["method"],
    "fallback_value_only": fallback["value_only"],
    "fallback_distinct_trial_qp": fallback["projection_qp_at_each_distinct_trial"],
    "fallback_evaluation_budget": fallback["maximum_function_evaluations"],
    "surrogate_embedded_ipopt_stages": 0,
    "direct_optimization": optimization["direct_protocol"],
    "direct_smoothing_stages": len(optimization["smooth_sequence"]),
    "local_optimum_label_requires": fallback["local_optimum_label_requires"],
    "surrogate_local_certification": certification["protocol"],
    "casewise_validation": reporting_contract["validation_protocol"],
    "timing_protocol": reporting_contract["timing_protocol"],
    "timing_cases": reporting_contract["timing_case_count"],
    "test_set_smooth_reference_equivalence": "retired",
    "unresolved_incumbent_is_called_local_optimum": False,
    "global_optimality_claimed": False,
    "full_run_wall_time_ceiling": None,
    "scientific_admission_gate_enforced_for_article_eligibility": True,
    "admission_gate_execution_policy": "advisory; scientific eligibility unchanged",
}, name="value").to_frame())
"""),
    markdown(r"""
## Status and artifact views

The helpers below are read-only. Each stage call publishes atomic checkpoints;
the `finally` block displays the latest state and paths. A scientific gate
failure is visible in those artifacts but does not stop this model-function
exercise. Hard computational and integrity failures still stop the stage.
"""),
    code(r"""
STAGE_ARTIFACTS = {
    "generation": (
        "inputs/contract.json",
        "inputs/contract_migrations/article-v3-generation-replacement-v1.json",
        "inputs/contract_migrations/article-v3-projection-audit-v1.json",
        "inputs/contract_migrations/article-v3-direct-active-set-v1.json",
        "inputs/contract_migrations/article-v3-casewise-common-reference-v1.json",
        "inputs/contract_migrations/article-v3-convergence-poll-refinement-v1.json",
        "inputs/contract_migrations/article-v3-poll-linesearch-v1.json",
        "inputs/contract_migrations/article-v3-poll-linesearch-v1-retained.json",
        "inputs/contract_migrations/article-v3-poll-linesearch-v1-reused-files.json",
        "inputs/contract_migrations/article-v3-casewise-timing-v1.json",
        "inputs/contract_migrations/article-v3-casewise-timing-v1-retained.json",
        "datasets/design.npz",
        "datasets/development/all_attempts.csv",
        "datasets/development/accepted_provenance.csv",
        "datasets/development/accepted_inputs.npz",
        "datasets/development/mechanistic_accepted_v3.npz",
        "datasets/development/accepted_diagnostics.csv",
        "datasets/development/base_checkpoint_migration.csv",
        "datasets/development/replacement_summary.json",
        "datasets/development/block_complete.json",
        "datasets/test/all_attempts.csv",
        "datasets/test/accepted_provenance.csv",
        "datasets/test/accepted_inputs.npz",
        "datasets/test/mechanistic_accepted_v3.npz",
        "datasets/test/accepted_diagnostics.csv",
        "datasets/test/base_checkpoint_migration.csv",
        "datasets/test/replacement_summary.json",
        "datasets/test/block_complete.json",
    ),
    "assessment": (
        "models/ridge_complete.json",
        "metrics/assessment_complete.json",
        "metrics/admission_gate.json",
        "metrics/untouched_prediction_metrics.csv",
        "metrics/physical_violations_assessment.csv",
    ),
    "complete": (
        "optimization/optimization_complete.json",
        "metrics/untouched_test_equivalence_retired.json",
        "metrics/convergence_poll_refinement.json",
        "metrics/selected_candidate_reference_evaluation.csv",
        "metrics/case_common_reference_comparison.csv",
        "metrics/selected_response_physical_audit.csv",
        "metrics/physical_violations_all_analysis.csv",
        "metrics/robustness_case_timing.csv",
        "metrics/robustness_case_timing_summary.json",
        "metrics/robustness_case_timing_complete.json",
        "report/tables/report_manifest.json",
    ),
}


def read_json(relative_path: str) -> dict:
    path = RUN_ROOT / relative_path
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_table(through: str) -> pd.DataFrame:
    ordered = ["run_state.json"]
    for stage in ("generation", "assessment", "complete"):
        ordered.extend(STAGE_ARTIFACTS[stage])
        if stage == through:
            break
    rows = []
    for relative in dict.fromkeys(ordered):
        path = RUN_ROOT / relative
        rows.append({
            "artifact": relative,
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "absolute_path": str(path),
        })
    return pd.DataFrame(rows)


def show_status(through: str) -> None:
    state = read_json("run_state.json")
    display(pd.Series(
        state or {"stage": through, "status": "no state published"},
        name="value",
    ).to_frame())
    display(artifact_table(through))


def invoke_stage(through: str) -> None:
    print(f"Invoking production runner through {through!r} for {RUN_ID!r}.")
    try:
        run_article(
            run_id=RUN_ID,
            through=through,
            reuse_from_run_id=(None if RUN_ROOT.exists() else REUSE_FROM_RUN_ID),
        )
    finally:
        show_status(through)
"""),
    markdown(r"""
## 1. Complete the accepted 4,000/1,000 mechanistic blocks

Candidate round 0 retains the independent development/test Latin hypercubes
and resumes their row-level, two-start nonsmooth mechanistic checkpoints. Each
rejected candidate remains in the attempt ledger with all available audits but
is excluded from the accepted dataset. The runner then continues that block's
persisted SplitMix64 state in deterministic row-major supplemental rounds,
each sized to the remaining deficit, until exactly 4,000 development and 1,000
test rows have been accepted. Accepted replacements fill failed original slots
in ascending order; neither candidates nor accepted rows cross block boundaries.

The accepted union is conditioned on mechanistic acceptance and is not one
global Latin hypercube. Generation reports therefore distinguish the attempted
candidate denominator from the accepted-row denominator and retain every
candidate-to-final-slot mapping. The phrase "untouched test" means untouched by
fitting and tuning, not unconditional sampling from the entire input box.

For every rerun, the runner first creates a new self-contained run directory.
It byte-copies and hash-verifies the accepted generation, fitted surrogate,
assessment, and completed primary case-search artifacts from the declared
source run, records their provenance, and generates all corrected downstream
artifacts in the new directory. It does not regenerate data, refit the
surrogate, or repeat a completed primary case search.
"""),
    code(r"""
invoke_stage("generation")

generation_rows = []
attempt_status_rows = []
for block, required in (("development", 4_000), ("test", 1_000)):
    attempts_path = RUN_ROOT / "datasets" / block / "all_attempts.csv"
    provenance_path = RUN_ROOT / "datasets" / block / "accepted_provenance.csv"
    attempts = pd.read_csv(attempts_path) if attempts_path.is_file() else pd.DataFrame()
    provenance = (
        pd.read_csv(provenance_path) if provenance_path.is_file() else pd.DataFrame()
    )
    summary = read_json(f"datasets/{block}/replacement_summary.json")
    generation_rows.append({
        "block": block,
        "candidate_attempts": len(attempts),
        "required_accepted_rows": required,
        "accepted_rows": len(provenance),
        "rejected_attempts": max(0, len(attempts) - len(provenance)),
        "base_accepted_rows": summary.get("base_accepted_count"),
        "supplemental_attempts": summary.get("supplemental_attempt_count"),
        "supplemental_rounds": summary.get("supplemental_round_count"),
        "provenance_complete": len(provenance) == required,
    })
    if "attempt_status" in attempts:
        attempt_status_rows.extend(
            {"block": block, "attempt_status": status, "count": int(count)}
            for status, count in attempts["attempt_status"].value_counts(
                dropna=False
            ).items()
        )
display(pd.DataFrame(generation_rows))
display(pd.DataFrame(
    attempt_status_rows,
    columns=("block", "attempt_status", "count"),
))
"""),
    markdown(r"""
## 2. Fit and assess on the untouched 1,000-input test block

This call reuses the generation checkpoints, performs the frozen five-fold
ridge selection and trust calibration, opens the untouched block once, and
publishes raw/projected/mechanistic accuracy and physical-violation ledgers.
The gate result remains the scientific article-eligibility decision. A failure
is not waived or refitted; it is recorded, while execution continues because
this run is currently serving as a complete model-function exercise.
"""),
    code(r"""
invoke_stage("assessment")

gate = read_json("metrics/admission_gate.json")
if gate:
    display(pd.Series(gate, name="value").to_frame())

prediction_path = RUN_ROOT / "metrics" / "untouched_prediction_metrics.csv"
if prediction_path.is_file():
    display(pd.read_csv(prediction_path))

physical_path = RUN_ROOT / "metrics" / "physical_violations_assessment.csv"
if physical_path.is_file():
    physical = pd.read_csv(physical_path)
    display(physical.groupby("method", dropna=False).agg(
        rows=("method", "size"),
        maximum_mass_violation=("mass_conservation_violation_max", "max"),
        maximum_nonnegativity_violation=("nonnegativity_violation_max", "max"),
        minimum_coordinate=("minimum_coordinate", "min"),
    ))
"""),
    markdown(r"""
## 3. Optimize, independently replay, audit, and report

This resumes the completed assessment, including any recorded scientific gate
failure, and runs the nominal plus ten robustness cases. Both routes use one
deterministic box-center start per case and search only that local basin. The
surrogate route primarily uses the seven-variable exact-QP active-set solver
with normalized constraints and no embedded-KKT IPOPT continuation. If its
exact derivatives are unavailable, deterministic value-only COBYQA evaluates
the same problem, cold-solving the unchanged projection QP at every distinct
trial. The best feasible visited point is cold-replayed; a budget-limited or
stationarity-unresolved result remains an incumbent, not a claimed optimum.
The direct route retains only its separate three-stage smoothing continuation.
Each retained surrogate endpoint is then certified either by its exact active-
set KKT audit or by a deterministic two-scale feasible poll; the latter
supports only finite-direction, finite-resolution convergence, not
mathematical stationarity. A
failed direct route receives at most one recovery solve initialized from the
certified surrogate decision.

Both returned decisions are evaluated on the same exact, nonsmooth two-start
mechanistic model in each case. Branch-boundary ambiguity is reported as a
qualifier instead of being mistaken for solver failure. This casewise replay
replaces the retired 1,000-row smooth/reference equivalence sweep. The driver
records raw, projected, optimizer-native, and exact-reference mass-conservation
and non-negativity audits before writing all article tables. A failed or
unresolved case remains in the denominator and does not suppress subsequent
cases or downstream audits. Runtime summaries are calculated only from the
durations recorded in robustness cases 01--10; no repeated timing benchmark is
run over the 1,000-row untouched test block. There is no 10-minute ceiling in
this phase.
"""),
    code(r"""
invoke_stage("complete")

all_physical_path = RUN_ROOT / "metrics" / "physical_violations_all_analysis.csv"
if all_physical_path.is_file():
    all_physical = pd.read_csv(all_physical_path)
    display(all_physical.groupby("method", dropna=False).agg(
        rows=("method", "size"),
        maximum_mass_violation=("mass_conservation_violation_max", "max"),
        mass_violating_rows=(
            "mass_conservation_violation_count", lambda values: int((values > 0).sum())
        ),
        maximum_nonnegativity_violation=("nonnegativity_violation_max", "max"),
        nonnegative_violating_rows=(
            "nonnegativity_violation_count", lambda values: int((values > 0).sum())
        ),
        minimum_coordinate=("minimum_coordinate", "min"),
    ))

report_directory = RUN_ROOT / "report" / "tables"
report_rows = [
    {
        "report_artifact": path.name,
        "bytes": path.stat().st_size,
        "absolute_path": str(path),
    }
    for path in sorted(report_directory.glob("*"))
    if path.is_file()
]
display(pd.DataFrame(
    report_rows,
    columns=("report_artifact", "bytes", "absolute_path"),
))
"""),
    markdown(r"""
## Resumption

If execution is interrupted, rerun the setup and helper cells with the same
target and source IDs, then rerun the cell for the desired terminal stage.
Calling `complete` is sufficient to resume every missing prerequisite in that
target directory. A later rerun must use a new target ID and name its prior run
with `ARTICLE_V3_REUSE_FROM_RUN_ID`; the driver creates another self-contained,
hash-pinned copy rather than overwriting the earlier run. Candidate and
accepted-row checkpoints are accepted only when their source, profile, input,
stream-state, and provenance bindings match. Any mismatch is a run-integrity
failure and is not hidden by regeneration.
"""),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12",
        },
        "surrogate_optimization_arch": {
            "interface": "scripts/run_article_v3_5000.py",
            "profile": "article_full",
            "schema": 5,
        },
    },
)
nbf.validate(notebook)
nbf.write(notebook, NOTEBOOK)
print(NOTEBOOK)
