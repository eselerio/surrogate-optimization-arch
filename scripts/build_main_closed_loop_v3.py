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
# Manuscript-v3 three-route closed-loop article run

This notebook is the canonical, resumable interface to
`scripts/run_article_v3_5000.py`. The production driver, rather than duplicated
notebook code, owns candidate generation and deterministic replacement, two-start
mechanistic acceptance, the whole-system and shared-unit ridge fits,
post-selection-holdout surrogate assessment, route-specific scientific admission gates, three-route optimization,
casewise exact-reference replay, convergence and physical audits, and
Results/Discussion tables.

The production interface supports the original **5,000 accepted-dataset**
workload, the interrupted 50,000-target run, and its user-frozen set of
**16,714 accepted datasets**. The frozen set contains 13,371 development and
3,343 post-selection holdout rows; it is the default for the revised analysis,
and no further mechanistic rows are generated. Rejected mechanistic candidates
remain fully audited but are excluded and deterministically replaced. Each uses ten
robustness cases plus the nominal case and a ten-layer
Clarifier in the mechanistic model. Each retained 170-coordinate mechanistic
response is deterministically reduced to 161 operational coordinates: mixer and
reactor states, Clarifier outlet component flows, and the scalar
Clarifier-solids inventory $M_{\rm cl}$. Neither surrogate predicts or
reconstructs the ten-layer profile. The existing whole-system projected
surrogate is route $S$. Added route $U$ reuses one 20-output quadratic CSTR map
in all five reactors, applies one 41-output quadratic Clarifier map, closes its
recycle from two target-independent starts, and then uses the same joint
161-coordinate physical projection as route $S$. The unchanged smooth
mechanistic NLP is route $M$. Each route uses one deterministic box-center
start per case and searches only that basin without claiming global
optimality. Route $S$ primarily runs the seven-variable exact-QP active-set optimizer. If
exact active-set derivatives are unavailable, deterministic value-only COBYQA
cold-solves the unchanged projection QP at every distinct fallback trial and
retains the best feasible visited incumbent. Route $U$ uses deterministic
value-only search; every distinct trial cold-solves and audits both recycle
roots before cold-solving the common projection QP. A budget-limited or
stationarity-unresolved incumbent is not called a local optimum. Each selected
surrogate endpoint then receives a bounded two-scale complete-evaluation feasible poll;
this can support a finite-direction, finite-resolution convergence
qualification without claiming differentiable KKT stationarity. Neither
surrogate route executes the retired embedded-KKT IPOPT problem or any of its
seven gap-continuation stages. Route $M$ retains its
separate three-stage smoothing continuation. There is no wall-time ceiling in
the full article run. The revised scientific admission thresholds are frozen
and determine whether results are article-eligible. For this model-function
exercise they are advisory for execution: failures are recorded and propagated
while later stages are attempted without refitting. Non-finite or incomplete
objects needed by a later stage and run-integrity failures remain fatal.

Both surrogate routes retain the same strictly convex QP over the reduced response.
Its 75 equalities exclude the former layer-endpoint identities; outlet TSS is
derived from the outlet component flows. Ten particulate-densification rows and
two tight, division-free inventory-envelope rows form its 12 physical
inequalities. Its independent dual
audit reconstructs multipliers with deterministic bounded-variable least
squares (BVLS); a failed cold numerical attempt may use the two declared cold
OSQP retry settings, without regularizing or otherwise changing the problem.

The separate, already-completed 500-input preflight record used 400 development
inputs, 100 test inputs, five robustness cases, and five Clarifier
layers. Under the then-current protocol, its limited optimization smoke used
one center start and a 600-second (10-minute) ceiling for each embedded-surrogate
IPOPT continuation stage. That historical solver path and every preflight
artifact are excluded from the revised full article optimization.
"""),
    markdown(r"""
## Immutable-source execution

The production contract hashes this notebook byte-for-byte. The three-route
implementation uses runner schema 11 and must start in a fresh target directory;
it cannot resume a two-route fit or optimization checkpoint as though that
checkpoint contained route $U$. Accepted mechanistic generation artifacts may
be copied from the declared predecessor only through the runner's verified
generation-reuse path. For a robust article execution, keep
`main_closed_loop.ipynb` unmodified while the run is in progress. The safest
command is to execute it into a different output file:

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
    ASSESSMENT_BATCH_PROTOCOL,
    AUTHORIZED_DATASET_TOTALS,
    DEFAULT_ASSESSMENT_BATCH_SIZE,
    DEFAULT_RUN_ID,
    LEGACY_RUN_ID,
    OPTIMIZATION_PROTOCOL,
    RESPONSE_SCHEMA,
    RUNNER_SCHEMA,
    RUN_ID_PATTERN,
    frozen_accepted_profile,
    main as run_article,
    profile_for_dataset_total,
    resolve_run_directory,
)

ROOT = Path.cwd().resolve()
if not (ROOT / "scripts" / "run_article_v3_5000.py").is_file():
    raise RuntimeError(
        "Run this notebook from the surrogate-optimization-arch repository root."
    )

# Every rerun gets its own directory. Configure the target and optional source
# with environment variables before starting, or edit these assignments.
DATASET_COUNT = int(os.environ.get("ARTICLE_V3_DATASET_COUNT", "50000"))
USE_FROZEN_ACCEPTED = os.environ.get(
    "ARTICLE_V3_USE_FROZEN_ACCEPTED_CHECKPOINTS", "1"
).strip().lower() in {"1", "true", "yes"}
if DATASET_COUNT not in AUTHORIZED_DATASET_TOTALS:
    raise ValueError(
        f"ARTICLE_V3_DATASET_COUNT must be one of {AUTHORIZED_DATASET_TOTALS}."
    )
PROFILE = (
    frozen_accepted_profile()
    if USE_FROZEN_ACCEPTED
    else profile_for_dataset_total(DATASET_COUNT)
)
ASSESSMENT_WORKERS = int(os.environ.get(
    "ARTICLE_V3_ASSESSMENT_WORKERS", str(PROFILE.parallel_workers)
))
ASSESSMENT_BATCH_SIZE = int(os.environ.get(
    "ARTICLE_V3_ASSESSMENT_BATCH_SIZE", str(DEFAULT_ASSESSMENT_BATCH_SIZE)
))
if ASSESSMENT_WORKERS < 1 or ASSESSMENT_BATCH_SIZE < 1:
    raise ValueError("Assessment workers and batch size must be positive.")
default_run_id = (
    "article_full_50000_three_route_001" if USE_FROZEN_ACCEPTED
    else DEFAULT_RUN_ID if DATASET_COUNT == 5_000
    else f"article_full_{DATASET_COUNT}_001"
)
RUN_ID = os.environ.get("ARTICLE_V3_RUN_ID", default_run_id)
if RUN_ID_PATTERN.fullmatch(RUN_ID) is None or ".." in RUN_ID:
    raise ValueError(
        "ARTICLE_V3_RUN_ID must match article_full_<5000-or-50000>_<identifier>."
    )
RUN_ROOT = resolve_run_directory(RUN_ID)
REUSE_FROM_RUN_ID = os.environ.get(
    "ARTICLE_V3_REUSE_FROM_RUN_ID",
    (
        "article_full_50000_003" if USE_FROZEN_ACCEPTED
        else LEGACY_RUN_ID if DATASET_COUNT == 5_000
        else None
    ),
)
if REUSE_FROM_RUN_ID == RUN_ID:
    raise ValueError("The rerun source and target run IDs must differ.")
AUTHORIZE_PARALLEL_ASSESSMENT_MIGRATION = os.environ.get(
    "ARTICLE_V3_AUTHORIZE_PARALLEL_ASSESSMENT_MIGRATION",
    "1" if RUN_ID == "article_full_50000_three_route_001" else "0",
).strip().lower() in {"1", "true", "yes"}

profile = asdict(PROFILE)
expected_profile = {
    "name": (
        "article_frozen_16714" if USE_FROZEN_ACCEPTED
        else "article_full" if DATASET_COUNT == 5_000
        else f"article_full_{DATASET_COUNT}"
    ),
    "development_count": 13_371 if USE_FROZEN_ACCEPTED else DATASET_COUNT * 4 // 5,
    "test_count": 3_343 if USE_FROZEN_ACCEPTED else DATASET_COUNT // 5,
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
expected_total = 16_714 if USE_FROZEN_ACCEPTED else DATASET_COUNT
if profile["development_count"] + profile["test_count"] != expected_total:
    raise RuntimeError("The article profile does not match ARTICLE_V3_DATASET_COUNT.")

contract_config = json.loads(
    (ROOT / "config" / "params_manuscript_v3.json").read_text(encoding="utf-8")
)
if contract_config.get("schema_version") != 5:
    raise RuntimeError("The three-route reduced-response configuration schema is required.")
if contract_config.get("execution", {}).get("runner_schema") != 11 or RUNNER_SCHEMA != 11:
    raise RuntimeError("The three-route calculation requires runner schema 11.")
execution_contract = contract_config["execution"]
if (
    execution_contract.get("assessment_batch_protocol")
    != ASSESSMENT_BATCH_PROTOCOL
    or int(execution_contract.get("assessment_batch_size", -1))
    != DEFAULT_ASSESSMENT_BATCH_SIZE
):
    raise RuntimeError("The deterministic assessment-batch contract is inconsistent.")
surrogate_contract = contract_config["surrogate"]
required_surrogate = {
    "route_id": "surrogate",
    "route_symbol": "S",
    "architecture": "whole_system",
    "response_schema": RESPONSE_SCHEMA,
    "mechanistic_target_dimension": 170,
    "response_dimension": 161,
    "mechanistic_layers_retained": True,
}
for field, expected in required_surrogate.items():
    if surrogate_contract.get(field) != expected:
        raise RuntimeError(
            f"Surrogate response contract mismatch for {field}: "
            f"{surrogate_contract.get(field)!r} != {expected!r}"
        )
unit_contract = contract_config["shared_unit_surrogate"]
required_unit = {
    "route_id": "shared_unit",
    "route_symbol": "U",
    "response_schema": RESPONSE_SCHEMA,
    "response_dimension": 161,
    "reactor_count": 5,
    "component_count": 20,
    "reactor_input_dimension": 22,
    "reactor_output_dimension": 20,
    "clarifier_input_dimension": 22,
    "clarifier_output_dimension": 41,
    "quadratic_feature_count_per_local_model": 276,
    "reactor_coefficient_shape": [20, 276],
    "clarifier_coefficient_shape": [41, 276],
    "shared_reactor_coefficients_across_all_stages": True,
    "reactor_stage_indicator_included": False,
    "internal_recycle_in_clarifier_input": False,
    "fold_count": 5,
    "development_reactor_transition_count": 66_855,
    "post_selection_holdout_reactor_transition_count": 16_715,
}
for field, expected in required_unit.items():
    if unit_contract.get(field) != expected:
        raise RuntimeError(
            f"Shared-unit contract mismatch for {field}: "
            f"{unit_contract.get(field)!r} != {expected!r}"
        )
root_contract = unit_contract.get("recycle_closure", {})
required_root = {
    "solver": "scipy.optimize.least_squares",
    "method": "trf",
    "ftol": 1e-10,
    "xtol": 1e-10,
    "gtol": 1e-10,
    "max_nfev": 1000,
    "target_independent_starts": True,
    "cold_solve_both_starts_at_every_distinct_control": True,
    "cross_control_warm_start_permitted": False,
    "both_starts_must_succeed": True,
    "normalized_residual_infinity_maximum": 1e-8,
    "mixer_scaled_agreement_infinity_maximum": 1e-6,
    "assembled_response_scaled_agreement_infinity_maximum": 1e-6,
    "required_jacobian_rank": 20,
    "condition_number_times_machine_epsilon_maximum": 1e-8,
    "uniqueness_claimed": False,
}
for field, expected in required_root.items():
    if root_contract.get(field) != expected:
        raise RuntimeError(
            f"Shared-unit recycle-root contract mismatch for {field}: "
            f"{root_contract.get(field)!r} != {expected!r}"
        )
projection_contract = contract_config["projection"]
required_projection = {
    "response_dimension": 161,
    "equality_count": 75,
    "physical_inequality_count": 12,
    "inequality_count_including_nonnegativity": 173,
    "predicts_or_reconstructs_layer_profile": False,
}
for field, expected in required_projection.items():
    if projection_contract.get(field) != expected:
        raise RuntimeError(
            f"Projection contract mismatch for {field}: "
            f"{projection_contract.get(field)!r} != {expected!r}"
        )
if contract_config.get("trust", {}).get("diagnostics") != [
    "correction", "regularized_leverage", "particulate_split", "reactor_residual",
]:
    raise RuntimeError("The unchanged route-S four-diagnostic trust contract is required.")
required_unit_trust = [
    "correction",
    "reactor_regularized_leverage_stage_1",
    "reactor_regularized_leverage_stage_2",
    "reactor_regularized_leverage_stage_3",
    "reactor_regularized_leverage_stage_4",
    "reactor_regularized_leverage_stage_5",
    "clarifier_regularized_leverage",
    "particulate_split",
    "reactor_residual",
]
trust_contract = contract_config.get("trust", {})
if trust_contract.get("route_specific_diagnostics", {}).get("shared_unit") != required_unit_trust:
    raise RuntimeError("The route-U local-leverage trust contract is required.")
if trust_contract.get("shared_unit_root_acceptance_is_trust_inequality") is not False:
    raise RuntimeError("Route-U root acceptance must remain outside upper trust inequalities.")
optimization = contract_config["optimization"]
article_config = contract_config["profiles"][profile["name"]]
required_optimization = {
    "protocol": "three_route_single_start_mixed_surrogate_v1",
    "runner_protocol": "three_route_single_center_v2",
    "route_ids": ["surrogate", "shared_unit", "direct"],
    "route_count": 3,
    "surrogate_protocol": "seven_variable_exact_qp_single_start_v1",
    "shared_unit_protocol": "shared_unit_value_only_single_center_v1",
    "direct_protocol": "smooth_direct_single_center_v1",
    "start_count": 1,
    "direct_start_count": 1,
    "surrogate_embedded_kkt_ipopt_enabled": False,
    "shared_unit_analytical_derivatives_enabled": False,
    "shared_unit_strong_first_order_certificate_available": False,
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
    raise RuntimeError("The surrogate routes must not execute gap-continuation stages.")
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
if reporting_contract.get("validation_protocol") != "casewise_exact_common_reference_v4":
    raise RuntimeError("The three-route casewise exact common-reference protocol is required.")
if reporting_contract.get("required_routes") != ["surrogate", "shared_unit", "direct"]:
    raise RuntimeError("The reporting denominator must contain routes S, U, and M.")
if reporting_contract.get("pairwise_comparisons") != ["S-U", "S-M", "U-M"]:
    raise RuntimeError("All three pairwise route comparisons are required.")
if reporting_contract.get("selected_candidate_maximum") != 33:
    raise RuntimeError("Eleven cases and three routes require up to 33 candidates.")
if reporting_contract.get("exact_reference_start_maximum") != 66:
    raise RuntimeError("The three-route comparison requires up to 66 exact starts.")
if reporting_contract.get("untouched_test_smooth_reference_equivalence_executed") is not False:
    raise RuntimeError("Test-set-wide smooth/reference equivalence must remain retired.")
if reporting_contract.get("common_reference_start_count_per_decision") != 2:
    raise RuntimeError("Every selected decision requires a two-start exact replay.")
if reporting_contract.get("timing_protocol") != "robustness_casewise_three_route_v2":
    raise RuntimeError("Three-route timing must be aggregated from the ten robustness cases.")
if reporting_contract.get("untouched_test_repeated_inference_benchmark_executed") is not False:
    raise RuntimeError("Repeated untouched-test timing must remain retired.")

display(pd.Series({
    "run_id": RUN_ID,
    "run_directory": str(RUN_ROOT),
    "reused_from_run_id": REUSE_FROM_RUN_ID,
    "runner_schema": RUNNER_SCHEMA,
    "optimization_routes": "S=surrogate, U=shared_unit, M=direct",
    "accepted_dataset_target": expected_total,
    "accepted_development_target": profile["development_count"],
    "accepted_post_selection_holdout_target": profile["test_count"],
    "candidate_attempt_count": (
        "frozen at 18,211" if USE_FROZEN_ACCEPTED
        else f"reported after generation; may exceed {expected_total:,}"
    ),
    "candidate_replacement_policy": "audit, exclude, deterministically replace",
    "robustness_cases": profile["robustness_count"],
    "mechanistic_clarifier_layers": profile["layer_count"],
    "mechanistic_response_coordinates": surrogate_contract["mechanistic_target_dimension"],
    "response_coordinates_per_surrogate": surrogate_contract["response_dimension"],
    "surrogate_clarifier_coordinates": "outlet flows plus M_cl; no layer profile",
    "route_S_features": 406,
    "route_U_reactor_features": unit_contract["quadratic_feature_count_per_local_model"],
    "route_U_clarifier_features": unit_contract["quadratic_feature_count_per_local_model"],
    "route_U_reactor_coefficients": unit_contract["reactor_coefficient_shape"],
    "route_U_clarifier_coefficients": unit_contract["clarifier_coefficient_shape"],
    "route_U_development_transitions": unit_contract["development_reactor_transition_count"],
    "route_U_root_starts": 2,
    "projection_equalities": projection_contract["equality_count"],
    "projection_physical_inequalities": projection_contract["physical_inequality_count"],
    "projection_total_inequalities": projection_contract["inequality_count_including_nonnegativity"],
    "starts_per_route_and_case": 1,
    "route_S_optimization": optimization["surrogate_protocol"],
    "route_U_optimization": optimization["shared_unit_protocol"],
    "route_M_optimization": optimization["direct_protocol"],
    "route_S_derivative_fallback": fallback["method"],
    "route_U_value_only": True,
    "fallback_value_only": fallback["value_only"],
    "fallback_distinct_trial_qp": fallback["projection_qp_at_each_distinct_trial"],
    "fallback_evaluation_budget": fallback["maximum_function_evaluations"],
    "surrogate_embedded_ipopt_stages": 0,
    "direct_smoothing_stages": len(optimization["smooth_sequence"]),
    "local_optimum_label_requires": fallback["local_optimum_label_requires"],
    "surrogate_local_certification": certification["protocol"],
    "casewise_validation": reporting_contract["validation_protocol"],
    "timing_protocol": reporting_contract["timing_protocol"],
    "timing_cases": reporting_contract["timing_case_count"],
    "maximum_selected_candidates": reporting_contract["selected_candidate_maximum"],
    "maximum_exact_reference_starts": reporting_contract["exact_reference_start_maximum"],
    "pairwise_comparisons": ", ".join(reporting_contract["pairwise_comparisons"]),
    "holdout_smooth_reference_equivalence": "retired",
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
        f"inputs/contract_migrations/article-v3-reduced-response-{RUN_ID}.json",
        f"inputs/contract_migrations/article-v3-reduced-response-{RUN_ID}-retained.json",
        f"inputs/contract_migrations/article-v3-reduced-response-{RUN_ID}-reused-files.json",
        f"inputs/contract_migrations/article-v3-three-route-{RUN_ID}.json",
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
        "datasets/development/surrogate_responses_inventory_v1.npz",
        "datasets/test/surrogate_responses_inventory_v1.npz",
        "models/ridge_complete.json",
        "models/shared_unit_reactor_ridge.npz",
        "models/shared_unit_clarifier_ridge.npz",
        "models/shared_unit_fold_models.npz",
        "models/shared_unit_complete.json",
        "metrics/shared_unit_cross_validation.csv",
        "metrics/shared_unit_reactor_cross_validation.csv",
        "metrics/shared_unit_clarifier_cross_validation.csv",
        "metrics/shared_unit_fold_membership.csv",
        "metrics/shared_unit_teacher_forced_metrics.csv",
        "metrics/shared_unit_development_oof_root_diagnostics.csv",
        "metrics/shared_unit_trust_development_oof.csv",
        "metrics/shared_unit_trust_limits.json",
        "models/shared_unit_trust_calibration.npz",
        "metrics/shared_unit_admission_gate.json",
        "metrics/shared_unit_assessment_complete.json",
        "predictions/shared_unit_post_selection_holdout.npz",
        "metrics/shared_unit_post_selection_prediction_metrics.csv",
        "metrics/shared_unit_post_selection_root_diagnostics.csv",
        "metrics/shared_unit_trust_post_selection_holdout.csv",
        "metrics/shared_unit_physical_violations_assessment.csv",
        "metrics/shared_unit_projection_qp_diagnostics.csv",
        "metrics/shared_unit_projection_feasibility_bound.csv",
        "metrics/assessment_complete.json",
        "metrics/admission_gate.json",
        "metrics/post_selection_prediction_metrics.csv",
        "metrics/trust_post_selection_holdout.csv",
        "metrics/physical_violations_assessment.csv",
    ),
    "complete": (
        "optimization/optimization_complete.json",
        "metrics/untouched_test_equivalence_retired.json",
        "metrics/convergence_poll_refinement.json",
        "metrics/selected_candidate_reference_evaluation.csv",
        "metrics/case_common_reference_comparison.csv",
        "metrics/pairwise_common_reference_eligibility.csv",
        "metrics/shared_unit_root_work.csv",
        "metrics/selected_response_physical_audit.csv",
        "metrics/physical_violations_all_analysis.csv",
        "metrics/robustness_case_timing.csv",
        "metrics/robustness_case_timing_summary.json",
        "metrics/robustness_case_timing_complete.json",
        "report/tables/scope_specific_nonlinear_audit.csv",
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
            profile=PROFILE,
            use_frozen_accepted_checkpoints=USE_FROZEN_ACCEPTED,
            reuse_from_run_id=(None if RUN_ROOT.exists() else REUSE_FROM_RUN_ID),
            authorize_parallel_assessment_migration=(
                AUTHORIZE_PARALLEL_ASSESSMENT_MIGRATION
            ),
            assessment_workers=ASSESSMENT_WORKERS,
            assessment_batch_size=ASSESSMENT_BATCH_SIZE,
        )
    finally:
        show_status(through)
"""),
    markdown(r"""
## 1. Complete the accepted mechanistic blocks

Candidate round 0 retains the independent development/test Latin hypercubes
and resumes their row-level, two-start nonsmooth mechanistic checkpoints. Each
rejected candidate remains in the attempt ledger with all available audits but
is excluded from the accepted dataset. The runner then continues that block's
persisted SplitMix64 state in deterministic row-major supplemental rounds,
each sized to the remaining deficit, until the selected profile's 80/20
development/test targets have been accepted. Accepted replacements fill failed original slots
in ascending order; neither candidates nor accepted rows cross block boundaries.

The accepted union is conditioned on mechanistic acceptance and is not one
global Latin hypercube. Generation reports therefore distinguish the attempted
candidate denominator from the accepted-row denominator and retain every
candidate-to-final-slot mapping. The frozen 3,343-row block is post-selection:
its superseded layer-wise summaries informed the reduced response definition.

For this revision, the runner first creates a new self-contained run directory.
It byte-copies and hash-verifies only the frozen design, accepted mechanistic
states, attempt ledgers, and provenance from the declared source run. The
historical 170-output fit, projections, trust calibration, optimization,
replays at the old decisions, timing, and reports are archived as superseded
and are never loaded as current results. The runner deterministically derives
the 161-coordinate response from each full state, then refits and reruns every
surrogate-dependent stage.
"""),
    code(r"""
invoke_stage("generation")

generation_rows = []
attempt_status_rows = []
for block, required in (
    ("development", PROFILE.development_count),
    ("test", PROFILE.test_count),
):
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
## 2. Transform, refit, and assess on the post-selection holdout

This call reuses only the full mechanistic checkpoints. It maps each response
to mixer, five reactors, two Clarifier outlet-flow vectors, and
$M_{\rm cl}=\sum_\ell V_{{\rm cl},\ell}s_\ell$. Route $S$ retains its frozen
five-fold whole-system ridge selection and four-diagnostic trust calibration.
Route $U$ reuses the same plant folds, keeps all five reactor transitions from
one plant together, selects $\gamma_{\rm rxn}$ and $\gamma_{\rm cl}$ separately,
and reports both teacher-forced unit metrics and the complete free-running
two-root recycle pipeline. The final local fits contain 66,855 development
reactor transitions and 13,371 Clarifier rows. Both routes assemble 161
coordinates and independently use the same projection QP.

Development-only gates are route-specific. Each route requires complete and
inventory raw out-of-fold nRMSE below one and successful projection audits;
route $U$ additionally requires every two-start root residual, agreement, rank,
conditioning, and finiteness audit. A failure is not waived or refitted and
does not reclassify the other two routes. The post-selection holdout remains
descriptive and cannot alter either fitted model or threshold.
"""),
    code(r"""
invoke_stage("assessment")

gate = read_json("metrics/admission_gate.json")
if gate:
    display(pd.Series(gate, name="value").to_frame())

unit_gate = read_json("metrics/shared_unit_admission_gate.json")
if unit_gate:
    display(pd.Series(unit_gate, name="value").to_frame())

prediction_path = RUN_ROOT / "metrics" / "post_selection_prediction_metrics.csv"
if prediction_path.is_file():
    display(pd.read_csv(prediction_path))

unit_prediction_path = (
    RUN_ROOT / "metrics" / "shared_unit_post_selection_prediction_metrics.csv"
)
if unit_prediction_path.is_file():
    display(pd.read_csv(unit_prediction_path))

unit_root_path = RUN_ROOT / "metrics" / "shared_unit_post_selection_root_diagnostics.csv"
if unit_root_path.is_file():
    display(pd.read_csv(unit_root_path))

unit_projection_path = RUN_ROOT / "metrics" / "shared_unit_projection_qp_diagnostics.csv"
if unit_projection_path.is_file():
    display(pd.read_csv(unit_projection_path))

unit_bound_path = RUN_ROOT / "metrics" / "shared_unit_projection_feasibility_bound.csv"
if unit_bound_path.is_file():
    display(pd.read_csv(unit_bound_path))

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
failure, and runs the nominal plus ten robustness cases for routes $S$, $U$,
and $M$. All three routes use one deterministic box-center start per case and
search only that local basin. Route $S$ primarily uses the seven-variable exact-QP active-set solver
with normalized constraints and no embedded-KKT IPOPT continuation. If its
exact derivatives are unavailable, deterministic value-only COBYQA evaluates
the same problem, cold-solving the unchanged projection QP at every distinct
trial. Route $U$ is an independent value-only calculation: every distinct
control first cold-solves the learned recycle closure from both declared starts,
audits residual, agreement, rank, and conditioning, assembles its raw response,
and then cold-solves the same projection QP. The best feasible visited point is
cold-replayed; a budget-limited or stationarity-unresolved result remains an
incumbent, not a claimed optimum. Route $M$ retains only its separate three-stage
smoothing continuation. Direct recovery remains conditional only on a certified
route-$S$ endpoint; route $U$ never changes route $M$'s algorithm or seed.
Each retained surrogate endpoint receives a deterministic two-scale feasible
poll; route $S$ may additionally retain its exact active-set KKT tier. The poll
supports only finite-direction, finite-resolution convergence, not
mathematical stationarity. A
failed route remains explicit rather than being replaced by another route.

Every available returned decision, at most 33, is evaluated on the same exact, nonsmooth two-start
mechanistic model in each case. Branch-boundary ambiguity is reported as a
qualifier instead of being mistaken for solver failure. This casewise replay
replaces the retired whole-test smooth/reference equivalence sweep. The driver
records raw and projected 161-coordinate responses together with layer-resolved
smooth-direct and exact-reference states. Clarifier-flux and layer-envelope
residuals are evaluated only where a mechanistic layer state exists; the
route-$S$ trust region contains its whole-system leverage, whereas route $U$
contains five reactor leverages and one Clarifier leverage. Root acceptance is
an evaluation prerequisite, not an upper trust inequality. Mass-conservation and non-negativity audits are
written before the article tables. A failed or
unresolved case remains in the denominator and does not suppress subsequent
cases or downstream audits. Runtime summaries are calculated only from the
durations recorded for all three routes in robustness cases 01--10, including
route-$U$ root work; no repeated timing benchmark is
run over the post-selection holdout block. There is no 10-minute ceiling in
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
target directory. The assessment uses deterministic process batches (64 rows
by default), one numerical thread per worker, and atomic input-bound
checkpoints below `assessment_checkpoints`; completed batches are reused in
fixed row order even when the worker count changes. Configure concurrency with
`ARTICLE_V3_ASSESSMENT_WORKERS` and batch geometry with
`ARTICLE_V3_ASSESSMENT_BATCH_SIZE`. A schema-10 two-route directory is never resumed as a
schema-11 three-route directory. The first three-route calculation uses a fresh
target such as `article_full_50000_three_route_001` and may reuse only verified
mechanistic-generation artifacts from its declared predecessor. A later rerun
must use a new target ID and name its prior run
with `ARTICLE_V3_REUSE_FROM_RUN_ID`; the driver creates another self-contained,
hash-pinned copy rather than overwriting the earlier run. Candidate and
accepted-row checkpoints are accepted only when their source, profile, input,
stream-state, and provenance bindings match. Any mismatch is a run-integrity
failure and is not hidden by regeneration.
"""),
]

def build_notebook():
    """Return the canonical notebook with stable cell identifiers."""

    for index, cell in enumerate(cells):
        cell["id"] = f"article-v3-{index:02d}"
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
                "profile": "article_frozen_16714",
                "response_schema": "clarifier_inventory_v1",
                "schema": 7,
                "runner_schema": 11,
                "routes": ["surrogate", "shared_unit", "direct"],
                "route_symbols": {"surrogate": "S", "shared_unit": "U", "direct": "M"},
            },
        },
    )
    nbf.validate(notebook)
    return notebook


def write_notebook(path: Path = NOTEBOOK) -> Path:
    """Write the canonical notebook to ``path`` and return the resolved path."""

    destination = path.resolve()
    nbf.write(build_notebook(), destination)
    return destination


if __name__ == "__main__":
    print(write_notebook())
