"""Build the manuscript-v3 execution notebook from auditable source cells."""

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "main_closed_loop.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


cells = [
    markdown(r"""
# Manuscript-v3 closed-loop study

This is the canonical executable companion to `article/wip_v3/manuscript.tex`
and `article/wip_v3/supplementary_material.tex`. It follows their six frozen
phases: two independent Latin-hypercube blocks; two-start nonsmooth
mechanistic generation; five-fold ridge selection using raw error only; one
untouched raw/projected assessment; nominal plus influent-scenario analysis;
and independent nonsmooth replay.

The notebook records mass-conservation and non-negativity violations for every
raw, projected, and mechanistic response. A failed row or gate remains visible
and stops the applicable scientific phase; it is never replaced or tuned away.

The default verification profile is deliberately article-ineligible: 400
development + 100 untouched test rows, five robustness influents, and five
Clarifier layers. The article profile restores the declared 800 + 200 rows,
ten robustness influents, and ten layers. Verification and article designs use
different seeds, so verification observations cannot leak into article results.
"""),
    code(r"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

from closed_loop.manuscript_v3 import (
    ARTICLE_FULL, TEST_500, assess_raw_projected_mechanistic,
    create_design, generate_mechanistic_block, optimize_surrogate_case,
    replay_selected_case, select_ridge, write_json,
)

ROOT = Path.cwd().resolve()
if not (ROOT / "article" / "wip_v3" / "manuscript.tex").is_file():
    raise RuntimeError("Run this notebook from the surrogate-optimization-arch root.")

PROFILE_NAME = os.environ.get("ARTICLE_V3_PROFILE", "test_500_l5")
PROFILES = {"test_500_l5": TEST_500, "article_full": ARTICLE_FULL}
if PROFILE_NAME not in PROFILES:
    raise ValueError(f"ARTICLE_V3_PROFILE must be one of {tuple(PROFILES)}")
PROFILE = PROFILES[PROFILE_NAME]
RUN_ID = os.environ.get("ARTICLE_V3_RUN_ID", PROFILE.name)
GATES_BYPASSED = os.environ.get("ARTICLE_V3_BYPASS_ADMISSION_GATE", "0") == "1"
RUN_ROOT = ROOT / "results" / "article_v3" / RUN_ID
for name in ("inputs", "datasets", "models", "predictions", "metrics", "optimization", "report"):
    (RUN_ROOT / name).mkdir(parents=True, exist_ok=True)

contract = {
    "profile": PROFILE.__dict__,
    "manuscript": "article/wip_v3/manuscript.tex",
    "supplement": "article/wip_v3/supplementary_material.tex",
    "decision_order": ["H", "a_3", "a_4", "a_5", "r_I", "r_R", "w"],
    "response_order": "mixer, reactors 1..5, overflow flow, underflow flow, layers 1..L",
    "failure_policy": "stop without replacement",
    "admission_gate_bypassed_by_user": GATES_BYPASSED,
}
write_json(RUN_ROOT / "inputs" / "contract.json", contract)
display(pd.Series(contract["profile"], name="value").to_frame())
"""),
    markdown(r"""
## 1. Frozen design and dimensional audit

The two 27-dimensional blocks restart independent SplitMix64 streams. The
seven controls are ordered exactly as in Eq. (casecontrols); the remaining
twenty coordinates use the manuscript component order. The feature count is
checked from the formula rather than hard-coded.
"""),
    code(r"""
design = create_design(PROFILE)
expected_features = 1 + 27 + 27 * 28 // 2
expected_response = (5 + 3) * 20 + PROFILE.layer_count
assert expected_features == 406
assert PROFILE.response_count == expected_response
assert design["development_decisions"].shape == (PROFILE.development_count, 7)
assert design["test_decisions"].shape == (PROFILE.test_count, 7)
assert design["robustness_influents"].shape == (PROFILE.robustness_count, 20)
np.savez_compressed(RUN_ROOT / "datasets" / "design.npz", **{
    key: value for key, value in design.items() if isinstance(value, np.ndarray)
})
write_json(RUN_ROOT / "inputs" / "generator_records.json", design["generators"])
print({"features": expected_features, "responses": expected_response,
       "development": PROFILE.development_count, "test": PROFILE.test_count,
       "robustness": PROFILE.robustness_count, "layers": PROFILE.layer_count})
"""),
    markdown(r"""
## 2. Two-start nonsmooth mechanistic generation

Every prescribed point is solved from both influent-defined starts. Both
routes must pass the scaled balances, non-negativity, Clarifier envelope,
reduced stability, and root-agreement checks. Checkpoints are resumable; a
failed point is reported and stops the design without resampling.
"""),
    code(r"""
development_targets, development_diagnostics = generate_mechanistic_block(
    design["development_decisions"], design["development_influents"], PROFILE,
    RUN_ROOT / "datasets" / "development",
)
test_targets, test_diagnostics = generate_mechanistic_block(
    design["test_decisions"], design["test_influents"], PROFILE,
    RUN_ROOT / "datasets" / "test",
)
generation_summary = pd.DataFrame([
    {"block": "development", "rows": len(development_diagnostics),
     "accepted": int(development_diagnostics.accepted.sum()),
     "max_mass_residual": development_diagnostics[["mass_residual_start_1", "mass_residual_start_2"]].max().max(),
     "minimum_state": development_diagnostics[["minimum_state_start_1", "minimum_state_start_2"]].min().min()},
    {"block": "test", "rows": len(test_diagnostics),
     "accepted": int(test_diagnostics.accepted.sum()),
     "max_mass_residual": test_diagnostics[["mass_residual_start_1", "mass_residual_start_2"]].max().max(),
     "minimum_state": test_diagnostics[["minimum_state_start_1", "minimum_state_start_2"]].min().min()},
])
generation_summary.to_csv(RUN_ROOT / "metrics" / "mechanistic_generation_summary.csv", index=False)
display(generation_summary)
"""),
    markdown(r"""
## 3. Five-fold ridge selection

Development rows are permuted once from state 271828 and split into five
folds. All centers and scales are fitted again inside each fold. The selected
penalty is the largest value within one descriptive fold standard error of the
minimum raw complete-state nRMSE. Projection results never select the penalty.
"""),
    code(r"""
model, cv_scores = select_ridge(
    design["development_decisions"], design["development_influents"],
    development_targets,
)
assert model.feature_map.feature_count == 406
assert model.response_count == PROFILE.response_count
cv_scores.to_csv(RUN_ROOT / "metrics" / "ridge_cross_validation.csv", index=False)
np.savez_compressed(
    RUN_ROOT / "models" / "ridge_surrogate.npz",
    decision_center=model.feature_map.decision_center,
    decision_scale=model.feature_map.decision_scale,
    influent_center=model.feature_map.influent_center,
    influent_scale=model.feature_map.influent_scale,
    term_center=model.feature_map.term_center,
    term_scale=model.feature_map.term_scale,
    response_center=model.response_center,
    response_scale=model.response_scale,
    coefficients=model.coefficients,
    ridge_penalty=np.asarray(model.ridge_penalty),
)
display(cv_scores.groupby("gamma")[["mean", "standard_error", "selected"]].first())
"""),
    markdown(r"""
## 4. Untouched raw/projected/mechanistic assessment

The test block is opened once. Raw and projected predictions use the same
frozen ridge model. For every test row, the physical ledger records scaled
mass/network equality residuals, network-direction violations, minimum
coordinate, and scaled non-negativity violation count and magnitude. The same
ledger includes the independently generated mechanistic reference.
"""),
    code(r"""
prediction_metrics, assessment_violations, raw_test, projected_test = (
    assess_raw_projected_mechanistic(
        model, design["development_decisions"], design["development_influents"],
        development_targets, design["test_decisions"], design["test_influents"],
        test_targets, PROFILE,
    )
)
prediction_metrics.to_csv(RUN_ROOT / "metrics" / "untouched_prediction_metrics.csv", index=False)
assessment_violations.to_csv(RUN_ROOT / "metrics" / "physical_violations_assessment.csv", index=False)
np.savez_compressed(RUN_ROOT / "predictions" / "untouched_test.npz",
                    raw=raw_test, projected=projected_test, mechanistic=test_targets)
display(prediction_metrics)
display(assessment_violations.groupby("method").agg(
    mass_max=("mass_conservation_violation_max", "max"),
    mass_rows_violating=("mass_conservation_violation_count", lambda x: int((x > 0).sum())),
    nonnegative_max=("nonnegativity_violation_max", "max"),
    nonnegative_rows_violating=("nonnegativity_violation_count", lambda x: int((x > 0).sum())),
    minimum_coordinate=("minimum_coordinate", "min"),
))
raw_gate = float(prediction_metrics.loc[prediction_metrics.method.eq("raw"), "nrmse"].iloc[0]) < 1.0
projection_gate = assessment_violations.loc[
    assessment_violations.method.eq("projected"),
    "mass_conservation_violation_count",
].eq(0).all()
ADMISSION_PASSED = bool(raw_gate and projection_gate)
OPTIMIZATION_AUTHORIZED = bool(ADMISSION_PASSED or GATES_BYPASSED)
write_json(RUN_ROOT / "metrics" / "admission_gate.json", {
    "passed": ADMISSION_PASSED,
    "bypassed_by_user": GATES_BYPASSED,
    "optimization_authorized": OPTIMIZATION_AUTHORIZED,
    "raw_nrmse_below_one": bool(raw_gate),
    "all_projected_mass_audits_passed": bool(projection_gate),
    "failure_action": None if ADMISSION_PASSED else "optimization prohibited without refitting",
})
print({"admission_passed": ADMISSION_PASSED,
       "bypassed_by_user": GATES_BYPASSED,
       "optimization_authorized": OPTIMIZATION_AUTHORIZED})
"""),
    markdown(r"""
## 5. Nominal and influent-scenario stress test

The reduced verification performs the manuscript's nine-start projected
surrogate outer refinement for the midpoint influent and all requested fresh
influent scenarios. Every selected point is then replayed independently by the
nonsmooth mechanism from both declared starts. Endpoint stationarity remains
explicitly labeled unresolved in this reduced profile; article release is not
authorized by a reduced test.
"""),
    code(r"""
nominal = (np.asarray([0.0, 20.0, 5.0, 12.0, 0.0, 0.0, 0.0, 2.0, 10.0, 1.6,
                       20.0, 60.0, 15.0, 5.0, 2.0, 1.0, 0.5, 0.5, 0.0, 0.0]) +
           np.asarray([0.5, 180.0, 80.0, 55.0, 3.0, 8.0, 2.0, 18.0, 90.0, 5.2,
                       120.0, 280.0, 100.0, 60.0, 20.0, 30.0, 8.0, 8.0, 12.0, 12.0])) / 2
case_influents = [("nominal", nominal)] + [
    (f"robustness_{i + 1:02d}", row)
    for i, row in enumerate(design["robustness_influents"])
]
case_rows, violation_frames = [], []
for case, influent in case_influents:
    if OPTIMIZATION_AUTHORIZED:
        selected = optimize_surrogate_case(
            model, influent, design["development_decisions"],
            design["development_influents"], development_targets, PROFILE,
        )
        replay, violations = replay_selected_case(
            case, selected, influent, model, design["development_decisions"],
            design["development_influents"], development_targets, PROFILE,
        )
        case_rows.append(replay)
        if len(violations):
            violation_frames.append(violations)
    else:
        case_rows.append({
            "case": case,
        "status": "not attempted: predeclared untouched-test admission gate failed",
        })
case_results = pd.DataFrame(case_rows)
case_violations = pd.concat(violation_frames, ignore_index=True) if violation_frames else pd.DataFrame()
case_results.to_csv(RUN_ROOT / "optimization" / "case_results.csv", index=False)
case_violations.to_csv(RUN_ROOT / "metrics" / "physical_violations_selected_cases.csv", index=False)
display(case_results)
display(case_violations)
"""),
    markdown(r"""
## 6. Test disposition and article-run lock

The verification run is complete only when its 500 fixed design rows, five
scenario rows plus nominal case, and every physical ledger are present. It can
validate implementation behavior but cannot supply article results. The full
profile is intentionally separate and uses the manuscript's 800/200 design,
ten scenarios, and ten-layer Clarifier.
"""),
    code(r"""
all_violations = pd.concat([assessment_violations, case_violations], ignore_index=True)
summary = {
    "profile": PROFILE.name,
    "article_eligible": PROFILE.article_eligible,
    "generated_dataset_count": PROFILE.development_count + PROFILE.test_count,
    "robustness_case_count": PROFILE.robustness_count,
    "clarifier_layer_count": PROFILE.layer_count,
    "ridge_penalty": model.ridge_penalty,
    "raw_test_nrmse": float(prediction_metrics.loc[prediction_metrics.method.eq("raw"), "nrmse"].iloc[0]),
    "projected_test_nrmse": float(prediction_metrics.loc[prediction_metrics.method.eq("projected"), "nrmse"].iloc[0]),
    "maximum_mass_violation_by_method": all_violations.groupby("method")["mass_conservation_violation_max"].max().to_dict(),
    "maximum_nonnegativity_violation_by_method": all_violations.groupby("method")["nonnegativity_violation_max"].max().to_dict(),
    "admission_gate_bypassed_by_user": GATES_BYPASSED,
    "status": (
        "verification_complete" if OPTIMIZATION_AUTHORIZED and not PROFILE.article_eligible
        else "article_run_complete" if OPTIMIZATION_AUTHORIZED
        else "stopped_at_untouched_test_admission_gate"
    ),
}
write_json(RUN_ROOT / "report" / "summary.json", summary)
display(pd.Series(summary, name="value").to_frame())
"""),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
)
nbf.write(notebook, NOTEBOOK)
print(NOTEBOOK)
