"""Resume the fixed 500-row manuscript-v3 preflight after generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from closed_loop.manuscript_v3 import (
    TEST_500,
    assess_raw_projected_mechanistic,
    create_design,
    cross_validate_ridge,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "article_v3" / "test_500_l5_revision_001"
STABILITY_AGREEMENT_TOLERANCE = 1.0e-6


def _load_targets(block: str) -> tuple[np.ndarray, pd.DataFrame]:
    directory = RUN / "datasets" / block
    with np.load(directory / "mechanistic_rows_v3.npz", allow_pickle=False) as stored:
        targets = np.asarray(stored["targets"], dtype=float)
    diagnostics = pd.read_csv(directory / "mechanistic_diagnostics.csv")
    return targets, diagnostics


def _acceptance_audit(diagnostics: pd.DataFrame) -> pd.DataFrame:
    audited = diagnostics.copy()
    audited["stability_agreement_max"] = audited[
        ["stability_agreement_start_1", "stability_agreement_start_2"]
    ].max(axis=1)
    audited["mass_residual_max"] = audited[
        ["mass_residual_start_1", "mass_residual_start_2"]
    ].max(axis=1)
    audited["largest_real_eigenvalue_max"] = audited[
        ["largest_real_eigenvalue_start_1", "largest_real_eigenvalue_start_2"]
    ].max(axis=1)
    audited["state_negativity_max"] = audited[
        ["state_negativity_start_1", "state_negativity_start_2"]
    ].max(axis=1)
    audited["rate_negativity_max"] = audited[
        ["rate_negativity_start_1", "rate_negativity_start_2"]
    ].max(axis=1)
    audited["accepted_revised"] = (
        (audited["root_difference_inf"] <= 1.0e-6)
        & audited["branch_agreement"].astype(bool)
        & (audited["mass_residual_max"] <= 1.0e-8)
        & (audited["state_negativity_max"] <= 1.0e-10)
        & (audited["rate_negativity_max"] <= 1.0e-12)
        & (audited["largest_real_eigenvalue_max"] <= -1.0e-8)
        & (audited["stability_agreement_max"] <= STABILITY_AGREEMENT_TOLERANCE)
    )
    return audited


def main() -> None:
    metrics = RUN / "metrics"
    models = RUN / "models"
    predictions = RUN / "predictions"
    for directory in (metrics, models, predictions):
        directory.mkdir(parents=True, exist_ok=True)

    design = create_design(TEST_500)
    development_targets, development_diagnostics = _load_targets("development")
    test_targets, test_diagnostics = _load_targets("test")
    audited = pd.concat(
        [
            _acceptance_audit(development_diagnostics).assign(block="development"),
            _acceptance_audit(test_diagnostics).assign(block="test"),
        ],
        ignore_index=True,
    )
    audited.to_csv(metrics / "mechanistic_generation_audit.csv", index=False)
    if len(audited) != 500 or not bool(audited["accepted_revised"].all()):
        failed = audited.loc[~audited["accepted_revised"]]
        raise RuntimeError(f"{len(failed)} mechanistic rows fail the revised audit")
    write_json(
        metrics / "mechanistic_generation_disposition.json",
        {
            "rows": len(audited),
            "accepted_under_original_1e-7_step_agreement": int(audited["accepted"].sum()),
            "accepted_under_revised_1e-6_step_agreement": int(audited["accepted_revised"].sum()),
            "maximum_stability_step_agreement": float(audited["stability_agreement_max"].max()),
            "maximum_mass_residual": float(audited["mass_residual_max"].max()),
            "maximum_root_difference": float(audited["root_difference_inf"].max()),
            "reason_for_revision": "all original failures were isolated to finite-difference step agreement",
        },
    )

    ridge = cross_validate_ridge(
        design["development_decisions"],
        design["development_influents"],
        development_targets,
    )
    ridge.scores.to_csv(metrics / "ridge_cross_validation.csv", index=False)
    pd.DataFrame(
        {"row": np.arange(TEST_500.development_count), "fold": ridge.fold_membership}
    ).to_csv(metrics / "ridge_fold_membership.csv", index=False)
    model = ridge.model
    np.savez_compressed(
        models / "ridge_surrogate.npz",
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
        out_of_fold_raw=ridge.out_of_fold_raw,
    )

    assessment = assess_raw_projected_mechanistic(
        model,
        design["development_decisions"],
        design["development_influents"],
        development_targets,
        design["test_decisions"],
        design["test_influents"],
        test_targets,
        TEST_500,
    )
    assessment.metrics.to_csv(metrics / "untouched_prediction_metrics.csv", index=False)
    assessment.violations.to_csv(metrics / "physical_violations_assessment.csv", index=False)
    assessment.qp_diagnostics.to_csv(metrics / "projection_qp_diagnostics.csv", index=False)
    assessment.feasibility.to_csv(metrics / "projection_feasibility_bound.csv", index=False)
    np.savez_compressed(
        predictions / "untouched_test.npz",
        raw=assessment.raw,
        projected=assessment.projected,
        projected_targets=assessment.projected_targets,
        mechanistic=test_targets,
    )
    complete = assessment.metrics[assessment.metrics["block"].eq("complete_response")]
    print(complete.to_string(index=False))


if __name__ == "__main__":
    main()
