from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from closed_loop import v3_reporting as reporting
from closed_loop.v3_reporting import (
    FAILURE_CLASSES,
    PENDING_CLASS,
    build_reporting_tables,
)


THETA = np.asarray([18.0, 0.2, 0.3, 0.4, 2.0, 0.75, 0.02])


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _direct_payload(*, with_selection: bool = True) -> dict[str, object]:
    start = {
        "start_index": 0,
        "nearest_development_row": 0,
        "status": "first_order_kkt_stationary_feasible",
        "objective": 0.75,
        "feasible": True,
        "stationary": True,
        "theta": THETA.tolist(),
        "stages": [
            {
                "epsilon": 1.0e-6,
                "receiver_half_width": 10.0,
                "status": "Solve_Succeeded",
                "solver_success": True,
                "elapsed_seconds": 2.0,
                "iterations": 4,
                "feasible": True,
            }
        ],
        "kkt": {"active_inequality_count": 2},
    }
    return {
        "status": "selected_stationary" if with_selection else "no_validated_feasible_start",
        "selected_start": 0 if with_selection else None,
        "starts": [start],
        "elapsed_seconds": 3.0,
        "preflight_stage_wall_time_seconds": 600.0,
    }


def _make_run(root: Path, robustness_count: int = 2) -> None:
    rng = np.random.default_rng(4)
    development_count = 12
    decisions = np.tile(THETA, (development_count, 1))
    decisions += rng.normal(0.0, 0.01, size=decisions.shape)
    decisions[:, 0] = np.clip(decisions[:, 0], 6.0, 36.0)
    decisions[:, 1:4] = np.clip(decisions[:, 1:4], 0.0, 1.0)
    decisions[:, 4] = np.clip(decisions[:, 4], 0.0, 4.0)
    decisions[:, 5] = np.clip(decisions[:, 5], 0.25, 1.25)
    decisions[:, 6] = np.clip(decisions[:, 6], 0.001, 0.05)
    influents = rng.uniform(1.0, 20.0, size=(development_count, 20))
    targets = rng.uniform(1.0, 100.0, size=(development_count, 165))
    (root / "datasets" / "development").mkdir(parents=True)
    np.savez_compressed(
        root / "datasets" / "design.npz",
        development_decisions=decisions,
        development_influents=influents,
        robustness_influents=rng.uniform(1.0, 20.0, size=(robustness_count, 20)),
    )
    np.savez_compressed(
        root / "datasets" / "development" / "mechanistic_rows_v3.npz",
        targets=targets,
    )
    (root / "models").mkdir(parents=True)
    np.savez_compressed(root / "models" / "ridge_surrogate.npz", response_scale=np.ones(161))
    (root / "metrics").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "case": "test_0000",
                "method": "raw",
                "mass_conservation_violation_max": 2.0,
                "mass_conservation_violation_count": 3,
                "nonnegativity_violation_max": 0.25,
                "nonnegativity_violation_count": 1,
                "minimum_coordinate": -0.25,
            },
            {
                "case": "test_0000",
                "method": "projected",
                "mass_conservation_violation_max": 1.0e-11,
                "mass_conservation_violation_count": 0,
                "nonnegativity_violation_max": 0.0,
                "nonnegativity_violation_count": 0,
                "minimum_coordinate": 0.0,
            },
        ]
    ).to_csv(root / "metrics" / "physical_violations_assessment.csv", index=False)
    _write_json(
        root / "metrics" / "trust_limits.json",
        {
            "correction": 0.5,
            "regularized_leverage": 2.0,
            "particulate_split": 0.1,
            "reactor_residual": 1.0,
        },
    )


class ReportingSnapshotTests(unittest.TestCase):
    def test_reporting_geometry_separates_surrogate_and_mechanistic_widths(self) -> None:
        geometry = reporting.StudyGeometry(layer_count=5, layer_volume_m3=1_200.0)
        self.assertEqual(geometry.surrogate_response_count, 161)
        self.assertEqual(geometry.response_count, 161)
        self.assertEqual(geometry.mechanistic_response_count, 165)
        self.assertEqual(geometry.mechanistic_state_count, 105)

        full = np.arange(165, dtype=float)
        full[-5:] = np.arange(1.0, 6.0)
        reduced = reporting._as_reduced_response(
            full, geometry, allow_mechanistic=True,
        )
        self.assertIsNotNone(reduced)
        assert reduced is not None
        self.assertEqual(reduced.shape, (161,))
        np.testing.assert_array_equal(reduced[:160], full[:160])
        self.assertEqual(reduced[-1], 1_200.0 * sum(range(1, 6)))
        self.assertIsNone(reporting._as_reduced_response(
            full, geometry, allow_mechanistic=False,
        ))

    def test_shared_engineering_uses_scalar_clarifier_inventory(self) -> None:
        geometry = reporting.StudyGeometry(layer_count=5, layer_volume_m3=1_200.0)
        response = np.zeros(geometry.surrogate_response_count)
        response[geometry.inventory_index] = 5_432.0
        quantities, _, _ = reporting._response_quantities(
            THETA, response, geometry, np.ones(4),
        )
        self.assertEqual(quantities["clarifier_solids_inventory"], 5_432.0)
        self.assertEqual(quantities["solids_inventory"], 5_432.0)

    def test_trust_reporting_has_four_reduced_response_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _make_run(run, robustness_count=0)
            table = reporting._trust_table(run, (), [])
            self.assertEqual(tuple(table["diagnostic"]), reporting.TRUST_DIAGNOSTICS)
            self.assertNotIn("clarifier_flux", set(table["diagnostic"]))

    def test_schema_10_trust_uses_only_post_selection_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _make_run(run, robustness_count=0)
            _write_json(run / "inputs" / "contract.json", {
                "runner_schema": 10,
                "response_schema": {"name": "clarifier_inventory_v1"},
                "profile": {"layer_count": 5},
            })
            legacy = pd.DataFrame([{
                "correction": 99.0,
                "regularized_leverage": 99.0,
                "particulate_split": 99.0,
                "reactor_residual": 99.0,
                "clarifier_flux": 99.0,
            }])
            legacy.to_csv(
                run / "metrics" / "trust_untouched_test.csv", index=False,
            )
            warnings: list[str] = []
            table = reporting._trust_table(run, (), warnings)
            self.assertTrue((table["test_count"] == 0).all())
            self.assertTrue(any("superseded schema-9" in item for item in warnings))

            pd.DataFrame([
                {
                    "correction": 0.1,
                    "regularized_leverage": 1.0,
                    "particulate_split": 0.2,
                    "reactor_residual": 0.3,
                },
                {
                    "correction": 0.4,
                    "regularized_leverage": 1.5,
                    "particulate_split": 0.5,
                    "reactor_residual": 0.6,
                },
            ]).to_csv(
                run / "metrics" / "trust_post_selection_holdout.csv", index=False,
            )
            table = reporting._trust_table(run, (), [])
            self.assertTrue((table["test_count"] == 2).all())
            correction = table.set_index("diagnostic").loc["correction"]
            self.assertEqual(correction["test_p95"], 0.4)

    def test_physical_summary_reports_reduced_inequality_families(self) -> None:
        detail = pd.DataFrame([{
            "analysis_scope": "post_selection_holdout",
            "case": "test_0000",
            "method": "raw",
            "mass_conservation_violation_max": 0.0,
            "mass_conservation_violation_count": 0,
            "nonnegativity_violation_max": 0.0,
            "nonnegativity_violation_count": 0,
            "particulate_densification_violation_max": 0.25,
            "clarifier_inventory_bound_violation_max": 0.75,
        }])
        summary = reporting._physical_summary(detail)
        row = summary[
            (summary["analysis_scope"] == "all_analysis")
            & (summary["method"] == "raw")
        ].iloc[0]
        self.assertEqual(row["particulate_densification_violation_max"], 0.25)
        self.assertEqual(row["clarifier_inventory_bound_violation_max"], 0.75)
        self.assertNotIn("mass_tss_endpoint_max", summary.columns)

    def test_scope_specific_nonlinear_audit_uses_only_saved_full_states(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _make_run(run, robustness_count=0)
            geometry = reporting.StudyGeometry(5, 1_200.0)
            states = np.ones((1, geometry.mechanistic_state_count))
            states[0, -5:] = np.arange(1.0, 6.0)
            alternate = states.copy()
            alternate[0, -5:] = np.arange(5.0, 0.0, -1.0)
            np.savez_compressed(
                run / "datasets" / "development" / "mechanistic_accepted_v3.npz",
                targets=np.ones((1, geometry.mechanistic_response_count)),
                states_start_1=states,
                states_start_2=alternate,
            )
            np.savez_compressed(
                run / "datasets" / "development" / "accepted_inputs.npz",
                decisions=THETA[None, :],
                influents=np.ones((1, 20)),
            )
            pd.DataFrame([{
                "accepted_slot": 0,
                "largest_real_eigenvalue_start_1": -0.2,
                "largest_real_eigenvalue_start_2": -0.1,
                "stability_agreement_start_1": 1.0e-8,
                "stability_agreement_start_2": 2.0e-8,
                "locally_stable_start_1": True,
                "locally_stable_start_2": True,
            }]).to_csv(
                run / "datasets" / "development" / "accepted_diagnostics.csv",
                index=False,
            )
            exact_diagnostics = {
                "diagnostics_start_1": {
                    "largest_real_eigenvalue": -0.3,
                    "stability_eigenvalue_agreement": 1.0e-8,
                    "locally_stable": True,
                },
                "diagnostics_start_2": {
                    "largest_real_eigenvalue": 0.0,
                    "stability_eigenvalue_agreement": 2.0e-6,
                    "locally_stable": False,
                },
            }
            snapshot = reporting.RouteSnapshot(
                case="nominal",
                route="surrogate",
                artifact_state="complete",
                outcome="selected",
                payload=None,
                starts=(),
                selected_start=0,
                selected={"final": {"theta": THETA.tolist()}},
                selected_arrays={
                    "theta": THETA,
                    "exact_state_start_1": states[0],
                    "exact_state_start_2": alternate[0],
                },
                equivalence=None,
                reference_arrays={},
                casewise_reference={
                    "candidate_available": True,
                    "reference": exact_diagnostics,
                },
                certification=None,
                recovery=None,
            )

            def balance_audit(state, *_args, **_kwargs):
                descending = bool(state[-1] < state[-2])
                return {
                    "balance_family_maxima": {
                        "clarifier_layer": 2.0e-8 if descending else 5.0e-9,
                    },
                    "balance_family_violation_counts": {
                        "clarifier_layer": 1 if descending else 0,
                    },
                }

            physical = pd.DataFrame({"method": ["raw", "projected"]})
            with patch.object(
                reporting, "mechanistic_balance_audit", side_effect=balance_audit,
            ) as audit:
                table = reporting._scope_specific_nonlinear_audit(
                    run, (snapshot,), ("nominal",), geometry, physical, [],
                ).set_index("source")

            self.assertEqual(audit.call_count, 4)
            for source in ("raw_reduced", "projected_reduced"):
                self.assertEqual(
                    table.loc[source, "applicability"],
                    "not_applicable_no_layer_state",
                )
                self.assertTrue(np.isnan(
                    table.loc[source, "layer_residual_max"]
                ))
            generation = table.loc["exact_mechanistic_generation"]
            self.assertEqual(generation["record_count"], 2)
            self.assertEqual(generation["audited_record_count"], 2)
            self.assertEqual(generation["layer_residual_violation_count"], 1)
            replay = table.loc["exact_mechanistic_replay"]
            self.assertEqual(replay["record_count"], 2)
            self.assertEqual(replay["layer_envelope_violation_count"], 6)
            self.assertEqual(replay["stability_violation_count"], 1)

    def test_physical_summary_excludes_unavailable_placeholders(self) -> None:
        detail = pd.DataFrame([
            {
                "analysis_scope": "selected_decision_common_reference",
                "case": "nominal:surrogate",
                "method": "optimizer_native",
                "audit_available": True,
                "mass_conservation_violation_max": 2.0e-8,
                "mass_conservation_violation_count": 2,
                "nonnegativity_violation_max": 3.0e-10,
                "nonnegativity_violation_count": 1,
                "minimum_coordinate": -3.0e-10,
                "network_inequality_violation_count": 4,
            },
            {
                "analysis_scope": "selected_decision_common_reference",
                "case": "robustness_01:surrogate",
                "method": "optimizer_native",
                "audit_available": False,
                "mass_conservation_violation_max": np.nan,
                "mass_conservation_violation_count": 0,
                "nonnegativity_violation_max": np.nan,
                "nonnegativity_violation_count": 0,
                "minimum_coordinate": np.nan,
                "network_inequality_violation_count": 0,
            },
            {
                # Missing availability is a legacy computed row, not a failed
                # placeholder; concatenated legacy tables acquire this NaN.
                "analysis_scope": "post_selection_holdout",
                "case": "test_0000",
                "method": "optimizer_native",
                "audit_available": np.nan,
                "mass_conservation_violation_max": 1.0e-9,
                "mass_conservation_violation_count": 0,
                "nonnegativity_violation_max": 0.0,
                "nonnegativity_violation_count": 0,
                "minimum_coordinate": 0.0,
                "network_inequality_violation_count": 0,
            },
        ])

        summary = reporting._physical_summary(detail)
        selected = summary[
            (summary["analysis_scope"] == "selected_decision_common_reference")
            & (summary["method"] == "optimizer_native")
        ].iloc[0]
        self.assertEqual(selected["availability"], "partially_available")
        self.assertEqual(selected["record_count"], 2)
        self.assertEqual(selected["audited_record_count"], 1)
        self.assertEqual(selected["unavailable_record_count"], 1)
        self.assertEqual(selected["audit_coverage_fraction"], 0.5)
        self.assertEqual(selected["mass_conservation_violation_count"], 2)
        self.assertEqual(selected["nonnegativity_violation_count"], 1)

        only_placeholder = detail.iloc[[1]].copy()
        unavailable = reporting._physical_summary(only_placeholder)
        unavailable = unavailable[
            (unavailable["analysis_scope"] == "selected_decision_common_reference")
            & (unavailable["method"] == "optimizer_native")
        ].iloc[0]
        self.assertEqual(unavailable["availability"], "not_available")
        self.assertEqual(unavailable["audited_record_count"], 0)
        self.assertEqual(unavailable["unavailable_record_count"], 1)
        self.assertTrue(np.isnan(unavailable["mass_conservation_violation_count"]))
        self.assertTrue(np.isnan(unavailable["nonnegativity_violation_count"]))

    def test_physical_summary_treats_legacy_rows_as_audited(self) -> None:
        legacy = pd.DataFrame([{
            "analysis_scope": "untouched_test",
            "case": "test_0000",
            "method": "mechanistic",
            "mass_conservation_violation_max": 1.0e-9,
            "mass_conservation_violation_count": 0,
            "nonnegativity_violation_max": 0.0,
            "nonnegativity_violation_count": 0,
        }])
        summary = reporting._physical_summary(legacy)
        row = summary[
            (summary["analysis_scope"] == "post_selection_holdout")
            & (summary["method"] == "mechanistic")
        ].iloc[0]
        self.assertEqual(row["availability"], "available")
        self.assertEqual(row["audited_record_count"], 1)
        self.assertEqual(row["unavailable_record_count"], 0)
        self.assertEqual(row["mass_conservation_violation_count"], 0)

    def test_route_status_uses_declared_single_attempt_and_reads_legacy_nine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _make_run(run, robustness_count=0)
            path = run / "optimization" / "nominal" / "direct.json"
            current = _direct_payload()
            current["optimization_attempt_count"] = 1
            _write_json(path, current)
            row = build_reporting_tables(run)["route_status"]
            row = row[(row["case"] == "nominal") & (row["route"] == "direct")].iloc[0]
            self.assertEqual(row["starts_expected"], 1)

            legacy = _direct_payload()
            _write_json(path, legacy)
            row = build_reporting_tables(run)["route_status"]
            row = row[(row["case"] == "nominal") & (row["route"] == "direct")].iloc[0]
            self.assertEqual(row["starts_expected"], 9)

    def test_replacement_generation_tables_and_effective_artifacts_are_preferred(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _make_run(run, robustness_count=0)
            with np.load(run / "datasets" / "design.npz", allow_pickle=False) as stored:
                effective = {name: np.asarray(stored[name]).copy() for name in stored.files}
            effective["development_decisions"] = effective["development_decisions"] + 0.125
            np.savez_compressed(run / "datasets" / "effective_design.npz", **effective)
            accepted_targets = np.full((12, 165), 7.0)
            np.savez_compressed(
                run / "datasets" / "development" / "mechanistic_accepted_v3.npz",
                targets=accepted_targets,
            )

            for block in ("development", "test"):
                directory = run / "datasets" / block
                directory.mkdir(parents=True, exist_ok=True)
                decisions = np.asarray([
                    [6.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.001],
                    [36.0, 1.0, 1.0, 1.0, 4.0, 1.25, 0.05],
                ])
                influents = np.vstack((np.ones(20), np.full(20, 2.0)))
                np.savez_compressed(
                    directory / "accepted_inputs.npz",
                    decisions=decisions,
                    influents=influents,
                )
                attempts = pd.DataFrame({
                    "candidate_id": [
                        f"{block}:r000000:c000000",
                        f"{block}:r000000:c000001",
                        f"{block}:r000001:c000000",
                    ],
                    "accepted": [True, False, True],
                    "rejection_reason": ["accepted", "branch_disagreement", "accepted"],
                    "rejected_solver_exception": [False, False, False],
                    "rejected_mass_or_residual": [False, False, False],
                    "rejected_stability": [False, False, False],
                    "rejected_nonnegativity": [False, False, False],
                    "rejected_domain": [False, False, False],
                    "rejected_root_distance": [False, False, False],
                    "rejected_branch_disagreement": [False, True, False],
                    "rejected_other_solver_rejection": [False, False, False],
                    "elapsed_seconds": [1.0, 2.0, 3.0],
                })
                attempts.to_csv(directory / "all_attempts.csv", index=False)
                pd.DataFrame({
                    "accepted_slot": [0, 1],
                    "source_candidate_id": [
                        f"{block}:r000000:c000000",
                        f"{block}:r000001:c000000",
                    ],
                    "source_candidate_round": [0, 1],
                }).to_csv(directory / "accepted_provenance.csv", index=False)
                pd.DataFrame({
                    "candidate_id": [
                        f"{block}:r000000:c000000",
                        f"{block}:r000000:c000001",
                    ],
                    "preserved_without_rewrite": [True, True],
                }).to_csv(directory / "base_checkpoint_migration.csv", index=False)
                _write_json(directory / "replacement_summary.json", {
                    "requested_accepted_count": 2,
                    "accepted_count": 2,
                    "base_attempt_count": 2,
                    "base_accepted_count": 1,
                    "supplemental_attempt_count": 1,
                    "supplemental_accepted_count": 1,
                    "supplemental_round_count": 1,
                })

            warnings: list[str] = []
            loaded_design = reporting._effective_design(run, warnings)
            loaded_targets = reporting._accepted_development(run, warnings)
            np.testing.assert_array_equal(
                loaded_design["development_decisions"],
                effective["development_decisions"],
            )
            np.testing.assert_array_equal(loaded_targets["targets"], accepted_targets)

            bundle = build_reporting_tables(run)
            summary = bundle["generation_summary"].set_index("block")
            self.assertEqual(summary.loc["development", "candidate_attempt_denominator"], 3)
            self.assertEqual(summary.loc["development", "accepted_row_denominator"], 2)
            self.assertEqual(summary.loc["development", "rejected_candidate_count"], 1)
            self.assertTrue(summary.loc["development", "accepted_slots_fully_traced"])
            self.assertFalse(summary.loc["development", "single_global_strength_one_lhs"])
            reasons = bundle["generation_rejection_reasons"]
            branch = reasons[
                (reasons["block"] == "development")
                & (reasons["rejection_reason"] == "branch_disagreement")
            ].iloc[0]
            self.assertEqual(branch["count"], 1)
            self.assertEqual(len(bundle["generation_accepted_coverage"]), 54)
            self.assertEqual(len(bundle["generation_attempt_ledger"]), 6)
            self.assertEqual(len(bundle["generation_accepted_provenance"]), 4)
            self.assertEqual(len(bundle["generation_checkpoint_migration"]), 4)

    def test_timing_tables_use_declared_artifact_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _make_run(run, robustness_count=0)
            np.savez_compressed(
                run / "models" / "ridge_surrogate.npz",
                response_scale=np.ones(161),
            )
            pd.DataFrame([
                {
                    "case": "robustness_01", "route": "surrogate",
                    "candidate_available": True,
                    "primary_optimization_seconds": 10.0,
                    "certification_seconds": 5.0,
                    "recovery_seconds": np.nan,
                    "complete_optimization_seconds": 15.0,
                    "exact_reference_seconds": 7.0,
                },
                {
                    "case": "robustness_02", "route": "surrogate",
                    "candidate_available": True,
                    "primary_optimization_seconds": 20.0,
                    "certification_seconds": 9.0,
                    "recovery_seconds": np.nan,
                    "complete_optimization_seconds": 29.0,
                    "exact_reference_seconds": 11.0,
                },
                {
                    "case": "robustness_01", "route": "direct",
                    "candidate_available": True,
                    "primary_optimization_seconds": 12.0,
                    "certification_seconds": np.nan,
                    "recovery_seconds": np.nan,
                    "complete_optimization_seconds": 12.0,
                    "exact_reference_seconds": 8.0,
                },
                {
                    "case": "robustness_02", "route": "direct",
                    "candidate_available": False,
                    "primary_optimization_seconds": 18.0,
                    "certification_seconds": np.nan,
                    "recovery_seconds": np.nan,
                    "complete_optimization_seconds": 18.0,
                    "exact_reference_seconds": np.nan,
                },
            ]).to_csv(
                run / "metrics" / "robustness_case_timing.csv",
                index=False,
            )

            bundle = build_reporting_tables(run)
            timing = bundle["timing_summary"].set_index("category")
            self.assertEqual(
                timing.loc["surrogate_complete_optimization", "unit"],
                "seconds_per_robustness_case",
            )
            self.assertAlmostEqual(
                timing.loc["surrogate_complete_optimization", "mean"], 22.0,
            )
            self.assertAlmostEqual(
                timing.loc["direct_complete_optimization", "mean"], 15.0,
            )
            self.assertAlmostEqual(
                timing.loc["surrogate_local_certification", "mean"], 7.0,
            )
            self.assertNotIn("raw_inference", timing.index)
            self.assertNotIn("qp_deployment", timing.index)
            workload = bundle["timing_workload"].set_index("route")
            self.assertAlmostEqual(
                workload.loc["direct", "reference_validation_time_seconds"],
                8.0,
            )

    def test_incomplete_cases_and_zero_count_failure_classes_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _make_run(run)
            _write_json(run / "optimization" / "nominal" / "direct.json", _direct_payload())
            _write_json(
                run / "optimization" / "nominal" / "direct_equivalence.json",
                {
                    "smooth_accepted": True,
                    "reference_accepted": True,
                    "accepted": True,
                    "state_rms": 1.0e-8,
                    "state_inf": 1.0e-7,
                    "own_smooth_residual": 1.0e-10,
                    "own_reference_residual": 1.0e-10,
                    "cross_residual": 1.0e-8,
                    "relative_objective_difference": 1.0e-8,
                    "engineering_difference": 1.0e-8,
                    "reference_root_difference_generation": 1.0e-8,
                    "reference_root_difference_state_scale": 1.0e-8,
                    "branch_agreement": True,
                    "feasibility_agreement": True,
                },
            )
            bundle = build_reporting_tables(run)

            self.assertEqual(
                bundle.expected_cases,
                ("nominal", "robustness_01", "robustness_02"),
            )
            self.assertEqual(len(bundle["route_status"]), 6)
            self.assertEqual(len(bundle["case_status"]), 3)
            nominal = bundle["case_status"].set_index("case").loc["nominal"]
            self.assertEqual(nominal["selected_n"], 1)
            self.assertEqual(nominal["mechanistic_disposition"], "validated result")
            pending = bundle["case_status"].set_index("case").loc["robustness_01"]
            self.assertEqual(pending["surrogate_disposition"], PENDING_CLASS)
            controls = bundle["selected_controls"]
            direct = controls[(controls["case"] == "nominal") & (controls["route"] == "direct")].iloc[0]
            self.assertAlmostEqual(direct["H"], 18.0)
            self.assertEqual(len(bundle["nominal_controls"]), 2)
            self.assertEqual(len(bundle["scenario_controls"]), 4)

            failure = bundle["failure_accounting"]
            self.assertEqual(len(failure), 2 * (len(FAILURE_CLASSES) + 1))
            projection = failure[
                (failure["route"] == "direct")
                & (failure["classification"] == "projection failure")
            ].iloc[0]
            self.assertEqual(projection["count"], 0)
            self.assertEqual(projection["denominator"], 3)

            physical = bundle["physical_violation_summary"].set_index(
                ["analysis_scope", "method"]
            )
            self.assertEqual(physical.loc[("post_selection_holdout", "raw"), "record_count"], 1)
            self.assertEqual(
                physical.loc[("post_selection_holdout", "raw"), "mass_conservation_violation_count"],
                3,
            )
            self.assertEqual(
                physical.loc[
                    ("selected_decision_common_reference", "exact_mechanistic_start_1"),
                    "availability",
                ],
                "not_available",
            )

    def test_selected_response_audit_is_reconstructed_and_tables_write_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _make_run(run, robustness_count=0)
            case = run / "optimization" / "nominal"
            _write_json(case / "direct.json", _direct_payload())
            case.mkdir(parents=True, exist_ok=True)
            response = np.linspace(1.0, 10.0, 165)
            np.savez_compressed(
                case / "direct_selected.npz",
                theta=THETA,
                response=response,
                state=np.ones(105),
            )
            pd.DataFrame([{
                "case": "nominal:direct",
                "method": "optimizer_native",
                "decision_route": "direct",
                "response_source": "optimizer_native",
                "audit_available": True,
                "mass_conservation_violation_max": 1.0e-9,
                "mass_conservation_violation_count": 0,
                "nonnegativity_violation_max": 0.0,
                "nonnegativity_violation_count": 0,
            }]).to_csv(
                run / "metrics" / "selected_response_physical_audit.csv",
                index=False,
            )

            bundle = build_reporting_tables(run)
            detail = bundle["physical_violation_detail"]
            selected = detail[
                (detail["analysis_scope"] == "selected_decision_common_reference")
                & (detail["method"] == "optimizer_native")
            ]
            self.assertEqual(len(selected), 1)
            self.assertTrue(np.isfinite(selected.iloc[0]["mass_conservation_violation_max"]))
            summary = bundle["physical_violation_summary"].set_index(
                ["analysis_scope", "method"]
            )
            self.assertEqual(
                summary.loc[
                    ("selected_decision_common_reference", "optimizer_native"),
                    "availability",
                ],
                "available",
            )
            self.assertEqual(len(bundle["process_profiles"]), 38)
            profiles = bundle["process_profiles"]
            inventory = profiles[profiles["location"].eq("clarifier_inventory")]
            layers = profiles[profiles["location"].str.startswith("clarifier_layer_")]
            self.assertEqual(len(inventory), 1)
            self.assertEqual(set(inventory["quantity"]), {"TSS_mass"})
            self.assertEqual(len(layers), 5)
            self.assertEqual(set(layers["response_method"]), {"smooth"})

            output = Path(temporary) / "report"
            written = bundle.write(output)
            self.assertTrue(written["physical_violation_summary"].is_file())
            self.assertTrue(written["scope_specific_nonlinear_audit"].is_file())
            self.assertTrue(written["manifest"].is_file())
            manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["expected_cases"], ["nominal"])
            self.assertEqual(
                manifest["table_rows"]["physical_violation_summary"],
                len(bundle["physical_violation_summary"]),
            )
            self.assertEqual(
                manifest["table_rows"]["scope_specific_nonlinear_audit"],
                len(bundle["scope_specific_nonlinear_audit"]),
            )


if __name__ == "__main__":
    unittest.main()
