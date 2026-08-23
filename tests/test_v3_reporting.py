from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

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
    np.savez_compressed(root / "models" / "ridge_surrogate.npz", response_scale=np.ones(165))
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
            "clarifier_flux": 1.5,
        },
    )


class ReportingSnapshotTests(unittest.TestCase):
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
            # Preserve the response scale needed elsewhere while adding the
            # one-time ridge-training duration consumed by the timing table.
            np.savez_compressed(
                run / "models" / "ridge_surrogate.npz",
                response_scale=np.ones(165),
                elapsed_seconds=np.asarray([17.0]),
            )
            pd.DataFrame({"elapsed_seconds": [11.0, 13.0]}).to_csv(
                run / "metrics" / "mechanistic_generation_summary.csv",
                index=False,
            )
            pd.DataFrame([
                {
                    "category": "raw_inference",
                    "elapsed_seconds": 10.0,
                    "per_response_latency_seconds": 0.01,
                },
                {
                    "category": "raw_inference",
                    "elapsed_seconds": 30.0,
                    "per_response_latency_seconds": 0.03,
                },
                {
                    "category": "qp_deployment",
                    "elapsed_seconds": 20.0,
                    "per_response_latency_seconds": 0.02,
                },
                {
                    "category": "qp_deployment",
                    "elapsed_seconds": 40.0,
                    "per_response_latency_seconds": 0.04,
                },
            ]).to_csv(
                run / "metrics" / "inference_timing_batches.csv", index=False,
            )
            # This fallback stream must be ignored when the dedicated batch
            # artifact exists.
            pd.DataFrame([
                {
                    "category": "raw_inference",
                    "elapsed_seconds": 999.0,
                    "per_response_latency_seconds": 999.0,
                }
            ]).to_csv(run / "metrics" / "timing_events.csv", index=False)
            _write_json(
                run / "optimization" / "nominal" / "direct.json",
                _direct_payload(),
            )
            _write_json(
                run / "optimization" / "nominal" / "direct_equivalence.json",
                {
                    "elapsed_seconds": 101.0,
                    "reference_replay": {"elapsed_seconds": 7.0},
                },
            )

            bundle = build_reporting_tables(run)
            timing = bundle["timing_summary"].set_index("category")
            self.assertEqual(timing.loc["raw_inference", "unit"], "seconds_per_response")
            self.assertEqual(timing.loc["qp_deployment", "unit"], "seconds_per_response")
            self.assertAlmostEqual(timing.loc["raw_inference", "total"], 0.04)
            self.assertAlmostEqual(timing.loc["raw_inference", "mean"], 0.02)
            self.assertAlmostEqual(timing.loc["qp_deployment", "total"], 0.06)
            self.assertAlmostEqual(timing.loc["qp_deployment", "mean"], 0.03)
            self.assertAlmostEqual(timing.loc["mechanistic_generation", "total"], 24.0)
            self.assertEqual(timing.loc["mechanistic_generation", "count"], 2)
            self.assertAlmostEqual(timing.loc["training", "total"], 17.0)
            self.assertEqual(timing.loc["training", "count"], 1)
            self.assertAlmostEqual(timing.loc["fixed_input_equivalence", "total"], 101.0)
            self.assertAlmostEqual(timing.loc["reference_replay", "total"], 7.0)
            self.assertNotEqual(
                timing.loc["reference_replay", "total"],
                timing.loc["fixed_input_equivalence", "total"],
            )
            workload = bundle["timing_workload"].set_index("route")
            self.assertAlmostEqual(
                workload.loc["direct", "reference_validation_time_seconds"],
                7.0,
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
            self.assertEqual(physical.loc[("untouched_test", "raw"), "record_count"], 1)
            self.assertEqual(
                physical.loc[("untouched_test", "raw"), "mass_conservation_violation_count"],
                3,
            )
            self.assertEqual(
                physical.loc[("selected_decisions", "reference"), "availability"],
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

            bundle = build_reporting_tables(run)
            detail = bundle["physical_violation_detail"]
            selected = detail[
                (detail["analysis_scope"] == "selected_decisions")
                & (detail["method"] == "smooth")
            ]
            self.assertEqual(len(selected), 1)
            self.assertTrue(np.isfinite(selected.iloc[0]["mass_conservation_violation_max"]))
            summary = bundle["physical_violation_summary"].set_index(
                ["analysis_scope", "method"]
            )
            self.assertEqual(summary.loc[("selected_decisions", "smooth"), "availability"], "available")
            self.assertEqual(len(bundle["process_profiles"]), 37)

            output = Path(temporary) / "report"
            written = bundle.write(output)
            self.assertTrue(written["physical_violation_summary"].is_file())
            self.assertTrue(written["manifest"].is_file())
            manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["expected_cases"], ["nominal"])
            self.assertEqual(
                manifest["table_rows"]["physical_violation_summary"],
                len(bundle["physical_violation_summary"]),
            )


if __name__ == "__main__":
    unittest.main()
