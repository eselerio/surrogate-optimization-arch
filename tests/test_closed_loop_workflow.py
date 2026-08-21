from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from closed_loop import model as mechanism
from closed_loop.design import unit_latin_hypercube
from closed_loop.surrogate import (
    NetworkRowScales,
    QuadraticFeatureMap,
    QuadraticSurrogate,
)
from closed_loop.workflow import (
    ContractMismatchError,
    ClosedLoopWorkflow,
    MechanisticRow,
    StageExecutionError,
    _qr_leverage,
    atomic_json,
    generate_latin_hypercube,
    load_surrogate_bundle,
    save_surrogate_bundle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "config" / "params_closed_loop.json"


def exact_no_conversion_solver(
    index: int, decisions: np.ndarray, influent: np.ndarray
) -> MechanisticRow:
    q_effluent = 1.0 - decisions[4]
    q_underflow = decisions[3] + decisions[4]
    tss = float(mechanism.TSS_VECTOR @ influent)
    target = np.concatenate(
        (
            influent,
            np.tile(influent, mechanism.N_STAGES),
            q_effluent * influent,
            q_underflow * influent,
            np.full(mechanism.N_LAYERS, tss),
        )
    )
    return MechanisticRow(
        index=index,
        decisions=decisions.copy(),
        influent=influent.copy(),
        target=target,
        accepted=True,
        elapsed_seconds=0.001,
        diagnostics={"passed": True, "fixture": "no-conversion"},
    )


def rejected_solver(index: int, decisions: np.ndarray, influent: np.ndarray) -> MechanisticRow:
    row = exact_no_conversion_solver(index, decisions, influent)
    if index == 2:
        return MechanisticRow(
            index=index,
            decisions=decisions.copy(),
            influent=influent.copy(),
            target=np.full(170, np.nan),
            accepted=False,
            elapsed_seconds=0.001,
            diagnostics={"passed": False},
            error="deliberate fixture rejection",
        )
    return row


class DesignAdapterTests(unittest.TestCase):
    def test_adapter_uses_exact_design_contract(self) -> None:
        adapted = generate_latin_hypercube(37, 7, 42)
        direct, _, _ = unit_latin_hypercube(37, 7, seed=42)
        np.testing.assert_array_equal(adapted, direct)
        strata = np.floor(37 * adapted).astype(int)
        for coordinate in range(7):
            np.testing.assert_array_equal(np.sort(strata[:, coordinate]), np.arange(37))


class WorkflowCheckpointTests(unittest.TestCase):
    def test_static_pilot_dataset_and_resume_preserve_exact_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "results"
            workflow = ClosedLoopWorkflow(
                config_path=CONFIG_PATH,
                profile="unit",
                run_id="workflow_unit",
                repository_root=REPOSITORY_ROOT,
                results_root=results,
                mechanistic_solver=exact_no_conversion_solver,
            )
            manifest = workflow.run(through="dataset")
            self.assertEqual(manifest["stages"]["dataset"]["status"], "complete")
            with np.load(
                workflow.run_root / "datasets" / "mechanistic_dataset.npz",
                allow_pickle=False,
            ) as payload:
                np.testing.assert_array_equal(payload["row"], np.arange(workflow.sample_count))
                self.assertEqual(payload["targets"].shape, (workflow.sample_count, 170))
                self.assertTrue(np.all(np.isfinite(payload["targets"])))
            self.assertTrue((workflow.run_root / "splits" / "ordered_split.json").is_file())

            resumed = ClosedLoopWorkflow(
                config_path=CONFIG_PATH,
                profile="unit",
                run_id="workflow_unit",
                repository_root=REPOSITORY_ROOT,
                results_root=results,
                mechanistic_solver=exact_no_conversion_solver,
            )
            resumed_manifest = resumed.run(through="dataset")
            self.assertEqual(
                manifest["stages"]["dataset"]["marker_sha256"],
                resumed_manifest["stages"]["dataset"]["marker_sha256"],
            )

    def test_rejected_row_is_checkpointed_and_blocks_downstream_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow = ClosedLoopWorkflow(
                config_path=CONFIG_PATH,
                profile="unit",
                run_id="workflow_rejection",
                repository_root=REPOSITORY_ROOT,
                results_root=Path(temporary) / "results",
                mechanistic_solver=rejected_solver,
            )
            with self.assertRaises(StageExecutionError):
                workflow.run(through="pilot")
            self.assertTrue(
                (workflow.run_root / "datasets" / "chunks" / "rows_000000_000004.npz").is_file()
            )
            self.assertTrue(
                (workflow.run_root / "datasets" / "chunks" / "rows_000000_000004.diagnostics.json").is_file()
            )

    def test_completed_stage_marker_detects_artifact_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "results"
            workflow = ClosedLoopWorkflow(
                config_path=CONFIG_PATH,
                profile="unit",
                run_id="workflow_tamper",
                repository_root=REPOSITORY_ROOT,
                results_root=results,
                mechanistic_solver=exact_no_conversion_solver,
            )
            workflow.run(through="dataset")
            dataset = workflow.run_root / "datasets" / "mechanistic_dataset.npz"
            dataset.write_bytes(dataset.read_bytes() + b"changed")
            resumed = ClosedLoopWorkflow(
                config_path=CONFIG_PATH,
                profile="unit",
                run_id="workflow_tamper",
                repository_root=REPOSITORY_ROOT,
                results_root=results,
                mechanistic_solver=exact_no_conversion_solver,
            )
            with self.assertRaises(ContractMismatchError):
                resumed.run(through="dataset")

    def test_unexpected_stage_exception_is_wrapped(self) -> None:
        class BrokenWorkflow(ClosedLoopWorkflow):
            def _stage_static(self):
                raise ValueError("unexpected fixture failure")

        with tempfile.TemporaryDirectory() as temporary:
            workflow = BrokenWorkflow(
                config_path=CONFIG_PATH,
                profile="unit",
                run_id="workflow_exception",
                repository_root=REPOSITORY_ROOT,
                results_root=Path(temporary) / "results",
                mechanistic_solver=exact_no_conversion_solver,
            )
            with self.assertRaisesRegex(StageExecutionError, "stage 'static' failed"):
                workflow.run(through="static")


class HardeningContractTests(unittest.TestCase):
    def test_full_workload_is_projected_for_verification_without_eligibility_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow = ClosedLoopWorkflow(
                config_path=CONFIG_PATH,
                profile="unit",
                run_id="projection_unit",
                repository_root=REPOSITORY_ROOT,
                results_root=Path(temporary) / "results",
                mechanistic_solver=exact_no_conversion_solver,
            )
            workflow._initialize()
            atomic_json(
                workflow.run_root / "timing" / "pilot_summary.json",
                {
                    "all_accepted": True,
                    "mechanistic_p95_seconds": 0.001,
                    "mechanistic_preflight_peak_resident_memory_bytes": 1024,
                },
            )
            qp = {
                "row_indices": np.arange(10),
                "p95_seconds": 0.001,
                "peak_resident_memory_bytes": 2048,
                "all_accepted": True,
            }
            result = workflow._evaluate_computational_feasibility(
                fit_seconds=0.1,
                fit_peak_resident_memory_bytes=4096,
                qp_preflight=qp,
            )
            self.assertEqual(result["scientific_workloads"]["qp_scientific_evaluations"], 2_549_101)
            self.assertEqual(result["scientific_workloads"]["mechanistic_scientific_evaluations"], 280_000)
            self.assertTrue(result["projection_passed"])
            self.assertTrue(result["verification_profile"])
            self.assertFalse(result["article_reporting_eligible"])

    def test_full_profile_enforces_projection_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow = ClosedLoopWorkflow(
                config_path=CONFIG_PATH,
                profile="full",
                run_id="projection_full",
                repository_root=REPOSITORY_ROOT,
                results_root=Path(temporary) / "results",
                mechanistic_solver=exact_no_conversion_solver,
            )
            workflow._initialize()
            atomic_json(
                workflow.run_root / "timing" / "pilot_summary.json",
                {
                    "all_accepted": True,
                    "mechanistic_p95_seconds": 100.0,
                    "mechanistic_preflight_peak_resident_memory_bytes": 1024,
                },
            )
            qp = {
                "row_indices": np.arange(1000),
                "p95_seconds": 0.001,
                "peak_resident_memory_bytes": 2048,
                "all_accepted": True,
            }
            with self.assertRaisesRegex(StageExecutionError, "feasibility gate failed"):
                workflow._evaluate_computational_feasibility(
                    fit_seconds=0.1,
                    fit_peak_resident_memory_bytes=4096,
                    qp_preflight=qp,
                )

    def test_qr_leverage_matches_definition_without_normal_equations_factor(self) -> None:
        rng = np.random.default_rng(12)
        design = rng.normal(size=(100, 7))
        _, upper = np.linalg.qr(design, mode="reduced")
        feature = rng.normal(size=7)
        expected = float(feature @ np.linalg.inv(design.T @ design) @ feature)
        self.assertAlmostEqual(_qr_leverage(upper, feature), expected, places=12)

    def test_relative_variance_rule_marks_undefined_r_squared(self) -> None:
        truth = np.column_stack(
            (
                1.0e9 + np.asarray([0.0, 1.0e-6, -1.0e-6, 0.0]),
                np.asarray([0.0, 1.0, 2.0, 3.0]),
            )
        )
        metrics = ClosedLoopWorkflow._coordinate_metrics(
            truth, np.zeros_like(truth), np.ones(2), "fixture", ("flat", "variable")
        )
        self.assertTrue(np.isnan(metrics.loc[0, "r_squared"]))
        self.assertTrue(np.isfinite(metrics.loc[1, "r_squared"]))

    def test_mechanistic_reference_uses_exact_distinct_budget_and_saves_selected_state(self) -> None:
        calls: list[tuple[float, ...]] = []

        def counting_solver(index, decisions, influent):
            calls.append(tuple(float(value) for value in decisions))
            return exact_no_conversion_solver(index, decisions, influent)

        with tempfile.TemporaryDirectory() as temporary:
            workflow = ClosedLoopWorkflow(
                config_path=CONFIG_PATH,
                profile="unit",
                run_id="mechanistic_budget",
                repository_root=REPOSITORY_ROOT,
                results_root=Path(temporary) / "results",
                mechanistic_solver=counting_solver,
            )
            workflow._initialize()
            case_root = workflow.run_root / "optimization" / "nominal"
            case_root.mkdir(parents=True, exist_ok=True)
            bounds = np.asarray(list(workflow.config["process"]["decision_bounds"].values()))
            selected = np.mean(bounds, axis=1)
            influent = np.asarray(workflow.config["process"]["nominal_influent"])
            result = workflow._mechanistic_reference(
                "nominal", influent, selected, np.ones(4), robustness=False
            )
            self.assertEqual(result["mechanistic_evaluations"], 20)
            self.assertTrue(result["mechanistic_budget_exhausted_exactly"])
            self.assertEqual(len(calls), 20)
            self.assertEqual(len(set(calls)), 20)
            self.assertEqual(calls.count(tuple(selected)), 1)
            self.assertIn("selected_mechanistic_accepted", result)
            self.assertTrue((case_root / "selected_mechanistic_state.npz").is_file())
            self.assertTrue((case_root / "selected_mechanistic_diagnostics.json").is_file())

    def test_final_inventory_excludes_mutable_seal_files_and_replays_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow = ClosedLoopWorkflow(
                config_path=CONFIG_PATH,
                profile="unit",
                run_id="seal_order",
                repository_root=REPOSITORY_ROOT,
                results_root=Path(temporary) / "results",
                mechanistic_solver=exact_no_conversion_solver,
            )
            workflow._initialize()
            atomic_json(workflow.run_root / "checks" / "stage_complete.json", {"fixture": True})
            atomic_json(workflow.run_root / "report" / "fixture.json", {"value": 1})
            workflow._manifest["stages"]["complete"] = {}
            workflow._finalize_seal()
            inventory = pd.read_csv(workflow.run_root / "artifact_inventory.csv")
            paths = set(inventory["path"])
            self.assertIn("checks/stage_complete.json", paths)
            self.assertIn("report/fixture.json", paths)
            self.assertNotIn("manifest.json", paths)
            self.assertNotIn("COMPLETED.json", paths)
            for record in inventory.to_dict(orient="records"):
                from closed_loop.workflow import sha256_file

                self.assertEqual(
                    sha256_file(workflow.run_root / record["path"]), record["sha256"]
                )


class BundleTests(unittest.TestCase):
    def test_surrogate_bundle_round_trip_is_prediction_identical(self) -> None:
        rng = np.random.default_rng(81)
        decisions = rng.normal(size=(390, 5))
        influents = rng.normal(size=(390, 20))
        feature_map = QuadraticFeatureMap.fit(decisions, influents)
        design = feature_map.transform(decisions, influents)
        responses = design @ rng.normal(scale=0.03, size=(351, 3))
        model = QuadraticSurrogate.fit(decisions, influents, responses)
        row_scales = NetworkRowScales(
            equality=np.linspace(1.0, 2.0, 77),
            inequality=np.linspace(2.0, 3.0, 26),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.npz"
            save_surrogate_bundle(
                path,
                model,
                row_scales,
                quality_scale=np.ones(4),
            )
            restored, restored_scales, extras = load_surrogate_bundle(path)
            np.testing.assert_array_equal(
                restored.predict(decisions[:5], influents[:5]),
                model.predict(decisions[:5], influents[:5]),
            )
            np.testing.assert_array_equal(restored_scales.equality, row_scales.equality)
            np.testing.assert_array_equal(extras["quality_scale"], np.ones(4))


if __name__ == "__main__":
    unittest.main()
