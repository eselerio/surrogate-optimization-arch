from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import closed_loop.workflow as workflow_module
from closed_loop import model as mechanism
from closed_loop import nlp as nlp_core
from closed_loop.surrogate import QuadraticFeatureMap, QuadraticSurrogate
from closed_loop.workflow import (
    ContractMismatchError,
    ClosedLoopWorkflow,
    MechanisticRow,
    STAGES,
    StageExecutionError,
    _derived_assessment_responses,
    _derived_metric_frame,
    _maximin_robustness_indices,
    _nearest_rank,
    atomic_json,
    load_surrogate_bundle,
    save_surrogate_bundle,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "params_closed_loop.json"


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
            target=np.full(mechanism.TARGET_SIZE, np.nan),
            accepted=False,
            elapsed_seconds=0.001,
            diagnostics={"passed": False},
            error="deliberate fixture rejection",
        )
    return row


class _InterruptibleNLPFacade:
    ObjectiveWeights = nlp_core.ObjectiveWeights
    CaseDefinition = nlp_core.CaseDefinition
    KKTDiagnostics = nlp_core.KKTDiagnostics
    NLPStartResult = nlp_core.NLPStartResult

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.fail_once_at = 2

    @staticmethod
    def ordered_normalized_starts() -> np.ndarray:
        values = np.zeros((9, 5), dtype=np.float64)
        values[:, 0] = np.linspace(0.1, 0.9, 9)
        return values

    @staticmethod
    def combined_initial_point(
        z, influent, development_decisions, development_influent,
        development_targets, assets,
    ):
        point = np.zeros(115, dtype=np.float64)
        point[:5] = np.asarray(z, dtype=np.float64)
        return point, 0

    def solve_nlp_start(self, problem, case, point, *, start_index):
        self.calls.append(int(start_index))
        if start_index == self.fail_once_at:
            self.fail_once_at = -1
            raise RuntimeError("deliberate interrupted start")
        return nlp_core.NLPStartResult(
            start_index=int(start_index),
            status="Solve_Succeeded",
            solver_success=True,
            accepted=True,
            objective=float(start_index),
            primal=np.asarray(point, dtype=np.float64),
            equality_multipliers=np.zeros(110),
            inequality_multipliers=np.zeros(9),
            bound_multipliers=np.zeros(115),
            equality=np.zeros(110),
            inequality=np.full(9, -1.0),
            normalized_controls=np.asarray(point[:5]),
            decisions=np.asarray([12.0, 0.5, 1.0, 0.5, 0.01]),
            state=np.full(110, float(start_index)),
            diagnostics={"engineering_objective": float(start_index)},
            kkt=nlp_core.KKTDiagnostics(
                bound_violation=0.0,
                primal_residual=0.0,
                stationarity_residual=0.0,
                dual_feasibility_residual=0.0,
                complementarity_residual=0.0,
                physical_nonnegativity_residual=0.0,
                finite=True,
            ),
            elapsed_seconds=0.01,
            iterations=3,
            error=None,
        )

    @staticmethod
    def evaluate_problem(problem, primal, case):
        return {"complete_state": np.full(170, float(primal[0]))}


class WorkflowTests(unittest.TestCase):
    def test_stages_are_the_frozen_nlp_pipeline(self) -> None:
        self.assertEqual(
            STAGES,
            (
                "static", "pilot", "dataset", "fit", "calibration",
                "assessment", "nlp_preflight", "optimization", "report", "complete",
            ),
        )

    def test_static_and_dataset_checkpoint_independent_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow = ClosedLoopWorkflow(
                config_path=CONFIG, profile="unit", run_id="checkpoint",
                repository_root=ROOT, results_root=Path(temporary),
                mechanistic_solver=exact_no_conversion_solver,
            )
            manifest = workflow.run(through="dataset")
            self.assertEqual(manifest["stages"]["dataset"]["status"], "complete")
            with np.load(workflow.run_root / "datasets" / "mechanistic_dataset.npz") as payload:
                self.assertEqual(payload["targets"].shape, (600, 170))
                np.testing.assert_array_equal(np.bincount(payload["block"]), [420, 60, 120])
            split = json.loads((workflow.run_root / "splits" / "ordered_blocks.json").read_text())
            self.assertEqual([block["seed"] for block in split["blocks"]], [60042, 60043, 60044])
            dataset_summary = json.loads(
                (workflow.run_root / "checks" / "dataset_validation.json").read_text()
            )
            self.assertEqual(
                {
                    name: value["rows"]
                    for name, value in dataset_summary["generation_by_block"].items()
                },
                {"development": 420, "calibration": 60, "assessment": 120},
            )
            self.assertGreater(
                manifest["stages"]["dataset"][
                    "stage_high_water_resident_memory_bytes"
                ],
                0,
            )
            self.assertEqual(
                sorted(path.name for path in (workflow.run_root / "datasets" / "chunks").glob("*.npz")),
                ["rows_000000_000004.npz", "rows_000004_000016.npz", "rows_000016_000064.npz", "rows_000064_000256.npz", "rows_000256_000420.npz", "rows_000420_000480.npz", "rows_000480_000600.npz"],
            )
            resumed = ClosedLoopWorkflow(
                config_path=CONFIG, profile="unit", run_id="checkpoint",
                repository_root=ROOT, results_root=Path(temporary),
                mechanistic_solver=exact_no_conversion_solver,
            )
            replayed = resumed.run(through="dataset")
            self.assertEqual(replayed["stages"]["pilot"]["status"], "complete")

    def test_failed_mechanistic_row_is_checkpointed_and_blocks_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow = ClosedLoopWorkflow(
                config_path=CONFIG, profile="unit", run_id="rejected",
                repository_root=ROOT, results_root=Path(temporary),
                mechanistic_solver=rejected_solver,
            )
            with self.assertRaises(StageExecutionError):
                workflow.run(through="pilot")
            self.assertTrue((workflow.run_root / "datasets" / "chunks" / "rows_000000_000004.npz").is_file())

    def test_resume_detects_changed_bound_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workflow = ClosedLoopWorkflow(
                config_path=CONFIG, profile="unit", run_id="tamper",
                repository_root=ROOT, results_root=root,
                mechanistic_solver=exact_no_conversion_solver,
            )
            workflow.run(through="static")
            design = workflow.run_root / "datasets" / "design.npz"
            design.write_bytes(design.read_bytes() + b"tamper")
            resumed = ClosedLoopWorkflow(
                config_path=CONFIG, profile="unit", run_id="tamper",
                repository_root=ROOT, results_root=root,
                mechanistic_solver=exact_no_conversion_solver,
            )
            with self.assertRaises(ContractMismatchError):
                resumed.run(through="static")

    def test_unexpected_stage_exception_is_wrapped(self) -> None:
        class Broken(ClosedLoopWorkflow):
            def _stage_static(self):
                raise ValueError("deliberate")

        with tempfile.TemporaryDirectory() as temporary:
            workflow = Broken(
                config_path=CONFIG, profile="unit", run_id="broken",
                repository_root=ROOT, results_root=Path(temporary),
                mechanistic_solver=exact_no_conversion_solver,
            )
            with self.assertRaisesRegex(StageExecutionError, "stage 'static' failed"):
                workflow.run(through="static")

    def test_declared_workloads_have_no_qp_or_direct_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow = ClosedLoopWorkflow(
                config_path=CONFIG, profile="test_2000", run_id="workload",
                repository_root=ROOT, results_root=Path(temporary),
                mechanistic_solver=exact_no_conversion_solver,
            )
            full = workflow._scientific_workloads("full")
            test = workflow._scientific_workloads("test_2000")
            self.assertEqual(full["bdf_invocations_maximum"], 20_113)
            self.assertEqual(full["combined_nlp_starts"], 1_017)
            self.assertEqual(test["bdf_invocations_maximum"], 2_023)
            self.assertEqual(test["combined_nlp_starts"], 207)
            self.assertEqual(test["optimization_cases"], 23)
            self.assertEqual(full["physical_qp_evaluations"], 0)
            self.assertEqual(full["direct_evaluations"], 0)

    def test_contract_binds_casadi_ipopt_and_mumps_binaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow = ClosedLoopWorkflow(
                config_path=CONFIG, profile="unit", run_id="contract",
                repository_root=ROOT, results_root=Path(temporary),
                mechanistic_solver=exact_no_conversion_solver,
            )
            binaries = workflow._contract["casadi_ipopt_binary_sha256"]
            self.assertTrue(any("ipopt" in name.lower() for name in binaries))
            self.assertTrue(any("mumps" in name.lower() for name in binaries))
            self.assertEqual(workflow._contract["dependency_lock"]["sha256"], sha256_file(ROOT / "uv.lock"))

    def test_maximin_panel_is_deterministic_and_ties_use_lower_index(self) -> None:
        points = np.asarray([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        selected = _maximin_robustness_indices(points, np.asarray([0.5, 0.5]), np.zeros(2), np.ones(2), count=3)
        np.testing.assert_array_equal(selected, [0, 1, 2])

    def test_nearest_rank_is_not_interpolated(self) -> None:
        values = [4.0, 1.0, 3.0, 2.0]
        self.assertEqual(_nearest_rank(values, 0.25), 1.0)
        self.assertEqual(_nearest_rank(values, 0.50), 2.0)
        self.assertEqual(_nearest_rank(values, 0.95), 4.0)

    def test_combined_outcome_table_retains_unobserved_categories(self) -> None:
        frame = pd.DataFrame(
            {"outcome": ["validated combined recommendation", "no accepted NLP start"]}
        )
        categories = (
            "no accepted NLP start",
            "exact integration failure",
            "validated combined recommendation",
        )
        table = ClosedLoopWorkflow._outcome_table(frame, "outcome", categories)
        self.assertEqual(table["count"].tolist(), [1, 0, 1])
        self.assertEqual(table["denominator"].tolist(), [2, 2, 2])

    def test_terminal_frame_replay_aligns_json_key_order_but_rejects_differences(self) -> None:
        stored = pd.DataFrame(
            {"case_id": ["nominal", "robustness_001"], "objective": [1.0, 2.0]}
        )
        json_order = pd.DataFrame(
            {"objective": [1.0, 2.0], "case_id": ["nominal", "robustness_001"]}
        )
        aligned = ClosedLoopWorkflow._validate_replayed_frame(
            stored, json_order, "fixture"
        )
        self.assertEqual(list(aligned.columns), list(stored.columns))
        with self.assertRaisesRegex(StageExecutionError, "schema changed"):
            ClosedLoopWorkflow._validate_replayed_frame(
                stored, json_order.drop(columns="objective"), "fixture"
            )
        changed = json_order.copy()
        changed.loc[1, "objective"] = 3.0
        with self.assertRaisesRegex(StageExecutionError, "values changed"):
            ClosedLoopWorkflow._validate_replayed_frame(stored, changed, "fixture")

    def test_derived_assessment_metrics_use_development_scales(self) -> None:
        decisions = np.asarray([[12.0, 0.5, 1.0, 0.5, 0.01], [24.0, 0.6, 2.0, 0.8, 0.02]])
        targets = np.zeros((2, mechanism.TARGET_SIZE))
        targets[:, 120:140] = np.asarray([[1.0] * 20, [2.0] * 20])
        targets[:, 140:160] = np.asarray([[2.0] * 20, [3.0] * 20])
        targets[:, 160:170] = np.asarray([[100.0] * 10, [200.0] * 10])
        truth = _derived_assessment_responses(decisions, targets)
        prediction = truth + np.arange(1.0, 10.0)
        scale = np.arange(2.0, 11.0)
        metrics = _derived_metric_frame(
            truth,
            prediction,
            scale,
            variance_relative_tolerance=1.0e-12,
        )
        np.testing.assert_allclose(metrics["rmse"], np.arange(1.0, 10.0))
        np.testing.assert_allclose(metrics["nrmse"], np.arange(1.0, 10.0) / scale)
        self.assertEqual(metrics["response"].iloc[-1], "normalized_clarifier_inventory")

    def test_exact_fidelity_uses_normalized_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow = ClosedLoopWorkflow(
                config_path=CONFIG, profile="unit", run_id="fidelity",
                repository_root=ROOT, results_root=Path(temporary),
                mechanistic_solver=exact_no_conversion_solver,
            )
            tolerance = float(
                workflow.config["upper_constraints"]["normalized_feasibility_tolerance"]
            )
            self.assertTrue(workflow._exact_fidelity_passed(1.0 + tolerance))
            self.assertFalse(workflow._exact_fidelity_passed(1.0 + 2.0 * tolerance))

    def test_projection_gate_is_enforced_for_unit_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow = ClosedLoopWorkflow(
                config_path=CONFIG, profile="unit", run_id="projection",
                repository_root=ROOT, results_root=Path(temporary),
                mechanistic_solver=exact_no_conversion_solver,
            )
            workflow._initialize()
            atomic_json(
                workflow.run_root / "timing" / "pilot_summary.json",
                {"p95_seconds": 0.001, "maximum_resident_memory_bytes": 1},
            )
            atomic_json(
                workflow.run_root / "models" / "development_assets.json",
                {"fit_seconds": 0.001, "peak_resident_memory_bytes": 1},
            )
            with self.assertRaisesRegex(StageExecutionError, "feasibility gate failed"):
                workflow._computational_projection([0.001], 30 * 1024**3)

    def test_zero_eligible_robustness_and_activity_rows_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow = ClosedLoopWorkflow(
                config_path=CONFIG, profile="unit", run_id="empty_tables",
                repository_root=ROOT, results_root=Path(temporary),
                mechanistic_solver=exact_no_conversion_solver,
            )
            workflow._initialize()
            atomic_json(workflow.run_root / "optimization" / "cases.json", {"cases": []})
            frame = pd.DataFrame(
                columns=(
                    "case_id", "case_class", "selected_start", "selected_H",
                    "selected_a", "selected_r_I", "selected_r_R", "selected_w",
                    "exact_objective",
                )
            )
            robustness = workflow._robustness_summary(frame)
            objective = robustness.loc[
                robustness["quantity"] == "exact_objective"
            ].iloc[0]
            self.assertEqual(objective["eligible_n"], 0)
            activity = workflow._bound_activity(frame)
            self.assertEqual(len(activity), 26)
            self.assertTrue((activity["eligible_n"] == 0).all())

    def test_flat_summary_exposes_signed_outlet_differences(self) -> None:
        flow = np.arange(40.0).reshape(2, 20) - 20.0
        concentration = np.arange(8.0).reshape(2, 4) - 4.0
        row = ClosedLoopWorkflow._flat_case_summary(
            {
                "case_id": "fixture",
                "case_class": "nominal",
                "sensitivity_family": None,
                "accepted_starts": 1,
                "selected_start": 0,
                "selected_objective": 1.0,
                "selected_decisions": np.ones(5),
                "nlp_diagnostics": {},
                "weights": np.ones(6),
                "exact": {
                    "outcome": "residual/stability failure",
                    "accepted": False,
                    "nlp_minus_bdf_outlet_component_flow": flow,
                    "nlp_minus_bdf_outlet_composite_concentration": concentration,
                    "dynamic_balance_scaled_residual_inf": 2.5e-7,
                },
            }
        )
        self.assertEqual(
            row["exact_nlp_minus_bdf_effluent_component_flow_S_O"], flow[0, 0]
        )
        self.assertEqual(
            row["exact_nlp_minus_bdf_underflow_tss_concentration"],
            concentration[1, 3],
        )
        self.assertEqual(row["exact_dynamic_balance_scaled_residual_inf"], 2.5e-7)

    def test_per_start_checkpoints_resume_after_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow = ClosedLoopWorkflow(
                config_path=CONFIG, profile="unit", run_id="start_resume",
                repository_root=ROOT, results_root=Path(temporary),
                mechanistic_solver=exact_no_conversion_solver,
            )
            workflow._initialize()
            facade = _InterruptibleNLPFacade()
            workflow._nlp = lambda: facade
            record = {
                "case_id": "fixture_case",
                "case_class": "nominal",
                "influent": workflow.config["process"]["nominal_influent"],
                "weights": workflow._primary_weights(),
                "underflow_tss_limit": workflow.config["upper_constraints"]
                ["underflow_tss_max_g_m3"],
            }
            development_decisions = np.zeros((2, 5))
            development_influent = np.zeros((2, 20))
            development_targets = np.zeros((2, 170))
            with self.assertRaisesRegex(RuntimeError, "interrupted start"):
                workflow._solve_nlp_case(
                    record, object(), object(), None,
                    development_decisions, development_influent,
                    development_targets,
                )
            results = workflow._solve_nlp_case(
                record, object(), object(), None,
                development_decisions, development_influent,
                development_targets,
            )
            self.assertEqual(len(results), 9)
            self.assertEqual(facade.calls.count(0), 1)
            self.assertEqual(facade.calls.count(1), 1)
            self.assertEqual(facade.calls.count(2), 2)
            self.assertEqual(len(facade.calls), 10)
            # The authoritative aggregate survives loss of its optional JSON
            # sidecar and is reused without another solver invocation.
            aggregate_json = (
                workflow.run_root / "optimization" / "cache" / "fixture_case"
                / "starts.json"
            )
            aggregate_json.unlink()
            resumed = workflow._solve_nlp_case(
                record, object(), object(), None,
                development_decisions, development_influent,
                development_targets,
            )
            self.assertEqual(len(resumed), 9)
            self.assertTrue(aggregate_json.is_file())
            self.assertEqual(len(facade.calls), 10)
            summary = workflow._invocation_summary()["kinds"]["combined_nlp_start"]
            self.assertEqual(summary["attempted"], 10)
            self.assertEqual(summary["completed"], 9)
            self.assertEqual(summary["interrupted"], 1)
            self.assertEqual(summary["reused"], 11)
            aggregate_npz = aggregate_json.with_suffix(".npz")
            with np.load(aggregate_npz, allow_pickle=False) as payload:
                changed = {name: payload[name].copy() for name in payload.files}
            changed["primal"][0, 0] += 1.0
            np.savez_compressed(aggregate_npz, **changed)
            with self.assertRaises(ContractMismatchError):
                workflow._load_nlp_case(workflow._case_definition(record))

    def test_exact_cache_resume_records_attempts_and_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow = ClosedLoopWorkflow(
                config_path=CONFIG, profile="unit", run_id="exact_resume",
                repository_root=ROOT, results_root=Path(temporary),
                mechanistic_solver=exact_no_conversion_solver,
            )
            workflow._initialize()
            atomic_json(workflow.run_root / "metrics" / "calibration.json", {"delta": 1.0})
            decisions = np.asarray([12.0, 0.5, 1.0, 0.5, 0.01])
            influent = np.asarray(
                workflow.config["process"]["nominal_influent"], dtype=np.float64
            )
            exact = exact_no_conversion_solver(0, decisions, influent)
            selected = SimpleNamespace(
                decisions=decisions,
                state=np.concatenate((exact.target[20:120], exact.target[160:170])),
            )

            class ExactModel:
                response_scale = np.ones(170)

                @staticmethod
                def predict(theta, x):
                    return exact.target.copy()

            case = {
                "case_id": "exact_fixture",
                "case_class": "nominal",
                "influent": influent,
                "weights": workflow._primary_weights(),
                "underflow_tss_limit": workflow.config["upper_constraints"]
                ["underflow_tss_max_g_m3"],
            }
            calls = 0

            def interrupted_then_complete(index, theta, x):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("deliberate exact interruption")
                return exact_no_conversion_solver(index, theta, x)

            workflow._solve_row = interrupted_then_complete
            arguments = (
                0, case, selected, exact.target, ExactModel(),
                SimpleNamespace(state_scale=np.ones(110)),
                np.ones(4), 1.0, np.ones(20),
            )
            with self.assertRaisesRegex(RuntimeError, "exact interruption"):
                workflow._exact_replay(*arguments)
            record = workflow._exact_replay(*arguments)
            self.assertIn("selected_engineering", record)
            self.assertIn("dynamic_balance_scaled_residual_inf", record)
            self.assertEqual(
                np.asarray(record["nlp_minus_bdf_outlet_component_flow"]).shape,
                (2, 20),
            )
            sidecar = (
                workflow.run_root / "optimization" / "cache" / "exact_fixture"
                / "exact_combined.json"
            )
            sidecar.unlink()
            replayed = workflow._exact_replay(*arguments)
            self.assertEqual(replayed["exact_cache_key"], record["exact_cache_key"])
            self.assertTrue(sidecar.is_file())
            self.assertEqual(calls, 2)
            summary = workflow._invocation_summary()["kinds"]["exact_bdf_replay"]
            self.assertEqual(summary["attempted"], 2)
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["interrupted"], 1)
            self.assertEqual(summary["reused"], 1)


class AtomicWriteTests(unittest.TestCase):
    @staticmethod
    def _windows_permission_error(winerror: int = 5) -> PermissionError:
        error = PermissionError(13, "transient Windows file lock")
        error.winerror = winerror
        return error

    def test_atomic_bytes_retries_transient_windows_replace_then_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            destination = directory / "checkpoint.json"
            destination.write_bytes(b"old")
            real_replace = workflow_module.os.replace
            attempts = 0

            def transient_then_success(source, target):
                nonlocal attempts
                attempts += 1
                if attempts <= 2:
                    raise self._windows_permission_error()
                real_replace(source, target)

            with (
                patch.object(workflow_module, "_WINDOWS_ATOMIC_REPLACE_RETRY", True),
                patch.object(workflow_module, "sleep") as mocked_sleep,
                patch.object(
                    workflow_module.os,
                    "replace",
                    side_effect=transient_then_success,
                ),
            ):
                workflow_module._atomic_bytes(destination, b"new")

            self.assertEqual(destination.read_bytes(), b"new")
            self.assertEqual(attempts, 3)
            self.assertEqual(
                [call.args[0] for call in mocked_sleep.call_args_list],
                [0.01, 0.02],
            )
            self.assertEqual(list(directory.glob(".checkpoint.json.*.tmp")), [])

    def test_atomic_bytes_exhaustion_preserves_destination_and_removes_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            destination = directory / "checkpoint.json"
            destination.write_bytes(b"old")
            with (
                patch.object(workflow_module, "_WINDOWS_ATOMIC_REPLACE_RETRY", True),
                patch.object(workflow_module, "sleep") as mocked_sleep,
                patch.object(
                    workflow_module.os,
                    "replace",
                    side_effect=lambda *_: (_ for _ in ()).throw(
                        self._windows_permission_error()
                    ),
                ) as mocked_replace,
                self.assertRaises(PermissionError),
            ):
                workflow_module._atomic_bytes(destination, b"new")

            self.assertEqual(destination.read_bytes(), b"old")
            self.assertEqual(
                mocked_replace.call_count,
                len(workflow_module._WINDOWS_ATOMIC_REPLACE_DELAYS_SECONDS) + 1,
            )
            self.assertEqual(
                mocked_sleep.call_count,
                len(workflow_module._WINDOWS_ATOMIC_REPLACE_DELAYS_SECONDS),
            )
            self.assertEqual(list(directory.glob(".checkpoint.json.*.tmp")), [])

    def test_atomic_bytes_does_not_retry_other_permission_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            destination = directory / "checkpoint.json"
            with (
                patch.object(workflow_module, "_WINDOWS_ATOMIC_REPLACE_RETRY", True),
                patch.object(workflow_module, "sleep") as mocked_sleep,
                patch.object(
                    workflow_module.os,
                    "replace",
                    side_effect=self._windows_permission_error(winerror=87),
                ) as mocked_replace,
                self.assertRaises(PermissionError),
            ):
                workflow_module._atomic_bytes(destination, b"new")

            self.assertEqual(mocked_replace.call_count, 1)
            mocked_sleep.assert_not_called()
            self.assertFalse(destination.exists())
            self.assertEqual(list(directory.glob(".checkpoint.json.*.tmp")), [])


class BundleTests(unittest.TestCase):
    def test_round_trip_preserves_qr_and_predictions(self) -> None:
        rng = np.random.default_rng(81)
        decisions = rng.normal(size=(390, 5))
        influents = rng.normal(size=(390, 20))
        feature_map = QuadraticFeatureMap.fit(decisions, influents)
        design = feature_map.transform(decisions, influents)
        responses = design @ rng.normal(scale=0.03, size=(351, 3))
        model = QuadraticSurrogate.fit(decisions, influents, responses)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.npz"
            save_surrogate_bundle(path, model, quality_scale=np.ones(4))
            restored, restored_scales, extras = load_surrogate_bundle(path)
            np.testing.assert_array_equal(restored.predict(decisions[:5], influents[:5]), model.predict(decisions[:5], influents[:5]))
            np.testing.assert_array_equal(restored.feature_qr_upper, model.feature_qr_upper)
            np.testing.assert_array_equal(restored.feature_qr_pivots, model.feature_qr_pivots)
            self.assertIsNone(restored_scales)
            np.testing.assert_array_equal(extras["quality_scale"], np.ones(4))


if __name__ == "__main__":
    unittest.main()
