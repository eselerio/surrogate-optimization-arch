from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from closed_loop.manuscript_v3 import (
    ARTICLE_FULL,
    DECISION_LOWER,
    DECISION_UPPER,
    reduce_mechanistic_responses,
)
from closed_loop.model import INFLUENT_LOWER, INFLUENT_UPPER
from closed_loop.projection import (
    NetworkLayout,
    ProjectionDiagnostics,
    ProjectionResult,
)
from closed_loop.v3_smooth import (
    BranchClassification,
    ContinuationStageResult as DirectStage,
    DirectMultistartResult,
    DirectStartResult,
    EquivalenceDiagnostics,
    KKTDiagnostics,
)
from closed_loop.v3_surrogate_nlp import (
    EXACT_QP_CENTER_START,
    EXACT_QP_SINGLE_START_PROTOCOL,
    GAP_CONTINUATION,
    ContinuationStageRecord,
    FeasibilityRecord,
    FinalCandidateRecord,
    OuterRefinementRecord,
    StationarityRecord,
    SurrogateMultistartResult,
    SurrogateStartResult,
)
from scripts import run_article_v3_5000 as runner


RESPONSE_COUNT = ARTICLE_FULL.surrogate_response_count
MECHANISTIC_RESPONSE_COUNT = ARTICLE_FULL.mechanistic_response_count
REDUCED_STATE_COUNT = 5 * 20 + ARTICLE_FULL.layer_count


def _branch() -> BranchClassification:
    return BranchClassification((), (), (), (), (), False, 1.0)


def _projection(response: np.ndarray) -> ProjectionResult:
    diagnostics = ProjectionDiagnostics(
        status="solved",
        status_value=1,
        iterations=4,
        equality_rank_tolerance=1.0e-12,
        equality_smallest_singular_value=1.0,
        equality_condition_number=1.0,
        equality_residual=1.0e-12,
        inequality_residual=0.0,
        nonnegativity_residual=0.0,
        dual_feasibility_residual=0.0,
        stationarity_residual=1.0e-12,
        complementarity_residual=1.0e-12,
        retried_cold=False,
        active_inequality_count=2,
        multipliers_reconstructed=True,
    )
    return ProjectionResult(
        state=np.asarray(response, dtype=float),
        displacement=np.zeros(RESPONSE_COUNT),
        equality_multipliers=np.zeros(1),
        inequality_multipliers=np.zeros(1),
        inequality_slack=np.ones(1),
        diagnostics=diagnostics,
        accepted=True,
    )


def _surrogate_start(index: int, normalized: np.ndarray) -> SurrogateStartResult:
    raw = np.full(RESPONSE_COUNT, 1.0 + 0.01 * index)
    projected = np.full(RESPONSE_COUNT, 2.0 + 0.01 * index)
    theta = DECISION_LOWER + (DECISION_UPPER - DECISION_LOWER) * normalized
    final = FinalCandidateRecord(
        normalized_controls=normalized.copy(),
        theta=theta,
        raw=raw,
        projected=projected,
        displacement=projected - raw,
        objective=1.0 + index,
        objective_components=np.full(6, 1.0 / 6.0),
        engineering_rows=np.full(7, -1.0),
        engineering_quantities=np.ones(7),
        trust_rows=np.full(4, -1.0),
        trust_values=np.zeros(4),
        projection=_projection(projected),
        feasibility=FeasibilityRecord(
            finite=True,
            cold_projection=True,
            projection_accepted=True,
            control_bound_residual=0.0,
            engineering_residual=0.0,
            trust_residual=0.0,
            maximum_upper_residual=0.0,
            feasible=True,
            projection_reproduction_residual=1.0e-12,
            projection_reproduction_passed=True,
        ),
        stationarity=StationarityRecord(
            classification="first_order_kkt_stationary_feasible",
            resolved=True,
            stationary=True,
            lower_qp_kkt_passed=True,
            upper_stationarity_residual=1.0e-12,
            reason="mocked independent audit",
        ),
        status="validated_stationary",
    )
    lower_active_set = {"stable": True, "active_count": 2}
    upper_kkt = {"feasible": True, "stationary": True}
    final = FinalCandidateRecord(
        **{
            **final.__dict__,
            "lower_active_set": lower_active_set,
            "upper_kkt": upper_kkt,
        }
    )
    return SurrogateStartResult(
        start_index=index,
        initial_normalized_controls=normalized.copy(),
        stages=(),
        outer_refinement=OuterRefinementRecord(
            attempted=True,
            solver_success=True,
            status="success",
            iterations=2,
            evaluations=3,
            elapsed_seconds=0.01,
            initial_objective=final.objective + 0.1,
            final_objective=final.objective,
            projection_reproduction_residual=1.0e-12,
            projection_reproduction_passed=True,
            lower_active_set=lower_active_set,
            upper_kkt=upper_kkt,
        ),
        final=final,
        status="validated_stationary",
        resume_contract="mock-resume-contract",
        protocol=EXACT_QP_SINGLE_START_PROTOCOL,
    )


def _direct_start(index: int, normalized: np.ndarray) -> DirectStartResult:
    theta = DECISION_LOWER + (DECISION_UPPER - DECISION_LOWER) * normalized
    stages = tuple(
        DirectStage(
            epsilon=epsilon,
            receiver_half_width=half_width,
            status="Solve_Succeeded",
            solver_success=True,
            elapsed_seconds=0.01,
            iterations=2,
            primal=np.zeros(8),
            feasible=True,
        )
        for epsilon, half_width in ((1.0e-6, 10.0), (1.0e-7, 3.0), (1.0e-8, 1.0))
    )
    kkt = KKTDiagnostics(
        finite=True,
        equality_residual=1.0e-12,
        inequality_residual=0.0,
        bound_residual=0.0,
        stationarity_residual=1.0e-12,
        dual_feasibility_residual=0.0,
        complementarity_residual=0.0,
        active_inequality_count=2,
        feasible=True,
        stationary=True,
        equality_multipliers=np.zeros(1),
        inequality_multipliers=np.zeros(1),
        lower_bound_multipliers=np.zeros(1),
        upper_bound_multipliers=np.zeros(1),
    )
    return DirectStartResult(
        start_index=index,
        initial_normalized_controls=normalized.copy(),
        resume_contract="mock-resume-contract",
        nearest_development_row=0,
        stages=stages,
        objective=2.0 + index,
        normalized_controls=normalized.copy(),
        theta=theta,
        state=np.ones(REDUCED_STATE_COUNT),
        feed_tss=100.0,
        response=np.full(MECHANISTIC_RESPONSE_COUNT, 3.0 + 0.01 * index),
        engineering=np.ones(11),
        objective_components=np.full(6, 1.0 / 6.0),
        branch=_branch(),
        kkt=kkt,
        feasible=True,
        stationary=True,
        status="first_order_kkt_stationary_feasible",
        error=None,
    )


def _surrogate_solver(*_args: object, **kwargs: object) -> SurrogateMultistartResult:
    normalized = np.asarray(EXACT_QP_CENTER_START, dtype=float)
    completed = kwargs.get("completed_result")
    callback = kwargs["progress_callback"]
    if completed is None:
        result = _surrogate_start(0, normalized)
        callback(result)
    else:
        result = completed
    return SurrogateMultistartResult(
        (result,), result, "selected_stationary",
        protocol=EXACT_QP_SINGLE_START_PROTOCOL,
    )


def _direct_solver(*_args: object, **kwargs: object) -> DirectMultistartResult:
    starts = np.asarray(kwargs["starts"], dtype=float)
    completed = dict(kwargs.get("completed_starts") or {})
    callback = kwargs["progress_callback"]
    for index, normalized in enumerate(starts):
        if index in completed:
            continue
        result = _direct_start(index, normalized)
        completed[index] = result
        callback(result)
    ordered = tuple(completed[index] for index in range(len(starts)))
    return DirectMultistartResult(ordered, ordered[0], "selected_stationary")


def _fixture() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, runner.AnalysisBundle]:
    midpoint = 0.5 * (DECISION_LOWER + DECISION_UPPER)
    influent = 0.5 * (INFLUENT_LOWER + INFLUENT_UPPER)
    development_count, test_count = 3, 2
    design = {
        "development_decisions": np.tile(midpoint, (development_count, 1)),
        "development_influents": np.tile(influent, (development_count, 1)),
        "test_decisions": np.tile(midpoint, (test_count, 1)),
        "test_influents": np.tile(influent, (test_count, 1)),
        "robustness_influents": np.vstack(
            [influent + 1.0e-3 * index for index in range(10)]
        ),
    }
    development_targets = np.ones((development_count, MECHANISTIC_RESPONSE_COUNT))
    test_targets = np.ones((test_count, MECHANISTIC_RESPONSE_COUNT))
    layout = NetworkLayout(layer_count=ARTICLE_FULL.layer_count)

    def predict(_theta: np.ndarray, _influent: np.ndarray) -> np.ndarray:
        return np.ones(RESPONSE_COUNT)

    model = SimpleNamespace(
        response_count=MECHANISTIC_RESPONSE_COUNT,
        response_scale=np.ones(RESPONSE_COUNT),
        predict=predict,
    )
    surrogate_assets = SimpleNamespace(
        layout=layout,
        row_scales=SimpleNamespace(equality=np.ones(1), inequality=np.ones(1)),
    )
    direct_assets = SimpleNamespace(
        response_count=RESPONSE_COUNT,
        state_count=REDUCED_STATE_COUNT,
        state_scale=np.ones(REDUCED_STATE_COUNT),
        feed_scale=1.0,
    )
    analysis = runner.AnalysisBundle(
        passed=True,
        model=model,
        direct_assets=direct_assets,
        surrogate_assets=surrogate_assets,
        assessment=None,
        gate={"passed": True},
    )
    return design, development_targets, test_targets, analysis


def _equivalence() -> EquivalenceDiagnostics:
    return EquivalenceDiagnostics(
        smooth_accepted=True,
        reference_accepted=True,
        accepted=True,
        state_rms=1.0e-9,
        state_inf=1.0e-8,
        own_smooth_residual=1.0e-10,
        own_reference_residual=1.0e-10,
        cross_residual=1.0e-8,
        relative_objective_difference=1.0e-8,
        engineering_difference=1.0e-8,
        reference_root_difference_generation=1.0e-8,
        reference_root_difference_state_scale=1.0e-8,
        branch_agreement=True,
        feasibility_agreement=True,
    )


def _mock_timing(run: Path, *_args: object, **_kwargs: object) -> pd.DataFrame:
    frame = pd.DataFrame([
        {
            "case": f"robustness_{case:02d}",
            "route": route,
            "primary_optimization_seconds": 1.0,
            "certification_seconds": 0.5 if route == "surrogate" else np.nan,
            "recovery_seconds": np.nan,
            "complete_optimization_seconds": 1.5 if route == "surrogate" else 1.0,
            "exact_reference_seconds": 0.25,
        }
        for case in range(1, 11)
        for route in ("surrogate", "direct")
    ])
    runner.atomic_dataframe(run / "metrics/robustness_case_timing.csv", frame)
    runner.atomic_dataframe(run / "metrics/timing_events.csv", pd.DataFrame([{
        "case": row.case,
        "route": row.route,
        "category": f"{row.route}_complete_optimization",
        "elapsed_seconds": row.complete_optimization_seconds,
    } for row in frame.itertuples()]))
    runner.atomic_json(run / "metrics/robustness_case_timing_summary.json", {
        "protocol": runner.TIMING_PROTOCOL,
        "robustness_case_count": 10,
    })
    runner.atomic_json(run / "metrics/robustness_case_timing_complete.json", {
        "stage": "robustness_case_timing_aggregation",
        "case_count": 10,
    })
    return frame


class ArticleV3OptimizationHookTests(unittest.TestCase):
    def test_derivative_audit_is_source_bound(self) -> None:
        self.assertIn("closed_loop/v3_derivative_audit.py", runner.SOURCE_FILES)

    def test_exact_qp_route_uses_endpoint_audit_without_continuation(self) -> None:
        _, _, _, analysis = _fixture()
        selected = _surrogate_start(0, np.asarray(EXACT_QP_CENTER_START))
        self.assertEqual(selected.stages, ())
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            runner, "audit_casadi_nlp_derivatives",
        ) as legacy_audit:
            case = Path(temporary)
            payload = runner._run_selected_derivative_audit(
                case,
                route="surrogate",
                case_id="nominal",
                influent=0.5 * (INFLUENT_LOWER + INFLUENT_UPPER),
                selected=selected,
                analysis=analysis,
                selection_contract="selection",
            )
            legacy_audit.assert_not_called()
            self.assertTrue(payload["passed"])
            self.assertEqual(payload["audited_point"], "exact_qp_active_set_endpoint")
            marker = json.loads((
                case / "surrogate_derivative_audit_complete.json"
            ).read_text())
            self.assertEqual(marker["stage"], "selected_exact_qp_active_set_audit")

    def test_repeated_untouched_test_inference_timing_is_retired(self) -> None:
        design, _, _, base_analysis = _fixture()
        cached_raw = np.full((2, RESPONSE_COUNT), 7.0)

        class CountingModel:
            response_count = RESPONSE_COUNT
            response_scale = np.ones(RESPONSE_COUNT)

            def __init__(self) -> None:
                self.calls = 0

            def predict(self, decisions: np.ndarray, influents: np.ndarray) -> np.ndarray:
                self.calls += 1
                self.assert_shapes(decisions, influents)
                return cached_raw.copy()

            @staticmethod
            def assert_shapes(decisions: np.ndarray, influents: np.ndarray) -> None:
                if np.asarray(decisions).shape != (2, 7):
                    raise AssertionError("raw timing was not evaluated as one fixed-order batch")
                if np.asarray(influents).shape != (2, 20):
                    raise AssertionError("raw timing influents changed shape or order")

        model = CountingModel()
        analysis = runner.AnalysisBundle(
            passed=True,
            model=model,
            direct_assets=base_analysis.direct_assets,
            surrogate_assets=base_analysis.surrogate_assets,
            assessment=SimpleNamespace(raw=cached_raw),
            gate={"passed": True},
        )

        class CountingProjector:
            def __init__(self) -> None:
                self.calls = 0

            def project(self, raw: np.ndarray, *_args: object, **_kwargs: object):
                # All 35 raw batches must have completed before the cached-only
                # projection section starts; no prediction may occur here.
                if model.calls != 5 + 30:
                    raise AssertionError("projection timing recomputed raw inference")
                self.calls += 1
                return SimpleNamespace(
                    accepted=bool(self.calls % 2 == 0),
                    state=np.asarray(raw).copy(),
                )

        projector = CountingProjector()
        ticks = iter(
            value
            for batch in range(60)
            for value in (batch * 1_000, batch * 1_000 + 100)
        )
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            runner, "PhysicalProjector", return_value=projector,
        ), patch.object(
            runner, "perf_counter_ns", side_effect=lambda: next(ticks),
        ), patch.object(runner, "assert_source_unchanged"):
            run = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "timing is retired"):
                runner._run_inference_timing_benchmark(
                    run,
                    design,
                    analysis,
                    source_files={"mock": "source"},
                    analysis_id="analysis",
                )
            return
            self.assertEqual(model.calls, 35)
            self.assertEqual(projector.calls, 35 * 2)
            self.assertEqual(
                frame.groupby("category").size().to_dict(),
                {"qp_deployment": 30, "raw_inference": 30},
            )
            np.testing.assert_allclose(
                frame["per_response_latency_seconds"],
                frame["elapsed_seconds"] / 2.0,
            )
            qp = frame.loc[frame["category"].eq("qp_deployment")]
            self.assertTrue((qp["projection_accepted_count"] == 1).all())
            self.assertTrue((qp["projection_accepted_fraction"] == 0.5).all())
            summary = json.loads(
                (run / "metrics/inference_timing_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["warmup_count"], 5)
            self.assertEqual(summary["timed_batch_count_per_route"], 30)
            self.assertTrue(summary["raw_inference_excluded_from_qp_timing"])
            qp_summary = summary["categories"]["qp_deployment"]
            self.assertEqual(qp_summary["projection_accepted_count"], 30)
            self.assertEqual(qp_summary["projection_attempt_count"], 60)
            self.assertEqual(qp_summary["projection_accepted_fraction"], 0.5)
            # The complete marker makes a second invocation a zero-work resume.
            second = runner._run_inference_timing_benchmark(
                run,
                design,
                analysis,
                source_files={"mock": "source"},
                analysis_id="analysis",
            )
            self.assertEqual(len(second), 60)
            self.assertEqual(model.calls, 35)
            self.assertEqual(projector.calls, 70)

    def test_projection_timing_still_rejects_nonfinite_state(self) -> None:
        design, _, _, analysis = _fixture()
        raw = np.full((2, RESPONSE_COUNT), 7.0)

        class InvalidProjector:
            @staticmethod
            def project(value: np.ndarray, *_args: object, **_kwargs: object):
                state = np.asarray(value, dtype=float).copy()
                state[0] = np.nan
                return SimpleNamespace(accepted=False, state=state)

        with self.assertRaisesRegex(RuntimeError, "invalid state at test row 0"):
            runner._projection_inference_batch(
                raw, np.asarray(design["test_decisions"]),
                np.asarray(design["test_influents"]), InvalidProjector(),
                analysis.surrogate_assets.layout,
            )

    def test_timing_is_aggregated_from_ten_robustness_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "metrics").mkdir()
            for index in range(1, 11):
                case_id = f"robustness_{index:02d}"
                case = run / "optimization" / case_id
                case.mkdir(parents=True)
                runner.atomic_json(case / "surrogate.json", {
                    "elapsed_seconds": float(index), "status": "selected",
                })
                runner.atomic_json(case / "direct.json", {
                    "elapsed_seconds": float(index + 10), "status": "selected",
                })
                runner.atomic_json(case / "shared_unit.json", {
                    "elapsed_seconds": float(index + 5), "status": "selected",
                    "root_attempts": 20, "failed_roots": 0,
                    "projection_solves": 10,
                })
                runner.atomic_json(case / "surrogate_local_convergence.json", {
                    "certificate": {"elapsed_seconds": 0.5},
                })
                for route in ("surrogate", "shared_unit", "direct"):
                    runner.atomic_json(case / f"{route}_casewise_reference.json", {
                        "candidate_available": True,
                        "comparison_valid": True,
                        "optimization_elapsed_seconds": (
                            float(index) + 0.5
                            if route == "surrogate"
                            else float(index + 5)
                            if route == "shared_unit"
                            else float(index + 10)
                        ),
                        "reference_elapsed_seconds": 0.25,
                        "recovery": {"attempted": False},
                    })
                comparison = case / "common_reference_comparison.json"
                runner.atomic_json(comparison, {"case": case_id})
                runner.atomic_json(case / "casewise_comparison_complete.json", {
                    "case": case_id,
                    "artifacts": {
                        comparison.relative_to(run).as_posix():
                        runner.file_digest(comparison),
                    },
                })

            frame = runner._run_robustness_case_timing_aggregation(
                run, source_files={"unit": "source"}, analysis_id="analysis",
            )
            self.assertEqual(len(frame), 30)
            means = frame.groupby("route")[
                "complete_optimization_seconds"
            ].mean()
            self.assertAlmostEqual(means["surrogate"], 6.0)
            self.assertAlmostEqual(means["shared_unit"], 10.5)
            self.assertAlmostEqual(means["direct"], 15.5)
            summary = json.loads((
                run / "metrics/robustness_case_timing_summary.json"
            ).read_text())
            self.assertEqual(summary["robustness_case_count"], 10)
            self.assertEqual(summary["repeated_test_batch_count"], 0)

    def test_derivative_failure_retains_cross_evaluation_but_rejects_route(self) -> None:
        _, _, _, analysis = _fixture()
        normalized = np.asarray(runner.direct_normalized_starts()[0], dtype=float)
        selected = _direct_start(0, normalized)
        result = DirectMultistartResult(
            (selected,), selected, "selected_stationary"
        )
        route_payload = result.as_dict()
        route_payload["route_contract"] = "direct-contract"
        influent = 0.5 * (INFLUENT_LOWER + INFLUENT_UPPER)
        fixed_route = SimpleNamespace(
            response=np.full(MECHANISTIC_RESPONSE_COUNT, 3.0),
            state=selected.state.copy(),
            feed_tss=selected.feed_tss,
            branch=selected.branch,
        )
        smooth = SimpleNamespace(
            accepted=True, routes=(fixed_route, fixed_route)
        )

        def failed_audit(case_directory: Path, *, route: str, **_kwargs: object):
            payload = {
                "passed": False,
                "status": "failed_tolerance",
                "maximum_jacobian_discrepancy": 2.0e-5,
            }
            audit = case_directory / f"{route}_derivative_audit.json"
            runner.atomic_json(audit, payload)
            runner.atomic_json(
                case_directory / f"{route}_derivative_audit_complete.json",
                {"passed": False, "artifacts": {audit.name: runner.file_digest(audit)}},
            )
            return payload

        replay = (
            np.full(MECHANISTIC_RESPONSE_COUNT, 4.0),
            np.ones(REDUCED_STATE_COUNT),
            np.ones(REDUCED_STATE_COUNT),
            {"accepted": True, "branch_agreement": True},
        )
        violation = lambda method, case, *_a, **_k: {
            "case": case,
            "method": method,
            "mass_conservation_violation_max": 0.0,
            "nonnegativity_violation_max": 0.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            case_directory = Path(temporary) / "nominal"
            case_directory.mkdir(parents=True)
            runner.atomic_json(
                case_directory / "direct.json", route_payload,
                nonfinite_to_none=True,
            )
            with patch.object(
                runner, "_run_selected_derivative_audit", side_effect=failed_audit,
            ), patch.object(
                runner, "cold_reproject", return_value=_projection(
                    np.full(RESPONSE_COUNT, 2.0)
                ),
            ), patch.object(
                runner, "solve_fixed_input_two_start", return_value=smooth,
            ), patch.object(
                runner, "compare_smooth_reference", return_value=_equivalence(),
            ), patch.object(
                runner, "_reference_two_start", return_value=replay,
            ), patch.object(
                runner, "violation_record", side_effect=violation,
            ):
                available, accepted, violations = runner._evaluate_selected_route(
                    case_directory,
                    case_id="nominal",
                    route="direct",
                    influent=influent,
                    result=result,
                    route_payload=route_payload,
                    analysis=analysis,
                    source_id="source",
                    analysis_id="analysis",
                )
            self.assertTrue(available)
            self.assertFalse(accepted)
            self.assertEqual(set(violations["method"]), {
                "raw", "projected", "smooth", "reference",
            })
            self.assertTrue((case_directory / "direct_selected.npz").is_file())
            self.assertTrue((case_directory / "direct_reference.npz").is_file())
            equivalence = json.loads(
                (case_directory / "direct_equivalence.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(equivalence["accepted"])
            self.assertFalse(equivalence["derivative_audit"]["passed"])
            marker = json.loads(
                (case_directory / "direct_selection_complete.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(marker["accepted"])

    def test_full_hook_runs_11_cases_and_cross_evaluates_three_routes(self) -> None:
        self.assertEqual(
            runner.COMPARISON_PROTOCOL, "casewise_exact_common_reference_v4"
        )
        design, development_targets, test_targets, analysis = _fixture()
        analysis = runner.AnalysisBundle(
            passed=analysis.passed,
            model=analysis.model,
            direct_assets=analysis.direct_assets,
            surrogate_assets=analysis.surrogate_assets,
            assessment=analysis.assessment,
            gate=analysis.gate,
            shared_unit_assets=SimpleNamespace(),
            shared_unit_gate={"passed": True},
        )
        surrogate_completed_on_entry: list[set[int]] = []
        direct_completed_on_entry: list[set[int]] = []

        def surrogate_solver(*args: object, **kwargs: object):
            surrogate_completed_on_entry.append(
                set() if kwargs.get("completed_result") is None else {0}
            )
            return _surrogate_solver(*args, **kwargs)

        def direct_solver(*args: object, **kwargs: object):
            direct_completed_on_entry.append(
                set((kwargs.get("completed_starts") or {}).keys())
            )
            return _direct_solver(*args, **kwargs)

        def shared_unit_route(
            case_directory: Path,
            *,
            case_id: str,
            **_kwargs: object,
        ):
            normalized = np.full(7, 0.5)
            candidate = SimpleNamespace(
                normalized_controls=normalized,
                theta=DECISION_LOWER + (DECISION_UPPER - DECISION_LOWER) * normalized,
                objective=1.5,
                status="validated_feasible_value_only_local",
            )
            result = SimpleNamespace(
                selected=candidate,
                status="selected_locally_converged",
                root_attempts=20,
                failed_roots=0,
                projection_solves=10,
            )
            payload = {
                "route": "shared_unit",
                "case_id": case_id,
                "status": result.status,
                "classification": "two_scale_feasible_no_descent",
                "locally_converged": True,
                "stationarity_resolved": False,
                "elapsed_seconds": 0.5,
                "root_attempts": result.root_attempts,
                "failed_roots": result.failed_roots,
                "projection_solves": result.projection_solves,
                "selected": {"objective": candidate.objective},
            }
            runner.atomic_json(case_directory / "shared_unit.json", payload)
            runner.atomic_json(
                case_directory / "shared_unit_complete.json",
                {"status": result.status},
            )
            return result, payload

        def certify(
            case_directory: Path,
            *,
            result: SurrogateMultistartResult,
            **_kwargs: object,
        ):
            self.assertIsNotNone(result.selected)
            assert result.selected is not None and result.selected.final is not None
            candidate = result.selected.final
            payload = {
                "selected": True,
                "locally_converged": True,
                "first_order_certified": True,
                "status": "exact_active_set_kkt",
                "candidate": candidate.as_dict(),
                "certificate": {"elapsed_seconds": 0.01},
            }
            runner.atomic_json(
                case_directory / "surrogate_local_convergence_complete.json",
                {"locally_converged": True},
            )
            return candidate, payload

        def casewise_reference(
            case_directory: Path,
            *,
            case_id: str,
            route: str,
            selected: object,
            surrogate_candidate: FinalCandidateRecord | None,
            route_payload: dict[str, object],
            **_kwargs: object,
        ):
            candidate = surrogate_candidate if route == "surrogate" else selected
            self.assertIsNotNone(candidate)
            assert candidate is not None
            objective = float(candidate.objective)
            normalized = np.asarray(candidate.normalized_controls, dtype=float)
            payload = {
                "case": case_id,
                "route": route,
                "candidate_available": True,
                "native_feasible": True,
                "exact_replay_valid": True,
                "comparison_valid": True,
                "status": "valid_interior",
                "native_status": candidate.status,
                "native_objective": objective,
                "exact_reference_objective": objective,
                "native_minus_reference_objective": 0.0,
                "normalized_controls": normalized.tolist(),
                "exact_reference_objective_components": np.full(6, 1.0 / 6.0).tolist(),
                "local_convergence_certified": True,
                "first_order_stationarity_certified": True,
                "optimization_elapsed_seconds": route_payload["elapsed_seconds"],
                "certification_elapsed_seconds": 0.01 if route == "surrogate" else None,
                "reference_elapsed_seconds": 0.02,
                "reference": {
                    "status": "valid_interior",
                    "branch_ambiguous": False,
                    "engineering_feasible": True,
                },
            }
            physical = pd.DataFrame([
                {
                    "case": f"{case_id}:{route}",
                    "method": method,
                    "decision_route": route,
                    "response_source": method,
                    "audit_available": True,
                    "mass_conservation_violation_max": 1.0e-12,
                    "mass_conservation_violation_count": 0,
                    "nonnegativity_violation_max": 0.0,
                    "nonnegativity_violation_count": 0,
                }
                for method in (
                    "raw",
                    "projected",
                    "optimizer_native",
                    "exact_mechanistic_start_1",
                    "exact_mechanistic_start_2",
                )
            ])
            runner.atomic_json(
                case_directory / f"{route}_casewise_reference_complete.json",
                {"candidate_available": True, "comparison_valid": True},
            )
            return payload, physical

        def report(run: Path, *, output_directory: Path, expected_cases: tuple[str, ...]):
            self.assertEqual(
                expected_cases,
                ("nominal", *(f"robustness_{index:02d}" for index in range(1, 11))),
            )
            output_directory.mkdir(parents=True, exist_ok=True)
            runner.atomic_json(output_directory / "report_manifest.json", {
                "expected_cases": list(expected_cases),
            })
            return SimpleNamespace(warnings=())

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "article"
            (run / "metrics").mkdir(parents=True)
            pd.DataFrame([
                {
                    "case": "test_0000",
                    "method": method,
                    "mass_conservation_violation_max": 0.0,
                    "mass_conservation_violation_count": 0,
                    "nonnegativity_violation_max": 0.0,
                    "nonnegativity_violation_count": 0,
                }
                for method in ("raw", "projected", "mechanistic")
            ]).to_csv(run / "metrics/physical_violations_assessment.csv", index=False)

            with patch.object(
                runner,
                "_run_untouched_test_equivalence",
                side_effect=AssertionError("retired whole-test equivalence was called"),
            ) as untouched_equivalence, patch.object(
                runner,
                "build_surrogate_nlp",
                return_value=SimpleNamespace(assets=analysis.surrogate_assets),
            ) as graph_builder, patch.object(
                runner, "_run_surrogate_certification", side_effect=certify,
            ) as certification, patch.object(
                runner,
                "_run_casewise_route_reference_evaluation",
                side_effect=casewise_reference,
            ) as reference_evaluation, patch.object(
                runner, "solve_surrogate_exact_qp_local", side_effect=surrogate_solver,
            ) as surrogate_solver, patch.object(
                runner, "solve_direct_multistart", side_effect=direct_solver,
            ) as direct_solver, patch.object(
                runner, "_run_shared_unit_route", side_effect=shared_unit_route,
            ) as shared_unit_solver, patch.object(
                runner, "_run_robustness_case_timing_aggregation", side_effect=_mock_timing,
            ) as timing_benchmark, patch.object(
                runner, "write_reporting_tables", side_effect=report,
            ) as reporting, patch.object(runner, "assert_source_unchanged"):
                passed = runner.run_optimization_stage(
                    run=run,
                    profile=ARTICLE_FULL,
                    design=design,
                    development_targets=development_targets,
                    test_targets=test_targets,
                    analysis=analysis,
                    source_files={"mock": "source"},
                )

                self.assertTrue(passed)
                untouched_equivalence.assert_not_called()
                self.assertEqual(surrogate_solver.call_count, 11)
                self.assertEqual(shared_unit_solver.call_count, 11)
                self.assertEqual(direct_solver.call_count, 11)
                self.assertEqual(certification.call_count, 11)
                self.assertEqual(reference_evaluation.call_count, 33)
                graph_builder.assert_called_once()
                for call in surrogate_solver.call_args_list:
                    self.assertNotIn("starts", call.kwargs)
                    self.assertIs(call.kwargs["problem"], graph_builder.return_value)
                    self.assertIsNone(call.kwargs["settings"].maximum_wall_time)
                for call in direct_solver.call_args_list:
                    np.testing.assert_array_equal(
                        np.asarray(call.kwargs["starts"]),
                        np.full((1, 7), 0.5),
                    )
                    self.assertTrue(call.kwargs["allow_reduced_starts"])
                    self.assertIsNone(call.kwargs["settings"].maximum_wall_time)
                self.assertEqual(surrogate_completed_on_entry, [set()] * 11)
                self.assertEqual(direct_completed_on_entry, [set()] * 11)
                self.assertEqual(timing_benchmark.call_count, 1)
                self.assertEqual(reporting.call_count, 1)

                selected = pd.read_csv(
                    run / "metrics/selected_response_physical_audit.csv"
                )
                self.assertEqual(len(selected), 11 * 3 * 5)
                self.assertEqual(
                    selected.groupby("method").size().to_dict(),
                    {
                        method: 33
                        for method in (
                            "raw",
                            "projected",
                            "optimizer_native",
                            "exact_mechanistic_start_1",
                            "exact_mechanistic_start_2",
                        )
                    },
                )
                comparison = pd.read_csv(
                    run / "metrics/case_common_reference_comparison.csv"
                )
                reference = pd.read_csv(
                    run / "metrics/selected_candidate_reference_evaluation.csv"
                )
                self.assertEqual(len(comparison), 11)
                self.assertTrue(comparison["comparison_eligible"].all())
                self.assertEqual(len(reference), 33)
                self.assertTrue(reference["comparison_valid"].all())
                retired = json.loads(
                    (run / "metrics/untouched_test_equivalence_retired.json").read_text()
                )
                self.assertEqual(
                    retired["status"], "retired_incomplete_excluded_from_analysis"
                )
                status = json.loads(
                    (run / "optimization/final_status.json").read_text(encoding="utf-8")
                )
                self.assertEqual(status["case_count"], 11)
                self.assertEqual(status["route_count"], 33)
                self.assertEqual(status["required_starts_per_route"], 1)
                self.assertEqual(status["required_attempts_per_route"], 1)
                self.assertEqual(status["surrogate_ipopt_continuation_stage_count"], 0)
                self.assertFalse(status["untouched_test_equivalence_executed"])
                self.assertEqual(status["selected_decision_count"], 33)
                self.assertTrue(status["scientific_validation_passed"])

                for case_id in (
                    "nominal", *(f"robustness_{index:02d}" for index in range(1, 11))
                ):
                    case = run / "optimization" / case_id
                    self.assertTrue((case / "casewise_comparison_complete.json").is_file())
                    self.assertTrue(
                        (case / "surrogate_local_convergence_complete.json").is_file()
                    )
                    for route in ("surrogate", "direct"):
                        payload = json.loads(
                            (case / f"{route}.json").read_text(encoding="utf-8")
                        )
                        self.assertEqual(len(payload["starts"]), 1)
                        self.assertEqual(payload["optimization_attempt_count"], 1)
                        self.assertIsNone(payload["maximum_wall_time"])
                        self.assertEqual(
                            len(list((case / "checkpoints").glob(f"{route}_start_*.json"))),
                            1,
                        )
                        self.assertTrue(
                            (case / f"{route}_casewise_reference_complete.json").is_file()
                        )
                    shared_payload = json.loads(
                        (case / "shared_unit.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(shared_payload["route"], "shared_unit")
                    self.assertTrue(
                        (case / "shared_unit_casewise_reference_complete.json").is_file()
                    )
                self.assertFalse(
                    any(path.name.endswith(".tmp") for path in run.rglob("*"))
                )

                surrogate_solver.reset_mock()
                shared_unit_solver.reset_mock()
                direct_solver.reset_mock()
                graph_builder.reset_mock()
                certification.reset_mock()
                reference_evaluation.reset_mock()
                timing_benchmark.reset_mock()
                reporting.reset_mock()
                self.assertTrue(runner.run_optimization_stage(
                    run=run,
                    profile=ARTICLE_FULL,
                    design=design,
                    development_targets=development_targets,
                    test_targets=test_targets,
                    analysis=analysis,
                    source_files={"mock": "source"},
                ))
                surrogate_solver.assert_not_called()
                self.assertEqual(shared_unit_solver.call_count, 11)
                direct_solver.assert_not_called()
                graph_builder.assert_called_once()
                self.assertEqual(certification.call_count, 11)
                self.assertEqual(reference_evaluation.call_count, 33)
                self.assertEqual(timing_benchmark.call_count, 1)
                self.assertEqual(reporting.call_count, 1)

    def test_surrogate_route_resumes_the_single_atomic_attempt(self) -> None:
        center = np.asarray(EXACT_QP_CENTER_START, dtype=float)
        phase = {"first": True}
        observed_completed: list[set[int]] = []

        def interrupted(*_args: object, **kwargs: object):
            completed = kwargs.get("completed_result")
            observed_completed.append(set() if completed is None else {0})
            callback = kwargs["progress_callback"]
            if phase["first"]:
                phase["first"] = False
                callback(_surrogate_start(0, center))
                raise RuntimeError("simulated interruption")
            result = completed
            self.assertIsNotNone(result)
            return SurrogateMultistartResult(
                (result,), result, "selected_stationary",
                protocol=EXACT_QP_SINGLE_START_PROTOCOL,
            )

        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / "nominal"
            # An interrupted schema-4 multistart artifact must never be treated
            # as the schema-5 exact-QP attempt, even though both use index zero.
            runner.atomic_json(
                case / "checkpoints/surrogate_start_00.json",
                {
                    "route_contract": "legacy-nine-start-contract",
                    "protocol": "embedded_kkt_continuation_multistart",
                    "start_index": 0,
                    "normalized_start": center,
                    "result": _surrogate_start(0, center).as_dict(),
                },
                nonfinite_to_none=True,
            )
            with patch.object(
                runner, "solve_surrogate_exact_qp_local", side_effect=interrupted,
            ) as solver:
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    runner._run_surrogate_route(
                        case,
                        case_id="nominal",
                        influent=0.5 * (INFLUENT_LOWER + INFLUENT_UPPER),
                        assets=object(),
                        source_id="source",
                        analysis_id="analysis",
                    )
                self.assertEqual(
                    sorted(path.name for path in (case / "checkpoints").glob("*.json")),
                    ["surrogate_start_00.json"],
                )
                result, payload = runner._run_surrogate_route(
                    case,
                    case_id="nominal",
                    influent=0.5 * (INFLUENT_LOWER + INFLUENT_UPPER),
                    assets=object(),
                    source_id="source",
                    analysis_id="analysis",
                )
                self.assertEqual(observed_completed, [set(), {0}])
                self.assertEqual(len(result.starts), 1)
                self.assertEqual(len(payload["starts"]), 1)
                self.assertEqual(payload["protocol"], EXACT_QP_SINGLE_START_PROTOCOL)
                self.assertEqual(payload["starts"][0]["stages"], [])
                self.assertTrue((case / "surrogate_complete.json").is_file())
                self.assertFalse(
                    any(path.name.endswith(".tmp") for path in case.rglob("*"))
                )
                solver.reset_mock()
                restored, restored_payload = runner._run_surrogate_route(
                    case,
                    case_id="nominal",
                    influent=0.5 * (INFLUENT_LOWER + INFLUENT_UPPER),
                    assets=object(),
                    source_id="source",
                    analysis_id="analysis",
                )
                solver.assert_not_called()
                self.assertEqual(len(restored.starts), 1)
                self.assertEqual(restored_payload["selected_start"], 0)

    def test_direct_route_resumes_the_single_atomic_attempt(self) -> None:
        starts = np.full((1, 7), 0.5)
        phase = {"first": True}
        observed_completed: list[set[int]] = []

        def interrupted(*_args: object, **kwargs: object):
            completed = dict(kwargs.get("completed_starts") or {})
            observed_completed.append(set(completed))
            callback = kwargs["progress_callback"]
            if phase["first"]:
                phase["first"] = False
                callback(_direct_start(0, starts[0]))
                raise RuntimeError("simulated direct interruption")
            for index in range(1):
                if index not in completed:
                    result = _direct_start(index, starts[index])
                    completed[index] = result
                    callback(result)
            ordered = tuple(completed[index] for index in range(1))
            return DirectMultistartResult(ordered, ordered[0], "selected_stationary")

        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / "nominal"
            with patch.object(
                runner, "solve_direct_multistart", side_effect=interrupted,
            ) as solver:
                arguments = dict(
                    case_id="nominal",
                    influent=0.5 * (INFLUENT_LOWER + INFLUENT_UPPER),
                    assets=object(),
                    development_decisions=np.zeros((1, 7)),
                    development_influents=np.zeros((1, 20)),
                    development_targets=np.zeros((1, MECHANISTIC_RESPONSE_COUNT)),
                    source_id="source",
                    analysis_id="analysis",
                )
                with self.assertRaisesRegex(
                    RuntimeError, "simulated direct interruption"
                ):
                    runner._run_direct_route(case, **arguments)
                self.assertEqual(
                    sorted(path.name for path in (case / "checkpoints").glob("*.json")),
                    ["direct_start_00.json"],
                )
                result, payload = runner._run_direct_route(case, **arguments)
                self.assertEqual(observed_completed, [set(), {0}])
                self.assertEqual(len(result.starts), 1)
                self.assertEqual(len(payload["starts"]), 1)
                self.assertTrue((case / "direct_complete.json").is_file())
                solver.reset_mock()
                restored, restored_payload = runner._run_direct_route(case, **arguments)
                solver.assert_not_called()
                self.assertEqual(len(restored.starts), 1)
                self.assertEqual(restored_payload["selected_start"], 0)

    def test_branch_boundary_is_a_qualifier_when_exact_replays_agree(self) -> None:
        theta = 0.5 * (DECISION_LOWER + DECISION_UPPER)
        influent = 0.5 * (INFLUENT_LOWER + INFLUENT_UPPER)
        state = np.ones(3)
        analysis = SimpleNamespace(
            direct_assets=SimpleNamespace(
                clarifier=object(),
                state_count=3,
                response_count=4,
                state_scale=np.ones(3),
            )
        )
        ambiguous = BranchClassification((), (), (), (), (), True, 0.0)
        solved = SimpleNamespace(accepted=True, state=state, message="accepted")
        with (
            patch.object(
                runner, "solve_steady_state", side_effect=(solved, solved),
            ) as solve,
            patch.object(runner, "classify_branches", return_value=ambiguous),
            patch.object(runner, "mechanistic_diagnostics", return_value={"passed": True}),
            patch.object(
                runner, "unpack_state", return_value=(np.ones((1, 20)), np.ones(1)),
            ),
            patch.object(runner, "generation_scale", return_value=np.ones(3)),
            patch.object(runner, "smooth_branches_match", return_value=True),
            patch.object(runner, "assemble_target", return_value=np.ones(4)),
            patch.object(runner, "objective_components", return_value=np.ones(6)),
            patch.object(runner, "engineering_quantities", return_value=np.ones(7)),
            patch.object(runner, "engineering_feasible", return_value=True),
        ):
            response, start_1, start_2, payload = runner._casewise_exact_reference(
                theta, influent, analysis,
            )

        self.assertTrue(payload["accepted"])
        self.assertEqual(payload["status"], "valid_branch_boundary")
        self.assertTrue(payload["branch_ambiguous"])
        self.assertTrue(payload["start_2_required"])
        self.assertTrue(payload["two_start_agreement_checked"])
        self.assertEqual([call.kwargs["starts"] for call in solve.call_args_list], [(1,), (2,)])
        np.testing.assert_array_equal(response, np.ones(4))
        np.testing.assert_array_equal(start_1, state)
        np.testing.assert_array_equal(start_2, state)

    def test_valid_exact_replay_retains_objective_when_engineering_is_infeasible(
        self,
    ) -> None:
        _, _, _, analysis = _fixture()
        analysis.direct_assets.clarifier = object()
        normalized = np.asarray(EXACT_QP_CENTER_START, dtype=float)
        selected = _direct_start(0, normalized)
        route_payload = DirectMultistartResult(
            (selected,), selected, "selected_stationary"
        ).as_dict()
        route_payload.update({"route_contract": "direct-contract", "elapsed_seconds": 0.5})
        reference = np.full(MECHANISTIC_RESPONSE_COUNT, 4.0)
        replay = (
            reference,
            np.ones(REDUCED_STATE_COUNT),
            np.ones(REDUCED_STATE_COUNT),
            {
                "accepted": True,
                "status": "valid_interior",
                "branch_ambiguous": False,
                "engineering_feasible": False,
                "objective": 12.5,
                "objective_components": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "elapsed_seconds": 0.2,
            },
        )

        def physical(method: str, case: str, *_args: object, **_kwargs: object):
            return {
                "case": case,
                "method": method,
                "audit_available": True,
                "mass_conservation_violation_max": 0.0,
                "mass_conservation_violation_count": 0,
                "nonnegativity_violation_max": 0.0,
                "nonnegativity_violation_count": 0,
            }

        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / "nominal"
            runner.atomic_json(
                case / "direct.json", route_payload, nonfinite_to_none=True,
            )
            with (
                patch.object(
                    runner,
                    "cold_reproject",
                    return_value=_projection(np.full(RESPONSE_COUNT, 2.0)),
                ),
                patch.object(
                    runner, "_casewise_exact_reference", return_value=replay,
                ),
                patch.object(runner, "assemble_target", return_value=reference),
                patch.object(runner, "_physical_record", side_effect=physical),
            ):
                payload, violations = (
                    runner._run_casewise_route_reference_evaluation(
                        case,
                        case_id="nominal",
                        route="direct",
                        influent=0.5 * (INFLUENT_LOWER + INFLUENT_UPPER),
                        selected=selected,
                        surrogate_candidate=None,
                        route_payload=route_payload,
                        certification_payload=None,
                        recovery_payload=None,
                        analysis=analysis,
                        source_id="source",
                        analysis_id="analysis",
                    )
                )

            self.assertTrue(payload["candidate_available"])
            self.assertTrue(payload["exact_replay_valid"])
            self.assertFalse(payload["comparison_valid"])
            self.assertEqual(payload["status"], "exact_valid_engineering_infeasible")
            self.assertEqual(payload["exact_reference_objective"], 12.5)
            self.assertEqual(
                payload["exact_reference_objective_components"],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )
            self.assertAlmostEqual(
                payload["native_minus_reference_objective"],
                selected.objective - 12.5,
            )
            self.assertEqual(len(violations), 5)
            marker = json.loads((
                case / "direct_casewise_reference_complete.json"
            ).read_text())
            self.assertFalse(marker["comparison_valid"])
            with np.load(case / "direct_casewise_reference.npz") as arrays:
                np.testing.assert_array_equal(
                    arrays["exact_reference"],
                    reduce_mechanistic_responses(reference, ARTICLE_FULL.layer_count),
                )
                np.testing.assert_array_equal(arrays["exact_reference_full"], reference)

    def test_direct_recovery_runs_only_after_primary_failure(self) -> None:
        normalized = np.asarray(EXACT_QP_CENTER_START, dtype=float)
        primary_start = _direct_start(0, normalized)
        primary = DirectMultistartResult(
            (primary_start,), primary_start, "selected_stationary"
        )
        surrogate_candidate = _surrogate_start(0, normalized).final
        assert surrogate_candidate is not None
        common = {
            "case_id": "nominal",
            "influent": 0.5 * (INFLUENT_LOWER + INFLUENT_UPPER),
            "surrogate_candidate": surrogate_candidate,
            "assets": object(),
            "development_decisions": np.zeros((1, 7)),
            "development_influents": np.zeros((1, 20)),
            "development_targets": np.zeros((1, MECHANISTIC_RESPONSE_COUNT)),
            "source_id": "source",
            "analysis_id": "analysis",
        }
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / "nominal"
            with patch.object(runner, "solve_direct_multistart") as solve:
                returned, payload = runner._run_direct_failure_recovery(
                    case, result=primary, **common,
                )
            self.assertIs(returned, primary)
            self.assertFalse(payload["attempted"])
            self.assertEqual(payload["status"], "not_required")
            solve.assert_not_called()

            case.mkdir(parents=True, exist_ok=True)
            runner.atomic_json(case / "direct.json", {"route": "direct"})
            failed = DirectMultistartResult(
                (), None, "no_validated_feasible_start"
            )
            recovered_start = _direct_start(0, normalized)
            recovered = DirectMultistartResult(
                (recovered_start,), recovered_start, "selected_stationary"
            )
            with patch.object(
                runner, "solve_direct_multistart", return_value=recovered,
            ) as solve:
                returned, payload = runner._run_direct_failure_recovery(
                    case, result=failed, **common,
                )
            self.assertIs(returned, recovered)
            self.assertTrue(payload["attempted"])
            self.assertEqual(payload["selected_from"], "single_surrogate_endpoint_recovery")
            solve.assert_called_once()
            np.testing.assert_array_equal(
                solve.call_args.kwargs["starts"], normalized.reshape(1, 7)
            )
            self.assertTrue(solve.call_args.kwargs["allow_reduced_starts"])
            self.assertTrue((case / "direct_recovery_complete.json").is_file())

    def test_shared_unit_metrics_record_zero_coverage_without_name_error(self) -> None:
        layout = NetworkLayout(layer_count=ARTICLE_FULL.layer_count)
        unavailable = np.full((1, layout.state_size), np.nan)
        metrics = runner._shared_unit_prediction_metrics(
            unavailable,
            unavailable,
            np.ones((1, layout.state_size)),
            np.asarray([False]),
            np.ones(layout.state_size),
            layout,
        )
        self.assertFalse(metrics.empty)
        self.assertTrue(metrics["coverage_fraction"].eq(0.0).all())
        self.assertTrue(metrics["nrmse"].isna().all())


if __name__ == "__main__":
    unittest.main()
