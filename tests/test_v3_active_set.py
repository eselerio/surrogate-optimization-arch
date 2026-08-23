from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import closed_loop.v3_surrogate_nlp as surrogate_module
from closed_loop.manuscript_v3 import DECISION_LOWER, DECISION_UPPER
from closed_loop.projection import (
    LeastSquaresDiagnostics,
    NetworkLayout,
    NetworkRowScales,
    QuadraticFeatureMap,
    QuadraticSurrogate,
)
from closed_loop.v3_active_set import (
    ActiveSetDerivativeError,
    ActiveSetRefinementSettings,
    ExactQPActiveSetRefiner,
    UpperKKTAudit,
)
from closed_loop.v3_surrogate_nlp import (
    EXACT_QP_CENTER_START,
    EXACT_QP_SINGLE_START_PROTOCOL,
    LOCAL_CONVERGENCE_PROTOCOL,
    EngineeringLimits,
    ContinuationStageRecord,
    OuterRefinementRecord,
    StationarityRecord,
    SurrogateCase,
    SurrogateCertificationSettings,
    SurrogateNLPAssets,
    SurrogateSolverSettings,
    SurrogateStartResult,
    TrustThresholds,
    _outer_refine,
    audit_exact_candidate,
    build_surrogate_nlp,
    certify_surrogate_local_convergence,
    cold_reproject,
    ordered_normalized_starts,
    solve_surrogate_exact_qp_local,
    solve_surrogate_multistart,
    surrogate_exact_qp_resume_contract,
    surrogate_start_resume_contract,
)


def _toy_assets(*, weak_active_set: bool = False) -> tuple[SurrogateNLPAssets, SurrogateCase]:
    """Return a small, physical one-stage network with a constant surrogate."""

    layout = NetworkLayout(
        stage_count=1,
        component_count=2,
        layer_count=3,
        soluble_indices=(0,),
        particulate_indices=(1,),
    )
    theta = 0.5 * (DECISION_LOWER + DECISION_UPPER)
    internal, returned, waste = theta[4], theta[5], theta[6]
    underflow = returned + waste
    effluent = 1.0 - waste
    influent = np.asarray([10.0, 10.0])
    if weak_active_set:
        # The no-conversion solution makes all three physical-direction rows
        # active with zero multipliers, so strict complementarity must fail.
        response_center = np.concatenate(
            (
                influent,
                influent,
                effluent * influent,
                underflow * influent,
                np.full(3, 10.0),
            )
        )
    else:
        final = np.asarray([10.0, 10.0])
        underflow_flow = np.asarray([underflow * 10.0, underflow * 15.0])
        overflow_flow = (1.0 + returned) * final - underflow_flow
        primary = 1.0 + internal + returned
        mixer = np.asarray(
            [
                10.0,
                (
                    10.0
                    + internal * final[1]
                    + returned / underflow * underflow_flow[1]
                )
                / primary,
            ]
        )
        response_center = np.concatenate(
            (
                mixer,
                final,
                overflow_flow,
                underflow_flow,
                np.asarray([overflow_flow[1] / effluent, 10.0, 15.0]),
            )
        )

    nonconstant = QuadraticFeatureMap.expected_feature_count(7, 2) - 1
    feature_map = QuadraticFeatureMap(
        decision_center=theta,
        decision_scale=0.5 * (DECISION_UPPER - DECISION_LOWER),
        influent_center=influent,
        influent_scale=np.ones(2),
        term_center=np.zeros(nonconstant),
        term_scale=np.ones(nonconstant),
    )
    diagnostics = LeastSquaresDiagnostics(
        sample_count=100,
        feature_count=nonconstant + 1,
        response_count=layout.state_size,
        rank_tolerance=1.0,
        smallest_singular_value=1.0,
        largest_singular_value=1.0,
        condition_number=1.0,
        optimality_residual=0.0,
        coefficient_agreement=0.0,
        acceptance_threshold=1.0,
    )
    model = QuadraticSurrogate(
        feature_map=feature_map,
        response_center=response_center,
        response_scale=np.ones(layout.state_size),
        coefficients=np.zeros((layout.state_size, nonconstant + 1)),
        diagnostics=diagnostics,
        ridge_penalty=1.0,
    )
    assets = SurrogateNLPAssets(
        model=model,
        layout=layout,
        invariant_operator=np.asarray([[1.0, 0.0]]),
        tss_weights=np.asarray([0.0, 1.0]),
        row_scales=NetworkRowScales(np.ones(8), np.ones(3)),
        leverage_precision=np.zeros((nonconstant + 1, nonconstant + 1)),
        trust_thresholds=TrustThresholds(0.5, 1.0),
        quality_operator=np.asarray([[0.0, 1.0]]),
        quality_scale=np.ones(1),
        engineering=EngineeringLimits(
            fresh_flow_m3_d=1.0,
            clarifier_area_m2=1.0,
            clarifier_volume_m3=1.0,
            srt_lower_d=0.1,
            srt_upper_d=10.0,
            external_loss_min_g_m3=0.1,
            slr_upper_kg_m2_d=100.0,
            underflow_tss_upper_g_m3=100.0,
            feed_tss_min_g_m3=0.1,
            inventory_scale=100.0,
        ),
    )
    return assets, SurrogateCase(influent=influent, case_id="active_set_unit")


class ExactQPActiveSetTests(unittest.TestCase):
    def test_primary_exact_qp_protocol_is_one_center_start_without_ipopt(self) -> None:
        assets, case = _toy_assets()
        settings = SurrogateSolverSettings(outer_maximum_iterations=40)
        reported: list[SurrogateStartResult] = []
        original_build = surrogate_module.build_surrogate_nlp

        with (
            patch(
                "closed_loop.v3_surrogate_nlp.build_surrogate_nlp",
                wraps=original_build,
            ) as build,
            patch(
                "closed_loop.v3_surrogate_nlp._solve_continuation_stage",
                side_effect=AssertionError("the direct protocol called IPOPT"),
            ) as continuation,
        ):
            result = solve_surrogate_exact_qp_local(
                assets,
                case,
                settings=settings,
                progress_callback=reported.append,
            )

        self.assertEqual(result.protocol, EXACT_QP_SINGLE_START_PROTOCOL)
        self.assertEqual(len(result.starts), 1)
        self.assertEqual(reported, [result.starts[0]])
        start = result.starts[0]
        self.assertEqual(start.start_index, 0)
        np.testing.assert_array_equal(
            start.initial_normalized_controls,
            np.asarray(EXACT_QP_CENTER_START),
        )
        self.assertEqual(start.stages, ())
        self.assertEqual(start.protocol, EXACT_QP_SINGLE_START_PROTOCOL)
        self.assertEqual(
            start.resume_contract,
            surrogate_exact_qp_resume_contract(assets, case, settings),
        )
        self.assertTrue(start.outer_refinement.attempted)
        self.assertGreater(start.outer_refinement.cold_qp_resolutions, 0)
        self.assertIsNotNone(start.final)
        build.assert_called_once()
        self.assertFalse(build.call_args.kwargs["compile_solver"])
        self.assertEqual(build.call_args.args[1], 1.0e-8)
        continuation.assert_not_called()
        self.assertEqual(result.as_dict()["protocol"], EXACT_QP_SINGLE_START_PROTOCOL)
        self.assertEqual(
            result.as_dict()["starts"][0]["protocol"],
            EXACT_QP_SINGLE_START_PROTOCOL,
        )

    def test_primary_exact_qp_protocol_uses_derivative_free_fallback(self) -> None:
        assets, case = _toy_assets(weak_active_set=True)
        result = solve_surrogate_exact_qp_local(
            assets,
            case,
            settings=SurrogateSolverSettings(outer_maximum_iterations=40),
        )

        start = result.starts[0]
        self.assertEqual(start.stages, ())
        self.assertEqual(
            start.outer_refinement.status,
            "derivative_free_budget_limited_candidate",
        )
        self.assertEqual(
            start.outer_refinement.method,
            "exact_qp_derivative_free_cobyqa",
        )
        self.assertTrue(start.outer_refinement.fallback_used)
        self.assertEqual(start.outer_refinement.fallback_method, "COBYQA")
        self.assertGreater(start.outer_refinement.fallback_evaluations, 1)
        self.assertGreaterEqual(
            start.outer_refinement.cold_qp_resolutions,
            start.outer_refinement.fallback_evaluations,
        )
        self.assertIsNotNone(start.outer_refinement.derivative_error)
        self.assertIsNotNone(start.final)
        assert start.final is not None
        self.assertLess(
            start.final.objective,
            start.outer_refinement.initial_objective,
        )
        self.assertFalse(start.final.stationarity.stationary)
        self.assertEqual(
            start.final.stationarity.classification,
            "budget_limited_derivative_free_feasible_incumbent_stationarity_unresolved",
        )
        self.assertIn("not an established local optimum", start.final.stationarity.reason)

    def test_primary_exact_qp_checkpoint_resume_performs_no_recomputation(self) -> None:
        assets, case = _toy_assets(weak_active_set=True)
        settings = SurrogateSolverSettings(outer_maximum_iterations=10)
        first = solve_surrogate_exact_qp_local(assets, case, settings=settings)
        checkpoint = first.starts[0]

        with patch(
            "closed_loop.v3_surrogate_nlp.build_surrogate_nlp",
            side_effect=AssertionError("a valid completed result was recomputed"),
        ) as build:
            resumed = solve_surrogate_exact_qp_local(
                assets,
                case,
                settings=settings,
                completed_result=checkpoint,
            )
        build.assert_not_called()
        self.assertIs(resumed.starts[0], checkpoint)

        stale = replace(checkpoint, resume_contract="stale")
        with self.assertRaisesRegex(ValueError, "stale exact-QP resume contract"):
            solve_surrogate_exact_qp_local(
                assets,
                case,
                settings=settings,
                completed_result=stale,
            )

    def test_primary_exact_qp_protocol_reuses_one_expression_problem(self) -> None:
        assets, case = _toy_assets(weak_active_set=True)
        settings = SurrogateSolverSettings(outer_maximum_iterations=10)
        problem = build_surrogate_nlp(
            assets,
            1.0e-8,
            settings=settings,
            compile_solver=False,
            name="active_set_reusable_expression_graph",
        )
        with patch(
            "closed_loop.v3_surrogate_nlp.build_surrogate_nlp",
            side_effect=AssertionError("the reusable expression graph was rebuilt"),
        ) as build:
            result = solve_surrogate_exact_qp_local(
                assets,
                case,
                settings=settings,
                problem=problem,
            )
        build.assert_not_called()
        self.assertEqual(result.starts[0].stages, ())

        with self.assertRaisesRegex(ValueError, "final gap value"):
            solve_surrogate_exact_qp_local(
                assets,
                case,
                settings=settings,
                problem=replace(problem, tau=1.0e-6),
            )

    def test_exact_sensitivity_matches_an_external_central_difference(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_nlp(
            assets, 1.0e-8, compile_solver=False, name="active_set_gradient"
        )
        refiner = ExactQPActiveSetRefiner(
            assets, case, problem=problem, name="active_set_gradient"
        )
        normalized = np.full(7, 0.5)
        trial = refiner.evaluate(normalized)

        self.assertTrue(trial.projection.accepted)
        self.assertTrue(trial.lower_active_set.stable)
        self.assertEqual(trial.lower_active_set.active_indices, ())
        self.assertEqual(len(trial.lower_active_set.perturbations), 14)
        self.assertEqual(refiner.cold_qp_resolutions, 15)
        self.assertLessEqual(trial.sensitivity.solve_residual, 1.0e-8)

        step = 2.0e-6
        finite_difference = np.empty(7)
        parameter = case.parameter_vector(assets)
        for coordinate in range(7):
            plus = normalized.copy()
            minus = normalized.copy()
            plus[coordinate] += step
            minus[coordinate] -= step
            plus_projection = cold_reproject(assets, case, plus)
            minus_projection = cold_reproject(assets, case, minus)
            plus_value = float(
                problem.upper_from_state_function(
                    plus, parameter, plus_projection.state
                )[0]
            )
            minus_value = float(
                problem.upper_from_state_function(
                    minus, parameter, minus_projection.state
                )[0]
            )
            finite_difference[coordinate] = (plus_value - minus_value) / (2.0 * step)
        np.testing.assert_allclose(
            trial.objective_gradient_normalized,
            finite_difference,
            rtol=2.0e-6,
            atol=2.0e-8,
        )

    def test_weakly_active_lower_constraints_fail_explicitly(self) -> None:
        assets, case = _toy_assets(weak_active_set=True)
        problem = build_surrogate_nlp(
            assets, 1.0e-8, compile_solver=False, name="active_set_weak"
        )
        refiner = ExactQPActiveSetRefiner(
            assets, case, problem=problem, name="active_set_weak"
        )
        with self.assertRaises(ActiveSetDerivativeError) as captured:
            refiner.evaluate(np.full(7, 0.5))
        audit = captured.exception.audit
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertFalse(audit.stable)
        self.assertFalse(audit.strict_complementarity_passed)
        self.assertEqual(audit.minimum_active_multiplier, 0.0)
        self.assertIn("active lower multiplier", audit.reason)

        result = refiner.refine(np.full(7, 0.5))
        self.assertEqual(result.status, "active_set_derivative_unavailable")
        self.assertIsNotNone(result.derivative_error)
        self.assertIsNone(result.final)

    def test_refinement_has_independent_final_qp_and_upper_kkt_audit(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_nlp(
            assets, 1.0e-8, compile_solver=False, name="active_set_refinement"
        )
        refiner = ExactQPActiveSetRefiner(
            assets,
            case,
            problem=problem,
            settings=ActiveSetRefinementSettings(maximum_iterations=40),
            name="active_set_refinement",
        )
        result = refiner.refine(np.full(7, 0.5))

        self.assertIsNotNone(result.final)
        self.assertIsNotNone(result.upper_kkt)
        assert result.final is not None
        assert result.upper_kkt is not None
        self.assertTrue(result.final.independent_final_replay)
        self.assertTrue(result.final.projection.accepted)
        self.assertTrue(result.upper_kkt.feasible)
        self.assertTrue(result.state_reproduction_passed)
        self.assertLessEqual(result.state_reproduction_residual, 1.0e-8)
        self.assertLess(result.final.objective, result.initial.objective)  # type: ignore[union-attr]
        self.assertGreater(result.cold_qp_resolutions, result.distinct_trials)
        self.assertIn(
            result.status,
            {"validated_stationary", "validated_feasible_stationarity_unresolved"},
        )
        json.dumps(result.as_dict())

    def test_surrogate_outer_path_serializes_active_set_and_upper_kkt(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_nlp(
            assets, 1.0e-8, compile_solver=False, name="active_set_integration"
        )
        final, refinement = _outer_refine(
            problem,
            case,
            np.full(7, 0.5),
            SurrogateSolverSettings(outer_maximum_iterations=40),
        )

        self.assertIsNotNone(final)
        assert final is not None
        self.assertTrue(final.feasibility.feasible)
        self.assertIsNotNone(final.lower_active_set)
        self.assertIsNotNone(final.upper_kkt)
        self.assertEqual(
            final.stationarity.upper_stationarity_residual,
            final.upper_kkt["stationarity_residual"],
        )
        self.assertGreater(refinement.cold_qp_resolutions, refinement.evaluations)
        self.assertEqual(refinement.lower_active_set, final.lower_active_set)
        self.assertEqual(refinement.upper_kkt, final.upper_kkt)
        if final.status == "validated_feasible_stationarity_unresolved":
            self.assertFalse(final.stationarity.resolved)
        serialized = final.as_dict()
        self.assertIn("lower_active_set", serialized)
        self.assertIn("upper_kkt", serialized)
        json.dumps(serialized)

    def test_multistart_prefers_stationary_candidate_over_better_unresolved_one(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_nlp(
            assets, 1.0e-8, compile_solver=False, name="active_set_selection"
        )
        base, refinement = _outer_refine(
            problem,
            case,
            np.full(7, 0.5),
            SurrogateSolverSettings(outer_maximum_iterations=40),
        )
        assert base is not None
        unresolved = replace(
            base,
            objective=base.objective - 1.0,
            stationarity=StationarityRecord(
                classification="stationarity_unresolved",
                resolved=True,
                stationary=False,
                lower_qp_kkt_passed=True,
                upper_stationarity_residual=2.0e-6,
                reason="unit unresolved candidate",
            ),
            status="validated_feasible_stationarity_unresolved",
        )
        stationary = replace(
            base,
            normalized_controls=np.full(7, 0.6),
            stationarity=StationarityRecord(
                classification="first_order_kkt_stationary_feasible",
                resolved=True,
                stationary=True,
                lower_qp_kkt_passed=True,
                upper_stationarity_residual=5.0e-7,
                reason="unit stationary candidate",
            ),
            status="validated_stationary",
        )
        starts = (
            SurrogateStartResult(
                0,
                np.full(7, 0.5),
                (),
                refinement,
                unresolved,
                unresolved.status,
                resume_contract=surrogate_start_resume_contract(
                    assets, case, SurrogateSolverSettings()
                ),
            ),
            SurrogateStartResult(
                1,
                np.full(7, 0.6),
                (),
                OuterRefinementRecord(
                    True, True, "unit", 1, 1, 0.1, stationary.objective, stationary.objective
                ),
                stationary,
                stationary.status,
                resume_contract=surrogate_start_resume_contract(
                    assets, case, SurrogateSolverSettings()
                ),
            ),
        )
        with (
            patch(
                "closed_loop.v3_surrogate_nlp.build_surrogate_nlp",
                return_value=problem,
            ),
            patch(
                "closed_loop.v3_surrogate_nlp.solve_surrogate_start",
                side_effect=starts,
            ),
        ):
            result = solve_surrogate_multistart(
                assets,
                case,
                starts=np.vstack((np.full(7, 0.5), np.full(7, 0.6))),
                allow_reduced_starts=True,
            )
        self.assertIs(result.selected, starts[1])
        self.assertEqual(result.status, "selected_stationary")

    def test_completed_start_map_skips_only_validated_indices(self) -> None:
        assets, case = _toy_assets()
        settings = SurrogateSolverSettings()
        normalized = ordered_normalized_starts()
        contract = surrogate_start_resume_contract(assets, case, settings)
        empty_refinement = OuterRefinementRecord(
            False, False, "not_attempted", 0, 0, 0.0, None, None
        )

        def checkpoint(index: int) -> SurrogateStartResult:
            return SurrogateStartResult(
                start_index=index,
                initial_normalized_controls=normalized[index].copy(),
                stages=(),
                outer_refinement=empty_refinement,
                final=None,
                status="initial_projection_failed",
                resume_contract=contract,
            )

        resumed = {0: checkpoint(0), 3: checkpoint(3)}
        newly_reported: list[int] = []

        def solve_missing(
            problems: object,
            supplied_case: SurrogateCase,
            supplied_start: np.ndarray,
            *,
            start_index: int,
            settings: SurrogateSolverSettings,
        ) -> SurrogateStartResult:
            self.assertIs(supplied_case, case)
            np.testing.assert_array_equal(supplied_start, normalized[start_index])
            return checkpoint(start_index)

        with (
            patch(
                "closed_loop.v3_surrogate_nlp.build_surrogate_nlp",
                return_value=object(),
            ) as build,
            patch(
                "closed_loop.v3_surrogate_nlp.solve_surrogate_start",
                side_effect=solve_missing,
            ) as solve,
        ):
            result = solve_surrogate_multistart(
                assets,
                case,
                settings=settings,
                completed_starts=resumed,
                progress_callback=lambda item: newly_reported.append(item.start_index),
            )
        self.assertEqual([item.start_index for item in result.starts], list(range(9)))
        self.assertEqual(solve.call_count, 7)
        self.assertEqual(build.call_count, 7)
        self.assertEqual(newly_reported, [1, 2, 4, 5, 6, 7, 8])

        stale = replace(resumed[0], resume_contract="stale")
        with self.assertRaisesRegex(ValueError, "stale case/settings"):
            solve_surrogate_multistart(
                assets,
                case,
                settings=settings,
                completed_starts={0: stale},
            )
        wrong_control = replace(
            resumed[0], initial_normalized_controls=normalized[0] + 1.0e-12
        )
        with self.assertRaisesRegex(ValueError, "declared normalized control"):
            solve_surrogate_multistart(
                assets,
                case,
                settings=settings,
                completed_starts={0: wrong_control},
            )

    def test_checkpoint_records_round_trip_and_restore_null_as_nan(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_nlp(
            assets, 1.0e-8, compile_solver=False, name="active_set_roundtrip"
        )
        final, refinement = _outer_refine(
            problem,
            case,
            np.full(7, 0.5),
            SurrogateSolverSettings(outer_maximum_iterations=40),
        )
        assert final is not None
        stage = ContinuationStageRecord(
            tau=1.0e-2,
            status="unit",
            solver_success=False,
            iterations=2,
            elapsed_seconds=0.1,
            feasible=False,
            equality_residual=np.inf,
            inequality_residual=0.0,
            bound_residual=0.0,
            normalized_gap=np.inf,
            primal=np.arange(problem.variable_count, dtype=float),
            error="unit failure",
        )
        original = SurrogateStartResult(
            start_index=0,
            initial_normalized_controls=np.full(7, 0.5),
            stages=(stage,),
            outer_refinement=refinement,
            final=final,
            status=final.status,
            resume_contract="unit-contract",
        )
        payload = original.as_dict()
        payload["stages"][0]["equality_residual"] = None
        payload["stages"][0]["normalized_gap"] = None
        payload["final"]["projection"]["state"][0] = None
        restored = SurrogateStartResult.from_dict(payload)

        self.assertTrue(np.isnan(restored.stages[0].equality_residual))
        self.assertTrue(np.isnan(restored.stages[0].normalized_gap))
        self.assertEqual(restored.resume_contract, "unit-contract")
        self.assertTrue(np.isnan(restored.final.projection.state[0]))  # type: ignore[union-attr]
        np.testing.assert_array_equal(restored.final.projected, final.projected)  # type: ignore[union-attr]
        np.testing.assert_array_equal(
            restored.final.projection.inequality_multipliers,  # type: ignore[union-attr]
            final.projection.inequality_multipliers,
        )
        json.dumps(restored.as_dict())

    def test_refinement_prefers_stationary_cached_trial_before_final_replay(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_nlp(
            assets, 1.0e-8, compile_solver=False, name="active_set_inner_selection"
        )
        refiner = ExactQPActiveSetRefiner(
            assets, case, problem=problem, name="active_set_inner_selection"
        )
        stationary_controls = np.full(7, 0.5)
        lower_objective_controls = np.full(7, 0.4)
        stationary_trial = refiner.evaluate(stationary_controls)
        lower_trial = refiner.evaluate(lower_objective_controls)
        self.assertLess(lower_trial.objective, stationary_trial.objective)

        def audit(trial: object) -> UpperKKTAudit:
            stationary = np.array_equal(
                trial.normalized_controls, stationary_controls  # type: ignore[attr-defined]
            )
            return UpperKKTAudit(
                active_indices=(),
                active_names=(),
                multipliers=np.zeros(trial.upper_constraints.size),  # type: ignore[attr-defined]
                primal_residual=0.0,
                dual_feasibility_residual=0.0,
                stationarity_residual=0.0 if stationary else 1.0e-3,
                complementarity_residual=0.0,
                feasible=True,
                stationary=stationary,
                classification=(
                    "first_order_kkt_stationary_feasible"
                    if stationary
                    else "validated_feasible_stationarity_unresolved"
                ),
                reason="unit audit",
            )

        with (
            patch(
                "closed_loop.v3_active_set.minimize",
                return_value=SimpleNamespace(
                    x=lower_objective_controls,
                    success=True,
                    message="unit",
                    nit=1,
                ),
            ),
            patch.object(refiner, "audit_upper_kkt", side_effect=audit),
        ):
            result = refiner.refine(stationary_controls)
        np.testing.assert_array_equal(
            result.final.normalized_controls, stationary_controls  # type: ignore[union-attr]
        )
        self.assertTrue(result.stationary)

    def test_independent_state_reproduction_is_an_acceptance_gate(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_nlp(
            assets, 1.0e-8, compile_solver=False, name="active_set_reproduction"
        )
        refiner = ExactQPActiveSetRefiner(
            assets, case, problem=problem, name="active_set_reproduction"
        )
        original_evaluate = refiner.evaluate

        def shifted_final(
            value: np.ndarray,
            *,
            force_cold: bool = False,
            independent_final_replay: bool = False,
        ) -> object:
            trial = original_evaluate(
                value,
                force_cold=force_cold,
                independent_final_replay=independent_final_replay,
            )
            if force_cold:
                return replace(
                    trial,
                    projected_state=(
                        trial.projected_state
                        + 2.0e-8 * assets.model.response_scale
                    ),
                )
            return trial

        with (
            patch(
                "closed_loop.v3_active_set.minimize",
                return_value=SimpleNamespace(
                    x=np.full(7, 0.5), success=True, message="unit", nit=1
                ),
            ),
            patch.object(refiner, "evaluate", side_effect=shifted_final),
        ):
            result = refiner.refine(np.full(7, 0.5))
        self.assertEqual(result.status, "projection_reproduction_failed")
        self.assertFalse(result.state_reproduction_passed)
        self.assertGreater(result.state_reproduction_residual, 1.0e-8)
        self.assertFalse(result.feasible)

    def test_unexpected_refinement_exception_retains_feasible_incumbent(self) -> None:
        assets, case = _toy_assets()
        base = build_surrogate_nlp(
            assets, 1.0e-8, compile_solver=False, name="active_set_exception"
        )
        problems = tuple(replace(base, tau=tau) for tau in surrogate_module.GAP_CONTINUATION)
        normalized = np.full(7, 0.5)
        incumbent = audit_exact_candidate(base, case, normalized)
        self.assertTrue(incumbent.feasibility.feasible)
        primal = np.zeros(base.variable_count)
        primal[base.theta_slice] = normalized
        stage = ContinuationStageRecord(
            tau=1.0e-8,
            status="unit",
            solver_success=True,
            iterations=1,
            elapsed_seconds=0.1,
            feasible=True,
            equality_residual=0.0,
            inequality_residual=0.0,
            bound_residual=0.0,
            normalized_gap=0.0,
            primal=primal,
        )
        solved_stage = SimpleNamespace(
            stage=stage,
            bound_multipliers=np.zeros(base.variable_count),
            constraint_multipliers=np.zeros(base.constraint_lower_bounds.size),
        )
        with (
            patch(
                "closed_loop.v3_surrogate_nlp.initial_primal_from_projection",
                return_value=(primal, incumbent.projection),
            ),
            patch(
                "closed_loop.v3_surrogate_nlp._solve_continuation_stage",
                return_value=solved_stage,
            ),
            patch(
                "closed_loop.v3_surrogate_nlp.audit_exact_candidate",
                return_value=incumbent,
            ),
            patch(
                "closed_loop.v3_surrogate_nlp._outer_refine",
                side_effect=RuntimeError("unit linear algebra failure"),
            ),
        ):
            result = surrogate_module.solve_surrogate_start(
                problems,
                case,
                normalized,
                start_index=0,
            )
        self.assertTrue(result.feasible)
        self.assertFalse(result.stationary)
        self.assertEqual(result.status, "validated_feasible_stationarity_unresolved")
        self.assertEqual(result.outer_refinement.status, "unexpected_refinement_exception")
        self.assertIn("unit linear algebra failure", result.error)

    def test_degenerate_endpoint_passes_complete_two_scale_poll(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_nlp(
            assets, 1.0e-8, compile_solver=False, name="certificate_flat"
        )
        controls = np.full(7, 0.5)
        initial = audit_exact_candidate(problem, case, controls)

        def exact_candidate(
            _problem: object,
            _case: object,
            normalized: np.ndarray,
            **_kwargs: object,
        ):
            value = np.asarray(normalized, dtype=float)
            feasible = bool(np.all(value[1:] == controls[1:]))
            return replace(
                initial,
                normalized_controls=value.copy(),
                theta=assets.theta_lower + assets.theta_span * value,
                objective=float(np.sum((value - controls) ** 2)),
                feasibility=replace(initial.feasibility, feasible=feasible),
            )

        class DegenerateRefiner:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.cold_qp_resolutions = 0

            def evaluate(self, *_args: object, **_kwargs: object) -> object:
                raise ActiveSetDerivativeError("unit degenerate active set")

        with (
            patch(
                "closed_loop.v3_surrogate_nlp.audit_exact_candidate",
                side_effect=exact_candidate,
            ),
            patch(
                "closed_loop.v3_active_set.ExactQPActiveSetRefiner",
                DegenerateRefiner,
            ),
        ):
            result = certify_surrogate_local_convergence(
                assets,
                case,
                initial,
                problem=problem,
            )

        certificate = result.certificate
        self.assertEqual(certificate.protocol, LOCAL_CONVERGENCE_PROTOCOL)
        self.assertEqual(
            certificate.protocol,
            "exact_qp_two_scale_accelerated_feasible_poll_v3",
        )
        self.assertEqual(SurrogateCertificationSettings().maximum_evaluations, 10_000)
        self.assertEqual(SurrogateCertificationSettings().acceleration_growth_factor, 2.0)
        self.assertEqual(SurrogateCertificationSettings().maximum_acceleration_probes, 16)
        self.assertEqual(
            certificate.classification, "finite_resolution_feasible_poll"
        )
        self.assertTrue(certificate.locally_converged)
        self.assertFalse(certificate.first_order_certified)
        self.assertFalse(certificate.stationarity_resolved)
        self.assertEqual(certificate.accepted_improvements, 0)
        self.assertEqual(len(certificate.poll_levels), 2)
        self.assertEqual(
            [level["direction_count"] for level in certificate.poll_levels],
            [14, 106],
        )
        self.assertTrue(all(level["passed"] for level in certificate.poll_levels))
        self.assertTrue(
            all(level["feasible_direction_rank"] == 1 for level in certificate.poll_levels)
        )
        self.assertTrue(
            all(level["required_direction_rank"] is None for level in certificate.poll_levels)
        )
        self.assertTrue(
            all(level["rank_is_diagnostic_only"] for level in certificate.poll_levels)
        )
        self.assertTrue(
            all(
                level["feasible_direction_coverage_passed"]
                for level in certificate.poll_levels
            )
        )
        self.assertTrue(
            all(level["acceleration_evaluation_requests"] == 0 for level in certificate.poll_levels)
        )
        self.assertTrue(
            all(level["acceleration_accepted_improvements"] == 0 for level in certificate.poll_levels)
        )
        self.assertIsNotNone(result.candidate)
        assert result.candidate is not None
        self.assertEqual(
            result.candidate.status,
            "validated_feasible_poll_converged_stationarity_unresolved",
        )
        self.assertFalse(result.candidate.stationarity.resolved)

    def test_poll_accepts_improvements_and_repeats_radius_before_passing(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_nlp(
            assets, 1.0e-8, compile_solver=False, name="certificate_repeat"
        )
        controls = np.full(7, 0.5)
        optimum = controls.copy()
        optimum[0] = 0.502
        initial = audit_exact_candidate(problem, case, controls)

        def exact_candidate(
            _problem: object,
            _case: object,
            normalized: np.ndarray,
            **_kwargs: object,
        ):
            value = np.asarray(normalized, dtype=float)
            return replace(
                initial,
                normalized_controls=value.copy(),
                theta=assets.theta_lower + assets.theta_span * value,
                objective=float(np.sum((value - optimum) ** 2)),
            )

        class DegenerateRefiner:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.cold_qp_resolutions = 0

            def evaluate(self, *_args: object, **_kwargs: object) -> object:
                raise ActiveSetDerivativeError("unit nonsmooth endpoint")

        settings = SurrogateCertificationSettings(
            poll_radii=(1.0e-3,),
            maximum_evaluations=1_000,
        )
        with (
            patch(
                "closed_loop.v3_surrogate_nlp.audit_exact_candidate",
                side_effect=exact_candidate,
            ),
            patch(
                "closed_loop.v3_active_set.ExactQPActiveSetRefiner",
                DegenerateRefiner,
            ),
        ):
            result = certify_surrogate_local_convergence(
                assets,
                case,
                initial,
                settings=settings,
                problem=problem,
            )

        self.assertEqual(
            result.certificate.classification, "finite_resolution_feasible_poll"
        )
        self.assertEqual(result.certificate.accepted_improvements, 2)
        level = result.certificate.poll_levels[0]
        self.assertEqual(level["accepted_improvements"], 2)
        self.assertEqual(level["poll_accepted_improvements"], 1)
        self.assertEqual(level["acceleration_accepted_improvements"], 1)
        self.assertEqual(level["acceleration_evaluation_requests"], 2)
        self.assertEqual(level["acceleration_unique_evaluations"], 2)
        self.assertEqual(level["acceleration_maximum_accepted_multiplier"], 2.0)
        self.assertEqual(level["acceleration_stops"], {"no_sufficient_descent": 1})
        self.assertEqual(level["direction_count"], 106)
        # Acceleration only locates a better center. Certification still comes
        # from the fresh, complete no-descent sweep recorded at that center.
        self.assertTrue(level["complete_no_descent_poll"])
        np.testing.assert_allclose(
            result.certificate.final_normalized_controls, optimum, atol=1.0e-14
        )

    def test_fine_scale_improvement_revalidates_the_coarse_scale(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_nlp(
            assets, 1.0e-8, compile_solver=False, name="certificate_revalidate"
        )
        controls = np.full(7, 0.5)
        optimum = controls.copy()
        optimum[0] += 1.0e-4
        initial = audit_exact_candidate(problem, case, controls)

        def exact_candidate(
            _problem: object,
            _case: object,
            normalized: np.ndarray,
            **_kwargs: object,
        ):
            value = np.asarray(normalized, dtype=float)
            return replace(
                initial,
                normalized_controls=value.copy(),
                theta=assets.theta_lower + assets.theta_span * value,
                objective=float(np.sum((value - optimum) ** 2)),
            )

        class DegenerateRefiner:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.cold_qp_resolutions = 0

            def evaluate(self, *_args: object, **_kwargs: object) -> object:
                raise ActiveSetDerivativeError("unit nonsmooth endpoint")

        settings = SurrogateCertificationSettings(
            absolute_decrease_tolerance=1.0e-12,
            relative_decrease_tolerance=1.0e-12,
        )
        with (
            patch(
                "closed_loop.v3_surrogate_nlp.audit_exact_candidate",
                side_effect=exact_candidate,
            ),
            patch(
                "closed_loop.v3_active_set.ExactQPActiveSetRefiner",
                DegenerateRefiner,
            ),
        ):
            result = certify_surrogate_local_convergence(
                assets,
                case,
                initial,
                settings=settings,
                problem=problem,
            )

        self.assertTrue(result.certificate.locally_converged)
        self.assertEqual(result.certificate.accepted_improvements, 1)
        levels = result.certificate.poll_levels
        self.assertEqual(
            [(level["validation_round"], level["radius"]) for level in levels],
            [
                (0, 1.0e-3),
                (0, 1.0e-4),
                (1, 1.0e-3),
                (1, 1.0e-4),
            ],
        )
        self.assertEqual(
            [level["accepted_improvements"] for level in levels], [0, 1, 0, 0]
        )
        self.assertTrue(all(level["passed"] for level in levels))
        np.testing.assert_allclose(
            result.certificate.final_normalized_controls, optimum, atol=1.0e-14
        )

    def test_cached_revalidation_completes_at_the_exact_evaluation_cap(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_nlp(
            assets, 1.0e-8, compile_solver=False, name="certificate_cached_cap"
        )
        controls = np.full(7, 0.5)
        optimum = controls.copy()
        optimum[0] += 1.0e-4
        initial = audit_exact_candidate(problem, case, controls)

        def exact_candidate(
            _problem: object,
            _case: object,
            normalized: np.ndarray,
            **_kwargs: object,
        ):
            value = np.asarray(normalized, dtype=float)
            return replace(
                initial,
                normalized_controls=value.copy(),
                theta=assets.theta_lower + assets.theta_span * value,
                objective=float(np.sum((value - optimum) ** 2)),
            )

        class DegenerateRefiner:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.cold_qp_resolutions = 0

            def evaluate(self, *_args: object, **_kwargs: object) -> object:
                raise ActiveSetDerivativeError("unit nonsmooth endpoint")

        axis = np.zeros(7)
        axis[0] = 1.0
        settings = SurrogateCertificationSettings(
            poll_radii=(2.0e-4, 1.0e-4),
            absolute_decrease_tolerance=1.0e-12,
            relative_decrease_tolerance=1.0e-12,
            maximum_evaluations=6,
        )
        with (
            patch(
                "closed_loop.v3_surrogate_nlp.audit_exact_candidate",
                side_effect=exact_candidate,
            ),
            patch(
                "closed_loop.v3_surrogate_nlp._local_poll_directions",
                return_value=np.vstack((axis, -axis)),
            ),
            patch(
                "closed_loop.v3_active_set.ExactQPActiveSetRefiner",
                DegenerateRefiner,
            ),
        ):
            result = certify_surrogate_local_convergence(
                assets,
                case,
                initial,
                settings=settings,
                problem=problem,
            )

        self.assertEqual(result.certificate.evaluations, settings.maximum_evaluations)
        self.assertEqual(
            result.certificate.classification, "finite_resolution_feasible_poll"
        )
        self.assertTrue(result.certificate.locally_converged)
        self.assertNotIn("budget", result.certificate.termination_reason)
        self.assertEqual(
            [level["validation_round"] for level in result.certificate.poll_levels],
            [0, 0, 1, 1],
        )
        self.assertTrue(all(level["passed"] for level in result.certificate.poll_levels))

    def test_poll_budget_exhaustion_is_explicit_and_not_certified(self) -> None:
        assets, case = _toy_assets()
        problem = build_surrogate_nlp(
            assets, 1.0e-8, compile_solver=False, name="certificate_budget"
        )
        controls = np.full(7, 0.5)
        initial = audit_exact_candidate(problem, case, controls)

        def exact_candidate(
            _problem: object,
            _case: object,
            normalized: np.ndarray,
            **_kwargs: object,
        ):
            value = np.asarray(normalized, dtype=float)
            return replace(
                initial,
                normalized_controls=value.copy(),
                theta=assets.theta_lower + assets.theta_span * value,
                objective=1.0,
            )

        class DegenerateRefiner:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.cold_qp_resolutions = 0

            def evaluate(self, *_args: object, **_kwargs: object) -> object:
                raise ActiveSetDerivativeError("unit degenerate active set")

        with (
            patch(
                "closed_loop.v3_surrogate_nlp.audit_exact_candidate",
                side_effect=exact_candidate,
            ),
            patch(
                "closed_loop.v3_active_set.ExactQPActiveSetRefiner",
                DegenerateRefiner,
            ),
        ):
            result = certify_surrogate_local_convergence(
                assets,
                case,
                initial,
                settings=SurrogateCertificationSettings(maximum_evaluations=2),
                problem=problem,
            )

        self.assertEqual(result.certificate.classification, "poll_budget_limited")
        self.assertFalse(result.certificate.locally_converged)
        self.assertFalse(result.certificate.first_order_certified)
        self.assertEqual(result.certificate.evaluations, 2)
        self.assertIn("budget", result.certificate.termination_reason)
        self.assertIsNotNone(result.candidate)
        assert result.candidate is not None
        self.assertTrue(result.candidate.feasibility.feasible)
        self.assertIn("poll_budget_limited", result.candidate.status)


if __name__ == "__main__":
    unittest.main()
