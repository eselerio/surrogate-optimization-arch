from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from closed_loop.projection import (
    LeastSquaresDiagnostics,
    NetworkLayout,
    NetworkRowScales,
    QuadraticFeatureMap,
    QuadraticSurrogate,
    build_network_operators,
    no_conversion_feasible_state,
)
from closed_loop.v3_shared_unit import (
    ROOT_MIXER_AGREEMENT_TOLERANCE,
    SharedUnitAssets,
    SharedUnitCase,
    SharedUnitClosureDiagnostics,
    SharedUnitClosureResult,
    SharedUnitEvaluation,
    SharedUnitFitResult,
    SharedUnitLeverageContract,
    SharedUnitModels,
    SharedUnitOptimizationSettings,
    SharedUnitOptimizationResult,
    SharedUnitRidgeScore,
    SharedUnitRootAttempt,
    SharedUnitTrustLimits,
    calibrate_shared_unit_trust,
    cross_validate_shared_unit_models,
    evaluate_shared_unit,
    evaluate_shared_unit_holdout_batches,
    extract_shared_unit_training,
    fit_shared_unit_leverage,
    optimize_shared_unit_case,
    project_shared_unit_raw,
    quadratic_prediction_jacobian,
    solve_shared_unit_closure,
)
from closed_loop.v3_surrogate_nlp import EngineeringLimits


def small_layout() -> NetworkLayout:
    return NetworkLayout(
        stage_count=5,
        component_count=2,
        layer_count=3,
        soluble_indices=(0,),
        particulate_indices=(1,),
    )


def diagnostics(feature_count: int, response_count: int) -> LeastSquaresDiagnostics:
    return LeastSquaresDiagnostics(
        sample_count=100,
        feature_count=feature_count,
        response_count=response_count,
        rank_tolerance=1.0e-12,
        smallest_singular_value=1.0,
        largest_singular_value=2.0,
        condition_number=2.0,
        optimality_residual=0.0,
        coefficient_agreement=0.0,
        acceptance_threshold=1.0e-12,
        augmented_condition_number=2.0,
        condition_times_machine_epsilon=2.0 * np.finfo(float).eps,
        effective_degrees_of_freedom=float(feature_count),
    )


def linear_model(
    decision_matrix: np.ndarray,
    state_matrix: np.ndarray,
    intercept: np.ndarray,
    *,
    seed: int,
) -> QuadraticSurrogate:
    """Encode a physical linear map exactly in the standardized basis."""

    rng = np.random.default_rng(seed)
    decision_count = decision_matrix.shape[1]
    state_count = state_matrix.shape[1]
    response_count = decision_matrix.shape[0]
    fit_decisions = rng.uniform(-2.0, 2.0, size=(80, decision_count))
    fit_states = rng.uniform(0.1, 4.0, size=(80, state_count))
    feature = QuadraticFeatureMap.fit(fit_decisions, fit_states)
    combined = np.column_stack((decision_matrix, state_matrix))
    input_center = np.concatenate((feature.decision_center, feature.influent_center))
    input_scale = np.concatenate((feature.decision_scale, feature.influent_scale))
    linear_term_center = feature.term_center[: decision_count + state_count]
    linear_term_scale = feature.term_scale[: decision_count + state_count]
    coefficients = np.zeros((response_count, feature.feature_count))
    coefficients[:, 0] = intercept + combined @ (
        input_center + input_scale * linear_term_center
    )
    coefficients[:, 1 : 1 + decision_count + state_count] = (
        combined * (input_scale * linear_term_scale)[None, :]
    )
    return QuadraticSurrogate(
        feature_map=feature,
        response_center=np.zeros(response_count),
        response_scale=np.ones(response_count),
        coefficients=coefficients,
        diagnostics=diagnostics(feature.feature_count, response_count),
        ridge_penalty=0.1,
    )


def identity_models() -> SharedUnitModels:
    reactor = linear_model(
        np.zeros((2, 2)), np.eye(2), np.zeros(2), seed=10
    )
    clarifier_state = np.vstack(
        (
            0.8 * np.eye(2),
            0.5 * np.eye(2),
            np.asarray([[20.0, 100.0]]),
        )
    )
    clarifier = linear_model(
        np.zeros((5, 2)), clarifier_state, np.zeros(5), seed=11
    )
    return SharedUnitModels(reactor=reactor, clarifier=clarifier)


def controls(rows: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    lower = np.asarray([6.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.001])
    upper = np.asarray([36.0, 1.0, 1.0, 1.0, 4.0, 1.25, 0.05])
    return lower + rng.uniform(0.1, 0.9, size=(rows, 7)) * (upper - lower)


def optimization_fixture() -> tuple[
    SharedUnitAssets,
    SharedUnitCase,
    SharedUnitLeverageContract,
]:
    layout = small_layout()
    models = identity_models()
    rng = np.random.default_rng(51)
    theta_rows = controls(20)
    target_rows = rng.uniform(0.5, 4.0, size=(20, layout.state_size))
    training = extract_shared_unit_training(theta_rows, target_rows, layout=layout)
    leverage = fit_shared_unit_leverage(models, training)
    assets = SharedUnitAssets(
        models=models,
        layout=layout,
        common_response_scale=np.r_[np.ones(16), 100.0],
        row_scales=NetworkRowScales(equality=np.ones(10), inequality=np.ones(3)),
        invariant_operator=np.asarray([[1.0, 0.0]]),
        tss_weights=np.asarray([0.0, 1.0]),
        leverage=leverage,
        trust_limits=SharedUnitTrustLimits(
            correction_rms=1.0e6,
            reactor_leverage=1.0e6,
            clarifier_leverage=1.0e6,
        ),
        quality_operator=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        quality_scale=np.ones(2),
        engineering=EngineeringLimits(
            srt_lower_d=1.0e-6,
            srt_upper_d=1.0e6,
            external_loss_min_g_m3=1.0e-8,
            slr_upper_kg_m2_d=1.0e8,
            underflow_tss_upper_g_m3=1.0e8,
            feed_tss_min_g_m3=1.0e-8,
        ),
    )
    case = SharedUnitCase(
        influent=np.asarray([2.0, 3.0]),
        case_id="synthetic",
        quality_weights=np.asarray([0.5, 0.5]),
    )
    return assets, case, leverage


class SharedUnitTests(unittest.TestCase):
    def test_parallel_calibration_matches_serial_and_resumes_batches(self) -> None:
        layout = small_layout()
        models = identity_models()
        theta = np.tile(
            0.5
            * (
                np.asarray([6.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.001])
                + np.asarray([36.0, 1.0, 1.0, 1.0, 4.0, 1.25, 0.05])
            ),
            (4, 1),
        )
        feed = np.tile(np.asarray([2.0, 3.0]), (4, 1))
        truth = np.tile(
            solve_shared_unit_closure(
                models,
                theta[0],
                feed[0],
                np.r_[np.ones(16), 100.0],
                layout=layout,
            ).raw,
            (4, 1),
        )
        training = extract_shared_unit_training(theta, truth, layout=layout)
        membership = np.asarray([1, 2, 1, 2])
        fit = SharedUnitFitResult(
            models=models,
            fold_models=(models, models),
            scores=tuple(),
            plant_fold_membership=membership,
            reactor_out_of_fold_raw=np.zeros((20, 2)),
            clarifier_out_of_fold_raw=np.zeros((4, 5)),
            elapsed_seconds=0.0,
        )
        row_scales = NetworkRowScales(
            equality=np.ones(10), inequality=np.ones(3)
        )
        arguments = dict(
            layout=layout,
            invariant_operator=np.asarray([[1.0, 0.0]]),
            tss_weights=np.asarray([0.0, 1.0]),
            batch_size=2,
        )
        serial = calibrate_shared_unit_trust(
            fit,
            training,
            theta,
            feed,
            truth,
            np.r_[np.ones(16), 100.0],
            row_scales,
            parallel_workers=1,
            **arguments,
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            parallel = calibrate_shared_unit_trust(
                fit,
                training,
                theta,
                feed,
                truth,
                np.r_[np.ones(16), 100.0],
                row_scales,
                parallel_workers=2,
                checkpoint_directory=checkpoint,
                checkpoint_contract="shared-calibration-test",
                **arguments,
            )
            with patch(
                "closed_loop.v3_shared_unit._shared_calibration_batch",
                side_effect=AssertionError("completed batches were recomputed"),
            ):
                resumed = calibrate_shared_unit_trust(
                    fit,
                    training,
                    theta,
                    feed,
                    truth,
                    np.r_[np.ones(16), 100.0],
                    row_scales,
                    parallel_workers=1,
                    checkpoint_directory=checkpoint,
                    checkpoint_contract="shared-calibration-test",
                    **arguments,
                )
        for observed in (parallel, resumed):
            np.testing.assert_allclose(
                observed.out_of_fold_raw, serial.out_of_fold_raw, rtol=0.0, atol=0.0
            )
            np.testing.assert_allclose(
                observed.out_of_fold_projected,
                serial.out_of_fold_projected,
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_array_equal(
                observed.closure_accepted, serial.closure_accepted
            )
            np.testing.assert_array_equal(
                observed.projection_accepted, serial.projection_accepted
            )
            np.testing.assert_allclose(
                observed.development_values,
                serial.development_values,
                rtol=0.0,
                atol=1.0e-14,
            )
            self.assertEqual(
                [item.as_dict() for item in observed.closure_diagnostics],
                [item.as_dict() for item in serial.closure_diagnostics],
            )

    def test_parallel_holdout_core_matches_serial_and_reuses_checkpoints(self) -> None:
        assets, case, _ = optimization_fixture()
        theta = np.tile(
            assets.theta_lower + 0.5 * assets.theta_span,
            (3, 1),
        )
        feed = np.tile(case.influent, (3, 1))
        evaluation = evaluate_shared_unit(assets, case, np.full(7, 0.5))
        self.assertTrue(evaluation.available, evaluation.reason)
        truth = np.tile(evaluation.projected, (3, 1))
        serial = evaluate_shared_unit_holdout_batches(
            assets,
            theta,
            feed,
            truth,
            parallel_workers=1,
            batch_size=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            parallel = evaluate_shared_unit_holdout_batches(
                assets,
                theta,
                feed,
                truth,
                parallel_workers=2,
                batch_size=2,
                checkpoint_directory=checkpoint,
                checkpoint_contract="shared-holdout-test",
            )
            with patch(
                "closed_loop.v3_shared_unit._shared_holdout_batch",
                side_effect=AssertionError("completed batches were recomputed"),
            ):
                resumed = evaluate_shared_unit_holdout_batches(
                    assets,
                    theta,
                    feed,
                    truth,
                    parallel_workers=1,
                    batch_size=2,
                    checkpoint_directory=checkpoint,
                    checkpoint_contract="shared-holdout-test",
                )

        def combined(batches, name):
            return np.concatenate([batch[name] for batch in batches], axis=0)

        for observed in (parallel, resumed):
            for name in (
                "raw",
                "projected",
                "projected_targets",
                "available",
                "target_accepted",
            ):
                np.testing.assert_allclose(
                    combined(observed, name),
                    combined(serial, name),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            self.assertEqual(
                [
                    json.loads(str(item))["case_id"]
                    for batch in observed for item in batch["evaluation_json"]
                ],
                ["test_000000", "test_000001", "test_000002"],
            )

    def test_extracts_plant_major_shared_rows_with_exact_case_inputs(self) -> None:
        layout = small_layout()
        theta = controls(3)
        targets = np.arange(3 * layout.state_size, dtype=float).reshape(
            3, layout.state_size
        ) + 1.0
        training = extract_shared_unit_training(theta, targets, layout=layout)

        self.assertEqual(training.reactor_decisions.shape, (15, 2))
        self.assertEqual(training.reactor_upstream.shape, (15, 2))
        self.assertEqual(training.reactor_targets.shape, (15, 2))
        np.testing.assert_array_equal(training.reactor_plant_index, np.repeat(range(3), 5))
        np.testing.assert_array_equal(training.reactor_stage_index, np.tile(range(5), 3))
        expected_d = 120.0 * (1.0 + theta[:, 4] + theta[:, 5]) / theta[:, 0]
        np.testing.assert_allclose(training.reactor_decisions[:, 0], np.repeat(expected_d, 5))
        np.testing.assert_allclose(
            training.reactor_decisions[:, 1].reshape(3, 5),
            np.column_stack((np.zeros((3, 2)), theta[:, 1:4])),
        )
        np.testing.assert_array_equal(
            training.reactor_upstream[0], targets[0, layout.mixer_slice]
        )
        np.testing.assert_array_equal(
            training.reactor_upstream[1], targets[0, layout.reactor_slice(0)]
        )
        np.testing.assert_array_equal(training.clarifier_feed, targets[:, 10:12])
        np.testing.assert_array_equal(training.clarifier_decisions, theta[:, [5, 6]])

    def test_quadratic_map_jacobian_is_exact_in_physical_coordinates(self) -> None:
        rng = np.random.default_rng(19)
        decision_matrix = rng.normal(size=(3, 2))
        state_matrix = rng.normal(size=(3, 4))
        model = linear_model(
            decision_matrix, state_matrix, rng.normal(size=3), seed=20
        )
        decision = np.asarray([0.3, -0.7])
        state = np.asarray([0.4, 1.2, -0.2, 0.8])
        prediction, decision_jacobian, state_jacobian = quadratic_prediction_jacobian(
            model, decision, state
        )
        np.testing.assert_allclose(
            prediction,
            model.coefficients[:, 0] * 0.0
            + model.predict(decision, state),
            rtol=0.0,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(decision_jacobian, decision_matrix, atol=2.0e-13)
        np.testing.assert_allclose(state_jacobian, state_matrix, atol=2.0e-13)

        step = 1.0e-6
        for coordinate in range(2):
            offset = np.zeros(2)
            offset[coordinate] = step
            finite = (
                model.predict(decision + offset, state)
                - model.predict(decision - offset, state)
            ) / (2.0 * step)
            np.testing.assert_allclose(finite, decision_jacobian[:, coordinate], atol=1.0e-8)

    def test_grouped_cross_validation_never_splits_one_plant_transitions(self) -> None:
        layout = small_layout()
        rng = np.random.default_rng(31)
        theta = controls(20)
        targets = rng.uniform(0.1, 5.0, size=(20, layout.state_size))
        training = extract_shared_unit_training(theta, targets, layout=layout)
        membership = np.tile(np.arange(1, 6), 4)
        result = cross_validate_shared_unit_models(
            training,
            plant_fold_membership=membership,
            ridge_grid=np.asarray([0.1]),
        )

        np.testing.assert_array_equal(result.plant_fold_membership, membership)
        self.assertEqual(len(result.fold_models), 5)
        self.assertEqual(result.reactor_out_of_fold_raw.shape, (100, 2))
        self.assertEqual(result.clarifier_out_of_fold_raw.shape, (20, 5))
        self.assertTrue(np.all(np.isfinite(result.reactor_out_of_fold_raw)))
        self.assertEqual(len(result.scores), 10)
        self.assertTrue(all(item.selected for item in result.scores))
        # Every one of the five rows for a plant is governed by its one plant fold.
        np.testing.assert_array_equal(
            result.plant_fold_membership[training.reactor_plant_index],
            np.repeat(membership, 5),
        )

    def test_two_start_closure_accepts_unique_root_and_analytic_derivative(self) -> None:
        layout = small_layout()
        models = identity_models()
        theta = controls(1)[0]
        influent = np.asarray([2.0, 3.0])
        scale = np.ones(layout.state_size)
        scale[-1] = 100.0
        result = solve_shared_unit_closure(
            models,
            theta,
            influent,
            scale,
            layout=layout,
            with_jacobian=True,
        )

        self.assertTrue(result.accepted, result.diagnostics.reason)
        self.assertIsNotNone(result.raw_jacobian_theta)
        self.assertLessEqual(
            result.diagnostics.mixer_agreement_inf,
            ROOT_MIXER_AGREEMENT_TOLERANCE,
        )
        self.assertEqual(result.raw.shape, (layout.state_size,))
        self.assertEqual(result.reactors.shape, (5, 2))
        self.assertEqual(result.clarifier.shape, (5,))
        # The implicit analytical derivative agrees with independently resolved roots.
        step = 2.0e-6
        for coordinate in (0, 4, 5, 6):
            plus = theta.copy()
            minus = theta.copy()
            plus[coordinate] += step
            minus[coordinate] -= step
            plus_result = solve_shared_unit_closure(
                models, plus, influent, scale, layout=layout
            )
            minus_result = solve_shared_unit_closure(
                models, minus, influent, scale, layout=layout
            )
            self.assertTrue(plus_result.accepted)
            self.assertTrue(minus_result.accepted)
            finite = (plus_result.raw - minus_result.raw) / (2.0 * step)
            np.testing.assert_allclose(
                finite,
                result.raw_jacobian_theta[:, coordinate],
                rtol=2.0e-5,
                atol=2.0e-6,
            )

    def test_closure_rejects_disagreeing_roots_and_uses_only_declared_starts(self) -> None:
        layout = small_layout()
        models = identity_models()
        theta = controls(1)[0]
        influent = np.asarray([2.0, 3.0])
        scale = np.ones(layout.state_size)

        def attempt(mixer: np.ndarray, raw_offset: float) -> SharedUnitRootAttempt:
            raw = np.zeros(layout.state_size)
            raw[:2] = mixer
            raw += raw_offset
            return SharedUnitRootAttempt(
                success=True,
                status=1,
                message="mock accepted",
                nfev=1,
                njev=1,
                cost=0.0,
                optimality=0.0,
                residual_inf=0.0,
                mixer=mixer,
                raw=raw,
                jacobian_rank=2,
                jacobian_condition=1.0,
                condition_times_epsilon=np.finfo(float).eps,
            )

        with patch(
            "closed_loop.v3_shared_unit._root_attempt",
            side_effect=(attempt(influent, 0.0), attempt(influent + 10.0, 10.0)),
        ) as mocked:
            result = solve_shared_unit_closure(
                models, theta, influent, scale, layout=layout
            )
        self.assertFalse(result.accepted)
        self.assertIsNone(result.raw)
        self.assertIn("two_start_agreement_failed", result.diagnostics.reason)
        first_start = mocked.call_args_list[0].args[4]
        second_start = mocked.call_args_list[1].args[4]
        np.testing.assert_array_equal(first_start, influent)
        np.testing.assert_array_equal(second_start, np.asarray([2.0, 10.5]))

    def test_common_projection_accepts_a_known_feasible_raw_state(self) -> None:
        layout = small_layout()
        invariant = np.asarray([[1.0, 0.0]])
        tss = np.asarray([0.0, 1.0])
        theta = controls(1)[0]
        influent = np.asarray([2.0, 3.0])
        operators = build_network_operators(
            influent,
            internal_recycle=theta[4],
            return_recycle=theta[5],
            waste_fraction=theta[6],
            invariant_operator=invariant,
            tss_weights=tss,
            layout=layout,
        )
        raw = no_conversion_feasible_state(
            influent, operators=operators, tss_weights=tss
        )
        row_scales = NetworkRowScales(
            equality=np.ones(operators.equality_matrix.shape[0]),
            inequality=np.ones(operators.inequality_matrix.shape[0]),
        )
        state_scale = np.maximum(1.0, np.abs(raw))
        projection = project_shared_unit_raw(
            raw,
            theta,
            influent,
            state_scale,
            row_scales,
            layout=layout,
            invariant_operator=invariant,
            tss_weights=tss,
        )
        self.assertTrue(projection.accepted, projection.diagnostics.as_dict())
        np.testing.assert_allclose(projection.state, raw, rtol=0.0, atol=1.0e-7)

    def test_leverage_and_value_only_optimizer_return_serializable_route_result(self) -> None:
        assets, case, leverage = optimization_fixture()
        self.assertGreaterEqual(leverage.reactor_limit, 0.0)
        self.assertGreaterEqual(leverage.clarifier_limit, 0.0)
        evaluation = evaluate_shared_unit(assets, case, np.full(7, 0.5))
        self.assertTrue(evaluation.available, evaluation.reason)
        self.assertEqual(evaluation.raw.shape, (17,))

        fake_optimized = SimpleNamespace(
            x=np.full(7, 0.5), success=True, message="synthetic", nit=0
        )
        settings = SharedUnitOptimizationSettings(
            maximum_iterations=2,
            maximum_function_evaluations=2,
            maximum_poll_evaluations=1,
            maximum_acceleration_probes=0,
        )
        with patch("closed_loop.v3_shared_unit.minimize", return_value=fake_optimized):
            result = optimize_shared_unit_case(assets, case, settings=settings)
        payload = result.as_dict()
        self.assertEqual(payload["route"], "shared_unit")
        self.assertEqual(payload["case_id"], "synthetic")
        self.assertFalse(payload["stationarity_resolved"])
        self.assertGreaterEqual(payload["root_attempts"], 2)
        self.assertIn("failed_closures", payload)
        self.assertGreater(payload["root_seconds"], 0.0)
        self.assertGreaterEqual(payload["projection_seconds"], 0.0)
        self.assertGreaterEqual(
            payload["evaluation_seconds"],
            payload["root_seconds"] + payload["projection_seconds"],
        )
        self.assertIn("selected", payload)
        restored = SharedUnitOptimizationResult.from_dict(
            json.loads(json.dumps(payload))
        )
        self.assertEqual(restored.route, "shared_unit")
        self.assertIsNotNone(restored.selected)
        np.testing.assert_array_equal(
            restored.selected.normalized_controls,
            result.selected.normalized_controls,
        )

        legacy_payload = dict(payload)
        for name in (
            "failed_closures",
            "root_seconds",
            "projection_seconds",
            "evaluation_seconds",
        ):
            legacy_payload.pop(name)
        restored_legacy = SharedUnitOptimizationResult.from_dict(legacy_payload)
        self.assertEqual(restored_legacy.failed_closures, 0)
        self.assertEqual(restored_legacy.root_seconds, 0.0)
        self.assertEqual(restored_legacy.projection_seconds, 0.0)
        self.assertEqual(restored_legacy.evaluation_seconds, 0.0)

    def test_failed_replay_counts_each_failed_root_start_and_is_never_validated(self) -> None:
        assets, case, _ = optimization_fixture()
        available = replace(
            evaluate_shared_unit(assets, case, np.full(7, 0.5)),
            elapsed_seconds=2.0,
            root_seconds=1.25,
            projection_seconds=0.5,
        )
        self.assertTrue(available.available)
        failed_attempt_1 = replace(
            available.closure.diagnostics.attempt_1,
            success=False,
            residual_inf=np.inf,
        )
        failed_attempt_2 = replace(
            available.closure.diagnostics.attempt_2,
            jacobian_rank=0,
        )
        failed_diagnostics = replace(
            available.closure.diagnostics,
            accepted=False,
            reason="both_starts_failed",
            attempt_1=failed_attempt_1,
            attempt_2=failed_attempt_2,
        )
        failed_closure = replace(
            available.closure,
            accepted=False,
            raw=None,
            mixer=None,
            reactors=None,
            clarifier=None,
            raw_jacobian_theta=None,
            diagnostics=failed_diagnostics,
        )
        failed_replay = SharedUnitEvaluation(
            available=False,
            reason="recycle_closure_unavailable:both_starts_failed",
            case_id=case.case_id,
            normalized_controls=np.full(7, 0.5),
            theta=available.theta.copy(),
            closure=failed_closure,
            projection=None,
            raw=None,
            projected=None,
            objective=None,
            objective_components=None,
            engineering_rows=None,
            engineering_quantities=None,
            trust=None,
            trust_rows=None,
            feasible=False,
            maximum_upper_residual=np.inf,
            elapsed_seconds=1.0,
            root_seconds=0.75,
            projection_seconds=0.0,
        )
        fake_optimized = SimpleNamespace(
            x=np.full(7, 0.5), success=True, message="synthetic", nit=0
        )
        with (
            patch("closed_loop.v3_shared_unit.minimize", return_value=fake_optimized),
            patch(
                "closed_loop.v3_shared_unit.evaluate_shared_unit",
                side_effect=(available, failed_replay),
            ),
        ):
            result = optimize_shared_unit_case(
                assets,
                case,
                settings=SharedUnitOptimizationSettings(
                    maximum_iterations=2,
                    maximum_function_evaluations=2,
                    maximum_poll_evaluations=1,
                    maximum_acceleration_probes=0,
                ),
            )

        self.assertEqual(result.status, "selected_replay_failed")
        self.assertEqual(result.classification, "primary_selected_replay_failed")
        self.assertFalse(result.locally_converged)
        self.assertFalse(result.stationarity_resolved)
        self.assertIs(result.selected, failed_replay)
        self.assertEqual(result.root_attempts, 4)
        self.assertEqual(result.failed_roots, 2)
        self.assertEqual(result.failed_closures, 1)
        self.assertEqual(result.projection_solves, 1)
        self.assertEqual(result.root_seconds, 2.0)
        self.assertEqual(result.projection_seconds, 0.5)
        self.assertEqual(result.evaluation_seconds, 3.0)

        with (
            patch("closed_loop.v3_shared_unit.minimize", return_value=fake_optimized),
            patch(
                "closed_loop.v3_shared_unit.evaluate_shared_unit",
                side_effect=(available, available, available, failed_replay),
            ),
        ):
            final_failure = optimize_shared_unit_case(
                assets,
                case,
                settings=SharedUnitOptimizationSettings(
                    maximum_iterations=2,
                    maximum_function_evaluations=2,
                    maximum_poll_evaluations=1,
                    maximum_acceleration_probes=0,
                ),
            )
        self.assertEqual(final_failure.status, "selected_replay_failed")
        self.assertEqual(
            final_failure.classification, "final_selected_replay_failed"
        )
        self.assertFalse(final_failure.locally_converged)
        self.assertFalse(final_failure.stationarity_resolved)
        self.assertIs(final_failure.selected, failed_replay)


if __name__ == "__main__":
    unittest.main()
