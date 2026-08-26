from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import closed_loop.v3_smooth as v3_smooth

from closed_loop.model import (
    ArticleOperatingPoint,
    ClarifierParameters,
    INFLUENT_LOWER,
    INFLUENT_UPPER,
    NOMINAL_INFLUENT,
    TSS_VECTOR,
    initial_state,
    process_rates,
)
from closed_loop.v3_smooth import (
    DECISION_LOWER,
    DECISION_UPPER,
    DirectAssets,
    DirectCase,
    SmoothScales,
    SolverSettings,
    branches_match,
    build_direct_nlp,
    classify_branches,
    direct_initial_point,
    evaluate_direct,
    evaluate_smooth_response,
    extract_reduced_states,
    fit_direct_assets,
    independent_kkt_diagnostics,
    ordered_normalized_starts,
    smooth_division,
    smooth_maximum,
    smooth_minimum,
    smooth_positive_part,
    smooth_process_rates,
)


def _clarifier(layer_count: int) -> ClarifierParameters:
    return ClarifierParameters(
        layer_count=layer_count,
        feed_layer=(layer_count - 1) // 2,
        layer_volume=6_000.0 / layer_count,
    )


def _assets(layer_count: int) -> DirectAssets:
    state_count = 100 + layer_count
    return DirectAssets(
        clarifier=_clarifier(layer_count),
        smoothing=SmoothScales(10.0, 100.0, 100.0, 100.0, 100.0,
                               100.0, 100.0, 250.0, 10_000.0),
        state_center=np.ones(state_count),
        state_scale=np.ones(state_count),
        feed_scale=100.0,
        balance_scale=np.ones(state_count),
        quality_scale=np.ones(4),
        envelope_scale=np.ones(2 * (layer_count - 2)),
        engineering_scale=np.ones(4),
        decision_center=(DECISION_LOWER + DECISION_UPPER) / 2.0,
        decision_scale=(DECISION_UPPER - DECISION_LOWER) / np.sqrt(12.0),
        influent_center=(INFLUENT_LOWER + INFLUENT_UPPER) / 2.0,
        influent_scale=(INFLUENT_UPPER - INFLUENT_LOWER) / np.sqrt(12.0),
    )


class SmoothingPrimitiveTests(unittest.TestCase):
    def test_positive_part_is_finite_and_nonzero_on_large_negative_input(self) -> None:
        value = smooth_positive_part(-1.0e8, epsilon=1.0e-8, scale=1.0)
        self.assertTrue(np.isfinite(value))
        self.assertGreater(value, 0.0)
        self.assertAlmostEqual(value, 2.5e-25, delta=1e-39)

    def test_smooth_primitives_obey_declared_formulas(self) -> None:
        epsilon, scale = 1.0e-4, 3.0
        a, b = -2.0, 5.0
        root = np.hypot(a - b, epsilon * scale)
        self.assertEqual(smooth_maximum(a, b, epsilon=epsilon, scale=scale),
                         0.5 * (a + b + root))
        self.assertEqual(smooth_minimum(a, b, epsilon=epsilon, scale=scale),
                         0.5 * (a + b - root))
        self.assertEqual(
            smooth_division(2.0, 3.0, epsilon=epsilon, scale=scale),
            6.0 / (9.0 + (epsilon * scale) ** 2),
        )

    def test_smooth_rates_converge_to_reference_away_from_branches(self) -> None:
        state = NOMINAL_INFLUENT + 1.0
        scales = _assets(5).smoothing
        smooth = smooth_process_rates(state, scales, epsilon=1.0e-10)
        reference = process_rates(state)
        np.testing.assert_allclose(smooth, reference, rtol=1.0e-12, atol=1.0e-12)
        self.assertGreaterEqual(float(np.min(smooth)), 0.0)


class VariableLayerModelTests(unittest.TestCase):
    def test_numeric_and_symbolic_paths_agree_for_five_and_ten_layers(self) -> None:
        theta = np.asarray([18.0, 0.2, 0.3, 0.4, 2.0, 0.75, 0.02])
        for layer_count in (5, 10):
            with self.subTest(layer_count=layer_count):
                assets = _assets(layer_count)
                state = initial_state(NOMINAL_INFLUENT, 2, assets.clarifier)
                feed_tss = float(TSS_VECTOR @ state[80:100])
                response, residual = evaluate_smooth_response(
                    theta, NOMINAL_INFLUENT, state, feed_tss, assets,
                )
                self.assertEqual(response.shape, (160 + layer_count,))
                self.assertEqual(residual.shape, (100 + layer_count,))
                self.assertTrue(np.all(np.isfinite(response)))
                self.assertTrue(np.all(np.isfinite(residual)))

                problem = build_direct_nlp(
                    assets, compile_solver=False, name=f"test_direct_L{layer_count}",
                )
                self.assertEqual(problem.variable_count, 108 + layer_count)
                self.assertEqual(problem.equality_count, 101 + layer_count)
                self.assertEqual(problem.inequality_count, 2 * (layer_count - 2) + 4)
                normalized = (theta - DECISION_LOWER) / (DECISION_UPPER - DECISION_LOWER)
                primal = np.concatenate((
                    normalized,
                    (state - assets.state_center) / assets.state_scale,
                    [feed_tss / assets.feed_scale],
                ))
                evaluated = evaluate_direct(problem, primal, DirectCase(NOMINAL_INFLUENT))
                np.testing.assert_allclose(evaluated["response"], response, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(evaluated["raw_residual"], residual, rtol=1e-12, atol=1e-12)
                diagnostics = independent_kkt_diagnostics(
                    problem, primal, DirectCase(NOMINAL_INFLUENT),
                )
                self.assertTrue(diagnostics.finite)
                self.assertEqual(diagnostics.equality_multipliers.shape,
                                 (problem.equality_count,))

    def test_branch_comparison_ignores_margin_magnitude(self) -> None:
        assets = _assets(5)
        state = initial_state(NOMINAL_INFLUENT, 2, assets.clarifier)
        first = classify_branches(state, assets)
        second = classify_branches(state * (1.0 + 1.0e-12), assets)
        self.assertTrue(branches_match(first, second))


class DevelopmentScalingAndStartTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(2048)
        self.layer_count = 5
        self.rows = 24
        self.decisions = rng.uniform(DECISION_LOWER, DECISION_UPPER,
                                     size=(self.rows, 7))
        self.influents = rng.uniform(INFLUENT_LOWER, INFLUENT_UPPER,
                                     size=(self.rows, 20))
        self.targets = rng.uniform(1.0, 200.0,
                                   size=(self.rows, 160 + self.layer_count))
        # Give Clarifier layers a physically ordered, non-degenerate profile.
        self.targets[:, 160:] = np.sort(
            rng.uniform(20.0, 8_000.0, size=(self.rows, self.layer_count)), axis=1,
        )

    def test_fit_uses_median_mad_and_all_v3_dimensions(self) -> None:
        clarifier = _clarifier(self.layer_count)
        assets = fit_direct_assets(
            self.decisions, self.influents, self.targets, clarifier=clarifier,
        )
        states = extract_reduced_states(self.targets, self.layer_count)
        expected_center = np.median(states, axis=0)
        expected_scale = np.maximum(
            1.0, np.median(np.abs(states - expected_center), axis=0),
        )
        np.testing.assert_allclose(assets.state_center, expected_center)
        np.testing.assert_allclose(assets.state_scale, expected_scale)
        self.assertEqual(assets.balance_scale.shape, (105,))
        self.assertEqual(assets.envelope_scale.shape, (6,))
        self.assertEqual(assets.engineering_scale.shape, (4,))
        self.assertTrue(np.all(assets.quality_scale > 0.0))

    def test_nine_starts_are_reproducible_open_lhs_and_initialize_by_nearest_row(self) -> None:
        first = ordered_normalized_starts()
        second = ordered_normalized_starts()
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[0], np.full(7, 0.5))
        self.assertTrue(np.all((first[1:] > 0.0) & (first[1:] < 1.0)))
        for column in range(7):
            np.testing.assert_array_equal(
                np.sort(np.floor(first[1:, column] * 8.0).astype(int)),
                np.arange(8),
            )

        assets = fit_direct_assets(
            self.decisions, self.influents, self.targets,
            clarifier=_clarifier(self.layer_count),
        )
        case = DirectCase(self.influents[0])
        primal, nearest = direct_initial_point(
            first[0], case, self.decisions, self.influents, self.targets, assets,
        )
        self.assertIn(nearest, range(self.rows))
        self.assertEqual(primal.shape, (113,))
        _, state, feed_tss = (
            DECISION_LOWER + (DECISION_UPPER - DECISION_LOWER) * primal[:7],
            assets.state_center + assets.state_scale * primal[7:-1],
            assets.feed_scale * primal[-1],
        )
        self.assertGreaterEqual(float(np.min(state)), 1.0e-8)
        self.assertGreaterEqual(feed_tss, 1.0)


class DirectContinuationWarmStartTests(unittest.TestCase):
    def test_stage_returns_and_reuses_ipopt_duals(self) -> None:
        class RecordingSolver:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def __call__(self, **arguments: object) -> dict[str, np.ndarray]:
                self.calls.append(arguments)
                return {
                    "x": np.asarray([0.2, 0.3, 0.4]),
                    "lam_x": np.asarray([1.0, 2.0, 3.0]),
                    "lam_g": np.asarray([-4.0, 5.0]),
                }

            @staticmethod
            def stats() -> dict[str, object]:
                return {"return_status": "Solve_Succeeded", "success": True,
                        "iter_count": 3}

        solver = RecordingSolver()
        problem = SimpleNamespace(
            solver=solver,
            variable_count=3,
            equality_count=1,
            inequality_count=1,
            lower_bounds=np.zeros(3),
            upper_bounds=np.ones(3),
            constraint_lower_bounds=np.asarray([0.0, -np.inf]),
            constraint_upper_bounds=np.asarray([0.0, 0.0]),
            settings=SolverSettings(),
            epsilon=1.0e-4,
            receiver_half_width=1.0e-3,
        )
        case = DirectCase(NOMINAL_INFLUENT)
        evaluated = {"equality": np.zeros(1), "inequality": -np.ones(1)}
        with patch.object(v3_smooth, "evaluate_direct", return_value=evaluated):
            first, primal, dual = v3_smooth._solve_direct_stage(
                problem, case, np.asarray([0.1, 0.2, 0.3]),
            )
            second, _, _ = v3_smooth._solve_direct_stage(
                problem, case, primal, dual,
            )

        self.assertTrue(first.feasible)
        self.assertTrue(second.feasible)
        self.assertNotIn("lam_x0", solver.calls[0])
        self.assertNotIn("lam_g0", solver.calls[0])
        np.testing.assert_array_equal(solver.calls[1]["lam_x0"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(solver.calls[1]["lam_g0"], [-4.0, 5.0])
        np.testing.assert_array_equal(first.constraint_multipliers, [-4.0, 5.0])
        restored = v3_smooth.ContinuationStageResult.from_dict(first.as_dict())
        np.testing.assert_array_equal(
            restored.constraint_multipliers, [-4.0, 5.0]
        )

    def test_completed_starts_round_trip_and_skip_solver_work(self) -> None:
        assets = _assets(5)
        case = DirectCase(NOMINAL_INFLUENT)
        settings = SolverSettings()
        starts = ordered_normalized_starts()
        contract = v3_smooth.direct_start_resume_contract(assets, case, settings)
        completed = {}
        for index, initial in enumerate(starts):
            result = v3_smooth.DirectStartResult(
                start_index=index,
                initial_normalized_controls=initial.copy(),
                resume_contract=contract,
                nearest_development_row=0,
                stages=(),
                objective=np.nan,
                normalized_controls=initial.copy(),
                theta=DECISION_LOWER + (DECISION_UPPER - DECISION_LOWER) * initial,
                state=np.full(assets.state_count, np.nan),
                feed_tss=np.nan,
                response=np.full(assets.response_count, np.nan),
                engineering=np.full(11, np.nan),
                objective_components=np.full(6, np.nan),
                branch=None,
                kkt=None,
                feasible=False,
                stationary=False,
                status="continuation_failed",
            )
            payload = result.as_dict()
            payload["objective"] = None
            payload["feed_tss"] = None
            completed[index] = v3_smooth.DirectStartResult.from_dict(payload)
        callbacks = []
        with patch.object(v3_smooth, "build_direct_nlp", return_value=SimpleNamespace()):
            outcome = v3_smooth.solve_direct_multistart(
                assets, case,
                np.empty((1, 7)), np.empty((1, 20)),
                np.empty((1, assets.response_count)),
                settings=settings, starts=starts,
                completed_starts=completed,
                progress_callback=callbacks.append,
            )
        self.assertEqual(len(outcome.starts), 9)
        self.assertEqual(callbacks, [])
        self.assertEqual(outcome.status, "no_validated_feasible_start")
        invalid = dict(completed)
        invalid[0] = v3_smooth.DirectStartResult.from_dict({
            **completed[0].as_dict(), "resume_contract": "wrong",
        })
        with self.assertRaisesRegex(ValueError, "completed direct start"):
            v3_smooth.solve_direct_multistart(
                assets, case,
                np.empty((1, 7)), np.empty((1, 20)),
                np.empty((1, assets.response_count)),
                settings=settings, starts=starts,
                completed_starts=invalid,
            )

    def test_final_audit_exception_is_a_retained_start_failure(self) -> None:
        assets = _assets(5)
        case = DirectCase(NOMINAL_INFLUENT)
        starts = ordered_normalized_starts()[:1]

        def problem(_assets, epsilon, receiver_half_width, **_kwargs):
            return SimpleNamespace(
                epsilon=epsilon, receiver_half_width=receiver_half_width,
            )

        def stage(problem_value, _case, primal, _dual=None):
            record = v3_smooth.ContinuationStageResult(
                problem_value.epsilon, problem_value.receiver_half_width,
                "Solve_Succeeded", True, 0.01, 1, primal.copy(), True,
            )
            return record, primal, (np.empty(0), np.empty(0))

        callbacks = []
        with patch.object(v3_smooth, "build_direct_nlp", side_effect=problem), \
                patch.object(v3_smooth, "direct_initial_point", return_value=(np.zeros(1), 0)), \
                patch.object(v3_smooth, "_solve_direct_stage", side_effect=stage), \
                patch.object(v3_smooth, "evaluate_direct", side_effect=FloatingPointError("audit")):
            outcome = v3_smooth.solve_direct_multistart(
                assets, case,
                np.empty((1, 7)), np.empty((1, 20)),
                np.empty((1, assets.response_count)),
                starts=starts, allow_reduced_starts=True,
                progress_callback=callbacks.append,
            )
        self.assertEqual(outcome.starts[0].status, "final_audit_exception")
        self.assertIn("FloatingPointError", outcome.starts[0].error)
        self.assertEqual(len(callbacks), 1)


if __name__ == "__main__":
    unittest.main()
