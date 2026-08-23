from __future__ import annotations

import unittest

import numpy as np

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
        engineering_scale=np.ones(5),
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
                self.assertEqual(problem.inequality_count, 2 * (layer_count - 2) + 5)
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
        self.assertEqual(assets.engineering_scale.shape, (5,))
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


if __name__ == "__main__":
    unittest.main()
