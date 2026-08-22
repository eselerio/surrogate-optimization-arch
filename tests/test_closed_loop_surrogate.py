from __future__ import annotations

import unittest

import numpy as np

import closed_loop.surrogate as surrogate_module
from closed_loop.surrogate import (
    NetworkLayout,
    QuadraticFeatureMap,
    QuadraticSurrogate,
    SurrogateValidationError,
    assess_raw_predictions,
    assess_raw_surrogate,
    calibrate_split_conformal,
    standardized_squared_error_scores,
)


class RetiredOptimizationPathTests(unittest.TestCase):
    def test_projection_qp_and_direct_search_are_not_public_or_present(self) -> None:
        retired = (
            "PhysicalProjector",
            "ProjectionResult",
            "SearchSettings",
            "NetworkOperators",
            "NetworkRowScales",
            "affine_projection",
            "build_network_operators",
            "deterministic_bounded_search",
            "fit_network_row_scales",
            "no_conversion_feasible_state",
        )
        for name in retired:
            self.assertNotIn(name, surrogate_module.__all__)
            self.assertFalse(hasattr(surrogate_module, name))


class QuadraticFeatureTests(unittest.TestCase):
    def test_article_dimensions_have_351_unique_features(self) -> None:
        rng = np.random.default_rng(71)
        decisions = rng.normal(size=(400, 5))
        influent = rng.normal(size=(400, 20))
        feature_map = QuadraticFeatureMap.fit(decisions, influent)
        design = feature_map.transform(decisions, influent)

        self.assertEqual(feature_map.feature_count, 351)
        self.assertEqual(design.shape, (400, 351))
        self.assertTrue(np.array_equal(design[:, 0], np.ones(400)))
        names = feature_map.feature_names(
            ("H", "a", "rI", "rR", "w"), tuple(f"x{i}" for i in range(20))
        )
        self.assertEqual(len(names), 351)
        self.assertLess(names.index("standardized[H*x0]"), names.index("standardized[H*x1]"))
        self.assertLess(names.index("standardized[H*x19]"), names.index("standardized[a*x0]"))

    def test_fixed_ols_recovers_exact_response_and_persists_pivoted_qr(self) -> None:
        rng = np.random.default_rng(910)
        decisions = rng.uniform(-2.0, 3.0, size=(430, 5))
        influent = rng.uniform(0.2, 5.0, size=(430, 20))
        generating_map = QuadraticFeatureMap.fit(decisions, influent)
        design = generating_map.transform(decisions, influent)
        generating_coefficients = rng.normal(scale=0.05, size=(351, 4))
        responses = design @ generating_coefficients + np.array((3.0, 7.0, 11.0, 13.0))

        model = QuadraticSurrogate.fit(decisions, influent, responses)
        fitted = model.predict(decisions, influent)

        self.assertEqual(model.coefficients.shape, (4, 351))
        self.assertEqual(model.feature_qr_upper.shape, (351, 351))
        np.testing.assert_array_equal(
            np.sort(model.feature_qr_pivots), np.arange(351, dtype=np.int64)
        )
        self.assertLess(model.diagnostics.condition_number, 1.0e8)
        self.assertLess(np.max(np.abs(fitted - responses)), 2.0e-11)

    def test_pivoted_qr_leverage_matches_definition_and_trace(self) -> None:
        rng = np.random.default_rng(72)
        decisions = rng.normal(size=(80, 2))
        influent = rng.normal(size=(80, 2))
        responses = rng.normal(size=(80, 3))
        model = QuadraticSurrogate.fit(decisions, influent, responses)
        design = model.feature_map.transform(decisions, influent)

        leverage = model.leverage(decisions, influent)
        gram_inverse_design_t = np.linalg.solve(design.T @ design, design.T)
        expected = np.sum(design.T * gram_inverse_design_t, axis=0)
        np.testing.assert_allclose(leverage, expected, rtol=2.0e-11, atol=2.0e-12)
        self.assertAlmostEqual(float(np.sum(leverage)), model.feature_map.feature_count, places=10)
        self.assertAlmostEqual(
            model.leverage(decisions[7], influent[7]), float(expected[7]), places=11
        )
        self.assertEqual(
            model.maximum_training_leverage(decisions, influent), float(np.max(leverage))
        )

    def test_zero_variance_is_rejected_instead_of_silently_rescaled(self) -> None:
        rng = np.random.default_rng(8)
        decisions = rng.normal(size=(400, 5))
        decisions[:, 2] = 1.0
        influent = rng.normal(size=(400, 20))
        with self.assertRaises(SurrogateValidationError):
            QuadraticFeatureMap.fit(decisions, influent)


class CalibrationAndAssessmentTests(unittest.TestCase):
    def test_conformal_quantile_uses_finite_sample_one_based_rule(self) -> None:
        scores = np.linspace(0.001, 0.2, 2000)
        observed = np.repeat(np.sqrt(scores)[:, None], 2, axis=1)
        predicted = np.zeros_like(observed)
        calibration = calibrate_split_conformal(
            observed, predicted, np.ones(2), alpha=0.05
        )

        self.assertEqual(calibration.n, 2000)
        self.assertEqual(calibration.k_one_based, 1901)
        self.assertEqual(calibration.index_zero_based, 1900)
        self.assertEqual(calibration.delta, float(np.sort(scores)[1900]))
        np.testing.assert_allclose(calibration.scores, scores, rtol=2.0e-15, atol=2.0e-16)
        self.assertEqual(len(calibration.scores_sha256), 64)
        self.assertEqual(
            calibration.scores_sha256,
            calibrate_split_conformal(observed, predicted, np.ones(2)).scores_sha256,
        )

    def test_calibration_rejects_zero_large_and_nonfinite_thresholds(self) -> None:
        zeros = np.zeros((20, 2))
        with self.assertRaisesRegex(SurrogateValidationError, "must lie"):
            calibrate_split_conformal(zeros, zeros, np.ones(2))
        with self.assertRaisesRegex(SurrogateValidationError, "must lie"):
            calibrate_split_conformal(
                np.full((20, 2), 2.0), zeros, np.ones(2)
            )
        nonfinite = zeros.copy()
        nonfinite[0, 0] = np.nan
        with self.assertRaisesRegex(SurrogateValidationError, "observed"):
            calibrate_split_conformal(nonfinite, zeros, np.ones(2))

    def test_raw_assessment_reports_coordinate_block_and_gate_metrics(self) -> None:
        truth = np.asarray(
            [[1.0, 3.0], [2.0, 5.0], [3.0, 7.0], [4.0, 9.0]], dtype=float
        )
        prediction = truth + np.asarray(
            [[0.2, -0.4], [-0.2, 0.4], [0.2, -0.4], [-0.2, 0.4]]
        )
        scale = np.asarray([2.0, 4.0])
        metrics = assess_raw_predictions(
            truth,
            prediction,
            scale,
            delta=0.011,
            blocks={"first": [0], "all": slice(0, 2)},
        )

        expected_scores = standardized_squared_error_scores(truth, prediction, scale)
        np.testing.assert_allclose(metrics.scores, expected_scores)
        np.testing.assert_allclose(metrics.coordinate_metrics.rmse, [0.2, 0.4])
        np.testing.assert_allclose(metrics.coordinate_metrics.nrmse, [0.1, 0.1])
        self.assertAlmostEqual(metrics.complete_state_standardized_rmse, 0.1)
        self.assertAlmostEqual(metrics.block_metric("all").standardized_rmse, 0.1)
        self.assertEqual(metrics.empirical_coverage, 1.0)
        self.assertTrue(metrics.predictions_finite)
        self.assertTrue(metrics.complete_state_rmse_passed)
        self.assertTrue(metrics.coverage_passed)
        self.assertTrue(metrics.passed)

    def test_nonfinite_prediction_is_a_failed_gate_not_a_refit(self) -> None:
        truth = np.arange(12, dtype=float).reshape(4, 3)
        prediction = truth.copy()
        prediction[0, 0] = np.inf
        metrics = assess_raw_predictions(
            truth, prediction, np.ones(3), delta=0.5
        )
        self.assertFalse(metrics.predictions_finite)
        self.assertFalse(metrics.complete_state_rmse_passed)
        self.assertFalse(metrics.coverage_passed)
        self.assertFalse(metrics.passed)

    def test_assess_raw_surrogate_uses_the_frozen_model_response_scale(self) -> None:
        rng = np.random.default_rng(97)
        decisions = rng.normal(size=(60, 2))
        influent = rng.normal(size=(60, 2))
        base_map = QuadraticFeatureMap.fit(decisions, influent)
        features = base_map.transform(decisions, influent)
        responses = features @ rng.normal(size=(15, 3))
        model = QuadraticSurrogate.fit(decisions, influent, responses)
        assessment_decisions = rng.normal(size=(8, 2))
        assessment_influent = rng.normal(size=(8, 2))
        observed = model.predict(assessment_decisions, assessment_influent)
        observed = observed + 0.05 * model.response_scale

        metrics = assess_raw_surrogate(
            model,
            assessment_decisions,
            assessment_influent,
            observed,
            delta=0.01,
        )
        self.assertAlmostEqual(metrics.complete_state_standardized_rmse, 0.05)
        self.assertEqual(metrics.empirical_coverage, 1.0)
        self.assertTrue(metrics.passed)


class NetworkLayoutTests(unittest.TestCase):
    def test_complete_response_layout_has_declared_order_and_size(self) -> None:
        layout = NetworkLayout()

        self.assertEqual(layout.state_size, 170)
        self.assertEqual(layout.mixer_slice, slice(0, 20))
        self.assertEqual(layout.reactor_slice(0), slice(20, 40))
        self.assertEqual(layout.reactor_slice(4), slice(100, 120))
        self.assertEqual(layout.overflow_flow_slice, slice(120, 140))
        self.assertEqual(layout.underflow_flow_slice, slice(140, 160))
        self.assertEqual(layout.layer_slice, slice(160, 170))


if __name__ == "__main__":
    unittest.main()
