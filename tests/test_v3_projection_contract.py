from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
from scipy import linalg

from closed_loop.projection import (
    PhysicalProjector,
    ProjectionWarmStart,
    QuadraticSurrogate,
    SurrogateValidationError,
)


class RidgeNumericalContractTests(unittest.TestCase):
    @staticmethod
    def _rank_deficient_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(731)
        base = np.linspace(-2.0, 2.0, 48)
        decisions = base[:, None]
        influent = base[:, None]
        responses = np.column_stack(
            (
                1.5 + 2.0 * base - 0.3 * base**2 + 0.01 * rng.normal(size=base.size),
                -2.0 + 0.7 * base + 0.1 * base**2 + 0.02 * rng.normal(size=base.size),
            )
        )
        return decisions, influent, responses

    def test_augmented_qr_svd_audit_and_effective_degrees_of_freedom(self) -> None:
        decisions, influent, responses = self._rank_deficient_data()
        gamma = 1.0e-8
        model = QuadraticSurrogate.fit_ridge(
            decisions, influent, responses, ridge_penalty=gamma
        )
        design = model.feature_map.transform(decisions, influent)
        standardized_response = (
            responses - model.response_center
        ) / model.response_scale
        rows, feature_count = design.shape
        penalty = np.diag(np.r_[0.0, np.ones(feature_count - 1)])
        augmented_design = np.vstack(
            (design, np.sqrt(rows * gamma) * penalty)
        )
        augmented_response = np.vstack(
            (standardized_response, np.zeros((feature_count, responses.shape[1])))
        )

        self.assertLess(np.linalg.matrix_rank(design), feature_count)
        q_qr, r_qr, pivot = linalg.qr(
            augmented_design, mode="economic", pivoting=True
        )
        permuted = linalg.solve_triangular(r_qr, q_qr.T @ augmented_response)
        coefficients_qr = np.empty_like(permuted)
        coefficients_qr[pivot] = permuted
        u_svd, singular_values, vt_svd = linalg.svd(
            augmented_design, full_matrices=False, lapack_driver="gesdd"
        )
        coefficients_svd = (
            (vt_svd.T / singular_values[None, :])
            @ (u_svd.T @ augmented_response)
        )
        condition = float(singular_values[0] / singular_values[-1])
        agreement = float(np.max(np.abs(coefficients_qr - coefficients_svd))) / (
            1.0
            + max(
                float(np.max(np.abs(coefficients_qr))),
                float(np.max(np.abs(coefficients_svd))),
            )
        )
        threshold = 100.0 * condition * np.finfo(np.float64).eps

        fitted_agreement = float(
            np.max(np.abs(model.coefficients.T - coefficients_qr))
        ) / (1.0 + float(np.max(np.abs(coefficients_qr))))
        self.assertLessEqual(fitted_agreement, threshold)
        self.assertAlmostEqual(model.diagnostics.condition_number, condition, places=9)
        self.assertAlmostEqual(
            model.diagnostics.augmented_condition_number, condition, places=9
        )
        self.assertAlmostEqual(
            model.diagnostics.smallest_singular_value,
            float(singular_values[-1]),
            places=14,
        )
        self.assertAlmostEqual(
            model.diagnostics.largest_singular_value,
            float(singular_values[0]),
            places=12,
        )
        self.assertAlmostEqual(model.diagnostics.coefficient_agreement, agreement, places=13)
        self.assertAlmostEqual(model.diagnostics.acceptance_threshold, threshold, places=18)
        self.assertLessEqual(model.diagnostics.coefficient_agreement, threshold)

        ridge_matrix = (
            design.T @ design + rows * gamma * (penalty.T @ penalty)
        )
        expected_df = float(
            np.trace(design @ np.linalg.solve(ridge_matrix, design.T))
        )
        self.assertAlmostEqual(model.effective_degrees_of_freedom, expected_df, places=8)
        self.assertAlmostEqual(
            model.diagnostics.effective_degrees_of_freedom, expected_df, places=8
        )

    def test_augmented_condition_gate_is_enforced(self) -> None:
        decisions, influent, responses = self._rank_deficient_data()
        with self.assertRaisesRegex(
            SurrogateValidationError, "augmented ridge-system condition gate"
        ):
            QuadraticSurrogate.fit_ridge(
                decisions, influent, responses, ridge_penalty=1.0e-16
            )


class PhysicalProjectionNumericalContractTests(unittest.TestCase):
    def test_cold_final_solve_reconstructs_multipliers_and_exact_residuals(self) -> None:
        state_scale = np.asarray([2.0, 0.5])
        equality_scale = np.asarray([4.0])
        inequality_scale = np.asarray([3.0])
        raw = np.asarray([-2.0, 2.0])
        equality = np.asarray([[2.0, 8.0]])
        equality_rhs = np.asarray([8.0])
        inequality = np.asarray([[1.5, 0.0]])
        delta = 2.0e-9

        projector = PhysicalProjector(
            state_scale, equality_scale, inequality_scale
        )
        calls: list[object] = []

        def mocked_solve_once(**kwargs: object) -> SimpleNamespace:
            calls.append(kwargs["warm_start"])
            displacement = (
                np.asarray([0.0, 0.0])
                if kwargs["warm_start"] is not None
                else np.asarray([1.0 + delta, -2.0])
            )
            return SimpleNamespace(
                x=displacement,
                # Deliberately invalid solver duals must never enter the result.
                y=np.asarray([999.0, -999.0, 888.0, -777.0]),
                info=SimpleNamespace(status="mock", status_val=1, iter=1),
            )

        projector._solve_once = mocked_solve_once  # type: ignore[method-assign]
        warm = ProjectionWarmStart(np.zeros(2), np.zeros(4))
        result = projector.project(
            raw,
            equality,
            equality_rhs,
            inequality,
            warm_start=warm,
        )

        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0], warm)
        self.assertIsNone(calls[1])
        self.assertTrue(result.diagnostics.retried_cold)
        self.assertTrue(result.diagnostics.multipliers_reconstructed)
        self.assertTrue(result.accepted)

        scaled_equality = equality * state_scale[None, :] / equality_scale[:, None]
        required_equality = (equality_rhs - equality @ raw) / equality_scale
        physical_scaled_inequality = inequality / inequality_scale[:, None]
        scaled_inequality = np.vstack(
            (-np.eye(2), physical_scaled_inequality * state_scale[None, :])
        )
        inequality_rhs_scaled = np.r_[
            raw / state_scale, -(physical_scaled_inequality @ raw)
        ]
        expected_displacement = np.asarray([1.0 + delta, -2.0])
        expected_state = raw + state_scale * expected_displacement
        expected_slack = inequality_rhs_scaled - scaled_inequality @ expected_displacement

        np.testing.assert_allclose(result.displacement, expected_displacement, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(result.state, expected_state, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(result.inequality_slack, expected_slack, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(result.equality_multipliers, [2.0], atol=2.0e-12)
        np.testing.assert_allclose(
            result.inequality_multipliers, [3.0 + delta, 0.0, 0.0], atol=2.0e-12
        )
        self.assertTrue(np.all(result.inequality_multipliers >= 0.0))

        equality_term = scaled_equality.T @ result.equality_multipliers
        inequality_term = scaled_inequality.T @ result.inequality_multipliers
        stationarity = expected_displacement + equality_term + inequality_term
        expected_residuals = {
            "equality_residual": float(
                np.linalg.norm(
                    scaled_equality @ expected_displacement - required_equality,
                    ord=np.inf,
                )
            ),
            "inequality_residual": float(
                np.max(np.maximum(physical_scaled_inequality @ expected_state, 0.0))
            ),
            "nonnegativity_residual": float(
                np.max(np.maximum(-expected_state / state_scale, 0.0))
            ),
            "dual_feasibility_residual": float(
                np.max(np.maximum(-result.inequality_multipliers, 0.0))
            ),
            "stationarity_residual": float(
                np.max(
                    np.abs(stationarity)
                    / (
                        1.0
                        + np.abs(expected_displacement)
                        + np.abs(equality_term)
                        + np.abs(inequality_term)
                    )
                )
            ),
            "complementarity_residual": float(
                np.linalg.norm(
                    result.inequality_multipliers * expected_slack, ord=np.inf
                )
            ),
        }
        for name, expected in expected_residuals.items():
            self.assertAlmostEqual(getattr(result.diagnostics, name), expected, places=16)
        self.assertAlmostEqual(
            result.diagnostics.complementarity_residual, 6.0e-9, places=14
        )
        for array in (
            result.state,
            result.displacement,
            result.equality_multipliers,
            result.inequality_multipliers,
            result.inequality_slack,
        ):
            self.assertTrue(np.all(np.isfinite(array)))


if __name__ == "__main__":
    unittest.main()
