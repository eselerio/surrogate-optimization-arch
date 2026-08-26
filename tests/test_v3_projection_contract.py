from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
from scipy import linalg

from closed_loop.projection import (
    LogOverflowTSSClosure,
    NetworkLayout,
    PhysicalProjector,
    ProjectionWarmStart,
    QuadraticSurrogate,
    SurrogateValidationError,
    build_network_operators,
    fit_network_row_scales,
    no_conversion_feasible_state,
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

    def test_log_overflow_closure_is_positive_and_recovers_quadratic_target(self) -> None:
        rng = np.random.default_rng(903)
        decisions = rng.uniform(-1.0, 1.0, size=(80, 1))
        influent = rng.uniform(-2.0, 2.0, size=(80, 1))
        log_target = (
            0.4
            + 0.3 * decisions[:, 0]
            - 0.2 * influent[:, 0]
            + 0.1 * decisions[:, 0] * influent[:, 0]
            + 0.05 * influent[:, 0] ** 2
        )
        exact = 2.5 * np.exp(log_target)
        closure = LogOverflowTSSClosure.fit_ridge(
            decisions,
            influent,
            exact,
            ridge_penalty=1.0e-10,
            reference_concentration=2.5,
        )
        predicted = closure.predict(decisions, influent)
        self.assertTrue(np.all(predicted > 0.0))
        np.testing.assert_allclose(predicted, exact, rtol=2.0e-9, atol=2.0e-9)


class ReducedResponseNetworkTests(unittest.TestCase):
    @staticmethod
    def _layout() -> NetworkLayout:
        return NetworkLayout(
            stage_count=1,
            component_count=2,
            layer_count=3,
            soluble_indices=(0,),
            particulate_indices=(1,),
        )

    def test_layout_and_no_conversion_state_follow_reduced_contract(self) -> None:
        layout = self._layout()
        self.assertEqual(layout.state_size, 9)
        self.assertEqual(layout.inventory_index, 8)
        self.assertEqual(layout.inventory_slice, slice(8, 9))
        self.assertEqual(layout.equality_count_without_invariants, 5)
        self.assertEqual(layout.inequality_count, 3)

        operators = build_network_operators(
            np.asarray([2.0, 10.0]),
            internal_recycle=1.0,
            return_recycle=0.5,
            waste_fraction=0.1,
            invariant_operator=np.asarray([[1.0, 0.0]]),
            tss_weights=np.asarray([0.0, 1.0]),
            layout=layout,
            clarifier_volume_m3=30.0,
        )
        state = no_conversion_feasible_state(
            np.asarray([2.0, 10.0]),
            operators=operators,
            tss_weights=np.asarray([0.0, 1.0]),
        )
        self.assertEqual(operators.equality_matrix.shape, (6, 9))
        self.assertEqual(operators.inequality_matrix.shape, (3, 9))
        self.assertEqual(state[layout.inventory_index], 300.0)
        np.testing.assert_allclose(
            operators.equality_matrix @ state,
            operators.equality_rhs,
            atol=1.0e-13,
        )
        self.assertLessEqual(
            float(np.max(operators.inequality_matrix @ state)), 1.0e-13
        )

    def test_inventory_rows_are_the_tight_endpoint_envelope(self) -> None:
        layout = self._layout()
        q_e, q_u = 0.9, 0.6
        x_e, x_u = 4.0, 16.0
        volume, endpoint_volume = 30.0, 10.0
        operators = build_network_operators(
            np.asarray([0.0, 0.0]),
            internal_recycle=0.0,
            return_recycle=0.5,
            waste_fraction=0.1,
            invariant_operator=np.asarray([[1.0, 0.0]]),
            tss_weights=np.asarray([0.0, 1.0]),
            layout=layout,
            clarifier_volume_m3=volume,
        )
        response = np.zeros(layout.state_size)
        response[layout.overflow_flow_slice.start + 1] = q_e * x_e
        response[layout.underflow_flow_slice.start + 1] = q_u * x_u
        lower = (volume - endpoint_volume) * x_e + endpoint_volume * x_u
        upper = endpoint_volume * x_e + (volume - endpoint_volume) * x_u

        response[layout.inventory_index] = lower
        residual = operators.inequality_matrix[-2:] @ response
        self.assertAlmostEqual(residual[0], 0.0, places=12)
        self.assertLess(residual[1], 0.0)
        response[layout.inventory_index] = upper
        residual = operators.inequality_matrix[-2:] @ response
        self.assertLess(residual[0], 0.0)
        self.assertAlmostEqual(residual[1], 0.0, places=12)

    def test_log_overflow_closure_adds_and_enforces_one_equality(self) -> None:
        layout = self._layout()
        operators = build_network_operators(
            np.asarray([2.0, 10.0]),
            internal_recycle=1.0,
            return_recycle=0.5,
            waste_fraction=0.1,
            invariant_operator=np.asarray([[1.0, 0.0]]),
            tss_weights=np.asarray([0.0, 1.0]),
            layout=layout,
            clarifier_volume_m3=30.0,
            overflow_tss_closure=4.0,
        )
        self.assertEqual(operators.equality_matrix.shape, (7, layout.state_size))
        np.testing.assert_array_equal(
            operators.equality_matrix[-1, layout.overflow_flow_slice],
            np.asarray([0.0, 1.0]),
        )
        self.assertAlmostEqual(operators.equality_rhs[-1], 0.9 * 4.0)

        projector = PhysicalProjector(
            np.ones(layout.state_size),
            np.ones(7),
            np.ones(layout.inequality_count),
        )
        result = projector.project(
            np.ones(layout.state_size),
            operators.equality_matrix,
            operators.equality_rhs,
            operators.inequality_matrix,
        )
        self.assertTrue(result.accepted)
        overflow_tss = result.state[layout.overflow_flow_slice][1] / 0.9
        self.assertAlmostEqual(overflow_tss, 4.0, places=8)

    def test_reduced_network_row_scales_are_positive_and_dimensioned(self) -> None:
        layout = self._layout()
        rows = 2
        influents = np.asarray([[2.0, 10.0], [3.0, 12.0]])
        internal = np.asarray([1.0, 1.5])
        returned = np.asarray([0.5, 0.7])
        waste = np.asarray([0.1, 0.2])
        states = []
        for row in range(rows):
            operators = build_network_operators(
                influents[row],
                internal_recycle=float(internal[row]),
                return_recycle=float(returned[row]),
                waste_fraction=float(waste[row]),
                invariant_operator=np.asarray([[1.0, 0.0]]),
                tss_weights=np.asarray([0.0, 1.0]),
                layout=layout,
                clarifier_volume_m3=30.0,
            )
            states.append(no_conversion_feasible_state(
                influents[row], operators=operators,
                tss_weights=np.asarray([0.0, 1.0]),
            ))
        scales = fit_network_row_scales(
            np.asarray(states),
            influents,
            internal_recycle=internal,
            return_recycle=returned,
            waste_fraction=waste,
            invariant_operator=np.asarray([[1.0, 0.0]]),
            tss_weights=np.asarray([0.0, 1.0]),
            layout=layout,
            clarifier_volume_m3=30.0,
            minimum_scale=1.0,
        )
        self.assertEqual(scales.equality.shape, (6,))
        self.assertEqual(scales.inequality.shape, (3,))
        self.assertTrue(np.all(scales.equality >= 1.0))
        self.assertTrue(np.all(scales.inequality >= 1.0))

        closure_scales = fit_network_row_scales(
            np.asarray(states),
            influents,
            internal_recycle=internal,
            return_recycle=returned,
            waste_fraction=waste,
            invariant_operator=np.asarray([[1.0, 0.0]]),
            tss_weights=np.asarray([0.0, 1.0]),
            layout=layout,
            clarifier_volume_m3=30.0,
            minimum_scale=1.0,
            overflow_tss_closure=np.asarray([10.0, 12.0]),
        )
        self.assertEqual(closure_scales.equality.shape, (7,))
        self.assertTrue(np.all(closure_scales.equality >= 1.0))


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
        self.assertEqual(result.diagnostics.solver_attempts, 1)
        self.assertFalse(result.diagnostics.fallback_used)
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

    def test_bvls_reconstructs_rank_deficient_active_set_accurately(self) -> None:
        # This deterministic system has two dependent active-constraint
        # columns.  Trust-region bounded least squares can stop with an
        # approximately 1e-7 stationarity defect, whereas BVLS resolves the
        # active set to roundoff accuracy.
        rng = np.random.default_rng(1_204_000)
        variable_count = 12
        equality_count = 4
        active_count = 10
        equality_columns = rng.normal(size=(variable_count, equality_count))
        inequality_columns = rng.normal(size=(variable_count, active_count))
        inequality_columns[:, -1] = (
            inequality_columns[:, 0] + inequality_columns[:, 1]
        )
        inequality_columns[:, -2] = (
            inequality_columns[:, 2]
            + 1.0e-9 * rng.normal(size=variable_count)
        )
        multiplier_matrix = np.column_stack(
            (equality_columns, inequality_columns)
        )
        equality_multiplier = (
            rng.normal(size=equality_count) * 10.0 ** rng.uniform(0.0, 4.0)
        )
        inequality_multiplier = (
            np.abs(rng.normal(size=active_count))
            * 10.0 ** rng.uniform(-2.0, 2.0)
        )
        inequality_multiplier[
            rng.choice(active_count, size=active_count // 3, replace=False)
        ] = 0.0
        displacement = -multiplier_matrix @ np.concatenate(
            (equality_multiplier, inequality_multiplier)
        )
        scaled_equality = equality_columns.T
        scaled_inequality = inequality_columns.T
        inequality_rhs = scaled_inequality @ displacement

        projector = PhysicalProjector(
            np.ones(variable_count),
            np.ones(equality_count),
            np.ones(active_count),
        )
        equality_dual, inequality_dual, reconstructed_count = (
            projector._reconstruct_multipliers(
                displacement,
                scaled_equality,
                scaled_inequality,
                inequality_rhs,
            )
        )
        stationarity = (
            displacement
            + scaled_equality.T @ equality_dual
            + scaled_inequality.T @ inequality_dual
        )

        self.assertLess(
            np.linalg.matrix_rank(multiplier_matrix), multiplier_matrix.shape[1]
        )
        self.assertEqual(reconstructed_count, active_count)
        self.assertTrue(np.all(inequality_dual >= 0.0))
        self.assertLess(float(np.linalg.norm(stationarity, ord=np.inf)), 1.0e-10)

    def test_failed_audits_use_deterministic_cold_fallback_sequence(self) -> None:
        state_scale = np.asarray([2.0, 0.5])
        equality_scale = np.asarray([4.0])
        inequality_scale = np.asarray([3.0])
        raw = np.asarray([-2.0, 2.0])
        equality = np.asarray([[2.0, 8.0]])
        equality_rhs = np.asarray([8.0])
        inequality = np.asarray([[1.5, 0.0]])
        projector = PhysicalProjector(
            state_scale, equality_scale, inequality_scale
        )
        calls: list[dict[str, object]] = []

        def mocked_solve_once(**kwargs: object) -> SimpleNamespace:
            calls.append(dict(kwargs))
            attempt = len(calls)
            displacement = (
                np.asarray([0.0, 0.0])
                if attempt < 3
                else np.asarray([1.0 + 2.0e-9, -2.0])
            )
            return SimpleNamespace(
                x=displacement,
                y=np.zeros(4),
                info=SimpleNamespace(
                    status=f"mock-{attempt}", status_val=attempt, iter=attempt,
                ),
            )

        projector._solve_once = mocked_solve_once  # type: ignore[method-assign]
        result = projector.project(raw, equality, equality_rhs, inequality)

        self.assertTrue(result.accepted)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call["warm_start"] is None for call in calls))
        self.assertNotIn("rho", calls[0])
        self.assertEqual(calls[1]["rho"], 0.01)
        self.assertIs(calls[1]["adaptive_rho"], False)
        self.assertEqual(calls[1]["maximum_iterations"], 200_000)
        self.assertEqual(calls[2]["rho"], 10.0)
        self.assertIs(calls[2]["adaptive_rho"], False)
        self.assertEqual(calls[2]["maximum_iterations"], 200_000)
        self.assertEqual(result.diagnostics.status, "mock-3")
        self.assertEqual(result.diagnostics.solver_attempts, 3)
        self.assertTrue(result.diagnostics.fallback_used)
        self.assertTrue(result.diagnostics.retried_cold)

    def test_finite_failed_candidate_is_returned_for_diagnostic_continuation(self) -> None:
        projector = PhysicalProjector(
            np.ones(2), np.ones(1), np.ones(1)
        )
        attempts = 0

        def mocked_solve_once(**_kwargs: object) -> SimpleNamespace:
            nonlocal attempts
            attempts += 1
            return SimpleNamespace(
                x=np.zeros(2),
                y=np.zeros(4),
                info=SimpleNamespace(
                    status=f"finite-failure-{attempts}",
                    status_val=7,
                    iter=200_000,
                ),
            )

        projector._solve_once = mocked_solve_once  # type: ignore[method-assign]
        result = projector.project(
            np.ones(2),
            np.asarray([[1.0, 0.0]]),
            np.asarray([3.0]),
            np.asarray([[0.0, 1.0]]),
            raise_on_failure=False,
        )

        self.assertFalse(result.accepted)
        self.assertEqual(attempts, 3)
        self.assertTrue(np.all(np.isfinite(result.state)))
        self.assertTrue(np.all(np.isfinite(result.displacement)))
        self.assertEqual(result.diagnostics.solver_attempts, 3)
        self.assertTrue(result.diagnostics.fallback_used)


if __name__ == "__main__":
    unittest.main()
