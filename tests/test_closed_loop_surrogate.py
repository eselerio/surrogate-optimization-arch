from __future__ import annotations

import unittest

import numpy as np

from closed_loop.surrogate import (
    NetworkLayout,
    PhysicalProjector,
    QuadraticFeatureMap,
    QuadraticSurrogate,
    SearchSettings,
    SurrogateValidationError,
    affine_projection,
    build_network_operators,
    deterministic_bounded_search,
    feasibility_first_merit,
    fit_network_row_scales,
    no_conversion_feasible_state,
)


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

    def test_fixed_ols_recovers_an_exact_quadratic_response(self) -> None:
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
        self.assertLess(model.diagnostics.condition_number, 1.0e8)
        self.assertLess(np.max(np.abs(fitted - responses)), 2.0e-11)

    def test_zero_variance_is_rejected_instead_of_silently_rescaled(self) -> None:
        rng = np.random.default_rng(8)
        decisions = rng.normal(size=(400, 5))
        decisions[:, 2] = 1.0
        influent = rng.normal(size=(400, 20))
        with self.assertRaises(SurrogateValidationError):
            QuadraticFeatureMap.fit(decisions, influent)


class PhysicalProjectionTests(unittest.TestCase):
    @staticmethod
    def network_fixture() -> tuple[NetworkLayout, np.ndarray, np.ndarray, np.ndarray]:
        layout = NetworkLayout()
        invariant = np.zeros((5, 20), dtype=np.float64)
        invariant[0, 0] = 1.0
        invariant[1, 1] = 1.0
        invariant[2, 2] = 1.0
        invariant[3, 3] = 1.0
        invariant[4, 4] = 1.0
        tss = np.zeros(20, dtype=np.float64)
        tss[10:] = 1.0
        influent = np.linspace(1.0, 20.0, 20)
        return layout, invariant, tss, influent

    def test_network_operator_order_and_analytical_feasible_point(self) -> None:
        layout, invariant, tss, influent = self.network_fixture()
        operators = build_network_operators(
            influent,
            internal_recycle=2.0,
            return_recycle=0.75,
            waste_fraction=0.02,
            invariant_operator=invariant,
            tss_weights=tss,
            layout=layout,
        )
        state = no_conversion_feasible_state(influent, operators=operators, tss_weights=tss)

        self.assertEqual(layout.state_size, 170)
        self.assertEqual(operators.equality_matrix.shape, (77, 170))
        self.assertEqual(operators.inequality_matrix.shape, (26, 170))
        self.assertLess(
            np.linalg.norm(operators.equality_matrix @ state - operators.equality_rhs, ord=np.inf),
            1.0e-12,
        )
        self.assertLessEqual(np.max(operators.inequality_matrix @ state), 1.0e-12)

    def test_affine_and_full_projection_satisfy_independent_contracts(self) -> None:
        layout, invariant, tss, influent = self.network_fixture()
        operators = build_network_operators(
            influent,
            internal_recycle=1.5,
            return_recycle=0.8,
            waste_fraction=0.015,
            invariant_operator=invariant,
            tss_weights=tss,
            layout=layout,
        )
        feasible = no_conversion_feasible_state(influent, operators=operators, tss_weights=tss)
        rng = np.random.default_rng(18)
        raw = feasible + rng.normal(scale=3.0, size=layout.state_size)
        state_scale = np.maximum(1.0, np.abs(feasible))

        affine = affine_projection(
            raw,
            operators.equality_matrix,
            operators.equality_rhs,
            state_scale,
        )
        self.assertLess(
            np.linalg.norm(
                operators.equality_matrix @ affine - operators.equality_rhs, ord=np.inf
            ),
            2.0e-10,
        )

        projector = PhysicalProjector(
            state_scale=state_scale,
            equality_scale=np.maximum(1.0, np.abs(operators.equality_rhs) + 1.0),
            inequality_scale=np.ones(operators.inequality_matrix.shape[0]),
        )
        result = projector.project(
            raw,
            operators.equality_matrix,
            operators.equality_rhs,
            operators.inequality_matrix,
        )
        self.assertTrue(result.accepted)
        self.assertGreaterEqual(np.min(result.state), -1.0e-8)
        self.assertLessEqual(np.max(operators.inequality_matrix @ result.state), 2.0e-7)
        self.assertLessEqual(result.diagnostics.equality_residual, 1.0e-8)
        self.assertLessEqual(result.diagnostics.stationarity_residual, 1.0e-8)

    def test_term_based_row_scales_are_positive_and_have_declared_sizes(self) -> None:
        layout, invariant, tss, _ = self.network_fixture()
        rng = np.random.default_rng(33)
        states = []
        influents = []
        internal = np.linspace(0.2, 3.8, 12)
        returned = np.linspace(0.3, 1.2, 12)
        wasted = np.linspace(0.002, 0.045, 12)
        for index in range(12):
            influent = rng.uniform(0.5, 100.0, size=20)
            operators = build_network_operators(
                influent,
                internal_recycle=float(internal[index]),
                return_recycle=float(returned[index]),
                waste_fraction=float(wasted[index]),
                invariant_operator=invariant,
                tss_weights=tss,
                layout=layout,
            )
            influents.append(influent)
            states.append(
                no_conversion_feasible_state(influent, operators=operators, tss_weights=tss)
            )
        scales = fit_network_row_scales(
            np.asarray(states),
            np.asarray(influents),
            internal_recycle=internal,
            return_recycle=returned,
            waste_fraction=wasted,
            invariant_operator=invariant,
            tss_weights=tss,
            layout=layout,
        )
        self.assertEqual(scales.equality.shape, (77,))
        self.assertEqual(scales.inequality.shape, (26,))
        self.assertTrue(np.all(scales.equality > 0.0))
        self.assertTrue(np.all(scales.inequality > 0.0))


class BoundedSearchTests(unittest.TestCase):
    @staticmethod
    def settings() -> SearchSettings:
        return SearchSettings(
            total_budget=150,
            full_direct_budget=60,
            face_direct_budget=10,
            direct_resolution=1.0 / 81.0,
            local_seed_count=3,
            initial_mesh=1.0 / 16.0,
            terminal_mesh=1.0 / 256.0,
        )

    def test_merit_separates_feasible_infeasible_and_failed_points(self) -> None:
        feasible = feasibility_first_merit(4.0, 0.0, accepted=True)
        infeasible = feasibility_first_merit(0.0, 0.2, accepted=True)
        failed = feasibility_first_merit(None, None, accepted=False)
        self.assertLess(feasible, 1.0)
        self.assertGreaterEqual(infeasible, 1.0)
        self.assertLess(infeasible, failed)

    def test_search_finds_boundary_basin_and_is_bitwise_deterministic(self) -> None:
        def objective(point: np.ndarray) -> float:
            return float(point[0] ** 2 + (point[1] - 0.73) ** 2)

        first = deterministic_bounded_search(
            objective,
            ((0.0, 1.0), (0.0, 1.0)),
            settings=self.settings(),
        )
        second = deterministic_bounded_search(
            objective,
            ((0.0, 1.0), (0.0, 1.0)),
            settings=self.settings(),
        )

        self.assertEqual(first.evaluations, second.evaluations)
        self.assertEqual(
            tuple(record.normalized_point for record in first.records),
            tuple(record.normalized_point for record in second.records),
        )
        self.assertEqual(first.fun, second.fun)
        self.assertEqual(first.x[0], 0.0)
        self.assertLess(abs(first.x[1] - 0.73), 0.012)
        self.assertLessEqual(first.evaluations, self.settings().total_budget)
        self.assertTrue(any(phase.startswith("face:0:0") for phase in first.phase_counts))


if __name__ == "__main__":
    unittest.main()
