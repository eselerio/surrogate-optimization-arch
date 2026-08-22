from __future__ import annotations

import json
import unittest

import casadi as ca
import numpy as np

from closed_loop import model as mechanism
from closed_loop import nlp as nlp_module
from closed_loop.nlp import (
    COMBINED_EQUALITY_COUNT,
    COMBINED_INEQUALITY_COUNT,
    COMBINED_VARIABLE_COUNT,
    CaseDefinition,
    CombinedNLPAssets,
    IPOPTSettings,
    NLPStartResult,
    NLPValidationError,
    ObjectiveWeights,
    SmoothingScales,
    SymbolicNLP,
    build_combined_nlp,
    combined_initial_point,
    evaluate_problem,
    evaluate_smooth_process_rates,
    evaluate_symbolic_mechanistic_model,
    evaluate_symbolic_surrogate_prediction,
    fit_mechanistic_residual_scales,
    fit_smoothing_scales,
    ordered_normalized_starts,
    replay_kkt,
    select_best_start,
    smooth_feed_reciprocal,
    solve_nlp_start,
)
from closed_loop.surrogate import (
    LeastSquaresDiagnostics,
    QuadraticFeatureMap,
    QuadraticSurrogate,
)


def _dummy_model(seed: int = 22) -> QuadraticSurrogate:
    rng = np.random.default_rng(seed)
    feature_map = QuadraticFeatureMap(
        decision_center=np.array([21.0, 0.5, 2.0, 0.75, 0.0255]),
        decision_scale=np.array([8.0, 0.25, 1.0, 0.25, 0.012]),
        influent_center=mechanism.NOMINAL_INFLUENT.copy(),
        influent_scale=np.maximum(1.0, mechanism.INFLUENT_UPPER - mechanism.INFLUENT_LOWER),
        term_center=rng.normal(scale=0.1, size=350),
        term_scale=rng.uniform(0.5, 2.0, size=350),
    )
    diagnostics = LeastSquaresDiagnostics(
        sample_count=400,
        feature_count=351,
        response_count=170,
        rank_tolerance=1.0e-12,
        smallest_singular_value=1.0,
        largest_singular_value=2.0,
        condition_number=2.0,
        optimality_residual=0.0,
        coefficient_agreement=0.0,
        acceptance_threshold=1.0e-12,
    )
    response_center = np.full(170, 100.0)
    response_center[160:170] = np.linspace(500.0, 2500.0, 10)
    response_scale = rng.uniform(1.0, 5.0, size=170)
    upper = np.diag(np.linspace(1.0, 3.0, 351))
    pivots = np.arange(350, -1, -1, dtype=np.int64)
    return QuadraticSurrogate(
        feature_map=feature_map,
        response_center=response_center,
        response_scale=response_scale,
        coefficients=rng.normal(scale=0.01, size=(170, 351)),
        diagnostics=diagnostics,
        feature_qr_upper=upper,
        feature_qr_pivots=pivots,
    )


def _smoothing() -> SmoothingScales:
    return SmoothingScales(
        nox=1000.0,
        fermentable_and_acetate=1000.0,
        hydrolysis=1000.0,
        pao=1000.0,
        positive_pp=1.0,
        settling_delta=5000.0,
    )


def _state_center() -> np.ndarray:
    reactors = np.tile(np.maximum(mechanism.NOMINAL_INFLUENT, 2.0), 5)
    layers = np.linspace(500.0, 2500.0, 10)
    return np.concatenate((reactors, layers))


def _combined_assets(model: QuadraticSurrogate) -> CombinedNLPAssets:
    return CombinedNLPAssets(
        model=model,
        fidelity_delta=0.5,
        leverage_max=1.0e9,
        state_center=_state_center(),
        state_scale=np.linspace(1.0, 3.0, 110),
        residual_scale=np.linspace(1.0, 2.0, 110),
        quality_scale=np.array([10.0, 2.0, 1.0, 5.0]),
        inventory_scale=1.0e10,
        smoothing=_smoothing(),
    )


class SymbolicParityTests(unittest.TestCase):
    def test_symbolic_351_feature_prediction_matches_numpy(self) -> None:
        model = _dummy_model()
        decisions = np.array([18.0, 0.63, 1.7, 0.82, 0.018])
        influent = mechanism.NOMINAL_INFLUENT + np.linspace(-0.1, 0.1, 20)
        symbolic_features, symbolic_prediction = evaluate_symbolic_surrogate_prediction(
            model, decisions, influent
        )
        np.testing.assert_allclose(
            symbolic_features,
            model.feature_map.transform(decisions, influent),
            rtol=0.0,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            symbolic_prediction,
            model.predict(decisions, influent),
            rtol=2.0e-14,
            atol=2.0e-12,
        )

    def test_smooth_mechanistic_model_matches_exact_model_away_from_branches(self) -> None:
        rng = np.random.default_rng(33)
        reactors = np.abs(rng.normal(100.0, 20.0, size=100)) + 1.0
        layers = np.linspace(500.0, 2500.0, 10)
        state = np.concatenate((reactors, layers))
        decisions = np.array([18.0, 0.6, 2.0, 0.75, 0.02])
        influent = mechanism.NOMINAL_INFLUENT
        complete, smooth_rhs = evaluate_symbolic_mechanistic_model(
            decisions, influent, state, _smoothing()
        )
        operating = mechanism.OperatingPoint(*decisions)
        exact_rhs = mechanism.coupled_rhs(state, operating, influent)
        exact_complete = mechanism.assemble_target(state, operating, influent)
        np.testing.assert_allclose(complete, exact_complete, rtol=0.0, atol=3.0e-11)
        np.testing.assert_allclose(smooth_rhs, exact_rhs, rtol=1.0e-7, atol=1.0e-5)

    def test_batched_rate_map_matches_exact_rates_away_from_guards(self) -> None:
        states = np.vstack((np.arange(1.0, 21.0) * 3.0, np.arange(2.0, 22.0) * 4.0))
        smooth = evaluate_smooth_process_rates(states, _smoothing())
        exact = np.vstack([mechanism.process_rates(row) for row in states])
        self.assertEqual(smooth.shape, (2, mechanism.N_PROCESSES))
        np.testing.assert_allclose(smooth, exact, rtol=1.0e-8, atol=1.0e-8)


class SmoothingAndScaleTests(unittest.TestCase):
    def test_c2_feed_reciprocal_is_exact_on_feasible_domain(self) -> None:
        np.testing.assert_allclose(
            smooth_feed_reciprocal(np.array([1.0, 2.0, 10.0])),
            np.array([1.0, 0.5, 0.1]),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(smooth_feed_reciprocal(0.0), 3.0)
        value = ca.MX.sym("value")
        reciprocal = smooth_feed_reciprocal(value)
        derivative = ca.jacobian(reciprocal, value)
        curvature = ca.jacobian(derivative, value)
        function = ca.Function("reciprocal_c2_test", [value], [reciprocal, derivative, curvature])
        at_join = [float(item) for item in function(1.0)]
        np.testing.assert_allclose(at_join, [1.0, -1.0, 2.0], rtol=0.0, atol=1.0e-14)

    def test_development_mechanistic_scales_are_batched_and_positive(self) -> None:
        rng = np.random.default_rng(44)
        rows = 4
        targets = np.abs(rng.normal(100.0, 25.0, size=(rows, 170))) + 1.0
        targets[:, 160:170] = np.linspace(500.0, 2500.0, 10)
        decisions = np.tile(np.array([18.0, 0.6, 2.0, 0.75, 0.02]), (rows, 1))
        influent = np.tile(mechanism.NOMINAL_INFLUENT, (rows, 1))
        smoothing = fit_smoothing_scales(targets)
        residual_scale = fit_mechanistic_residual_scales(
            decisions, influent, targets, smoothing
        )
        self.assertEqual(residual_scale.shape, (110,))
        self.assertTrue(np.all(np.isfinite(residual_scale)))
        self.assertTrue(np.all(residual_scale >= 1.0))

    def test_combined_assets_reject_invalid_statistical_limits(self) -> None:
        model = _dummy_model()
        values = dict(
            model=model,
            leverage_max=1.0,
            state_center=_state_center(),
            state_scale=np.ones(110),
            residual_scale=np.ones(110),
            quality_scale=np.ones(4),
            inventory_scale=1.0,
            smoothing=_smoothing(),
        )
        with self.assertRaises(NLPValidationError):
            CombinedNLPAssets(fidelity_delta=0.0, **values)
        with self.assertRaises(NLPValidationError):
            CombinedNLPAssets(fidelity_delta=0.5, **{**values, "leverage_max": np.inf})


class SolverContractTests(unittest.TestCase):
    def test_every_ipopt_option_is_explicit_and_separate_from_kkt_replay(self) -> None:
        self.assertEqual(IPOPTSettings().maximum_iterations, 2500)
        settings = IPOPTSettings(
            primal_tolerance=3.0e-9,
            stationarity_tolerance=4.0e-7,
            dual_tolerance=5.0e-7,
            complementarity_tolerance=6.0e-7,
            physical_nonnegativity_tolerance=7.0e-11,
            tol=8.0e-9,
            constraint_violation_tolerance=9.0e-9,
            dual_infeasibility_tolerance=1.1e-6,
            ipopt_complementarity_tolerance=1.2e-6,
            maximum_iterations=2500,
            bound_relax_factor=0.0,
            linear_solver="MUMPS",
            mu_strategy="ADAPTIVE",
            hessian_approximation="EXACT",
            accepted_return_statuses=(
                "Solved_To_Acceptable_Level",
                "Solve_Succeeded",
            ),
        )
        options = settings.solver_options()
        expected = {
            "ipopt.tol": 8.0e-9,
            "ipopt.constr_viol_tol": 9.0e-9,
            "ipopt.dual_inf_tol": 1.1e-6,
            "ipopt.compl_inf_tol": 1.2e-6,
            "ipopt.max_iter": 2500,
            "ipopt.bound_relax_factor": 0.0,
            "ipopt.linear_solver": "mumps",
            "ipopt.mu_strategy": "adaptive",
            "ipopt.hessian_approximation": "exact",
        }
        for key, value in expected.items():
            self.assertEqual(options[key], value, msg=key)
        self.assertEqual(settings.primal_tolerance, 3.0e-9)
        self.assertEqual(settings.complementarity_tolerance, 6.0e-7)
        self.assertTrue(settings.accepts_return_status("Solve_Succeeded"))
        self.assertTrue(settings.accepts_return_status("Solved_To_Acceptable_Level"))
        self.assertFalse(settings.accepts_return_status("Maximum_Iterations_Exceeded"))

    def test_ipopt_contract_rejects_unfrozen_choices_and_bad_statuses(self) -> None:
        invalid = (
            {"bound_relax_factor": 1.0e-12},
            {"linear_solver": "ma57"},
            {"mu_strategy": "monotone"},
            {"hessian_approximation": "limited-memory"},
            {"maximum_iterations": 2500.0},
            {"accepted_return_statuses": ("Solve_Succeeded",)},
            {"accepted_return_statuses": (
                "Solve_Succeeded", "Maximum_Iterations_Exceeded",
            )},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(NLPValidationError):
                IPOPTSettings(**values)


class StartAndStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = _dummy_model()
        cls.assets = _combined_assets(cls.model)
        cls.problem = build_combined_nlp(cls.assets, compile_solver=False)

    def test_nine_start_order_is_exact(self) -> None:
        starts = ordered_normalized_starts()
        self.assertEqual(starts.shape, (9, 5))
        np.testing.assert_array_equal(starts[0], np.full(5, 0.5))
        np.testing.assert_allclose(
            starts[1],
            [
                0.02663004232826169,
                0.29179325776038795,
                0.881289146262954,
                0.28302834586154596,
                0.2431478444919397,
            ],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            starts[-1],
            [
                0.5735787513199975,
                0.5131255488958242,
                0.11530597273325514,
                0.1035912579996088,
                0.1049531134399749,
            ],
            rtol=0.0,
            atol=0.0,
        )

    def test_nearest_initializer_uses_first_index_tie_and_only_floors_start(self) -> None:
        model = _dummy_model()
        z = np.full(5, 0.5)
        theta = np.array([21.0, 0.5, 2.0, 0.75, 0.0255])
        decisions = np.vstack((theta, theta))
        influents = np.vstack((mechanism.NOMINAL_INFLUENT, mechanism.NOMINAL_INFLUENT))
        targets = np.zeros((2, 170))
        targets[1] = 10.0
        assets = CombinedNLPAssets(
            model=model,
            fidelity_delta=0.5,
            leverage_max=1.0,
            state_center=np.ones(110),
            state_scale=np.full(110, 2.0),
            residual_scale=np.ones(110),
            quality_scale=np.ones(4),
            inventory_scale=1.0e9,
            smoothing=_smoothing(),
        )
        point, index = combined_initial_point(
            z, mechanism.NOMINAL_INFLUENT, decisions, influents, targets, assets
        )
        self.assertEqual(index, 0)
        np.testing.assert_array_equal(point[:5], z)
        physical_y = assets.state_center + assets.state_scale * point[5:]
        np.testing.assert_allclose(
            physical_y,
            np.full(110, 2.0e-8),
            rtol=0.0,
            atol=2.0e-14,
        )
        np.testing.assert_array_equal(targets, np.vstack((np.zeros(170), np.full(170, 10.0))))

    def test_combined_dimensions_objective_and_constraint_order(self) -> None:
        case = CaseDefinition(
            mechanism.NOMINAL_INFLUENT,
            weights=ObjectiveWeights(),
            underflow_tss_limit=15_000.0,
        )
        point = np.concatenate((np.full(5, 0.5), np.zeros(110)))
        evaluated = evaluate_problem(self.problem, point, case)
        self.assertEqual(
            (
                self.problem.variable_count,
                self.problem.equality_count,
                self.problem.inequality_count,
            ),
            (COMBINED_VARIABLE_COUNT, COMBINED_EQUALITY_COUNT, COMBINED_INEQUALITY_COUNT),
        )
        self.assertEqual(evaluated["state"].shape, (110,))
        self.assertEqual(evaluated["complete_state"].shape, (170,))
        self.assertEqual(evaluated["equality"].shape, (110,))
        self.assertEqual(evaluated["inequality"].shape, (9,))
        parameter = case.parameter_vector()
        gradient = np.asarray(
            self.problem.gradient_function(point, parameter), dtype=float
        )
        equality_jacobian = np.asarray(
            self.problem.equality_jacobian_function(point, parameter), dtype=float
        )
        inequality_jacobian = np.asarray(
            self.problem.inequality_jacobian_function(point, parameter), dtype=float
        )
        self.assertEqual(gradient.shape, (115, 1))
        self.assertEqual(equality_jacobian.shape, (110, 115))
        self.assertEqual(inequality_jacobian.shape, (9, 115))
        self.assertTrue(np.all(np.isfinite(gradient)))
        self.assertTrue(np.all(np.isfinite(equality_jacobian)))
        self.assertTrue(np.all(np.isfinite(inequality_jacobian)))

        decisions = evaluated["decisions"]
        state = evaluated["state"]
        complete, raw_residual = evaluate_symbolic_mechanistic_model(
            decisions, case.influent, state, self.assets.smoothing
        )
        np.testing.assert_allclose(evaluated["complete_state"], complete, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            evaluated["equality"], raw_residual / self.assets.residual_scale,
            rtol=2.0e-14, atol=2.0e-12,
        )

        prediction = self.model.predict(decisions, case.influent)
        fidelity = float(np.mean(np.square(
            (complete - prediction) / self.model.response_scale
        )))
        leverage = float(self.model.leverage(decisions, case.influent))
        self.assertAlmostEqual(evaluated["diagnostics"]["fidelity"], fidelity, places=10)
        self.assertAlmostEqual(evaluated["diagnostics"]["leverage"], leverage, places=10)
        self.assertAlmostEqual(
            evaluated["inequality"][-2], fidelity / self.assets.fidelity_delta - 1.0,
            places=10,
        )
        self.assertAlmostEqual(
            evaluated["inequality"][-1],
            (leverage - self.assets.leverage_max) / self.assets.leverage_max,
            places=12,
        )

        final = complete[100:120]
        overflow_mass = complete[120:140]
        underflow_mass = complete[140:160]
        layers = complete[160:170]
        hrt, aeration, internal, returned, waste = decisions
        q_c, q_u, q_e = 1.0 + returned, returned + waste, 1.0 - waste
        feed_tss = float(mechanism.TSS_VECTOR @ final)
        underflow_tss = float(mechanism.TSS_VECTOR @ (underflow_mass / q_u))
        boundary = float(
            mechanism.TSS_VECTOR @ overflow_mass
            + waste * mechanism.TSS_VECTOR @ (underflow_mass / q_u)
        )
        reactor = complete[20:120].reshape(5, 20)
        stage_volume = mechanism.CLARIFIER.fresh_flow * hrt / (24.0 * 5.0)
        inventory = float(
            stage_volume * np.sum(reactor @ mechanism.TSS_VECTOR)
            + mechanism.CLARIFIER.layer_volume * np.sum(layers)
        )
        sor = mechanism.CLARIFIER.fresh_flow * q_e / mechanism.CLARIFIER.area
        slr = (
            mechanism.CLARIFIER.fresh_flow * q_c * feed_tss
            / (1000.0 * mechanism.CLARIFIER.area)
        )
        expected_domain = np.array([1.0 - feed_tss, 1.0 - boundary])
        expected_engineering = np.array([
            (8.0 * mechanism.CLARIFIER.fresh_flow * boundary - inventory)
            / self.assets.inventory_scale,
            (inventory - 30.0 * mechanism.CLARIFIER.fresh_flow * boundary)
            / self.assets.inventory_scale,
            (sor - 20.0) / 20.0,
            (slr - 100.0) / 100.0,
            (underflow_tss - case.underflow_tss_limit) / 15_000.0,
        ])
        np.testing.assert_allclose(evaluated["inequality"][:2], expected_domain)
        np.testing.assert_allclose(evaluated["inequality"][2:7], expected_engineering)

        effluent = overflow_mass / q_e
        composites = mechanism.COMPOSITE_MATRIX @ effluent
        components = np.array([
            np.dot(np.full(4, 0.25) / self.assets.quality_scale, composites),
            (hrt - 12.0) / 18.0,
            aeration,
            (internal - 1.0) / 2.0,
            (returned - 0.5) / 0.5,
            waste * underflow_tss / (0.05 * 15_000.0),
        ])
        expected_objective = float(case.weights.as_array() @ components)
        self.assertAlmostEqual(evaluated["objective"], expected_objective, places=12)
        self.assertAlmostEqual(
            evaluated["objective"],
            evaluated["diagnostics"]["engineering_objective"],
            places=14,
        )

    def test_only_combined_builder_is_public(self) -> None:
        for obsolete in (
            "SurrogateNLPAssets",
            "MechanisticNLPAssets",
            "build_surrogate_nlp",
            "build_mechanistic_nlp",
            "surrogate_initial_point",
            "mechanistic_initial_point",
            "fit_recovery_pair_scales",
            "fit_surrogate_clarifier_residual_scales",
        ):
            self.assertFalse(hasattr(nlp_module, obsolete), msg=obsolete)
            self.assertNotIn(obsolete, nlp_module.__all__)
        self.assertIn("build_combined_nlp", nlp_module.__all__)

    def test_combined_solver_graph_is_finite_at_zero_physical_state_bound(self) -> None:
        case = CaseDefinition(mechanism.NOMINAL_INFLUENT)
        point = self.problem.lower_bounds.copy()
        point[:5] = 0.5
        parameter = case.parameter_vector()
        expressions = (
            self.problem.objective_function(point, parameter),
            self.problem.equality_function(point, parameter),
            self.problem.inequality_function(point, parameter),
            self.problem.gradient_function(point, parameter),
            self.problem.equality_jacobian_function(point, parameter),
            self.problem.inequality_jacobian_function(point, parameter),
        )
        self.assertTrue(all(
            np.all(np.isfinite(np.asarray(value, dtype=float))) for value in expressions
        ))


def _toy_problem() -> SymbolicNLP:
    settings = IPOPTSettings()
    variable = ca.MX.sym("toy_variable", 6)
    parameter = ca.MX.sym("toy_parameter", 27)
    target = ca.DM([0.2, 0.4, 0.3, 0.5, 0.6, 1.0])
    objective = ca.sumsqr(variable - target)
    equality = ca.vertcat(variable[2] - 0.3)
    inequality = ca.vertcat(variable[0] - 0.8)
    constraint = ca.vertcat(equality, inequality)
    solver = ca.nlpsol(
        "toy_ipopt",
        "ipopt",
        {"x": variable, "p": parameter, "f": objective, "g": constraint},
        settings.solver_options(),
    )
    return SymbolicNLP(
        name="toy",
        variable_count=6,
        equality_count=1,
        inequality_count=1,
        state_count=1,
        lower_bounds=np.zeros(6),
        upper_bounds=np.array([1.0, 1.0, 1.0, 1.0, 1.0, np.inf]),
        solver=solver,
        objective_function=ca.Function("toy_objective", [variable, parameter], [objective]),
        equality_function=ca.Function("toy_equality", [variable, parameter], [equality]),
        inequality_function=ca.Function("toy_inequality", [variable, parameter], [inequality]),
        gradient_function=ca.Function(
            "toy_gradient", [variable, parameter], [ca.gradient(objective, variable)]
        ),
        equality_jacobian_function=ca.Function(
            "toy_equality_jacobian", [variable, parameter], [ca.jacobian(equality, variable)]
        ),
        inequality_jacobian_function=ca.Function(
            "toy_inequality_jacobian", [variable, parameter], [ca.jacobian(inequality, variable)]
        ),
        physical_function=ca.Function("toy_physical", [variable, parameter], [variable]),
        complete_state_function=ca.Function(
            "toy_complete_state", [variable, parameter], [variable[5:]]
        ),
        diagnostics_function=ca.Function(
            "toy_diagnostics", [variable, parameter], [ca.vertcat(objective)]
        ),
        diagnostic_names=("objective",),
        physical_scale=np.ones(1),
        settings=settings,
    )


class SolverAndKKTTests(unittest.TestCase):
    def test_cold_ipopt_solve_and_independent_kkt_accept(self) -> None:
        problem = _toy_problem()
        result = solve_nlp_start(
            problem,
            CaseDefinition(mechanism.NOMINAL_INFLUENT),
            np.array([0.7, 0.7, 0.7, 0.7, 0.7, 2.0]),
            start_index=4,
        )
        self.assertTrue(result.accepted, msg=result.error or result.kkt.as_dict())
        np.testing.assert_allclose(
            result.primal, [0.2, 0.4, 0.3, 0.5, 0.6, 1.0], atol=2.0e-7
        )
        self.assertLessEqual(result.kkt.primal_residual, 1.0e-8)
        self.assertLessEqual(result.kkt.stationarity_residual, 1.0e-6)
        json.dumps(result.as_dict())

    def test_lower_only_lam_x_sign_is_checked_and_tie_selection_is_lexicographic(self) -> None:
        problem = _toy_problem()
        case = CaseDefinition(mechanism.NOMINAL_INFLUENT)
        point = np.array([0.2, 0.4, 0.3, 0.5, 0.6, 1.0])
        bad_bound_multiplier = np.zeros(6)
        bad_bound_multiplier[-1] = 1.0
        diagnostics = replay_kkt(
            problem, point, case, np.zeros(1), np.zeros(1), bad_bound_multiplier
        )
        self.assertGreater(diagnostics.dual_feasibility_residual, 0.0)

        base = solve_nlp_start(problem, case, point, start_index=8)
        self.assertTrue(base.accepted)
        result_a = NLPStartResult(
            **{
                **base.__dict__,
                "start_index": 3,
                "normalized_controls": np.array([0.3, 0.4, 0.3, 0.5, 0.6]),
            }
        )
        result_b = NLPStartResult(
            **{
                **base.__dict__,
                "start_index": 7,
                "objective": base.objective + 5.0e-11,
                "normalized_controls": np.array([0.2, 0.4, 0.3, 0.5, 0.6]),
            }
        )
        self.assertIs(select_best_start((result_a, result_b)), result_b)


if __name__ == "__main__":
    unittest.main()
