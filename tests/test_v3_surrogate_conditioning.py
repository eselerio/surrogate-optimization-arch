from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import casadi as ca
import numpy as np

import closed_loop.v3_surrogate_nlp as surrogate_nlp
from closed_loop.projection import NetworkLayout, build_network_operators
from closed_loop.v3_surrogate_nlp import (
    GAP_CONTINUATION,
    EngineeringLimits,
    NamedTrustRows,
    SurrogateSolverSettings,
    TrustDiagnosticCallbacks,
    TrustThresholds,
    symbolic_network_operators,
)


class _RecordingSolver:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **arguments: object) -> dict[str, ca.DM]:
        self.calls.append(arguments)
        return {
            "x": ca.DM(np.asarray(arguments["x0"], dtype=float)),
            "lam_x": ca.DM([1.0, 2.0, 3.0]),
            "lam_g": ca.DM([-4.0, 5.0]),
        }

    @staticmethod
    def stats() -> dict[str, object]:
        return {"success": True, "return_status": "Solve_Succeeded", "iter_count": 3}


class _Case:
    @staticmethod
    def parameter_vector(_assets: object) -> np.ndarray:
        return np.asarray([42.0])


class SurrogateConditioningTests(unittest.TestCase):
    def test_declared_schedule_has_finer_final_gap_stages(self) -> None:
        self.assertEqual(
            GAP_CONTINUATION,
            (1.0e-2, 1.0e-4, 1.0e-6, 3.0e-7, 1.0e-7, 3.0e-8, 1.0e-8),
        )

    def test_all_trust_rows_are_relative_dimensionless_residuals(self) -> None:
        def rows(first: float, second: float):
            return lambda *_arguments: ca.vertcat(first, second)

        assets = SimpleNamespace(
            layout=SimpleNamespace(state_size=2),
            leverage_precision=np.eye(2),
            trust_thresholds=TrustThresholds(
                correction_rms=2.0,
                regularized_leverage=4.0,
                split_rms=3.0,
                reactor_rms=5.0,
            ),
            trust_callbacks=TrustDiagnosticCallbacks(
                split_rows=rows(3.0, 0.0),
                reactor_rows=rows(10.0, 0.0),
                additional=(NamedTrustRows("extra", rows(22.0, 0.0), 11.0),),
            ),
        )
        constraints, names, values = surrogate_nlp._trust_expressions(
            ca.DM.zeros(7),
            ca.DM.zeros(2),
            ca.DM.zeros(2),
            ca.DM([2.0, 0.0]),
            ca.DM.zeros(2),
            ca.DM([2.0, 0.0]),
            assets,
        )
        self.assertEqual(
            names, ("correction", "leverage", "split", "reactor", "extra")
        )
        np.testing.assert_allclose(
            np.asarray(values).reshape(-1),
            [2.0, 4.0, 4.5, 50.0, 242.0],
        )
        np.testing.assert_allclose(
            np.asarray(constraints).reshape(-1),
            [-0.5, 0.0, -0.5, 1.0, 1.0],
        )
        self.assertEqual(surrogate_nlp._normalized_limit_residual(0.25, 0.0), 0.25)

    def test_engineering_rows_use_only_positive_fixed_scales(self) -> None:
        layout = NetworkLayout(
            stage_count=1,
            component_count=2,
            layer_count=3,
            soluble_indices=(0,),
            particulate_indices=(1,),
        )
        limits = EngineeringLimits(
            fresh_flow_m3_d=100.0,
            clarifier_area_m2=10.0,
            clarifier_volume_m3=30.0,
            srt_lower_d=2.0,
            srt_upper_d=5.0,
            external_loss_min_g_m3=2.0,
            slr_upper_kg_m2_d=100.0,
            underflow_tss_upper_g_m3=1_000.0,
            feed_tss_min_g_m3=10.0,
            sor_upper_m_d=20.0,
            inventory_scale=100_000.0,
        )
        assets = SimpleNamespace(
            layout=layout,
            engineering=limits,
            tss_weights=np.asarray([0.0, 1.0]),
        )
        theta = ca.DM([24.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.1])
        state = np.zeros(layout.state_size)
        state[layout.reactor_slice(0)] = [0.0, 100.0]
        state[layout.overflow_flow_slice] = [0.0, 10.0]
        state[layout.underflow_flow_slice] = [0.0, 200.0]
        state[layout.inventory_index] = 3_000.0
        constraints, names, quantities = surrogate_nlp._engineering_expressions(
            theta, ca.DM(state), assets
        )

        external_loss = 10.0 + 0.1 * 200.0 / 0.6
        inventory = 100.0 * 100.0 + 10.0 * 300.0
        expected = np.asarray(
            [
                (2.0 * 100.0 * external_loss - inventory) / 100_000.0,
                (inventory - 5.0 * 100.0 * external_loss) / 100_000.0,
                (2.0 - external_loss) / 2.0,
                (1.5 - 100.0) / 100.0,
                (200.0 / 0.6 - 1_000.0) / 1_000.0,
                (10.0 - 100.0) / 10.0,
                (9.0 - 20.0) / 20.0,
            ]
        )
        self.assertEqual(
            names,
            (
                "srt_lower",
                "srt_upper",
                "external_solids_loss_guard",
                "slr_upper",
                "underflow_tss_upper",
                "feed_tss_lower",
                "sor_upper",
            ),
        )
        np.testing.assert_allclose(np.asarray(constraints).reshape(-1), expected)
        self.assertTrue(np.all(np.isfinite(np.asarray(quantities))))

    def test_symbolic_and_numeric_reduced_network_operators_match(self) -> None:
        layout = NetworkLayout(
            stage_count=1,
            component_count=2,
            layer_count=3,
            soluble_indices=(0,),
            particulate_indices=(1,),
        )
        invariant = np.asarray([[1.0, 0.0]])
        tss = np.asarray([0.0, 1.0])
        assets = SimpleNamespace(
            layout=layout,
            invariant_operator=invariant,
            tss_weights=tss,
            equality_count=6,
            engineering=SimpleNamespace(clarifier_volume_m3=30.0),
        )
        theta_symbol = ca.MX.sym("network_theta", 7)
        feed_symbol = ca.MX.sym("network_feed", 2)
        symbolic = symbolic_network_operators(theta_symbol, feed_symbol, assets)
        evaluate = ca.Function(
            "reduced_network_operator_test",
            [theta_symbol, feed_symbol],
            [symbolic.equality_matrix, symbolic.equality_rhs, symbolic.inequality_matrix],
        )
        theta = np.asarray([24.0, 0.2, 0.3, 0.4, 1.0, 0.5, 0.1])
        feed = np.asarray([2.0, 10.0])
        symbolic_values = evaluate(theta, feed)
        numeric = build_network_operators(
            feed,
            internal_recycle=theta[4],
            return_recycle=theta[5],
            waste_fraction=theta[6],
            invariant_operator=invariant,
            tss_weights=tss,
            layout=layout,
            clarifier_volume_m3=30.0,
        )
        np.testing.assert_allclose(symbolic_values[0], numeric.equality_matrix)
        np.testing.assert_allclose(
            np.asarray(symbolic_values[1]).reshape(-1), numeric.equality_rhs
        )
        np.testing.assert_allclose(symbolic_values[2], numeric.inequality_matrix)

    def test_ipopt_options_and_stage_call_propagate_outer_duals(self) -> None:
        options = SurrogateSolverSettings().ipopt_options()
        self.assertEqual(options["ipopt.warm_start_init_point"], "yes")
        for name in (
            "ipopt.warm_start_bound_push",
            "ipopt.warm_start_bound_frac",
            "ipopt.warm_start_slack_bound_push",
            "ipopt.warm_start_slack_bound_frac",
            "ipopt.warm_start_mult_bound_push",
        ):
            self.assertEqual(options[name], 1.0e-9)

        solver = _RecordingSolver()
        problem = SimpleNamespace(
            solver=solver,
            assets=object(),
            variable_count=3,
            tau=1.0e-4,
            lower_bounds=np.full(3, -np.inf),
            upper_bounds=np.full(3, np.inf),
            constraint_lower_bounds=np.asarray([0.0, -np.inf]),
            constraint_upper_bounds=np.asarray([0.0, 0.0]),
        )
        evaluation = {
            "objective": 0.0,
            "equality": np.asarray([0.0]),
            "inequality": np.asarray([-1.0]),
            "normalized_gap": 5.0e-5,
        }
        with patch.object(
            surrogate_nlp, "evaluate_surrogate_problem", return_value=evaluation
        ):
            first = surrogate_nlp._solve_continuation_stage(
                problem, _Case(), np.zeros(3), SurrogateSolverSettings()
            )
            second = surrogate_nlp._solve_continuation_stage(
                problem,
                _Case(),
                first.stage.primal,
                SurrogateSolverSettings(),
                (first.bound_multipliers, first.constraint_multipliers),
            )

        self.assertNotIn("lam_x0", solver.calls[0])
        self.assertNotIn("lam_g0", solver.calls[0])
        np.testing.assert_array_equal(solver.calls[1]["lam_x0"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(solver.calls[1]["lam_g0"], [-4.0, 5.0])
        self.assertTrue(first.stage.feasible)
        self.assertTrue(second.stage.feasible)
        np.testing.assert_array_equal(
            first.stage.constraint_multipliers, [-4.0, 5.0]
        )
        restored = surrogate_nlp.ContinuationStageRecord.from_dict(
            first.stage.as_dict()
        )
        np.testing.assert_array_equal(
            restored.constraint_multipliers, [-4.0, 5.0]
        )


if __name__ == "__main__":
    unittest.main()
