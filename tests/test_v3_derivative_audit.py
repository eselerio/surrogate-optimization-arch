from __future__ import annotations

import json
import unittest

import casadi as ca
import numpy as np

from closed_loop.v3_derivative_audit import (
    DerivativeAuditError,
    DerivativeAuditFailure,
    audit_casadi_nlp_derivatives,
)


class SurrogateLikeDerivativeAuditTests(unittest.TestCase):
    @staticmethod
    def _functions() -> tuple[ca.Function, ca.Function]:
        variable = ca.MX.sym("surrogate_audit_x", 4)
        parameter = ca.MX.sym("surrogate_audit_p", 2)
        control, displacement, equality_dual, inequality_dual = ca.vertsplit(variable)
        raw = parameter[0] + control**2
        projected = raw + displacement
        objective = (projected - parameter[1]) ** 2 + 0.1 * control**3
        constraints = ca.vertcat(
            displacement + control - 0.3,
            equality_dual * control + displacement**2 - parameter[0],
            inequality_dual * projected - 0.2,
        )
        return (
            ca.Function("surrogate_audit_objective", [variable, parameter], [objective]),
            ca.Function("surrogate_audit_constraints", [variable, parameter], [constraints]),
        )

    def test_interior_and_inward_columns_pass_both_declared_steps(self) -> None:
        objective, constraints = self._functions()
        result = audit_casadi_nlp_derivatives(
            objective,
            constraints,
            point=np.asarray([0.4, -0.1, 0.0, 1.0]),
            parameters=np.asarray([0.2, 0.8]),
            lower_bounds=np.asarray([0.0, -np.inf, 0.0, 0.0]),
            upper_bounds=np.asarray([1.0, np.inf, np.inf, 1.0]),
            constraint_multipliers=np.asarray([0.3, -0.2, 0.4]),
            name="surrogate_like_derivative_test",
        )

        self.assertTrue(result.passed, msg=result.as_dict())
        self.assertEqual(result.variable_count, 4)
        self.assertEqual(result.constraint_count, 3)
        self.assertEqual(len(result.columns), 8)
        for step in (1.0e-6, 5.0e-7):
            by_coordinate = {
                item.coordinate: item
                for item in result.columns
                if item.step == step
            }
            self.assertEqual(by_coordinate[0].scheme, "central")
            self.assertEqual(by_coordinate[1].scheme, "central")
            self.assertEqual(by_coordinate[2].scheme, "forward")
            self.assertEqual(by_coordinate[2].direction, 1)
            self.assertEqual(by_coordinate[3].scheme, "backward")
            self.assertEqual(by_coordinate[3].direction, -1)
        self.assertLessEqual(result.maximum_jacobian_discrepancy, 1.0e-5)
        self.assertLessEqual(result.maximum_hessian_discrepancy, 1.0e-5)
        json.dumps(result.as_dict(), allow_nan=False)

    def test_no_feasible_two_step_stencil_fails_explicitly(self) -> None:
        variable = ca.MX.sym("narrow_x", 1)
        objective = ca.Function("narrow_objective", [variable], [variable[0] ** 2])
        constraints = ca.Function("narrow_constraints", [variable], [variable])
        with self.assertRaisesRegex(DerivativeAuditError, "no feasible two-step stencil"):
            audit_casadi_nlp_derivatives(
                objective,
                constraints,
                point=np.asarray([5.0e-7]),
                parameters=np.empty(0),
                lower_bounds=np.asarray([0.0]),
                upper_bounds=np.asarray([1.0e-6]),
                constraint_multipliers=np.asarray([0.0]),
            )


class DirectLikeDerivativeAuditTests(unittest.TestCase):
    @staticmethod
    def _functions() -> tuple[ca.Function, ca.Function]:
        variable = ca.MX.sym("direct_audit_x", 3)
        parameter = ca.MX.sym("direct_audit_p", 1)
        control, state, feed = ca.vertsplit(variable)
        objective = (
            0.4 * ca.sin(control)
            + 0.3 * state**2
            + 0.2 * feed**3
            + 0.1 * control * state
        )
        constraints = ca.vertcat(
            state - ca.exp(control) - parameter[0],
            feed + control * state - 0.5,
            state * feed + control**2 - 0.25,
        )
        return (
            ca.Function("direct_audit_objective", [variable, parameter], [objective]),
            ca.Function("direct_audit_constraints", [variable, parameter], [constraints]),
        )

    def test_nonlinear_direct_like_lagrangian_hessian_passes(self) -> None:
        objective, constraints = self._functions()
        arguments = dict(
            objective_function=objective,
            constraint_function=constraints,
            point=np.asarray([0.3, 1.2, 0.0]),
            parameters=np.asarray([0.1]),
            lower_bounds=np.asarray([0.0, 0.0, 0.0]),
            upper_bounds=np.asarray([1.0, np.inf, np.inf]),
            constraint_multipliers=np.asarray([-0.4, 0.2, 0.1]),
            name="direct_like_derivative_test",
        )
        result = audit_casadi_nlp_derivatives(**arguments)
        self.assertTrue(result.passed, msg=result.as_dict())
        self.assertEqual(result.function_evaluations, 1 + result.unique_perturbation_points)
        self.assertGreater(result.elapsed_seconds, 0.0)

        failed = audit_casadi_nlp_derivatives(**arguments, tolerance=1.0e-14)
        self.assertFalse(failed.passed)
        with self.assertRaises(DerivativeAuditFailure) as captured:
            audit_casadi_nlp_derivatives(
                **arguments,
                tolerance=1.0e-14,
                raise_on_failure=True,
            )
        self.assertIs(captured.exception.result.passed, False)


if __name__ == "__main__":
    unittest.main()
