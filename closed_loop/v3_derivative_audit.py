"""Domain-aware derivative audit required by the manuscript-v3 supplement.

The audit compares CasADi's algorithmic objective/constraint Jacobian and
Lagrangian Hessian columns against second-order finite differences in the
caller's dimensionless internal coordinates.  It uses both declared step
sizes, switches to an inward three-point formula near finite bounds, and does
not alter the supplied point or multipliers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Sequence

import casadi as ca
import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
DEFAULT_STEPS: tuple[float, float] = (1.0e-6, 5.0e-7)
DEFAULT_TOLERANCE = 1.0e-5


class DerivativeAuditError(RuntimeError):
    """Raised when an audit cannot be constructed or evaluated."""


class DerivativeAuditFailure(DerivativeAuditError):
    """Raised on request when a completed derivative audit does not pass."""

    def __init__(self, result: "DerivativeAuditResult") -> None:
        super().__init__(
            "CasADi derivative audit failed: "
            f"Jacobian={result.maximum_jacobian_discrepancy:.3e}, "
            f"Hessian={result.maximum_hessian_discrepancy:.3e}, "
            f"limit={result.tolerance:.3e}."
        )
        self.result = result


def _finite_vector(value: npt.ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {size}.")
    return array.copy()


def _bound_vector(value: npt.ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or np.any(np.isnan(array)):
        raise ValueError(f"{name} must be a length-{size} vector without NaN.")
    return array.copy()


def _json_float(value: float) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def _function_output(function: ca.Function, arguments: Sequence[Any], name: str) -> Any:
    if function.n_out() != 1:
        raise ValueError(f"{name} must have exactly one output.")
    if function.n_in() == 1:
        return function(arguments[0])
    if function.n_in() == 2:
        return function(arguments[0], arguments[1])
    raise ValueError(f"{name} must accept either (variable) or (variable, parameter).")


@dataclass(frozen=True)
class DerivativeColumnAudit:
    step: float
    coordinate: int
    scheme: str
    direction: int
    jacobian_algorithmic_norm: float
    jacobian_finite_difference_norm: float
    jacobian_error_norm: float
    jacobian_discrepancy: float
    hessian_algorithmic_norm: float
    hessian_finite_difference_norm: float
    hessian_error_norm: float
    hessian_discrepancy: float
    finite: bool
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key, item in tuple(value.items()):
            if isinstance(item, float):
                value[key] = _json_float(item)
        return value


@dataclass(frozen=True)
class DerivativeAuditResult:
    variable_count: int
    constraint_count: int
    output_count: int
    step_sizes: tuple[float, ...]
    tolerance: float
    columns: tuple[DerivativeColumnAudit, ...]
    maximum_jacobian_discrepancy: float
    maximum_hessian_discrepancy: float
    finite: bool
    passed: bool
    unique_perturbation_points: int
    function_evaluations: int
    graph_build_seconds: float
    algorithmic_evaluation_seconds: float
    finite_difference_evaluation_seconds: float
    elapsed_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "variable_count": self.variable_count,
            "constraint_count": self.constraint_count,
            "output_count": self.output_count,
            "step_sizes": list(self.step_sizes),
            "tolerance": self.tolerance,
            "columns": [column.as_dict() for column in self.columns],
            "maximum_jacobian_discrepancy": _json_float(
                self.maximum_jacobian_discrepancy
            ),
            "maximum_hessian_discrepancy": _json_float(
                self.maximum_hessian_discrepancy
            ),
            "finite": self.finite,
            "passed": self.passed,
            "unique_perturbation_points": self.unique_perturbation_points,
            "function_evaluations": self.function_evaluations,
            "graph_build_seconds": self.graph_build_seconds,
            "algorithmic_evaluation_seconds": self.algorithmic_evaluation_seconds,
            "finite_difference_evaluation_seconds": (
                self.finite_difference_evaluation_seconds
            ),
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class _Stencil:
    coordinate: int
    step: float
    scheme: str
    direction: int
    first_key: bytes
    second_key: bytes


def _stencil(
    point: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    coordinate: int,
    step: float,
) -> tuple[_Stencil, tuple[FloatArray, FloatArray]]:
    value = point[coordinate]
    central = bool(
        value - 2.0 * step >= lower[coordinate]
        and value + 2.0 * step <= upper[coordinate]
    )
    if central:
        first = point.copy()
        second = point.copy()
        first[coordinate] += step
        second[coordinate] -= step
        return (
            _Stencil(
                coordinate,
                step,
                "central",
                0,
                first.tobytes(),
                second.tobytes(),
            ),
            (first, second),
        )
    positive_available = value + 2.0 * step <= upper[coordinate]
    negative_available = value - 2.0 * step >= lower[coordinate]
    if positive_available:
        direction = 1
        scheme = "forward"
    elif negative_available:
        direction = -1
        scheme = "backward"
    else:
        raise DerivativeAuditError(
            f"coordinate {coordinate} has no feasible two-step stencil at step {step:.3e}."
        )
    first = point.copy()
    second = point.copy()
    first[coordinate] += direction * step
    second[coordinate] += direction * 2.0 * step
    return (
        _Stencil(
            coordinate,
            step,
            scheme,
            direction,
            first.tobytes(),
            second.tobytes(),
        ),
        (first, second),
    )


def audit_casadi_nlp_derivatives(
    objective_function: ca.Function,
    constraint_function: ca.Function,
    point: npt.ArrayLike,
    parameters: npt.ArrayLike,
    lower_bounds: npt.ArrayLike,
    upper_bounds: npt.ArrayLike,
    constraint_multipliers: npt.ArrayLike,
    *,
    step_sizes: Sequence[float] = DEFAULT_STEPS,
    tolerance: float = DEFAULT_TOLERANCE,
    name: str = "v3_derivative_audit",
    raise_on_failure: bool = False,
) -> DerivativeAuditResult:
    """Audit algorithmic first and second derivatives at one NLP point.

    ``objective_function`` and ``constraint_function`` must each accept either
    ``(variable)`` or ``(variable, parameter)`` and have one output.  The
    Lagrangian is ``objective + constraint_multipliers.T @ constraints``.
    Bound multipliers need not be supplied because bound rows are linear and
    therefore contribute zero to the Hessian.
    """

    total_started = perf_counter()
    if not isinstance(objective_function, ca.Function) or not isinstance(
        constraint_function, ca.Function
    ):
        raise TypeError("objective_function and constraint_function must be CasADi Functions.")
    variable_count = int(objective_function.numel_in(0))
    if variable_count < 1 or int(constraint_function.numel_in(0)) != variable_count:
        raise ValueError("objective and constraint variable dimensions must match and be nonzero.")
    variable = _finite_vector(point, variable_count, "point")
    lower = _bound_vector(lower_bounds, variable_count, "lower_bounds")
    upper = _bound_vector(upper_bounds, variable_count, "upper_bounds")
    if np.any(lower > upper):
        raise ValueError("lower_bounds cannot exceed upper_bounds.")
    if np.any(variable < lower) or np.any(variable > upper):
        raise ValueError("the audit point must lie within its declared bounds.")

    parameter_count = 0
    for function, function_name in (
        (objective_function, "objective_function"),
        (constraint_function, "constraint_function"),
    ):
        if function.n_in() not in (1, 2):
            raise ValueError(
                f"{function_name} must accept either one or two inputs."
            )
        if function.n_in() == 2:
            count = int(function.numel_in(1))
            if parameter_count not in (0, count):
                raise ValueError("objective and constraint parameter dimensions differ.")
            parameter_count = count
    parameter = _finite_vector(parameters, parameter_count, "parameters")
    steps = tuple(float(item) for item in step_sizes)
    if (
        len(steps) != 2
        or any(not np.isfinite(item) or item <= 0.0 for item in steps)
        or steps[0] != 1.0e-6
        or steps[1] != 5.0e-7
    ):
        raise ValueError("step_sizes must be exactly (1e-6, 5e-7).")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")

    safe_name = "".join(character if character.isalnum() else "_" for character in name)
    safe_name = safe_name or "v3_derivative_audit"
    graph_started = perf_counter()
    x_symbol = ca.MX.sym(f"{safe_name}_x", variable_count)
    p_symbol = ca.MX.sym(f"{safe_name}_p", parameter_count)
    objective = ca.vec(
        _function_output(objective_function, (x_symbol, p_symbol), "objective_function")
    )
    if objective.numel() != 1:
        raise ValueError("objective_function output must be scalar.")
    constraints = ca.vec(
        _function_output(constraint_function, (x_symbol, p_symbol), "constraint_function")
    )
    constraint_count = int(constraints.numel())
    multipliers = _finite_vector(
        constraint_multipliers,
        constraint_count,
        "constraint_multipliers",
    )
    multiplier_symbol = ca.MX.sym(f"{safe_name}_lambda", constraint_count)
    audit_stack = ca.vertcat(objective, constraints)
    lagrangian = objective[0] + ca.dot(multiplier_symbol, constraints)
    lagrangian_gradient = ca.gradient(lagrangian, x_symbol)
    algorithmic = ca.Function(
        f"{safe_name}_algorithmic",
        [x_symbol, p_symbol, multiplier_symbol],
        [
            ca.jacobian(audit_stack, x_symbol),
            ca.jacobian(lagrangian_gradient, x_symbol),
            audit_stack,
            lagrangian_gradient,
        ],
    )
    numerical = ca.Function(
        f"{safe_name}_numerical",
        [x_symbol, p_symbol, multiplier_symbol],
        [audit_stack, lagrangian_gradient],
    )
    graph_seconds = perf_counter() - graph_started

    algorithmic_started = perf_counter()
    algorithmic_values = algorithmic(variable, parameter, multipliers)
    jacobian = np.asarray(algorithmic_values[0], dtype=np.float64).reshape(
        1 + constraint_count, variable_count
    )
    hessian = np.asarray(algorithmic_values[1], dtype=np.float64).reshape(
        variable_count, variable_count
    )
    base_stack = np.asarray(algorithmic_values[2], dtype=np.float64).reshape(-1)
    base_gradient = np.asarray(algorithmic_values[3], dtype=np.float64).reshape(-1)
    algorithmic_seconds = perf_counter() - algorithmic_started

    stencils: list[_Stencil] = []
    perturbations: dict[bytes, FloatArray] = {}
    for step in steps:
        for coordinate in range(variable_count):
            stencil, points = _stencil(variable, lower, upper, coordinate, step)
            stencils.append(stencil)
            perturbations[points[0].tobytes()] = points[0]
            perturbations[points[1].tobytes()] = points[1]

    numerical_started = perf_counter()
    keys = tuple(perturbations)
    points = np.column_stack([perturbations[key] for key in keys])
    mapped = numerical.map(len(keys), "serial")
    parameter_matrix = np.repeat(parameter[:, None], len(keys), axis=1)
    multiplier_matrix = np.repeat(multipliers[:, None], len(keys), axis=1)
    mapped_values = mapped(points, parameter_matrix, multiplier_matrix)
    stack_values = np.asarray(mapped_values[0], dtype=np.float64).reshape(
        1 + constraint_count, len(keys), order="F"
    )
    gradient_values = np.asarray(mapped_values[1], dtype=np.float64).reshape(
        variable_count, len(keys), order="F"
    )
    lookup = {
        key: (stack_values[:, index], gradient_values[:, index])
        for index, key in enumerate(keys)
    }
    numerical_seconds = perf_counter() - numerical_started

    columns: list[DerivativeColumnAudit] = []
    for stencil in stencils:
        first_stack, first_gradient = lookup[stencil.first_key]
        second_stack, second_gradient = lookup[stencil.second_key]
        if stencil.scheme == "central":
            finite_jacobian = (first_stack - second_stack) / (2.0 * stencil.step)
            finite_hessian = (first_gradient - second_gradient) / (2.0 * stencil.step)
        else:
            denominator = 2.0 * stencil.direction * stencil.step
            finite_jacobian = (
                -3.0 * base_stack + 4.0 * first_stack - second_stack
            ) / denominator
            finite_hessian = (
                -3.0 * base_gradient + 4.0 * first_gradient - second_gradient
            ) / denominator
        algorithmic_jacobian = jacobian[:, stencil.coordinate]
        algorithmic_hessian = hessian[:, stencil.coordinate]
        jacobian_algorithmic_norm = float(
            np.linalg.norm(algorithmic_jacobian, ord=np.inf)
        )
        jacobian_finite_norm = float(
            np.linalg.norm(finite_jacobian, ord=np.inf)
        )
        jacobian_error = float(
            np.linalg.norm(finite_jacobian - algorithmic_jacobian, ord=np.inf)
        )
        jacobian_discrepancy = jacobian_error / (1.0 + jacobian_algorithmic_norm)
        hessian_algorithmic_norm = float(
            np.linalg.norm(algorithmic_hessian, ord=np.inf)
        )
        hessian_finite_norm = float(np.linalg.norm(finite_hessian, ord=np.inf))
        hessian_error = float(
            np.linalg.norm(finite_hessian - algorithmic_hessian, ord=np.inf)
        )
        hessian_discrepancy = hessian_error / (1.0 + hessian_algorithmic_norm)
        finite = bool(
            np.all(np.isfinite(finite_jacobian))
            and np.all(np.isfinite(finite_hessian))
            and np.all(np.isfinite(algorithmic_jacobian))
            and np.all(np.isfinite(algorithmic_hessian))
        )
        columns.append(
            DerivativeColumnAudit(
                step=stencil.step,
                coordinate=stencil.coordinate,
                scheme=stencil.scheme,
                direction=stencil.direction,
                jacobian_algorithmic_norm=jacobian_algorithmic_norm,
                jacobian_finite_difference_norm=jacobian_finite_norm,
                jacobian_error_norm=jacobian_error,
                jacobian_discrepancy=jacobian_discrepancy,
                hessian_algorithmic_norm=hessian_algorithmic_norm,
                hessian_finite_difference_norm=hessian_finite_norm,
                hessian_error_norm=hessian_error,
                hessian_discrepancy=hessian_discrepancy,
                finite=finite,
                passed=bool(
                    finite
                    and jacobian_discrepancy <= tolerance
                    and hessian_discrepancy <= tolerance
                ),
            )
        )
    maximum_jacobian = float(
        np.max([item.jacobian_discrepancy for item in columns], initial=0.0)
    )
    maximum_hessian = float(
        np.max([item.hessian_discrepancy for item in columns], initial=0.0)
    )
    finite = bool(all(item.finite for item in columns))
    passed = bool(finite and all(item.passed for item in columns))
    result = DerivativeAuditResult(
        variable_count=variable_count,
        constraint_count=constraint_count,
        output_count=1 + constraint_count,
        step_sizes=steps,
        tolerance=float(tolerance),
        columns=tuple(columns),
        maximum_jacobian_discrepancy=maximum_jacobian,
        maximum_hessian_discrepancy=maximum_hessian,
        finite=finite,
        passed=passed,
        unique_perturbation_points=len(keys),
        function_evaluations=1 + len(keys),
        graph_build_seconds=graph_seconds,
        algorithmic_evaluation_seconds=algorithmic_seconds,
        finite_difference_evaluation_seconds=numerical_seconds,
        elapsed_seconds=perf_counter() - total_started,
    )
    if raise_on_failure and not result.passed:
        raise DerivativeAuditFailure(result)
    return result


__all__ = [
    "DEFAULT_STEPS",
    "DEFAULT_TOLERANCE",
    "DerivativeAuditError",
    "DerivativeAuditFailure",
    "DerivativeAuditResult",
    "DerivativeColumnAudit",
    "audit_casadi_nlp_derivatives",
]
