"""The simultaneous physics-constrained statistical nonlinear program.

The complete plant response is reconstructed from 110 mechanistic state
coordinates inside one NLP.  Its smooth balances enforce physical steady
state, while the frozen statistical model supplies fidelity and development-
support constraints.  Model construction, cold-start solution, and the
independent KKT replay remain separate so solver status cannot substitute for
the stated acceptance tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from time import perf_counter_ns
from typing import Any, Iterable, Sequence

import casadi as ca
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .design import DECISION_BOUNDS, DECISION_COLUMNS, unit_latin_hypercube
from .model import (
    CLARIFIER,
    COMPONENT_INDEX,
    COMPOSITE_MATRIX,
    N_COMPONENTS,
    N_LAYERS,
    N_PROCESSES,
    N_STAGES,
    PARAMETERS,
    PARTICULATE,
    SOLUBLE,
    STOICHIOMETRIC_MATRIX,
    TSS_VECTOR,
)
from .surrogate import NetworkLayout, QuadraticSurrogate


FloatArray = NDArray[np.float64]

DECISION_LOWER = np.asarray([DECISION_BOUNDS[name][0] for name in DECISION_COLUMNS], dtype=float)
DECISION_UPPER = np.asarray([DECISION_BOUNDS[name][1] for name in DECISION_COLUMNS], dtype=float)
DECISION_SPAN = DECISION_UPPER - DECISION_LOWER

COMBINED_VARIABLE_COUNT = 115
COMBINED_EQUALITY_COUNT = 110
COMBINED_INEQUALITY_COUNT = 9
CASE_PARAMETER_COUNT = 27
ACCEPTED_STATUSES = frozenset(("Solve_Succeeded", "Solved_To_Acceptable_Level"))


class NLPValidationError(ValueError):
    """Raised when frozen NLP inputs do not satisfy the manuscript contract."""


def _vector(value: ArrayLike, size: int, name: str, *, positive: bool = False) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise NLPValidationError(f"{name} must be a finite vector of length {size}.")
    if positive and np.any(result <= 0.0):
        raise NLPValidationError(f"{name} must be strictly positive.")
    return result.copy()


@dataclass(frozen=True)
class ObjectiveWeights:
    """The six ordered coefficients in the engineering objective."""

    quality: float = 0.50
    hrt: float = 0.15
    aeration: float = 0.20
    internal_recycle: float = 0.05
    return_recycle: float = 0.05
    wasted_solids: float = 0.05

    def __post_init__(self) -> None:
        values = self.as_array()
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise NLPValidationError("objective weights must be finite and positive.")
        if not np.isclose(np.sum(values), 1.0, rtol=0.0, atol=1.0e-12):
            raise NLPValidationError("objective weights must sum to one.")

    def as_array(self) -> FloatArray:
        return np.asarray(
            [self.quality, self.hrt, self.aeration, self.internal_recycle,
             self.return_recycle, self.wasted_solids],
            dtype=np.float64,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "quality": self.quality, "hrt": self.hrt, "aeration": self.aeration,
            "internal_recycle": self.internal_recycle,
            "return_recycle": self.return_recycle,
            "wasted_solids": self.wasted_solids,
        }


@dataclass(frozen=True)
class CaseDefinition:
    """Case-specific influent, scalarization weights, and underflow limit."""

    influent: FloatArray
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    underflow_tss_limit: float = 15_000.0
    case_id: str = "nominal"

    def __post_init__(self) -> None:
        object.__setattr__(self, "influent", _vector(self.influent, N_COMPONENTS, "influent"))
        if np.any(self.influent < 0.0):
            raise NLPValidationError("influent concentrations must be nonnegative.")
        if not np.isfinite(self.underflow_tss_limit) or self.underflow_tss_limit <= 0.0:
            raise NLPValidationError("underflow_tss_limit must be positive and finite.")
        if not self.case_id:
            raise NLPValidationError("case_id must not be empty.")

    def parameter_vector(self) -> FloatArray:
        return np.concatenate(
            (self.influent, self.weights.as_array(), [float(self.underflow_tss_limit)])
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "influent": self.influent.tolist(),
            "weights": self.weights.as_dict(),
            "underflow_tss_limit": float(self.underflow_tss_limit),
        }


@dataclass(frozen=True)
class SmoothingScales:
    """Frozen development scales used by every smooth branch primitive."""

    nox: float
    fermentable_and_acetate: float
    hydrolysis: float
    pao: float
    positive_pp: float
    settling_delta: float
    epsilon: float = 1.0e-8
    velocity: float = 474.0
    flux: float = 474.0 * 15_000.0
    receiver_half_width: float = 1.0

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.nox, self.fermentable_and_acetate, self.hydrolysis, self.pao,
             self.positive_pp, self.settling_delta, self.epsilon,
             self.velocity, self.flux, self.receiver_half_width], dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise NLPValidationError("all smoothing scales must be finite and positive.")


@dataclass(frozen=True)
class CombinedNLPAssets:
    """Frozen development-only objects required by the combined NLP."""

    model: QuadraticSurrogate
    fidelity_delta: float
    leverage_max: float
    state_center: FloatArray
    state_scale: FloatArray
    residual_scale: FloatArray
    quality_scale: FloatArray
    inventory_scale: float
    smoothing: SmoothingScales

    def __post_init__(self) -> None:
        if self.model.feature_map.decision_count != 5 or self.model.feature_map.influent_count != 20:
            raise NLPValidationError("the NLP requires the fixed 5-decision, 20-influent feature map.")
        if self.model.response_count != 170 or self.model.feature_map.feature_count != 351:
            raise NLPValidationError("the NLP requires a 351-feature, 170-response surrogate.")
        object.__setattr__(self, "state_center", _vector(self.state_center, 110, "state_center"))
        raw_scale = _vector(self.state_scale, 110, "state_scale", positive=True)
        if np.any(raw_scale < 1.0):
            raise NLPValidationError("state_scale must already equal max(1, development scale).")
        object.__setattr__(self, "state_scale", raw_scale)
        object.__setattr__(self, "residual_scale", _vector(
            self.residual_scale, 110, "residual_scale", positive=True,
        ))
        object.__setattr__(self, "quality_scale", _vector(
            self.quality_scale, 4, "quality_scale", positive=True,
        ))
        if not np.isfinite(self.fidelity_delta) or not 0.0 < self.fidelity_delta <= 1.0:
            raise NLPValidationError("fidelity_delta must lie in (0, 1].")
        if not np.isfinite(self.leverage_max) or self.leverage_max <= 0.0:
            raise NLPValidationError("leverage_max must be positive and finite.")
        if not np.isfinite(self.inventory_scale) or self.inventory_scale <= 0.0:
            raise NLPValidationError("inventory_scale must be positive and finite.")
        upper = self.model.feature_qr_upper
        pivots = self.model.feature_qr_pivots
        if upper.shape != (351, 351) or np.any(~np.isfinite(upper)):
            raise NLPValidationError("the frozen surrogate QR factor must be finite and 351 by 351.")
        if np.any(np.diag(upper) == 0.0) or np.any(np.tril(upper, k=-1) != 0.0):
            raise NLPValidationError("the frozen surrogate QR factor must be nonsingular and upper triangular.")
        if pivots.shape != (351,) or not np.array_equal(np.sort(pivots), np.arange(351)):
            raise NLPValidationError("the frozen surrogate QR pivots must permute 0,...,350.")


@dataclass(frozen=True)
class IPOPTSettings:
    """Explicit IPOPT options and separate independent-KKT tolerances."""

    # Independent replay acceptance tolerances.
    primal_tolerance: float = 1.0e-8
    stationarity_tolerance: float = 1.0e-6
    dual_tolerance: float = 1.0e-6
    complementarity_tolerance: float = 1.0e-6
    physical_nonnegativity_tolerance: float = 1.0e-10

    # IPOPT convergence and algorithm options.  These are deliberately
    # distinct from the replay tolerances above, even when a configuration
    # supplies the same numerical value for both contracts.
    tol: float = 1.0e-8
    constraint_violation_tolerance: float = 1.0e-8
    dual_infeasibility_tolerance: float = 1.0e-6
    ipopt_complementarity_tolerance: float = 1.0e-6
    maximum_iterations: int = 2500
    bound_relax_factor: float = 0.0
    linear_solver: str = "mumps"
    mu_strategy: str = "adaptive"
    hessian_approximation: str = "exact"
    accepted_return_statuses: tuple[str, ...] = (
        "Solve_Succeeded",
        "Solved_To_Acceptable_Level",
    )

    def __post_init__(self) -> None:
        replay_tolerances = np.asarray(
            [self.primal_tolerance, self.stationarity_tolerance, self.dual_tolerance,
             self.complementarity_tolerance, self.physical_nonnegativity_tolerance],
            dtype=float,
        )
        solver_tolerances = np.asarray(
            [self.tol, self.constraint_violation_tolerance,
             self.dual_infeasibility_tolerance,
             self.ipopt_complementarity_tolerance],
            dtype=float,
        )
        if not np.all(np.isfinite(replay_tolerances)) or np.any(replay_tolerances <= 0.0):
            raise NLPValidationError("independent KKT tolerances must be finite and positive.")
        if not np.all(np.isfinite(solver_tolerances)) or np.any(solver_tolerances <= 0.0):
            raise NLPValidationError("IPOPT convergence tolerances must be finite and positive.")
        if (
            isinstance(self.maximum_iterations, (bool, np.bool_))
            or not isinstance(self.maximum_iterations, (int, np.integer))
            or self.maximum_iterations <= 0
        ):
            raise NLPValidationError("maximum_iterations must be a positive integer.")
        if not np.isfinite(self.bound_relax_factor) or self.bound_relax_factor != 0.0:
            raise NLPValidationError("bound_relax_factor must equal the fixed value zero.")

        fixed_choices = (
            ("linear_solver", self.linear_solver, "mumps"),
            ("mu_strategy", self.mu_strategy, "adaptive"),
            ("hessian_approximation", self.hessian_approximation, "exact"),
        )
        for name, value, required in fixed_choices:
            if not isinstance(value, str) or value.casefold() != required:
                raise NLPValidationError(f"{name} must equal {required!r}.")
            object.__setattr__(self, name, required)

        if isinstance(self.accepted_return_statuses, (str, bytes)):
            raise NLPValidationError("accepted_return_statuses must be a sequence of statuses.")
        statuses = tuple(self.accepted_return_statuses)
        if (
            any(not isinstance(status, str) or not status for status in statuses)
            or len(statuses) != len(set(statuses))
            or frozenset(statuses) != ACCEPTED_STATUSES
        ):
            raise NLPValidationError(
                "accepted_return_statuses must contain exactly the two frozen IPOPT statuses."
            )
        object.__setattr__(self, "accepted_return_statuses", statuses)

    def accepts_return_status(self, status: str) -> bool:
        """Return whether an IPOPT status belongs to the configured contract."""

        return status in self.accepted_return_statuses

    def solver_options(self) -> dict[str, Any]:
        return {
            "print_time": False,
            "ipopt.linear_solver": self.linear_solver,
            "ipopt.mu_strategy": self.mu_strategy,
            "ipopt.hessian_approximation": self.hessian_approximation,
            "ipopt.tol": float(self.tol),
            "ipopt.constr_viol_tol": float(self.constraint_violation_tolerance),
            "ipopt.dual_inf_tol": float(self.dual_infeasibility_tolerance),
            "ipopt.compl_inf_tol": float(self.ipopt_complementarity_tolerance),
            "ipopt.max_iter": int(self.maximum_iterations),
            "ipopt.bound_relax_factor": float(self.bound_relax_factor),
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
        }


@dataclass(frozen=True)
class KKTDiagnostics:
    bound_violation: float
    primal_residual: float
    stationarity_residual: float
    dual_feasibility_residual: float
    complementarity_residual: float
    physical_nonnegativity_residual: float
    finite: bool

    def accepted(self, settings: IPOPTSettings) -> bool:
        return bool(
            self.finite
            and self.primal_residual <= settings.primal_tolerance
            and self.physical_nonnegativity_residual <= settings.physical_nonnegativity_tolerance
            and self.stationarity_residual <= settings.stationarity_tolerance
            and self.dual_feasibility_residual <= settings.dual_tolerance
            and self.complementarity_residual <= settings.complementarity_tolerance
        )

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "bound_violation": self.bound_violation,
            "primal_residual": self.primal_residual,
            "stationarity_residual": self.stationarity_residual,
            "dual_feasibility_residual": self.dual_feasibility_residual,
            "complementarity_residual": self.complementarity_residual,
            "physical_nonnegativity_residual": self.physical_nonnegativity_residual,
            "finite": self.finite,
        }


@dataclass(frozen=True)
class NLPStartResult:
    start_index: int
    status: str
    solver_success: bool
    accepted: bool
    objective: float
    primal: FloatArray
    equality_multipliers: FloatArray
    inequality_multipliers: FloatArray
    bound_multipliers: FloatArray
    equality: FloatArray
    inequality: FloatArray
    normalized_controls: FloatArray
    decisions: FloatArray
    state: FloatArray
    diagnostics: dict[str, float]
    kkt: KKTDiagnostics
    elapsed_seconds: float
    iterations: int
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "status": self.status,
            "solver_success": self.solver_success,
            "accepted": self.accepted,
            "objective": self.objective,
            "primal": self.primal.tolist(),
            "equality_multipliers": self.equality_multipliers.tolist(),
            "inequality_multipliers": self.inequality_multipliers.tolist(),
            "bound_multipliers": self.bound_multipliers.tolist(),
            "equality": self.equality.tolist(),
            "inequality": self.inequality.tolist(),
            "normalized_controls": self.normalized_controls.tolist(),
            "decisions": self.decisions.tolist(),
            "state": self.state.tolist(),
            "diagnostics": dict(self.diagnostics),
            "kkt": self.kkt.as_dict(),
            "elapsed_seconds": self.elapsed_seconds,
            "iterations": self.iterations,
            "error": self.error,
        }


@dataclass
class SymbolicNLP:
    """Compiled CasADi expressions and immutable bounds for the NLP."""

    name: str
    variable_count: int
    equality_count: int
    inequality_count: int
    state_count: int
    lower_bounds: FloatArray
    upper_bounds: FloatArray
    solver: ca.Function | None
    objective_function: ca.Function
    equality_function: ca.Function
    inequality_function: ca.Function
    gradient_function: ca.Function
    equality_jacobian_function: ca.Function
    inequality_jacobian_function: ca.Function
    physical_function: ca.Function
    complete_state_function: ca.Function
    diagnostics_function: ca.Function
    diagnostic_names: tuple[str, ...]
    physical_scale: FloatArray
    settings: IPOPTSettings

    @property
    def constraint_lower_bounds(self) -> FloatArray:
        return np.concatenate((np.zeros(self.equality_count), np.full(self.inequality_count, -np.inf)))

    @property
    def constraint_upper_bounds(self) -> FloatArray:
        return np.zeros(self.equality_count + self.inequality_count)


def ordered_normalized_starts() -> FloatArray:
    """Return the exact nine normalized control starts in manuscript order."""

    lhs, final_state, draw_count = unit_latin_hypercube(8, 5, seed=271828)
    if final_state != 6503384058600783867 or draw_count != 75:
        raise RuntimeError("the fixed nine-start generator replay did not match its contract.")
    return np.vstack((np.full((1, 5), 0.5, dtype=np.float64), lhs))


def nearest_development_index(
    normalized_controls: ArrayLike,
    influent: ArrayLike,
    development_decisions: ArrayLike,
    development_influent: ArrayLike,
    model: QuadraticSurrogate,
) -> int:
    """Find the nearest frozen development input; NumPy argmin gives the index tie rule."""

    z = _vector(normalized_controls, 5, "normalized_controls")
    if np.any(z < 0.0) or np.any(z > 1.0):
        raise NLPValidationError("normalized_controls must lie in [0, 1].")
    x = _vector(influent, 20, "influent")
    decisions = np.asarray(development_decisions, dtype=np.float64)
    influents = np.asarray(development_influent, dtype=np.float64)
    if decisions.ndim != 2 or decisions.shape[1] != 5 or influents.shape != (decisions.shape[0], 20):
        raise NLPValidationError("development input blocks have inconsistent shapes.")
    if decisions.shape[0] == 0 or not np.all(np.isfinite(decisions)) or not np.all(np.isfinite(influents)):
        raise NLPValidationError("development input blocks must be finite and nonempty.")
    theta = DECISION_LOWER + DECISION_SPAN * z
    fm = model.feature_map
    query = np.concatenate(((theta - fm.decision_center) / fm.decision_scale,
                            (x - fm.influent_center) / fm.influent_scale))
    rows = np.concatenate(
        ((decisions - fm.decision_center) / fm.decision_scale,
         (influents - fm.influent_center) / fm.influent_scale), axis=1,
    )
    distance = np.sum(np.square(rows - query[None, :]), axis=1)
    return int(np.argmin(distance))


def combined_initial_point(
    normalized_controls: ArrayLike,
    influent: ArrayLike,
    development_decisions: ArrayLike,
    development_influent: ArrayLike,
    development_targets: ArrayLike,
    assets: CombinedNLPAssets,
) -> tuple[FloatArray, int]:
    """Build a combined-NLP start from the nearest development target.

    Nearest-neighbor distances use the frozen 25-dimensional surrogate input
    standardization.  The coordinate floor is applied only to this returned
    initialization and never alters the stored development response.
    """

    z = _vector(normalized_controls, 5, "normalized_controls")
    targets = np.asarray(development_targets, dtype=np.float64)
    decisions = np.asarray(development_decisions, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 170 or not np.all(np.isfinite(targets)):
        raise NLPValidationError("development_targets must be a finite matrix with 170 columns.")
    if targets.shape[0] != (decisions.shape[0] if decisions.ndim == 2 else -1):
        raise NLPValidationError("development targets and inputs must contain the same rows.")
    index = nearest_development_index(
        z, influent, decisions, development_influent, assets.model,
    )
    physical = np.concatenate((targets[index, 20:120], targets[index, 160:170]))
    floor = 1.0e-8 * np.maximum(1.0, assets.state_scale)
    physical = np.maximum(physical, floor)
    scaled = (physical - assets.state_center) / assets.state_scale
    return np.concatenate((z, scaled)), index


def _is_casadi(value: Any) -> bool:
    return isinstance(value, (ca.SX, ca.MX, ca.DM))


def smooth_maximum(a: Any, b: Any, scale: float, epsilon: float = 1.0e-8) -> Any:
    root = ca.sqrt((a - b) ** 2 + (epsilon * scale) ** 2) if (_is_casadi(a) or _is_casadi(b)) else np.sqrt((a - b) ** 2 + (epsilon * scale) ** 2)
    return 0.5 * (a + b + root)


def smooth_minimum(a: Any, b: Any, scale: float, epsilon: float = 1.0e-8) -> Any:
    root = ca.sqrt((a - b) ** 2 + (epsilon * scale) ** 2) if (_is_casadi(a) or _is_casadi(b)) else np.sqrt((a - b) ** 2 + (epsilon * scale) ** 2)
    return 0.5 * (a + b - root)


def smooth_division(a: Any, b: Any, scale: float, epsilon: float = 1.0e-8) -> Any:
    return a * b / (b * b + (epsilon * scale) ** 2)


def smooth_positive_part(value: Any, scale: float, epsilon: float = 1.0e-8) -> Any:
    root = ca.sqrt(value * value + (epsilon * scale) ** 2) if _is_casadi(value) else np.sqrt(value * value + (epsilon * scale) ** 2)
    return 0.5 * (value + root)


def receiver_transition(solids: Any, *, threshold: float = 3000.0, half_width: float = 1.0) -> Any:
    xi = (solids - (threshold - half_width)) / (2.0 * half_width)
    polynomial = 6.0 * xi**5 - 15.0 * xi**4 + 10.0 * xi**3
    if _is_casadi(solids):
        return ca.if_else(solids <= threshold - half_width, 0.0,
                          ca.if_else(solids >= threshold + half_width, 1.0, polynomial))
    return np.where(np.asarray(solids) <= threshold - half_width, 0.0,
                    np.where(np.asarray(solids) >= threshold + half_width, 1.0, polynomial))


def smooth_feed_reciprocal(feed_tss: Any, reference: float = 1.0) -> Any:
    """C2 reciprocal extension that equals ``1/feed_tss`` on the feasible domain."""

    normalized = feed_tss / reference
    extension = (normalized**2 - 3.0 * normalized + 3.0) / reference
    if _is_casadi(feed_tss):
        reciprocal = 1.0 / feed_tss
        return ca.if_else(feed_tss < reference, extension, reciprocal)
    values = np.asarray(feed_tss)
    reciprocal = np.divide(1.0, values, out=np.zeros_like(values, dtype=float), where=values != 0.0)
    result = np.where(values < reference, extension, reciprocal)
    return float(result) if values.ndim == 0 else result


def _casadi_feature_vector(model: QuadraticSurrogate, theta: ca.MX, influent: ca.MX) -> ca.MX:
    fm = model.feature_map
    d = (theta - ca.DM(fm.decision_center)) / ca.DM(fm.decision_scale)
    x = (influent - ca.DM(fm.influent_center)) / ca.DM(fm.influent_scale)
    terms: list[ca.MX] = [d, x]
    terms.extend(d[j] * d[k] for j in range(5) for k in range(j, 5))
    terms.extend(x[j] * x[k] for j in range(20) for k in range(j, 20))
    terms.extend(d[j] * x[k] for j in range(5) for k in range(20))
    unscaled = ca.vertcat(*terms)
    standardized = (unscaled - ca.DM(fm.term_center)) / ca.DM(fm.term_scale)
    return ca.vertcat(1.0, standardized)


def symbolic_surrogate_prediction(model: QuadraticSurrogate, theta: ca.MX, influent: ca.MX) -> tuple[ca.MX, ca.MX]:
    """Return the symbolic 351-vector and raw 170-coordinate prediction."""

    phi = _casadi_feature_vector(model, theta, influent)
    standardized = ca.DM(model.coefficients) @ phi
    raw = ca.DM(model.response_center) + ca.DM(model.response_scale) * standardized
    return phi, raw


def _theta(z: ca.MX) -> ca.MX:
    return ca.DM(DECISION_LOWER) + ca.DM(DECISION_SPAN) * z


def _smooth_rates(c: ca.MX, scales: SmoothingScales) -> ca.MX:
    """ASM2d-TSN rates with only the manuscript-declared smooth guards."""

    p, ix, eps = PARAMETERS, COMPONENT_INDEX, scales.epsilon
    so, sf, sa = c[ix["S_O"]], c[ix["S_F"]], c[ix["S_A"]]
    snh4, sno2, sno3 = c[ix["S_NH4"]], c[ix["S_NO2"]], c[ix["S_NO3"]]
    spo4, salk = c[ix["S_PO4"]], c[ix["S_ALK"]]
    xs, xh, xpao = c[ix["X_S"]], c[ix["X_H"]], c[ix["X_PAO"]]
    xpp, xpha = c[ix["X_PP"]], c[ix["X_PHA"]]
    xaob, xnob = c[ix["X_AOB"]], c[ix["X_NOB"]]
    xmep, xmeoh = c[ix["X_MeP"]], c[ix["X_MeOH"]]
    snox, carbon = sno2 + sno3, sf + sa
    alpha2 = smooth_division(sno2, snox, scales.nox, eps)
    alpha3 = smooth_division(sno3, snox, scales.nox, eps)
    alpha_f = smooth_division(sf, carbon, scales.fermentable_and_acetate, eps)
    alpha_a = smooth_division(sa, carbon, scales.fermentable_and_acetate, eps)
    theta_x = smooth_division(xs, p["K_X"] * xh + xs, scales.hydrolysis, eps)
    r_pp = smooth_division(xpp, xpao, scales.pao, eps)
    r_pha = smooth_division(xpha, xpao, scales.pao, eps)

    def monod(value: Any, half: float) -> Any:
        return value / (half + value)

    def inhibit(value: Any, half: float) -> Any:
        return half / (half + value)

    pi_pp, pi_pha = monod(r_pp, p["K_PP"]), monod(r_pha, p["K_PHA"])
    capacity = smooth_positive_part(p["K_max"] - r_pp, scales.positive_pp, eps)
    c_pp = capacity / (p["K_IPP"] + capacity)
    lh = monod(snh4, p["K_NH4_H"]) * monod(spo4, p["K_PO4_H"]) * monod(salk, p["K_ALK_H"])
    lp = monod(snh4, p["K_NH4_PAO"]) * monod(spo4, p["K_PO4_PAO"]) * monod(salk, p["K_ALK_PAO"])
    ln = monod(spo4, p["K_PO4_nit"]) * monod(salk, p["K_ALK_nit"])
    mo_hyd, io_hyd = monod(so, p["K_O_hyd"]), inhibit(so, p["K_O_hyd"])
    mo_h, io_h = monod(so, p["K_O_H"]), inhibit(so, p["K_O_H"])
    mo_p, io_p = monod(so, p["K_O_PAO"]), inhibit(so, p["K_O_PAO"])
    alk_p = monod(salk, p["K_ALK_PAO"])
    common_pp = p["q_PP"] * monod(spo4, p["K_PS"]) * alk_p * pi_pha * c_pp
    rates = [
        p["K_H"] * mo_hyd * theta_x * xh,
        p["eta_hyd_NO2"] * p["K_H"] * io_hyd * monod(sno2, p["K_NO2_hyd"]) * alpha2 * theta_x * xh,
        p["eta_hyd_NO3"] * p["K_H"] * io_hyd * monod(sno3, p["K_NO3_hyd"]) * alpha3 * theta_x * xh,
        p["eta_hyd_fe"] * p["K_H"] * io_hyd * inhibit(snox, p["K_NOx_hyd"]) * theta_x * xh,
        p["mu_H"] * mo_h * monod(sf, p["K_F"]) * alpha_f * lh * xh,
        p["mu_H"] * mo_h * monod(sa, p["K_A"]) * alpha_a * lh * xh,
        p["mu_H"] * io_h * monod(sf, p["K_F"]) * alpha_f * lh * p["eta_H_NO3"] * monod(sno3, p["K_NO3_H"]) * alpha3 * xh,
        p["mu_H"] * io_h * monod(sf, p["K_F"]) * alpha_f * lh * p["eta_H_NO2"] * monod(sno2, p["K_NO2_H"]) * alpha2 * xh,
        p["mu_H"] * io_h * monod(sa, p["K_A"]) * alpha_a * lh * p["eta_H_NO3"] * monod(sno3, p["K_NO3_H"]) * alpha3 * xh,
        p["mu_H"] * io_h * monod(sa, p["K_A"]) * alpha_a * lh * p["eta_H_NO2"] * monod(sno2, p["K_NO2_H"]) * alpha2 * xh,
        p["q_fe"] * io_h * inhibit(snox, p["K_NOx_H"]) * monod(sf, p["K_fe"]) * monod(salk, p["K_ALK_H"]) * xh,
        p["b_H"] * xh,
        p["q_PHA"] * monod(sa, p["K_A"]) * io_p * inhibit(snox, p["K_NOx_PAO"]) * alk_p * pi_pp * xpao,
        common_pp * mo_p * xpao,
        common_pp * io_p * p["eta_PAO_NO3"] * monod(sno3, p["K_NO3_PAO"]) * alpha3 * xpao,
        common_pp * io_p * p["eta_PAO_NO2"] * monod(sno2, p["K_NO2_PAO"]) * alpha2 * xpao,
        p["mu_PAO"] * mo_p * lp * pi_pha * xpao,
        p["mu_PAO"] * io_p * lp * pi_pha * p["eta_PAO_NO3"] * monod(sno3, p["K_NO3_PAO"]) * alpha3 * xpao,
        p["mu_PAO"] * io_p * lp * pi_pha * p["eta_PAO_NO2"] * monod(sno2, p["K_NO2_PAO"]) * alpha2 * xpao,
        p["b_PAO"] * alk_p * xpao,
        p["b_PP"] * alk_p * xpp,
        p["b_PHA"] * alk_p * xpha,
        p["mu_AOB"] * monod(so, p["K_O_AOB"]) * monod(snh4, p["K_NH4_AOB"]) * ln * xaob,
        p["mu_NOB"] * monod(so, p["K_O_NOB"]) * monod(sno2, p["K_NO2_NOB"]) * ln * xnob,
        p["b_AOB"] * xaob,
        p["b_NOB"] * xnob,
        p["k_PRE"] * spo4 * xmeoh,
        p["k_RED"] * p["i_PMeP"] * monod(salk, p["K_ALK_chem"]) * xmep,
    ]
    return ca.vertcat(*rates)


def _smooth_clarifier_fluxes(
    layers: ca.MX, feed_tss: ca.MX, theta: ca.MX, scales: SmoothingScales,
) -> ca.MX:
    eps = scales.epsilon
    minimum_solids = CLARIFIER.nonsettleable_fraction * feed_tss
    gravity: list[ca.MX] = []
    for layer in range(N_LAYERS):
        delta = smooth_positive_part(layers[layer] - minimum_solids, scales.settling_delta, eps)
        raw = CLARIFIER.theoretical_settling_velocity * (
            ca.exp(-CLARIFIER.hindered_coefficient * delta)
            - ca.exp(-CLARIFIER.low_concentration_coefficient * delta)
        )
        bounded = smooth_maximum(
            0.0,
            smooth_minimum(CLARIFIER.maximum_settling_velocity, raw, scales.velocity, eps),
            scales.velocity,
            eps,
        )
        gravity.append(layers[layer] * bounded)
    r_r, waste = theta[3], theta[4]
    q_c, q_u, q_e = 1.0 + r_r, r_r + waste, 1.0 - waste
    v_e = CLARIFIER.fresh_flow * q_e / CLARIFIER.area
    v_u = CLARIFIER.fresh_flow * q_u / CLARIFIER.area
    flux: list[ca.MX] = [-v_e * layers[0]]
    for upper in range(N_LAYERS - 1):
        lower = upper + 1
        receiver_weight = receiver_transition(
            layers[lower], threshold=CLARIFIER.flux_threshold,
            half_width=scales.receiver_half_width,
        )
        limited = (1.0 - receiver_weight) * gravity[upper] + receiver_weight * smooth_minimum(
            gravity[upper], gravity[lower], scales.flux, eps
        )
        if upper < CLARIFIER.feed_layer:
            flux.append(-v_e * layers[lower] + limited)
        else:
            flux.append(v_u * layers[upper] + limited)
    flux.append(v_u * layers[-1])
    return ca.vertcat(*flux)


def _smooth_clarifier_rhs(
    layers: ca.MX, feed_tss: ca.MX, theta: ca.MX, scales: SmoothingScales,
) -> ca.MX:
    flux = _smooth_clarifier_fluxes(layers, feed_tss, theta, scales)
    q_c = 1.0 + theta[3]
    rhs = [CLARIFIER.area * (flux[layer] - flux[layer + 1]) / CLARIFIER.layer_volume
           for layer in range(N_LAYERS)]
    rhs[CLARIFIER.feed_layer] += CLARIFIER.fresh_flow * q_c * feed_tss / CLARIFIER.layer_volume
    return ca.vertcat(*rhs)


@lru_cache(maxsize=32)
def _smooth_rate_function(scales: SmoothingScales) -> ca.Function:
    symbol = ca.MX.sym("rate_state", 20)
    return ca.Function("smooth_rates_kernel", [symbol], [_smooth_rates(symbol, scales)])


def evaluate_smooth_process_rates(states: ArrayLike, scales: SmoothingScales) -> FloatArray:
    """Evaluate smooth rates for one state or an n-by-20 batch using ``Function.map``."""

    values = np.asarray(states, dtype=np.float64)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 20 or not np.all(np.isfinite(values)):
        raise NLPValidationError("states must have shape (20,) or (n, 20) and be finite.")
    function = _smooth_rate_function(scales)
    mapped = function if values.shape[0] == 1 else function.map(values.shape[0], "serial")
    result = np.asarray(mapped(values.T), dtype=float).T
    return result[0] if single else result


@lru_cache(maxsize=32)
def _smooth_clarifier_flux_function(scales: SmoothingScales) -> ca.Function:
    layer_symbol = ca.MX.sym("flux_layers", 10)
    theta_symbol = ca.MX.sym("flux_theta", 5)
    feed_symbol = ca.MX.sym("flux_feed")
    return ca.Function(
        "smooth_clarifier_flux_kernel", [layer_symbol, feed_symbol, theta_symbol],
        [_smooth_clarifier_fluxes(layer_symbol, feed_symbol, theta_symbol, scales)],
    )


def evaluate_smooth_clarifier_fluxes(
    layers: ArrayLike, feed_tss: ArrayLike | float, decisions: ArrayLike,
    scales: SmoothingScales,
) -> FloatArray:
    """Evaluate the 11 smooth interface fluxes for one row or aligned batches."""

    layer_values = np.asarray(layers, dtype=np.float64)
    decision_values = np.asarray(decisions, dtype=np.float64)
    feed_values = np.asarray(feed_tss, dtype=np.float64)
    single = layer_values.ndim == 1
    if single:
        layer_values = layer_values[None, :]
        decision_values = decision_values[None, :]
        feed_values = feed_values.reshape(1)
    if (
        layer_values.ndim != 2 or layer_values.shape[1] != 10
        or decision_values.shape != (layer_values.shape[0], 5)
        or feed_values.shape != (layer_values.shape[0],)
        or not np.all(np.isfinite(layer_values))
        or not np.all(np.isfinite(decision_values))
        or not np.all(np.isfinite(feed_values))
    ):
        raise NLPValidationError("Clarifier inputs must be aligned finite (n,10), (n,), and (n,5) arrays.")
    function = _smooth_clarifier_flux_function(scales)
    mapped = function if layer_values.shape[0] == 1 else function.map(layer_values.shape[0], "serial")
    result = np.asarray(mapped(layer_values.T, feed_values.reshape(1, -1), decision_values.T), dtype=float).T
    return result[0] if single else result


def evaluate_smooth_clarifier_rhs(
    layers: ArrayLike, feed_tss: float, decisions: ArrayLike, scales: SmoothingScales,
) -> FloatArray:
    """Numerically evaluate the exact symbolic smooth Clarifier balance."""

    flux = evaluate_smooth_clarifier_fluxes(layers, feed_tss, decisions, scales)
    theta = _vector(decisions, 5, "decisions")
    rhs = CLARIFIER.area * (flux[:-1] - flux[1:]) / CLARIFIER.layer_volume
    rhs[CLARIFIER.feed_layer] += (
        CLARIFIER.fresh_flow * (1.0 + theta[3]) * float(feed_tss) / CLARIFIER.layer_volume
    )
    return rhs


def fit_smoothing_scales(development_targets: ArrayLike) -> SmoothingScales:
    """Fit every named smooth-guard scale from complete development targets."""

    targets = np.asarray(development_targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 170 or targets.shape[0] == 0 or not np.all(np.isfinite(targets)):
        raise NLPValidationError("development_targets must be a nonempty finite n-by-170 matrix.")
    reactors = targets[:, 20:120].reshape(-1, N_COMPONENTS)
    ix = COMPONENT_INDEX
    nox = reactors[:, ix["S_NO2"]] + reactors[:, ix["S_NO3"]]
    fermentable = reactors[:, ix["S_F"]] + reactors[:, ix["S_A"]]
    hydrolysis = PARAMETERS["K_X"] * reactors[:, ix["X_H"]] + reactors[:, ix["X_S"]]
    pao = reactors[:, ix["X_PAO"]]
    r_pp = np.divide(
        reactors[:, ix["X_PP"]], pao,
        out=np.zeros(reactors.shape[0], dtype=float), where=pao != 0.0,
    )
    positive_pp = PARAMETERS["K_max"] - r_pp
    final = targets[:, 100:120]
    feed_tss = final @ TSS_VECTOR
    layers = targets[:, 160:170]
    settling_delta = layers - CLARIFIER.nonsettleable_fraction * feed_tss[:, None]

    def scale(values: FloatArray) -> float:
        return max(1.0, float(np.max(np.abs(values))))

    return SmoothingScales(
        nox=scale(nox),
        fermentable_and_acetate=scale(fermentable),
        hydrolysis=scale(hydrolysis),
        pao=scale(pao),
        positive_pp=scale(positive_pp),
        settling_delta=scale(settling_delta),
    )


def fit_mechanistic_residual_scales(
    development_decisions: ArrayLike,
    development_influent: ArrayLike,
    development_targets: ArrayLike,
    smoothing: SmoothingScales,
) -> FloatArray:
    """Fit D_f from the manuscript's structurally nonzero signed physical terms."""

    decisions = np.asarray(development_decisions, dtype=np.float64)
    influent = np.asarray(development_influent, dtype=np.float64)
    targets = np.asarray(development_targets, dtype=np.float64)
    rows = decisions.shape[0] if decisions.ndim == 2 else 0
    if decisions.shape != (rows, 5) or influent.shape != (rows, 20) or targets.shape != (rows, 170) or rows == 0:
        raise NLPValidationError("development decisions, influents, and targets have inconsistent shapes.")
    if not all(np.all(np.isfinite(block)) for block in (decisions, influent, targets)):
        raise NLPValidationError("development blocks must be finite.")
    reactors = targets[:, 20:120].reshape(rows, N_STAGES, N_COMPONENTS)
    layers = targets[:, 160:170]
    final = reactors[:, -1, :]
    feed_tss = final @ TSS_VECTOR
    fractions = final[:, PARTICULATE] * smooth_feed_reciprocal(feed_tss)[:, None]
    underflow = final.copy()
    underflow[:, PARTICULATE] = layers[:, -1, None] * fractions
    mixer = (
        influent + decisions[:, 2, None] * final + decisions[:, 3, None] * underflow
    ) / (1.0 + decisions[:, 2] + decisions[:, 3])[:, None]
    dilution = 120.0 * (1.0 + decisions[:, 2] + decisions[:, 3]) / decisions[:, 0]
    stoich_nonzero = STOICHIOMETRIC_MATRIX != 0.0
    reaction_count = np.sum(stoich_nonzero, axis=0).astype(float)
    reactor_scales = np.empty((N_STAGES, N_COMPONENTS), dtype=float)
    upstream = mixer
    for stage in range(N_STAGES):
        current = reactors[:, stage, :]
        hydraulic = dilution[:, None] * (upstream - current)
        rates = evaluate_smooth_process_rates(current, smoothing)
        reaction_terms = rates[:, :, None] * STOICHIOMETRIC_MATRIX[None, :, :]
        square_sum = np.square(hydraulic) + np.sum(np.square(reaction_terms), axis=1)
        count = 1.0 + reaction_count
        if stage >= 2:
            oxygen = 47.0 * decisions[:, 1] * (8.5 - current[:, COMPONENT_INDEX["S_O"]])
            square_sum[:, COMPONENT_INDEX["S_O"]] += np.square(oxygen)
            count = count.copy()
            count[COMPONENT_INDEX["S_O"]] += 1.0
        reactor_scales[stage] = np.maximum(1.0, np.sqrt(np.mean(square_sum / count[None, :], axis=0)))
        upstream = current
    flux = evaluate_smooth_clarifier_fluxes(layers, feed_tss, decisions, smoothing)
    factor = CLARIFIER.area / CLARIFIER.layer_volume
    layer_square_sum = np.square(factor * flux[:, :-1]) + np.square(factor * flux[:, 1:])
    layer_count = np.full(10, 2.0)
    feed_term = CLARIFIER.fresh_flow * (1.0 + decisions[:, 3]) * feed_tss / CLARIFIER.layer_volume
    layer_square_sum[:, CLARIFIER.feed_layer] += np.square(feed_term)
    layer_count[CLARIFIER.feed_layer] += 1.0
    layer_scales = np.maximum(1.0, np.sqrt(np.mean(layer_square_sum / layer_count[None, :], axis=0)))
    return np.concatenate((reactor_scales.reshape(-1), layer_scales))


def extract_mechanistic_states(development_targets: ArrayLike) -> FloatArray:
    """Extract ``(c1,...,c5,s1,...,s10)`` from complete 170-coordinate targets."""

    targets = np.asarray(development_targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 170 or not np.all(np.isfinite(targets)):
        raise NLPValidationError("development_targets must be a finite matrix with 170 columns.")
    return np.concatenate((targets[:, 20:120], targets[:, 160:170]), axis=1)


def fit_mechanistic_state_scaling(development_targets: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return development means and ``max(1,population standard deviation)`` scales."""

    states = extract_mechanistic_states(development_targets)
    return np.mean(states, axis=0), np.maximum(1.0, np.std(states, axis=0, ddof=0))


def fit_inventory_scale(development_decisions: ArrayLike, development_targets: ArrayLike) -> float:
    """Fit S_M from complete development targets using the cross-multiplied SRT terms."""

    decisions = np.asarray(development_decisions, dtype=np.float64)
    targets = np.asarray(development_targets, dtype=np.float64)
    if decisions.ndim != 2 or decisions.shape[1] != 5 or targets.shape != (decisions.shape[0], 170):
        raise NLPValidationError("development decisions and targets have inconsistent shapes.")
    reactor = targets[:, 20:120].reshape(-1, N_STAGES, N_COMPONENTS)
    layers = targets[:, 160:170]
    g_e, g_u = targets[:, 120:140], targets[:, 140:160]
    q_u = decisions[:, 3] + decisions[:, 4]
    c_u = g_u / q_u[:, None]
    boundary = g_e @ TSS_VECTOR + decisions[:, 4] * (c_u @ TSS_VECTOR)
    stage_volume = CLARIFIER.fresh_flow * decisions[:, 0] / (24.0 * N_STAGES)
    inventory = stage_volume * np.sum(reactor @ TSS_VECTOR, axis=1)
    inventory += CLARIFIER.layer_volume * np.sum(layers, axis=1)
    return max(1.0, float(np.max(np.maximum(inventory, 30.0 * CLARIFIER.fresh_flow * boundary))))


def fit_quality_scale(
    development_decisions: ArrayLike,
    development_targets: ArrayLike,
    *,
    variance_relative_tolerance: float = 1.0e-12,
) -> FloatArray:
    """Fit population standard deviations of overflow COD, TN, TP, and TSS."""

    decisions = np.asarray(development_decisions, dtype=np.float64)
    targets = np.asarray(development_targets, dtype=np.float64)
    if decisions.ndim != 2 or decisions.shape[1] != 5 or targets.shape != (decisions.shape[0], 170):
        raise NLPValidationError("development decisions and targets have inconsistent shapes.")
    effluent = targets[:, 120:140] / (1.0 - decisions[:, 4])[:, None]
    if targets.shape[0] < 2:
        raise NLPValidationError("quality scaling requires at least two development rows.")
    composites = effluent @ COMPOSITE_MATRIX.T
    scale = np.std(composites, axis=0, ddof=0)
    reference = np.maximum(1.0, np.max(np.abs(composites), axis=0))
    if (
        not np.isfinite(variance_relative_tolerance)
        or variance_relative_tolerance <= 0.0
        or np.any(~np.isfinite(scale))
        or np.any(scale <= variance_relative_tolerance * reference)
    ):
        raise NLPValidationError("a development effluent composite fails the nonzero-variance rule.")
    return scale


def _physical_quantities(
    theta: ca.MX,
    chi: ca.MX,
    weights: ca.MX,
    underflow_limit: ca.MX,
    quality_scale: FloatArray,
    inventory_scale: float,
) -> tuple[ca.MX, ca.MX, ca.MX, ca.MX, ca.MX]:
    """Return engineering objective, components, domain rows, and engineering rows."""

    layout = NetworkLayout()
    hrt, aeration, r_i, r_r, waste = (theta[k] for k in range(5))
    q_c, q_u, q_e = 1.0 + r_r, r_r + waste, 1.0 - waste
    reactors = [chi[layout.reactor_slice(stage)] for stage in range(N_STAGES)]
    g_e = chi[layout.overflow_flow_slice]
    g_u = chi[layout.underflow_flow_slice]
    layers = chi[layout.layer_slice]
    c_e, c_u = g_e / q_e, g_u / q_u
    tss = ca.DM(TSS_VECTOR)
    feed_tss = ca.dot(tss, reactors[-1])
    underflow_tss = ca.dot(tss, c_u)
    effluent_tss = ca.dot(tss, c_e)
    boundary_solids = q_e * effluent_tss + waste * underflow_tss
    stage_volume = CLARIFIER.fresh_flow * hrt / (24.0 * N_STAGES)
    inventory = sum(stage_volume * ca.dot(tss, reactor) for reactor in reactors)
    inventory += CLARIFIER.layer_volume * ca.sum1(layers)
    surface_overflow_rate = CLARIFIER.fresh_flow * q_e / CLARIFIER.area
    solids_loading_rate = (
        CLARIFIER.fresh_flow * q_c * feed_tss / (1000.0 * CLARIFIER.area)
    )
    quality_composites = ca.DM(COMPOSITE_MATRIX) @ c_e
    quality = ca.dot(ca.DM(np.full(4, 0.25) / quality_scale), quality_composites)
    components = ca.vertcat(
        quality,
        (hrt - DECISION_LOWER[0]) / DECISION_SPAN[0],
        aeration,
        (r_i - DECISION_LOWER[2]) / DECISION_SPAN[2],
        (r_r - DECISION_LOWER[3]) / DECISION_SPAN[3],
        waste * underflow_tss / (DECISION_UPPER[4] * 15_000.0),
    )
    objective = ca.dot(weights, components)
    domain = ca.vertcat(1.0 - feed_tss, 1.0 - boundary_solids)
    engineering = ca.vertcat(
        (8.0 * CLARIFIER.fresh_flow * boundary_solids - inventory) / inventory_scale,
        (inventory - 30.0 * CLARIFIER.fresh_flow * boundary_solids) / inventory_scale,
        (surface_overflow_rate - 20.0) / 20.0,
        (solids_loading_rate - 100.0) / 100.0,
        (underflow_tss - underflow_limit) / 15_000.0,
    )
    details = ca.vertcat(
        components,
        weights * components,
        feed_tss,
        boundary_solids,
        inventory,
        inventory / (CLARIFIER.fresh_flow * boundary_solids),
        surface_overflow_rate,
        solids_loading_rate,
        underflow_tss,
        quality_composites,
    )
    return objective, components, domain, engineering, details


_PHYSICAL_DIAGNOSTIC_NAMES: tuple[str, ...] = (
    "component_quality", "component_H", "component_a", "component_r_I",
    "component_r_R", "component_wasted_solids",
    "weighted_quality", "weighted_H", "weighted_a", "weighted_r_I",
    "weighted_r_R", "weighted_wasted_solids",
    "feed_tss", "boundary_solids", "solids_inventory", "srt_days",
    "surface_overflow_rate", "solids_loading_rate", "underflow_tss",
    "effluent_cod", "effluent_tn", "effluent_tp", "effluent_tss",
)


def _reconstruct_mechanistic_chi(
    theta: ca.MX, influent: ca.MX, y: ca.MX, scales: SmoothingScales,
) -> tuple[ca.MX, list[ca.MX], ca.MX]:
    reactors = [y[stage * N_COMPONENTS:(stage + 1) * N_COMPONENTS] for stage in range(N_STAGES)]
    layers = y[N_STAGES * N_COMPONENTS:]
    final = reactors[-1]
    feed_tss = ca.dot(ca.DM(TSS_VECTOR), final)
    fractions: list[ca.MX] = []
    for component in range(N_COMPONENTS):
        if component in PARTICULATE:
            fractions.append(final[component] * smooth_feed_reciprocal(feed_tss))
        else:
            fractions.append(0.0)
    c_e = ca.vertcat(*[
        final[j] if j in SOLUBLE else layers[0] * fractions[j]
        for j in range(N_COMPONENTS)
    ])
    c_u = ca.vertcat(*[
        final[j] if j in SOLUBLE else layers[-1] * fractions[j]
        for j in range(N_COMPONENTS)
    ])
    r_i, r_r, waste = theta[2], theta[3], theta[4]
    q_p, q_c, q_u, q_e = 1.0 + r_i + r_r, 1.0 + r_r, r_r + waste, 1.0 - waste
    mixer = (influent + r_i * final + r_r * c_u) / q_p
    chi = ca.vertcat(mixer, *reactors, q_e * c_e, q_u * c_u, layers)
    return chi, reactors, layers


def _mechanistic_raw_expressions(
    theta: ca.MX, influent: ca.MX, y: ca.MX, smoothing: SmoothingScales,
) -> tuple[ca.MX, ca.MX]:
    """Reconstruct the complete state and the ordered 110 smooth balance rows."""

    chi, reactors, layers = _reconstruct_mechanistic_chi(theta, influent, y, smoothing)
    mixer = chi[:20]
    dilution = 120.0 * (1.0 + theta[2] + theta[3]) / theta[0]
    reactor_rhs: list[ca.MX] = []
    upstream = mixer
    stoichiometry = ca.DM(STOICHIOMETRIC_MATRIX)
    for stage, reactor in enumerate(reactors):
        source = stoichiometry.T @ _smooth_rates(reactor, smoothing)
        if stage >= 2:
            source = source + ca.vertcat(
                47.0 * theta[1] * (8.5 - reactor[COMPONENT_INDEX["S_O"]]),
                ca.MX.zeros(N_COMPONENTS - 1),
            )
        reactor_rhs.append(dilution * (upstream - reactor) + source)
        upstream = reactor
    feed_tss = ca.dot(ca.DM(TSS_VECTOR), reactors[-1])
    layer_rhs = _smooth_clarifier_rhs(layers, feed_tss, theta, smoothing)
    return chi, ca.vertcat(*reactor_rhs, layer_rhs)


@lru_cache(maxsize=32)
def _mechanistic_raw_function(smoothing: SmoothingScales) -> ca.Function:
    theta = ca.MX.sym("raw_theta", 5)
    influent = ca.MX.sym("raw_influent", 20)
    state = ca.MX.sym("raw_state", 110)
    chi, residual = _mechanistic_raw_expressions(theta, influent, state, smoothing)
    return ca.Function("mechanistic_raw_kernel", [theta, influent, state], [chi, residual])


def evaluate_symbolic_mechanistic_model(
    decisions: ArrayLike,
    influent: ArrayLike,
    state: ArrayLike,
    smoothing: SmoothingScales,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate reconstructed chi and the 110 raw smooth equations for parity tests."""

    values = _mechanistic_raw_function(smoothing)(
        _vector(decisions, 5, "decisions"), _vector(influent, 20, "influent"),
        _vector(state, 110, "state"),
    )
    return _flat(values[0]), _flat(values[1])


def _lower_triangular_solve_from_upper_transpose(upper: FloatArray, rhs: ca.MX) -> ca.MX:
    """Forward substitution for R.T w=rhs; avoids forming normal equations."""

    values: list[ca.MX] = []
    for row in range(upper.shape[0]):
        accumulated: Any = 0.0
        for column in range(row):
            coefficient = upper[column, row]
            if coefficient != 0.0:
                accumulated += coefficient * values[column]
        values.append((rhs[row] - accumulated) / upper[row, row])
    return ca.vertcat(*values)


def _make_symbolic_problem(
    *,
    name: str,
    variable: ca.MX,
    parameter: ca.MX,
    objective: ca.MX,
    equality: ca.MX,
    inequality: ca.MX,
    physical: ca.MX,
    complete_state: ca.MX,
    diagnostics: ca.MX,
    diagnostic_names: tuple[str, ...],
    lower_bounds: FloatArray,
    upper_bounds: FloatArray,
    physical_scale: FloatArray,
    settings: IPOPTSettings,
    compile_solver: bool,
) -> SymbolicNLP:
    equality_count = int(equality.numel())
    inequality_count = int(inequality.numel())
    constraint = ca.vertcat(equality, inequality)
    solver = None
    if compile_solver:
        solver = ca.nlpsol(
            f"{name}_ipopt", "ipopt",
            {"x": variable, "p": parameter, "f": objective, "g": constraint},
            settings.solver_options(),
        )
    return SymbolicNLP(
        name=name,
        variable_count=int(variable.numel()),
        equality_count=equality_count,
        inequality_count=inequality_count,
        state_count=int(physical.numel()) - 5,
        lower_bounds=lower_bounds.copy(),
        upper_bounds=upper_bounds.copy(),
        solver=solver,
        objective_function=ca.Function(f"{name}_objective", [variable, parameter], [objective]),
        equality_function=ca.Function(f"{name}_equality", [variable, parameter], [equality]),
        inequality_function=ca.Function(f"{name}_inequality", [variable, parameter], [inequality]),
        gradient_function=ca.Function(f"{name}_gradient", [variable, parameter], [ca.gradient(objective, variable)]),
        equality_jacobian_function=ca.Function(f"{name}_equality_jacobian", [variable, parameter], [ca.jacobian(equality, variable)]),
        inequality_jacobian_function=ca.Function(f"{name}_inequality_jacobian", [variable, parameter], [ca.jacobian(inequality, variable)]),
        physical_function=ca.Function(f"{name}_physical", [variable, parameter], [physical]),
        complete_state_function=ca.Function(
            f"{name}_complete_state", [variable, parameter], [complete_state],
        ),
        diagnostics_function=ca.Function(f"{name}_diagnostics", [variable, parameter], [diagnostics]),
        diagnostic_names=diagnostic_names,
        physical_scale=physical_scale.copy(),
        settings=settings,
    )


def build_combined_nlp(
    assets: CombinedNLPAssets,
    *,
    settings: IPOPTSettings | None = None,
    compile_solver: bool = True,
) -> SymbolicNLP:
    """Compile the fixed 115-variable, 110-equality, nine-inequality NLP."""

    settings = settings or IPOPTSettings()
    v = ca.MX.sym("combined_v", COMBINED_VARIABLE_COUNT)
    p_case = ca.MX.sym("combined_case", CASE_PARAMETER_COUNT)
    z, scaled_state = v[:5], v[5:]
    influent, weights, underflow_limit = p_case[:20], p_case[20:26], p_case[26]
    theta = _theta(z)
    y = ca.DM(assets.state_center) + ca.DM(assets.state_scale) * scaled_state
    chi, raw_residual = _mechanistic_raw_expressions(theta, influent, y, assets.smoothing)
    equality = raw_residual / ca.DM(assets.residual_scale)
    engineering_objective, _, domain, engineering, details = _physical_quantities(
        theta, chi, weights, underflow_limit, assets.quality_scale, assets.inventory_scale
    )

    phi, prediction = symbolic_surrogate_prediction(assets.model, theta, influent)
    displacement = (chi - prediction) / ca.DM(assets.model.response_scale)
    fidelity = ca.dot(displacement, displacement) / 170.0
    qr_solution = _lower_triangular_solve_from_upper_transpose(
        assets.model.feature_qr_upper,
        phi[assets.model.feature_qr_pivots.tolist()],
    )
    leverage = ca.dot(qr_solution, qr_solution)
    normalized_fidelity = fidelity / assets.fidelity_delta
    normalized_leverage = (
        (leverage - assets.leverage_max) / max(assets.leverage_max, 1.0e-12)
    )
    inequality = ca.vertcat(
        domain,
        engineering,
        normalized_fidelity - 1.0,
        normalized_leverage,
    )
    diagnostics = ca.vertcat(
        details,
        engineering_objective,
        fidelity,
        normalized_fidelity,
        leverage,
        ca.mmax(ca.fabs(equality)),
    )
    names = _PHYSICAL_DIAGNOSTIC_NAMES + (
        "engineering_objective", "fidelity", "normalized_fidelity", "leverage",
        "maximum_scaled_dynamic_residual",
    )
    lower = np.concatenate((np.zeros(5), -assets.state_center / assets.state_scale))
    upper = np.concatenate((np.ones(5), np.full(110, np.inf)))
    problem = _make_symbolic_problem(
        name="combined_nlp", variable=v, parameter=p_case,
        objective=engineering_objective, equality=equality, inequality=inequality,
        physical=ca.vertcat(theta, y), complete_state=chi, diagnostics=diagnostics,
        diagnostic_names=names, lower_bounds=lower, upper_bounds=upper,
        physical_scale=assets.state_scale, settings=settings,
        compile_solver=compile_solver,
    )
    if (problem.variable_count, problem.equality_count, problem.inequality_count) != (
        COMBINED_VARIABLE_COUNT, COMBINED_EQUALITY_COUNT, COMBINED_INEQUALITY_COUNT
    ):
        raise AssertionError("combined NLP dimensions do not match the manuscript.")
    return problem


def _flat(value: Any) -> FloatArray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def evaluate_problem(
    problem: SymbolicNLP, primal: ArrayLike, case: CaseDefinition,
) -> dict[str, Any]:
    """Evaluate all stored functions without invoking IPOPT."""

    point = _vector(primal, problem.variable_count, "primal")
    parameter = case.parameter_vector()
    physical = _flat(problem.physical_function(point, parameter))
    complete_state = _flat(problem.complete_state_function(point, parameter))
    diagnostics = _flat(problem.diagnostics_function(point, parameter))
    return {
        "objective": float(problem.objective_function(point, parameter)),
        "equality": _flat(problem.equality_function(point, parameter)),
        "inequality": _flat(problem.inequality_function(point, parameter)),
        "decisions": physical[:5],
        "state": physical[5:],
        "complete_state": complete_state,
        "diagnostics": dict(zip(problem.diagnostic_names, diagnostics, strict=True)),
    }


def evaluate_symbolic_surrogate_prediction(
    model: QuadraticSurrogate, decisions: ArrayLike, influent: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate the CasADi feature and prediction expressions for parity audits."""

    theta_symbol = ca.MX.sym("prediction_theta", 5)
    influent_symbol = ca.MX.sym("prediction_influent", 20)
    phi, prediction = symbolic_surrogate_prediction(model, theta_symbol, influent_symbol)
    function = ca.Function("surrogate_prediction_eval", [theta_symbol, influent_symbol], [phi, prediction])
    values = function(_vector(decisions, 5, "decisions"), _vector(influent, 20, "influent"))
    return _flat(values[0]), _flat(values[1])


def replay_kkt(
    problem: SymbolicNLP,
    primal: ArrayLike,
    case: CaseDefinition,
    equality_multipliers: ArrayLike,
    inequality_multipliers: ArrayLike,
    bound_multipliers: ArrayLike,
) -> KKTDiagnostics:
    """Independently recompute every scaled KKT residual from raw multipliers."""

    point = _vector(primal, problem.variable_count, "primal")
    lam_h = _vector(equality_multipliers, problem.equality_count, "equality_multipliers")
    mu_g = _vector(inequality_multipliers, problem.inequality_count, "inequality_multipliers")
    lam_x = _vector(bound_multipliers, problem.variable_count, "bound_multipliers")
    parameter = case.parameter_vector()
    equality = _flat(problem.equality_function(point, parameter))
    inequality = _flat(problem.inequality_function(point, parameter))
    gradient = _flat(problem.gradient_function(point, parameter))
    equality_jacobian = np.asarray(problem.equality_jacobian_function(point, parameter), dtype=float)
    inequality_jacobian = np.asarray(problem.inequality_jacobian_function(point, parameter), dtype=float)
    physical = _flat(problem.physical_function(point, parameter))[5:]

    finite_upper = np.isfinite(problem.upper_bounds)
    two_sided = finite_upper
    mu_lower = np.empty(problem.variable_count, dtype=np.float64)
    mu_upper = np.zeros(problem.variable_count, dtype=np.float64)
    mu_lower[two_sided] = np.maximum(-lam_x[two_sided], 0.0)
    mu_upper[two_sided] = np.maximum(lam_x[two_sided], 0.0)
    mu_lower[~two_sided] = -lam_x[~two_sided]

    lower_slack = point - problem.lower_bounds
    upper_slack = problem.upper_bounds[finite_upper] - point[finite_upper]
    general_slack = -inequality
    lower_violation = float(np.max(np.maximum(-lower_slack, 0.0), initial=0.0))
    upper_violation = float(np.max(np.maximum(-upper_slack, 0.0), initial=0.0))
    bound_violation = max(lower_violation, upper_violation)
    equality_violation = float(np.max(np.abs(equality), initial=0.0))
    inequality_violation = float(np.max(np.maximum(inequality, 0.0), initial=0.0))
    primal_residual = max(equality_violation, inequality_violation, bound_violation)

    equality_term = equality_jacobian.T @ lam_h
    inequality_term = inequality_jacobian.T @ mu_g
    stationarity_vector = gradient + equality_term + inequality_term - mu_lower + mu_upper
    stationarity_denominator = (
        1.0 + np.linalg.norm(gradient, ord=np.inf)
        + np.linalg.norm(equality_term, ord=np.inf)
        + np.linalg.norm(inequality_term, ord=np.inf)
        + np.linalg.norm(mu_lower, ord=np.inf)
        + np.linalg.norm(mu_upper, ord=np.inf)
    )
    stationarity_residual = float(np.linalg.norm(stationarity_vector, ord=np.inf) / stationarity_denominator)
    general_dual = float(np.max(np.maximum(-mu_g, 0.0), initial=0.0)) / (
        1.0 + float(np.linalg.norm(mu_g, ord=np.inf))
    )
    lower_dual = float(np.max(np.maximum(-mu_lower, 0.0), initial=0.0)) / (
        1.0 + float(np.linalg.norm(mu_lower, ord=np.inf))
    )
    dual_residual = max(general_dual, lower_dual)

    general_product = float(np.max(np.abs(mu_g * general_slack), initial=0.0))
    lower_product = float(np.max(np.abs(mu_lower * lower_slack), initial=0.0))
    upper_product = float(
        np.max(np.abs(mu_upper[finite_upper] * upper_slack), initial=0.0)
    )
    complementarity_numerator = max(general_product, lower_product, upper_product)
    general_scale = float(np.linalg.norm(mu_g, ord=np.inf) * np.linalg.norm(general_slack, ord=np.inf))
    lower_scale = float(np.linalg.norm(mu_lower, ord=np.inf) * np.linalg.norm(lower_slack, ord=np.inf))
    upper_scale = float(
        np.linalg.norm(mu_upper[finite_upper], ord=np.inf) * np.linalg.norm(upper_slack, ord=np.inf)
    )
    complementarity_residual = complementarity_numerator / (
        1.0 + max(general_scale, lower_scale, upper_scale)
    )
    physical_nonnegativity = float(
        np.max(np.maximum(-physical / np.maximum(1.0, problem.physical_scale), 0.0), initial=0.0)
    )
    all_values = (
        point, lam_h, mu_g, lam_x, equality, inequality, gradient,
        equality_jacobian, inequality_jacobian, physical,
    )
    finite = bool(all(np.all(np.isfinite(value)) for value in all_values))
    residuals = np.asarray(
        [bound_violation, primal_residual, stationarity_residual, dual_residual,
         complementarity_residual, physical_nonnegativity], dtype=float,
    )
    finite = finite and bool(np.all(np.isfinite(residuals)))
    if not finite:
        residuals[:] = np.inf
    return KKTDiagnostics(
        bound_violation=float(residuals[0]),
        primal_residual=float(residuals[1]),
        stationarity_residual=float(residuals[2]),
        dual_feasibility_residual=float(residuals[3]),
        complementarity_residual=float(residuals[4]),
        physical_nonnegativity_residual=float(residuals[5]),
        finite=finite,
    )


def _failed_result(
    problem: SymbolicNLP,
    initial_point: FloatArray,
    case: CaseDefinition,
    start_index: int,
    elapsed_seconds: float,
    status: str,
    error: str,
) -> NLPStartResult:
    try:
        evaluated = evaluate_problem(problem, initial_point, case)
        decisions = evaluated["decisions"]
        state = evaluated["state"]
        equality = evaluated["equality"]
        inequality = evaluated["inequality"]
    except Exception:
        decisions = np.full(5, np.nan)
        state = np.full(problem.state_count, np.nan)
        equality = np.full(problem.equality_count, np.nan)
        inequality = np.full(problem.inequality_count, np.nan)
    failed_kkt = KKTDiagnostics(*(np.inf for _ in range(6)), finite=False)
    return NLPStartResult(
        start_index=start_index, status=status, solver_success=False, accepted=False,
        objective=np.nan, primal=initial_point.copy(),
        equality_multipliers=np.full(problem.equality_count, np.nan),
        inequality_multipliers=np.full(problem.inequality_count, np.nan),
        bound_multipliers=np.full(problem.variable_count, np.nan),
        equality=equality, inequality=inequality,
        normalized_controls=initial_point[:5].copy(), decisions=decisions, state=state,
        diagnostics={}, kkt=failed_kkt, elapsed_seconds=elapsed_seconds,
        iterations=0, error=error,
    )


def solve_nlp_start(
    problem: SymbolicNLP,
    case: CaseDefinition,
    initial_point: ArrayLike,
    *,
    start_index: int,
) -> NLPStartResult:
    """Run one cold IPOPT start and apply the independent acceptance replay."""

    point0 = _vector(initial_point, problem.variable_count, "initial_point")
    if problem.solver is None:
        raise NLPValidationError("the problem was built with compile_solver=False.")
    parameter = case.parameter_vector()
    started = perf_counter_ns()
    try:
        solution = problem.solver(
            x0=point0,
            p=parameter,
            lbx=problem.lower_bounds,
            ubx=problem.upper_bounds,
            lbg=problem.constraint_lower_bounds,
            ubg=problem.constraint_upper_bounds,
        )
        elapsed = (perf_counter_ns() - started) * 1.0e-9
        stats = problem.solver.stats()
        status = str(stats.get("return_status", "unknown"))
        primal = _flat(solution["x"])
        objective = float(solution["f"])
        lam_g = _flat(solution["lam_g"])
        lam_h = lam_g[:problem.equality_count]
        mu_g = lam_g[problem.equality_count:]
        lam_x = _flat(solution["lam_x"])
        kkt = replay_kkt(problem, primal, case, lam_h, mu_g, lam_x)
        evaluated = evaluate_problem(problem, primal, case)
        finite = bool(
            np.isfinite(objective)
            and np.all(np.isfinite(primal))
            and np.all(np.isfinite(lam_g))
            and np.all(np.isfinite(lam_x))
        )
        accepted = bool(
            problem.settings.accepts_return_status(status)
            and finite
            and kkt.accepted(problem.settings)
        )
        return NLPStartResult(
            start_index=int(start_index), status=status,
            solver_success=bool(stats.get("success", False)), accepted=accepted,
            objective=objective, primal=primal,
            equality_multipliers=lam_h, inequality_multipliers=mu_g,
            bound_multipliers=lam_x, equality=evaluated["equality"],
            inequality=evaluated["inequality"],
            normalized_controls=primal[:5].copy(), decisions=evaluated["decisions"],
            state=evaluated["state"], diagnostics=evaluated["diagnostics"],
            kkt=kkt, elapsed_seconds=elapsed,
            iterations=int(stats.get("iter_count", 0)), error=None,
        )
    except Exception as exc:
        elapsed = (perf_counter_ns() - started) * 1.0e-9
        return _failed_result(
            problem, point0, case, int(start_index), elapsed,
            status="solver_exception", error=f"{type(exc).__name__}: {exc}",
        )


def solve_nlp_multistart(
    problem: SymbolicNLP,
    case: CaseDefinition,
    initial_points: Iterable[ArrayLike],
) -> tuple[tuple[NLPStartResult, ...], NLPStartResult | None]:
    """Solve independent cold starts in order and select deterministically."""

    points = tuple(initial_points)
    if len(points) != 9:
        raise NLPValidationError("the fixed multistart protocol requires exactly nine starts.")
    results = tuple(
        solve_nlp_start(problem, case, point, start_index=index)
        for index, point in enumerate(points)
    )
    return results, select_best_start(results)


def select_best_start(results: Sequence[NLPStartResult]) -> NLPStartResult | None:
    """Apply the objective-tolerance, lexicographic-control, start-index rule."""

    accepted = [result for result in results if result.accepted]
    if not accepted:
        return None
    minimum = min(result.objective for result in accepted)
    tolerance = 1.0e-10 * (1.0 + abs(minimum))
    tied = [result for result in accepted if result.objective <= minimum + tolerance]
    return min(tied, key=lambda result: (*result.normalized_controls.tolist(), result.start_index))


__all__ = [
    "ACCEPTED_STATUSES", "CASE_PARAMETER_COUNT", "COMBINED_EQUALITY_COUNT",
    "COMBINED_INEQUALITY_COUNT", "COMBINED_VARIABLE_COUNT", "CaseDefinition",
    "CombinedNLPAssets", "IPOPTSettings", "KKTDiagnostics", "NLPStartResult",
    "NLPValidationError", "ObjectiveWeights", "SmoothingScales", "SymbolicNLP",
    "build_combined_nlp", "combined_initial_point",
    "evaluate_problem", "evaluate_smooth_clarifier_fluxes", "evaluate_smooth_clarifier_rhs",
    "evaluate_smooth_process_rates", "fit_mechanistic_residual_scales",
    "evaluate_symbolic_mechanistic_model", "extract_mechanistic_states",
    "fit_inventory_scale", "fit_mechanistic_state_scaling", "fit_quality_scale",
    "fit_smoothing_scales", "evaluate_symbolic_surrogate_prediction",
    "nearest_development_index", "ordered_normalized_starts",
    "receiver_transition", "replay_kkt", "select_best_start", "smooth_division",
    "smooth_feed_reciprocal", "smooth_maximum", "smooth_minimum", "smooth_positive_part",
    "solve_nlp_multistart", "solve_nlp_start",
]
