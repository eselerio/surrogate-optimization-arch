"""Exact-QP active-set refinement for the manuscript-v3 surrogate route.

The gap-continuation problem in :mod:`closed_loop.v3_surrogate_nlp` is useful
for reaching the neighbourhood of a lower-QP solution, but it is deliberately
not the reportable outer problem.  This module implements that outer problem
in the seven normalized control coordinates.  Every distinct trial control
is evaluated by a newly initialized OSQP projection, and all derivatives are
total derivatives through the active lower-QP KKT system.

No finite-difference derivative fallback is provided.  A rank-deficient or
ill-conditioned lower active set, a weakly active lower inequality, or an
active-set change under the declared perturbation raises
``ActiveSetDerivativeError``.  This is intentional: such an endpoint may be a
validated feasible incumbent, but the manuscript does not permit it to be
called first-order stationary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Any, Sequence

import casadi as ca
import numpy as np
import numpy.typing as npt
import osqp
from scipy import linalg, sparse
from scipy.optimize import lsq_linear, minimize

from .projection import ProjectionResult
from .v3_surrogate_nlp import (
    SurrogateCase,
    SurrogateNLP,
    SurrogateNLPAssets,
    build_surrogate_nlp,
    cold_reproject,
)


FloatArray = npt.NDArray[np.float64]


class ActiveSetRefinementError(RuntimeError):
    """Base error for exact-QP outer refinement."""


class ExactQPProjectionError(ActiveSetRefinementError):
    """Raised when a cold lower-QP trial fails its independent audit."""


class ActiveSetDerivativeError(ActiveSetRefinementError):
    """Raised when the manuscript's active-set derivative gate is not met."""

    def __init__(self, message: str, audit: "LowerActiveSetAudit | None" = None) -> None:
        super().__init__(message)
        self.audit = audit


def _vector(value: npt.ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {size}.")
    return array.copy()


def _maximum_positive(value: npt.ArrayLike) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(np.max(np.maximum(array, 0.0), initial=0.0))


def _safe_name(value: str) -> str:
    result = "".join(character if character.isalnum() else "_" for character in value)
    return result or "v3_active_set"


@dataclass(frozen=True)
class ActiveSetRefinementSettings:
    """Frozen tolerances from the supplementary active-set protocol."""

    active_tolerance: float = 1.0e-7
    multiplier_tolerance: float = 1.0e-8
    perturbation: float = 1.0e-6
    condition_epsilon_limit: float = 1.0e-8
    sensitivity_residual_tolerance: float = 1.0e-8
    state_reproduction_tolerance: float = 1.0e-8
    upper_acceptance_tolerance: float = 1.0e-6
    maximum_iterations: int = 250
    function_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        positive = (
            self.active_tolerance,
            self.multiplier_tolerance,
            self.perturbation,
            self.condition_epsilon_limit,
            self.sensitivity_residual_tolerance,
            self.state_reproduction_tolerance,
            self.upper_acceptance_tolerance,
            self.function_tolerance,
        )
        if not all(np.isfinite(item) and item > 0.0 for item in positive):
            raise ValueError("active-set tolerances must be finite and positive.")
        if self.perturbation >= 0.5:
            raise ValueError("the normalized active-set perturbation must be below 0.5.")
        if self.maximum_iterations < 1:
            raise ValueError("maximum_iterations must be positive.")


@dataclass(frozen=True)
class ActiveSetPerturbation:
    coordinate: int
    direction: int
    accepted_projection: bool
    active_indices: tuple[int, ...]
    active_set_preserved: bool
    multiplier_signs_preserved: bool
    minimum_active_multiplier: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LowerActiveSetAudit:
    active_indices: tuple[int, ...]
    active_count: int
    active_row_rank: int
    required_active_row_rank: int
    rank_tolerance: float
    smallest_active_row_singular_value: float
    kkt_condition_number: float
    condition_times_machine_epsilon: float
    minimum_active_multiplier: float | None
    active_kkt_residual: float
    rank_passed: bool
    conditioning_passed: bool
    strict_complementarity_passed: bool
    perturbation_passed: bool
    perturbations: tuple[ActiveSetPerturbation, ...]
    stable: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["active_indices"] = list(self.active_indices)
        value["perturbations"] = [item.as_dict() for item in self.perturbations]
        return value


@dataclass(frozen=True)
class ExactQPSensitivity:
    displacement_wrt_normalized: FloatArray
    state_wrt_normalized: FloatArray
    active_multiplier_wrt_normalized: FloatArray
    displacement_wrt_physical: FloatArray
    state_wrt_physical: FloatArray
    solve_residual: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "displacement_wrt_normalized": self.displacement_wrt_normalized.tolist(),
            "state_wrt_normalized": self.state_wrt_normalized.tolist(),
            "active_multiplier_wrt_normalized": (
                self.active_multiplier_wrt_normalized.tolist()
            ),
            "displacement_wrt_physical": self.displacement_wrt_physical.tolist(),
            "state_wrt_physical": self.state_wrt_physical.tolist(),
            "solve_residual": self.solve_residual,
        }


@dataclass(frozen=True)
class ExactQPTrial:
    normalized_controls: FloatArray
    physical_controls: FloatArray
    raw_state: FloatArray
    projected_state: FloatArray
    objective: float
    objective_gradient_normalized: FloatArray
    objective_gradient_physical: FloatArray
    upper_constraint_names: tuple[str, ...]
    upper_constraints: FloatArray
    upper_constraint_jacobian_normalized: FloatArray
    upper_constraint_jacobian_physical: FloatArray
    engineering_rows: FloatArray
    trust_rows: FloatArray
    projection: ProjectionResult
    lower_active_set: LowerActiveSetAudit
    sensitivity: ExactQPSensitivity
    independent_final_replay: bool
    elapsed_seconds: float

    @property
    def feasible(self) -> bool:
        return bool(
            self.projection.accepted
            and _maximum_positive(self.upper_constraints) <= 1.0e-6
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "normalized_controls": self.normalized_controls.tolist(),
            "physical_controls": self.physical_controls.tolist(),
            "raw_state": self.raw_state.tolist(),
            "projected_state": self.projected_state.tolist(),
            "objective": self.objective,
            "objective_gradient_normalized": self.objective_gradient_normalized.tolist(),
            "objective_gradient_physical": self.objective_gradient_physical.tolist(),
            "upper_constraint_names": list(self.upper_constraint_names),
            "upper_constraints": self.upper_constraints.tolist(),
            "upper_constraint_jacobian_normalized": (
                self.upper_constraint_jacobian_normalized.tolist()
            ),
            "upper_constraint_jacobian_physical": (
                self.upper_constraint_jacobian_physical.tolist()
            ),
            "engineering_rows": self.engineering_rows.tolist(),
            "trust_rows": self.trust_rows.tolist(),
            "projection": {
                "accepted": self.projection.accepted,
                "diagnostics": self.projection.diagnostics.as_dict(),
            },
            "lower_active_set": self.lower_active_set.as_dict(),
            "sensitivity": self.sensitivity.as_dict(),
            "independent_final_replay": self.independent_final_replay,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class UpperKKTAudit:
    active_indices: tuple[int, ...]
    active_names: tuple[str, ...]
    multipliers: FloatArray
    primal_residual: float
    dual_feasibility_residual: float
    stationarity_residual: float
    complementarity_residual: float
    feasible: bool
    stationary: bool
    classification: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["active_indices"] = list(self.active_indices)
        value["active_names"] = list(self.active_names)
        value["multipliers"] = self.multipliers.tolist()
        return value


@dataclass(frozen=True)
class ExactQPRefinementResult:
    initial_controls: FloatArray
    initial: ExactQPTrial | None
    final: ExactQPTrial | None
    upper_kkt: UpperKKTAudit | None
    solver_success: bool
    solver_status: str
    iterations: int
    distinct_trials: int
    cold_qp_resolutions: int
    elapsed_seconds: float
    status: str
    derivative_error: str | None = None
    derivative_audit: LowerActiveSetAudit | None = None
    state_reproduction_residual: float | None = None
    state_reproduction_passed: bool | None = None

    @property
    def feasible(self) -> bool:
        return bool(
            self.upper_kkt is not None
            and self.upper_kkt.feasible
            and self.state_reproduction_passed is True
        )

    @property
    def stationary(self) -> bool:
        return bool(self.feasible and self.upper_kkt is not None and self.upper_kkt.stationary)

    def as_dict(self) -> dict[str, Any]:
        return {
            "initial_controls": self.initial_controls.tolist(),
            "initial": None if self.initial is None else self.initial.as_dict(),
            "final": None if self.final is None else self.final.as_dict(),
            "upper_kkt": None if self.upper_kkt is None else self.upper_kkt.as_dict(),
            "solver_success": self.solver_success,
            "solver_status": self.solver_status,
            "iterations": self.iterations,
            "distinct_trials": self.distinct_trials,
            "cold_qp_resolutions": self.cold_qp_resolutions,
            "elapsed_seconds": self.elapsed_seconds,
            "status": self.status,
            "derivative_error": self.derivative_error,
            "derivative_audit": (
                None if self.derivative_audit is None else self.derivative_audit.as_dict()
            ),
            "state_reproduction_residual": self.state_reproduction_residual,
            "state_reproduction_passed": self.state_reproduction_passed,
        }


@dataclass(frozen=True)
class _LowerMatrices:
    raw: FloatArray
    equality: FloatArray
    equality_rhs: FloatArray
    inequality: FloatArray
    inequality_rhs: FloatArray
    raw_derivative: FloatArray
    equality_derivative: FloatArray
    equality_rhs_derivative: FloatArray
    inequality_derivative: FloatArray
    inequality_rhs_derivative: FloatArray


class ExactQPActiveSetRefiner:
    """Seven-variable outer evaluator and optimizer with exact-QP derivatives."""

    def __init__(
        self,
        assets: SurrogateNLPAssets,
        case: SurrogateCase,
        *,
        problem: SurrogateNLP | None = None,
        settings: ActiveSetRefinementSettings | None = None,
        name: str = "v3_active_set",
    ) -> None:
        self.assets = assets
        self.case = case
        self.settings = settings or ActiveSetRefinementSettings()
        self.problem = problem or build_surrogate_nlp(
            assets,
            1.0e-8,
            name=f"{name}_expressions",
            compile_solver=False,
        )
        if self.problem.assets is not assets:
            raise ValueError("the supplied surrogate problem must use the supplied assets object.")
        self.parameter = case.parameter_vector(assets)
        self._cache: dict[bytes, ExactQPTrial] = {}
        self._cold_qp_resolutions = 0
        self._compile_functions(_safe_name(f"{name}_{case.case_id}"))

    @property
    def distinct_trials(self) -> int:
        return len(self._cache)

    @property
    def cold_qp_resolutions(self) -> int:
        return self._cold_qp_resolutions

    def _compile_functions(self, name: str) -> None:
        n_state = self.assets.layout.state_size
        n_equality = self.assets.equality_count
        n_inequality = self.assets.projection_inequality_count
        normalized = ca.MX.sym(f"{name}_z", 7)
        parameter = ca.MX.sym(f"{name}_p", self.problem.parameter_count)
        state = ca.MX.sym(f"{name}_state", n_state)
        raw = self.problem.raw_function(normalized, parameter)
        network = self.problem.network_function(normalized, parameter)
        physical_equality, physical_rhs, physical_inequality = network
        state_scale = ca.DM(self.assets.model.response_scale)
        equality_scale = ca.DM(self.assets.row_scales.equality)
        inequality_scale = ca.DM(self.assets.row_scales.inequality)
        equality = (
            ca.diag(1.0 / equality_scale)
            @ physical_equality
            @ ca.diag(state_scale)
        )
        equality_rhs = (
            physical_rhs - physical_equality @ raw
        ) / equality_scale
        scaled_network = (
            ca.diag(1.0 / inequality_scale)
            @ physical_inequality
            @ ca.diag(state_scale)
        )
        inequality = ca.vertcat(-ca.DM.eye(n_state), scaled_network)
        inequality_rhs = ca.vertcat(
            raw / state_scale,
            -(physical_inequality @ raw) / inequality_scale,
        )
        self._lower_function = ca.Function(
            f"{name}_lower",
            [normalized, parameter],
            [raw, equality, equality_rhs, inequality, inequality_rhs],
        )
        self._lower_derivative_function = ca.Function(
            f"{name}_lower_derivative",
            [normalized, parameter],
            [
                ca.jacobian(raw, normalized),
                ca.jacobian(ca.vec(equality), normalized),
                ca.jacobian(equality_rhs, normalized),
                ca.jacobian(ca.vec(inequality), normalized),
                ca.jacobian(inequality_rhs, normalized),
            ],
        )
        upper = self.problem.upper_from_state_function(normalized, parameter, state)
        upper_vector = ca.vertcat(upper[0], upper[1], upper[2])
        self._upper_function = ca.Function(
            f"{name}_upper",
            [normalized, parameter, state],
            [upper_vector, ca.jacobian(upper_vector, normalized), ca.jacobian(upper_vector, state)],
        )
        if int(equality.numel()) != n_equality * n_state:
            raise AssertionError("lower equality serialization has an inconsistent size.")
        if int(inequality.numel()) != n_inequality * n_state:
            raise AssertionError("lower inequality serialization has an inconsistent size.")

    def _normalized(self, value: npt.ArrayLike) -> FloatArray:
        normalized = _vector(value, 7, "normalized_controls")
        tolerance = 1.0e-12
        if np.any(normalized < -tolerance) or np.any(normalized > 1.0 + tolerance):
            raise ValueError("normalized_controls must lie in [0, 1].")
        return np.clip(normalized, 0.0, 1.0)

    def _cold_project(self, normalized: FloatArray) -> ProjectionResult:
        self._cold_qp_resolutions += 1
        return cold_reproject(
            self.assets,
            self.case,
            normalized,
            raise_on_failure=False,
        )

    def _lower_matrices(self, normalized: FloatArray) -> _LowerMatrices:
        n_state = self.assets.layout.state_size
        n_equality = self.assets.equality_count
        n_inequality = self.assets.projection_inequality_count
        values = self._lower_function(normalized, self.parameter)
        derivatives = self._lower_derivative_function(normalized, self.parameter)
        return _LowerMatrices(
            raw=np.asarray(values[0], dtype=np.float64).reshape(n_state),
            equality=np.asarray(values[1], dtype=np.float64).reshape(
                n_equality, n_state
            ),
            equality_rhs=np.asarray(values[2], dtype=np.float64).reshape(n_equality),
            inequality=np.asarray(values[3], dtype=np.float64).reshape(
                n_inequality, n_state
            ),
            inequality_rhs=np.asarray(values[4], dtype=np.float64).reshape(n_inequality),
            raw_derivative=np.asarray(derivatives[0], dtype=np.float64).reshape(
                n_state, 7
            ),
            equality_derivative=np.asarray(derivatives[1], dtype=np.float64).reshape(
                n_equality, n_state, 7, order="F"
            ),
            equality_rhs_derivative=np.asarray(
                derivatives[2], dtype=np.float64
            ).reshape(n_equality, 7),
            inequality_derivative=np.asarray(
                derivatives[3], dtype=np.float64
            ).reshape(n_inequality, n_state, 7, order="F"),
            inequality_rhs_derivative=np.asarray(
                derivatives[4], dtype=np.float64
            ).reshape(n_inequality, 7),
        )

    def _perturbation_directions(self, normalized: FloatArray) -> Sequence[tuple[int, int]]:
        delta = self.settings.perturbation
        result: list[tuple[int, int]] = []
        for coordinate in range(7):
            if normalized[coordinate] - delta >= 0.0:
                result.append((coordinate, -1))
            if normalized[coordinate] + delta <= 1.0:
                result.append((coordinate, 1))
        return result

    def _audit_active_set(
        self,
        normalized: FloatArray,
        projection: ProjectionResult,
        matrices: _LowerMatrices,
    ) -> tuple[LowerActiveSetAudit, FloatArray, FloatArray, FloatArray]:
        settings = self.settings
        active = np.flatnonzero(
            projection.inequality_slack <= settings.active_tolerance
        )
        block = np.vstack((matrices.equality, matrices.inequality[active]))
        singular = linalg.svdvals(block, check_finite=True)
        epsilon = np.finfo(np.float64).eps
        rank_tolerance = (
            max(block.shape) * epsilon * singular[0] if singular.size else 0.0
        )
        numerical_rank = int(np.count_nonzero(singular > rank_tolerance))
        required_rank = int(block.shape[0])
        rank_passed = numerical_rank == required_rank
        smallest = float(singular[-1]) if singular.size else np.inf

        n_state = self.assets.layout.state_size
        kkt = np.block(
            [
                [np.eye(n_state), block.T],
                [block, np.zeros((block.shape[0], block.shape[0]))],
            ]
        )
        kkt_singular = linalg.svdvals(kkt, check_finite=True)
        condition = (
            float(kkt_singular[0] / kkt_singular[-1])
            if kkt_singular[-1] > 0.0
            else np.inf
        )
        condition_epsilon = condition * epsilon
        conditioning_passed = bool(
            np.isfinite(condition_epsilon)
            and condition_epsilon <= settings.condition_epsilon_limit
        )
        active_multiplier = projection.inequality_multipliers[active]
        minimum_multiplier = (
            float(np.min(active_multiplier)) if active.size else None
        )
        strict_passed = bool(
            active.size == 0
            or np.all(active_multiplier > settings.multiplier_tolerance)
        )
        multipliers = np.concatenate(
            (projection.equality_multipliers, active_multiplier)
        )
        right_hand_side = np.concatenate(
            (
                np.zeros(n_state),
                matrices.equality_rhs,
                matrices.inequality_rhs[active],
            )
        )
        vector = np.concatenate((projection.displacement, multipliers))
        kkt_residual = float(np.linalg.norm(kkt @ vector - right_hand_side, ord=np.inf))

        perturbations: list[ActiveSetPerturbation] = []
        perturbation_passed = True
        if rank_passed and conditioning_passed and strict_passed:
            for coordinate, direction in self._perturbation_directions(normalized):
                perturbed = normalized.copy()
                perturbed[coordinate] += direction * settings.perturbation
                result = self._cold_project(perturbed)
                perturbed_active = np.flatnonzero(
                    result.inequality_slack <= settings.active_tolerance
                ) if result.accepted else np.asarray([], dtype=int)
                preserved = bool(
                    result.accepted and np.array_equal(perturbed_active, active)
                )
                if preserved and perturbed_active.size:
                    perturbed_minimum = float(
                        np.min(result.inequality_multipliers[perturbed_active])
                    )
                    signs = bool(
                        np.all(
                            result.inequality_multipliers[perturbed_active]
                            > settings.multiplier_tolerance
                        )
                    )
                elif preserved:
                    perturbed_minimum = None
                    signs = True
                else:
                    perturbed_minimum = None
                    signs = False
                perturbations.append(
                    ActiveSetPerturbation(
                        coordinate=coordinate,
                        direction=direction,
                        accepted_projection=bool(result.accepted),
                        active_indices=tuple(int(item) for item in perturbed_active),
                        active_set_preserved=preserved,
                        multiplier_signs_preserved=signs,
                        minimum_active_multiplier=perturbed_minimum,
                    )
                )
                perturbation_passed = perturbation_passed and preserved and signs
        else:
            perturbation_passed = False

        stable = bool(
            rank_passed
            and conditioning_passed
            and strict_passed
            and perturbation_passed
        )
        reasons: list[str] = []
        if not rank_passed:
            reasons.append("active lower rows fail numerical LICQ")
        if not conditioning_passed:
            reasons.append("lower KKT condition-times-epsilon exceeds 1e-8")
        if not strict_passed:
            reasons.append("an active lower multiplier is not greater than 1e-8")
        if not perturbation_passed:
            reasons.append("the 1e-6 domain-aware perturbation test failed")
        reason = "active-set sensitivity gates passed" if stable else "; ".join(reasons)
        audit = LowerActiveSetAudit(
            active_indices=tuple(int(item) for item in active),
            active_count=int(active.size),
            active_row_rank=numerical_rank,
            required_active_row_rank=required_rank,
            rank_tolerance=float(rank_tolerance),
            smallest_active_row_singular_value=smallest,
            kkt_condition_number=condition,
            condition_times_machine_epsilon=condition_epsilon,
            minimum_active_multiplier=minimum_multiplier,
            active_kkt_residual=kkt_residual,
            rank_passed=rank_passed,
            conditioning_passed=conditioning_passed,
            strict_complementarity_passed=strict_passed,
            perturbation_passed=perturbation_passed,
            perturbations=tuple(perturbations),
            stable=stable,
            reason=reason,
        )
        return audit, block, kkt, multipliers

    def _sensitivity(
        self,
        projection: ProjectionResult,
        matrices: _LowerMatrices,
        audit: LowerActiveSetAudit,
        block: FloatArray,
        kkt: FloatArray,
        multipliers: FloatArray,
    ) -> ExactQPSensitivity:
        if not audit.stable:
            raise ActiveSetDerivativeError(audit.reason, audit)
        active = np.asarray(audit.active_indices, dtype=int)
        n_state = self.assets.layout.state_size
        n_equality = self.assets.equality_count
        right_hand_sides = np.empty((kkt.shape[0], 7), dtype=np.float64)
        for coordinate in range(7):
            derivative_block = np.vstack(
                (
                    matrices.equality_derivative[:, :, coordinate],
                    matrices.inequality_derivative[active, :, coordinate],
                )
            )
            derivative_rhs = np.concatenate(
                (
                    matrices.equality_rhs_derivative[:, coordinate],
                    matrices.inequality_rhs_derivative[active, coordinate],
                )
            )
            right_hand_sides[:n_state, coordinate] = -derivative_block.T @ multipliers
            right_hand_sides[n_state:, coordinate] = (
                derivative_rhs - derivative_block @ projection.displacement
            )
        try:
            derivative = linalg.solve(
                kkt,
                right_hand_sides,
                assume_a="sym",
                check_finite=True,
            )
        except linalg.LinAlgError as exc:
            raise ActiveSetDerivativeError(
                "the accepted lower KKT sensitivity system could not be solved", audit
            ) from exc
        residual = float(
            np.linalg.norm(kkt @ derivative - right_hand_sides, ord=np.inf)
            / max(1.0, float(np.linalg.norm(right_hand_sides, ord=np.inf)))
        )
        if not np.all(np.isfinite(derivative)) or residual > self.settings.sensitivity_residual_tolerance:
            raise ActiveSetDerivativeError(
                "the lower KKT sensitivity solve failed its residual audit", audit
            )
        displacement_derivative = derivative[:n_state]
        state_derivative = (
            matrices.raw_derivative
            + self.assets.model.response_scale[:, None] * displacement_derivative
        )
        inverse_span = 1.0 / self.assets.theta_span
        return ExactQPSensitivity(
            displacement_wrt_normalized=displacement_derivative,
            state_wrt_normalized=state_derivative,
            active_multiplier_wrt_normalized=derivative[n_state:],
            displacement_wrt_physical=displacement_derivative * inverse_span[None, :],
            state_wrt_physical=state_derivative * inverse_span[None, :],
            solve_residual=residual,
        )

    def evaluate(
        self,
        normalized_controls: npt.ArrayLike,
        *,
        force_cold: bool = False,
        independent_final_replay: bool = False,
    ) -> ExactQPTrial:
        """Cold-resolve a trial and return exact values and total gradients.

        Repeated objective/constraint callbacks at the identical bytewise
        control vector share one trial evaluation.  Every *distinct* trial is
        cold-solved.  ``force_cold`` bypasses that cache for the independent
        final replay.
        """

        normalized = self._normalized(normalized_controls)
        key = normalized.tobytes()
        if not force_cold and key in self._cache:
            return self._cache[key]
        started = perf_counter()
        projection = self._cold_project(normalized)
        if not projection.accepted:
            raise ExactQPProjectionError(
                "cold projection failed its independent lower-QP KKT audit: "
                f"{projection.diagnostics.as_dict()}"
            )
        matrices = self._lower_matrices(normalized)
        audit, block, kkt, multipliers = self._audit_active_set(
            normalized, projection, matrices
        )
        sensitivity = self._sensitivity(
            projection, matrices, audit, block, kkt, multipliers
        )
        upper, partial_normalized, partial_state = self._upper_function(
            normalized, self.parameter, projection.state
        )
        upper_value = np.asarray(upper, dtype=np.float64).reshape(-1)
        partial_normalized_array = np.asarray(
            partial_normalized, dtype=np.float64
        )
        partial_state_array = np.asarray(partial_state, dtype=np.float64)
        total = (
            partial_normalized_array
            + partial_state_array @ sensitivity.state_wrt_normalized
        )
        engineering_count = len(self.problem.engineering_names)
        trust_count = len(self.problem.trust_names)
        expected = 1 + engineering_count + trust_count
        if upper_value.size != expected or total.shape != (expected, 7):
            raise AssertionError("upper expression serialization is inconsistent.")
        engineering = upper_value[1 : 1 + engineering_count]
        trust = upper_value[1 + engineering_count :]
        upper_constraints = np.concatenate(
            (engineering, trust, -normalized, normalized - 1.0)
        )
        upper_jacobian = np.vstack((total[1:], -np.eye(7), np.eye(7)))
        upper_names = (
            *self.problem.engineering_names,
            *self.problem.trust_names,
            *(f"control_{index}_lower" for index in range(7)),
            *(f"control_{index}_upper" for index in range(7)),
        )
        inverse_span = 1.0 / self.assets.theta_span
        result = ExactQPTrial(
            normalized_controls=normalized,
            physical_controls=(
                self.assets.theta_lower + self.assets.theta_span * normalized
            ),
            raw_state=matrices.raw,
            projected_state=projection.state.copy(),
            objective=float(upper_value[0]),
            objective_gradient_normalized=total[0].copy(),
            objective_gradient_physical=total[0] * inverse_span,
            upper_constraint_names=tuple(upper_names),
            upper_constraints=upper_constraints,
            upper_constraint_jacobian_normalized=upper_jacobian,
            upper_constraint_jacobian_physical=(
                upper_jacobian * inverse_span[None, :]
            ),
            engineering_rows=engineering.copy(),
            trust_rows=trust.copy(),
            projection=projection,
            lower_active_set=audit,
            sensitivity=sensitivity,
            independent_final_replay=bool(independent_final_replay),
            elapsed_seconds=perf_counter() - started,
        )
        if not force_cold:
            self._cache[key] = result
        return result

    def _minimum_norm_nonnegative_multipliers(
        self,
        matrix: FloatArray,
        rhs: FloatArray,
    ) -> FloatArray:
        if matrix.shape[1] == 0:
            return np.empty(0, dtype=np.float64)
        fit = lsq_linear(
            matrix,
            rhs,
            bounds=(np.zeros(matrix.shape[1]), np.full(matrix.shape[1], np.inf)),
            method="trf",
            tol=1.0e-12,
            lsq_solver="exact",
            max_iter=10_000,
        )
        if not fit.success or not np.all(np.isfinite(fit.x)):
            raise ActiveSetRefinementError(
                "upper active-multiplier nonnegative least-squares failed."
            )
        multipliers = np.asarray(fit.x, dtype=np.float64)
        singular = linalg.svdvals(matrix, check_finite=True)
        tolerance = (
            max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
            if singular.size
            else 0.0
        )
        rank = int(np.count_nonzero(singular > tolerance))
        if rank < matrix.shape[1]:
            _, _, right_vectors = linalg.svd(
                matrix,
                full_matrices=False,
                check_finite=True,
                lapack_driver="gesdd",
            )
            row_basis = right_vectors[:rank]
            row_rhs = row_basis @ multipliers
            selector = sparse.eye(matrix.shape[1], format="csc")
            constraint = sparse.vstack(
                (sparse.csc_matrix(row_basis), selector), format="csc"
            )
            lower = np.concatenate((row_rhs, np.zeros(matrix.shape[1])))
            upper = np.concatenate(
                (row_rhs, np.full(matrix.shape[1], np.inf))
            )
            solver = osqp.OSQP()
            solver.setup(
                P=sparse.eye(matrix.shape[1], format="csc"),
                q=np.zeros(matrix.shape[1]),
                A=constraint,
                l=lower,
                u=upper,
                eps_abs=1.0e-12,
                eps_rel=1.0e-12,
                max_iter=100_000,
                polishing=True,
                verbose=False,
            )
            solved = solver.solve(raise_error=False)
            candidate = np.asarray(
                solved.x
                if solved.x is not None
                else np.full(matrix.shape[1], np.nan),
                dtype=np.float64,
            )
            reference = matrix @ multipliers
            error = float(np.linalg.norm(matrix @ candidate - reference, ord=np.inf))
            if (
                not np.all(np.isfinite(candidate))
                or np.min(candidate) < -1.0e-9
                or error > 1.0e-9 * max(1.0, float(np.linalg.norm(reference, ord=np.inf)))
            ):
                raise ActiveSetRefinementError(
                    "minimum-norm upper multiplier reconstruction failed its audit."
                )
            multipliers = candidate
        return np.maximum(multipliers, 0.0)

    def audit_upper_kkt(self, trial: ExactQPTrial) -> UpperKKTAudit:
        """Independently reconstruct upper multipliers and first-order residuals."""

        active = np.flatnonzero(
            trial.upper_constraints >= -self.settings.active_tolerance
        )
        active_jacobian = trial.upper_constraint_jacobian_normalized[active]
        multipliers = self._minimum_norm_nonnegative_multipliers(
            active_jacobian.T,
            -trial.objective_gradient_normalized,
        )
        lagrangian_gradient = (
            trial.objective_gradient_normalized
            + active_jacobian.T @ multipliers
        )
        primal = _maximum_positive(trial.upper_constraints)
        dual = _maximum_positive(-multipliers)
        stationarity = float(np.linalg.norm(lagrangian_gradient, ord=np.inf))
        complementarity = float(
            np.linalg.norm(
                multipliers * trial.upper_constraints[active], ord=np.inf
            )
        ) if active.size else 0.0
        tolerance = self.settings.upper_acceptance_tolerance
        feasible = bool(trial.projection.accepted and primal <= tolerance)
        stationary = bool(
            feasible
            and trial.lower_active_set.stable
            and dual <= tolerance
            and stationarity <= tolerance
            and complementarity <= tolerance
        )
        classification = (
            "first_order_kkt_stationary_feasible"
            if stationary
            else (
                "validated_feasible_stationarity_unresolved"
                if feasible
                else "final_feasibility_failed"
            )
        )
        reason = (
            "independent lower and upper KKT audits passed"
            if stationary
            else (
                "the independently projected point is feasible but its upper KKT "
                "residuals do not meet 1e-6"
                if feasible
                else "the independently projected point violates an upper feasibility gate"
            )
        )
        full_multipliers = np.zeros(trial.upper_constraints.size, dtype=np.float64)
        full_multipliers[active] = multipliers
        return UpperKKTAudit(
            active_indices=tuple(int(item) for item in active),
            active_names=tuple(trial.upper_constraint_names[item] for item in active),
            multipliers=full_multipliers,
            primal_residual=primal,
            dual_feasibility_residual=dual,
            stationarity_residual=stationarity,
            complementarity_residual=complementarity,
            feasible=feasible,
            stationary=stationary,
            classification=classification,
            reason=reason,
        )

    def refine(self, normalized_start: npt.ArrayLike) -> ExactQPRefinementResult:
        """Run analytical-gradient SLSQP and independently replay its endpoint."""

        started = perf_counter()
        initial_controls = self._normalized(normalized_start)
        initial: ExactQPTrial | None = None
        final: ExactQPTrial | None = None
        upper_kkt: UpperKKTAudit | None = None
        derivative_error: str | None = None
        derivative_audit: LowerActiveSetAudit | None = None
        solver_success = False
        solver_status = "not_started"
        iterations = 0
        try:
            initial = self.evaluate(initial_controls)
        except ActiveSetDerivativeError as exc:
            derivative_error = str(exc)
            derivative_audit = exc.audit
            return ExactQPRefinementResult(
                initial_controls=initial_controls,
                initial=None,
                final=None,
                upper_kkt=None,
                solver_success=False,
                solver_status="initial_active_set_derivative_unavailable",
                iterations=0,
                distinct_trials=self.distinct_trials,
                cold_qp_resolutions=self.cold_qp_resolutions,
                elapsed_seconds=perf_counter() - started,
                status="active_set_derivative_unavailable",
                derivative_error=derivative_error,
                derivative_audit=derivative_audit,
            )
        except ActiveSetRefinementError as exc:
            return ExactQPRefinementResult(
                initial_controls=initial_controls,
                initial=None,
                final=None,
                upper_kkt=None,
                solver_success=False,
                solver_status="initial_exact_qp_failed",
                iterations=0,
                distinct_trials=self.distinct_trials,
                cold_qp_resolutions=self.cold_qp_resolutions,
                elapsed_seconds=perf_counter() - started,
                status="exact_qp_failed",
                derivative_error=str(exc),
            )

        upper_without_bounds = len(self.problem.engineering_names) + len(
            self.problem.trust_names
        )

        def objective(value: npt.ArrayLike) -> float:
            return self.evaluate(value).objective

        def objective_jacobian(value: npt.ArrayLike) -> FloatArray:
            return self.evaluate(value).objective_gradient_normalized

        def constraints(value: npt.ArrayLike) -> FloatArray:
            return -self.evaluate(value).upper_constraints[:upper_without_bounds]

        def constraint_jacobian(value: npt.ArrayLike) -> FloatArray:
            return -self.evaluate(value).upper_constraint_jacobian_normalized[
                :upper_without_bounds
            ]

        proposed = initial_controls.copy()
        try:
            result = minimize(
                objective,
                initial_controls,
                method="SLSQP",
                jac=objective_jacobian,
                bounds=[(0.0, 1.0)] * 7,
                constraints={
                    "type": "ineq",
                    "fun": constraints,
                    "jac": constraint_jacobian,
                },
                options={
                    "maxiter": self.settings.maximum_iterations,
                    "ftol": self.settings.function_tolerance,
                    "disp": False,
                },
            )
            proposed = self._normalized(result.x)
            solver_success = bool(result.success)
            solver_status = str(result.message)
            iterations = int(result.nit)
        except ActiveSetDerivativeError as exc:
            derivative_error = str(exc)
            derivative_audit = exc.audit
            solver_status = f"active_set_derivative_unavailable: {exc}"
        except ActiveSetRefinementError as exc:
            solver_status = f"exact_qp_evaluation_failed: {exc}"

        # Ensure a normally returned optimizer endpoint is represented in the
        # exact-QP cache even if SLSQP did not request its value last.
        try:
            self.evaluate(proposed)
        except ActiveSetDerivativeError as exc:
            derivative_error = str(exc)
            derivative_audit = exc.audit
        except ActiveSetRefinementError as exc:
            solver_status = f"exact_qp_endpoint_evaluation_failed: {exc}"

        candidates = list(self._cache.values())
        feasible = [
            item
            for item in candidates
            if _maximum_positive(item.upper_constraints)
            <= self.settings.upper_acceptance_tolerance
        ]
        candidate_audits: dict[bytes, UpperKKTAudit] = {}
        for item in feasible:
            try:
                candidate_audits[item.normalized_controls.tobytes()] = (
                    self.audit_upper_kkt(item)
                )
            except ActiveSetRefinementError as exc:
                # A failed multiplier reconstruction cannot be treated as a
                # stationary candidate, but the physical point remains a
                # feasible local incumbent.
                derivative_error = str(exc)
        stationary_candidates = [
            item
            for item in feasible
            if (
                item.normalized_controls.tobytes() in candidate_audits
                and candidate_audits[item.normalized_controls.tobytes()].stationary
            )
        ]
        pool = stationary_candidates or feasible
        selected_cached: ExactQPTrial | None = None
        if pool:
            best_objective = min(item.objective for item in pool)
            tie = 1.0e-10 * max(1.0, abs(best_objective))
            selected_cached = min(
                (item for item in pool if item.objective <= best_objective + tie),
                key=lambda item: tuple(item.normalized_controls.tolist()),
            )
            proposed = selected_cached.normalized_controls
        elif proposed.tobytes() in self._cache:
            selected_cached = self._cache[proposed.tobytes()]

        reproduction_residual: float | None = None
        reproduction_passed: bool | None = None
        try:
            final = self.evaluate(
                proposed,
                force_cold=True,
                independent_final_replay=True,
            )
            upper_kkt = self.audit_upper_kkt(final)
            if selected_cached is None:
                reproduction_passed = False
            else:
                reproduction_residual = float(
                    np.linalg.norm(
                        (
                            final.projected_state
                            - selected_cached.projected_state
                        )
                        / self.assets.model.response_scale,
                        ord=np.inf,
                    )
                )
                reproduction_passed = bool(
                    np.isfinite(reproduction_residual)
                    and reproduction_residual
                    <= self.settings.state_reproduction_tolerance
                )
            if not reproduction_passed:
                upper_kkt = replace(
                    upper_kkt,
                    feasible=False,
                    stationary=False,
                    classification="projection_reproduction_failed",
                    reason=(
                        "the independent final cold QP did not reproduce the "
                        "cached exact-QP state within the scaled 1e-8 tolerance"
                    ),
                )
        except ActiveSetDerivativeError as exc:
            derivative_error = str(exc)
            derivative_audit = exc.audit
            solver_status = f"final_active_set_derivative_unavailable: {exc}"
        except ActiveSetRefinementError as exc:
            solver_status = f"final_exact_qp_failed: {exc}"

        if (
            upper_kkt is not None
            and upper_kkt.stationary
            and reproduction_passed is True
        ):
            status = "validated_stationary"
        elif (
            upper_kkt is not None
            and upper_kkt.feasible
            and reproduction_passed is True
        ):
            status = "validated_feasible_stationarity_unresolved"
        elif reproduction_passed is False:
            status = "projection_reproduction_failed"
        elif derivative_error is not None:
            status = "active_set_derivative_unavailable"
        else:
            status = "refinement_failed"
        return ExactQPRefinementResult(
            initial_controls=initial_controls,
            initial=initial,
            final=final,
            upper_kkt=upper_kkt,
            solver_success=solver_success,
            solver_status=solver_status,
            iterations=iterations,
            distinct_trials=self.distinct_trials,
            cold_qp_resolutions=self.cold_qp_resolutions,
            elapsed_seconds=perf_counter() - started,
            status=status,
            derivative_error=derivative_error,
            derivative_audit=derivative_audit,
            state_reproduction_residual=reproduction_residual,
            state_reproduction_passed=reproduction_passed,
        )


def evaluate_exact_qp_active_set(
    assets: SurrogateNLPAssets,
    case: SurrogateCase,
    normalized_controls: npt.ArrayLike,
    *,
    problem: SurrogateNLP | None = None,
    settings: ActiveSetRefinementSettings | None = None,
    name: str = "v3_active_set_evaluation",
) -> ExactQPTrial:
    """Convenience wrapper for one cold exact-QP value/gradient evaluation."""

    return ExactQPActiveSetRefiner(
        assets,
        case,
        problem=problem,
        settings=settings,
        name=name,
    ).evaluate(normalized_controls)


def refine_exact_qp_active_set(
    assets: SurrogateNLPAssets,
    case: SurrogateCase,
    normalized_start: npt.ArrayLike,
    *,
    problem: SurrogateNLP | None = None,
    settings: ActiveSetRefinementSettings | None = None,
    name: str = "v3_active_set_refinement",
) -> ExactQPRefinementResult:
    """Convenience wrapper for seven-variable exact-QP outer refinement."""

    return ExactQPActiveSetRefiner(
        assets,
        case,
        problem=problem,
        settings=settings,
        name=name,
    ).refine(normalized_start)


__all__ = [
    "ActiveSetDerivativeError",
    "ActiveSetPerturbation",
    "ActiveSetRefinementError",
    "ActiveSetRefinementSettings",
    "ExactQPActiveSetRefiner",
    "ExactQPProjectionError",
    "ExactQPRefinementResult",
    "ExactQPSensitivity",
    "ExactQPTrial",
    "LowerActiveSetAudit",
    "UpperKKTAudit",
    "evaluate_exact_qp_active_set",
    "refine_exact_qp_active_set",
]
