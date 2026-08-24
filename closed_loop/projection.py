"""Statistical surrogate, physical projection, and bounded search utilities.

The module implements the numerical contracts stated in the closed-loop
mixer--reactor--clarifier manuscript.  It deliberately keeps the mechanistic
model out of the regression layer: callers provide accepted mechanistic
states, and this module supplies the fixed quadratic response, the linear
network projection, and a deterministic finite-budget outer search.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from typing import Callable, Iterable, Sequence

import numpy as np
import numpy.typing as npt
import osqp
from scipy import linalg, sparse
from scipy.optimize import lsq_linear


FloatArray = npt.NDArray[np.float64]


class SurrogateValidationError(ValueError):
    """Raised when a scientific precondition for fitting is not satisfied."""


class ProjectionError(RuntimeError):
    """Raised when the physical lower problem cannot be accepted."""


class SearchBudgetError(RuntimeError):
    """Raised when an evaluation is attempted after the declared budget."""


def _finite_array(value: npt.ArrayLike, *, name: str, ndim: int | None = None) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise SurrogateValidationError(f"{name} must be {ndim}-dimensional; got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise SurrogateValidationError(f"{name} contains a non-finite value.")
    return array


def _sample_matrix(value: npt.ArrayLike, *, name: str) -> tuple[FloatArray, bool]:
    array = _finite_array(value, name=name)
    if array.ndim == 1:
        return array[None, :], True
    if array.ndim != 2:
        raise SurrogateValidationError(f"{name} must be a vector or a sample-by-coordinate matrix.")
    return array, False


def _fit_coordinate_scale(
    values: FloatArray,
    *,
    name: str,
    variance_relative_tolerance: float,
) -> tuple[FloatArray, FloatArray]:
    if values.ndim != 2 or values.shape[0] < 2:
        raise SurrogateValidationError(f"{name} requires at least two sample rows.")
    center = np.mean(values, axis=0)
    scale = np.sqrt(np.mean(np.square(values - center), axis=0))
    reference = np.maximum(1.0, np.max(np.abs(values), axis=0))
    bad = np.flatnonzero(scale <= variance_relative_tolerance * reference)
    if bad.size:
        preview = ", ".join(str(int(index)) for index in bad[:10])
        suffix = "..." if bad.size > 10 else ""
        raise SurrogateValidationError(
            f"{name} coordinates [{preview}{suffix}] fail the nonzero-variance rule."
        )
    return center, scale


def _upper_triangle_products(values: FloatArray) -> FloatArray:
    """Return v_j v_k in lexicographic (j, k), j <= k, order."""

    indices = np.triu_indices(values.shape[1])
    return values[:, indices[0]] * values[:, indices[1]]


@dataclass(frozen=True)
class QuadraticFeatureMap:
    """Fitted standardization and serialization for the unique quadratic basis."""

    decision_center: FloatArray
    decision_scale: FloatArray
    influent_center: FloatArray
    influent_scale: FloatArray
    term_center: FloatArray
    term_scale: FloatArray
    variance_relative_tolerance: float = 1.0e-12

    @property
    def decision_count(self) -> int:
        return int(self.decision_center.size)

    @property
    def influent_count(self) -> int:
        return int(self.influent_center.size)

    @property
    def nonconstant_count(self) -> int:
        return int(self.term_center.size)

    @property
    def feature_count(self) -> int:
        return self.nonconstant_count + 1

    @staticmethod
    def expected_feature_count(decision_count: int, influent_count: int) -> int:
        return (
            1
            + decision_count
            + influent_count
            + decision_count * (decision_count + 1) // 2
            + influent_count * (influent_count + 1) // 2
            + decision_count * influent_count
        )

    @staticmethod
    def _unscaled_terms(decisions_standardized: FloatArray, influent_standardized: FloatArray) -> FloatArray:
        decision_quadratic = _upper_triangle_products(decisions_standardized)
        influent_quadratic = _upper_triangle_products(influent_standardized)
        # C-order flattening makes the influent index vary fastest.
        interactions = np.einsum(
            "ni,nj->nij", decisions_standardized, influent_standardized
        ).reshape(decisions_standardized.shape[0], -1)
        return np.concatenate(
            (
                decisions_standardized,
                influent_standardized,
                decision_quadratic,
                influent_quadratic,
                interactions,
            ),
            axis=1,
        )

    @classmethod
    def fit(
        cls,
        decisions: npt.ArrayLike,
        influent: npt.ArrayLike,
        *,
        variance_relative_tolerance: float = 1.0e-12,
    ) -> "QuadraticFeatureMap":
        decision_matrix = _finite_array(decisions, name="decisions", ndim=2)
        influent_matrix = _finite_array(influent, name="influent", ndim=2)
        if decision_matrix.shape[0] != influent_matrix.shape[0]:
            raise SurrogateValidationError("decisions and influent must have the same row count.")
        if decision_matrix.shape[1] == 0 or influent_matrix.shape[1] == 0:
            raise SurrogateValidationError("decision and influent blocks must both be nonempty.")
        decision_center, decision_scale = _fit_coordinate_scale(
            decision_matrix,
            name="decision",
            variance_relative_tolerance=variance_relative_tolerance,
        )
        influent_center, influent_scale = _fit_coordinate_scale(
            influent_matrix,
            name="influent",
            variance_relative_tolerance=variance_relative_tolerance,
        )
        decision_standardized = (decision_matrix - decision_center) / decision_scale
        influent_standardized = (influent_matrix - influent_center) / influent_scale
        terms = cls._unscaled_terms(decision_standardized, influent_standardized)
        term_center, term_scale = _fit_coordinate_scale(
            terms,
            name="nonconstant feature",
            variance_relative_tolerance=variance_relative_tolerance,
        )
        expected = cls.expected_feature_count(decision_matrix.shape[1], influent_matrix.shape[1])
        if terms.shape[1] + 1 != expected:
            raise AssertionError("quadratic feature serialization has an inconsistent size")
        return cls(
            decision_center=decision_center,
            decision_scale=decision_scale,
            influent_center=influent_center,
            influent_scale=influent_scale,
            term_center=term_center,
            term_scale=term_scale,
            variance_relative_tolerance=float(variance_relative_tolerance),
        )

    def transform(self, decisions: npt.ArrayLike, influent: npt.ArrayLike) -> FloatArray:
        decision_matrix, decision_single = _sample_matrix(decisions, name="decisions")
        influent_matrix, influent_single = _sample_matrix(influent, name="influent")
        if decision_single != influent_single:
            raise SurrogateValidationError("decisions and influent must both be vectors or both matrices.")
        if decision_matrix.shape != (influent_matrix.shape[0], self.decision_count):
            raise SurrogateValidationError(
                f"decisions must have shape ({influent_matrix.shape[0]}, {self.decision_count})."
            )
        if influent_matrix.shape[1] != self.influent_count:
            raise SurrogateValidationError(
                f"influent must have {self.influent_count} coordinates; got {influent_matrix.shape[1]}."
            )
        decision_standardized = (decision_matrix - self.decision_center) / self.decision_scale
        influent_standardized = (influent_matrix - self.influent_center) / self.influent_scale
        terms = self._unscaled_terms(decision_standardized, influent_standardized)
        standardized_terms = (terms - self.term_center) / self.term_scale
        features = np.concatenate(
            (np.ones((terms.shape[0], 1), dtype=np.float64), standardized_terms), axis=1
        )
        return features[0] if decision_single else features

    def feature_names(
        self,
        decision_names: Sequence[str] | None = None,
        influent_names: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        decisions = tuple(decision_names or (f"decision_{i}" for i in range(self.decision_count)))
        influents = tuple(influent_names or (f"influent_{i}" for i in range(self.influent_count)))
        if len(decisions) != self.decision_count or len(influents) != self.influent_count:
            raise SurrogateValidationError("feature-name blocks do not match the fitted dimensions.")
        labels: list[str] = list(decisions) + list(influents)
        labels.extend(
            f"{decisions[j]}*{decisions[k]}"
            for j in range(self.decision_count)
            for k in range(j, self.decision_count)
        )
        labels.extend(
            f"{influents[j]}*{influents[k]}"
            for j in range(self.influent_count)
            for k in range(j, self.influent_count)
        )
        labels.extend(f"{decision}*{influent}" for decision in decisions for influent in influents)
        return ("1", *(f"standardized[{label}]" for label in labels))


@dataclass(frozen=True)
class LeastSquaresDiagnostics:
    sample_count: int
    feature_count: int
    response_count: int
    rank_tolerance: float
    smallest_singular_value: float
    largest_singular_value: float
    condition_number: float
    optimality_residual: float
    coefficient_agreement: float
    acceptance_threshold: float
    augmented_condition_number: float | None = None
    condition_times_machine_epsilon: float | None = None
    effective_degrees_of_freedom: float | None = None

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True)
class QuadraticSurrogate:
    """A fixed standardized multiresponse quadratic model."""

    feature_map: QuadraticFeatureMap
    response_center: FloatArray
    response_scale: FloatArray
    coefficients: FloatArray
    diagnostics: LeastSquaresDiagnostics
    ridge_penalty: float = 0.0

    @classmethod
    def fit(
        cls,
        decisions: npt.ArrayLike,
        influent: npt.ArrayLike,
        responses: npt.ArrayLike,
        *,
        variance_relative_tolerance: float = 1.0e-12,
        maximum_condition_number: float = 1.0e8,
        coefficient_acceptance_factor: float = 100.0,
    ) -> "QuadraticSurrogate":
        decision_matrix = _finite_array(decisions, name="decisions", ndim=2)
        influent_matrix = _finite_array(influent, name="influent", ndim=2)
        response_matrix = _finite_array(responses, name="responses", ndim=2)
        row_count = decision_matrix.shape[0]
        if influent_matrix.shape[0] != row_count or response_matrix.shape[0] != row_count:
            raise SurrogateValidationError("all fitting blocks must have the same row count.")
        feature_map = QuadraticFeatureMap.fit(
            decision_matrix,
            influent_matrix,
            variance_relative_tolerance=variance_relative_tolerance,
        )
        design = feature_map.transform(decision_matrix, influent_matrix)
        if row_count < design.shape[1]:
            raise SurrogateValidationError(
                f"the fit has {row_count} rows for {design.shape[1]} coefficients per response."
            )
        response_center, response_scale = _fit_coordinate_scale(
            response_matrix,
            name="response",
            variance_relative_tolerance=variance_relative_tolerance,
        )
        standardized_response = (response_matrix - response_center) / response_scale

        # The SVD is the independent rank/conditioning check and coefficient audit.
        u_svd, singular_values, vt_svd = linalg.svd(
            design, full_matrices=False, check_finite=True, lapack_driver="gesdd"
        )
        epsilon = np.finfo(np.float64).eps
        rank_tolerance = max(design.shape) * epsilon * singular_values[0]
        smallest = float(singular_values[-1])
        condition = float(singular_values[0] / singular_values[-1]) if smallest > 0.0 else np.inf
        if smallest <= rank_tolerance:
            raise SurrogateValidationError(
                "the standardized quadratic design is not full column rank at the declared tolerance."
            )
        if condition > maximum_condition_number:
            raise SurrogateValidationError(
                f"design condition number {condition:.6g} exceeds {maximum_condition_number:.6g}."
            )

        q_qr, r_qr, pivot = linalg.qr(
            design, mode="economic", pivoting=True, check_finite=True
        )
        permuted_coefficients = linalg.solve_triangular(
            r_qr, q_qr.T @ standardized_response, lower=False, check_finite=True
        )
        coefficients_qr = np.empty_like(permuted_coefficients)
        coefficients_qr[pivot, :] = permuted_coefficients
        # One QR-based iterative-refinement step prevents a nearly exact
        # polynomial response from failing the declared normal-equation audit
        # solely because its residual is at roundoff scale.
        preliminary_residual = design @ coefficients_qr - standardized_response
        permuted_correction = linalg.solve_triangular(
            r_qr, q_qr.T @ (-preliminary_residual), lower=False, check_finite=True
        )
        correction = np.empty_like(permuted_correction)
        correction[pivot, :] = permuted_correction
        coefficients_qr += correction
        coefficients_svd = (
            (vt_svd.T / singular_values[None, :]) @ (u_svd.T @ standardized_response)
        )

        residual = design @ coefficients_qr - standardized_response
        residual_norm = float(linalg.norm(residual, ord="fro"))
        optimality = float(linalg.norm(design.T @ residual, ord="fro")) / max(
            1.0, float(singular_values[0]) * residual_norm
        )
        agreement = float(linalg.norm(coefficients_qr - coefficients_svd, ord="fro")) / max(
            1.0, float(linalg.norm(coefficients_svd, ord="fro"))
        )
        threshold = float(coefficient_acceptance_factor * condition * epsilon)
        if not np.all(np.isfinite(coefficients_qr)) or max(optimality, agreement) > threshold:
            raise SurrogateValidationError(
                "QR/SVD coefficient acceptance failed: "
                f"optimality={optimality:.3e}, agreement={agreement:.3e}, limit={threshold:.3e}."
            )

        diagnostics = LeastSquaresDiagnostics(
            sample_count=row_count,
            feature_count=design.shape[1],
            response_count=response_matrix.shape[1],
            rank_tolerance=float(rank_tolerance),
            smallest_singular_value=smallest,
            largest_singular_value=float(singular_values[0]),
            condition_number=condition,
            optimality_residual=optimality,
            coefficient_agreement=agreement,
            acceptance_threshold=threshold,
        )
        return cls(
            feature_map=feature_map,
            response_center=response_center,
            response_scale=response_scale,
            coefficients=coefficients_qr.T,
            diagnostics=diagnostics,
            ridge_penalty=0.0,
        )

    @classmethod
    def fit_ridge(
        cls,
        decisions: npt.ArrayLike,
        influent: npt.ArrayLike,
        responses: npt.ArrayLike,
        *,
        ridge_penalty: float,
        variance_relative_tolerance: float = 1.0e-12,
    ) -> "QuadraticSurrogate":
        """Fit the manuscript ridge estimator, leaving the intercept unpenalized.

        The coefficient estimate is the column-pivoted-QR solution of the
        augmented least-squares system.  A separate divide-and-conquer SVD
        solve supplies the coefficient audit; forming and solving the ridge
        normal equations is deliberately avoided.
        """

        if not np.isfinite(ridge_penalty) or ridge_penalty <= 0.0:
            raise SurrogateValidationError("ridge_penalty must be finite and positive.")
        decision_matrix = _finite_array(decisions, name="decisions", ndim=2)
        influent_matrix = _finite_array(influent, name="influent", ndim=2)
        response_matrix = _finite_array(responses, name="responses", ndim=2)
        rows = decision_matrix.shape[0]
        if influent_matrix.shape[0] != rows or response_matrix.shape[0] != rows:
            raise SurrogateValidationError("all fitting blocks must have the same row count.")
        feature_map = QuadraticFeatureMap.fit(
            decision_matrix, influent_matrix,
            variance_relative_tolerance=variance_relative_tolerance,
        )
        design = feature_map.transform(decision_matrix, influent_matrix)
        response_center, response_scale = _fit_coordinate_scale(
            response_matrix, name="response",
            variance_relative_tolerance=variance_relative_tolerance,
        )
        standardized = (response_matrix - response_center) / response_scale
        feature_count = design.shape[1]
        penalty_diagonal = np.ones(feature_count, dtype=np.float64)
        penalty_diagonal[0] = 0.0
        penalty_operator = np.diag(penalty_diagonal)
        ridge_scale = np.sqrt(float(rows)) * np.sqrt(ridge_penalty)
        augmented_design = np.vstack((design, ridge_scale * penalty_operator))
        augmented_response = np.vstack(
            (standardized, np.zeros((feature_count, response_matrix.shape[1])))
        )

        # This solve is the estimator named by the manuscript.
        q_qr, r_qr, pivot = linalg.qr(
            augmented_design, mode="economic", pivoting=True, check_finite=True
        )
        permuted_coefficients = linalg.solve_triangular(
            r_qr,
            q_qr.T @ augmented_response,
            lower=False,
            check_finite=True,
        )
        coefficients_qr = np.empty_like(permuted_coefficients)
        coefficients_qr[pivot, :] = permuted_coefficients

        # This independent thin SVD supplies both the second coefficient solve
        # and the augmented-system condition gate.
        u_svd, singular_values, vt_svd = linalg.svd(
            augmented_design,
            full_matrices=False,
            check_finite=True,
            lapack_driver="gesdd",
        )
        epsilon = np.finfo(np.float64).eps
        rank_tolerance = max(augmented_design.shape) * epsilon * singular_values[0]
        svd_rank = int(np.count_nonzero(singular_values > rank_tolerance))
        coefficients_svd = (
            (vt_svd.T / singular_values[None, :])
            @ (u_svd.T @ augmented_response)
        )
        smallest = float(singular_values[-1])
        largest = float(singular_values[0])
        condition = largest / smallest if smallest > 0.0 else np.inf
        condition_product = condition * epsilon
        if svd_rank != feature_count or condition_product > 1.0e-8:
            raise SurrogateValidationError(
                "augmented ridge-system condition gate failed: "
                f"rank={svd_rank}/{feature_count}, "
                f"kappa*eps={condition_product:.3e} > 1.000e-08."
            )

        coefficient_scale = 1.0 + max(
            float(np.max(np.abs(coefficients_qr))),
            float(np.max(np.abs(coefficients_svd))),
        )
        agreement = float(np.max(np.abs(coefficients_qr - coefficients_svd))) / coefficient_scale
        threshold = float(100.0 * condition_product)
        if (
            not np.all(np.isfinite(coefficients_qr))
            or not np.all(np.isfinite(coefficients_svd))
            or agreement > threshold
        ):
            raise SurrogateValidationError(
                "augmented QR/SVD coefficient acceptance failed: "
                f"agreement={agreement:.3e}, limit={threshold:.3e}."
            )

        augmented_residual = augmented_design @ coefficients_qr - augmented_response
        augmented_gradient = augmented_design.T @ augmented_residual
        optimality = float(linalg.norm(augmented_gradient, ord="fro")) / max(
            1.0,
            largest * float(linalg.norm(augmented_residual, ord="fro")),
        )

        # If A[:, pivot] = Q R, then M_R^{-1}=R^{-1}R^{-T} in
        # pivoted coordinates.  This evaluates tr(Phi M_R^{-1} Phi^T)
        # without using normal equations.
        design_pivoted_times_r_inverse = linalg.solve_triangular(
            r_qr.T,
            design[:, pivot].T,
            lower=True,
            check_finite=True,
        ).T
        effective_degrees_of_freedom = float(
            np.sum(np.square(design_pivoted_times_r_inverse))
        )
        diagnostics = LeastSquaresDiagnostics(
            sample_count=rows,
            feature_count=feature_count,
            response_count=response_matrix.shape[1],
            rank_tolerance=float(rank_tolerance),
            smallest_singular_value=smallest,
            largest_singular_value=largest,
            condition_number=condition,
            optimality_residual=optimality,
            coefficient_agreement=agreement,
            acceptance_threshold=threshold,
            augmented_condition_number=condition,
            condition_times_machine_epsilon=condition_product,
            effective_degrees_of_freedom=effective_degrees_of_freedom,
        )
        return cls(
            feature_map=feature_map,
            response_center=response_center,
            response_scale=response_scale,
            coefficients=coefficients_qr.T,
            diagnostics=diagnostics,
            ridge_penalty=float(ridge_penalty),
        )

    @property
    def response_count(self) -> int:
        return int(self.response_center.size)

    @property
    def effective_degrees_of_freedom(self) -> float:
        """Return the ridge hat-matrix trace (or the OLS column count)."""

        value = self.diagnostics.effective_degrees_of_freedom
        return float(self.feature_map.feature_count if value is None else value)

    def predict_standardized(self, decisions: npt.ArrayLike, influent: npt.ArrayLike) -> FloatArray:
        features = self.feature_map.transform(decisions, influent)
        if features.ndim == 1:
            return self.coefficients @ features
        return features @ self.coefficients.T

    def predict(self, decisions: npt.ArrayLike, influent: npt.ArrayLike) -> FloatArray:
        standardized = self.predict_standardized(decisions, influent)
        return self.response_center + standardized * self.response_scale


@dataclass(frozen=True)
class NetworkLayout:
    """Coordinate layout for ``chi=(m,c_1,...,c_N,g_E,g_U,M_cl)``.

    ``layer_count`` remains part of the mechanistic Clarifier geometry, but
    the statistical response contains only its total solids inventory.  The
    envelope implemented here assumes that total Clarifier volume is divided
    equally among those mechanistic layers.
    """

    stage_count: int = 5
    component_count: int = 20
    layer_count: int = 10
    soluble_indices: tuple[int, ...] = tuple(range(10))
    particulate_indices: tuple[int, ...] = tuple(range(10, 20))

    def __post_init__(self) -> None:
        if self.stage_count < 1 or self.component_count < 1 or self.layer_count < 3:
            raise SurrogateValidationError("network dimensions are outside the supported range.")
        soluble = tuple(int(index) for index in self.soluble_indices)
        particulate = tuple(int(index) for index in self.particulate_indices)
        if sorted(soluble + particulate) != list(range(self.component_count)):
            raise SurrogateValidationError(
                "soluble_indices and particulate_indices must partition all components exactly once."
            )

    @property
    def state_size(self) -> int:
        return (self.stage_count + 3) * self.component_count + 1

    @property
    def equality_count_without_invariants(self) -> int:
        return 2 * self.component_count + len(self.soluble_indices)

    @property
    def inequality_count(self) -> int:
        return len(self.particulate_indices) + 2

    @property
    def mixer_slice(self) -> slice:
        return slice(0, self.component_count)

    def reactor_slice(self, stage: int) -> slice:
        if not 0 <= stage < self.stage_count:
            raise IndexError("reactor stage is out of range")
        start = (stage + 1) * self.component_count
        return slice(start, start + self.component_count)

    @property
    def overflow_flow_slice(self) -> slice:
        start = (self.stage_count + 1) * self.component_count
        return slice(start, start + self.component_count)

    @property
    def underflow_flow_slice(self) -> slice:
        start = (self.stage_count + 2) * self.component_count
        return slice(start, start + self.component_count)

    @property
    def inventory_index(self) -> int:
        return (self.stage_count + 3) * self.component_count

    @property
    def inventory_slice(self) -> slice:
        return slice(self.inventory_index, self.inventory_index + 1)


@dataclass(frozen=True)
class NetworkOperators:
    layout: NetworkLayout
    equality_matrix: FloatArray
    equality_rhs: FloatArray
    inequality_matrix: FloatArray
    primary_flow: float
    clarifier_flow: float
    underflow: float
    effluent_flow: float
    clarifier_volume_m3: float


def build_network_operators(
    influent: npt.ArrayLike,
    *,
    internal_recycle: float,
    return_recycle: float,
    waste_fraction: float,
    invariant_operator: npt.ArrayLike,
    tss_weights: npt.ArrayLike,
    layout: NetworkLayout | None = None,
    clarifier_volume_m3: float = 6_000.0,
) -> NetworkOperators:
    """Assemble the manuscript's ordered H chi=b and G chi<=0 matrices.

    ``clarifier_volume_m3`` is divided equally over ``layout.layer_count``;
    unequal layer volumes are outside this reduced projection contract.
    """

    layout = layout or NetworkLayout()
    component_count = layout.component_count
    state_size = layout.state_size
    x = _finite_array(influent, name="influent", ndim=1)
    invariant = _finite_array(invariant_operator, name="invariant_operator", ndim=2)
    tss = _finite_array(tss_weights, name="tss_weights", ndim=1)
    if x.size != component_count or invariant.shape[1] != component_count or tss.size != component_count:
        raise SurrogateValidationError("network component dimensions are inconsistent.")
    if invariant.shape[0] == 0 or np.linalg.matrix_rank(invariant) != invariant.shape[0]:
        raise SurrogateValidationError("invariant_operator must have full row rank.")
    if internal_recycle < 0.0 or return_recycle < 0.0 or not 0.0 < waste_fraction < 1.0:
        raise SurrogateValidationError("recycle ratios must be nonnegative and 0 < waste_fraction < 1.")
    if not np.isfinite(clarifier_volume_m3) or clarifier_volume_m3 <= 0.0:
        raise SurrogateValidationError("clarifier_volume_m3 must be positive and finite.")
    underflow = float(return_recycle + waste_fraction)
    if underflow <= 0.0:
        raise SurrogateValidationError("return plus waste flow must be positive.")
    primary_flow = float(1.0 + internal_recycle + return_recycle)
    clarifier_flow = float(1.0 + return_recycle)
    effluent_flow = float(1.0 - waste_fraction)

    invariant_count = invariant.shape[0]
    equality_count = (
        component_count
        + layout.stage_count * invariant_count
        + component_count
        + len(layout.soluble_indices)
    )
    equality = np.zeros((equality_count, state_size), dtype=np.float64)
    rhs = np.zeros(equality_count, dtype=np.float64)
    identity = np.eye(component_count, dtype=np.float64)
    row = 0

    equality[row : row + component_count, layout.mixer_slice] = primary_flow * identity
    equality[row : row + component_count, layout.reactor_slice(layout.stage_count - 1)] = (
        -float(internal_recycle) * identity
    )
    equality[row : row + component_count, layout.underflow_flow_slice] = (
        -float(return_recycle) / underflow * identity
    )
    rhs[row : row + component_count] = x
    row += component_count

    previous = layout.mixer_slice
    for stage in range(layout.stage_count):
        current = layout.reactor_slice(stage)
        equality[row : row + invariant_count, current] = invariant
        equality[row : row + invariant_count, previous] = -invariant
        row += invariant_count
        previous = current

    final_reactor = layout.reactor_slice(layout.stage_count - 1)
    equality[row : row + component_count, layout.overflow_flow_slice] = identity
    equality[row : row + component_count, layout.underflow_flow_slice] = identity
    equality[row : row + component_count, final_reactor] = -clarifier_flow * identity
    row += component_count

    for component in layout.soluble_indices:
        equality[row, layout.underflow_flow_slice.start + component] = 1.0
        equality[row, final_reactor.start + component] = -underflow
        row += 1

    if row != equality_count:
        raise AssertionError("equality row assembly is inconsistent")

    inequality = np.zeros((layout.inequality_count, state_size), dtype=np.float64)
    row = 0
    for component in layout.particulate_indices:
        inequality[row, final_reactor.start + component] = underflow
        inequality[row, layout.underflow_flow_slice.start + component] = -1.0
        row += 1
    endpoint_layer_volume = float(clarifier_volume_m3) / layout.layer_count
    remaining_volume = float(clarifier_volume_m3) - endpoint_layer_volume
    inequality[row, layout.overflow_flow_slice] = (
        underflow * remaining_volume * tss
    )
    inequality[row, layout.underflow_flow_slice] = (
        effluent_flow * endpoint_layer_volume * tss
    )
    inequality[row, layout.inventory_index] = -effluent_flow * underflow
    row += 1
    inequality[row, layout.overflow_flow_slice] = (
        -underflow * endpoint_layer_volume * tss
    )
    inequality[row, layout.underflow_flow_slice] = (
        -effluent_flow * remaining_volume * tss
    )
    inequality[row, layout.inventory_index] = effluent_flow * underflow
    row += 1
    if row != layout.inequality_count:
        raise AssertionError("inequality row assembly is inconsistent")

    return NetworkOperators(
        layout=layout,
        equality_matrix=equality,
        equality_rhs=rhs,
        inequality_matrix=inequality,
        primary_flow=primary_flow,
        clarifier_flow=clarifier_flow,
        underflow=underflow,
        effluent_flow=effluent_flow,
        clarifier_volume_m3=float(clarifier_volume_m3),
    )


def no_conversion_feasible_state(
    influent: npt.ArrayLike,
    *,
    operators: NetworkOperators,
    tss_weights: npt.ArrayLike,
) -> FloatArray:
    """Construct the analytical feasible point used to prove QP nonemptiness."""

    x = _finite_array(influent, name="influent", ndim=1)
    tss = _finite_array(tss_weights, name="tss_weights", ndim=1)
    layout = operators.layout
    if x.size != layout.component_count or tss.size != layout.component_count:
        raise SurrogateValidationError("no-conversion state has inconsistent component dimensions.")
    state = np.empty(layout.state_size, dtype=np.float64)
    state[layout.mixer_slice] = x
    for stage in range(layout.stage_count):
        state[layout.reactor_slice(stage)] = x
    state[layout.overflow_flow_slice] = operators.effluent_flow * x
    state[layout.underflow_flow_slice] = operators.underflow * x
    state[layout.inventory_index] = operators.clarifier_volume_m3 * float(tss @ x)
    return state


@dataclass(frozen=True)
class NetworkRowScales:
    equality: FloatArray
    inequality: FloatArray


def _flow_vector(value: npt.ArrayLike, row_count: int, *, name: str) -> FloatArray:
    array = _finite_array(value, name=name)
    if array.ndim == 0:
        return np.full(row_count, float(array), dtype=np.float64)
    if array.ndim != 1 or array.size != row_count:
        raise SurrogateValidationError(f"{name} must be scalar or have one value per state row.")
    return array


def fit_network_row_scales(
    states: npt.ArrayLike,
    influents: npt.ArrayLike,
    *,
    internal_recycle: npt.ArrayLike,
    return_recycle: npt.ArrayLike,
    waste_fraction: npt.ArrayLike,
    invariant_operator: npt.ArrayLike,
    tss_weights: npt.ArrayLike,
    layout: NetworkLayout | None = None,
    clarifier_volume_m3: float = 6_000.0,
    minimum_scale: float = 1.0e-12,
) -> NetworkRowScales:
    """Fit D_b and D_g from the named physical terms in the manuscript."""

    layout = layout or NetworkLayout()
    state_matrix = _finite_array(states, name="states", ndim=2)
    influent_matrix = _finite_array(influents, name="influents", ndim=2)
    invariant = _finite_array(invariant_operator, name="invariant_operator", ndim=2)
    tss = _finite_array(tss_weights, name="tss_weights", ndim=1)
    row_count = state_matrix.shape[0]
    if state_matrix.shape[1] != layout.state_size:
        raise SurrogateValidationError(f"states must have {layout.state_size} columns.")
    if influent_matrix.shape != (row_count, layout.component_count):
        raise SurrogateValidationError("influents do not match the state rows and component count.")
    if invariant.shape[1] != layout.component_count or tss.size != layout.component_count:
        raise SurrogateValidationError("invariant or TSS component dimensions are inconsistent.")
    if minimum_scale <= 0.0:
        raise SurrogateValidationError("minimum_scale must be positive.")
    if not np.isfinite(clarifier_volume_m3) or clarifier_volume_m3 <= 0.0:
        raise SurrogateValidationError("clarifier_volume_m3 must be positive and finite.")
    r_internal = _flow_vector(internal_recycle, row_count, name="internal_recycle")
    r_return = _flow_vector(return_recycle, row_count, name="return_recycle")
    waste = _flow_vector(waste_fraction, row_count, name="waste_fraction")
    if np.any(r_internal < 0.0) or np.any(r_return < 0.0) or np.any((waste <= 0.0) | (waste >= 1.0)):
        raise SurrogateValidationError("row-scale flow inputs are outside their physical ranges.")
    q_primary = 1.0 + r_internal + r_return
    q_clarifier = 1.0 + r_return
    q_underflow = r_return + waste
    q_effluent = 1.0 - waste

    invariant_count = invariant.shape[0]
    equality_count = (
        layout.component_count
        + layout.stage_count * invariant_count
        + layout.component_count
        + len(layout.soluble_indices)
    )
    equality_mean_square = np.zeros((row_count, equality_count), dtype=np.float64)
    equality_term_count = np.zeros(equality_count, dtype=np.float64)
    inequality_mean_square = np.zeros((row_count, layout.inequality_count), dtype=np.float64)
    inequality_term_count = np.zeros(layout.inequality_count, dtype=np.float64)

    mixer = state_matrix[:, layout.mixer_slice]
    final_reactor = state_matrix[:, layout.reactor_slice(layout.stage_count - 1)]
    overflow = state_matrix[:, layout.overflow_flow_slice]
    underflow_flow = state_matrix[:, layout.underflow_flow_slice]
    clarifier_inventory = state_matrix[:, layout.inventory_index]
    row = 0

    mixer_terms = (
        q_primary[:, None] * mixer,
        -influent_matrix,
        -r_internal[:, None] * final_reactor,
        -(r_return / q_underflow)[:, None] * underflow_flow,
    )
    for term in mixer_terms:
        equality_mean_square[:, row : row + layout.component_count] += np.square(term)
    equality_term_count[row : row + layout.component_count] = len(mixer_terms)
    row += layout.component_count

    previous = mixer
    for stage in range(layout.stage_count):
        current = state_matrix[:, layout.reactor_slice(stage)]
        current_invariant = current @ invariant.T
        previous_invariant = previous @ invariant.T
        equality_mean_square[:, row : row + invariant_count] = (
            np.square(current_invariant) + np.square(previous_invariant)
        )
        equality_term_count[row : row + invariant_count] = 2.0
        row += invariant_count
        previous = current

    clarifier_terms = (overflow, underflow_flow, -q_clarifier[:, None] * final_reactor)
    for term in clarifier_terms:
        equality_mean_square[:, row : row + layout.component_count] += np.square(term)
    equality_term_count[row : row + layout.component_count] = len(clarifier_terms)
    row += layout.component_count

    for component in layout.soluble_indices:
        equality_mean_square[:, row] = (
            np.square(underflow_flow[:, component])
            + np.square(q_underflow * final_reactor[:, component])
        )
        equality_term_count[row] = 2.0
        row += 1

    if row != equality_count:
        raise AssertionError("equality scale serialization is inconsistent")

    row = 0
    for component in layout.particulate_indices:
        inequality_mean_square[:, row] = (
            np.square(q_underflow * final_reactor[:, component])
            + np.square(underflow_flow[:, component])
        )
        inequality_term_count[row] = 2.0
        row += 1
    endpoint_layer_volume = float(clarifier_volume_m3) / layout.layer_count
    remaining_volume = float(clarifier_volume_m3) - endpoint_layer_volume
    overflow_tss_flow = overflow @ tss
    underflow_tss_flow = underflow_flow @ tss
    lower_inventory_terms = (
        q_underflow * remaining_volume * overflow_tss_flow,
        q_effluent * endpoint_layer_volume * underflow_tss_flow,
        -q_effluent * q_underflow * clarifier_inventory,
    )
    for term in lower_inventory_terms:
        inequality_mean_square[:, row] += np.square(term)
    inequality_term_count[row] = len(lower_inventory_terms)
    row += 1
    upper_inventory_terms = (
        q_effluent * q_underflow * clarifier_inventory,
        -q_underflow * endpoint_layer_volume * overflow_tss_flow,
        -q_effluent * remaining_volume * underflow_tss_flow,
    )
    for term in upper_inventory_terms:
        inequality_mean_square[:, row] += np.square(term)
    inequality_term_count[row] = len(upper_inventory_terms)
    row += 1
    if row != layout.inequality_count:
        raise AssertionError("inequality scale serialization is inconsistent")

    equality_scale = np.sqrt(
        np.mean(equality_mean_square / equality_term_count[None, :], axis=0)
    )
    inequality_scale = np.sqrt(
        np.mean(inequality_mean_square / inequality_term_count[None, :], axis=0)
    )
    equality_scale = np.maximum(minimum_scale, equality_scale)
    inequality_scale = np.maximum(minimum_scale, inequality_scale)
    if not np.all(np.isfinite(equality_scale)) or not np.all(np.isfinite(inequality_scale)):
        raise SurrogateValidationError("a fitted network row scale is non-finite.")
    return NetworkRowScales(equality=equality_scale, inequality=inequality_scale)


@dataclass(frozen=True)
class ProjectionWarmStart:
    displacement: FloatArray
    dual: FloatArray


@dataclass(frozen=True)
class ProjectionDiagnostics:
    status: str
    status_value: int
    iterations: int
    equality_rank_tolerance: float
    equality_smallest_singular_value: float
    equality_condition_number: float
    equality_residual: float
    inequality_residual: float
    nonnegativity_residual: float
    dual_feasibility_residual: float
    stationarity_residual: float
    complementarity_residual: float
    retried_cold: bool
    active_inequality_count: int = 0
    multipliers_reconstructed: bool = False
    solver_attempts: int = 1
    fallback_used: bool = False

    def as_dict(self) -> dict[str, str | int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectionResult:
    state: FloatArray
    displacement: FloatArray
    equality_multipliers: FloatArray
    inequality_multipliers: FloatArray
    inequality_slack: FloatArray
    diagnostics: ProjectionDiagnostics
    accepted: bool

    @property
    def warm_start(self) -> ProjectionWarmStart:
        dual = np.concatenate((self.equality_multipliers, self.inequality_multipliers))
        return ProjectionWarmStart(self.displacement.copy(), dual)


def affine_projection(
    raw_state: npt.ArrayLike,
    equality_matrix: npt.ArrayLike,
    equality_rhs: npt.ArrayLike,
    state_scale: npt.ArrayLike,
) -> FloatArray:
    """Return the equality-only scaled-L2 projection used in assessment."""

    raw = _finite_array(raw_state, name="raw_state", ndim=1)
    equality = _finite_array(equality_matrix, name="equality_matrix", ndim=2)
    rhs = _finite_array(equality_rhs, name="equality_rhs", ndim=1)
    scale = _finite_array(state_scale, name="state_scale", ndim=1)
    if equality.shape != (rhs.size, raw.size) or scale.size != raw.size or np.any(scale <= 0.0):
        raise SurrogateValidationError("affine projection dimensions or state scales are invalid.")
    scaled_equality = equality * scale[None, :]
    singular_values = linalg.svdvals(scaled_equality, check_finite=True)
    tolerance = max(scaled_equality.shape) * np.finfo(np.float64).eps * singular_values[0]
    if singular_values[-1] <= tolerance:
        raise ProjectionError("scaled equality matrix is not full row rank.")
    required = rhs - equality @ raw
    displacement, _, rank, _ = linalg.lstsq(
        scaled_equality, required, cond=None, lapack_driver="gelsd", check_finite=True
    )
    if rank != equality.shape[0]:
        raise ProjectionError("affine projection solve lost equality rank.")
    projected = raw + scale * displacement
    residual = float(np.linalg.norm(equality @ projected - rhs, ord=np.inf))
    reference = max(1.0, float(np.linalg.norm(rhs, ord=np.inf)))
    if residual > 1.0e-10 * reference:
        raise ProjectionError(f"affine projection equality residual is {residual:.3e}.")
    return projected


class PhysicalProjector:
    """Solve and independently accept the strictly convex network QP."""

    def __init__(
        self,
        state_scale: npt.ArrayLike,
        equality_scale: npt.ArrayLike,
        inequality_scale: npt.ArrayLike,
        *,
        absolute_tolerance: float = 1.0e-8,
        relative_tolerance: float = 1.0e-8,
        maximum_iterations: int = 100_000,
        polish: bool = True,
        equality_acceptance_tolerance: float = 1.0e-8,
        inequality_acceptance_tolerance: float = 1.0e-8,
        nonnegativity_acceptance_tolerance: float = 1.0e-10,
        active_set_tolerance: float = 1.0e-7,
    ) -> None:
        self.state_scale = _finite_array(state_scale, name="state_scale", ndim=1).copy()
        self.equality_scale = _finite_array(equality_scale, name="equality_scale", ndim=1).copy()
        self.inequality_scale = _finite_array(
            inequality_scale, name="inequality_scale", ndim=1
        ).copy()
        if (
            np.any(self.state_scale <= 0.0)
            or np.any(self.equality_scale <= 0.0)
            or np.any(self.inequality_scale <= 0.0)
        ):
            raise SurrogateValidationError("all projection scales must be strictly positive.")
        if absolute_tolerance <= 0.0 or relative_tolerance <= 0.0 or maximum_iterations < 1:
            raise SurrogateValidationError("OSQP tolerances and iteration limit must be positive.")
        if not np.isfinite(active_set_tolerance) or active_set_tolerance <= 0.0:
            raise SurrogateValidationError("active_set_tolerance must be finite and positive.")
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = float(relative_tolerance)
        self.maximum_iterations = int(maximum_iterations)
        self.polish = bool(polish)
        self.equality_acceptance_tolerance = float(equality_acceptance_tolerance)
        self.inequality_acceptance_tolerance = float(inequality_acceptance_tolerance)
        self.nonnegativity_acceptance_tolerance = float(nonnegativity_acceptance_tolerance)
        self.active_set_tolerance = float(active_set_tolerance)

    @staticmethod
    def _positive_part_norm(values: FloatArray) -> float:
        if values.size == 0:
            return 0.0
        return float(np.max(np.maximum(values, 0.0)))

    def _solve_once(
        self,
        *,
        scaled_equality: FloatArray,
        equality_rhs: FloatArray,
        scaled_inequality: FloatArray,
        inequality_rhs: FloatArray,
        warm_start: ProjectionWarmStart | None,
        rho: float | None = None,
        adaptive_rho: bool | None = None,
        maximum_iterations: int | None = None,
    ) -> object:
        variable_count = self.state_scale.size
        constraint_matrix = sparse.csc_matrix(
            np.vstack((scaled_equality, scaled_inequality)), dtype=np.float64
        )
        lower = np.concatenate(
            (equality_rhs, np.full(inequality_rhs.size, -np.inf, dtype=np.float64))
        )
        upper = np.concatenate((equality_rhs, inequality_rhs))
        solver = osqp.OSQP()
        setup_options: dict[str, object] = {}
        if rho is not None:
            setup_options["rho"] = float(rho)
        if adaptive_rho is not None:
            setup_options["adaptive_rho"] = bool(adaptive_rho)
        solver.setup(
            P=sparse.eye(variable_count, format="csc", dtype=np.float64),
            q=np.zeros(variable_count, dtype=np.float64),
            A=constraint_matrix,
            l=lower,
            u=upper,
            eps_abs=self.absolute_tolerance,
            eps_rel=self.relative_tolerance,
            max_iter=(
                self.maximum_iterations
                if maximum_iterations is None
                else int(maximum_iterations)
            ),
            polishing=self.polish,
            verbose=False,
            **setup_options,
        )
        if warm_start is not None:
            displacement = _finite_array(
                warm_start.displacement, name="warm displacement", ndim=1
            )
            dual = _finite_array(warm_start.dual, name="warm dual", ndim=1)
            if displacement.size != variable_count or dual.size != constraint_matrix.shape[0]:
                raise SurrogateValidationError("projection warm-start dimensions are inconsistent.")
            solver.warm_start(x=displacement, y=dual)
        return solver.solve(raise_error=False)

    def _reconstruct_multipliers(
        self,
        displacement: FloatArray,
        scaled_equality: FloatArray,
        scaled_inequality: FloatArray,
        inequality_rhs: FloatArray,
    ) -> tuple[FloatArray, FloatArray, int]:
        """Recover an independently audited KKT multiplier representation.

        BVLS is used because its active-set termination directly controls the
        residual of this small dense bounded problem.  The previous TRF solve
        could declare first-order convergence while leaving a stationarity
        residual above the projection gate, and a subsequent minimum-norm QP
        could introduce another unrelated numerical failure.  No
        solver-reported projection-QP dual is used.
        """

        equality_count = scaled_equality.shape[0]
        inequality_count = scaled_inequality.shape[0]
        inequality_value = scaled_inequality @ displacement - inequality_rhs
        active = np.flatnonzero(inequality_value >= -self.active_set_tolerance)
        multiplier_matrix = np.column_stack(
            (scaled_equality.T, scaled_inequality[active].T)
        )
        lower = np.concatenate(
            (
                np.full(equality_count, -np.inf, dtype=np.float64),
                np.zeros(active.size, dtype=np.float64),
            )
        )
        upper = np.full(lower.size, np.inf, dtype=np.float64)
        least_squares = lsq_linear(
            multiplier_matrix,
            -displacement,
            bounds=(lower, upper),
            method="bvls",
            tol=1.0e-12,
            max_iter=10_000,
        )
        if not least_squares.success or not np.all(np.isfinite(least_squares.x)):
            raise ProjectionError(
                "active-set multiplier reconstruction did not solve its bounded least-squares problem."
            )
        multipliers = np.asarray(least_squares.x, dtype=np.float64)

        # Enforce the declared sign domain exactly; bounded solvers can return
        # negative roundoff at an active lower bound.
        multipliers[equality_count:] = np.maximum(
            multipliers[equality_count:], 0.0
        )

        equality_multipliers = multipliers[:equality_count]
        inequality_multipliers = np.zeros(inequality_count, dtype=np.float64)
        inequality_multipliers[active] = multipliers[equality_count:]
        return equality_multipliers, inequality_multipliers, int(active.size)

    def project(
        self,
        raw_state: npt.ArrayLike,
        equality_matrix: npt.ArrayLike,
        equality_rhs: npt.ArrayLike,
        inequality_matrix: npt.ArrayLike,
        *,
        warm_start: ProjectionWarmStart | None = None,
        raise_on_failure: bool = True,
    ) -> ProjectionResult:
        raw = _finite_array(raw_state, name="raw_state", ndim=1)
        equality = _finite_array(equality_matrix, name="equality_matrix", ndim=2)
        rhs = _finite_array(equality_rhs, name="equality_rhs", ndim=1)
        inequality = _finite_array(inequality_matrix, name="inequality_matrix", ndim=2)
        variable_count = raw.size
        if variable_count != self.state_scale.size:
            raise SurrogateValidationError("raw state and state_scale sizes differ.")
        if equality.shape != (self.equality_scale.size, variable_count) or rhs.size != equality.shape[0]:
            raise SurrogateValidationError("equality dimensions do not match their fitted scales.")
        if inequality.shape != (self.inequality_scale.size, variable_count):
            raise SurrogateValidationError("inequality dimensions do not match their fitted scales.")

        scaled_equality = (
            equality * self.state_scale[None, :] / self.equality_scale[:, None]
        )
        required_equality = (rhs - equality @ raw) / self.equality_scale
        physical_scaled_inequality = inequality / self.inequality_scale[:, None]
        scaled_network_inequality = physical_scaled_inequality * self.state_scale[None, :]
        network_inequality_rhs = -(physical_scaled_inequality @ raw)
        scaled_inequality = np.vstack((-np.eye(variable_count), scaled_network_inequality))
        inequality_rhs = np.concatenate((raw / self.state_scale, network_inequality_rhs))

        singular_values = linalg.svdvals(scaled_equality, check_finite=True)
        if singular_values.size == 0:
            raise ProjectionError("the physical projection requires at least one equality row.")
        rank_tolerance = (
            max(scaled_equality.shape) * np.finfo(np.float64).eps * singular_values[0]
        )
        if singular_values[-1] <= rank_tolerance:
            raise ProjectionError("scaled equality matrix is not full row rank.")
        equality_condition = float(singular_values[0] / singular_values[-1])

        # A supplied warm point may be used for an expendable preliminary
        # solve, but the reportable candidate is always independently resolved
        # from a cold start, as required by the manuscript.
        if warm_start is not None:
            self._solve_once(
                scaled_equality=scaled_equality,
                equality_rhs=required_equality,
                scaled_inequality=scaled_inequality,
                inequality_rhs=inequality_rhs,
                warm_start=warm_start,
            )
        def audited_result(solver_result: object, attempt: int) -> ProjectionResult:
            displacement = np.asarray(
                solver_result.x
                if solver_result.x is not None
                else np.full(variable_count, np.nan),
                dtype=np.float64,
            )
            state = raw + self.state_scale * displacement
            slack = inequality_rhs - scaled_inequality @ displacement

            multipliers_reconstructed = False
            active_inequality_count = 0
            if np.all(np.isfinite(displacement)):
                try:
                    equality_dual, inequality_dual, active_inequality_count = (
                        self._reconstruct_multipliers(
                            displacement,
                            scaled_equality,
                            scaled_inequality,
                            inequality_rhs,
                        )
                    )
                    multipliers_reconstructed = True
                except ProjectionError:
                    equality_dual = np.full(scaled_equality.shape[0], np.nan)
                    inequality_dual = np.full(scaled_inequality.shape[0], np.nan)
            else:
                equality_dual = np.full(scaled_equality.shape[0], np.nan)
                inequality_dual = np.full(scaled_inequality.shape[0], np.nan)

            arrays_finite = all(
                np.all(np.isfinite(array))
                for array in (
                    displacement,
                    state,
                    slack,
                    equality_dual,
                    inequality_dual,
                )
            )
            if arrays_finite:
                equality_residual = float(
                    np.linalg.norm(
                        scaled_equality @ displacement - required_equality,
                        ord=np.inf,
                    )
                )
                inequality_residual = self._positive_part_norm(
                    physical_scaled_inequality @ state
                )
                nonnegativity_residual = self._positive_part_norm(
                    -state / self.state_scale
                )
                dual_residual = self._positive_part_norm(-inequality_dual)
                equality_stationarity = scaled_equality.T @ equality_dual
                inequality_stationarity = scaled_inequality.T @ inequality_dual
                stationarity = (
                    displacement + equality_stationarity + inequality_stationarity
                )
                stationarity_residual = float(
                    np.max(
                        np.abs(stationarity)
                        / (
                            1.0
                            + np.abs(displacement)
                            + np.abs(equality_stationarity)
                            + np.abs(inequality_stationarity)
                        )
                    )
                )
                complementarity_residual = float(
                    np.linalg.norm(inequality_dual * slack, ord=np.inf)
                )
            else:
                equality_residual = inequality_residual = nonnegativity_residual = np.inf
                dual_residual = stationarity_residual = complementarity_residual = np.inf

            # The independently reconstructed KKT audit is authoritative.
            # Solver status is retained for diagnosis but is not itself an
            # acceptance gate.
            accepted = (
                arrays_finite
                and multipliers_reconstructed
                and equality_residual <= self.equality_acceptance_tolerance
                and inequality_residual <= self.inequality_acceptance_tolerance
                and nonnegativity_residual <= self.nonnegativity_acceptance_tolerance
                and dual_residual <= self.inequality_acceptance_tolerance
                and stationarity_residual <= self.inequality_acceptance_tolerance
                and complementarity_residual <= self.inequality_acceptance_tolerance
            )
            diagnostics = ProjectionDiagnostics(
                status=str(solver_result.info.status),
                status_value=int(solver_result.info.status_val),
                iterations=int(solver_result.info.iter),
                equality_rank_tolerance=float(rank_tolerance),
                equality_smallest_singular_value=float(singular_values[-1]),
                equality_condition_number=equality_condition,
                equality_residual=equality_residual,
                inequality_residual=inequality_residual,
                nonnegativity_residual=nonnegativity_residual,
                dual_feasibility_residual=dual_residual,
                stationarity_residual=stationarity_residual,
                complementarity_residual=complementarity_residual,
                retried_cold=warm_start is not None or attempt > 1,
                active_inequality_count=active_inequality_count,
                multipliers_reconstructed=multipliers_reconstructed,
                solver_attempts=attempt,
                fallback_used=attempt > 1,
            )
            return ProjectionResult(
                state=state,
                displacement=displacement,
                equality_multipliers=equality_dual,
                inequality_multipliers=inequality_dual,
                inequality_slack=slack,
                diagnostics=diagnostics,
                accepted=accepted,
            )

        cold_attempts = (
            {},
            {"rho": 0.01, "adaptive_rho": False, "maximum_iterations": 200_000},
            {"rho": 10.0, "adaptive_rho": False, "maximum_iterations": 200_000},
        )
        attempted_results: list[ProjectionResult] = []
        for attempt, options in enumerate(cold_attempts, start=1):
            solver_result = self._solve_once(
                scaled_equality=scaled_equality,
                equality_rhs=required_equality,
                scaled_inequality=scaled_inequality,
                inequality_rhs=inequality_rhs,
                warm_start=None,
                **options,
            )
            candidate = audited_result(solver_result, attempt)
            attempted_results.append(candidate)
            if candidate.accepted:
                return candidate

        def failure_score(result: ProjectionResult) -> float:
            diagnostics = result.diagnostics
            scaled_residuals = (
                diagnostics.equality_residual / self.equality_acceptance_tolerance,
                diagnostics.inequality_residual / self.inequality_acceptance_tolerance,
                diagnostics.nonnegativity_residual
                / self.nonnegativity_acceptance_tolerance,
                diagnostics.dual_feasibility_residual
                / self.inequality_acceptance_tolerance,
                diagnostics.stationarity_residual
                / self.inequality_acceptance_tolerance,
                diagnostics.complementarity_residual
                / self.inequality_acceptance_tolerance,
            )
            return float(max(scaled_residuals))

        final_result = min(attempted_results, key=failure_score)
        final_result = replace(
            final_result,
            diagnostics=replace(
                final_result.diagnostics,
                retried_cold=True,
                solver_attempts=len(attempted_results),
                fallback_used=True,
            ),
        )

        if raise_on_failure:
            diagnostics = final_result.diagnostics
            raise ProjectionError(
                "physical QP failed independent acceptance: "
                f"status={diagnostics.status!r}, r_E={diagnostics.equality_residual:.3e}, "
                f"r_G={diagnostics.inequality_residual:.3e}, "
                f"r_+={diagnostics.nonnegativity_residual:.3e}, "
                f"r_stat={diagnostics.stationarity_residual:.3e}."
            )
        return final_result


def project_network_state(
    raw_state: npt.ArrayLike,
    operators: NetworkOperators,
    *,
    state_scale: npt.ArrayLike,
    row_scales: NetworkRowScales,
    warm_start: ProjectionWarmStart | None = None,
    raise_on_failure: bool = True,
) -> ProjectionResult:
    projector = PhysicalProjector(
        state_scale=state_scale,
        equality_scale=row_scales.equality,
        inequality_scale=row_scales.inequality,
    )
    return projector.project(
        raw_state,
        operators.equality_matrix,
        operators.equality_rhs,
        operators.inequality_matrix,
        warm_start=warm_start,
        raise_on_failure=raise_on_failure,
    )


def feasibility_first_merit(
    objective: float | None,
    maximum_scaled_violation: float | None,
    *,
    accepted: bool,
    feasibility_tolerance: float = 1.0e-8,
) -> float:
    """Map an evaluated candidate to the manuscript's finite search merit."""

    if not accepted or objective is None or maximum_scaled_violation is None:
        return 2.0
    objective_value = float(objective)
    violation = float(maximum_scaled_violation)
    if not np.isfinite(objective_value) or not np.isfinite(violation) or objective_value < 0.0:
        return 2.0
    violation = max(0.0, violation)
    if violation <= feasibility_tolerance:
        return objective_value / (1.0 + objective_value)
    return 1.0 + violation / (1.0 + violation)


@dataclass(frozen=True)
class SearchSettings:
    """Finite-budget center/corner, DIRECT, face, and pattern-search contract."""

    total_budget: int = 25_000
    full_direct_budget: int = 18_000
    face_direct_budget: int = 300
    direct_epsilon: float = 1.0e-4
    direct_resolution: float = 1.0 / 1024.0
    local_seed_count: int = 4
    initial_seed_separation: float = 1.0 / 8.0
    initial_mesh: float = 1.0 / 16.0
    terminal_mesh: float = 1.0 / 512.0
    failure_value: float = 2.0

    def validate(self, dimension: int) -> None:
        if dimension < 1:
            raise SurrogateValidationError("search dimension must be positive.")
        if self.total_budget < 1 + 2**dimension:
            raise SurrogateValidationError("search budget cannot cover the center and all corners.")
        if self.full_direct_budget < 1 or self.face_direct_budget < 0:
            raise SurrogateValidationError("DIRECT phase budgets are invalid.")
        if not 0.0 < self.direct_resolution < 1.0:
            raise SurrogateValidationError("direct_resolution must lie strictly between zero and one.")
        if self.direct_epsilon < 0.0:
            raise SurrogateValidationError("direct_epsilon must be nonnegative.")
        if self.local_seed_count < 1:
            raise SurrogateValidationError("at least one local seed is required.")
        if not 0.0 < self.initial_seed_separation <= 1.0:
            raise SurrogateValidationError("initial seed separation must lie in (0, 1].")
        if not 0.0 < self.terminal_mesh <= self.initial_mesh <= 1.0:
            raise SurrogateValidationError("pattern-search mesh sizes are inconsistent.")
        if not np.isfinite(self.failure_value):
            raise SurrogateValidationError("failure_value must be finite.")


@dataclass(frozen=True)
class SearchRecord:
    evaluation: int
    normalized_point: tuple[float, ...]
    physical_point: tuple[float, ...]
    value: float
    phase: str
    error: str | None = None


@dataclass(frozen=True)
class SearchResult:
    x: FloatArray
    normalized_x: FloatArray
    fun: float
    evaluations: int
    records: tuple[SearchRecord, ...]
    phase_counts: dict[str, int]
    message: str


class _EvaluationCache:
    def __init__(
        self,
        objective: Callable[[FloatArray], float],
        lower: FloatArray,
        upper: FloatArray,
        settings: SearchSettings,
    ) -> None:
        self.objective = objective
        self.lower = lower
        self.span = upper - lower
        self.settings = settings
        self.values: dict[bytes, float] = {}
        self.points: dict[bytes, FloatArray] = {}
        self.records: list[SearchRecord] = []
        self.phase_counts: dict[str, int] = {}

    @staticmethod
    def key(point: FloatArray) -> bytes:
        normalized = np.ascontiguousarray(point, dtype="<f8")
        return normalized.tobytes()

    @property
    def evaluations(self) -> int:
        return len(self.records)

    @property
    def remaining(self) -> int:
        return self.settings.total_budget - self.evaluations

    def uncached_count(self, points: Iterable[FloatArray]) -> int:
        keys: set[bytes] = set()
        for point in points:
            key = self.key(point)
            if key not in self.values:
                keys.add(key)
        return len(keys)

    def evaluate(self, point: FloatArray, *, phase: str) -> float:
        normalized = np.asarray(point, dtype=np.float64)
        if normalized.ndim != 1 or normalized.size != self.lower.size:
            raise SurrogateValidationError("search point has an inconsistent dimension.")
        if np.any(normalized < 0.0) or np.any(normalized > 1.0) or not np.all(np.isfinite(normalized)):
            raise SurrogateValidationError("normalized search point lies outside [0, 1].")
        key = self.key(normalized)
        if key in self.values:
            return self.values[key]
        if self.remaining <= 0:
            raise SearchBudgetError("the declared distinct-evaluation budget is exhausted.")
        physical = self.lower + self.span * normalized
        error: str | None = None
        try:
            value = float(self.objective(physical.copy()))
            if not np.isfinite(value):
                error = "objective returned a non-finite value"
                value = self.settings.failure_value
        except Exception as exc:  # scientific evaluators encode a failed point this way
            error = f"{type(exc).__name__}: {exc}"
            value = self.settings.failure_value
        self.values[key] = value
        self.points[key] = normalized.copy()
        evaluation = self.evaluations + 1
        self.records.append(
            SearchRecord(
                evaluation=evaluation,
                normalized_point=tuple(float(item) for item in normalized),
                physical_point=tuple(float(item) for item in physical),
                value=value,
                phase=phase,
                error=error,
            )
        )
        self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1
        return value


@dataclass(frozen=True)
class _Rectangle:
    identifier: int
    center: FloatArray
    lengths: FloatArray
    value: float

    @property
    def diagonal(self) -> float:
        return 0.5 * float(np.linalg.norm(self.lengths))

    @property
    def lower_corner(self) -> tuple[float, ...]:
        return tuple(float(item) for item in self.center - 0.5 * self.lengths)


class _DirectDomain:
    def __init__(
        self,
        cache: _EvaluationCache,
        expand: Callable[[FloatArray], FloatArray],
        dimension: int,
    ) -> None:
        self.cache = cache
        self.expand = expand
        self.dimension = dimension

    def evaluate(self, point: FloatArray, *, phase: str) -> float:
        return self.cache.evaluate(self.expand(point), phase=phase)

    def uncached_count(self, points: Iterable[FloatArray]) -> int:
        return self.cache.uncached_count(self.expand(point) for point in points)


class _DirectPartition:
    def __init__(self, center_value: float, dimension: int) -> None:
        center = np.full(dimension, 0.5, dtype=np.float64)
        self.rectangles: dict[int, _Rectangle] = {
            0: _Rectangle(0, center, np.ones(dimension, dtype=np.float64), center_value)
        }
        self.next_identifier = 1
        self.iteration = 0

    def add(self, center: FloatArray, lengths: FloatArray, value: float) -> None:
        identifier = self.next_identifier
        self.next_identifier += 1
        self.rectangles[identifier] = _Rectangle(
            identifier, center.copy(), lengths.copy(), float(value)
        )

    def potentially_optimal(self, epsilon: float) -> list[_Rectangle]:
        rectangles = tuple(self.rectangles.values())
        global_minimum = min(rectangle.value for rectangle in rectangles)
        by_diagonal: dict[float, list[_Rectangle]] = {}
        for rectangle in rectangles:
            by_diagonal.setdefault(rectangle.diagonal, []).append(rectangle)
        groups: list[tuple[float, float, list[_Rectangle]]] = []
        for diagonal, members in by_diagonal.items():
            best_value = min(member.value for member in members)
            ties = sorted(
                (member for member in members if member.value == best_value),
                key=lambda member: member.lower_corner,
            )
            groups.append((diagonal, best_value, ties))
        groups.sort(key=lambda group: group[0])

        selected: list[_Rectangle] = []
        target = global_minimum - epsilon * abs(global_minimum)
        for index, (diagonal, value, ties) in enumerate(groups):
            lower_slope = 0.0
            upper_slope = np.inf
            for other_diagonal, other_value, _ in groups[:index]:
                lower_slope = max(
                    lower_slope, (value - other_value) / (diagonal - other_diagonal)
                )
            for other_diagonal, other_value, _ in groups[index + 1 :]:
                upper_slope = min(
                    upper_slope, (other_value - value) / (other_diagonal - diagonal)
                )
            if diagonal > 0.0:
                lower_slope = max(lower_slope, (value - target) / diagonal)
            elif value > target:
                continue
            tolerance = 64.0 * np.finfo(np.float64).eps * max(
                1.0, abs(lower_slope), abs(upper_slope) if np.isfinite(upper_slope) else 1.0
            )
            if lower_slope <= upper_slope + tolerance:
                tie_index = self.iteration % len(ties)
                selected.append(ties[tie_index])
        selected.sort(
            key=lambda rectangle: (
                -rectangle.diagonal,
                rectangle.value,
                rectangle.lower_corner,
            )
        )
        return selected


def _advance_direct(
    partition: _DirectPartition,
    domain: _DirectDomain,
    *,
    phase: str,
    maximum_new_evaluations: int,
    epsilon: float,
    resolution: float,
) -> int:
    start_count = domain.cache.evaluations
    if maximum_new_evaluations <= 0:
        return 0
    while domain.cache.remaining > 0:
        if max(float(np.max(rectangle.lengths)) for rectangle in partition.rectangles.values()) <= resolution:
            break
        selected = partition.potentially_optimal(epsilon)
        if not selected:
            break
        partition.iteration += 1
        made_split = False
        phase_exhausted = False
        for rectangle in selected:
            current_new = domain.cache.evaluations - start_count
            phase_remaining = maximum_new_evaluations - current_new
            longest = float(np.max(rectangle.lengths))
            coordinates = np.flatnonzero(rectangle.lengths == longest)
            trial_points: list[FloatArray] = []
            trial_lookup: dict[tuple[int, int], FloatArray] = {}
            for coordinate in coordinates:
                delta = rectangle.lengths[coordinate] / 3.0
                for sign in (-1, 1):
                    point = rectangle.center.copy()
                    point[coordinate] += sign * delta
                    trial_points.append(point)
                    trial_lookup[(int(coordinate), sign)] = point
            required = domain.uncached_count(trial_points)
            if required > min(phase_remaining, domain.cache.remaining):
                phase_exhausted = True
                break

            trial_values: dict[tuple[int, int], float] = {}
            for coordinate in coordinates:
                for sign in (-1, 1):
                    point = trial_lookup[(int(coordinate), sign)]
                    trial_values[(int(coordinate), sign)] = domain.evaluate(point, phase=phase)
            split_order = sorted(
                (int(coordinate) for coordinate in coordinates),
                key=lambda coordinate: (
                    min(
                        trial_values[(coordinate, -1)],
                        trial_values[(coordinate, 1)],
                    ),
                    coordinate,
                ),
            )
            partition.rectangles.pop(rectangle.identifier)
            central_lengths = rectangle.lengths.copy()
            for coordinate in split_order:
                child_lengths = central_lengths.copy()
                child_lengths[coordinate] /= 3.0
                delta = rectangle.lengths[coordinate] / 3.0
                for sign in (-1, 1):
                    child_center = rectangle.center.copy()
                    child_center[coordinate] += sign * delta
                    partition.add(
                        child_center,
                        child_lengths,
                        trial_values[(coordinate, sign)],
                    )
                central_lengths[coordinate] /= 3.0
            partition.add(rectangle.center, central_lengths, rectangle.value)
            made_split = True
        if phase_exhausted or not made_split:
            break
    return domain.cache.evaluations - start_count


def _pattern_directions(dimension: int) -> tuple[FloatArray, ...]:
    directions: list[FloatArray] = []
    for coordinate in range(dimension):
        for sign in (-1.0, 1.0):
            direction = np.zeros(dimension, dtype=np.float64)
            direction[coordinate] = sign
            directions.append(direction)
    for first in range(dimension):
        for second in range(first + 1, dimension):
            for first_sign, second_sign in product((-1.0, 1.0), repeat=2):
                direction = np.zeros(dimension, dtype=np.float64)
                direction[first] = first_sign
                direction[second] = second_sign
                directions.append(direction)
    return tuple(directions)


def _select_local_seeds(cache: _EvaluationCache, settings: SearchSettings) -> list[FloatArray]:
    candidates = sorted(
        ((value, tuple(float(item) for item in cache.points[key]), cache.points[key])
         for key, value in cache.values.items()),
        key=lambda item: (item[0], item[1]),
    )
    desired = min(settings.local_seed_count, len(candidates))
    threshold = settings.initial_seed_separation
    selected: list[FloatArray] = []
    while True:
        selected = []
        for _, _, point in candidates:
            if not selected or all(
                float(np.linalg.norm(point - existing, ord=np.inf)) >= threshold
                for existing in selected
            ):
                selected.append(point.copy())
                if len(selected) == desired:
                    return selected
        if len(selected) == desired or threshold == 0.0:
            return selected
        threshold *= 0.5
        if threshold < np.finfo(np.float64).eps:
            threshold = 0.0


def _run_pattern_search(cache: _EvaluationCache, settings: SearchSettings) -> None:
    seeds = _select_local_seeds(cache, settings)
    if not seeds:
        return
    directions = _pattern_directions(cache.lower.size)
    centers = [seed.copy() for seed in seeds]
    values = [cache.evaluate(seed, phase="pattern:seed") for seed in centers]
    meshes = [settings.initial_mesh for _ in centers]
    active = [True for _ in centers]
    while cache.remaining > 0 and any(active):
        progressed = False
        for seed_index in range(len(centers)):
            if not active[seed_index]:
                continue
            center = centers[seed_index]
            mesh = meshes[seed_index]
            points_by_key: dict[bytes, FloatArray] = {}
            for direction in directions:
                point = np.clip(center + mesh * direction, 0.0, 1.0)
                key = cache.key(point)
                if key != cache.key(center):
                    points_by_key.setdefault(key, point)
            # Dictionaries preserve the declared direction order: axial
            # trials first, followed by lexicographic pairwise directions.
            poll_points = list(points_by_key.values())
            required = cache.uncached_count(poll_points)
            if required > cache.remaining:
                return
            poll_values = [cache.evaluate(point, phase=f"pattern:{seed_index}") for point in poll_points]
            best_index = min(
                range(len(poll_points)),
                key=lambda index: (
                    poll_values[index], tuple(float(item) for item in poll_points[index])
                ),
            )
            if poll_values[best_index] < values[seed_index]:
                centers[seed_index] = poll_points[best_index].copy()
                values[seed_index] = poll_values[best_index]
            else:
                if mesh <= settings.terminal_mesh:
                    active[seed_index] = False
                else:
                    meshes[seed_index] = mesh * 0.5
            progressed = True
            if cache.remaining <= 0:
                break
        if not progressed:
            break


def deterministic_bounded_search(
    objective: Callable[[FloatArray], float],
    bounds: Sequence[tuple[float, float]],
    *,
    settings: SearchSettings | None = None,
) -> SearchResult:
    """Run the manuscript's deterministic boundary-augmented bounded search.

    The objective receives physical coordinates.  Exact binary64 normalized
    coordinates are cached, so a duplicate does not consume the scientific
    evaluation budget.
    """

    settings = settings or SearchSettings()
    bound_array = _finite_array(bounds, name="bounds", ndim=2)
    if bound_array.shape[1] != 2:
        raise SurrogateValidationError("bounds must be a sequence of (lower, upper) pairs.")
    lower = bound_array[:, 0]
    upper = bound_array[:, 1]
    if np.any(upper <= lower):
        raise SurrogateValidationError("every upper bound must exceed its lower bound.")
    dimension = lower.size
    settings.validate(dimension)
    cache = _EvaluationCache(objective, lower, upper, settings)

    identity_domain = _DirectDomain(cache, lambda point: point.copy(), dimension)
    center = np.full(dimension, 0.5, dtype=np.float64)
    center_value = cache.evaluate(center, phase="full_direct:center")
    full_partition = _DirectPartition(center_value, dimension)

    for corner in product((0.0, 1.0), repeat=dimension):
        cache.evaluate(np.asarray(corner, dtype=np.float64), phase="corner")

    full_remaining = max(0, settings.full_direct_budget - 1)
    _advance_direct(
        full_partition,
        identity_domain,
        phase="full_direct",
        maximum_new_evaluations=min(full_remaining, cache.remaining),
        epsilon=settings.direct_epsilon,
        resolution=settings.direct_resolution,
    )

    if dimension > 1 and settings.face_direct_budget > 0:
        for fixed_coordinate in range(dimension):
            for side in (0.0, 1.0):
                if cache.remaining <= 0:
                    break

                def expand(
                    reduced: FloatArray,
                    *,
                    coordinate: int = fixed_coordinate,
                    fixed_side: float = side,
                ) -> FloatArray:
                    point = np.empty(dimension, dtype=np.float64)
                    point[coordinate] = fixed_side
                    point[np.arange(dimension) != coordinate] = reduced
                    return point

                face_phase = f"face:{fixed_coordinate}:{int(side)}"
                face_domain = _DirectDomain(cache, expand, dimension - 1)
                face_center = np.full(dimension - 1, 0.5, dtype=np.float64)
                face_start = cache.evaluations
                face_center_value = face_domain.evaluate(face_center, phase=f"{face_phase}:center")
                face_partition = _DirectPartition(face_center_value, dimension - 1)
                center_cost = cache.evaluations - face_start
                face_remaining = max(0, settings.face_direct_budget - center_cost)
                _advance_direct(
                    face_partition,
                    face_domain,
                    phase=face_phase,
                    maximum_new_evaluations=min(face_remaining, cache.remaining),
                    epsilon=settings.direct_epsilon,
                    resolution=settings.direct_resolution,
                )

    if cache.remaining > 0:
        _run_pattern_search(cache, settings)

    best = min(
        cache.records,
        key=lambda record: (record.value, record.normalized_point),
    )
    normalized_best = np.asarray(best.normalized_point, dtype=np.float64)
    physical_best = np.asarray(best.physical_point, dtype=np.float64)
    if cache.remaining == 0:
        message = "declared distinct-evaluation budget exhausted"
    else:
        message = "all DIRECT partitions and local meshes reached their stopping rules"
    return SearchResult(
        x=physical_best,
        normalized_x=normalized_best,
        fun=float(best.value),
        evaluations=cache.evaluations,
        records=tuple(cache.records),
        phase_counts=dict(cache.phase_counts),
        message=message,
    )


__all__ = [
    "LeastSquaresDiagnostics",
    "NetworkLayout",
    "NetworkOperators",
    "NetworkRowScales",
    "PhysicalProjector",
    "ProjectionDiagnostics",
    "ProjectionError",
    "ProjectionResult",
    "ProjectionWarmStart",
    "QuadraticFeatureMap",
    "QuadraticSurrogate",
    "SearchBudgetError",
    "SearchRecord",
    "SearchResult",
    "SearchSettings",
    "SurrogateValidationError",
    "affine_projection",
    "build_network_operators",
    "deterministic_bounded_search",
    "feasibility_first_merit",
    "fit_network_row_scales",
    "no_conversion_feasible_state",
    "project_network_state",
]
