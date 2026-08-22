"""Fixed statistical surrogate and physical-network utilities.

The module implements the numerical contracts stated in the closed-loop
mixer--reactor--clarifier manuscript.  It deliberately keeps the mechanistic
model out of the regression layer: callers provide accepted mechanistic
states.  The active statistical path supplies the development-only quadratic
response, split-conformal calibration, untouched raw assessment, and the
complete-response coordinate layout used by the combined NLP.  Retired
projection-QP and bounded-search implementations are intentionally absent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from scipy import linalg


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class SurrogateValidationError(ValueError):
    """Raised when a scientific precondition for fitting is not satisfied."""


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

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class QuadraticSurrogate:
    """A fixed standardized multiresponse ordinary-least-squares model."""

    feature_map: QuadraticFeatureMap
    response_center: FloatArray
    response_scale: FloatArray
    coefficients: FloatArray
    diagnostics: LeastSquaresDiagnostics
    feature_qr_upper: FloatArray
    feature_qr_pivots: IntArray

    def __post_init__(self) -> None:
        response_center = _finite_array(
            self.response_center, name="response_center", ndim=1
        )
        response_scale = _finite_array(
            self.response_scale, name="response_scale", ndim=1
        )
        coefficients = _finite_array(self.coefficients, name="coefficients", ndim=2)
        upper = _finite_array(
            self.feature_qr_upper, name="feature_qr_upper", ndim=2
        )
        pivots = np.asarray(self.feature_qr_pivots)
        feature_count = self.feature_map.feature_count
        response_count = response_center.size
        if response_scale.shape != (response_count,) or np.any(response_scale <= 0.0):
            raise SurrogateValidationError(
                "response_scale must be positive and match response_center."
            )
        if coefficients.shape != (response_count, feature_count):
            raise SurrogateValidationError(
                "coefficients must have shape (response_count, feature_count)."
            )
        if upper.shape != (feature_count, feature_count):
            raise SurrogateValidationError(
                "feature_qr_upper must be square with one row per feature."
            )
        if np.any(np.diag(upper) == 0.0):
            raise SurrogateValidationError("feature_qr_upper must be nonsingular.")
        if pivots.shape != (feature_count,) or not np.issubdtype(pivots.dtype, np.integer):
            raise SurrogateValidationError(
                "feature_qr_pivots must be an integer vector with one entry per feature."
            )
        pivots = pivots.astype(np.int64, copy=False)
        if not np.array_equal(np.sort(pivots), np.arange(feature_count, dtype=np.int64)):
            raise SurrogateValidationError(
                "feature_qr_pivots must be a permutation of the feature indices."
            )
        object.__setattr__(self, "response_center", response_center)
        object.__setattr__(self, "response_scale", response_scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "feature_qr_upper", upper)
        object.__setattr__(self, "feature_qr_pivots", pivots)

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
            feature_qr_upper=r_qr,
            feature_qr_pivots=np.asarray(pivot, dtype=np.int64),
        )

    @property
    def response_count(self) -> int:
        return int(self.response_center.size)

    def predict_standardized(self, decisions: npt.ArrayLike, influent: npt.ArrayLike) -> FloatArray:
        features = self.feature_map.transform(decisions, influent)
        if features.ndim == 1:
            return self.coefficients @ features
        return features @ self.coefficients.T

    def predict(self, decisions: npt.ArrayLike, influent: npt.ArrayLike) -> FloatArray:
        standardized = self.predict_standardized(decisions, influent)
        return self.response_center + standardized * self.response_scale

    def leverage(self, decisions: npt.ArrayLike, influent: npt.ArrayLike) -> float | FloatArray:
        """Evaluate ``phi' (Phi_D' Phi_D)^-1 phi`` by pivoted QR solves.

        If ``Phi_D[:, pivots] = Q R``, the leverage is
        ``||R^-T phi[pivots]||_2^2``.  Applying the stored permutation before
        the triangular solve is essential; no normal equations are formed.
        """

        features = self.feature_map.transform(decisions, influent)
        if features.ndim == 1:
            permuted = features[self.feature_qr_pivots]
            solved = linalg.solve_triangular(
                self.feature_qr_upper.T,
                permuted,
                lower=True,
                check_finite=True,
            )
            return float(np.dot(solved, solved))
        permuted = features[:, self.feature_qr_pivots]
        solved = linalg.solve_triangular(
            self.feature_qr_upper.T,
            permuted.T,
            lower=True,
            check_finite=True,
        )
        return np.sum(np.square(solved), axis=0)

    def maximum_training_leverage(
        self, decisions: npt.ArrayLike, influent: npt.ArrayLike
    ) -> float:
        """Return the largest leverage over the supplied development rows."""

        values = np.asarray(self.leverage(decisions, influent), dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise SurrogateValidationError(
                "development leverage must be a finite nonempty vector."
            )
        return float(np.max(values))


def standardized_squared_error_scores(
    observed: npt.ArrayLike,
    predicted: npt.ArrayLike,
    response_scale: npt.ArrayLike,
) -> FloatArray:
    """Return rowwise squared standardized RMS discrepancies.

    This is the manuscript score ``d_i=(1/L)||D_chi^-1(y_i-yhat_i)||^2``.
    The response scales must be the population standard deviations frozen on
    the development block.
    """

    truth = _finite_array(observed, name="observed", ndim=2)
    prediction = np.asarray(predicted, dtype=np.float64)
    scale = _finite_array(response_scale, name="response_scale", ndim=1)
    if prediction.ndim != 2 or prediction.shape != truth.shape:
        raise SurrogateValidationError(
            "predicted must be a matrix with the same shape as observed."
        )
    if scale.shape != (truth.shape[1],) or np.any(scale <= 0.0):
        raise SurrogateValidationError(
            "response_scale must be positive with one entry per response."
        )
    with np.errstate(over="ignore", invalid="ignore"):
        standardized = (truth - prediction) / scale
        return np.mean(np.square(standardized), axis=1)


def _float64_sha256(values: npt.ArrayLike) -> str:
    canonical = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return sha256(canonical.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class ConformalCalibration:
    """Frozen split-conformal order statistic and its input-order scores."""

    alpha: float
    sample_count: int
    order_statistic_one_based: int
    sorted_index_zero_based: int
    delta: float
    scores: FloatArray
    scores_sha256: str

    @property
    def n(self) -> int:
        return self.sample_count

    @property
    def k_one_based(self) -> int:
        return self.order_statistic_one_based

    @property
    def index_zero_based(self) -> int:
        return self.sorted_index_zero_based

    def as_dict(self, *, include_scores: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "alpha": self.alpha,
            "sample_count": self.sample_count,
            "order_statistic_one_based": self.order_statistic_one_based,
            "sorted_index_zero_based": self.sorted_index_zero_based,
            "delta": self.delta,
            "scores_sha256": self.scores_sha256,
        }
        if include_scores:
            payload["scores"] = self.scores.tolist()
        return payload


def calibrate_split_conformal(
    observed: npt.ArrayLike,
    predicted: npt.ArrayLike,
    response_scale: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    maximum_delta: float = 1.0,
) -> ConformalCalibration:
    """Freeze the finite-sample split-conformal fidelity threshold.

    The returned order statistic is ``k=min(n, ceil((n+1)*(1-alpha)))``.
    Calibration is a gate: nonfinite scores or a threshold outside
    ``(0, maximum_delta]`` raise rather than triggering model selection or a
    refit.
    """

    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise SurrogateValidationError("alpha must lie strictly between zero and one.")
    if not np.isfinite(maximum_delta) or maximum_delta <= 0.0:
        raise SurrogateValidationError("maximum_delta must be positive and finite.")
    scores = standardized_squared_error_scores(observed, predicted, response_scale)
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise SurrogateValidationError("all calibration scores must be finite.")
    count = int(scores.size)
    one_based = min(count, int(np.ceil((count + 1) * (1.0 - alpha))))
    zero_based = one_based - 1
    delta = float(np.sort(scores)[zero_based])
    if not 0.0 < delta <= maximum_delta:
        raise SurrogateValidationError(
            f"calibration delta must lie in (0, {maximum_delta:g}]; got {delta:.6g}."
        )
    frozen_scores = np.asarray(scores, dtype=np.float64).copy()
    return ConformalCalibration(
        alpha=float(alpha),
        sample_count=count,
        order_statistic_one_based=one_based,
        sorted_index_zero_based=zero_based,
        delta=delta,
        scores=frozen_scores,
        scores_sha256=_float64_sha256(frozen_scores),
    )


@dataclass(frozen=True)
class CoordinateAssessmentMetrics:
    """Raw response-coordinate metrics in target-coordinate order."""

    rmse: FloatArray
    mae: FloatArray
    bias: FloatArray
    nrmse: FloatArray
    r_squared: FloatArray
    r_squared_defined: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class BlockAssessmentMetric:
    """Standardized RMS error for one declared coordinate block."""

    name: str
    coordinate_count: int
    standardized_rmse: float


@dataclass(frozen=True)
class AssessmentMetrics:
    """Untouched-assessment metrics and the three predeclared gates."""

    sample_count: int
    response_count: int
    coordinate_metrics: CoordinateAssessmentMetrics
    block_metrics: tuple[BlockAssessmentMetric, ...]
    scores: FloatArray
    scores_sha256: str
    complete_state_standardized_rmse: float
    empirical_coverage: float
    predictions_finite: bool
    complete_state_rmse_passed: bool
    coverage_passed: bool
    passed: bool

    def block_metric(self, name: str) -> BlockAssessmentMetric:
        for metric in self.block_metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)

    def as_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "response_count": self.response_count,
            "complete_state_standardized_rmse": self.complete_state_standardized_rmse,
            "empirical_coverage": self.empirical_coverage,
            "predictions_finite": self.predictions_finite,
            "complete_state_rmse_passed": self.complete_state_rmse_passed,
            "coverage_passed": self.coverage_passed,
            "passed": self.passed,
            "scores_sha256": self.scores_sha256,
            "block_metrics": [asdict(metric) for metric in self.block_metrics],
        }


DEFAULT_STATE_BLOCKS: tuple[tuple[str, slice], ...] = (
    ("mixer", slice(0, 20)),
    ("reactor_1", slice(20, 40)),
    ("reactor_2", slice(40, 60)),
    ("reactor_3", slice(60, 80)),
    ("reactor_4", slice(80, 100)),
    ("reactor_5", slice(100, 120)),
    ("overflow_flow", slice(120, 140)),
    ("underflow_flow", slice(140, 160)),
    ("clarifier_layers", slice(160, 170)),
    ("complete_state", slice(0, 170)),
)


def _assessment_from_predictions(
    observed: npt.ArrayLike,
    predicted: npt.ArrayLike,
    response_scale: npt.ArrayLike,
    *,
    delta: float,
    blocks: Mapping[str, slice | Sequence[int]] | None,
    complete_state_rmse_max: float,
    minimum_coverage: float,
    variance_relative_tolerance: float,
) -> AssessmentMetrics:
    truth = _finite_array(observed, name="observed", ndim=2)
    prediction = np.asarray(predicted, dtype=np.float64)
    scale = _finite_array(response_scale, name="response_scale", ndim=1)
    if prediction.ndim != 2 or prediction.shape != truth.shape:
        raise SurrogateValidationError(
            "predicted must be a matrix with the same shape as observed."
        )
    if truth.shape[0] == 0 or truth.shape[1] == 0:
        raise SurrogateValidationError("assessment must contain at least one row and response.")
    if scale.shape != (truth.shape[1],) or np.any(scale <= 0.0):
        raise SurrogateValidationError(
            "response_scale must be positive with one entry per response."
        )
    scalar_settings = np.asarray(
        [delta, complete_state_rmse_max, minimum_coverage, variance_relative_tolerance],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(scalar_settings)):
        raise SurrogateValidationError("assessment thresholds must be finite.")
    if not 0.0 < delta <= 1.0 or complete_state_rmse_max <= 0.0:
        raise SurrogateValidationError(
            "delta must lie in (0, 1] and complete_state_rmse_max must be positive."
        )
    if not 0.0 <= minimum_coverage <= 1.0 or variance_relative_tolerance < 0.0:
        raise SurrogateValidationError("assessment coverage/tolerance settings are invalid.")

    predictions_finite = bool(np.all(np.isfinite(prediction)))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        error = prediction - truth
        squared_error = np.square(error)
        rmse = np.sqrt(np.mean(squared_error, axis=0))
        mae = np.mean(np.abs(error), axis=0)
        bias = np.mean(error, axis=0)
        nrmse = rmse / scale
        centered_truth = truth - np.mean(truth, axis=0)
        denominator = np.sum(np.square(centered_truth), axis=0)
        assessment_scale = np.sqrt(np.mean(np.square(centered_truth), axis=0))
        reference = np.maximum(1.0, np.max(np.abs(truth), axis=0))
        r_squared_defined = assessment_scale > variance_relative_tolerance * reference
        r_squared = np.full(truth.shape[1], np.nan, dtype=np.float64)
        r_squared[r_squared_defined] = (
            1.0
            - np.sum(squared_error[:, r_squared_defined], axis=0)
            / denominator[r_squared_defined]
        )
        standardized_error = error / scale
        scores = np.mean(np.square(standardized_error), axis=1)
        complete_rmse = float(np.sqrt(np.mean(np.square(standardized_error))))

    if blocks is None:
        selected_blocks: Mapping[str, slice | Sequence[int]] = (
            dict(DEFAULT_STATE_BLOCKS)
            if truth.shape[1] == 170
            else {"complete_state": slice(0, truth.shape[1])}
        )
    else:
        selected_blocks = blocks
    block_metrics: list[BlockAssessmentMetric] = []
    for name, selection in selected_blocks.items():
        indices = np.arange(truth.shape[1], dtype=np.int64)[selection]
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0 or np.any(indices < 0) or np.any(indices >= truth.shape[1]):
            raise SurrogateValidationError(f"assessment block {name!r} is empty or out of range.")
        value = float(np.sqrt(np.mean(np.square(standardized_error[:, indices]))))
        block_metrics.append(
            BlockAssessmentMetric(str(name), int(indices.size), value)
        )

    empirical_coverage = float(np.mean(np.isfinite(scores) & (scores <= delta)))
    complete_passed = bool(np.isfinite(complete_rmse) and complete_rmse < complete_state_rmse_max)
    coverage_passed = bool(empirical_coverage >= minimum_coverage)
    passed = bool(predictions_finite and complete_passed and coverage_passed)
    coordinate = CoordinateAssessmentMetrics(
        rmse=np.asarray(rmse, dtype=np.float64),
        mae=np.asarray(mae, dtype=np.float64),
        bias=np.asarray(bias, dtype=np.float64),
        nrmse=np.asarray(nrmse, dtype=np.float64),
        r_squared=np.asarray(r_squared, dtype=np.float64),
        r_squared_defined=np.asarray(r_squared_defined, dtype=np.bool_),
    )
    score_array = np.asarray(scores, dtype=np.float64)
    return AssessmentMetrics(
        sample_count=int(truth.shape[0]),
        response_count=int(truth.shape[1]),
        coordinate_metrics=coordinate,
        block_metrics=tuple(block_metrics),
        scores=score_array,
        scores_sha256=_float64_sha256(score_array),
        complete_state_standardized_rmse=complete_rmse,
        empirical_coverage=empirical_coverage,
        predictions_finite=predictions_finite,
        complete_state_rmse_passed=complete_passed,
        coverage_passed=coverage_passed,
        passed=passed,
    )


def assess_raw_predictions(
    observed: npt.ArrayLike,
    predicted: npt.ArrayLike,
    response_scale: npt.ArrayLike,
    *,
    delta: float,
    blocks: Mapping[str, slice | Sequence[int]] | None = None,
    complete_state_rmse_max: float = 1.0,
    minimum_coverage: float = 0.90,
    variance_relative_tolerance: float = 1.0e-12,
) -> AssessmentMetrics:
    """Assess an already-computed raw response without fitting or correction."""

    return _assessment_from_predictions(
        observed,
        predicted,
        response_scale,
        delta=delta,
        blocks=blocks,
        complete_state_rmse_max=complete_state_rmse_max,
        minimum_coverage=minimum_coverage,
        variance_relative_tolerance=variance_relative_tolerance,
    )


def assess_raw_surrogate(
    model: QuadraticSurrogate,
    decisions: npt.ArrayLike,
    influent: npt.ArrayLike,
    observed: npt.ArrayLike,
    *,
    delta: float,
    blocks: Mapping[str, slice | Sequence[int]] | None = None,
    complete_state_rmse_max: float = 1.0,
    minimum_coverage: float = 0.90,
    variance_relative_tolerance: float = 1.0e-12,
) -> AssessmentMetrics:
    """Run the untouched raw-response assessment for a frozen surrogate."""

    prediction = model.predict(decisions, influent)
    return _assessment_from_predictions(
        observed,
        prediction,
        model.response_scale,
        delta=delta,
        blocks=blocks,
        complete_state_rmse_max=complete_state_rmse_max,
        minimum_coverage=minimum_coverage,
        variance_relative_tolerance=variance_relative_tolerance,
    )


@dataclass(frozen=True)
class NetworkLayout:
    """Coordinate layout for chi=(m,c_1,...,c_N,g_E,g_U,s_1,...,s_L)."""

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
        return (self.stage_count + 3) * self.component_count + self.layer_count

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
    def layer_slice(self) -> slice:
        start = (self.stage_count + 3) * self.component_count
        return slice(start, start + self.layer_count)


__all__ = [
    "AssessmentMetrics",
    "BlockAssessmentMetric",
    "ConformalCalibration",
    "CoordinateAssessmentMetrics",
    "DEFAULT_STATE_BLOCKS",
    "LeastSquaresDiagnostics",
    "NetworkLayout",
    "QuadraticFeatureMap",
    "QuadraticSurrogate",
    "SurrogateValidationError",
    "assess_raw_predictions",
    "assess_raw_surrogate",
    "calibrate_split_conformal",
    "standardized_squared_error_scores",
]
