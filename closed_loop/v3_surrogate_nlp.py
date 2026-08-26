"""Manuscript-v3 projected-surrogate operational optimization.

The primary article route is a single deterministic local optimization over
the seven normalized operating controls. Every distinct trial is evaluated by
a cold solve of the exact physical projection QP. Analytical active-set
sensitivities are used when their numerical audit passes; otherwise a
deterministic value-only COBYQA solve continues on the same exact-QP values and
constraints. The selected endpoint is independently cold-replayed. It is
classified as stationarity-unresolved unless an independent endpoint
active-set and upper-KKT audit passes.

The former embedded-KKT IPOPT continuation implementation remains available
for checkpoint compatibility and methodological comparison, but it is not
used by :func:`solve_surrogate_exact_qp_local`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence
import json

import casadi as ca
import numpy as np
import numpy.typing as npt
from scipy import linalg
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from .manuscript_v3 import DECISION_LOWER, DECISION_UPPER
from .model import COMPOSITE_MATRIX, INVARIANT_MATRIX, TSS_VECTOR
from .projection import (
    LogOverflowTSSClosure,
    NetworkLayout,
    NetworkRowScales,
    PhysicalProjector,
    ProjectionDiagnostics,
    ProjectionResult,
    QuadraticSurrogate,
    SurrogateValidationError,
    build_network_operators,
    fit_network_row_scales,
)


FloatArray = npt.NDArray[np.float64]
TrustRowCallback = Callable[[Any, Any, Any, Any], Any]

GAP_CONTINUATION: tuple[float, ...] = (
    1.0e-2,
    1.0e-4,
    1.0e-6,
    3.0e-7,
    1.0e-7,
    3.0e-8,
    1.0e-8,
)
START_SEED = 271_828
DEFAULT_OBJECTIVE_WEIGHTS = np.asarray([0.50, 0.15, 0.20, 0.05, 0.05, 0.05])
LEGACY_SURROGATE_PROTOCOL = "embedded_kkt_gap_continuation_multistart_v1"
EXACT_QP_SINGLE_START_PROTOCOL = "seven_variable_exact_qp_single_start_v1"
EXACT_QP_CENTER_START: tuple[float, ...] = (0.5,) * 7
LOCAL_CONVERGENCE_PROTOCOL = "exact_qp_two_scale_accelerated_feasible_poll_v3"


class SurrogateNLPError(RuntimeError):
    """Raised when the v3 surrogate route cannot satisfy its contract."""


def _vector(value: npt.ArrayLike, size: int, name: str, *, positive: bool = False) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {size}.")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")
    return array.copy()


def _matrix(value: npt.ArrayLike, shape: tuple[int, int], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite matrix with shape {shape}.")
    return array.copy()


def _flat(value: Any) -> FloatArray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _float_or_nan(value: Any) -> float:
    """Restore strict-JSON ``null`` failure placeholders as IEEE NaN."""

    return float("nan") if value is None else float(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _maximum_positive(values: npt.ArrayLike) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.max(np.maximum(array, 0.0), initial=0.0))


def _scaled_upper_residual(residual: Any, positive_scale: float) -> Any:
    """Return a dimensionless upper-inequality row without changing its sign."""

    scale = float(positive_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("an upper-constraint row scale must be finite and positive.")
    return residual / scale


def _normalized_limit_residual(value: Any, nonnegative_limit: float) -> Any:
    """Normalize ``value <= limit`` by its positive limit when available.

    A zero limit denotes an exact-zero diagnostic. Unit scaling is then the
    only finite fixed scale that preserves the feasible set without inserting
    an arbitrary floor into the scientific limit.
    """

    limit = float(nonnegative_limit)
    if not np.isfinite(limit) or limit < 0.0:
        raise ValueError("an upper-constraint limit must be finite and nonnegative.")
    scale = limit if limit > 0.0 else 1.0
    return _scaled_upper_residual(value - limit, scale)


def _safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in value)
    return cleaned or "v3_surrogate"


@dataclass(frozen=True)
class EngineeringLimits:
    """Physical constants and case-study engineering limits.

    SRT rows are evaluated in cross-multiplied form.  ``inventory_scale`` is
    only a positive numerical row scale and cannot change feasibility.
    """

    fresh_flow_m3_d: float = 10_000.0
    clarifier_area_m2: float = 1_500.0
    clarifier_volume_m3: float = 6_000.0
    srt_lower_d: float = 8.0
    srt_upper_d: float = 30.0
    external_loss_min_g_m3: float = 1.0
    slr_upper_kg_m2_d: float = 100.0
    underflow_tss_upper_g_m3: float = 15_000.0
    feed_tss_min_g_m3: float = 1.0
    sor_upper_m_d: float | None = None
    inventory_scale: float | None = None

    def __post_init__(self) -> None:
        positive = (
            self.fresh_flow_m3_d,
            self.clarifier_area_m2,
            self.clarifier_volume_m3,
            self.srt_lower_d,
            self.srt_upper_d,
            self.external_loss_min_g_m3,
            self.slr_upper_kg_m2_d,
            self.underflow_tss_upper_g_m3,
            self.feed_tss_min_g_m3,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("engineering constants and finite limits must be positive.")
        if self.srt_lower_d >= self.srt_upper_d:
            raise ValueError("the SRT lower limit must be below its upper limit.")
        if self.sor_upper_m_d is not None and (
            not np.isfinite(self.sor_upper_m_d) or self.sor_upper_m_d <= 0.0
        ):
            raise ValueError("sor_upper_m_d must be positive when supplied.")
        if self.inventory_scale is not None and (
            not np.isfinite(self.inventory_scale) or self.inventory_scale <= 0.0
        ):
            raise ValueError("inventory_scale must be positive when supplied.")

    @property
    def resolved_inventory_scale(self) -> float:
        if self.inventory_scale is not None:
            return float(self.inventory_scale)
        return float(
            self.srt_upper_d
            * self.fresh_flow_m3_d
            * self.underflow_tss_upper_g_m3
        )


@dataclass(frozen=True)
class TrustThresholds:
    """Frozen development-supported limits used by the upper problem."""

    correction_rms: float
    regularized_leverage: float
    split_rms: float | None = None
    reactor_rms: float | None = None

    def __post_init__(self) -> None:
        required = (self.correction_rms, self.regularized_leverage)
        if not all(np.isfinite(value) and value >= 0.0 for value in required):
            raise ValueError("native trust thresholds must be finite and nonnegative.")
        for value in (self.split_rms, self.reactor_rms):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError("optional trust thresholds must be finite and nonnegative.")


@dataclass(frozen=True)
class NamedTrustRows:
    """One additional scaled-residual family.

    The callback receives ``(theta, raw, projected, influent)`` and returns a
    CasADi-compatible vector of already scaled residual rows.  Its constraint
    is ``mean(rows**2) <= rms_threshold**2``.
    """

    name: str
    callback: TrustRowCallback
    rms_threshold: float

    def __post_init__(self) -> None:
        if not self.name or not callable(self.callback):
            raise ValueError("a named trust row requires a name and callable.")
        if not np.isfinite(self.rms_threshold) or self.rms_threshold < 0.0:
            raise ValueError("a trust-row RMS threshold must be finite and nonnegative.")


@dataclass(frozen=True)
class TrustDiagnosticCallbacks:
    """Optional v3 particulate-split and smooth-reactor residual rows."""

    split_rows: TrustRowCallback | None = None
    reactor_rows: TrustRowCallback | None = None
    additional: tuple[NamedTrustRows, ...] = ()


@dataclass(frozen=True)
class SurrogateCase:
    influent: FloatArray
    case_id: str = "nominal"
    objective_weights: FloatArray = field(
        default_factory=lambda: DEFAULT_OBJECTIVE_WEIGHTS.copy()
    )
    quality_weights: FloatArray | None = None

    def parameter_vector(self, assets: "SurrogateNLPAssets") -> FloatArray:
        influent = _vector(
            self.influent, assets.layout.component_count, "case influent"
        )
        objective = _vector(self.objective_weights, 6, "objective_weights")
        if np.any(objective < 0.0) or not np.isclose(np.sum(objective), 1.0, atol=1.0e-12):
            raise ValueError("objective_weights must be nonnegative and sum to one.")
        if self.quality_weights is None:
            quality = np.full(assets.quality_count, 1.0 / assets.quality_count)
        else:
            quality = _vector(
                self.quality_weights, assets.quality_count, "quality_weights"
            )
            if np.any(quality < 0.0) or not np.isclose(
                np.sum(quality), 1.0, atol=1.0e-12
            ):
                raise ValueError("quality_weights must be nonnegative and sum to one.")
        return np.concatenate((influent, objective, quality))


@dataclass(frozen=True)
class SurrogateNLPAssets:
    """Frozen numerical data shared by all influent cases and gap stages."""

    model: QuadraticSurrogate
    layout: NetworkLayout
    invariant_operator: FloatArray
    tss_weights: FloatArray
    row_scales: NetworkRowScales
    leverage_precision: FloatArray
    trust_thresholds: TrustThresholds
    quality_operator: FloatArray
    quality_scale: FloatArray
    overflow_closure: LogOverflowTSSClosure | None = None
    trust_callbacks: TrustDiagnosticCallbacks = field(
        default_factory=TrustDiagnosticCallbacks
    )
    engineering: EngineeringLimits = field(default_factory=EngineeringLimits)
    theta_lower: FloatArray = field(default_factory=lambda: DECISION_LOWER.copy())
    theta_upper: FloatArray = field(default_factory=lambda: DECISION_UPPER.copy())

    def __post_init__(self) -> None:
        if self.layout.layer_count < 3:
            raise ValueError("the v3 surrogate route requires at least three Clarifier layers.")
        if self.model.feature_map.decision_count != 7:
            raise ValueError("the manuscript-v3 surrogate route requires exactly seven controls.")
        if self.model.feature_map.influent_count != self.layout.component_count:
            raise ValueError("surrogate influent coordinates must equal the component count.")
        if self.model.response_center.size != self.layout.state_size:
            raise ValueError("surrogate response coordinates do not match the network layout.")
        component_count = self.layout.component_count
        invariant = np.asarray(self.invariant_operator, dtype=np.float64)
        if (
            invariant.ndim != 2
            or invariant.shape[1] != component_count
            or invariant.shape[0] == 0
            or not np.all(np.isfinite(invariant))
            or np.linalg.matrix_rank(invariant) != invariant.shape[0]
        ):
            raise ValueError("invariant_operator must be finite and full row rank.")
        equality_count = (
            2 * component_count
            + self.layout.stage_count * invariant.shape[0]
            + len(self.layout.soluble_indices)
            + int(self.overflow_closure is not None)
        )
        equality_scale = _vector(
            self.row_scales.equality, equality_count, "equality row scales", positive=True
        )
        inequality_scale = _vector(
            self.row_scales.inequality,
            self.layout.inequality_count,
            "inequality row scales",
            positive=True,
        )
        tss = _vector(self.tss_weights, component_count, "tss_weights")
        if np.any(tss < 0.0) or not np.any(tss > 0.0):
            raise ValueError("tss_weights must be nonnegative and nonzero.")
        if self.overflow_closure is not None:
            closure_map = self.overflow_closure.model.feature_map
            if (
                closure_map.decision_count != self.model.feature_map.decision_count
                or closure_map.influent_count != self.model.feature_map.influent_count
            ):
                raise ValueError("overflow closure input dimensions do not match the surrogate")
        feature_count = self.model.feature_map.feature_count
        leverage = _matrix(
            self.leverage_precision,
            (feature_count, feature_count),
            "leverage_precision",
        )
        if not np.allclose(leverage, leverage.T, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("leverage_precision must be symmetric.")
        minimum_eigenvalue = float(linalg.eigvalsh(leverage, check_finite=True)[0])
        if minimum_eigenvalue < -1.0e-10 * max(1.0, float(np.linalg.norm(leverage, 2))):
            raise ValueError("leverage_precision must be positive semidefinite.")
        quality = np.asarray(self.quality_operator, dtype=np.float64)
        if (
            quality.ndim != 2
            or quality.shape[0] == 0
            or quality.shape[1] != component_count
            or not np.all(np.isfinite(quality))
        ):
            raise ValueError("quality_operator must have one column per component.")
        quality_scale = _vector(
            self.quality_scale, quality.shape[0], "quality_scale", positive=True
        )
        lower = _vector(self.theta_lower, 7, "theta_lower")
        upper = _vector(self.theta_upper, 7, "theta_upper")
        if np.any(upper <= lower):
            raise ValueError("every control must have a positive operating span.")
        # The network uses q_U=r_R+w and q_E=1-w at every trial point.
        if lower[5] + lower[6] <= 0.0 or upper[6] >= 1.0:
            raise ValueError("control bounds must keep q_U positive and q_E positive.")
        pairs = (
            (self.trust_callbacks.split_rows, self.trust_thresholds.split_rms, "split"),
            (self.trust_callbacks.reactor_rows, self.trust_thresholds.reactor_rms, "reactor"),
        )
        for callback, threshold, name in pairs:
            if (callback is None) != (threshold is None):
                raise ValueError(
                    f"{name} trust rows and their threshold must be supplied together."
                )
        names = [item.name for item in self.trust_callbacks.additional]
        if len(set(names)) != len(names) or any(
            name in {"correction", "leverage", "split", "reactor"}
            for name in names
        ):
            raise ValueError("additional trust diagnostic names must be unique and reserved-name free.")
        object.__setattr__(self, "invariant_operator", invariant.copy())
        object.__setattr__(self, "tss_weights", tss)
        object.__setattr__(
            self, "row_scales", NetworkRowScales(equality_scale, inequality_scale)
        )
        object.__setattr__(self, "leverage_precision", 0.5 * (leverage + leverage.T))
        object.__setattr__(self, "quality_operator", quality.copy())
        object.__setattr__(self, "quality_scale", quality_scale)
        object.__setattr__(self, "theta_lower", lower)
        object.__setattr__(self, "theta_upper", upper)

    @property
    def quality_count(self) -> int:
        return int(self.quality_operator.shape[0])

    @property
    def equality_count(self) -> int:
        return int(self.row_scales.equality.size)

    @property
    def network_inequality_count(self) -> int:
        return int(self.row_scales.inequality.size)

    @property
    def projection_inequality_count(self) -> int:
        return self.layout.state_size + self.network_inequality_count

    @property
    def theta_span(self) -> FloatArray:
        return self.theta_upper - self.theta_lower


def regularized_leverage_contract(
    model: QuadraticSurrogate,
    development_decisions: npt.ArrayLike,
    development_influents: npt.ArrayLike,
) -> tuple[FloatArray, float]:
    """Return ``M_R^{-1}`` and the maximum development-row leverage."""

    decisions = np.asarray(development_decisions, dtype=np.float64)
    influents = np.asarray(development_influents, dtype=np.float64)
    if decisions.ndim != 2 or influents.ndim != 2 or decisions.shape[0] != influents.shape[0]:
        raise ValueError("development input blocks must be two-dimensional with equal rows.")
    design = np.asarray(model.feature_map.transform(decisions, influents), dtype=np.float64)
    rows, feature_count = design.shape
    penalty = np.ones(feature_count, dtype=np.float64)
    penalty[0] = 0.0
    matrix = design.T @ design + rows * float(model.ridge_penalty) * np.diag(penalty)
    try:
        factor = linalg.cho_factor(matrix, lower=True, check_finite=True)
        precision = linalg.cho_solve(factor, np.eye(feature_count), check_finite=True)
    except linalg.LinAlgError as exc:
        raise SurrogateNLPError(
            "the regularized feature matrix is not positive definite; a positive ridge "
            "penalty or full-column-rank design is required."
        ) from exc
    precision = 0.5 * (precision + precision.T)
    leverage = np.einsum("ij,jk,ik->i", design, precision, design)
    if not np.all(np.isfinite(leverage)) or np.min(leverage) < -1.0e-10:
        raise SurrogateNLPError("regularized leverage calculation failed its finite/PSD check.")
    return precision, float(np.max(leverage))


def build_surrogate_assets(
    model: QuadraticSurrogate,
    development_decisions: npt.ArrayLike,
    development_influents: npt.ArrayLike,
    development_targets: npt.ArrayLike,
    *,
    layout: NetworkLayout | None = None,
    invariant_operator: npt.ArrayLike = INVARIANT_MATRIX,
    tss_weights: npt.ArrayLike = TSS_VECTOR,
    quality_operator: npt.ArrayLike = COMPOSITE_MATRIX,
    correction_rms_threshold: float = 0.50,
    trust_callbacks: TrustDiagnosticCallbacks | None = None,
    split_rms_threshold: float | None = None,
    reactor_rms_threshold: float | None = None,
    engineering: EngineeringLimits | None = None,
    overflow_closure: LogOverflowTSSClosure | None = None,
    development_overflow_tss_closure: npt.ArrayLike | None = None,
) -> SurrogateNLPAssets:
    """Fit the projection scales, leverage contract, and quality normalization.

    Optional split/reactor callbacks must return *already scaled* residual
    rows.  Their thresholds are the frozen RMS limits computed from the
    development/out-of-fold calculations described by the supplement.
    """

    layout = layout or NetworkLayout()
    decisions = np.asarray(development_decisions, dtype=np.float64)
    influents = np.asarray(development_influents, dtype=np.float64)
    targets = np.asarray(development_targets, dtype=np.float64)
    if decisions.ndim != 2 or decisions.shape[1] != 7:
        raise ValueError("development_decisions must have seven columns.")
    if influents.shape != (decisions.shape[0], layout.component_count):
        raise ValueError("development_influents have inconsistent dimensions.")
    if targets.shape != (decisions.shape[0], layout.state_size):
        raise ValueError("development_targets have inconsistent dimensions.")
    engineering_limits = engineering or EngineeringLimits()
    if overflow_closure is None:
        closure_prediction = None
        if development_overflow_tss_closure is not None:
            raise ValueError(
                "development_overflow_tss_closure requires an overflow closure model"
            )
    else:
        closure_prediction = np.asarray(
            development_overflow_tss_closure
            if development_overflow_tss_closure is not None
            else overflow_closure.predict(decisions, influents),
            dtype=np.float64,
        )
        if (
            closure_prediction.shape != (len(decisions),)
            or not np.all(np.isfinite(closure_prediction))
            or np.any(closure_prediction <= 0.0)
        ):
            raise ValueError("development overflow-TSS closure predictions are invalid")
    row_scales = fit_network_row_scales(
        targets,
        influents,
        internal_recycle=decisions[:, 4],
        return_recycle=decisions[:, 5],
        waste_fraction=decisions[:, 6],
        invariant_operator=invariant_operator,
        tss_weights=tss_weights,
        layout=layout,
        clarifier_volume_m3=engineering_limits.clarifier_volume_m3,
        minimum_scale=1.0,
        overflow_tss_closure=closure_prediction,
    )
    precision, leverage_limit = regularized_leverage_contract(model, decisions, influents)
    quality_matrix = np.asarray(quality_operator, dtype=np.float64)
    q_effluent = 1.0 - decisions[:, 6]
    effluent = targets[:, layout.overflow_flow_slice] / q_effluent[:, None]
    quality_values = effluent @ quality_matrix.T
    quality_scale = np.maximum(1.0, np.std(quality_values, axis=0, ddof=0))
    thresholds = TrustThresholds(
        correction_rms=float(correction_rms_threshold),
        regularized_leverage=leverage_limit,
        split_rms=split_rms_threshold,
        reactor_rms=reactor_rms_threshold,
    )
    return SurrogateNLPAssets(
        model=model,
        layout=layout,
        invariant_operator=np.asarray(invariant_operator, dtype=np.float64),
        tss_weights=np.asarray(tss_weights, dtype=np.float64),
        row_scales=row_scales,
        leverage_precision=precision,
        trust_thresholds=thresholds,
        quality_operator=quality_matrix,
        quality_scale=quality_scale,
        overflow_closure=overflow_closure,
        trust_callbacks=trust_callbacks or TrustDiagnosticCallbacks(),
        engineering=engineering_limits,
    )


def ordered_normalized_starts(seed: int = START_SEED) -> FloatArray:
    """Return the manuscript's center plus eight deterministic 7-D LHS rows."""

    # Import locally to avoid making the study-design implementation a hidden
    # dependency of symbolic graph construction.
    from .design import SplitMix64

    rows, dimensions = 8, 7
    stream = SplitMix64(seed)
    design = np.empty((rows, dimensions), dtype=np.float64)
    denominator = float(1 << 53)
    for dimension in range(dimensions):
        permutation = list(range(rows))
        for index in range(rows - 1, 0, -1):
            swap = stream.randbelow(index + 1)
            permutation[index], permutation[swap] = permutation[swap], permutation[index]
        for row in range(rows):
            jitter = ((stream.next_uint64() >> 11) + 0.5) / denominator
            design[row, dimension] = (permutation[row] + jitter) / rows
    return np.vstack((np.full((1, dimensions), 0.5), design))


def symbolic_quadratic_prediction(
    model: QuadraticSurrogate,
    theta: ca.MX,
    influent: ca.MX,
) -> tuple[ca.MX, ca.MX]:
    """Construct the exact fitted feature serialization and raw response."""

    feature_map = model.feature_map
    if theta.numel() != 7 or influent.numel() != feature_map.influent_count:
        raise ValueError("symbolic predictor inputs have inconsistent dimensions.")
    decision_z = (theta - ca.DM(feature_map.decision_center)) / ca.DM(
        feature_map.decision_scale
    )
    influent_z = (influent - ca.DM(feature_map.influent_center)) / ca.DM(
        feature_map.influent_scale
    )
    terms: list[Any] = [decision_z, influent_z]
    terms.extend(
        decision_z[j] * decision_z[k]
        for j in range(feature_map.decision_count)
        for k in range(j, feature_map.decision_count)
    )
    terms.extend(
        influent_z[j] * influent_z[k]
        for j in range(feature_map.influent_count)
        for k in range(j, feature_map.influent_count)
    )
    terms.extend(
        decision_z[j] * influent_z[k]
        for j in range(feature_map.decision_count)
        for k in range(feature_map.influent_count)
    )
    unscaled = ca.vertcat(*terms)
    standardized = (unscaled - ca.DM(feature_map.term_center)) / ca.DM(
        feature_map.term_scale
    )
    phi = ca.vertcat(1.0, standardized)
    raw_standardized = ca.DM(model.coefficients) @ phi
    raw = ca.DM(model.response_center) + ca.DM(model.response_scale) * raw_standardized
    return raw, phi


@dataclass(frozen=True)
class SymbolicNetworkOperators:
    equality_matrix: ca.MX
    equality_rhs: ca.MX
    inequality_matrix: ca.MX
    primary_flow: ca.MX
    clarifier_flow: ca.MX
    underflow: ca.MX
    effluent_flow: ca.MX


def symbolic_network_operators(
    theta: ca.MX,
    influent: ca.MX,
    assets: SurrogateNLPAssets,
) -> SymbolicNetworkOperators:
    """Assemble the symbolic manuscript H/b/G operators."""

    layout = assets.layout
    component_count = layout.component_count
    state_count = layout.state_size
    invariant = assets.invariant_operator
    invariant_count = invariant.shape[0]
    equality_count = assets.equality_count
    equality = ca.MX.zeros(equality_count, state_count)
    rhs = ca.MX.zeros(equality_count, 1)
    identity = ca.DM.eye(component_count)
    internal, returned, waste = theta[4], theta[5], theta[6]
    underflow = returned + waste
    primary = 1.0 + internal + returned
    clarifier = 1.0 + returned
    effluent = 1.0 - waste
    row = 0

    equality[row : row + component_count, layout.mixer_slice] = primary * identity
    equality[
        row : row + component_count,
        layout.reactor_slice(layout.stage_count - 1),
    ] = -internal * identity
    equality[row : row + component_count, layout.underflow_flow_slice] = (
        -returned / underflow * identity
    )
    rhs[row : row + component_count] = influent
    row += component_count

    previous = layout.mixer_slice
    invariant_dm = ca.DM(invariant)
    for stage in range(layout.stage_count):
        current = layout.reactor_slice(stage)
        equality[row : row + invariant_count, current] = invariant_dm
        equality[row : row + invariant_count, previous] = -invariant_dm
        row += invariant_count
        previous = current

    final_reactor = layout.reactor_slice(layout.stage_count - 1)
    equality[row : row + component_count, layout.overflow_flow_slice] = identity
    equality[row : row + component_count, layout.underflow_flow_slice] = identity
    equality[row : row + component_count, final_reactor] = -clarifier * identity
    row += component_count
    for component in layout.soluble_indices:
        equality[row, layout.underflow_flow_slice.start + component] = 1.0
        equality[row, final_reactor.start + component] = -underflow
        row += 1

    overflow_closure = getattr(assets, "overflow_closure", None)
    if overflow_closure is not None:
        log_prediction, _ = symbolic_quadratic_prediction(
            overflow_closure.model, theta, influent,
        )
        overflow_tss = (
            overflow_closure.reference_concentration * ca.exp(log_prediction[0])
        )
        equality[row, layout.overflow_flow_slice] = ca.DM(assets.tss_weights).T
        rhs[row] = effluent * overflow_tss
        row += 1

    if row != equality_count:
        raise AssertionError("symbolic equality-row construction is inconsistent.")

    inequality = ca.MX.zeros(layout.inequality_count, state_count)
    row = 0
    for component in layout.particulate_indices:
        inequality[row, final_reactor.start + component] = underflow
        inequality[row, layout.underflow_flow_slice.start + component] = -1.0
        row += 1
    endpoint_layer_volume = assets.engineering.clarifier_volume_m3 / layout.layer_count
    remaining_volume = assets.engineering.clarifier_volume_m3 - endpoint_layer_volume
    tss = ca.DM(assets.tss_weights).T
    inequality[row, layout.overflow_flow_slice] = underflow * remaining_volume * tss
    inequality[row, layout.underflow_flow_slice] = effluent * endpoint_layer_volume * tss
    inequality[row, layout.inventory_index] = -effluent * underflow
    row += 1
    inequality[row, layout.overflow_flow_slice] = -underflow * endpoint_layer_volume * tss
    inequality[row, layout.underflow_flow_slice] = -effluent * remaining_volume * tss
    inequality[row, layout.inventory_index] = effluent * underflow
    row += 1
    if row != layout.inequality_count:
        raise AssertionError("symbolic inequality-row construction is inconsistent.")
    return SymbolicNetworkOperators(
        equality,
        rhs,
        inequality,
        primary,
        clarifier,
        underflow,
        effluent,
    )


def _engineering_expressions(
    theta: ca.MX,
    state: ca.MX,
    assets: SurrogateNLPAssets,
) -> tuple[ca.MX, tuple[str, ...], ca.MX]:
    layout = assets.layout
    limits = assets.engineering
    tss = ca.DM(assets.tss_weights)
    final = state[layout.reactor_slice(layout.stage_count - 1)]
    overflow = state[layout.overflow_flow_slice]
    underflow_flow = state[layout.underflow_flow_slice]
    clarifier_inventory = state[layout.inventory_index]
    returned, waste = theta[5], theta[6]
    q_underflow = returned + waste
    q_effluent = 1.0 - waste
    q_clarifier = 1.0 + returned
    feed_tss = ca.dot(tss, final)
    underflow_tss = ca.dot(tss, underflow_flow) / q_underflow
    external_loss = ca.dot(tss, overflow) + waste * ca.dot(tss, underflow_flow) / q_underflow
    stage_volume = limits.fresh_flow_m3_d * theta[0] / (24.0 * layout.stage_count)
    reactor_inventory = 0.0
    for stage in range(layout.stage_count):
        reactor_inventory += stage_volume * ca.dot(tss, state[layout.reactor_slice(stage)])
    inventory = reactor_inventory + clarifier_inventory
    sor = limits.fresh_flow_m3_d * q_effluent / limits.clarifier_area_m2
    slr = (
        1.0e-3
        * limits.fresh_flow_m3_d
        * q_clarifier
        * feed_tss
        / limits.clarifier_area_m2
    )
    scale = limits.resolved_inventory_scale
    rows: list[Any] = [
        _scaled_upper_residual(
            limits.srt_lower_d * limits.fresh_flow_m3_d * external_loss - inventory,
            scale,
        ),
        _scaled_upper_residual(
            inventory - limits.srt_upper_d * limits.fresh_flow_m3_d * external_loss,
            scale,
        ),
        _scaled_upper_residual(
            limits.external_loss_min_g_m3 - external_loss,
            limits.external_loss_min_g_m3,
        ),
        _scaled_upper_residual(
            slr - limits.slr_upper_kg_m2_d,
            limits.slr_upper_kg_m2_d,
        ),
        _scaled_upper_residual(
            underflow_tss - limits.underflow_tss_upper_g_m3,
            limits.underflow_tss_upper_g_m3,
        ),
        _scaled_upper_residual(
            limits.feed_tss_min_g_m3 - feed_tss,
            limits.feed_tss_min_g_m3,
        ),
    ]
    names = [
        "srt_lower",
        "srt_upper",
        "external_solids_loss_guard",
        "slr_upper",
        "underflow_tss_upper",
        "feed_tss_lower",
    ]
    if limits.sor_upper_m_d is not None:
        rows.append(
            _scaled_upper_residual(
                sor - limits.sor_upper_m_d,
                limits.sor_upper_m_d,
            )
        )
        names.append("sor_upper")
    # Reported quantities avoid division only where the constraint contract
    # requires it; q_U and external loss are positive on accepted points.
    srt = inventory / (limits.fresh_flow_m3_d * external_loss)
    quantities = ca.vertcat(
        srt,
        sor,
        slr,
        underflow_tss,
        feed_tss,
        external_loss,
        inventory,
    )
    return ca.vertcat(*rows), tuple(names), quantities


def _objective_expressions(
    theta: ca.MX,
    state: ca.MX,
    objective_weights: ca.MX,
    quality_weights: ca.MX,
    assets: SurrogateNLPAssets,
) -> tuple[ca.MX, ca.MX]:
    layout = assets.layout
    lower = assets.theta_lower
    span = assets.theta_span
    q_effluent = 1.0 - theta[6]
    effluent = state[layout.overflow_flow_slice] / q_effluent
    quality_composites = ca.DM(assets.quality_operator) @ effluent
    quality = ca.dot(
        quality_weights,
        quality_composites / ca.DM(assets.quality_scale),
    )
    hrt = (theta[0] - lower[0]) / span[0]
    # Equal-volume case: the common Q0/(24N) factor cancels against its
    # declared maximum, leaving H*sum(a_3:a_5)/(H_U*3).
    aeration = theta[0] * ca.sum1(theta[1:4]) / (assets.theta_upper[0] * 3.0)
    internal = (theta[4] - lower[4]) / span[4]
    returned = (theta[5] - lower[5]) / span[5]
    q_underflow = theta[5] + theta[6]
    underflow_tss = ca.dot(
        ca.DM(assets.tss_weights), state[layout.underflow_flow_slice]
    ) / q_underflow
    wasting = theta[6] * underflow_tss / (
        assets.theta_upper[6] * assets.engineering.underflow_tss_upper_g_m3
    )
    components = ca.vertcat(quality, hrt, aeration, internal, returned, wasting)
    return ca.dot(objective_weights, components), components


def _callback_rows(
    callback: TrustRowCallback,
    theta: ca.MX,
    raw: ca.MX,
    projected: ca.MX,
    influent: ca.MX,
    name: str,
) -> ca.MX:
    try:
        result = callback(theta, raw, projected, influent)
        if isinstance(result, (tuple, list)):
            result = ca.vertcat(*result)
        rows = ca.vec(result)
    except Exception as exc:  # pragma: no cover - message path is deterministic
        raise ValueError(f"{name} trust callback failed during symbolic construction: {exc}") from exc
    if rows.numel() < 1:
        raise ValueError(f"{name} trust callback returned no rows.")
    return rows


def _trust_expressions(
    theta: ca.MX,
    raw: ca.MX,
    projected: ca.MX,
    u: ca.MX,
    influent: ca.MX,
    phi: ca.MX,
    assets: SurrogateNLPAssets,
) -> tuple[ca.MX, tuple[str, ...], ca.MX]:
    thresholds = assets.trust_thresholds
    correction_squared = ca.dot(u, u) / assets.layout.state_size
    leverage = ca.mtimes(
        [phi.T, ca.DM(assets.leverage_precision), phi]
    )
    rows: list[Any] = [
        _normalized_limit_residual(
            correction_squared, thresholds.correction_rms**2
        ),
        _normalized_limit_residual(
            leverage, thresholds.regularized_leverage
        ),
    ]
    values: list[Any] = [correction_squared, leverage]
    names = ["correction", "leverage"]
    specifications = (
        (
            "split",
            assets.trust_callbacks.split_rows,
            thresholds.split_rms,
        ),
        (
            "reactor",
            assets.trust_callbacks.reactor_rows,
            thresholds.reactor_rms,
        ),
    )
    for name, callback, threshold in specifications:
        if callback is None or threshold is None:
            continue
        residual = _callback_rows(callback, theta, raw, projected, influent, name)
        squared = ca.dot(residual, residual) / residual.numel()
        rows.append(_normalized_limit_residual(squared, threshold**2))
        values.append(squared)
        names.append(name)
    for specification in assets.trust_callbacks.additional:
        residual = _callback_rows(
            specification.callback,
            theta,
            raw,
            projected,
            influent,
            specification.name,
        )
        squared = ca.dot(residual, residual) / residual.numel()
        rows.append(
            _normalized_limit_residual(
                squared, specification.rms_threshold**2
            )
        )
        values.append(squared)
        names.append(specification.name)
    return ca.vertcat(*rows), tuple(names), ca.vertcat(*values)


@dataclass(frozen=True)
class SurrogateSolverSettings:
    maximum_iterations: int = 2_500
    tolerance: float = 1.0e-9
    acceptable_tolerance: float = 1.0e-8
    stage_feasibility_tolerance: float = 1.0e-7
    final_upper_tolerance: float = 1.0e-6
    outer_maximum_iterations: int = 250
    outer_function_tolerance: float = 1.0e-10
    perform_outer_refinement: bool = True
    print_level: int = 0
    maximum_wall_time: float | None = None

    def __post_init__(self) -> None:
        if self.maximum_iterations < 1 or self.outer_maximum_iterations < 1:
            raise ValueError("solver iteration limits must be positive.")
        for value in (
            self.tolerance,
            self.acceptable_tolerance,
            self.stage_feasibility_tolerance,
            self.final_upper_tolerance,
            self.outer_function_tolerance,
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("solver tolerances must be finite and positive.")
        if self.maximum_wall_time is not None and (
            not np.isfinite(self.maximum_wall_time) or self.maximum_wall_time <= 0.0
        ):
            raise ValueError("maximum_wall_time must be positive when supplied.")

    def ipopt_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "print_time": False,
            "ipopt.print_level": self.print_level,
            "ipopt.sb": "yes",
            "ipopt.max_iter": self.maximum_iterations,
            "ipopt.tol": self.tolerance,
            "ipopt.acceptable_tol": self.acceptable_tolerance,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.bound_relax_factor": 0.0,
            "ipopt.honor_original_bounds": "yes",
            "ipopt.warm_start_init_point": "yes",
            "ipopt.warm_start_bound_push": 1.0e-9,
            "ipopt.warm_start_bound_frac": 1.0e-9,
            "ipopt.warm_start_slack_bound_push": 1.0e-9,
            "ipopt.warm_start_slack_bound_frac": 1.0e-9,
            "ipopt.warm_start_mult_bound_push": 1.0e-9,
        }
        if self.maximum_wall_time is not None:
            options["ipopt.max_wall_time"] = float(self.maximum_wall_time)
        return options


@dataclass(frozen=True)
class SurrogateCertificationSettings:
    """Frozen finite-resolution local-convergence audit settings.

    This audit is deliberately separate from the optimizer.  It can therefore
    be applied to an already selected endpoint without repeating the global
    data, fit, or single-start search stages.  A passing poll is evidence of
    local convergence at the declared resolution; it is not a KKT or global
    optimality certificate.
    """

    poll_radii: tuple[float, ...] = (1.0e-3, 1.0e-4)
    absolute_decrease_tolerance: float = 1.0e-8
    relative_decrease_tolerance: float = 1.0e-8
    feasibility_tolerance: float = 1.0e-6
    direction_rank_tolerance: float = 1.0e-10
    maximum_evaluations: int = 10_000
    acceleration_growth_factor: float = 2.0
    maximum_acceleration_probes: int = 16

    def __post_init__(self) -> None:
        if not self.poll_radii:
            raise ValueError("poll_radii must not be empty.")
        radii = np.asarray(self.poll_radii, dtype=np.float64)
        if (
            not np.all(np.isfinite(radii))
            or np.any(radii <= 0.0)
            or np.any(np.diff(radii) >= 0.0)
        ):
            raise ValueError("poll_radii must be finite, positive, and decreasing.")
        for value in (
            self.absolute_decrease_tolerance,
            self.relative_decrease_tolerance,
            self.feasibility_tolerance,
            self.direction_rank_tolerance,
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("certification tolerances must be finite and positive.")
        if self.maximum_evaluations < 1:
            raise ValueError("maximum_evaluations must be positive.")
        if (
            not np.isfinite(self.acceleration_growth_factor)
            or self.acceleration_growth_factor <= 1.0
        ):
            raise ValueError("acceleration_growth_factor must be finite and above one.")
        if self.maximum_acceleration_probes < 1:
            raise ValueError("maximum_acceleration_probes must be positive.")


@dataclass(frozen=True)
class LocalConvergenceCertificate:
    protocol: str
    classification: str
    locally_converged: bool
    first_order_certified: bool
    stationarity_resolved: bool
    initial_objective: float
    final_objective: float
    initial_normalized_controls: FloatArray
    final_normalized_controls: FloatArray
    evaluations: int
    cold_qp_resolutions: int
    accepted_improvements: int
    elapsed_seconds: float
    termination_reason: str
    poll_levels: tuple[dict[str, Any], ...]
    lower_active_set: dict[str, Any] | None = None
    upper_kkt: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "initial_normalized_controls": self.initial_normalized_controls.tolist(),
            "final_normalized_controls": self.final_normalized_controls.tolist(),
            "poll_levels": [dict(item) for item in self.poll_levels],
        }


@dataclass(frozen=True)
class SurrogateCertificationResult:
    candidate: FinalCandidateRecord | None
    certificate: LocalConvergenceCertificate

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": None if self.candidate is None else self.candidate.as_dict(),
            "certificate": self.certificate.as_dict(),
        }


def _update_overflow_closure_digest(
    digest: Any, closure: LogOverflowTSSClosure | None,
) -> None:
    if closure is None:
        digest.update(b"\0no-log-overflow-closure\0")
        return
    digest.update(b"\0log-overflow-closure-v1\0")
    feature_map = closure.model.feature_map
    for value in (
        feature_map.decision_center,
        feature_map.decision_scale,
        feature_map.influent_center,
        feature_map.influent_scale,
        feature_map.term_center,
        feature_map.term_scale,
        closure.model.response_center,
        closure.model.response_scale,
        closure.model.coefficients,
        np.asarray([
            feature_map.variance_relative_tolerance,
            closure.model.ridge_penalty,
            closure.reference_concentration,
        ]),
    ):
        digest.update(np.asarray(value, dtype="<f8").tobytes(order="C"))


def surrogate_start_resume_contract(
    assets: SurrogateNLPAssets,
    case: SurrogateCase,
    settings: SurrogateSolverSettings,
) -> str:
    """Return the stable case/settings fingerprint required for start reuse.

    The full-run manifest separately binds model and source artifacts.  This
    library-level token prevents a completed start from being reused for a
    different case vector, case identifier, continuation schedule, or solver
    contract.
    """

    digest = sha256()
    digest.update(b"manuscript-v3-surrogate-start-resume-v1\0")
    digest.update(case.case_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(
        np.asarray(case.parameter_vector(assets), dtype="<f8").tobytes(order="C")
    )
    digest.update(np.asarray(GAP_CONTINUATION, dtype="<f8").tobytes(order="C"))
    _update_overflow_closure_digest(digest, assets.overflow_closure)
    digest.update(
        json.dumps(
            asdict(settings),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def surrogate_exact_qp_resume_contract(
    assets: SurrogateNLPAssets,
    case: SurrogateCase,
    settings: SurrogateSolverSettings,
) -> str:
    """Fingerprint the direct seven-variable, single-start protocol.

    This token is deliberately disjoint from
    :func:`surrogate_start_resume_contract`: a checkpoint produced by the
    embedded-KKT continuation route can never be mistaken for a direct
    exact-QP local-optimization result. IPOPT-only settings are omitted
    because this protocol does not construct or call an IPOPT solver.
    """

    digest = sha256()
    digest.update(b"manuscript-v3-surrogate-exact-qp-single-start-resume-v1\0")
    digest.update(EXACT_QP_SINGLE_START_PROTOCOL.encode("utf-8"))
    digest.update(b"\0")
    digest.update(case.case_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(
        np.asarray(case.parameter_vector(assets), dtype="<f8").tobytes(order="C")
    )
    digest.update(np.asarray(EXACT_QP_CENTER_START, dtype="<f8").tobytes(order="C"))
    _update_overflow_closure_digest(digest, assets.overflow_closure)
    digest.update(
        json.dumps(
            {
                "final_upper_tolerance": settings.final_upper_tolerance,
                "outer_maximum_iterations": settings.outer_maximum_iterations,
                "outer_function_tolerance": settings.outer_function_tolerance,
                "perform_outer_refinement": settings.perform_outer_refinement,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


@dataclass
class SurrogateNLP:
    assets: SurrogateNLPAssets
    tau: float
    name: str
    variable_count: int
    equality_count: int
    inequality_count: int
    theta_slice: slice
    displacement_slice: slice
    equality_multiplier_slice: slice
    inequality_multiplier_slice: slice
    lower_bounds: FloatArray
    upper_bounds: FloatArray
    constraint_lower_bounds: FloatArray
    constraint_upper_bounds: FloatArray
    solver: Any | None
    objective_function: ca.Function
    raw_function: ca.Function
    feature_function: ca.Function
    state_function: ca.Function
    network_function: ca.Function
    equality_function: ca.Function
    inequality_function: ca.Function
    gap_function: ca.Function
    engineering_function: ca.Function
    trust_function: ca.Function
    upper_from_state_function: ca.Function
    objective_components_function: ca.Function
    engineering_quantity_function: ca.Function
    engineering_names: tuple[str, ...]
    trust_names: tuple[str, ...]

    @property
    def parameter_count(self) -> int:
        return self.assets.layout.component_count + 6 + self.assets.quality_count


def build_surrogate_nlp(
    assets: SurrogateNLPAssets,
    tau: float,
    *,
    settings: SurrogateSolverSettings | None = None,
    name: str = "v3_surrogate",
    compile_solver: bool = True,
) -> SurrogateNLP:
    """Build one primal-dual-gap stage of the v3 surrogate route."""

    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive.")
    settings = settings or SurrogateSolverSettings()
    safe = _safe_name(f"{name}_{tau:.0e}")
    n_state = assets.layout.state_size
    n_equality = assets.equality_count
    n_q = assets.projection_inequality_count
    theta_slice = slice(0, 7)
    displacement_slice = slice(theta_slice.stop, theta_slice.stop + n_state)
    equality_multiplier_slice = slice(
        displacement_slice.stop, displacement_slice.stop + n_equality
    )
    inequality_multiplier_slice = slice(
        equality_multiplier_slice.stop, equality_multiplier_slice.stop + n_q
    )
    variable_count = inequality_multiplier_slice.stop
    variable = ca.MX.sym(f"{safe}_variable", variable_count)
    parameter_count = assets.layout.component_count + 6 + assets.quality_count
    parameter = ca.MX.sym(f"{safe}_parameter", parameter_count)
    normalized_theta = variable[theta_slice]
    theta = ca.DM(assets.theta_lower) + ca.DM(assets.theta_span) * normalized_theta
    u = variable[displacement_slice]
    lambda_equality = variable[equality_multiplier_slice]
    lambda_inequality = variable[inequality_multiplier_slice]
    influent = parameter[: assets.layout.component_count]
    objective_weights = parameter[
        assets.layout.component_count : assets.layout.component_count + 6
    ]
    quality_weights = parameter[assets.layout.component_count + 6 :]

    raw, phi = symbolic_quadratic_prediction(assets.model, theta, influent)
    network = symbolic_network_operators(theta, influent, assets)
    state_scale = ca.DM(assets.model.response_scale)
    equality_scale = ca.DM(assets.row_scales.equality)
    inequality_scale = ca.DM(assets.row_scales.inequality)
    projected = raw + state_scale * u
    scaled_equality = ca.diag(1.0 / equality_scale) @ network.equality_matrix @ ca.diag(
        state_scale
    )
    required_equality = (
        network.equality_rhs - network.equality_matrix @ raw
    ) / equality_scale
    scaled_network_inequality = (
        ca.diag(1.0 / inequality_scale)
        @ network.inequality_matrix
        @ ca.diag(state_scale)
    )
    projection_inequality = ca.vertcat(-ca.DM.eye(n_state), scaled_network_inequality)
    projection_rhs = ca.vertcat(
        raw / state_scale,
        -(network.inequality_matrix @ raw) / inequality_scale,
    )
    lower_equality = scaled_equality @ u - required_equality
    lower_inequality = projection_inequality @ u - projection_rhs
    stationarity = (
        u
        + scaled_equality.T @ lambda_equality
        + projection_inequality.T @ lambda_inequality
    )
    slack = projection_rhs - projection_inequality @ u
    gap = 0.5 * ca.dot(stationarity, stationarity) + ca.dot(lambda_inequality, slack)
    normalized_gap = gap / (n_state + n_q)

    engineering, engineering_names, quantities = _engineering_expressions(
        theta, projected, assets
    )
    trust, trust_names, trust_values = _trust_expressions(
        theta, raw, projected, u, influent, phi, assets
    )
    objective, components = _objective_expressions(
        theta,
        projected,
        objective_weights,
        quality_weights,
        assets,
    )
    inequality = ca.vertcat(
        lower_inequality,
        -normalized_gap,
        normalized_gap - tau,
        engineering,
        trust,
    )
    constraints = ca.vertcat(lower_equality, inequality)
    lower_bounds = np.full(variable_count, -np.inf, dtype=np.float64)
    upper_bounds = np.full(variable_count, np.inf, dtype=np.float64)
    lower_bounds[theta_slice] = 0.0
    upper_bounds[theta_slice] = 1.0
    lower_bounds[inequality_multiplier_slice] = 0.0
    constraint_lower = np.concatenate(
        (np.zeros(n_equality), np.full(int(inequality.numel()), -np.inf))
    )
    constraint_upper = np.zeros(int(constraints.numel()), dtype=np.float64)
    solver = None
    if compile_solver:
        solver = ca.nlpsol(
            f"{safe}_ipopt",
            "ipopt",
            {"x": variable, "p": parameter, "f": objective, "g": constraints},
            settings.ipopt_options(),
        )

    # Diagnostic and exact-QP functions take a standalone seven-vector.
    # A slice of ``variable`` is an expression rather than a pure CasADi
    # symbol and therefore cannot be used as a Function input.
    evaluation_normalized = ca.MX.sym(f"{safe}_normalized", 7)
    evaluation_theta = (
        ca.DM(assets.theta_lower)
        + ca.DM(assets.theta_span) * evaluation_normalized
    )
    evaluation_influent = parameter[: assets.layout.component_count]
    evaluation_raw, evaluation_phi = symbolic_quadratic_prediction(
        assets.model, evaluation_theta, evaluation_influent
    )
    evaluation_network = symbolic_network_operators(
        evaluation_theta, evaluation_influent, assets
    )

    # Upper expressions evaluated on an independently projected state.  This
    # graph is also the sole implementation used by exact-QP refinement.
    exact_state = ca.MX.sym(f"{safe}_exact_state", n_state)
    exact_u = (exact_state - evaluation_raw) / state_scale
    exact_engineering, _, exact_quantities = _engineering_expressions(
        evaluation_theta, exact_state, assets
    )
    exact_trust, _, exact_trust_values = _trust_expressions(
        evaluation_theta, evaluation_raw, exact_state, exact_u,
        evaluation_influent, evaluation_phi, assets
    )
    exact_objective, exact_components = _objective_expressions(
        evaluation_theta,
        exact_state,
        objective_weights,
        quality_weights,
        assets,
    )
    network_outputs = (
        evaluation_network.equality_matrix,
        evaluation_network.equality_rhs,
        evaluation_network.inequality_matrix,
    )
    return SurrogateNLP(
        assets=assets,
        tau=float(tau),
        name=safe,
        variable_count=variable_count,
        equality_count=n_equality,
        inequality_count=int(inequality.numel()),
        theta_slice=theta_slice,
        displacement_slice=displacement_slice,
        equality_multiplier_slice=equality_multiplier_slice,
        inequality_multiplier_slice=inequality_multiplier_slice,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        constraint_lower_bounds=constraint_lower,
        constraint_upper_bounds=constraint_upper,
        solver=solver,
        objective_function=ca.Function(f"{safe}_objective", [variable, parameter], [objective]),
        raw_function=ca.Function(
            f"{safe}_raw", [evaluation_normalized, parameter], [evaluation_raw]
        ),
        feature_function=ca.Function(
            f"{safe}_feature", [evaluation_normalized, parameter], [evaluation_phi]
        ),
        state_function=ca.Function(f"{safe}_state", [variable, parameter], [projected]),
        network_function=ca.Function(
            f"{safe}_network", [evaluation_normalized, parameter], list(network_outputs)
        ),
        equality_function=ca.Function(
            f"{safe}_equality", [variable, parameter], [lower_equality]
        ),
        inequality_function=ca.Function(
            f"{safe}_inequality", [variable, parameter], [inequality]
        ),
        gap_function=ca.Function(f"{safe}_gap", [variable, parameter], [normalized_gap]),
        engineering_function=ca.Function(
            f"{safe}_engineering", [variable, parameter], [engineering]
        ),
        trust_function=ca.Function(f"{safe}_trust", [variable, parameter], [trust, trust_values]),
        upper_from_state_function=ca.Function(
            f"{safe}_upper_exact",
            [evaluation_normalized, parameter, exact_state],
            [exact_objective, exact_engineering, exact_trust, exact_components,
             exact_quantities, exact_trust_values],
        ),
        objective_components_function=ca.Function(
            f"{safe}_components", [variable, parameter], [components]
        ),
        engineering_quantity_function=ca.Function(
            f"{safe}_quantities", [variable, parameter], [quantities]
        ),
        engineering_names=engineering_names,
        trust_names=trust_names,
    )


def unpack_primal(
    problem: SurrogateNLP,
    primal: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    value = _vector(primal, problem.variable_count, "primal")
    return (
        value[problem.theta_slice],
        value[problem.displacement_slice],
        value[problem.equality_multiplier_slice],
        value[problem.inequality_multiplier_slice],
    )


def evaluate_surrogate_problem(
    problem: SurrogateNLP,
    primal: npt.ArrayLike,
    case: SurrogateCase,
) -> dict[str, Any]:
    value = _vector(primal, problem.variable_count, "primal")
    parameter = case.parameter_vector(problem.assets)
    normalized, displacement, lambda_eq, lambda_ineq = unpack_primal(problem, value)
    theta = problem.assets.theta_lower + problem.assets.theta_span * normalized
    trust_output = problem.trust_function(value, parameter)
    return {
        "objective": float(problem.objective_function(value, parameter)),
        "normalized_controls": normalized,
        "theta": theta,
        "displacement": displacement,
        "lambda_eq": lambda_eq,
        "lambda_ineq": lambda_ineq,
        "raw": _flat(problem.raw_function(normalized, parameter)),
        "projected": _flat(problem.state_function(value, parameter)),
        "equality": _flat(problem.equality_function(value, parameter)),
        "inequality": _flat(problem.inequality_function(value, parameter)),
        "normalized_gap": float(problem.gap_function(value, parameter)),
        "engineering": _flat(problem.engineering_function(value, parameter)),
        "trust": _flat(trust_output[0]),
        "trust_values": _flat(trust_output[1]),
        "objective_components": _flat(
            problem.objective_components_function(value, parameter)
        ),
        "engineering_quantities": _flat(
            problem.engineering_quantity_function(value, parameter)
        ),
    }


def cold_reproject(
    assets: SurrogateNLPAssets,
    case: SurrogateCase,
    normalized_controls: npt.ArrayLike,
    *,
    raise_on_failure: bool = False,
) -> ProjectionResult:
    """Resolve a newly initialized OSQP projection at one control vector."""

    normalized = _vector(normalized_controls, 7, "normalized_controls")
    if np.any(normalized < 0.0) or np.any(normalized > 1.0):
        raise ValueError("normalized_controls must lie in [0, 1].")
    influent = _vector(case.influent, assets.layout.component_count, "case influent")
    theta = assets.theta_lower + assets.theta_span * normalized
    raw = np.asarray(assets.model.predict(theta, influent), dtype=np.float64)
    operators = build_network_operators(
        influent,
        internal_recycle=float(theta[4]),
        return_recycle=float(theta[5]),
        waste_fraction=float(theta[6]),
        invariant_operator=assets.invariant_operator,
        tss_weights=assets.tss_weights,
        layout=assets.layout,
        clarifier_volume_m3=assets.engineering.clarifier_volume_m3,
        overflow_tss_closure=(
            None
            if assets.overflow_closure is None
            else float(assets.overflow_closure.predict(theta, influent))
        ),
    )
    # A new projector means a new OSQP workspace; no primal or dual warm start
    # can leak from the continuation or a preceding outer trial.
    projector = PhysicalProjector(
        assets.model.response_scale,
        assets.row_scales.equality,
        assets.row_scales.inequality,
        absolute_tolerance=1.0e-10,
        relative_tolerance=1.0e-10,
        maximum_iterations=100_000,
        polish=True,
    )
    return projector.project(
        raw,
        operators.equality_matrix,
        operators.equality_rhs,
        operators.inequality_matrix,
        warm_start=None,
        raise_on_failure=raise_on_failure,
    )


def initial_primal_from_projection(
    problem: SurrogateNLP,
    case: SurrogateCase,
    normalized_controls: npt.ArrayLike,
) -> tuple[FloatArray, ProjectionResult]:
    normalized = _vector(normalized_controls, 7, "normalized_controls")
    projection = cold_reproject(problem.assets, case, normalized, raise_on_failure=False)
    if not projection.accepted:
        raise SurrogateNLPError("the initial cold projection failed its independent KKT audit.")
    primal = np.concatenate(
        (
            normalized,
            projection.displacement,
            projection.equality_multipliers,
            projection.inequality_multipliers,
        )
    )
    return primal, projection


@dataclass(frozen=True)
class ContinuationStageRecord:
    tau: float
    status: str
    solver_success: bool
    iterations: int
    elapsed_seconds: float
    feasible: bool
    equality_residual: float
    inequality_residual: float
    bound_residual: float
    normalized_gap: float
    primal: FloatArray
    error: str | None = None
    constraint_multipliers: FloatArray | None = None

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["primal"] = self.primal.tolist()
        value["constraint_multipliers"] = (
            None
            if self.constraint_multipliers is None
            else self.constraint_multipliers.tolist()
        )
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContinuationStageRecord":
        return cls(
            tau=_float_or_nan(value.get("tau")),
            status=str(value.get("status", "unknown")),
            solver_success=bool(value.get("solver_success", False)),
            iterations=int(value.get("iterations", 0)),
            elapsed_seconds=_float_or_nan(value.get("elapsed_seconds")),
            feasible=bool(value.get("feasible", False)),
            equality_residual=_float_or_nan(value.get("equality_residual")),
            inequality_residual=_float_or_nan(value.get("inequality_residual")),
            bound_residual=_float_or_nan(value.get("bound_residual")),
            normalized_gap=_float_or_nan(value.get("normalized_gap")),
            primal=np.asarray(value.get("primal"), dtype=np.float64).reshape(-1),
            error=None if value.get("error") is None else str(value["error"]),
            constraint_multipliers=(
                None
                if value.get("constraint_multipliers") is None
                else np.asarray(
                    value["constraint_multipliers"], dtype=np.float64
                ).reshape(-1)
            ),
        )


@dataclass(frozen=True)
class _ContinuationSolveResult:
    """One stage plus IPOPT's outer duals for the next gap stage."""

    stage: ContinuationStageRecord
    bound_multipliers: FloatArray | None
    constraint_multipliers: FloatArray | None


@dataclass(frozen=True)
class FeasibilityRecord:
    finite: bool
    cold_projection: bool
    projection_accepted: bool
    control_bound_residual: float
    engineering_residual: float
    trust_residual: float
    maximum_upper_residual: float
    feasible: bool
    projection_reproduction_residual: float | None = None
    projection_reproduction_passed: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeasibilityRecord":
        return cls(
            finite=bool(value.get("finite", False)),
            cold_projection=bool(value.get("cold_projection", False)),
            projection_accepted=bool(value.get("projection_accepted", False)),
            control_bound_residual=_float_or_nan(value.get("control_bound_residual")),
            engineering_residual=_float_or_nan(value.get("engineering_residual")),
            trust_residual=_float_or_nan(value.get("trust_residual")),
            maximum_upper_residual=_float_or_nan(value.get("maximum_upper_residual")),
            feasible=bool(value.get("feasible", False)),
            projection_reproduction_residual=_optional_float(
                value.get("projection_reproduction_residual")
            ),
            projection_reproduction_passed=(
                None
                if value.get("projection_reproduction_passed") is None
                else bool(value["projection_reproduction_passed"])
            ),
        )


@dataclass(frozen=True)
class StationarityRecord:
    classification: str
    resolved: bool
    stationary: bool
    lower_qp_kkt_passed: bool
    upper_stationarity_residual: float | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StationarityRecord":
        return cls(
            classification=str(value.get("classification", "stationarity_unresolved")),
            resolved=bool(value.get("resolved", False)),
            stationary=bool(value.get("stationary", False)),
            lower_qp_kkt_passed=bool(value.get("lower_qp_kkt_passed", False)),
            upper_stationarity_residual=_optional_float(
                value.get("upper_stationarity_residual")
            ),
            reason=str(value.get("reason", "checkpoint did not record a reason")),
        )


@dataclass(frozen=True)
class OuterRefinementRecord:
    attempted: bool
    solver_success: bool
    status: str
    iterations: int
    evaluations: int
    elapsed_seconds: float
    initial_objective: float | None
    final_objective: float | None
    cold_qp_resolutions: int = 0
    derivative_error: str | None = None
    lower_active_set: dict[str, Any] | None = None
    upper_kkt: dict[str, Any] | None = None
    projection_reproduction_residual: float | None = None
    projection_reproduction_passed: bool | None = None
    method: str = "exact_qp_active_set_slsqp"
    fallback_used: bool = False
    fallback_method: str | None = None
    fallback_solver_success: bool | None = None
    fallback_status: str | None = None
    fallback_iterations: int = 0
    fallback_evaluations: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OuterRefinementRecord":
        lower = value.get("lower_active_set")
        upper = value.get("upper_kkt")
        return cls(
            attempted=bool(value.get("attempted", False)),
            solver_success=bool(value.get("solver_success", False)),
            status=str(value.get("status", "unknown")),
            iterations=int(value.get("iterations", 0)),
            evaluations=int(value.get("evaluations", 0)),
            elapsed_seconds=_float_or_nan(value.get("elapsed_seconds")),
            initial_objective=_optional_float(value.get("initial_objective")),
            final_objective=_optional_float(value.get("final_objective")),
            cold_qp_resolutions=int(value.get("cold_qp_resolutions", 0)),
            derivative_error=(
                None if value.get("derivative_error") is None else str(value["derivative_error"])
            ),
            lower_active_set=None if lower is None else dict(lower),
            upper_kkt=None if upper is None else dict(upper),
            projection_reproduction_residual=_optional_float(
                value.get("projection_reproduction_residual")
            ),
            projection_reproduction_passed=(
                None
                if value.get("projection_reproduction_passed") is None
                else bool(value["projection_reproduction_passed"])
            ),
            method=str(value.get("method", "exact_qp_active_set_slsqp")),
            fallback_used=bool(value.get("fallback_used", False)),
            fallback_method=(
                None
                if value.get("fallback_method") is None
                else str(value["fallback_method"])
            ),
            fallback_solver_success=(
                None
                if value.get("fallback_solver_success") is None
                else bool(value["fallback_solver_success"])
            ),
            fallback_status=(
                None
                if value.get("fallback_status") is None
                else str(value["fallback_status"])
            ),
            fallback_iterations=int(value.get("fallback_iterations", 0)),
            fallback_evaluations=int(value.get("fallback_evaluations", 0)),
        )


@dataclass(frozen=True)
class FinalCandidateRecord:
    normalized_controls: FloatArray
    theta: FloatArray
    raw: FloatArray
    projected: FloatArray
    displacement: FloatArray
    objective: float
    objective_components: FloatArray
    engineering_rows: FloatArray
    engineering_quantities: FloatArray
    trust_rows: FloatArray
    trust_values: FloatArray
    projection: ProjectionResult
    feasibility: FeasibilityRecord
    stationarity: StationarityRecord
    status: str
    lower_active_set: dict[str, Any] | None = None
    upper_kkt: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "normalized_controls": self.normalized_controls.tolist(),
            "theta": self.theta.tolist(),
            "raw": self.raw.tolist(),
            "projected": self.projected.tolist(),
            "displacement": self.displacement.tolist(),
            "objective": self.objective,
            "objective_components": self.objective_components.tolist(),
            "engineering_rows": self.engineering_rows.tolist(),
            "engineering_quantities": self.engineering_quantities.tolist(),
            "trust_rows": self.trust_rows.tolist(),
            "trust_values": self.trust_values.tolist(),
            "projection": {
                "accepted": self.projection.accepted,
                "state": self.projection.state.tolist(),
                "displacement": self.projection.displacement.tolist(),
                "equality_multipliers": self.projection.equality_multipliers.tolist(),
                "inequality_multipliers": self.projection.inequality_multipliers.tolist(),
                "inequality_slack": self.projection.inequality_slack.tolist(),
                "diagnostics": self.projection.diagnostics.as_dict(),
            },
            "feasibility": self.feasibility.as_dict(),
            "stationarity": self.stationarity.as_dict(),
            "status": self.status,
            "lower_active_set": self.lower_active_set,
            "upper_kkt": self.upper_kkt,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FinalCandidateRecord":
        projection_value = value["projection"]
        diagnostics_value = projection_value["diagnostics"]
        diagnostics = ProjectionDiagnostics(
            status=str(diagnostics_value.get("status", "unknown")),
            status_value=int(diagnostics_value.get("status_value", 0)),
            iterations=int(diagnostics_value.get("iterations", 0)),
            equality_rank_tolerance=_float_or_nan(
                diagnostics_value.get("equality_rank_tolerance")
            ),
            equality_smallest_singular_value=_float_or_nan(
                diagnostics_value.get("equality_smallest_singular_value")
            ),
            equality_condition_number=_float_or_nan(
                diagnostics_value.get("equality_condition_number")
            ),
            equality_residual=_float_or_nan(
                diagnostics_value.get("equality_residual")
            ),
            inequality_residual=_float_or_nan(
                diagnostics_value.get("inequality_residual")
            ),
            nonnegativity_residual=_float_or_nan(
                diagnostics_value.get("nonnegativity_residual")
            ),
            dual_feasibility_residual=_float_or_nan(
                diagnostics_value.get("dual_feasibility_residual")
            ),
            stationarity_residual=_float_or_nan(
                diagnostics_value.get("stationarity_residual")
            ),
            complementarity_residual=_float_or_nan(
                diagnostics_value.get("complementarity_residual")
            ),
            retried_cold=bool(diagnostics_value.get("retried_cold", False)),
            active_inequality_count=int(
                diagnostics_value.get("active_inequality_count", 0)
            ),
            multipliers_reconstructed=bool(
                diagnostics_value.get("multipliers_reconstructed", False)
            ),
            solver_attempts=int(diagnostics_value.get("solver_attempts", 1)),
            fallback_used=bool(diagnostics_value.get("fallback_used", False)),
        )
        projection = ProjectionResult(
            state=np.asarray(projection_value.get("state"), dtype=np.float64).reshape(-1),
            displacement=np.asarray(
                projection_value.get("displacement"), dtype=np.float64
            ).reshape(-1),
            equality_multipliers=np.asarray(
                projection_value.get("equality_multipliers"), dtype=np.float64
            ).reshape(-1),
            inequality_multipliers=np.asarray(
                projection_value.get("inequality_multipliers"), dtype=np.float64
            ).reshape(-1),
            inequality_slack=np.asarray(
                projection_value.get("inequality_slack"), dtype=np.float64
            ).reshape(-1),
            diagnostics=diagnostics,
            accepted=bool(projection_value.get("accepted", False)),
        )
        lower = value.get("lower_active_set")
        upper = value.get("upper_kkt")
        return cls(
            normalized_controls=np.asarray(
                value.get("normalized_controls"), dtype=np.float64
            ).reshape(-1),
            theta=np.asarray(value.get("theta"), dtype=np.float64).reshape(-1),
            raw=np.asarray(value.get("raw"), dtype=np.float64).reshape(-1),
            projected=np.asarray(value.get("projected"), dtype=np.float64).reshape(-1),
            displacement=np.asarray(
                value.get("displacement"), dtype=np.float64
            ).reshape(-1),
            objective=_float_or_nan(value.get("objective")),
            objective_components=np.asarray(
                value.get("objective_components"), dtype=np.float64
            ).reshape(-1),
            engineering_rows=np.asarray(
                value.get("engineering_rows"), dtype=np.float64
            ).reshape(-1),
            engineering_quantities=np.asarray(
                value.get("engineering_quantities"), dtype=np.float64
            ).reshape(-1),
            trust_rows=np.asarray(value.get("trust_rows"), dtype=np.float64).reshape(-1),
            trust_values=np.asarray(
                value.get("trust_values"), dtype=np.float64
            ).reshape(-1),
            projection=projection,
            feasibility=FeasibilityRecord.from_dict(value["feasibility"]),
            stationarity=StationarityRecord.from_dict(value["stationarity"]),
            status=str(value.get("status", "unknown")),
            lower_active_set=None if lower is None else dict(lower),
            upper_kkt=None if upper is None else dict(upper),
        )


@dataclass(frozen=True)
class SurrogateStartResult:
    start_index: int
    initial_normalized_controls: FloatArray
    stages: tuple[ContinuationStageRecord, ...]
    outer_refinement: OuterRefinementRecord
    final: FinalCandidateRecord | None
    status: str
    error: str | None = None
    resume_contract: str | None = None
    protocol: str = LEGACY_SURROGATE_PROTOCOL

    @property
    def feasible(self) -> bool:
        return self.final is not None and self.final.feasibility.feasible

    @property
    def stationary(self) -> bool:
        return self.final is not None and self.final.stationarity.stationary

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "initial_normalized_controls": self.initial_normalized_controls.tolist(),
            "stages": [stage.as_dict() for stage in self.stages],
            "outer_refinement": self.outer_refinement.as_dict(),
            "final": None if self.final is None else self.final.as_dict(),
            "status": self.status,
            "error": self.error,
            "resume_contract": self.resume_contract,
            "protocol": self.protocol,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SurrogateStartResult":
        final_value = value.get("final")
        return cls(
            start_index=int(value["start_index"]),
            initial_normalized_controls=np.asarray(
                value.get("initial_normalized_controls"), dtype=np.float64
            ).reshape(-1),
            stages=tuple(
                ContinuationStageRecord.from_dict(item)
                for item in value.get("stages", ())
            ),
            outer_refinement=OuterRefinementRecord.from_dict(
                value.get("outer_refinement", {})
            ),
            final=(
                None
                if final_value is None
                else FinalCandidateRecord.from_dict(final_value)
            ),
            status=str(value.get("status", "unknown")),
            error=None if value.get("error") is None else str(value["error"]),
            resume_contract=(
                None
                if value.get("resume_contract") is None
                else str(value["resume_contract"])
            ),
            protocol=str(value.get("protocol", LEGACY_SURROGATE_PROTOCOL)),
        )


@dataclass(frozen=True)
class SurrogateMultistartResult:
    starts: tuple[SurrogateStartResult, ...]
    selected: SurrogateStartResult | None
    status: str
    protocol: str = LEGACY_SURROGATE_PROTOCOL

    def as_dict(self) -> dict[str, Any]:
        return {
            "starts": [result.as_dict() for result in self.starts],
            "selected_start": None if self.selected is None else self.selected.start_index,
            "status": self.status,
            "protocol": self.protocol,
        }


def _solve_continuation_stage(
    problem: SurrogateNLP,
    case: SurrogateCase,
    initial: FloatArray,
    settings: SurrogateSolverSettings,
    dual_warm_start: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
) -> _ContinuationSolveResult:
    if problem.solver is None:
        raise SurrogateNLPError("the continuation problem was built without an NLP solver.")
    started = perf_counter()
    try:
        arguments: dict[str, Any] = dict(
            x0=initial,
            p=case.parameter_vector(problem.assets),
            lbx=problem.lower_bounds,
            ubx=problem.upper_bounds,
            lbg=problem.constraint_lower_bounds,
            ubg=problem.constraint_upper_bounds,
        )
        if dual_warm_start is not None:
            arguments["lam_x0"] = _vector(
                dual_warm_start[0], problem.variable_count, "IPOPT bound warm start"
            )
            arguments["lam_g0"] = _vector(
                dual_warm_start[1],
                problem.constraint_lower_bounds.size,
                "IPOPT constraint warm start",
            )
        solution = problem.solver(**arguments)
        elapsed = perf_counter() - started
        primal = _flat(solution["x"])
        bound_multipliers = _vector(
            _flat(solution["lam_x"]),
            problem.variable_count,
            "IPOPT bound multipliers",
        )
        constraint_multipliers = _vector(
            _flat(solution["lam_g"]),
            problem.constraint_lower_bounds.size,
            "IPOPT constraint multipliers",
        )
        evaluated = evaluate_surrogate_problem(problem, primal, case)
        equality_residual = float(
            np.max(np.abs(evaluated["equality"]), initial=0.0)
        )
        inequality_residual = _maximum_positive(evaluated["inequality"])
        bound_residual = max(
            _maximum_positive(problem.lower_bounds - primal),
            _maximum_positive(primal - problem.upper_bounds),
        )
        finite = all(
            np.all(np.isfinite(np.asarray(value)))
            for value in (
                primal,
                evaluated["objective"],
                evaluated["equality"],
                evaluated["inequality"],
            )
        )
        feasible = bool(
            finite
            and equality_residual <= settings.stage_feasibility_tolerance
            and inequality_residual <= settings.stage_feasibility_tolerance
            and bound_residual <= settings.stage_feasibility_tolerance
        )
        stats = problem.solver.stats()
        return _ContinuationSolveResult(
            stage=ContinuationStageRecord(
                tau=problem.tau,
                status=str(stats.get("return_status", "unknown")),
                solver_success=bool(stats.get("success", False)),
                iterations=int(stats.get("iter_count", 0)),
                elapsed_seconds=elapsed,
                feasible=feasible,
                equality_residual=equality_residual,
                inequality_residual=inequality_residual,
                bound_residual=bound_residual,
                normalized_gap=float(evaluated["normalized_gap"]),
                primal=primal,
                constraint_multipliers=constraint_multipliers,
            ),
            bound_multipliers=bound_multipliers,
            constraint_multipliers=constraint_multipliers,
        )
    except Exception as exc:
        return _ContinuationSolveResult(
            stage=ContinuationStageRecord(
                tau=problem.tau,
                status="solver_exception",
                solver_success=False,
                iterations=0,
                elapsed_seconds=perf_counter() - started,
                feasible=False,
                equality_residual=np.inf,
                inequality_residual=np.inf,
                bound_residual=np.inf,
                normalized_gap=np.inf,
                primal=initial.copy(),
                error=f"{type(exc).__name__}: {exc}",
            ),
            bound_multipliers=None,
            constraint_multipliers=None,
        )


@dataclass(frozen=True)
class _ExactEvaluation:
    normalized: FloatArray
    projection: ProjectionResult
    objective: float
    engineering: FloatArray
    trust: FloatArray
    components: FloatArray
    quantities: FloatArray
    trust_values: FloatArray


def _exact_evaluation(
    problem: SurrogateNLP,
    case: SurrogateCase,
    normalized: FloatArray,
) -> _ExactEvaluation:
    projection = cold_reproject(problem.assets, case, normalized, raise_on_failure=False)
    parameter = case.parameter_vector(problem.assets)
    outputs = problem.upper_from_state_function(normalized, parameter, projection.state)
    return _ExactEvaluation(
        normalized=normalized.copy(),
        projection=projection,
        objective=float(outputs[0]),
        engineering=_flat(outputs[1]),
        trust=_flat(outputs[2]),
        components=_flat(outputs[3]),
        quantities=_flat(outputs[4]),
        trust_values=_flat(outputs[5]),
    )


def _feasibility_record(
    evaluation: _ExactEvaluation,
    settings: SurrogateSolverSettings,
) -> FeasibilityRecord:
    normalized = evaluation.normalized
    control = max(
        _maximum_positive(-normalized),
        _maximum_positive(normalized - 1.0),
    )
    engineering = _maximum_positive(evaluation.engineering)
    trust = _maximum_positive(evaluation.trust)
    finite = all(
        np.all(np.isfinite(np.asarray(value)))
        for value in (
            normalized,
            evaluation.projection.state,
            evaluation.objective,
            evaluation.engineering,
            evaluation.trust,
        )
    )
    maximum = max(control, engineering, trust)
    feasible = bool(
        finite
        and evaluation.projection.accepted
        and maximum <= settings.final_upper_tolerance
    )
    return FeasibilityRecord(
        finite=finite,
        cold_projection=True,
        projection_accepted=bool(evaluation.projection.accepted),
        control_bound_residual=control,
        engineering_residual=engineering,
        trust_residual=trust,
        maximum_upper_residual=maximum,
        feasible=feasible,
    )


def audit_exact_candidate(
    problem: SurrogateNLP,
    case: SurrogateCase,
    normalized_controls: npt.ArrayLike,
    *,
    settings: SurrogateSolverSettings | None = None,
) -> FinalCandidateRecord:
    """Cold-reproject and retain feasibility, lower KKT, and status records."""

    settings = settings or SurrogateSolverSettings()
    normalized = _vector(normalized_controls, 7, "normalized_controls")
    evaluation = _exact_evaluation(problem, case, normalized)
    feasibility = _feasibility_record(evaluation, settings)
    stationarity = StationarityRecord(
        classification="stationarity_unresolved",
        resolved=False,
        stationary=False,
        lower_qp_kkt_passed=bool(evaluation.projection.accepted),
        upper_stationarity_residual=None,
        reason=(
            "A cold lower-QP KKT audit is available, but the rank, conditioning, "
            "strict-complementarity, active-set perturbation, total-sensitivity, "
            "and independently reconstructed upper-multiplier contract has not "
            "been established for this endpoint."
        ),
    )
    status = (
        "validated_feasible_stationarity_unresolved"
        if feasibility.feasible
        else "final_feasibility_failed"
    )
    theta = problem.assets.theta_lower + problem.assets.theta_span * normalized
    raw = np.asarray(
        problem.assets.model.predict(theta, case.influent), dtype=np.float64
    )
    return FinalCandidateRecord(
        normalized_controls=normalized,
        theta=theta,
        raw=raw,
        projected=evaluation.projection.state.copy(),
        displacement=evaluation.projection.displacement.copy(),
        objective=evaluation.objective,
        objective_components=evaluation.components,
        engineering_rows=evaluation.engineering,
        engineering_quantities=evaluation.quantities,
        trust_rows=evaluation.trust,
        trust_values=evaluation.trust_values,
        projection=evaluation.projection,
        feasibility=feasibility,
        stationarity=stationarity,
        status=status,
    )


def _local_poll_directions(*, include_simplex: bool) -> FloatArray:
    """Return a deterministic poll set in seven normalized dimensions."""

    axes = np.concatenate((np.eye(7), -np.eye(7)), axis=0)
    if not include_simplex:
        return axes
    pairwise = []
    for left in range(7):
        for right in range(left + 1, 7):
            for left_sign in (-1.0, 1.0):
                for right_sign in (-1.0, 1.0):
                    direction = np.zeros(7)
                    direction[left] = left_sign / np.sqrt(2.0)
                    direction[right] = right_sign / np.sqrt(2.0)
                    pairwise.append(direction)
    # The columns of the Helmert matrix are eight equiangular vertices of a
    # regular simplex in R^7.  Together with the signed coordinate basis they
    # and all signed pairwise diagonals improve finite directional coverage at
    # coupled boundaries.  This remains a declared finite poll, not an exact
    # characterization of the feasible tangent cone.
    simplex = linalg.helmert(8, full=False).T
    simplex /= np.linalg.norm(simplex, axis=1, keepdims=True)
    return np.concatenate((axes, np.asarray(pairwise), simplex), axis=0)


def certify_surrogate_local_convergence(
    assets: SurrogateNLPAssets,
    case: SurrogateCase,
    initial_candidate: FinalCandidateRecord,
    *,
    settings: SurrogateCertificationSettings | None = None,
    problem: SurrogateNLP | None = None,
    name: str = "v3_surrogate_local_certificate",
) -> SurrogateCertificationResult:
    """Polish and audit one selected exact-QP surrogate endpoint.

    A stable lower active set followed by a passing upper KKT audit is the
    strong first-order certificate.  Otherwise a deterministic feasible poll
    is completed at each declared normalized-control radius.  The latter is a
    finite-resolution local-convergence certificate only: it deliberately
    leaves classical stationarity unresolved at a nonsmooth or degenerate
    endpoint.
    """

    from .v3_active_set import (  # local import avoids a module cycle
        ActiveSetDerivativeError,
        ActiveSetRefinementSettings,
        ExactQPActiveSetRefiner,
    )

    configuration = settings or SurrogateCertificationSettings()
    started = perf_counter()
    solver_settings = SurrogateSolverSettings(
        final_upper_tolerance=configuration.feasibility_tolerance,
        maximum_wall_time=None,
    )
    if problem is None:
        problem = build_surrogate_nlp(
            assets,
            GAP_CONTINUATION[-1],
            settings=solver_settings,
            name=f"{name}_expressions",
            compile_solver=False,
        )
    elif problem.assets is not assets:
        raise ValueError("the certification problem uses different surrogate assets.")

    initial_controls = _vector(
        initial_candidate.normalized_controls, 7, "initial candidate controls"
    )
    initial_objective = float(initial_candidate.objective)
    evaluations = 0
    cold_qp_resolutions = 0
    accepted_improvements = 0
    poll_levels: list[dict[str, Any]] = []
    cache: dict[bytes, tuple[FinalCandidateRecord | None, str | None]] = {}

    def evaluate(controls: npt.ArrayLike) -> tuple[FinalCandidateRecord | None, str | None]:
        nonlocal evaluations, cold_qp_resolutions
        value = np.clip(_vector(controls, 7, "poll controls"), 0.0, 1.0)
        key = np.ascontiguousarray(value, dtype=np.float64).tobytes()
        if key in cache:
            return cache[key]
        if evaluations >= configuration.maximum_evaluations:
            return None, "certification evaluation budget exhausted"
        evaluations += 1
        cold_qp_resolutions += 1
        try:
            candidate = audit_exact_candidate(
                problem, case, value, settings=solver_settings,
            )
            if (
                not candidate.projection.accepted
                or not candidate.feasibility.finite
                or not np.isfinite(candidate.objective)
            ):
                raise SurrogateNLPError("poll trial failed the exact-QP finite audit")
            result = (candidate, None)
        except Exception as exc:
            result = (None, f"{type(exc).__name__}: {exc}")
        cache[key] = result
        return result

    current, initial_error = evaluate(initial_controls)
    if current is None or not current.feasibility.feasible:
        reason = initial_error or "initial endpoint failed upper feasibility"
        certificate = LocalConvergenceCertificate(
            protocol=LOCAL_CONVERGENCE_PROTOCOL,
            classification="initial_candidate_invalid",
            locally_converged=False,
            first_order_certified=False,
            stationarity_resolved=False,
            initial_objective=initial_objective,
            final_objective=float("nan") if current is None else float(current.objective),
            initial_normalized_controls=initial_controls,
            final_normalized_controls=(
                initial_controls if current is None else current.normalized_controls.copy()
            ),
            evaluations=evaluations,
            cold_qp_resolutions=cold_qp_resolutions,
            accepted_improvements=0,
            elapsed_seconds=perf_counter() - started,
            termination_reason=reason,
            poll_levels=(),
        )
        return SurrogateCertificationResult(None, certificate)

    # Independently reconstruct the active-set sensitivities and upper KKT
    # multipliers even when the retained search already reported stationarity.
    # This makes the new certificate self-contained rather than trusting an
    # earlier status bit.
    initial_kkt_error: str | None = None
    initial_lower_active_set: dict[str, Any] | None = None
    initial_upper_kkt: dict[str, Any] | None = None
    initial_refiner = ExactQPActiveSetRefiner(
        assets,
        case,
        problem=problem,
        settings=ActiveSetRefinementSettings(
            upper_acceptance_tolerance=configuration.feasibility_tolerance,
        ),
        name=f"{name}_initial_endpoint",
    )
    try:
        initial_trial = initial_refiner.evaluate(
            current.normalized_controls,
            force_cold=True,
            independent_final_replay=True,
        )
        initial_upper = initial_refiner.audit_upper_kkt(initial_trial)
        initial_lower_active_set = initial_trial.lower_active_set.as_dict()
        initial_upper_kkt = initial_upper.as_dict()
        initial_kkt_passed = bool(initial_upper.stationary)
    except ActiveSetDerivativeError as exc:
        initial_lower_active_set = None if exc.audit is None else exc.audit.as_dict()
        initial_kkt_error = str(exc)
        initial_kkt_passed = False
    except Exception as exc:
        initial_kkt_error = f"{type(exc).__name__}: {exc}"
        initial_kkt_passed = False
    finally:
        cold_qp_resolutions += initial_refiner.cold_qp_resolutions

    if initial_kkt_passed:
        reason = (
            "the independently reconstructed lower active set and upper KKT "
            "audit passed at the cold-replayed endpoint"
        )
        current = replace(
            current,
            stationarity=StationarityRecord(
                classification="first_order_kkt_stationary_feasible",
                resolved=True,
                stationary=True,
                lower_qp_kkt_passed=True,
                upper_stationarity_residual=(
                    None
                    if initial_upper_kkt is None
                    else float(initial_upper_kkt["stationarity_residual"])
                ),
                reason=reason,
            ),
            status="validated_stationary",
            lower_active_set=initial_lower_active_set,
            upper_kkt=initial_upper_kkt,
        )
        certificate = LocalConvergenceCertificate(
            protocol=LOCAL_CONVERGENCE_PROTOCOL,
            classification="exact_active_set_kkt",
            locally_converged=True,
            first_order_certified=True,
            stationarity_resolved=True,
            initial_objective=initial_objective,
            final_objective=float(current.objective),
            initial_normalized_controls=initial_controls,
            final_normalized_controls=current.normalized_controls.copy(),
            evaluations=evaluations,
            cold_qp_resolutions=cold_qp_resolutions,
            accepted_improvements=0,
            elapsed_seconds=perf_counter() - started,
            termination_reason=reason,
            poll_levels=(),
            lower_active_set=initial_lower_active_set,
            upper_kkt=initial_upper_kkt,
        )
        return SurrogateCertificationResult(current, certificate)

    termination_reason = "all declared poll levels completed without sufficient descent"
    all_levels_passed = False
    validation_round = 0
    while True:
        round_passed = True
        earlier_level_became_stale = False
        for level_index, radius in enumerate(configuration.poll_radii):
            directions = _local_poll_directions(
                include_simplex=level_index == len(configuration.poll_radii) - 1,
            )
            level_evaluations = 0
            level_failures: list[str] = []
            level_improvements = 0
            level_poll_improvements = 0
            acceleration_requests = 0
            acceleration_unique_evaluations = 0
            acceleration_accepted_improvements = 0
            acceleration_maximum_accepted_multiplier = 1.0
            acceleration_stops: dict[str, int] = {}
            last_feasible_directions: list[FloatArray] = []
            last_best_decrease = 0.0
            complete_no_descent_poll = False
            while True:
                center = current.normalized_controls.copy()
                center_objective = float(current.objective)
                decrease_tolerance = max(
                    configuration.absolute_decrease_tolerance,
                    configuration.relative_decrease_tolerance * abs(center_objective),
                )
                feasible_trials: list[FinalCandidateRecord] = []
                feasible_directions: list[FloatArray] = []
                sweep_complete = True
                seen: set[bytes] = set()
                for direction in directions:
                    trial_controls = np.clip(
                        center + float(radius) * direction, 0.0, 1.0,
                    )
                    displacement = trial_controls - center
                    norm = float(np.linalg.norm(displacement))
                    if norm <= np.finfo(float).eps:
                        continue
                    key = np.ascontiguousarray(
                        trial_controls, dtype=np.float64,
                    ).tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    trial, error = evaluate(trial_controls)
                    level_evaluations += 1
                    if error is not None or trial is None:
                        level_failures.append(
                            error or "unknown exact-QP evaluation failure"
                        )
                        sweep_complete = False
                        continue
                    if trial.feasibility.feasible:
                        feasible_trials.append(trial)
                        feasible_directions.append(displacement / norm)
                last_feasible_directions = feasible_directions
                if not sweep_complete:
                    if any("budget exhausted" in item for item in level_failures):
                        termination_reason = "certification evaluation budget exhausted"
                    elif level_failures:
                        termination_reason = "one or more required poll QPs failed"
                    break
                improving = [
                    trial for trial in feasible_trials
                    if float(trial.objective) < center_objective - decrease_tolerance
                ]
                if not improving:
                    last_best_decrease = max(
                        (
                            center_objective - float(trial.objective)
                            for trial in feasible_trials
                        ),
                        default=float("nan"),
                    )
                    complete_no_descent_poll = True
                    break
                best_objective = min(float(trial.objective) for trial in improving)
                tie = 1.0e-12 * max(1.0, abs(best_objective))
                selected = min(
                    (
                        trial for trial in improving
                        if float(trial.objective) <= best_objective + tie
                    ),
                    key=lambda item: tuple(item.normalized_controls.tolist()),
                )
                current = selected
                accepted_improvements += 1
                level_improvements += 1
                level_poll_improvements += 1

                # Accelerate repeated fixed-radius progress along the winning
                # poll ray.  Each geometrically expanded point is subjected to
                # the same cold exact-QP and upper-feasibility audit as a poll
                # point.  This is only a polishing acceleration: after the ray
                # search stops, the surrounding while loop always performs a
                # new complete declared-direction sweep at the accepted point.
                # Consequently, no expansion evaluation is used as evidence
                # for the final two-scale no-descent certificate.
                base_step = current.normalized_controls - center
                nonzero = np.abs(base_step) > np.finfo(float).eps
                maximum_multiplier = float("inf")
                if np.any(nonzero):
                    positive = base_step > np.finfo(float).eps
                    negative = base_step < -np.finfo(float).eps
                    bounds: list[FloatArray] = []
                    if np.any(positive):
                        bounds.append(
                            (1.0 - center[positive]) / base_step[positive]
                        )
                    if np.any(negative):
                        bounds.append(-center[negative] / base_step[negative])
                    if bounds:
                        maximum_multiplier = float(
                            np.min(np.concatenate(bounds))
                        )

                multiplier = float(configuration.acceleration_growth_factor)
                previous_controls = current.normalized_controls.copy()
                for _ in range(configuration.maximum_acceleration_probes):
                    bounded_multiplier = min(multiplier, maximum_multiplier)
                    if bounded_multiplier <= 1.0 + 1.0e-12:
                        acceleration_stops["box_boundary"] = (
                            acceleration_stops.get("box_boundary", 0) + 1
                        )
                        break
                    trial_controls = np.clip(
                        center + bounded_multiplier * base_step, 0.0, 1.0,
                    )
                    if np.array_equal(trial_controls, previous_controls):
                        acceleration_stops["duplicate_or_boundary"] = (
                            acceleration_stops.get("duplicate_or_boundary", 0) + 1
                        )
                        break
                    evaluations_before = evaluations
                    accelerated, acceleration_error = evaluate(trial_controls)
                    acceleration_requests += 1
                    acceleration_unique_evaluations += evaluations - evaluations_before
                    if acceleration_error is not None or accelerated is None:
                        stop = (
                            "evaluation_budget_exhausted"
                            if acceleration_error is not None
                            and "budget exhausted" in acceleration_error
                            else "exact_qp_failure"
                        )
                        acceleration_stops[stop] = acceleration_stops.get(stop, 0) + 1
                        break
                    if not accelerated.feasibility.feasible:
                        acceleration_stops["upper_infeasible"] = (
                            acceleration_stops.get("upper_infeasible", 0) + 1
                        )
                        break
                    accelerated_decrease_tolerance = max(
                        configuration.absolute_decrease_tolerance,
                        configuration.relative_decrease_tolerance
                        * abs(float(current.objective)),
                    )
                    if not (
                        float(accelerated.objective)
                        < float(current.objective) - accelerated_decrease_tolerance
                    ):
                        acceleration_stops["no_sufficient_descent"] = (
                            acceleration_stops.get("no_sufficient_descent", 0) + 1
                        )
                        break
                    current = accelerated
                    previous_controls = accelerated.normalized_controls.copy()
                    accepted_improvements += 1
                    level_improvements += 1
                    acceleration_accepted_improvements += 1
                    acceleration_maximum_accepted_multiplier = max(
                        acceleration_maximum_accepted_multiplier,
                        float(bounded_multiplier),
                    )
                    if bounded_multiplier >= maximum_multiplier - 1.0e-12:
                        acceleration_stops["box_boundary"] = (
                            acceleration_stops.get("box_boundary", 0) + 1
                        )
                        break
                    multiplier *= configuration.acceleration_growth_factor
                else:
                    acceleration_stops["expansion_limit"] = (
                        acceleration_stops.get("expansion_limit", 0) + 1
                    )

            feasible_rank = (
                int(np.linalg.matrix_rank(
                    np.vstack(last_feasible_directions),
                    tol=configuration.direction_rank_tolerance,
                ))
                if last_feasible_directions else 0
            )
            # At an active constrained endpoint, the locally feasible cone need
            # not linearly span R^7.  Rank is therefore diagnostic, not a valid
            # pass/fail condition.  Every distinct point in the frozen poll set
            # must complete, and at least one nonzero feasible displacement must
            # exist; the claim remains explicitly finite-direction/resolution.
            coverage_passed = bool(last_feasible_directions)
            level_passed = bool(
                complete_no_descent_poll
                and not level_failures
                and coverage_passed
            )
            poll_levels.append({
                "validation_round": validation_round,
                "radius": float(radius),
                "direction_set": (
                    "signed_coordinate_pairwise_plus_helmert_simplex"
                    if level_index == len(configuration.poll_radii) - 1
                    else "signed_coordinate"
                ),
                "direction_count": int(len(directions)),
                "evaluations": level_evaluations,
                "feasible_direction_count": len(last_feasible_directions),
                "feasible_direction_rank": feasible_rank,
                "required_direction_rank": None,
                "rank_is_diagnostic_only": True,
                "feasible_direction_coverage_passed": coverage_passed,
                "accepted_improvements": level_improvements,
                "poll_accepted_improvements": level_poll_improvements,
                "acceleration_evaluation_requests": acceleration_requests,
                "acceleration_unique_evaluations": acceleration_unique_evaluations,
                "acceleration_accepted_improvements": (
                    acceleration_accepted_improvements
                ),
                "acceleration_maximum_accepted_multiplier": (
                    acceleration_maximum_accepted_multiplier
                ),
                "acceleration_stops": dict(sorted(acceleration_stops.items())),
                "best_objective_decrease_in_final_sweep": last_best_decrease,
                "complete_no_descent_poll": complete_no_descent_poll,
                "passed": level_passed,
                "failures": level_failures,
            })
            if not level_passed:
                round_passed = False
                if complete_no_descent_poll and not level_failures:
                    termination_reason = (
                        "the declared poll contained no feasible nonzero displacement"
                    )
                break
            # A move at a finer radius changes the center at which every
            # already-passed coarser radius was tested. Revalidate from the
            # coarsest radius before issuing a certificate.
            if level_index > 0 and level_improvements:
                earlier_level_became_stale = True

        if not round_passed:
            break
        if earlier_level_became_stale:
            validation_round += 1
            continue
        all_levels_passed = True
        break

    if current is None:  # pragma: no cover - guarded by the initial audit
        raise AssertionError("local certification lost its feasible incumbent")

    # Reproduce the final poll point independently, bypassing the poll cache,
    # then retry the strong KKT audit because a short poll can move away from a
    # degenerate active set.
    cold_qp_resolutions += 1
    try:
        final_replay = audit_exact_candidate(
            problem,
            case,
            current.normalized_controls,
            settings=solver_settings,
        )
        replay_error = None
    except Exception as exc:
        final_replay = None
        replay_error = f"{type(exc).__name__}: {exc}"
    if final_replay is None or not final_replay.feasibility.feasible:
        all_levels_passed = False
        termination_reason = replay_error or "final endpoint replay failed feasibility"
    else:
        current = final_replay

    lower_active_set: dict[str, Any] | None = None
    upper_kkt: dict[str, Any] | None = None
    first_order_certified = False
    kkt_error: str | None = initial_kkt_error
    if final_replay is not None:
        active_settings = ActiveSetRefinementSettings(
            upper_acceptance_tolerance=configuration.feasibility_tolerance,
        )
        refiner = ExactQPActiveSetRefiner(
            assets,
            case,
            problem=problem,
            settings=active_settings,
            name=f"{name}_endpoint",
        )
        try:
            trial = refiner.evaluate(
                current.normalized_controls,
                force_cold=True,
                independent_final_replay=True,
            )
            upper = refiner.audit_upper_kkt(trial)
            lower_active_set = trial.lower_active_set.as_dict()
            upper_kkt = upper.as_dict()
            first_order_certified = bool(upper.stationary)
        except ActiveSetDerivativeError as exc:
            lower_active_set = None if exc.audit is None else exc.audit.as_dict()
            kkt_error = str(exc)
        except Exception as exc:
            kkt_error = f"{type(exc).__name__}: {exc}"
        finally:
            cold_qp_resolutions += refiner.cold_qp_resolutions

    locally_converged = bool(first_order_certified or all_levels_passed)
    if first_order_certified:
        classification = "exact_active_set_kkt"
        termination_reason = "the independently replayed endpoint passed the lower and upper KKT audits"
        stationarity = StationarityRecord(
            classification="first_order_kkt_stationary_feasible",
            resolved=True,
            stationary=True,
            lower_qp_kkt_passed=True,
            upper_stationarity_residual=(
                None if upper_kkt is None else float(upper_kkt["stationarity_residual"])
            ),
            reason=termination_reason,
        )
        status = "validated_stationary"
    elif all_levels_passed:
        classification = "finite_resolution_feasible_poll"
        stationarity = StationarityRecord(
            classification="poll_converged_stationarity_unresolved",
            resolved=False,
            stationary=False,
            lower_qp_kkt_passed=bool(current.projection.accepted),
            upper_stationarity_residual=None,
            reason=(
                "The exact-QP endpoint passed the complete two-scale feasible "
                "no-descent poll. This establishes finite-resolution local "
                "convergence, not classical stationarity."
            ),
        )
        status = "validated_feasible_poll_converged_stationarity_unresolved"
    else:
        classification = (
            "poll_budget_limited"
            if "budget" in termination_reason
            else "poll_inconclusive"
        )
        stationarity = StationarityRecord(
            classification=f"{classification}_stationarity_unresolved",
            resolved=False,
            stationary=False,
            lower_qp_kkt_passed=bool(current.projection.accepted),
            upper_stationarity_residual=None,
            reason=(
                f"{termination_reason}; endpoint KKT audit: "
                f"{kkt_error or 'did not pass'}"
            ),
        )
        status = f"validated_feasible_{classification}_stationarity_unresolved"
    current = replace(
        current,
        stationarity=stationarity,
        status=status,
        lower_active_set=lower_active_set,
        upper_kkt=upper_kkt,
    )
    certificate = LocalConvergenceCertificate(
        protocol=LOCAL_CONVERGENCE_PROTOCOL,
        classification=classification,
        locally_converged=locally_converged,
        first_order_certified=first_order_certified,
        stationarity_resolved=first_order_certified,
        initial_objective=initial_objective,
        final_objective=float(current.objective),
        initial_normalized_controls=initial_controls,
        final_normalized_controls=current.normalized_controls.copy(),
        evaluations=evaluations,
        cold_qp_resolutions=cold_qp_resolutions,
        accepted_improvements=accepted_improvements,
        elapsed_seconds=perf_counter() - started,
        termination_reason=termination_reason,
        poll_levels=tuple(poll_levels),
        lower_active_set=lower_active_set,
        upper_kkt=upper_kkt,
    )
    return SurrogateCertificationResult(current, certificate)


def _outer_refine(
    problem: SurrogateNLP,
    case: SurrogateCase,
    initial: FloatArray,
    settings: SurrogateSolverSettings,
) -> tuple[FinalCandidateRecord | None, OuterRefinementRecord]:
    """Run the seven-variable exact-QP active-set refinement.

    The import is local because :mod:`closed_loop.v3_active_set` builds on the
    public data structures in this module.  The implementation has no
    finite-difference fallback: an unstable lower active set is retained as
    an explicit stationarity-unresolved result.
    """

    if not settings.perform_outer_refinement:
        return None, OuterRefinementRecord(
            attempted=False,
            solver_success=False,
            status="disabled",
            iterations=0,
            evaluations=0,
            elapsed_seconds=0.0,
            initial_objective=None,
            final_objective=None,
        )

    from .v3_active_set import (  # local import avoids a module cycle
        ActiveSetRefinementSettings,
        ExactQPActiveSetRefiner,
    )

    active_settings = ActiveSetRefinementSettings(
        upper_acceptance_tolerance=settings.final_upper_tolerance,
        maximum_iterations=settings.outer_maximum_iterations,
        function_tolerance=settings.outer_function_tolerance,
    )
    exact = ExactQPActiveSetRefiner(
        problem.assets,
        case,
        problem=problem,
        settings=active_settings,
        name=f"{problem.name}_outer",
    ).refine(initial)
    initial_objective = (
        None if exact.initial is None else float(exact.initial.objective)
    )
    final_objective = None if exact.final is None else float(exact.final.objective)
    lower_audit = (
        exact.final.lower_active_set.as_dict()
        if exact.final is not None
        else (
            None
            if exact.derivative_audit is None
            else exact.derivative_audit.as_dict()
        )
    )
    upper_audit = None if exact.upper_kkt is None else exact.upper_kkt.as_dict()
    record = OuterRefinementRecord(
        attempted=True,
        solver_success=exact.solver_success,
        status=exact.solver_status,
        iterations=exact.iterations,
        evaluations=exact.distinct_trials,
        elapsed_seconds=exact.elapsed_seconds,
        initial_objective=initial_objective,
        final_objective=final_objective,
        cold_qp_resolutions=exact.cold_qp_resolutions,
        derivative_error=exact.derivative_error,
        lower_active_set=lower_audit,
        upper_kkt=upper_audit,
        projection_reproduction_residual=exact.state_reproduction_residual,
        projection_reproduction_passed=exact.state_reproduction_passed,
    )
    if exact.final is None or exact.upper_kkt is None:
        return None, record

    trial = exact.final
    parameter = case.parameter_vector(problem.assets)
    outputs = problem.upper_from_state_function(
        trial.normalized_controls,
        parameter,
        trial.projected_state,
    )
    evaluation = _ExactEvaluation(
        normalized=trial.normalized_controls.copy(),
        projection=trial.projection,
        objective=float(outputs[0]),
        engineering=_flat(outputs[1]),
        trust=_flat(outputs[2]),
        components=_flat(outputs[3]),
        quantities=_flat(outputs[4]),
        trust_values=_flat(outputs[5]),
    )
    feasibility = _feasibility_record(evaluation, settings)
    reproduction_passed = exact.state_reproduction_passed is True
    feasibility = replace(
        feasibility,
        feasible=bool(feasibility.feasible and reproduction_passed),
        projection_reproduction_residual=exact.state_reproduction_residual,
        projection_reproduction_passed=exact.state_reproduction_passed,
    )
    upper = exact.upper_kkt
    stationary = bool(upper.stationary and reproduction_passed)
    unresolved = not stationary
    stationarity = StationarityRecord(
        classification=(
            upper.classification
            if reproduction_passed
            else "projection_reproduction_failed"
        ),
        resolved=not unresolved,
        stationary=stationary,
        lower_qp_kkt_passed=bool(
            trial.projection.accepted and trial.lower_active_set.stable
        ),
        upper_stationarity_residual=float(upper.stationarity_residual),
        reason=(
            upper.reason
            if reproduction_passed
            else "independent cold-QP state reproduction failed"
        ),
    )
    status = (
        "projection_reproduction_failed"
        if not reproduction_passed
        else "validated_stationary"
        if feasibility.feasible and stationarity.stationary
        else (
            "validated_feasible_stationarity_unresolved"
            if feasibility.feasible
            else "final_feasibility_failed"
        )
    )
    return FinalCandidateRecord(
        normalized_controls=trial.normalized_controls.copy(),
        theta=trial.physical_controls.copy(),
        raw=trial.raw_state.copy(),
        projected=trial.projected_state.copy(),
        displacement=trial.projection.displacement.copy(),
        objective=evaluation.objective,
        objective_components=evaluation.components,
        engineering_rows=evaluation.engineering,
        engineering_quantities=evaluation.quantities,
        trust_rows=evaluation.trust,
        trust_values=evaluation.trust_values,
        projection=trial.projection,
        feasibility=feasibility,
        stationarity=stationarity,
        status=status,
        lower_active_set=trial.lower_active_set.as_dict(),
        upper_kkt=upper.as_dict(),
    ), record


def solve_surrogate_start(
    problems: Sequence[SurrogateNLP],
    case: SurrogateCase,
    normalized_start: npt.ArrayLike,
    *,
    start_index: int,
    settings: SurrogateSolverSettings | None = None,
) -> SurrogateStartResult:
    """Run one start through every gap stage and exact-QP refinement."""

    settings = settings or SurrogateSolverSettings()
    if tuple(problem.tau for problem in problems) != GAP_CONTINUATION:
        raise ValueError(f"problems must follow the exact continuation {GAP_CONTINUATION}.")
    assets = problems[0].assets
    if any(problem.assets is not assets for problem in problems):
        raise ValueError("all continuation stages must share one assets object.")
    resume_contract = surrogate_start_resume_contract(assets, case, settings)
    initial_normalized = _vector(normalized_start, 7, "normalized_start")
    stages: list[ContinuationStageRecord] = []
    try:
        primal, _ = initial_primal_from_projection(
            problems[0], case, initial_normalized
        )
    except Exception as exc:
        return SurrogateStartResult(
            start_index=start_index,
            initial_normalized_controls=initial_normalized,
            stages=(),
            outer_refinement=OuterRefinementRecord(
                False, False, "not_attempted", 0, 0, 0.0, None, None
            ),
            final=None,
            status="initial_projection_failed",
            error=f"{type(exc).__name__}: {exc}",
            resume_contract=resume_contract,
        )
    dual_warm_start: tuple[FloatArray, FloatArray] | None = None
    for problem in problems:
        solved = _solve_continuation_stage(
            problem, case, primal, settings, dual_warm_start
        )
        stage = solved.stage
        stages.append(stage)
        primal = stage.primal
        if not stage.feasible:
            return SurrogateStartResult(
                start_index=start_index,
                initial_normalized_controls=initial_normalized,
                stages=tuple(stages),
                outer_refinement=OuterRefinementRecord(
                    False, False, "not_attempted", 0, 0, 0.0, None, None
                ),
                final=None,
                status="continuation_failed",
                error=stage.error,
                resume_contract=resume_contract,
            )
        if solved.bound_multipliers is None or solved.constraint_multipliers is None:
            raise AssertionError("a feasible IPOPT stage did not return its dual solution.")
        dual_warm_start = (
            solved.bound_multipliers,
            solved.constraint_multipliers,
        )
    normalized = primal[problems[-1].theta_slice]
    # Only an independently projected feasible point is eligible for exact-QP
    # refinement.  The refinement itself cold-solves OSQP at every trial.
    initial_final = audit_exact_candidate(
        problems[-1], case, normalized, settings=settings
    )
    refinement_error: str | None = None
    if initial_final.feasibility.feasible:
        try:
            refined_final, refinement = _outer_refine(
                problems[-1], case, normalized, settings
            )
        except Exception as exc:
            refinement_error = f"{type(exc).__name__}: {exc}"
            refined_final = None
            refinement = OuterRefinementRecord(
                attempted=True,
                solver_success=False,
                status="unexpected_refinement_exception",
                iterations=0,
                evaluations=0,
                elapsed_seconds=0.0,
                initial_objective=(
                    initial_final.objective
                    if np.isfinite(initial_final.objective)
                    else None
                ),
                final_objective=None,
                derivative_error=refinement_error,
            )
        if refined_final is not None:
            # The active-set refiner's endpoint is already an additional,
            # uncached cold OSQP replay with an independent upper KKT audit.
            final = refined_final
        else:
            reason = refinement.derivative_error or refinement.status
            final = replace(
                initial_final,
                stationarity=StationarityRecord(
                    classification="stationarity_unresolved",
                    resolved=False,
                    stationary=False,
                    lower_qp_kkt_passed=bool(initial_final.projection.accepted),
                    upper_stationarity_residual=None,
                    reason=(
                        "Exact-QP active-set refinement was unavailable; the "
                        f"independently projected continuation incumbent is retained: {reason}"
                    ),
                ),
                status="validated_feasible_stationarity_unresolved",
                lower_active_set=refinement.lower_active_set,
                upper_kkt=refinement.upper_kkt,
            )
    else:
        refinement = OuterRefinementRecord(
            False,
            False,
            "initial_exact_qp_candidate_infeasible",
            0,
            1,
            0.0,
            initial_final.objective if np.isfinite(initial_final.objective) else None,
            None,
        )
        final = initial_final
    return SurrogateStartResult(
        start_index=start_index,
        initial_normalized_controls=initial_normalized,
        stages=tuple(stages),
        outer_refinement=refinement,
        final=final,
        status=final.status,
        error=refinement_error,
        resume_contract=resume_contract,
    )


def _validated_completed_starts(
    completed_starts: Mapping[int, SurrogateStartResult] | None,
    normalized_starts: FloatArray,
    expected_contract: str,
) -> dict[int, SurrogateStartResult]:
    if completed_starts is None:
        return {}
    if not isinstance(completed_starts, Mapping):
        raise TypeError("completed_starts must be an index-to-SurrogateStartResult mapping.")
    validated: dict[int, SurrogateStartResult] = {}
    for raw_index, result in completed_starts.items():
        if isinstance(raw_index, bool) or not isinstance(raw_index, (int, np.integer)):
            raise TypeError("every completed-start key must be an integer index.")
        index = int(raw_index)
        if not 0 <= index < normalized_starts.shape[0]:
            raise ValueError(f"completed start index {index} is outside the requested schedule.")
        if not isinstance(result, SurrogateStartResult):
            raise TypeError("every completed-start value must be a SurrogateStartResult.")
        if result.start_index != index:
            raise ValueError(
                f"completed start key {index} does not match result index {result.start_index}."
            )
        initial = np.asarray(result.initial_normalized_controls, dtype=np.float64)
        if initial.shape != (7,) or not np.array_equal(initial, normalized_starts[index]):
            raise ValueError(
                f"completed start {index} does not match its declared normalized control."
            )
        if result.resume_contract != expected_contract:
            raise ValueError(
                f"completed start {index} has a stale case/settings resume contract."
            )
        stage_taus = tuple(float(stage.tau) for stage in result.stages)
        if stage_taus != GAP_CONTINUATION[: len(stage_taus)]:
            raise ValueError(
                f"completed start {index} has a non-prefix continuation schedule."
            )
        if result.final is not None and stage_taus != GAP_CONTINUATION:
            raise ValueError(
                f"completed start {index} reports a final candidate before all stages."
            )
        validated[index] = result
    return validated


def solve_surrogate_multistart(
    assets: SurrogateNLPAssets,
    case: SurrogateCase,
    *,
    settings: SurrogateSolverSettings | None = None,
    starts: npt.ArrayLike | None = None,
    completed_starts: Mapping[int, SurrogateStartResult] | None = None,
    name: str = "v3_surrogate",
    progress_callback: Callable[[SurrogateStartResult], None] | None = None,
    allow_reduced_starts: bool = False,
) -> SurrogateMultistartResult:
    """Run the deterministic starts through the exact v3 gap schedule.

    Production keeps the declared nine-start contract.  A caller must opt in
    explicitly to fewer starts for an article-ineligible smoke test.
    """

    settings = settings or SurrogateSolverSettings()
    normalized_starts = (
        ordered_normalized_starts()
        if starts is None
        else np.asarray(starts, dtype=np.float64)
    )
    if (
        normalized_starts.ndim != 2
        or normalized_starts.shape[1:] != (7,)
        or normalized_starts.shape[0] < 1
        or (not allow_reduced_starts and normalized_starts.shape[0] != 9)
        or not np.all(np.isfinite(normalized_starts))
    ):
        requirement = "one or more" if allow_reduced_starts else "exactly nine"
        raise ValueError(f"starts must contain {requirement} finite seven-control rows.")
    if np.any(normalized_starts < 0.0) or np.any(normalized_starts > 1.0):
        raise ValueError("all normalized starts must lie in [0, 1].")
    expected_contract = surrogate_start_resume_contract(assets, case, settings)
    resumed = _validated_completed_starts(
        completed_starts,
        normalized_starts,
        expected_contract,
    )
    problems: tuple[SurrogateNLP, ...] = ()
    if len(resumed) < normalized_starts.shape[0]:
        problems = tuple(
            build_surrogate_nlp(
                assets,
                tau,
                settings=settings,
                name=f"{name}_{case.case_id}_{index}",
                compile_solver=True,
            )
            for index, tau in enumerate(GAP_CONTINUATION)
        )
    completed: dict[int, SurrogateStartResult] = dict(resumed)
    for index, normalized in enumerate(normalized_starts):
        if index in completed:
            continue
        result = solve_surrogate_start(
            problems,
            case,
            normalized,
            start_index=index,
            settings=settings,
        )
        if result.resume_contract != expected_contract:
            raise AssertionError("a newly solved start returned an inconsistent resume contract.")
        completed[index] = result
        if progress_callback is not None:
            progress_callback(result)
    results = tuple(completed[index] for index in range(normalized_starts.shape[0]))
    feasible = [result for result in results if result.feasible and result.final is not None]
    if not feasible:
        return SurrogateMultistartResult(results, None, "no_validated_feasible_start")
    # The manuscript selects among stationary feasible points whenever any
    # exist.  Feasible stationarity-unresolved incumbents remain reportable,
    # but cannot displace a genuinely stationary candidate.
    stationary = [result for result in feasible if result.stationary]
    eligible = stationary if stationary else feasible
    minimum = min(result.final.objective for result in eligible if result.final is not None)
    tolerance = 1.0e-10 * max(1.0, abs(minimum))
    tied = [
        result
        for result in eligible
        if result.final is not None and result.final.objective <= minimum + tolerance
    ]
    selected = min(
        tied,
        key=lambda result: (
            *result.final.normalized_controls.tolist(),  # type: ignore[union-attr]
            result.start_index,
        ),
    )
    return SurrogateMultistartResult(
        results,
        selected,
        "selected_stationary" if stationary else "selected_stationarity_unresolved",
    )


@dataclass(frozen=True)
class _DerivativeFreeEvaluation:
    normalized_controls: FloatArray
    candidate: FinalCandidateRecord | None
    error: str | None


def _derivative_free_exact_qp_refine(
    problem: SurrogateNLP,
    case: SurrogateCase,
    initial: FloatArray,
    settings: SurrogateSolverSettings,
    active_record: OuterRefinementRecord,
) -> tuple[FinalCandidateRecord | None, OuterRefinementRecord]:
    """Continue an unavailable active-set derivative with exact-QP COBYQA.

    Bounds are unrelaxable in COBYQA and every distinct in-box point is
    evaluated by a newly initialized projection QP. Objective and nonlinear
    constraint callbacks share only the resulting value cache. The selected
    visited point is then cold-replayed independently, and analytical
    active-set/upper-KKT auditing is attempted again only at that endpoint.
    """

    from .v3_active_set import (  # local import avoids a module cycle
        ActiveSetDerivativeError,
        ActiveSetRefinementError,
        ActiveSetRefinementSettings,
        ExactQPActiveSetRefiner,
    )

    started = perf_counter()
    normalized_initial = _vector(initial, 7, "normalized_start")
    upper_count = len(problem.engineering_names) + len(problem.trust_names)
    cache: dict[bytes, _DerivativeFreeEvaluation] = {}
    cold_qp_resolutions = 0
    evaluation_errors: list[str] = []

    def evaluate(value: npt.ArrayLike) -> _DerivativeFreeEvaluation:
        nonlocal cold_qp_resolutions
        normalized = _vector(value, 7, "derivative-free normalized controls")
        # COBYQA treats finite bounds as unrelaxable, but its scaled internal
        # coordinates can return roundoff-sized excursions at a boundary.
        # Canonicalize those values to the physical box before cache lookup.
        normalized = np.clip(normalized, 0.0, 1.0)
        key = np.ascontiguousarray(normalized, dtype=np.float64).tobytes()
        if key in cache:
            return cache[key]
        if (
            settings.maximum_wall_time is not None
            and perf_counter() - started >= settings.maximum_wall_time
        ):
            raise TimeoutError("derivative-free exact-QP wall-time limit reached")
        cold_qp_resolutions += 1
        try:
            candidate = audit_exact_candidate(
                problem,
                case,
                normalized,
                settings=settings,
            )
            finite = bool(
                candidate.feasibility.finite
                and np.isfinite(candidate.objective)
                and np.all(np.isfinite(candidate.engineering_rows))
                and np.all(np.isfinite(candidate.trust_rows))
            )
            if not finite:
                raise SurrogateNLPError(
                    "exact-QP derivative-free evaluation was non-finite"
                )
            item = _DerivativeFreeEvaluation(normalized, candidate, None)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            item = _DerivativeFreeEvaluation(normalized, None, message)
            evaluation_errors.append(message)
        cache[key] = item
        return item

    def usable(item: _DerivativeFreeEvaluation) -> bool:
        return bool(
            item.candidate is not None
            and item.candidate.projection.accepted
            and item.candidate.feasibility.finite
        )

    def objective(value: npt.ArrayLike) -> float:
        item = evaluate(value)
        if usable(item):
            assert item.candidate is not None
            return float(item.candidate.objective)
        distance = float(
            np.linalg.norm(item.normalized_controls - normalized_initial) ** 2
        )
        return 1.0e6 + distance

    def upper_constraints(value: npt.ArrayLike) -> FloatArray:
        item = evaluate(value)
        if usable(item):
            assert item.candidate is not None
            rows = np.concatenate(
                (item.candidate.engineering_rows, item.candidate.trust_rows)
            )
            if rows.shape != (upper_count,):
                raise AssertionError(
                    "derivative-free upper constraints have inconsistent dimensions"
                )
            return rows
        return np.ones(upper_count, dtype=np.float64)

    solver_success = False
    solver_status = "not_started"
    iterations = 0
    proposed = normalized_initial.copy()
    try:
        optimized = minimize(
            objective,
            normalized_initial,
            method="COBYQA",
            bounds=Bounds(np.zeros(7), np.ones(7)),
            constraints=NonlinearConstraint(
                upper_constraints,
                np.full(upper_count, -np.inf),
                np.zeros(upper_count),
            ),
            options={
                "maxiter": settings.outer_maximum_iterations,
                "maxfev": settings.outer_maximum_iterations,
                "initial_tr_radius": 0.25,
                # Resolving controls more tightly than the independently
                # audited upper-feasibility scale cannot strengthen the
                # scientific claim and needlessly exhausts value evaluations.
                "final_tr_radius": max(
                    settings.outer_function_tolerance,
                    settings.final_upper_tolerance,
                ),
                "feasibility_tol": settings.final_upper_tolerance,
                # Controls are already normalized. Leaving COBYQA scaling
                # disabled also keeps its objective and nonlinear-constraint
                # callbacks in the identical coordinate system.
                "scale": False,
                "disp": False,
            },
        )
        proposed = np.clip(
            _vector(optimized.x, 7, "derivative-free optimizer endpoint"),
            0.0,
            1.0,
        )
        solver_success = bool(optimized.success)
        solver_status = str(optimized.message)
        iterations = int(optimized.nit)
    except Exception as exc:
        solver_status = f"{type(exc).__name__}: {exc}"

    # Make the solver endpoint visible even if COBYQA did not request both
    # callbacks there. A failed endpoint does not displace a feasible visited
    # point.
    try:
        evaluate(proposed)
    except Exception as exc:
        evaluation_errors.append(f"{type(exc).__name__}: {exc}")

    candidates = [
        item.candidate
        for item in cache.values()
        if usable(item) and item.candidate is not None
    ]
    feasible = [item for item in candidates if item.feasibility.feasible]
    pool = feasible or candidates
    selected_cached: FinalCandidateRecord | None = None
    if pool:
        if feasible:
            best_objective = min(item.objective for item in pool)
            tie = 1.0e-10 * max(1.0, abs(best_objective))
            selected_cached = min(
                (item for item in pool if item.objective <= best_objective + tie),
                key=lambda item: tuple(item.normalized_controls.tolist()),
            )
        else:
            selected_cached = min(
                pool,
                key=lambda item: (
                    item.feasibility.maximum_upper_residual,
                    item.objective,
                    *item.normalized_controls.tolist(),
                ),
            )

    final: FinalCandidateRecord | None = None
    reproduction_residual: float | None = None
    reproduction_passed: bool | None = None
    endpoint_lower = active_record.lower_active_set
    endpoint_upper: dict[str, Any] | None = None
    endpoint_stationary = False
    endpoint_stationarity_residual: float | None = None
    endpoint_derivative_error: str | None = None

    if selected_cached is not None:
        cold_qp_resolutions += 1
        try:
            replay = audit_exact_candidate(
                problem,
                case,
                selected_cached.normalized_controls,
                settings=settings,
            )
            reproduction_residual = float(
                np.linalg.norm(
                    (replay.projected - selected_cached.projected)
                    / problem.assets.model.response_scale,
                    ord=np.inf,
                )
            )
            reproduction_passed = bool(
                replay.projection.accepted
                and np.isfinite(reproduction_residual)
                and reproduction_residual <= 1.0e-8
            )
            final = replay
        except Exception as exc:
            endpoint_derivative_error = f"endpoint replay: {type(exc).__name__}: {exc}"

    # The derivative-free optimizer itself cannot establish stationarity.
    # Attempt the full analytical audit at the chosen endpoint; failure is a
    # recorded scientific result, not an execution stop.
    if final is not None and reproduction_passed is True:
        active_settings = ActiveSetRefinementSettings(
            upper_acceptance_tolerance=settings.final_upper_tolerance,
            maximum_iterations=settings.outer_maximum_iterations,
            function_tolerance=settings.outer_function_tolerance,
        )
        endpoint_refiner = ExactQPActiveSetRefiner(
            problem.assets,
            case,
            problem=problem,
            settings=active_settings,
            name=f"{problem.name}_derivative_free_endpoint",
        )
        try:
            endpoint_trial = endpoint_refiner.evaluate(
                final.normalized_controls,
                force_cold=True,
                independent_final_replay=True,
            )
            upper_audit = endpoint_refiner.audit_upper_kkt(endpoint_trial)
            endpoint_lower = endpoint_trial.lower_active_set.as_dict()
            endpoint_upper = upper_audit.as_dict()
            endpoint_stationary = bool(upper_audit.stationary)
            endpoint_stationarity_residual = float(
                upper_audit.stationarity_residual
            )
        except ActiveSetDerivativeError as exc:
            endpoint_lower = None if exc.audit is None else exc.audit.as_dict()
            endpoint_derivative_error = f"endpoint active-set audit: {exc}"
        except ActiveSetRefinementError as exc:
            endpoint_derivative_error = f"endpoint active-set audit: {exc}"
        except Exception as exc:
            endpoint_derivative_error = (
                f"endpoint active-set audit: {type(exc).__name__}: {exc}"
            )
        finally:
            cold_qp_resolutions += endpoint_refiner.cold_qp_resolutions

    derivative_messages = [
        item
        for item in (active_record.derivative_error, endpoint_derivative_error)
        if item
    ]
    derivative_error = "; ".join(dict.fromkeys(derivative_messages)) or None
    if final is not None:
        final_feasibility = replace(
            final.feasibility,
            feasible=bool(final.feasibility.feasible and reproduction_passed is True),
            projection_reproduction_residual=reproduction_residual,
            projection_reproduction_passed=reproduction_passed,
        )
        stationary = bool(
            final_feasibility.feasible
            and endpoint_stationary
            and endpoint_upper is not None
        )
        if stationary:
            stationarity = StationarityRecord(
                classification="first_order_kkt_stationary_feasible",
                resolved=True,
                stationary=True,
                lower_qp_kkt_passed=True,
                upper_stationarity_residual=endpoint_stationarity_residual,
                reason=(
                    "the derivative-free local endpoint passed independent "
                    "lower active-set and upper KKT audits"
                ),
            )
            final_status = "validated_stationary"
        else:
            normalized_solver_status = solver_status.lower()
            budget_limited = bool(
                not solver_success
                and "maximum number" in normalized_solver_status
                and (
                    "evaluation" in normalized_solver_status
                    or "iteration" in normalized_solver_status
                )
            )
            nonconverged_label = (
                "budget_limited_derivative_free_feasible_incumbent_"
                "stationarity_unresolved"
                if budget_limited
                else "nonconverged_derivative_free_feasible_incumbent_"
                "stationarity_unresolved"
            )
            stationarity = StationarityRecord(
                classification=(
                    nonconverged_label
                    if final_feasibility.feasible and not solver_success
                    else "derivative_free_local_candidate_stationarity_unresolved"
                    if final_feasibility.feasible
                    else "derivative_free_local_candidate_feasibility_failed"
                ),
                resolved=False,
                stationary=False,
                lower_qp_kkt_passed=bool(final.projection.accepted),
                upper_stationarity_residual=endpoint_stationarity_residual,
                reason=(
                    (
                        "COBYQA reached its iteration/evaluation budget; this is "
                        "the best feasible exact-QP point visited, not an "
                        "established local optimum. "
                        if budget_limited
                        else "COBYQA did not report convergence; this is the best "
                        "feasible exact-QP point visited, not an established "
                        "local optimum. "
                        if not solver_success
                        else "COBYQA converged to a derivative-free exact-QP local candidate. "
                    )
                    + "Stationarity remains unresolved because the independent "
                    + "active-set/KKT audit did not pass: "
                    + str(derivative_error or solver_status)
                ),
            )
            final_status = (
                "validated_feasible_budget_limited_derivative_free_incumbent_"
                "stationarity_unresolved"
                if final_feasibility.feasible and budget_limited
                else "validated_feasible_nonconverged_derivative_free_incumbent_"
                "stationarity_unresolved"
                if final_feasibility.feasible and not solver_success
                else "validated_feasible_derivative_free_local_candidate_"
                "stationarity_unresolved"
                if final_feasibility.feasible
                else (
                    "projection_reproduction_failed"
                    if reproduction_passed is False
                    else "final_feasibility_failed"
                )
            )
        final = replace(
            final,
            feasibility=final_feasibility,
            stationarity=stationarity,
            status=final_status,
            lower_active_set=endpoint_lower,
            upper_kkt=endpoint_upper,
        )

    initial_item = cache.get(
        np.ascontiguousarray(normalized_initial, dtype=np.float64).tobytes()
    )
    initial_objective = (
        None
        if initial_item is None or initial_item.candidate is None
        else float(initial_item.candidate.objective)
    )
    fallback_status = solver_status
    if evaluation_errors:
        fallback_status = (
            f"{fallback_status}; failed exact-QP evaluations={len(evaluation_errors)}"
        )
    record = OuterRefinementRecord(
        attempted=True,
        solver_success=solver_success,
        status=(
            "derivative_free_local_candidate"
            if final is not None and solver_success
            else "derivative_free_budget_limited_candidate"
            if final is not None
            and "maximum number" in solver_status.lower()
            and (
                "evaluation" in solver_status.lower()
                or "iteration" in solver_status.lower()
            )
            else "derivative_free_nonconverged_candidate"
            if final is not None
            else "derivative_free_local_optimization_failed"
        ),
        iterations=iterations,
        evaluations=len(cache),
        elapsed_seconds=active_record.elapsed_seconds + (perf_counter() - started),
        initial_objective=initial_objective,
        final_objective=None if final is None else float(final.objective),
        cold_qp_resolutions=(
            active_record.cold_qp_resolutions + cold_qp_resolutions
        ),
        derivative_error=derivative_error,
        lower_active_set=endpoint_lower,
        upper_kkt=endpoint_upper,
        projection_reproduction_residual=reproduction_residual,
        projection_reproduction_passed=reproduction_passed,
        method="exact_qp_derivative_free_cobyqa",
        fallback_used=True,
        fallback_method="COBYQA",
        fallback_solver_success=solver_success,
        fallback_status=fallback_status,
        fallback_iterations=iterations,
        fallback_evaluations=len(cache),
    )
    return final, record


def solve_surrogate_exact_qp_local(
    assets: SurrogateNLPAssets,
    case: SurrogateCase,
    *,
    settings: SurrogateSolverSettings | None = None,
    problem: SurrogateNLP | None = None,
    completed_result: SurrogateStartResult | None = None,
    name: str = "v3_surrogate_exact_qp_local",
    progress_callback: Callable[[SurrogateStartResult], None] | None = None,
) -> SurrogateMultistartResult:
    """Solve one case by one seven-variable exact-QP local optimization.

    The sole initial point is the deterministic center of the normalized
    operating box. This route builds only the expression graph needed by the
    exact-QP active-set evaluator (``compile_solver=False``): it never
    constructs an embedded-KKT IPOPT solver and has no continuation stages.

    Numerical projection, active-set, derivative, or endpoint failures are
    returned as one structured, stationarity-unresolved start result. They
    therefore remain visible to scientific reporting without aborting other
    influent cases. Invalid inputs and stale checkpoints still raise.
    """

    settings = settings or SurrogateSolverSettings()
    if not settings.perform_outer_refinement:
        raise ValueError(
            "the exact-QP single-start protocol requires perform_outer_refinement=True"
        )
    center = np.asarray(EXACT_QP_CENTER_START, dtype=np.float64)
    resume_contract = surrogate_exact_qp_resume_contract(assets, case, settings)

    if completed_result is not None:
        if not isinstance(completed_result, SurrogateStartResult):
            raise TypeError("completed_result must be a SurrogateStartResult.")
        if completed_result.start_index != 0:
            raise ValueError("the completed exact-QP result must have start_index zero.")
        if not np.array_equal(completed_result.initial_normalized_controls, center):
            raise ValueError("the completed exact-QP result does not use the center start.")
        if completed_result.stages:
            raise ValueError(
                "an exact-QP single-start checkpoint cannot contain continuation stages."
            )
        if completed_result.protocol != EXACT_QP_SINGLE_START_PROTOCOL:
            raise ValueError("the completed result uses a different surrogate protocol.")
        if completed_result.resume_contract != resume_contract:
            raise ValueError("the completed result has a stale exact-QP resume contract.")
        result = completed_result
    else:
        if problem is None:
            problem = build_surrogate_nlp(
                assets,
                GAP_CONTINUATION[-1],
                settings=settings,
                name=f"{name}_expressions",
                compile_solver=False,
            )
        else:
            if problem.assets is not assets:
                raise ValueError("the reusable exact-QP problem uses different assets.")
            if problem.tau != GAP_CONTINUATION[-1]:
                raise ValueError("the reusable exact-QP problem must use the final gap value.")
            if problem.solver is not None:
                raise ValueError(
                    "the reusable exact-QP expression problem must not contain an IPOPT solver."
                )
        refinement_error: str | None = None
        try:
            final, refinement = _outer_refine(problem, case, center, settings)
        except Exception as exc:
            # One bad case must not terminate the remaining cases. The
            # independent center audit below still preserves any finite,
            # physically projected incumbent for reporting.
            refinement_error = f"{type(exc).__name__}: {exc}"
            final = None
            refinement = OuterRefinementRecord(
                attempted=True,
                solver_success=False,
                status="unexpected_exact_qp_exception",
                iterations=0,
                evaluations=0,
                elapsed_seconds=0.0,
                initial_objective=None,
                final_objective=None,
                derivative_error=refinement_error,
            )

        if final is None:
            try:
                final, refinement = _derivative_free_exact_qp_refine(
                    problem,
                    case,
                    center,
                    settings,
                    refinement,
                )
            except Exception as exc:
                fallback_error = f"{type(exc).__name__}: {exc}"
                refinement_error = (
                    fallback_error
                    if refinement_error is None
                    else f"{refinement_error}; derivative-free fallback: {fallback_error}"
                )
                refinement = replace(
                    refinement,
                    status="unexpected_derivative_free_exception",
                    derivative_error=(
                        refinement.derivative_error or refinement_error
                    ),
                    method="exact_qp_derivative_free_cobyqa",
                    fallback_used=True,
                    fallback_method="COBYQA",
                    fallback_solver_success=False,
                    fallback_status=fallback_error,
                )

        if final is None:
            try:
                incumbent = audit_exact_candidate(
                    problem,
                    case,
                    center,
                    settings=settings,
                )
            except Exception as exc:
                audit_error = f"{type(exc).__name__}: {exc}"
                refinement_error = (
                    audit_error
                    if refinement_error is None
                    else f"{refinement_error}; center audit: {audit_error}"
                )
            else:
                reason = refinement.derivative_error or refinement.status
                final = replace(
                    incumbent,
                    stationarity=StationarityRecord(
                        classification="stationarity_unresolved",
                        resolved=False,
                        stationary=False,
                        lower_qp_kkt_passed=bool(incumbent.projection.accepted),
                        upper_stationarity_residual=None,
                        reason=(
                            "The primary seven-variable exact-QP local optimizer "
                            "did not establish endpoint stationarity; the "
                            "independently cold-projected center incumbent is "
                            f"retained: {reason}"
                        ),
                    ),
                    status=(
                        "validated_feasible_stationarity_unresolved"
                        if incumbent.feasibility.feasible
                        else incumbent.status
                    ),
                    lower_active_set=refinement.lower_active_set,
                    upper_kkt=refinement.upper_kkt,
                )
                if refinement_error is None:
                    refinement_error = refinement.derivative_error

        result = SurrogateStartResult(
            start_index=0,
            initial_normalized_controls=center.copy(),
            stages=(),
            outer_refinement=refinement,
            final=final,
            status=(
                final.status
                if final is not None
                else "exact_qp_local_optimization_unresolved"
            ),
            error=refinement_error,
            resume_contract=resume_contract,
            protocol=EXACT_QP_SINGLE_START_PROTOCOL,
        )
        if progress_callback is not None:
            progress_callback(result)

    selected = result if result.feasible else None
    return SurrogateMultistartResult(
        starts=(result,),
        selected=selected,
        status=(
            "selected_stationary"
            if result.stationary
            else (
                "selected_stationarity_unresolved"
                if result.feasible
                else "no_validated_feasible_start"
            )
        ),
        protocol=EXACT_QP_SINGLE_START_PROTOCOL,
    )


IMPLEMENTATION_LIMITATIONS = (
    "Exact-QP analytical derivatives are available only when the lower active "
    "set passes the LICQ, conditioning, strict-complementarity, and local "
    "perturbation audits. When they do not, deterministic value-only COBYQA "
    "continues with a cold exact projection QP at every distinct trial. Its "
    "independently replayed endpoint remains stationarity-unresolved unless "
    "the endpoint active-set and upper-KKT audits pass. No finite-difference "
    "derivative is substituted for a failed analytical audit."
)


__all__ = [
    "DEFAULT_OBJECTIVE_WEIGHTS",
    "EXACT_QP_CENTER_START",
    "EXACT_QP_SINGLE_START_PROTOCOL",
    "LOCAL_CONVERGENCE_PROTOCOL",
    "GAP_CONTINUATION",
    "IMPLEMENTATION_LIMITATIONS",
    "LEGACY_SURROGATE_PROTOCOL",
    "START_SEED",
    "EngineeringLimits",
    "FeasibilityRecord",
    "FinalCandidateRecord",
    "NamedTrustRows",
    "OuterRefinementRecord",
    "LocalConvergenceCertificate",
    "StationarityRecord",
    "SurrogateCase",
    "SurrogateCertificationResult",
    "SurrogateCertificationSettings",
    "SurrogateMultistartResult",
    "SurrogateNLP",
    "SurrogateNLPAssets",
    "SurrogateNLPError",
    "SurrogateSolverSettings",
    "SurrogateStartResult",
    "TrustDiagnosticCallbacks",
    "TrustThresholds",
    "audit_exact_candidate",
    "build_surrogate_assets",
    "build_surrogate_nlp",
    "cold_reproject",
    "certify_surrogate_local_convergence",
    "evaluate_surrogate_problem",
    "initial_primal_from_projection",
    "ordered_normalized_starts",
    "regularized_leverage_contract",
    "solve_surrogate_exact_qp_local",
    "solve_surrogate_multistart",
    "solve_surrogate_start",
    "surrogate_exact_qp_resume_contract",
    "surrogate_start_resume_contract",
    "symbolic_network_operators",
    "symbolic_quadratic_prediction",
    "unpack_primal",
]
