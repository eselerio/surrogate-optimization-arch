"""Manuscript-v3 projected-surrogate operational optimization.

This module implements the route in ``article/wip_v3`` without hiding the
lower physical projection behind a black-box callback.  The continuation NLP
has exactly the variables displayed by the manuscript,

``(theta, u, lambda_eq, lambda_ineq)``,

where ``theta`` is represented in the normalized seven-dimensional operating
box.  The raw quadratic predictor, the network matrices, and the primal-dual
gap are CasADi expressions.  Reportable endpoints are never accepted from the
single-level relaxation alone: a newly constructed OSQP problem resolves the
original projection and the independently reconstructed lower-QP KKT
residuals supplied by :mod:`closed_loop.projection` are retained.

The optional exact-QP outer refinement also resolves a cold projection at
every trial control.  This module deliberately labels upper stationarity as
unresolved.  A solver success flag (or SLSQP's approximate multipliers) is not
the active-set sensitivity and independently reconstructed upper KKT audit
required by the manuscript.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import casadi as ca
import numpy as np
import numpy.typing as npt
from scipy import linalg
from scipy.optimize import minimize

from .manuscript_v3 import DECISION_LOWER, DECISION_UPPER
from .model import COMPOSITE_MATRIX, INVARIANT_MATRIX, TSS_VECTOR
from .projection import (
    NetworkLayout,
    NetworkRowScales,
    PhysicalProjector,
    ProjectionResult,
    QuadraticSurrogate,
    SurrogateValidationError,
    build_network_operators,
    fit_network_row_scales,
)


FloatArray = npt.NDArray[np.float64]
TrustRowCallback = Callable[[Any, Any, Any, Any], Any]

GAP_CONTINUATION: tuple[float, ...] = (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)
START_SEED = 271_828
DEFAULT_OBJECTIVE_WEIGHTS = np.asarray([0.50, 0.15, 0.20, 0.05, 0.05, 0.05])


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


def _maximum_positive(values: npt.ArrayLike) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.max(np.maximum(array, 0.0), initial=0.0))


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
    flux_rms: float | None = None

    def __post_init__(self) -> None:
        required = (self.correction_rms, self.regularized_leverage)
        if not all(np.isfinite(value) and value >= 0.0 for value in required):
            raise ValueError("native trust thresholds must be finite and nonnegative.")
        for value in (self.split_rms, self.reactor_rms, self.flux_rms):
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
    """Optional v3 split, smooth-reactor, and Clarifier-flux residual rows."""

    split_rows: TrustRowCallback | None = None
    reactor_rows: TrustRowCallback | None = None
    flux_rows: TrustRowCallback | None = None
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
            + 2
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
            (self.trust_callbacks.flux_rows, self.trust_thresholds.flux_rms, "flux"),
        )
        for callback, threshold, name in pairs:
            if (callback is None) != (threshold is None):
                raise ValueError(
                    f"{name} trust rows and their threshold must be supplied together."
                )
        names = [item.name for item in self.trust_callbacks.additional]
        if len(set(names)) != len(names) or any(
            name in {"correction", "leverage", "split", "reactor", "flux"}
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
    flux_rms_threshold: float | None = None,
    engineering: EngineeringLimits | None = None,
) -> SurrogateNLPAssets:
    """Fit the projection scales, leverage contract, and quality normalization.

    Optional split/reactor/flux callbacks must return *already scaled* residual
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
    row_scales = fit_network_row_scales(
        targets,
        influents,
        internal_recycle=decisions[:, 4],
        return_recycle=decisions[:, 5],
        waste_fraction=decisions[:, 6],
        invariant_operator=invariant_operator,
        tss_weights=tss_weights,
        layout=layout,
        minimum_scale=1.0,
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
        flux_rms=flux_rms_threshold,
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
        trust_callbacks=trust_callbacks or TrustDiagnosticCallbacks(),
        engineering=engineering or EngineeringLimits(),
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

    first_layer = layout.layer_slice.start
    last_layer = layout.layer_slice.stop - 1
    equality[row, first_layer] = effluent
    equality[row, layout.overflow_flow_slice] = -ca.DM(assets.tss_weights).T
    row += 1
    equality[row, last_layer] = underflow
    equality[row, layout.underflow_flow_slice] = -ca.DM(assets.tss_weights).T
    row += 1
    if row != equality_count:
        raise AssertionError("symbolic equality-row construction is inconsistent.")

    inequality = ca.MX.zeros(layout.inequality_count, state_count)
    row = 0
    for component in layout.particulate_indices:
        inequality[row, final_reactor.start + component] = underflow
        inequality[row, layout.underflow_flow_slice.start + component] = -1.0
        row += 1
    for layer in range(1, layout.layer_count - 1):
        inequality[row, first_layer] = 1.0
        inequality[row, first_layer + layer] = -1.0
        row += 1
    for layer in range(1, layout.layer_count - 1):
        inequality[row, first_layer + layer] = 1.0
        inequality[row, last_layer] = -1.0
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
    layers = state[layout.layer_slice]
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
    layer_volume = limits.clarifier_volume_m3 / layout.layer_count
    inventory = reactor_inventory + layer_volume * ca.sum1(layers)
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
        (limits.srt_lower_d * limits.fresh_flow_m3_d * external_loss - inventory) / scale,
        (inventory - limits.srt_upper_d * limits.fresh_flow_m3_d * external_loss) / scale,
        (limits.external_loss_min_g_m3 - external_loss) / limits.external_loss_min_g_m3,
        slr / limits.slr_upper_kg_m2_d - 1.0,
        underflow_tss / limits.underflow_tss_upper_g_m3 - 1.0,
        (limits.feed_tss_min_g_m3 - feed_tss) / limits.feed_tss_min_g_m3,
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
        rows.append(sor / limits.sor_upper_m_d - 1.0)
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
        correction_squared - thresholds.correction_rms**2,
        leverage - thresholds.regularized_leverage,
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
        (
            "flux",
            assets.trust_callbacks.flux_rows,
            thresholds.flux_rms,
        ),
    )
    for name, callback, threshold in specifications:
        if callback is None or threshold is None:
            continue
        residual = _callback_rows(callback, theta, raw, projected, influent, name)
        squared = ca.dot(residual, residual) / residual.numel()
        rows.append(squared - threshold**2)
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
        rows.append(squared - specification.rms_threshold**2)
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

    def ipopt_options(self) -> dict[str, Any]:
        return {
            "print_time": False,
            "ipopt.print_level": self.print_level,
            "ipopt.sb": "yes",
            "ipopt.max_iter": self.maximum_iterations,
            "ipopt.tol": self.tolerance,
            "ipopt.acceptable_tol": self.acceptable_tolerance,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.bound_relax_factor": 0.0,
            "ipopt.honor_original_bounds": "yes",
        }


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

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["primal"] = self.primal.tolist()
        return value


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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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
                "diagnostics": self.projection.diagnostics.as_dict(),
            },
            "feasibility": self.feasibility.as_dict(),
            "stationarity": self.stationarity.as_dict(),
            "status": self.status,
        }


@dataclass(frozen=True)
class SurrogateStartResult:
    start_index: int
    initial_normalized_controls: FloatArray
    stages: tuple[ContinuationStageRecord, ...]
    outer_refinement: OuterRefinementRecord
    final: FinalCandidateRecord | None
    status: str
    error: str | None = None

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
        }


@dataclass(frozen=True)
class SurrogateMultistartResult:
    starts: tuple[SurrogateStartResult, ...]
    selected: SurrogateStartResult | None
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "starts": [result.as_dict() for result in self.starts],
            "selected_start": None if self.selected is None else self.selected.start_index,
            "status": self.status,
        }


def _solve_continuation_stage(
    problem: SurrogateNLP,
    case: SurrogateCase,
    initial: FloatArray,
    settings: SurrogateSolverSettings,
) -> ContinuationStageRecord:
    if problem.solver is None:
        raise SurrogateNLPError("the continuation problem was built without an NLP solver.")
    started = perf_counter()
    try:
        solution = problem.solver(
            x0=initial,
            p=case.parameter_vector(problem.assets),
            lbx=problem.lower_bounds,
            ubx=problem.upper_bounds,
            lbg=problem.constraint_lower_bounds,
            ubg=problem.constraint_upper_bounds,
        )
        elapsed = perf_counter() - started
        primal = _flat(solution["x"])
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
        return ContinuationStageRecord(
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
        )
    except Exception as exc:
        return ContinuationStageRecord(
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


def _outer_refine(
    problem: SurrogateNLP,
    case: SurrogateCase,
    initial: FloatArray,
    settings: SurrogateSolverSettings,
) -> tuple[FloatArray, OuterRefinementRecord]:
    initial_evaluation = _exact_evaluation(problem, case, initial)
    initial_feasibility = _feasibility_record(initial_evaluation, settings)
    if not initial_feasibility.feasible or not settings.perform_outer_refinement:
        return initial.copy(), OuterRefinementRecord(
            attempted=False,
            solver_success=False,
            status=(
                "disabled"
                if not settings.perform_outer_refinement
                else "initial_exact_qp_candidate_infeasible"
            ),
            iterations=0,
            evaluations=1,
            elapsed_seconds=0.0,
            initial_objective=(
                initial_evaluation.objective
                if np.isfinite(initial_evaluation.objective)
                else None
            ),
            final_objective=None,
        )
    cache: dict[bytes, _ExactEvaluation] = {initial.tobytes(): initial_evaluation}

    def evaluate(value: npt.ArrayLike) -> _ExactEvaluation:
        normalized = np.asarray(value, dtype=np.float64).reshape(7)
        key = normalized.tobytes()
        if key not in cache:
            cache[key] = _exact_evaluation(problem, case, normalized)
        return cache[key]

    def objective(value: npt.ArrayLike) -> float:
        evaluated = evaluate(value)
        if not evaluated.projection.accepted or not np.isfinite(evaluated.objective):
            return 1.0e20
        return evaluated.objective

    def constraints(value: npt.ArrayLike) -> FloatArray:
        evaluated = evaluate(value)
        if not evaluated.projection.accepted:
            return np.full(
                len(problem.engineering_names) + len(problem.trust_names),
                -1.0e6,
            )
        return -np.concatenate((evaluated.engineering, evaluated.trust))

    started = perf_counter()
    try:
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * 7,
            constraints={"type": "ineq", "fun": constraints},
            options={
                "maxiter": settings.outer_maximum_iterations,
                "ftol": settings.outer_function_tolerance,
                "disp": False,
            },
        )
        elapsed = perf_counter() - started
        proposed = np.asarray(result.x, dtype=np.float64)
        candidates = [initial_evaluation, evaluate(proposed)]
        feasible = [
            item
            for item in candidates
            if _feasibility_record(item, settings).feasible
        ]
        selected = min(
            feasible,
            key=lambda item: (item.objective, *item.normalized.tolist()),
        )
        return selected.normalized.copy(), OuterRefinementRecord(
            attempted=True,
            solver_success=bool(result.success),
            status=str(result.message),
            iterations=int(result.nit),
            evaluations=len(cache),
            elapsed_seconds=elapsed,
            initial_objective=initial_evaluation.objective,
            final_objective=selected.objective,
        )
    except Exception as exc:
        return initial.copy(), OuterRefinementRecord(
            attempted=True,
            solver_success=False,
            status=f"{type(exc).__name__}: {exc}",
            iterations=0,
            evaluations=len(cache),
            elapsed_seconds=perf_counter() - started,
            initial_objective=initial_evaluation.objective,
            final_objective=initial_evaluation.objective,
        )


def solve_surrogate_start(
    problems: Sequence[SurrogateNLP],
    case: SurrogateCase,
    normalized_start: npt.ArrayLike,
    *,
    start_index: int,
    settings: SurrogateSolverSettings | None = None,
) -> SurrogateStartResult:
    """Run one start through all four gap stages and exact-QP refinement."""

    settings = settings or SurrogateSolverSettings()
    if tuple(problem.tau for problem in problems) != GAP_CONTINUATION:
        raise ValueError(f"problems must follow the exact continuation {GAP_CONTINUATION}.")
    assets = problems[0].assets
    if any(problem.assets is not assets for problem in problems):
        raise ValueError("all continuation stages must share one assets object.")
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
        )
    for problem in problems:
        stage = _solve_continuation_stage(problem, case, primal, settings)
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
            )
    normalized = primal[problems[-1].theta_slice]
    # Only an independently projected feasible point is eligible for exact-QP
    # refinement.  The refinement itself cold-solves OSQP at every trial.
    initial_final = audit_exact_candidate(
        problems[-1], case, normalized, settings=settings
    )
    if initial_final.feasibility.feasible:
        refined, refinement = _outer_refine(
            problems[-1], case, normalized, settings
        )
        # This is an additional, uncached cold OSQP replay for the reportable
        # endpoint, even if SLSQP returned its initial vector.
        final = audit_exact_candidate(
            problems[-1], case, refined, settings=settings
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
    )


def solve_surrogate_multistart(
    assets: SurrogateNLPAssets,
    case: SurrogateCase,
    *,
    settings: SurrogateSolverSettings | None = None,
    starts: npt.ArrayLike | None = None,
    name: str = "v3_surrogate",
) -> SurrogateMultistartResult:
    """Run all nine deterministic starts through the exact v3 gap schedule."""

    settings = settings or SurrogateSolverSettings()
    normalized_starts = (
        ordered_normalized_starts()
        if starts is None
        else np.asarray(starts, dtype=np.float64)
    )
    if normalized_starts.shape != (9, 7) or not np.all(np.isfinite(normalized_starts)):
        raise ValueError("starts must be a finite 9-by-7 matrix.")
    if np.any(normalized_starts < 0.0) or np.any(normalized_starts > 1.0):
        raise ValueError("all normalized starts must lie in [0, 1].")
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
    results = tuple(
        solve_surrogate_start(
            problems,
            case,
            normalized,
            start_index=index,
            settings=settings,
        )
        for index, normalized in enumerate(normalized_starts)
    )
    feasible = [result for result in results if result.feasible and result.final is not None]
    if not feasible:
        return SurrogateMultistartResult(results, None, "no_validated_feasible_start")
    # No candidate is promoted to stationary by this module.  The stable
    # active-set sensitivity and upper KKT reconstruction remain an explicit
    # downstream audit, so selection is among validated local incumbents.
    minimum = min(result.final.objective for result in feasible if result.final is not None)
    tolerance = 1.0e-10 * max(1.0, abs(minimum))
    tied = [
        result
        for result in feasible
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
        "selected_stationarity_unresolved",
    )


IMPLEMENTATION_LIMITATIONS = (
    "The module records a cold lower-QP KKT audit and exact-QP upper "
    "feasibility, but intentionally does not claim upper stationarity. "
    "The manuscript's active-set rank/conditioning/strict-complementarity "
    "sensitivity and independently reconstructed upper multipliers must be "
    "implemented and passed before changing stationarity_unresolved."
)


__all__ = [
    "DEFAULT_OBJECTIVE_WEIGHTS",
    "GAP_CONTINUATION",
    "IMPLEMENTATION_LIMITATIONS",
    "START_SEED",
    "EngineeringLimits",
    "FeasibilityRecord",
    "FinalCandidateRecord",
    "NamedTrustRows",
    "OuterRefinementRecord",
    "StationarityRecord",
    "SurrogateCase",
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
    "evaluate_surrogate_problem",
    "initial_primal_from_projection",
    "ordered_normalized_starts",
    "regularized_leverage_contract",
    "solve_surrogate_multistart",
    "solve_surrogate_start",
    "symbolic_network_operators",
    "symbolic_quadratic_prediction",
    "unpack_primal",
]
