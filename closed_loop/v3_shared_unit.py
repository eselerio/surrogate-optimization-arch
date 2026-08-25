"""Shared-reactor/Clarifier surrogate route for the manuscript-v3 study.

This module implements route ``U`` without changing the existing whole-system
surrogate or smooth mechanistic routes.  One quadratic reactor map is reused at
every CSTR, one quadratic Clarifier map is used once, and an independently
audited two-start recycle solve closes the learned plant before the unchanged
physical projection is applied.

The implementation deliberately keeps the learned recycle root outside the
projection QP.  A route value exists only when both prescribed, target-free
starts converge to the same scaled mixer and complete raw response, and both
root Jacobians pass the declared rank and conditioning gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from time import perf_counter, perf_counter_ns
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import linalg
from scipy.optimize import Bounds, NonlinearConstraint, least_squares, minimize

from .design import SplitMix64
from .manuscript_v3 import DECISION_LOWER, DECISION_UPPER, RIDGE_GRID
from .model import COMPOSITE_MATRIX, INVARIANT_MATRIX, TSS_VECTOR
from .projection import (
    LeastSquaresDiagnostics,
    NetworkLayout,
    NetworkRowScales,
    PhysicalProjector,
    ProjectionDiagnostics,
    ProjectionResult,
    QuadraticFeatureMap,
    QuadraticSurrogate,
    build_network_operators,
)
from .v3_surrogate_nlp import DEFAULT_OBJECTIVE_WEIGHTS, EngineeringLimits
from .v3_parallel import BatchProgress, run_resumable_batches


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
TrustRows = Callable[[FloatArray, FloatArray, FloatArray, FloatArray], npt.ArrayLike]

ROOT_RESIDUAL_TOLERANCE = 1.0e-8
ROOT_MIXER_AGREEMENT_TOLERANCE = 1.0e-6
ROOT_RESPONSE_AGREEMENT_TOLERANCE = 1.0e-6
ROOT_CONDITION_EPSILON_LIMIT = 1.0e-8
ROOT_SOLVER_TOLERANCE = 1.0e-10
ROOT_MAXIMUM_EVALUATIONS = 1_000


class SharedUnitError(RuntimeError):
    """Base error for an undefined or invalid shared-unit calculation."""


def _finite_vector(value: npt.ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {size}.")
    return array.copy()


def _finite_matrix(
    value: npt.ArrayLike,
    columns: int,
    name: str,
    *,
    rows: int | None = None,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    expected = (rows, columns) if rows is not None else None
    if array.ndim != 2 or array.shape[1] != columns or (
        expected is not None and array.shape != expected
    ) or not np.all(np.isfinite(array)):
        description = f"{expected}" if expected is not None else f"(n, {columns})"
        raise ValueError(f"{name} must be a finite matrix with shape {description}.")
    return array.copy()


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _json_array(value: npt.ArrayLike | None) -> list[Any] | None:
    return None if value is None else np.asarray(value).tolist()


def _optional_array(value: Any) -> FloatArray | None:
    """Restore an optional numeric array emitted by :func:`_json_array`."""

    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def _maximum_positive(value: npt.ArrayLike) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(np.max(np.maximum(array, 0.0), initial=0.0))


def _normalized_limit(value: float, limit: float) -> float:
    if not np.isfinite(limit) or limit < 0.0:
        raise ValueError("a trust limit must be finite and nonnegative")
    return float((value - limit) / (limit if limit > 0.0 else 1.0))


def _nearest_rank_95(value: npt.ArrayLike) -> float:
    array = np.sort(np.asarray(value, dtype=np.float64).reshape(-1))
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("nearest-rank calibration requires finite observations")
    return float(array[math.ceil(0.95 * array.size) - 1])


@dataclass(frozen=True)
class SharedUnitTrainingData:
    """Teacher-forced unit rows with stable plant and stage provenance."""

    reactor_decisions: FloatArray
    reactor_upstream: FloatArray
    reactor_targets: FloatArray
    reactor_plant_index: IntArray
    reactor_stage_index: IntArray
    clarifier_decisions: FloatArray
    clarifier_feed: FloatArray
    clarifier_targets: FloatArray
    plant_count: int
    stage_count: int
    component_count: int

    def __post_init__(self) -> None:
        plants = int(self.plant_count)
        stages = int(self.stage_count)
        components = int(self.component_count)
        if plants < 1 or stages < 1 or components < 1:
            raise ValueError("shared-unit training dimensions must be positive")
        transitions = plants * stages
        reactor_decisions = _finite_matrix(
            self.reactor_decisions, 2, "reactor_decisions", rows=transitions
        )
        reactor_upstream = _finite_matrix(
            self.reactor_upstream, components, "reactor_upstream", rows=transitions
        )
        reactor_targets = _finite_matrix(
            self.reactor_targets, components, "reactor_targets", rows=transitions
        )
        plant_index = np.asarray(self.reactor_plant_index, dtype=np.int64).reshape(-1)
        stage_index = np.asarray(self.reactor_stage_index, dtype=np.int64).reshape(-1)
        if plant_index.shape != (transitions,) or not np.array_equal(
            plant_index, np.repeat(np.arange(plants, dtype=np.int64), stages)
        ):
            raise ValueError("reactor_plant_index must use plant-major ordering")
        if stage_index.shape != (transitions,) or not np.array_equal(
            stage_index, np.tile(np.arange(stages, dtype=np.int64), plants)
        ):
            raise ValueError("reactor_stage_index must use stage-minor ordering")
        clarifier_decisions = _finite_matrix(
            self.clarifier_decisions, 2, "clarifier_decisions", rows=plants
        )
        clarifier_feed = _finite_matrix(
            self.clarifier_feed, components, "clarifier_feed", rows=plants
        )
        clarifier_targets = _finite_matrix(
            self.clarifier_targets,
            2 * components + 1,
            "clarifier_targets",
            rows=plants,
        )
        object.__setattr__(self, "reactor_decisions", reactor_decisions)
        object.__setattr__(self, "reactor_upstream", reactor_upstream)
        object.__setattr__(self, "reactor_targets", reactor_targets)
        object.__setattr__(self, "reactor_plant_index", plant_index.copy())
        object.__setattr__(self, "reactor_stage_index", stage_index.copy())
        object.__setattr__(self, "clarifier_decisions", clarifier_decisions)
        object.__setattr__(self, "clarifier_feed", clarifier_feed)
        object.__setattr__(self, "clarifier_targets", clarifier_targets)

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "reactor_decisions": self.reactor_decisions.copy(),
            "reactor_upstream": self.reactor_upstream.copy(),
            "reactor_targets": self.reactor_targets.copy(),
            "reactor_plant_index": self.reactor_plant_index.copy(),
            "reactor_stage_index": self.reactor_stage_index.copy(),
            "clarifier_decisions": self.clarifier_decisions.copy(),
            "clarifier_feed": self.clarifier_feed.copy(),
            "clarifier_targets": self.clarifier_targets.copy(),
        }

    def as_dict(self) -> dict[str, int]:
        return {
            "plant_count": self.plant_count,
            "stage_count": self.stage_count,
            "component_count": self.component_count,
            "reactor_row_count": int(self.reactor_targets.shape[0]),
            "clarifier_row_count": int(self.clarifier_targets.shape[0]),
            "reactor_feature_count": QuadraticFeatureMap.expected_feature_count(
                2, self.component_count
            ),
            "clarifier_feature_count": QuadraticFeatureMap.expected_feature_count(
                2, self.component_count
            ),
        }


def extract_shared_unit_training(
    decisions: npt.ArrayLike,
    reduced_targets: npt.ArrayLike,
    *,
    layout: NetworkLayout | None = None,
) -> SharedUnitTrainingData:
    """Create plant-grouped CSTR transitions and one Clarifier row per plant."""

    layout = layout or NetworkLayout()
    theta = _finite_matrix(decisions, 7, "decisions")
    targets = _finite_matrix(
        reduced_targets, layout.state_size, "reduced_targets", rows=len(theta)
    )
    if layout.stage_count != 5:
        raise ValueError("the seven-control shared-unit case requires five reactor stages")
    q_process = 1.0 + theta[:, 4] + theta[:, 5]
    dilution = 120.0 * q_process / theta[:, 0]
    aeration = np.column_stack(
        (np.zeros((len(theta), 2), dtype=np.float64), theta[:, 1:4])
    )
    reactors = np.stack(
        [targets[:, layout.reactor_slice(stage)] for stage in range(layout.stage_count)],
        axis=1,
    )
    upstream = np.empty_like(reactors)
    upstream[:, 0, :] = targets[:, layout.mixer_slice]
    upstream[:, 1:, :] = reactors[:, :-1, :]
    reactor_decisions = np.column_stack(
        (np.repeat(dilution, layout.stage_count), aeration.reshape(-1))
    )
    final = reactors[:, -1, :]
    clarifier_targets = np.concatenate(
        (
            targets[:, layout.overflow_flow_slice],
            targets[:, layout.underflow_flow_slice],
            targets[:, layout.inventory_slice],
        ),
        axis=1,
    )
    return SharedUnitTrainingData(
        reactor_decisions=reactor_decisions,
        reactor_upstream=upstream.reshape(-1, layout.component_count),
        reactor_targets=reactors.reshape(-1, layout.component_count),
        reactor_plant_index=np.repeat(
            np.arange(len(theta), dtype=np.int64), layout.stage_count
        ),
        reactor_stage_index=np.tile(
            np.arange(layout.stage_count, dtype=np.int64), len(theta)
        ),
        clarifier_decisions=theta[:, [5, 6]],
        clarifier_feed=final,
        clarifier_targets=clarifier_targets,
        plant_count=len(theta),
        stage_count=layout.stage_count,
        component_count=layout.component_count,
    )


@dataclass(frozen=True)
class SharedUnitRidgeScore:
    family: str
    fold: int
    gamma: float
    raw_nrmse: float
    selected: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _model_to_arrays(model: QuadraticSurrogate, prefix: str) -> dict[str, np.ndarray]:
    feature = model.feature_map
    return {
        f"{prefix}_decision_center": feature.decision_center,
        f"{prefix}_decision_scale": feature.decision_scale,
        f"{prefix}_influent_center": feature.influent_center,
        f"{prefix}_influent_scale": feature.influent_scale,
        f"{prefix}_term_center": feature.term_center,
        f"{prefix}_term_scale": feature.term_scale,
        f"{prefix}_variance_relative_tolerance": np.asarray(
            feature.variance_relative_tolerance
        ),
        f"{prefix}_response_center": model.response_center,
        f"{prefix}_response_scale": model.response_scale,
        f"{prefix}_coefficients": model.coefficients,
        f"{prefix}_ridge_penalty": np.asarray(model.ridge_penalty),
        f"{prefix}_diagnostics_json": np.asarray(
            json.dumps(asdict(model.diagnostics), sort_keys=True)
        ),
    }


def _model_from_arrays(values: Mapping[str, npt.ArrayLike], prefix: str) -> QuadraticSurrogate:
    feature = QuadraticFeatureMap(
        decision_center=np.asarray(values[f"{prefix}_decision_center"], dtype=float),
        decision_scale=np.asarray(values[f"{prefix}_decision_scale"], dtype=float),
        influent_center=np.asarray(values[f"{prefix}_influent_center"], dtype=float),
        influent_scale=np.asarray(values[f"{prefix}_influent_scale"], dtype=float),
        term_center=np.asarray(values[f"{prefix}_term_center"], dtype=float),
        term_scale=np.asarray(values[f"{prefix}_term_scale"], dtype=float),
        variance_relative_tolerance=float(
            np.asarray(values[f"{prefix}_variance_relative_tolerance"]).item()
        ),
    )
    diagnostics = LeastSquaresDiagnostics(
        **json.loads(str(np.asarray(values[f"{prefix}_diagnostics_json"]).item()))
    )
    return QuadraticSurrogate(
        feature_map=feature,
        response_center=np.asarray(values[f"{prefix}_response_center"], dtype=float),
        response_scale=np.asarray(values[f"{prefix}_response_scale"], dtype=float),
        coefficients=np.asarray(values[f"{prefix}_coefficients"], dtype=float),
        diagnostics=diagnostics,
        ridge_penalty=float(np.asarray(values[f"{prefix}_ridge_penalty"]).item()),
    )


@dataclass(frozen=True)
class SharedUnitModels:
    reactor: QuadraticSurrogate
    clarifier: QuadraticSurrogate

    def __post_init__(self) -> None:
        component_count = self.reactor.response_count
        for name, model in (("reactor", self.reactor), ("clarifier", self.clarifier)):
            if model.feature_map.decision_count != 2:
                raise ValueError(f"{name} model must have two scalar operating inputs")
            if model.feature_map.influent_count != component_count:
                raise ValueError(f"{name} state-input dimension is inconsistent")
        if self.clarifier.response_count != 2 * component_count + 1:
            raise ValueError("clarifier response must contain g_E, g_U, and M_cl")

    @property
    def component_count(self) -> int:
        return self.reactor.response_count

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            **_model_to_arrays(self.reactor, "reactor"),
            **_model_to_arrays(self.clarifier, "clarifier"),
        }

    @classmethod
    def from_arrays(cls, values: Mapping[str, npt.ArrayLike]) -> "SharedUnitModels":
        return cls(
            reactor=_model_from_arrays(values, "reactor"),
            clarifier=_model_from_arrays(values, "clarifier"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "reactor_ridge_penalty": float(self.reactor.ridge_penalty),
            "clarifier_ridge_penalty": float(self.clarifier.ridge_penalty),
            "reactor_shape": list(self.reactor.coefficients.shape),
            "clarifier_shape": list(self.clarifier.coefficients.shape),
            "reactor_diagnostics": asdict(self.reactor.diagnostics),
            "clarifier_diagnostics": asdict(self.clarifier.diagnostics),
        }


@dataclass(frozen=True)
class SharedUnitFitResult:
    models: SharedUnitModels
    fold_models: tuple[SharedUnitModels, ...]
    scores: tuple[SharedUnitRidgeScore, ...]
    plant_fold_membership: IntArray
    reactor_out_of_fold_raw: FloatArray
    clarifier_out_of_fold_raw: FloatArray
    elapsed_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": self.models.as_dict(),
            "fold_count": len(self.fold_models),
            "scores": [item.as_dict() for item in self.scores],
            "plant_fold_membership": self.plant_fold_membership.tolist(),
            "reactor_out_of_fold_shape": list(self.reactor_out_of_fold_raw.shape),
            "clarifier_out_of_fold_shape": list(self.clarifier_out_of_fold_raw.shape),
            "elapsed_seconds": float(self.elapsed_seconds),
        }

    def score_frame(self) -> pd.DataFrame:
        return pd.DataFrame([item.as_dict() for item in self.scores])


def _deterministic_fold_membership(count: int, fold_count: int) -> IntArray:
    if count < fold_count or fold_count < 2:
        raise ValueError("grouped cross-validation needs at least one plant per fold")
    order = list(range(count))
    stream = SplitMix64(271_828)
    for index in range(count - 1, 0, -1):
        swap = stream.randbelow(index + 1)
        order[index], order[swap] = order[swap], order[index]
    membership = np.empty(count, dtype=np.int64)
    for fold, indices in enumerate(np.array_split(np.asarray(order), fold_count), start=1):
        membership[indices] = fold
    return membership


def _validate_fold_membership(
    value: npt.ArrayLike | None,
    count: int,
    fold_count: int,
) -> IntArray:
    membership = (
        _deterministic_fold_membership(count, fold_count)
        if value is None
        else np.asarray(value, dtype=np.int64).reshape(-1)
    )
    if membership.shape != (count,) or set(membership.tolist()) != set(
        range(1, fold_count + 1)
    ):
        raise ValueError("plant fold membership is incomplete or invalid")
    counts = np.bincount(membership, minlength=fold_count + 1)[1:]
    if int(np.max(counts) - np.min(counts)) > 1:
        raise ValueError("plant folds must differ in size by at most one")
    return membership.copy()


def _select_ridge(scores: Sequence[SharedUnitRidgeScore], family: str) -> float:
    family_scores = [item for item in scores if item.family == family]
    candidates = sorted(set(item.gamma for item in family_scores))
    summary: list[tuple[float, float, float]] = []
    for gamma in candidates:
        values = np.asarray(
            [item.raw_nrmse for item in family_scores if item.gamma == gamma],
            dtype=float,
        )
        mean = float(np.mean(values))
        standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
        summary.append((gamma, mean, standard_error))
    minimum = min(summary, key=lambda item: (item[1], -item[0]))
    eligible = [
        gamma for gamma, mean, _ in summary if mean <= minimum[1] + minimum[2]
    ]
    return float(max(eligible))


def cross_validate_shared_unit_models(
    training: SharedUnitTrainingData,
    *,
    plant_fold_membership: npt.ArrayLike | None = None,
    ridge_grid: npt.ArrayLike = RIDGE_GRID,
    fold_count: int = 5,
) -> SharedUnitFitResult:
    """Select reactor and Clarifier penalties with leakage-free plant folds."""

    membership = _validate_fold_membership(
        plant_fold_membership, training.plant_count, fold_count
    )
    grid = np.asarray(ridge_grid, dtype=np.float64).reshape(-1)
    if grid.size == 0 or np.any(~np.isfinite(grid)) or np.any(grid <= 0.0):
        raise ValueError("ridge_grid must contain positive finite penalties")
    grid = np.unique(grid)
    started = perf_counter()
    score_items: list[SharedUnitRidgeScore] = []
    cached: dict[tuple[str, int, float], QuadraticSurrogate] = {}
    reactor_membership = membership[training.reactor_plant_index]
    families = (
        (
            "reactor",
            training.reactor_decisions,
            training.reactor_upstream,
            training.reactor_targets,
            reactor_membership,
        ),
        (
            "clarifier",
            training.clarifier_decisions,
            training.clarifier_feed,
            training.clarifier_targets,
            membership,
        ),
    )
    for family, decisions, state_input, targets, row_membership in families:
        for fold in range(1, fold_count + 1):
            fitting = row_membership != fold
            validation = ~fitting
            for gamma in grid:
                model = QuadraticSurrogate.fit_ridge(
                    decisions[fitting],
                    state_input[fitting],
                    targets[fitting],
                    ridge_penalty=float(gamma),
                )
                prediction = model.predict(decisions[validation], state_input[validation])
                score = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                (prediction - targets[validation]) / model.response_scale
                            )
                        )
                    )
                )
                cached[(family, fold, float(gamma))] = model
                score_items.append(
                    SharedUnitRidgeScore(family, fold, float(gamma), score, False)
                )
    selected_reactor = _select_ridge(score_items, "reactor")
    selected_clarifier = _select_ridge(score_items, "clarifier")
    selected_by_family = {
        "reactor": selected_reactor,
        "clarifier": selected_clarifier,
    }
    scores = tuple(
        SharedUnitRidgeScore(
            item.family,
            item.fold,
            item.gamma,
            item.raw_nrmse,
            item.gamma == selected_by_family[item.family],
        )
        for item in score_items
    )
    reactor_oof = np.full_like(training.reactor_targets, np.nan)
    clarifier_oof = np.full_like(training.clarifier_targets, np.nan)
    fold_models: list[SharedUnitModels] = []
    for fold in range(1, fold_count + 1):
        reactor = cached[("reactor", fold, selected_reactor)]
        clarifier = cached[("clarifier", fold, selected_clarifier)]
        fold_models.append(SharedUnitModels(reactor, clarifier))
        reactor_validation = reactor_membership == fold
        clarifier_validation = membership == fold
        reactor_oof[reactor_validation] = reactor.predict(
            training.reactor_decisions[reactor_validation],
            training.reactor_upstream[reactor_validation],
        )
        clarifier_oof[clarifier_validation] = clarifier.predict(
            training.clarifier_decisions[clarifier_validation],
            training.clarifier_feed[clarifier_validation],
        )
    final = SharedUnitModels(
        QuadraticSurrogate.fit_ridge(
            training.reactor_decisions,
            training.reactor_upstream,
            training.reactor_targets,
            ridge_penalty=selected_reactor,
        ),
        QuadraticSurrogate.fit_ridge(
            training.clarifier_decisions,
            training.clarifier_feed,
            training.clarifier_targets,
            ridge_penalty=selected_clarifier,
        ),
    )
    if not np.all(np.isfinite(reactor_oof)) or not np.all(np.isfinite(clarifier_oof)):
        raise SharedUnitError("selected shared-unit OOF predictions are incomplete")
    return SharedUnitFitResult(
        models=final,
        fold_models=tuple(fold_models),
        scores=scores,
        plant_fold_membership=membership,
        reactor_out_of_fold_raw=reactor_oof,
        clarifier_out_of_fold_raw=clarifier_oof,
        elapsed_seconds=perf_counter() - started,
    )


def quadratic_prediction_jacobian(
    model: QuadraticSurrogate,
    decisions: npt.ArrayLike,
    state_input: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate one quadratic map and its exact physical-input Jacobians."""

    feature = model.feature_map
    decision = _finite_vector(decisions, feature.decision_count, "local decisions")
    state = _finite_vector(state_input, feature.influent_count, "local state input")
    decision_z = (decision - feature.decision_center) / feature.decision_scale
    state_z = (state - feature.influent_center) / feature.influent_scale
    total = feature.decision_count + feature.influent_count
    jacobian_z = np.zeros((feature.nonconstant_count, total), dtype=np.float64)
    row = 0
    jacobian_z[row : row + feature.decision_count, : feature.decision_count] = np.eye(
        feature.decision_count
    )
    row += feature.decision_count
    jacobian_z[
        row : row + feature.influent_count, feature.decision_count :
    ] = np.eye(feature.influent_count)
    row += feature.influent_count
    for left in range(feature.decision_count):
        for right in range(left, feature.decision_count):
            jacobian_z[row, left] += decision_z[right]
            jacobian_z[row, right] += decision_z[left]
            row += 1
    offset = feature.decision_count
    for left in range(feature.influent_count):
        for right in range(left, feature.influent_count):
            jacobian_z[row, offset + left] += state_z[right]
            jacobian_z[row, offset + right] += state_z[left]
            row += 1
    for decision_index in range(feature.decision_count):
        for state_index in range(feature.influent_count):
            jacobian_z[row, decision_index] = state_z[state_index]
            jacobian_z[row, offset + state_index] = decision_z[decision_index]
            row += 1
    if row != feature.nonconstant_count:
        raise AssertionError("quadratic Jacobian serialization is inconsistent")
    input_scale = np.concatenate((feature.decision_scale, feature.influent_scale))
    standardized_jacobian = (
        jacobian_z / feature.term_scale[:, None] / input_scale[None, :]
    )
    physical_jacobian = model.response_scale[:, None] * (
        model.coefficients[:, 1:] @ standardized_jacobian
    )
    prediction = np.asarray(model.predict(decision, state), dtype=np.float64)
    return (
        prediction,
        physical_jacobian[:, : feature.decision_count],
        physical_jacobian[:, feature.decision_count :],
    )


@dataclass(frozen=True)
class SharedUnitRootAttempt:
    success: bool
    status: int
    message: str
    nfev: int
    njev: int | None
    cost: float
    optimality: float
    residual_inf: float
    mixer: FloatArray | None
    raw: FloatArray | None
    jacobian_rank: int | None
    jacobian_condition: float | None
    condition_times_epsilon: float | None

    def as_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        result = {
            "success": bool(self.success),
            "status": int(self.status),
            "message": self.message,
            "nfev": int(self.nfev),
            "njev": None if self.njev is None else int(self.njev),
            "cost": float(self.cost),
            "optimality": float(self.optimality),
            "residual_inf": float(self.residual_inf),
            "jacobian_rank": self.jacobian_rank,
            "jacobian_condition": _optional_float(self.jacobian_condition),
            "condition_times_epsilon": _optional_float(self.condition_times_epsilon),
        }
        if include_arrays:
            result.update({"mixer": _json_array(self.mixer), "raw": _json_array(self.raw)})
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SharedUnitRootAttempt":
        return cls(
            success=bool(value["success"]),
            status=int(value["status"]),
            message=str(value["message"]),
            nfev=int(value["nfev"]),
            njev=None if value.get("njev") is None else int(value["njev"]),
            cost=float(value["cost"]),
            optimality=float(value["optimality"]),
            residual_inf=float(value["residual_inf"]),
            mixer=_optional_array(value.get("mixer")),
            raw=_optional_array(value.get("raw")),
            jacobian_rank=(
                None
                if value.get("jacobian_rank") is None
                else int(value["jacobian_rank"])
            ),
            jacobian_condition=(
                None
                if value.get("jacobian_condition") is None
                else float(value["jacobian_condition"])
            ),
            condition_times_epsilon=(
                None
                if value.get("condition_times_epsilon") is None
                else float(value["condition_times_epsilon"])
            ),
        )


@dataclass(frozen=True)
class SharedUnitClosureDiagnostics:
    accepted: bool
    reason: str
    attempt_1: SharedUnitRootAttempt
    attempt_2: SharedUnitRootAttempt
    mixer_agreement_inf: float | None
    raw_agreement_inf: float | None

    def as_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "residual_inf_start_1": float(self.attempt_1.residual_inf),
            "residual_inf_start_2": float(self.attempt_2.residual_inf),
            "mixer_agreement_inf": _optional_float(self.mixer_agreement_inf),
            "raw_agreement_inf": _optional_float(self.raw_agreement_inf),
            "jacobian_rank_start_1": self.attempt_1.jacobian_rank,
            "jacobian_rank_start_2": self.attempt_2.jacobian_rank,
            "jacobian_condition_start_1": _optional_float(
                self.attempt_1.jacobian_condition
            ),
            "jacobian_condition_start_2": _optional_float(
                self.attempt_2.jacobian_condition
            ),
            "condition_times_epsilon_start_1": _optional_float(
                self.attempt_1.condition_times_epsilon
            ),
            "condition_times_epsilon_start_2": _optional_float(
                self.attempt_2.condition_times_epsilon
            ),
            "attempt_1": self.attempt_1.as_dict(include_arrays=include_arrays),
            "attempt_2": self.attempt_2.as_dict(include_arrays=include_arrays),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SharedUnitClosureDiagnostics":
        return cls(
            accepted=bool(value["accepted"]),
            reason=str(value["reason"]),
            attempt_1=SharedUnitRootAttempt.from_dict(value["attempt_1"]),
            attempt_2=SharedUnitRootAttempt.from_dict(value["attempt_2"]),
            mixer_agreement_inf=(
                None
                if value.get("mixer_agreement_inf") is None
                else float(value["mixer_agreement_inf"])
            ),
            raw_agreement_inf=(
                None
                if value.get("raw_agreement_inf") is None
                else float(value["raw_agreement_inf"])
            ),
        )


@dataclass(frozen=True)
class SharedUnitClosureResult:
    accepted: bool
    raw: FloatArray | None
    mixer: FloatArray | None
    reactors: FloatArray | None
    clarifier: FloatArray | None
    raw_jacobian_theta: FloatArray | None
    diagnostics: SharedUnitClosureDiagnostics

    def as_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        result = {
            "accepted": bool(self.accepted),
            "diagnostics": self.diagnostics.as_dict(include_arrays=include_arrays),
            "has_analytic_jacobian": self.raw_jacobian_theta is not None,
        }
        if include_arrays:
            result.update(
                {
                    "raw": _json_array(self.raw),
                    "mixer": _json_array(self.mixer),
                    "reactors": _json_array(self.reactors),
                    "clarifier": _json_array(self.clarifier),
                    "raw_jacobian_theta": _json_array(self.raw_jacobian_theta),
                }
            )
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SharedUnitClosureResult":
        return cls(
            accepted=bool(value["accepted"]),
            raw=_optional_array(value.get("raw")),
            mixer=_optional_array(value.get("mixer")),
            reactors=_optional_array(value.get("reactors")),
            clarifier=_optional_array(value.get("clarifier")),
            raw_jacobian_theta=_optional_array(value.get("raw_jacobian_theta")),
            diagnostics=SharedUnitClosureDiagnostics.from_dict(value["diagnostics"]),
        )


@dataclass(frozen=True)
class _ForwardResult:
    reactors: FloatArray
    clarifier: FloatArray
    reactor_wrt_mixer: tuple[FloatArray, ...]
    reactor_partial_theta: tuple[FloatArray, ...]
    clarifier_wrt_mixer: FloatArray
    clarifier_partial_theta: FloatArray


def _shared_unit_forward(
    models: SharedUnitModels,
    theta: FloatArray,
    mixer: FloatArray,
) -> _ForwardResult:
    component_count = models.component_count
    q_process = 1.0 + theta[4] + theta[5]
    dilution = 120.0 * q_process / theta[0]
    dilution_gradient = np.zeros(7, dtype=np.float64)
    dilution_gradient[0] = -120.0 * q_process / theta[0] ** 2
    dilution_gradient[4] = 120.0 / theta[0]
    dilution_gradient[5] = 120.0 / theta[0]
    aeration = np.asarray([0.0, 0.0, theta[1], theta[2], theta[3]])
    reactors: list[FloatArray] = []
    wrt_mixer: list[FloatArray] = []
    partial_theta: list[FloatArray] = []
    upstream = mixer
    upstream_mixer = np.eye(component_count)
    upstream_theta = np.zeros((component_count, 7), dtype=np.float64)
    for stage, aeration_value in enumerate(aeration):
        prediction, operating_jacobian, state_jacobian = quadratic_prediction_jacobian(
            models.reactor,
            np.asarray([dilution, aeration_value]),
            upstream,
        )
        aeration_gradient = np.zeros(7, dtype=np.float64)
        if stage >= 2:
            aeration_gradient[stage - 1] = 1.0
        stage_mixer = state_jacobian @ upstream_mixer
        stage_theta = (
            state_jacobian @ upstream_theta
            + np.outer(operating_jacobian[:, 0], dilution_gradient)
            + np.outer(operating_jacobian[:, 1], aeration_gradient)
        )
        reactors.append(prediction)
        wrt_mixer.append(stage_mixer)
        partial_theta.append(stage_theta)
        upstream = prediction
        upstream_mixer = stage_mixer
        upstream_theta = stage_theta
    clarifier, operating_jacobian, state_jacobian = quadratic_prediction_jacobian(
        models.clarifier, theta[[5, 6]], reactors[-1]
    )
    clarifier_theta = state_jacobian @ partial_theta[-1]
    clarifier_theta[:, 5] += operating_jacobian[:, 0]
    clarifier_theta[:, 6] += operating_jacobian[:, 1]
    clarifier_mixer = state_jacobian @ wrt_mixer[-1]
    return _ForwardResult(
        reactors=np.asarray(reactors),
        clarifier=clarifier,
        reactor_wrt_mixer=tuple(wrt_mixer),
        reactor_partial_theta=tuple(partial_theta),
        clarifier_wrt_mixer=clarifier_mixer,
        clarifier_partial_theta=clarifier_theta,
    )


def _assemble_raw(mixer: FloatArray, forward: _ForwardResult) -> FloatArray:
    return np.concatenate((mixer, forward.reactors.reshape(-1), forward.clarifier))


def _closure_residual_and_jacobian(
    models: SharedUnitModels,
    theta: FloatArray,
    influent: FloatArray,
    mixer_scale: FloatArray,
    mixer: FloatArray,
) -> tuple[FloatArray, FloatArray, _ForwardResult]:
    forward = _shared_unit_forward(models, theta, mixer)
    component_count = models.component_count
    final = forward.reactors[-1]
    underflow_flow = forward.clarifier[component_count : 2 * component_count]
    q_process = 1.0 + theta[4] + theta[5]
    q_underflow = theta[5] + theta[6]
    alpha = theta[5] / q_underflow
    residual = (
        q_process * mixer
        - influent
        - theta[4] * final
        - alpha * underflow_flow
    ) / mixer_scale
    jacobian = (
        q_process * np.eye(component_count)
        - theta[4] * forward.reactor_wrt_mixer[-1]
        - alpha
        * forward.clarifier_wrt_mixer[component_count : 2 * component_count]
    ) / mixer_scale[:, None]
    return residual, jacobian, forward


def _root_attempt(
    models: SharedUnitModels,
    theta: FloatArray,
    influent: FloatArray,
    common_scale: FloatArray,
    start: FloatArray,
) -> SharedUnitRootAttempt:
    component_count = models.component_count
    mixer_scale = common_scale[:component_count]
    last: dict[str, Any] = {}

    def evaluate(mixer: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        residual, jacobian, forward = _closure_residual_and_jacobian(
            models,
            theta,
            influent,
            mixer_scale,
            np.asarray(mixer, dtype=np.float64),
        )
        last["mixer"] = np.asarray(mixer, dtype=np.float64).copy()
        last["residual"] = residual
        last["jacobian"] = jacobian
        last["forward"] = forward
        return residual, jacobian

    try:
        result = least_squares(
            lambda value: evaluate(value)[0],
            start,
            jac=lambda value: evaluate(value)[1],
            method="trf",
            bounds=(-np.inf, np.inf),
            ftol=ROOT_SOLVER_TOLERANCE,
            xtol=ROOT_SOLVER_TOLERANCE,
            gtol=ROOT_SOLVER_TOLERANCE,
            max_nfev=ROOT_MAXIMUM_EVALUATIONS,
        )
        mixer = np.asarray(result.x, dtype=np.float64)
        residual, jacobian, forward = _closure_residual_and_jacobian(
            models, theta, influent, mixer_scale, mixer
        )
        raw = _assemble_raw(mixer, forward)
        singular = linalg.svdvals(jacobian, check_finite=True)
        tolerance = max(jacobian.shape) * np.finfo(np.float64).eps * singular[0]
        rank = int(np.count_nonzero(singular > tolerance))
        condition = (
            float(singular[0] / singular[-1]) if singular[-1] > 0.0 else np.inf
        )
        finite = all(
            np.all(np.isfinite(value)) for value in (mixer, raw, residual, jacobian)
        )
        success = bool(result.success and finite)
        return SharedUnitRootAttempt(
            success=success,
            status=int(result.status),
            message=str(result.message),
            nfev=int(result.nfev),
            njev=None if result.njev is None else int(result.njev),
            cost=float(result.cost),
            optimality=float(result.optimality),
            residual_inf=float(np.linalg.norm(residual, ord=np.inf)) if finite else np.inf,
            mixer=mixer if finite else None,
            raw=raw if finite else None,
            jacobian_rank=rank if finite else None,
            jacobian_condition=condition if finite else None,
            condition_times_epsilon=(
                condition * np.finfo(np.float64).eps if finite else None
            ),
        )
    except Exception as exc:
        return SharedUnitRootAttempt(
            success=False,
            status=-1,
            message=f"{type(exc).__name__}: {exc}",
            nfev=0,
            njev=None,
            cost=np.inf,
            optimality=np.inf,
            residual_inf=np.inf,
            mixer=None,
            raw=None,
            jacobian_rank=None,
            jacobian_condition=None,
            condition_times_epsilon=None,
        )


def _root_attempt_accepted(
    attempt: SharedUnitRootAttempt,
    component_count: int,
) -> bool:
    """Return whether one root start passes every per-attempt audit."""

    return bool(
        attempt.success
        and attempt.residual_inf <= ROOT_RESIDUAL_TOLERANCE
        and attempt.jacobian_rank == component_count
        and attempt.condition_times_epsilon is not None
        and attempt.condition_times_epsilon <= ROOT_CONDITION_EPSILON_LIMIT
    )


def _raw_theta_jacobian(
    models: SharedUnitModels,
    theta: FloatArray,
    influent: FloatArray,
    common_scale: FloatArray,
    mixer: FloatArray,
) -> FloatArray:
    component_count = models.component_count
    mixer_scale = common_scale[:component_count]
    _, root_jacobian, forward = _closure_residual_and_jacobian(
        models, theta, influent, mixer_scale, mixer
    )
    final = forward.reactors[-1]
    underflow = forward.clarifier[component_count : 2 * component_count]
    q_underflow = theta[5] + theta[6]
    alpha = theta[5] / q_underflow
    q_gradient = np.zeros(7)
    q_gradient[4:6] = 1.0
    internal_gradient = np.zeros(7)
    internal_gradient[4] = 1.0
    alpha_gradient = np.zeros(7)
    alpha_gradient[5] = theta[6] / q_underflow**2
    alpha_gradient[6] = -theta[5] / q_underflow**2
    final_theta = forward.reactor_partial_theta[-1]
    underflow_theta = forward.clarifier_partial_theta[
        component_count : 2 * component_count
    ]
    physical_partial = (
        np.outer(mixer, q_gradient)
        - np.outer(final, internal_gradient)
        - theta[4] * final_theta
        - np.outer(underflow, alpha_gradient)
        - alpha * underflow_theta
    )
    scaled_partial = physical_partial / mixer_scale[:, None]
    mixer_theta = -linalg.solve(
        root_jacobian, scaled_partial, assume_a="gen", check_finite=True
    )
    reactor_total = [
        partial + derivative @ mixer_theta
        for partial, derivative in zip(
            forward.reactor_partial_theta,
            forward.reactor_wrt_mixer,
            strict=True,
        )
    ]
    clarifier_total = (
        forward.clarifier_partial_theta + forward.clarifier_wrt_mixer @ mixer_theta
    )
    result = np.vstack((mixer_theta, *reactor_total, clarifier_total))
    if result.shape != (common_scale.size, 7) or not np.all(np.isfinite(result)):
        raise SharedUnitError("implicit route-U response derivative is invalid")
    return result


def solve_shared_unit_closure(
    models: SharedUnitModels,
    theta: npt.ArrayLike,
    influent: npt.ArrayLike,
    common_response_scale: npt.ArrayLike,
    *,
    layout: NetworkLayout | None = None,
    with_jacobian: bool = False,
) -> SharedUnitClosureResult:
    """Cold-solve and independently audit the two prescribed recycle starts."""

    layout = layout or NetworkLayout(component_count=models.component_count)
    if layout.stage_count != 5 or layout.component_count != models.component_count:
        raise ValueError("shared-unit closure layout is inconsistent with the fitted maps")
    controls = _finite_vector(theta, 7, "theta")
    feed = _finite_vector(influent, layout.component_count, "influent")
    scale = _finite_vector(
        common_response_scale, layout.state_size, "common_response_scale"
    )
    if np.any(scale <= 0.0):
        raise ValueError("common_response_scale must be strictly positive")
    if controls[0] <= 0.0 or controls[5] + controls[6] <= 0.0:
        raise ValueError("theta does not define positive hydraulic factors")
    start_1 = feed.copy()
    start_2 = feed.copy()
    start_2[np.asarray(layout.particulate_indices, dtype=int)] *= 3.5
    attempt_1 = _root_attempt(models, controls, feed, scale, start_1)
    attempt_2 = _root_attempt(models, controls, feed, scale, start_2)
    mixer_agreement: float | None = None
    raw_agreement: float | None = None
    reasons: list[str] = []
    if not attempt_1.success:
        reasons.append("start_1_solver_failed")
    if not attempt_2.success:
        reasons.append("start_2_solver_failed")
    for index, attempt in enumerate((attempt_1, attempt_2), start=1):
        if attempt.residual_inf > ROOT_RESIDUAL_TOLERANCE:
            reasons.append(f"start_{index}_residual_failed")
        if attempt.jacobian_rank != layout.component_count:
            reasons.append(f"start_{index}_jacobian_rank_failed")
        condition_epsilon = attempt.condition_times_epsilon
        if condition_epsilon is None or condition_epsilon > ROOT_CONDITION_EPSILON_LIMIT:
            reasons.append(f"start_{index}_jacobian_condition_failed")
    if (
        attempt_1.mixer is not None
        and attempt_2.mixer is not None
        and attempt_1.raw is not None
        and attempt_2.raw is not None
    ):
        mixer_agreement = float(
            np.linalg.norm(
                (attempt_1.mixer - attempt_2.mixer) / scale[: layout.component_count],
                ord=np.inf,
            )
        )
        raw_agreement = float(
            np.linalg.norm((attempt_1.raw - attempt_2.raw) / scale, ord=np.inf)
        )
        if mixer_agreement > ROOT_MIXER_AGREEMENT_TOLERANCE:
            reasons.append("mixer_two_start_agreement_failed")
        if raw_agreement > ROOT_RESPONSE_AGREEMENT_TOLERANCE:
            reasons.append("response_two_start_agreement_failed")
    else:
        reasons.append("two_start_outputs_unavailable")
    accepted = not reasons
    reason = "accepted" if accepted else ";".join(dict.fromkeys(reasons))
    diagnostics = SharedUnitClosureDiagnostics(
        accepted=accepted,
        reason=reason,
        attempt_1=attempt_1,
        attempt_2=attempt_2,
        mixer_agreement_inf=mixer_agreement,
        raw_agreement_inf=raw_agreement,
    )
    if not accepted:
        return SharedUnitClosureResult(False, None, None, None, None, None, diagnostics)
    assert attempt_1.mixer is not None and attempt_1.raw is not None
    forward = _shared_unit_forward(models, controls, attempt_1.mixer)
    raw_jacobian = (
        _raw_theta_jacobian(models, controls, feed, scale, attempt_1.mixer)
        if with_jacobian
        else None
    )
    return SharedUnitClosureResult(
        accepted=True,
        raw=attempt_1.raw.copy(),
        mixer=attempt_1.mixer.copy(),
        reactors=forward.reactors.copy(),
        clarifier=forward.clarifier.copy(),
        raw_jacobian_theta=raw_jacobian,
        diagnostics=diagnostics,
    )


def project_shared_unit_raw(
    raw: npt.ArrayLike,
    theta: npt.ArrayLike,
    influent: npt.ArrayLike,
    common_response_scale: npt.ArrayLike,
    row_scales: NetworkRowScales,
    *,
    layout: NetworkLayout | None = None,
    invariant_operator: npt.ArrayLike = INVARIANT_MATRIX,
    tss_weights: npt.ArrayLike = TSS_VECTOR,
    clarifier_volume_m3: float = 6_000.0,
    raise_on_failure: bool = False,
) -> ProjectionResult:
    """Apply the unchanged cold physical QP to an assembled route-U response."""

    layout = layout or NetworkLayout()
    state = _finite_vector(raw, layout.state_size, "shared-unit raw response")
    controls = _finite_vector(theta, 7, "theta")
    feed = _finite_vector(influent, layout.component_count, "influent")
    scale = _finite_vector(
        common_response_scale, layout.state_size, "common_response_scale"
    )
    if np.any(scale <= 0.0):
        raise ValueError("common_response_scale must be strictly positive")
    operators = build_network_operators(
        feed,
        internal_recycle=float(controls[4]),
        return_recycle=float(controls[5]),
        waste_fraction=float(controls[6]),
        invariant_operator=invariant_operator,
        tss_weights=tss_weights,
        layout=layout,
        clarifier_volume_m3=clarifier_volume_m3,
    )
    # A new instance is intentional: no QP warm start crosses controls.
    projector = PhysicalProjector(
        state_scale=scale,
        equality_scale=row_scales.equality,
        inequality_scale=row_scales.inequality,
        absolute_tolerance=1.0e-10,
        relative_tolerance=1.0e-10,
        maximum_iterations=100_000,
        polish=True,
    )
    return projector.project(
        state,
        operators.equality_matrix,
        operators.equality_rhs,
        operators.inequality_matrix,
        warm_start=None,
        raise_on_failure=raise_on_failure,
    )


@dataclass(frozen=True)
class SharedUnitLeverageContract:
    reactor_precision: FloatArray
    clarifier_precision: FloatArray
    reactor_limit: float
    clarifier_limit: float

    def __post_init__(self) -> None:
        reactor = np.asarray(self.reactor_precision, dtype=np.float64)
        clarifier = np.asarray(self.clarifier_precision, dtype=np.float64)
        if (
            reactor.ndim != 2
            or reactor.shape[0] != reactor.shape[1]
            or clarifier.ndim != 2
            or clarifier.shape[0] != clarifier.shape[1]
            or not np.all(np.isfinite(reactor))
            or not np.all(np.isfinite(clarifier))
            or not np.allclose(reactor, reactor.T, rtol=1.0e-10, atol=1.0e-12)
            or not np.allclose(clarifier, clarifier.T, rtol=1.0e-10, atol=1.0e-12)
        ):
            raise ValueError("local leverage precision matrices must be finite symmetric squares")
        if not all(
            np.isfinite(value) and value >= 0.0
            for value in (self.reactor_limit, self.clarifier_limit)
        ):
            raise ValueError("local leverage limits must be finite and nonnegative")
        object.__setattr__(self, "reactor_precision", 0.5 * (reactor + reactor.T))
        object.__setattr__(self, "clarifier_precision", 0.5 * (clarifier + clarifier.T))

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "reactor_leverage_precision": self.reactor_precision.copy(),
            "clarifier_leverage_precision": self.clarifier_precision.copy(),
            "reactor_leverage_limit": np.asarray(self.reactor_limit),
            "clarifier_leverage_limit": np.asarray(self.clarifier_limit),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "reactor_limit": float(self.reactor_limit),
            "clarifier_limit": float(self.clarifier_limit),
            "reactor_feature_count": int(self.reactor_precision.shape[0]),
            "clarifier_feature_count": int(self.clarifier_precision.shape[0]),
        }


def _leverage_precision(
    model: QuadraticSurrogate,
    decisions: FloatArray,
    state_input: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    design = np.asarray(model.feature_map.transform(decisions, state_input), dtype=float)
    rows, feature_count = design.shape
    penalty = np.ones(feature_count, dtype=np.float64)
    penalty[0] = 0.0
    matrix = design.T @ design + rows * float(model.ridge_penalty) * np.diag(penalty)
    try:
        factor = linalg.cho_factor(matrix, lower=True, check_finite=True)
        precision = linalg.cho_solve(factor, np.eye(feature_count), check_finite=True)
    except linalg.LinAlgError as exc:
        raise SharedUnitError("local penalized feature matrix is not positive definite") from exc
    precision = 0.5 * (precision + precision.T)
    leverage = np.einsum("ij,jk,ik->i", design, precision, design)
    if not np.all(np.isfinite(leverage)) or np.min(leverage) < -1.0e-10:
        raise SharedUnitError("local leverage calculation failed")
    return precision, np.maximum(leverage, 0.0)


def fit_shared_unit_leverage(
    models: SharedUnitModels,
    training: SharedUnitTrainingData,
) -> SharedUnitLeverageContract:
    """Freeze final-fit local precision matrices and maximum-row limits."""

    if models.component_count != training.component_count:
        raise ValueError("shared-unit models and training data have different dimensions")
    reactor_precision, reactor_leverage = _leverage_precision(
        models.reactor,
        training.reactor_decisions,
        training.reactor_upstream,
    )
    clarifier_precision, clarifier_leverage = _leverage_precision(
        models.clarifier,
        training.clarifier_decisions,
        training.clarifier_feed,
    )
    return SharedUnitLeverageContract(
        reactor_precision=reactor_precision,
        clarifier_precision=clarifier_precision,
        reactor_limit=float(np.max(reactor_leverage)),
        clarifier_limit=float(np.max(clarifier_leverage)),
    )


@dataclass(frozen=True)
class SharedUnitTrustLimits:
    correction_rms: float
    reactor_leverage: float
    clarifier_leverage: float
    particulate_split_rms: float | None = None
    reactor_residual_rms: float | None = None

    def __post_init__(self) -> None:
        required = (
            self.correction_rms,
            self.reactor_leverage,
            self.clarifier_leverage,
        )
        if not all(np.isfinite(item) and item >= 0.0 for item in required):
            raise ValueError("required route-U trust limits must be finite and nonnegative")
        for item in (self.particulate_split_rms, self.reactor_residual_rms):
            if item is not None and (not np.isfinite(item) or item < 0.0):
                raise ValueError("optional route-U trust limits must be finite and nonnegative")

    def as_dict(self) -> dict[str, float | None]:
        return {
            "correction_rms": float(self.correction_rms),
            "reactor_leverage": float(self.reactor_leverage),
            "clarifier_leverage": float(self.clarifier_leverage),
            "particulate_split_rms": _optional_float(self.particulate_split_rms),
            "reactor_residual_rms": _optional_float(self.reactor_residual_rms),
        }


@dataclass(frozen=True)
class SharedUnitTrustValues:
    correction_rms: float
    reactor_leverage: FloatArray
    clarifier_leverage: float
    particulate_split_rms: float | None = None
    reactor_residual_rms: float | None = None

    def __post_init__(self) -> None:
        reactor = np.asarray(self.reactor_leverage, dtype=np.float64).reshape(-1)
        values = [self.correction_rms, self.clarifier_leverage, *reactor.tolist()]
        values.extend(
            item
            for item in (self.particulate_split_rms, self.reactor_residual_rms)
            if item is not None
        )
        if reactor.size < 1 or not all(np.isfinite(item) and item >= 0.0 for item in values):
            raise ValueError("route-U trust values must be finite and nonnegative")
        object.__setattr__(self, "reactor_leverage", reactor.copy())

    def constraint_rows(
        self, limits: SharedUnitTrustLimits
    ) -> tuple[tuple[str, ...], FloatArray]:
        names = ["correction"]
        rows = [_normalized_limit(self.correction_rms**2, limits.correction_rms**2)]
        for stage, value in enumerate(self.reactor_leverage, start=1):
            names.append(f"reactor_leverage_{stage}")
            rows.append(_normalized_limit(float(value), limits.reactor_leverage))
        names.append("clarifier_leverage")
        rows.append(_normalized_limit(self.clarifier_leverage, limits.clarifier_leverage))
        if self.particulate_split_rms is not None:
            if limits.particulate_split_rms is None:
                raise ValueError("particulate split value has no frozen limit")
            names.append("particulate_split")
            rows.append(
                _normalized_limit(
                    self.particulate_split_rms**2,
                    limits.particulate_split_rms**2,
                )
            )
        if self.reactor_residual_rms is not None:
            if limits.reactor_residual_rms is None:
                raise ValueError("reactor residual value has no frozen limit")
            names.append("reactor_residual")
            rows.append(
                _normalized_limit(
                    self.reactor_residual_rms**2,
                    limits.reactor_residual_rms**2,
                )
            )
        return tuple(names), np.asarray(rows, dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        return {
            "correction_rms": float(self.correction_rms),
            "reactor_leverage": self.reactor_leverage.tolist(),
            "clarifier_leverage": float(self.clarifier_leverage),
            "particulate_split_rms": _optional_float(self.particulate_split_rms),
            "reactor_residual_rms": _optional_float(self.reactor_residual_rms),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SharedUnitTrustValues":
        return cls(
            correction_rms=float(value["correction_rms"]),
            reactor_leverage=np.asarray(value["reactor_leverage"], dtype=np.float64),
            clarifier_leverage=float(value["clarifier_leverage"]),
            particulate_split_rms=(
                None
                if value.get("particulate_split_rms") is None
                else float(value["particulate_split_rms"])
            ),
            reactor_residual_rms=(
                None
                if value.get("reactor_residual_rms") is None
                else float(value["reactor_residual_rms"])
            ),
        )


@dataclass(frozen=True)
class SharedUnitTrustCalibration:
    passed: bool
    reason: str
    limits: SharedUnitTrustLimits | None
    leverage: SharedUnitLeverageContract
    development_values: FloatArray
    development_columns: tuple[str, ...]
    out_of_fold_raw: FloatArray
    out_of_fold_projected: FloatArray
    closure_accepted: BoolArray
    projection_accepted: BoolArray
    closure_diagnostics: tuple[SharedUnitClosureDiagnostics, ...]
    full_raw_nrmse: float | None
    inventory_raw_nrmse: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "reason": self.reason,
            "limits": None if self.limits is None else self.limits.as_dict(),
            "leverage": self.leverage.as_dict(),
            "development_columns": list(self.development_columns),
            "plant_count": int(self.closure_accepted.size),
            "closure_accepted_count": int(np.count_nonzero(self.closure_accepted)),
            "projection_accepted_count": int(np.count_nonzero(self.projection_accepted)),
            "full_raw_nrmse": _optional_float(self.full_raw_nrmse),
            "inventory_raw_nrmse": _optional_float(self.inventory_raw_nrmse),
        }


def _local_leverages(
    models: SharedUnitModels,
    leverage: SharedUnitLeverageContract,
    theta: FloatArray,
    closure: SharedUnitClosureResult,
) -> tuple[FloatArray, float]:
    if not closure.accepted or closure.mixer is None or closure.reactors is None:
        raise SharedUnitError("local leverage requires an accepted recycle closure")
    q_process = 1.0 + theta[4] + theta[5]
    dilution = 120.0 * q_process / theta[0]
    aeration = np.asarray([0.0, 0.0, theta[1], theta[2], theta[3]])
    upstream = np.vstack((closure.mixer, closure.reactors[:-1]))
    reactor_decisions = np.column_stack((np.full(5, dilution), aeration))
    reactor_features = np.asarray(
        models.reactor.feature_map.transform(reactor_decisions, upstream), dtype=float
    )
    reactor_values = np.einsum(
        "ij,jk,ik->i", reactor_features, leverage.reactor_precision, reactor_features
    )
    clarifier_features = np.asarray(
        models.clarifier.feature_map.transform(theta[[5, 6]], closure.reactors[-1]),
        dtype=float,
    )
    clarifier_value = float(
        clarifier_features @ leverage.clarifier_precision @ clarifier_features
    )
    if np.min(reactor_values) < -1.0e-10 or clarifier_value < -1.0e-10:
        raise SharedUnitError("local leverage violated its PSD numerical contract")
    return np.maximum(reactor_values, 0.0), max(clarifier_value, 0.0)


def _rms_callback(
    callback: TrustRows | None,
    theta: FloatArray,
    raw: FloatArray,
    projected: FloatArray,
    influent: FloatArray,
) -> float | None:
    if callback is None:
        return None
    rows = np.asarray(callback(theta, raw, projected, influent), dtype=np.float64).reshape(-1)
    if rows.size == 0 or not np.all(np.isfinite(rows)):
        raise SharedUnitError("route-U trust callback returned invalid scaled rows")
    return float(np.sqrt(np.mean(np.square(rows))))


def evaluate_shared_unit_trust(
    models: SharedUnitModels,
    leverage: SharedUnitLeverageContract,
    limits: SharedUnitTrustLimits,
    theta: npt.ArrayLike,
    influent: npt.ArrayLike,
    closure: SharedUnitClosureResult,
    projected: npt.ArrayLike,
    common_response_scale: npt.ArrayLike,
    *,
    split_rows: TrustRows | None = None,
    reactor_rows: TrustRows | None = None,
) -> SharedUnitTrustValues:
    controls = _finite_vector(theta, 7, "theta")
    feed = _finite_vector(influent, models.component_count, "influent")
    if not closure.accepted or closure.raw is None:
        raise SharedUnitError("trust diagnostics require an accepted route-U closure")
    state = _finite_vector(projected, closure.raw.size, "projected response")
    scale = _finite_vector(
        common_response_scale, closure.raw.size, "common_response_scale"
    )
    correction = float(np.sqrt(np.mean(np.square((state - closure.raw) / scale))))
    reactor_leverage, clarifier_leverage = _local_leverages(
        models, leverage, controls, closure
    )
    split = _rms_callback(split_rows, controls, closure.raw, state, feed)
    reactor = _rms_callback(reactor_rows, controls, closure.raw, state, feed)
    values = SharedUnitTrustValues(
        correction_rms=correction,
        reactor_leverage=reactor_leverage,
        clarifier_leverage=clarifier_leverage,
        particulate_split_rms=split,
        reactor_residual_rms=reactor,
    )
    # Validate that value/limit optional families agree now, not during optimization.
    values.constraint_rows(limits)
    return values


_SHARED_CALIBRATION_CONTEXT: tuple[
    tuple[SharedUnitModels, ...],
    SharedUnitModels,
    np.ndarray,
    SharedUnitLeverageContract,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    NetworkRowScales,
    NetworkLayout,
    np.ndarray,
    np.ndarray,
    EngineeringLimits,
    TrustRows | None,
    TrustRows | None,
] | None = None


def _initialize_shared_calibration_worker(
    fold_models: tuple[SharedUnitModels, ...],
    final_models: SharedUnitModels,
    fold_membership: np.ndarray,
    leverage: SharedUnitLeverageContract,
    theta: np.ndarray,
    feed: np.ndarray,
    scale: np.ndarray,
    row_scales: NetworkRowScales,
    layout: NetworkLayout,
    invariant_operator: np.ndarray,
    tss_weights: np.ndarray,
    engineering: EngineeringLimits,
    split_rows: TrustRows | None,
    reactor_rows: TrustRows | None,
) -> None:
    global _SHARED_CALIBRATION_CONTEXT
    _SHARED_CALIBRATION_CONTEXT = (
        fold_models,
        final_models,
        np.asarray(fold_membership, dtype=np.int64),
        leverage,
        np.asarray(theta, dtype=np.float64),
        np.asarray(feed, dtype=np.float64),
        np.asarray(scale, dtype=np.float64),
        row_scales,
        layout,
        np.asarray(invariant_operator, dtype=np.float64),
        np.asarray(tss_weights, dtype=np.float64),
        engineering,
        split_rows,
        reactor_rows,
    )


def _shared_calibration_batch(
    bounds: tuple[int, int],
) -> Mapping[str, np.ndarray]:
    if _SHARED_CALIBRATION_CONTEXT is None:
        raise RuntimeError("shared calibration worker was not initialized")
    (
        fold_models,
        final_models,
        fold_membership,
        leverage,
        theta,
        feed,
        scale,
        row_scales,
        layout,
        invariant_operator,
        tss_weights,
        engineering,
        split_rows,
        reactor_rows,
    ) = _SHARED_CALIBRATION_CONTEXT
    start, stop = bounds
    count = stop - start
    value_count = 7 + int(split_rows is not None) + int(reactor_rows is not None)
    raw = np.full((count, layout.state_size), np.nan, dtype=np.float64)
    projected = np.full_like(raw, np.nan)
    values = np.full((count, value_count), np.nan, dtype=np.float64)
    closure_accepted = np.zeros(count, dtype=bool)
    projection_accepted = np.zeros(count, dtype=bool)
    diagnostics: list[str] = []
    for local, row in enumerate(range(start, stop)):
        fold = int(fold_membership[row])
        closure = solve_shared_unit_closure(
            fold_models[fold - 1],
            theta[row],
            feed[row],
            scale,
            layout=layout,
        )
        diagnostics.append(json.dumps(closure.diagnostics.as_dict()))
        closure_accepted[local] = closure.accepted
        if not closure.accepted or closure.raw is None:
            continue
        raw[local] = closure.raw
        projection = project_shared_unit_raw(
            closure.raw,
            theta[row],
            feed[row],
            scale,
            row_scales,
            layout=layout,
            invariant_operator=invariant_operator,
            tss_weights=tss_weights,
            clarifier_volume_m3=engineering.clarifier_volume_m3,
            raise_on_failure=False,
        )
        projection_accepted[local] = projection.accepted
        if not projection.accepted or not np.all(np.isfinite(projection.state)):
            continue
        projected[local] = projection.state
        correction = float(
            np.sqrt(np.mean(np.square((projection.state - closure.raw) / scale)))
        )
        reactor_leverage, clarifier_leverage = _local_leverages(
            final_models, leverage, theta[row], closure
        )
        row_values = [
            correction,
            *reactor_leverage.tolist(),
            clarifier_leverage,
        ]
        split = _rms_callback(
            split_rows, theta[row], closure.raw, projection.state, feed[row]
        )
        reactor = _rms_callback(
            reactor_rows, theta[row], closure.raw, projection.state, feed[row]
        )
        if split_rows is not None:
            assert split is not None
            row_values.append(split)
        if reactor_rows is not None:
            assert reactor is not None
            row_values.append(reactor)
        values[local] = row_values
    return {
        "raw": raw,
        "projected": projected,
        "values": values,
        "closure_accepted": closure_accepted,
        "projection_accepted": projection_accepted,
        "diagnostics_json": np.asarray(diagnostics),
    }


def _validate_shared_calibration_batch(
    start: int, stop: int, payload: Mapping[str, np.ndarray],
) -> None:
    count = stop - start
    raw = np.asarray(payload["raw"])
    projected = np.asarray(payload["projected"])
    values = np.asarray(payload["values"])
    closure = np.asarray(payload["closure_accepted"])
    projection = np.asarray(payload["projection_accepted"])
    diagnostics = np.asarray(payload["diagnostics_json"])
    if (
        raw.ndim != 2
        or raw.shape[0] != count
        or projected.shape != raw.shape
        or values.ndim != 2
        or values.shape[0] != count
        or closure.shape != (count,)
        or projection.shape != (count,)
        or closure.dtype.kind != "b"
        or projection.dtype.kind != "b"
        or diagnostics.shape != (count,)
        or np.any(projection & ~closure)
    ):
        raise ValueError("shared calibration batch payload has invalid dimensions")
    finite_raw = np.all(np.isfinite(raw), axis=1)
    finite_projected = np.all(np.isfinite(projected), axis=1)
    finite_values = np.all(np.isfinite(values), axis=1)
    if (
        np.any(closure != finite_raw)
        or np.any(projection != finite_projected)
        or np.any(projection != finite_values)
    ):
        raise ValueError("shared calibration batch finite-state contract failed")
    for local, value in enumerate(diagnostics.tolist()):
        restored = SharedUnitClosureDiagnostics.from_dict(json.loads(str(value)))
        if bool(restored.accepted) != bool(closure[local]):
            raise ValueError("shared calibration diagnostic acceptance is inconsistent")


def calibrate_shared_unit_trust(
    fit: SharedUnitFitResult,
    training: SharedUnitTrainingData,
    decisions: npt.ArrayLike,
    influents: npt.ArrayLike,
    targets: npt.ArrayLike,
    common_response_scale: npt.ArrayLike,
    row_scales: NetworkRowScales,
    *,
    layout: NetworkLayout | None = None,
    invariant_operator: npt.ArrayLike = INVARIANT_MATRIX,
    tss_weights: npt.ArrayLike = TSS_VECTOR,
    engineering: EngineeringLimits | None = None,
    split_rows: TrustRows | None = None,
    reactor_rows: TrustRows | None = None,
    parallel_workers: int = 1,
    batch_size: int = 64,
    checkpoint_directory: Path | None = None,
    checkpoint_contract: str | None = None,
    progress: Callable[[BatchProgress], None] | None = None,
) -> SharedUnitTrustCalibration:
    """Run fold-specific free closure/projection and freeze route-U limits.

    The expensive row kernel can run in deterministic process batches.  Batch
    checkpoints contain the complete row result needed for the unchanged
    population reductions below, so an interrupted calibration resumes only
    missing batches.
    """

    layout = layout or NetworkLayout(component_count=training.component_count)
    theta = _finite_matrix(decisions, 7, "development decisions", rows=training.plant_count)
    feed = _finite_matrix(
        influents,
        layout.component_count,
        "development influents",
        rows=training.plant_count,
    )
    truth = _finite_matrix(
        targets, layout.state_size, "development targets", rows=training.plant_count
    )
    scale = _finite_vector(
        common_response_scale, layout.state_size, "common_response_scale"
    )
    if len(fit.fold_models) != int(np.max(fit.plant_fold_membership)):
        raise ValueError("selected fold models do not cover plant membership")
    leverage = fit_shared_unit_leverage(fit.models, training)
    value_columns = [
        "correction",
        *(f"reactor_leverage_{stage}" for stage in range(1, 6)),
        "clarifier_leverage",
    ]
    split_column: int | None = None
    reactor_column: int | None = None
    if split_rows is not None:
        split_column = len(value_columns)
        value_columns.append("particulate_split")
    if reactor_rows is not None:
        reactor_column = len(value_columns)
        value_columns.append("reactor_residual")
    limits_config = engineering or EngineeringLimits()
    if checkpoint_directory is not None and not checkpoint_contract:
        raise ValueError(
            "checkpoint_contract is required when shared calibration checkpoints "
            "are enabled"
        )
    batches = run_resumable_batches(
        stage="shared_unit_development_calibration",
        row_count=training.plant_count,
        batch_size=batch_size,
        parallel_workers=parallel_workers,
        checkpoint_directory=checkpoint_directory,
        contract_digest=checkpoint_contract or "unpersisted",
        payload_names=(
            "raw",
            "projected",
            "values",
            "closure_accepted",
            "projection_accepted",
            "diagnostics_json",
        ),
        worker=_shared_calibration_batch,
        validate=_validate_shared_calibration_batch,
        initializer=_initialize_shared_calibration_worker,
        initargs=(
            fit.fold_models,
            fit.models,
            fit.plant_fold_membership,
            leverage,
            theta,
            feed,
            scale,
            row_scales,
            layout,
            np.asarray(invariant_operator, dtype=np.float64),
            np.asarray(tss_weights, dtype=np.float64),
            limits_config,
            split_rows,
            reactor_rows,
        ),
        progress=progress,
    )
    raw = np.vstack([batch["raw"] for batch in batches])
    projected = np.vstack([batch["projected"] for batch in batches])
    values = np.vstack([batch["values"] for batch in batches])
    closure_accepted = np.concatenate([
        batch["closure_accepted"].astype(bool, copy=False) for batch in batches
    ])
    projection_accepted = np.concatenate([
        batch["projection_accepted"].astype(bool, copy=False) for batch in batches
    ])
    diagnostics = [
        SharedUnitClosureDiagnostics.from_dict(json.loads(str(record)))
        for batch in batches for record in batch["diagnostics_json"]
    ]
    finite_rows = (
        closure_accepted
        & projection_accepted
        & np.all(np.isfinite(raw), axis=1)
        & np.all(np.isfinite(projected), axis=1)
        & np.all(np.isfinite(values), axis=1)
    )
    complete = bool(np.all(finite_rows))
    full_nrmse: float | None = None
    inventory_nrmse: float | None = None
    frozen_limits: SharedUnitTrustLimits | None = None
    reasons: list[str] = []
    if not np.all(closure_accepted):
        reasons.append("out_of_fold_closure_failed")
    if not np.all(projection_accepted):
        reasons.append("out_of_fold_projection_failed")
    if not np.all(np.all(np.isfinite(values), axis=1)):
        reasons.append("out_of_fold_trust_values_failed")
    if np.any(finite_rows):
        scaled_error = (raw[finite_rows] - truth[finite_rows]) / scale
        full_nrmse = float(np.sqrt(np.mean(np.square(scaled_error))))
        inventory_nrmse = float(
            np.sqrt(np.mean(np.square(scaled_error[:, layout.inventory_index])))
        )
        frozen_limits = SharedUnitTrustLimits(
            correction_rms=_nearest_rank_95(values[finite_rows, 0]),
            reactor_leverage=leverage.reactor_limit,
            clarifier_leverage=leverage.clarifier_limit,
            particulate_split_rms=(
                _nearest_rank_95(values[finite_rows, split_column])
                if split_column is not None
                else None
            ),
            reactor_residual_rms=(
                _nearest_rank_95(values[finite_rows, reactor_column])
                if reactor_column is not None
                else None
            ),
        )
        if full_nrmse >= 1.0:
            reasons.append("full_raw_nrmse_failed")
        if inventory_nrmse >= 1.0:
            reasons.append("inventory_raw_nrmse_failed")
        if frozen_limits.correction_rms > 0.50:
            reasons.append("correction_limit_failed")
    else:
        reasons.append("trust_limits_not_frozen")
    passed = complete and not reasons
    return SharedUnitTrustCalibration(
        passed=passed,
        reason="passed" if passed else ";".join(dict.fromkeys(reasons)),
        limits=frozen_limits,
        leverage=leverage,
        development_values=values,
        development_columns=tuple(value_columns),
        out_of_fold_raw=raw,
        out_of_fold_projected=projected,
        closure_accepted=closure_accepted,
        projection_accepted=projection_accepted,
        closure_diagnostics=tuple(diagnostics),
        full_raw_nrmse=full_nrmse,
        inventory_raw_nrmse=inventory_nrmse,
    )


@dataclass(frozen=True)
class SharedUnitAssets:
    """Frozen data needed by every route-U case evaluation."""

    models: SharedUnitModels
    layout: NetworkLayout
    common_response_scale: FloatArray
    row_scales: NetworkRowScales
    invariant_operator: FloatArray
    tss_weights: FloatArray
    leverage: SharedUnitLeverageContract
    trust_limits: SharedUnitTrustLimits
    quality_operator: FloatArray
    quality_scale: FloatArray
    split_rows: TrustRows | None = None
    reactor_rows: TrustRows | None = None
    engineering: EngineeringLimits = field(default_factory=EngineeringLimits)
    theta_lower: FloatArray = field(default_factory=lambda: DECISION_LOWER.copy())
    theta_upper: FloatArray = field(default_factory=lambda: DECISION_UPPER.copy())

    def __post_init__(self) -> None:
        layout = self.layout
        if layout.stage_count != 5 or layout.component_count != self.models.component_count:
            raise ValueError("route-U assets require five stages matching the local models")
        scale = _finite_vector(
            self.common_response_scale, layout.state_size, "common_response_scale"
        )
        if np.any(scale <= 0.0):
            raise ValueError("common_response_scale must be strictly positive")
        invariant = np.asarray(self.invariant_operator, dtype=np.float64)
        if (
            invariant.ndim != 2
            or invariant.shape[1] != layout.component_count
            or invariant.shape[0] < 1
            or not np.all(np.isfinite(invariant))
            or np.linalg.matrix_rank(invariant) != invariant.shape[0]
        ):
            raise ValueError("invariant_operator must be finite and full row rank")
        tss = _finite_vector(self.tss_weights, layout.component_count, "tss_weights")
        if np.any(tss < 0.0) or not np.any(tss > 0.0):
            raise ValueError("tss_weights must be nonnegative and nonzero")
        equality_count = (
            2 * layout.component_count
            + layout.stage_count * invariant.shape[0]
            + len(layout.soluble_indices)
        )
        equality_scale = _finite_vector(
            self.row_scales.equality, equality_count, "equality row scales"
        )
        inequality_scale = _finite_vector(
            self.row_scales.inequality,
            layout.inequality_count,
            "inequality row scales",
        )
        if np.any(equality_scale <= 0.0) or np.any(inequality_scale <= 0.0):
            raise ValueError("projection row scales must be strictly positive")
        quality = np.asarray(self.quality_operator, dtype=np.float64)
        if (
            quality.ndim != 2
            or quality.shape[0] < 1
            or quality.shape[1] != layout.component_count
            or not np.all(np.isfinite(quality))
        ):
            raise ValueError("quality_operator has inconsistent component dimensions")
        quality_scale = _finite_vector(
            self.quality_scale, quality.shape[0], "quality_scale"
        )
        if np.any(quality_scale <= 0.0):
            raise ValueError("quality_scale must be strictly positive")
        lower = _finite_vector(self.theta_lower, 7, "theta_lower")
        upper = _finite_vector(self.theta_upper, 7, "theta_upper")
        if np.any(upper <= lower):
            raise ValueError("every route-U control must have a positive span")
        if self.leverage.reactor_precision.shape != (
            self.models.reactor.feature_map.feature_count,
            self.models.reactor.feature_map.feature_count,
        ) or self.leverage.clarifier_precision.shape != (
            self.models.clarifier.feature_map.feature_count,
            self.models.clarifier.feature_map.feature_count,
        ):
            raise ValueError("route-U leverage matrices do not match local features")
        if (self.split_rows is None) != (self.trust_limits.particulate_split_rms is None):
            raise ValueError("split callback and frozen limit must be supplied together")
        if (self.reactor_rows is None) != (self.trust_limits.reactor_residual_rms is None):
            raise ValueError("reactor callback and frozen limit must be supplied together")
        object.__setattr__(self, "common_response_scale", scale)
        object.__setattr__(self, "row_scales", NetworkRowScales(equality_scale, inequality_scale))
        object.__setattr__(self, "invariant_operator", invariant.copy())
        object.__setattr__(self, "tss_weights", tss)
        object.__setattr__(self, "quality_operator", quality.copy())
        object.__setattr__(self, "quality_scale", quality_scale)
        object.__setattr__(self, "theta_lower", lower)
        object.__setattr__(self, "theta_upper", upper)

    @property
    def theta_span(self) -> FloatArray:
        return self.theta_upper - self.theta_lower

    @property
    def engineering_row_names(self) -> tuple[str, ...]:
        names = [
            "srt_lower",
            "srt_upper",
            "external_solids_loss_guard",
            "slr_upper",
            "underflow_tss_upper",
            "feed_tss_lower",
        ]
        if self.engineering.sor_upper_m_d is not None:
            names.append("sor_upper")
        return tuple(names)

    @property
    def trust_row_names(self) -> tuple[str, ...]:
        dummy = SharedUnitTrustValues(
            correction_rms=0.0,
            reactor_leverage=np.zeros(self.layout.stage_count),
            clarifier_leverage=0.0,
            particulate_split_rms=0.0 if self.split_rows is not None else None,
            reactor_residual_rms=0.0 if self.reactor_rows is not None else None,
        )
        return dummy.constraint_rows(self.trust_limits)[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": self.models.as_dict(),
            "layout_state_size": self.layout.state_size,
            "leverage": self.leverage.as_dict(),
            "trust_limits": self.trust_limits.as_dict(),
            "engineering_rows": list(self.engineering_row_names),
            "trust_rows": list(self.trust_row_names),
        }


def build_shared_unit_assets(
    models: SharedUnitModels,
    common_response_scale: npt.ArrayLike,
    row_scales: NetworkRowScales,
    calibration: SharedUnitTrustCalibration,
    *,
    layout: NetworkLayout | None = None,
    invariant_operator: npt.ArrayLike = INVARIANT_MATRIX,
    tss_weights: npt.ArrayLike = TSS_VECTOR,
    quality_operator: npt.ArrayLike = COMPOSITE_MATRIX,
    quality_scale: npt.ArrayLike | None = None,
    split_rows: TrustRows | None = None,
    reactor_rows: TrustRows | None = None,
    engineering: EngineeringLimits | None = None,
    theta_lower: npt.ArrayLike = DECISION_LOWER,
    theta_upper: npt.ArrayLike = DECISION_UPPER,
) -> SharedUnitAssets:
    """Bind final local fits to common scales, projection, and frozen trust."""

    if calibration.limits is None:
        raise SharedUnitError("route-U trust limits were not successfully frozen")
    layout = layout or NetworkLayout(component_count=models.component_count)
    quality = np.asarray(quality_operator, dtype=np.float64)
    resolved_quality_scale = (
        np.ones(quality.shape[0], dtype=np.float64)
        if quality_scale is None
        else np.asarray(quality_scale, dtype=np.float64)
    )
    return SharedUnitAssets(
        models=models,
        layout=layout,
        common_response_scale=np.asarray(common_response_scale, dtype=float),
        row_scales=row_scales,
        invariant_operator=np.asarray(invariant_operator, dtype=float),
        tss_weights=np.asarray(tss_weights, dtype=float),
        leverage=calibration.leverage,
        trust_limits=calibration.limits,
        quality_operator=quality,
        quality_scale=resolved_quality_scale,
        split_rows=split_rows,
        reactor_rows=reactor_rows,
        engineering=engineering or EngineeringLimits(),
        theta_lower=np.asarray(theta_lower, dtype=float),
        theta_upper=np.asarray(theta_upper, dtype=float),
    )


@dataclass(frozen=True)
class SharedUnitCase:
    influent: FloatArray
    case_id: str = "nominal"
    objective_weights: FloatArray = field(
        default_factory=lambda: DEFAULT_OBJECTIVE_WEIGHTS.copy()
    )
    quality_weights: FloatArray | None = None

    def validated(self, assets: SharedUnitAssets) -> tuple[FloatArray, FloatArray, FloatArray]:
        influent = _finite_vector(
            self.influent, assets.layout.component_count, "case influent"
        )
        objective = _finite_vector(self.objective_weights, 6, "objective_weights")
        if np.any(objective < 0.0) or not np.isclose(
            np.sum(objective), 1.0, atol=1.0e-12
        ):
            raise ValueError("objective_weights must be nonnegative and sum to one")
        if self.quality_weights is None:
            quality = np.full(
                assets.quality_operator.shape[0],
                1.0 / assets.quality_operator.shape[0],
            )
        else:
            quality = _finite_vector(
                self.quality_weights,
                assets.quality_operator.shape[0],
                "quality_weights",
            )
            if np.any(quality < 0.0) or not np.isclose(
                np.sum(quality), 1.0, atol=1.0e-12
            ):
                raise ValueError("quality_weights must be nonnegative and sum to one")
        return influent, objective, quality


def _engineering_values(
    assets: SharedUnitAssets,
    theta: FloatArray,
    state: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    layout = assets.layout
    limits = assets.engineering
    final = state[layout.reactor_slice(layout.stage_count - 1)]
    overflow = state[layout.overflow_flow_slice]
    underflow = state[layout.underflow_flow_slice]
    inventory_clarifier = float(state[layout.inventory_index])
    returned, waste = theta[5], theta[6]
    q_underflow = returned + waste
    q_effluent = 1.0 - waste
    q_clarifier = 1.0 + returned
    feed_tss = float(assets.tss_weights @ final)
    underflow_tss = float(assets.tss_weights @ underflow / q_underflow)
    external_loss = float(
        assets.tss_weights @ overflow
        + waste * (assets.tss_weights @ underflow) / q_underflow
    )
    stage_volume = (
        limits.fresh_flow_m3_d * theta[0] / (24.0 * layout.stage_count)
    )
    reactor_inventory = sum(
        stage_volume
        * float(assets.tss_weights @ state[layout.reactor_slice(stage)])
        for stage in range(layout.stage_count)
    )
    inventory = reactor_inventory + inventory_clarifier
    sor = limits.fresh_flow_m3_d * q_effluent / limits.clarifier_area_m2
    slr = (
        1.0e-3
        * limits.fresh_flow_m3_d
        * q_clarifier
        * feed_tss
        / limits.clarifier_area_m2
    )
    scale = limits.resolved_inventory_scale
    rows = [
        (limits.srt_lower_d * limits.fresh_flow_m3_d * external_loss - inventory)
        / scale,
        (inventory - limits.srt_upper_d * limits.fresh_flow_m3_d * external_loss)
        / scale,
        (limits.external_loss_min_g_m3 - external_loss)
        / limits.external_loss_min_g_m3,
        (slr - limits.slr_upper_kg_m2_d) / limits.slr_upper_kg_m2_d,
        (underflow_tss - limits.underflow_tss_upper_g_m3)
        / limits.underflow_tss_upper_g_m3,
        (limits.feed_tss_min_g_m3 - feed_tss) / limits.feed_tss_min_g_m3,
    ]
    if limits.sor_upper_m_d is not None:
        rows.append((sor - limits.sor_upper_m_d) / limits.sor_upper_m_d)
    srt = (
        inventory / (limits.fresh_flow_m3_d * external_loss)
        if external_loss > 0.0
        else np.inf
    )
    quantities = np.asarray(
        [srt, sor, slr, underflow_tss, feed_tss, external_loss, inventory],
        dtype=np.float64,
    )
    return np.asarray(rows, dtype=np.float64), quantities


def _objective_values(
    assets: SharedUnitAssets,
    theta: FloatArray,
    state: FloatArray,
    objective_weights: FloatArray,
    quality_weights: FloatArray,
) -> tuple[float, FloatArray]:
    q_effluent = 1.0 - theta[6]
    effluent = state[assets.layout.overflow_flow_slice] / q_effluent
    composites = assets.quality_operator @ effluent
    quality = float(quality_weights @ (composites / assets.quality_scale))
    hrt = (theta[0] - assets.theta_lower[0]) / assets.theta_span[0]
    aeration = theta[0] * float(np.sum(theta[1:4])) / (
        assets.theta_upper[0] * 3.0
    )
    internal = (theta[4] - assets.theta_lower[4]) / assets.theta_span[4]
    returned = (theta[5] - assets.theta_lower[5]) / assets.theta_span[5]
    q_underflow = theta[5] + theta[6]
    underflow_tss = float(
        assets.tss_weights @ state[assets.layout.underflow_flow_slice] / q_underflow
    )
    wasting = theta[6] * underflow_tss / (
        assets.theta_upper[6] * assets.engineering.underflow_tss_upper_g_m3
    )
    components = np.asarray(
        [quality, hrt, aeration, internal, returned, wasting], dtype=np.float64
    )
    return float(objective_weights @ components), components


@dataclass(frozen=True)
class SharedUnitEvaluation:
    available: bool
    reason: str
    case_id: str
    normalized_controls: FloatArray
    theta: FloatArray
    closure: SharedUnitClosureResult
    projection: ProjectionResult | None
    raw: FloatArray | None
    projected: FloatArray | None
    objective: float | None
    objective_components: FloatArray | None
    engineering_rows: FloatArray | None
    engineering_quantities: FloatArray | None
    trust: SharedUnitTrustValues | None
    trust_rows: FloatArray | None
    feasible: bool
    maximum_upper_residual: float
    elapsed_seconds: float
    root_seconds: float
    projection_seconds: float

    def as_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        projection = None
        if self.projection is not None:
            projection = {
                "accepted": bool(self.projection.accepted),
                "diagnostics": self.projection.diagnostics.as_dict(),
            }
            if include_arrays:
                projection.update(
                    {
                        "state": self.projection.state.tolist(),
                        "displacement": self.projection.displacement.tolist(),
                        "equality_multipliers": (
                            self.projection.equality_multipliers.tolist()
                        ),
                        "inequality_multipliers": (
                            self.projection.inequality_multipliers.tolist()
                        ),
                        "inequality_slack": self.projection.inequality_slack.tolist(),
                    }
                )
        result: dict[str, Any] = {
            "available": bool(self.available),
            "reason": self.reason,
            "case_id": self.case_id,
            "normalized_controls": self.normalized_controls.tolist(),
            "theta": self.theta.tolist(),
            "objective": _optional_float(self.objective),
            "feasible": bool(self.feasible),
            "maximum_upper_residual": float(self.maximum_upper_residual),
            "closure": self.closure.as_dict(include_arrays=include_arrays),
            "projection": projection,
            "trust": None if self.trust is None else self.trust.as_dict(),
            "elapsed_seconds": float(self.elapsed_seconds),
            "root_seconds": float(self.root_seconds),
            "projection_seconds": float(self.projection_seconds),
        }
        if include_arrays:
            result.update(
                {
                    "raw": _json_array(self.raw),
                    "projected": _json_array(self.projected),
                    "objective_components": _json_array(self.objective_components),
                    "engineering_rows": _json_array(self.engineering_rows),
                    "engineering_quantities": _json_array(
                        self.engineering_quantities
                    ),
                    "trust_rows": _json_array(self.trust_rows),
                }
            )
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SharedUnitEvaluation":
        """Restore the complete atomic case checkpoint emitted by ``as_dict``."""

        closure = SharedUnitClosureResult.from_dict(value["closure"])
        projection_value = value.get("projection")
        projection: ProjectionResult | None = None
        if projection_value is not None:
            required_arrays = (
                "state",
                "displacement",
                "equality_multipliers",
                "inequality_multipliers",
                "inequality_slack",
            )
            if any(name not in projection_value for name in required_arrays):
                raise ValueError(
                    "SharedUnitEvaluation.from_dict requires an array-inclusive "
                    "checkpoint"
                )
            projection = ProjectionResult(
                state=np.asarray(projection_value["state"], dtype=np.float64),
                displacement=np.asarray(
                    projection_value["displacement"], dtype=np.float64
                ),
                equality_multipliers=np.asarray(
                    projection_value["equality_multipliers"], dtype=np.float64
                ),
                inequality_multipliers=np.asarray(
                    projection_value["inequality_multipliers"], dtype=np.float64
                ),
                inequality_slack=np.asarray(
                    projection_value["inequality_slack"], dtype=np.float64
                ),
                diagnostics=ProjectionDiagnostics(
                    **dict(projection_value["diagnostics"])
                ),
                accepted=bool(projection_value["accepted"]),
            )
        trust_value = value.get("trust")
        return cls(
            available=bool(value["available"]),
            reason=str(value["reason"]),
            case_id=str(value["case_id"]),
            normalized_controls=np.asarray(
                value["normalized_controls"], dtype=np.float64
            ),
            theta=np.asarray(value["theta"], dtype=np.float64),
            closure=closure,
            projection=projection,
            raw=_optional_array(value.get("raw")),
            projected=_optional_array(value.get("projected")),
            objective=(
                None if value.get("objective") is None else float(value["objective"])
            ),
            objective_components=_optional_array(value.get("objective_components")),
            engineering_rows=_optional_array(value.get("engineering_rows")),
            engineering_quantities=_optional_array(
                value.get("engineering_quantities")
            ),
            trust=(
                None
                if trust_value is None
                else SharedUnitTrustValues.from_dict(trust_value)
            ),
            trust_rows=_optional_array(value.get("trust_rows")),
            feasible=bool(value["feasible"]),
            maximum_upper_residual=float(value["maximum_upper_residual"]),
            elapsed_seconds=float(value["elapsed_seconds"]),
            root_seconds=float(value["root_seconds"]),
            projection_seconds=float(value["projection_seconds"]),
        )


def evaluate_shared_unit(
    assets: SharedUnitAssets,
    case: SharedUnitCase,
    normalized_controls: npt.ArrayLike,
) -> SharedUnitEvaluation:
    """Cold-evaluate root, common QP, objective, engineering, and trust rows."""

    started = perf_counter()
    normalized = _finite_vector(normalized_controls, 7, "normalized_controls")
    tolerance = 1.0e-12
    if np.any(normalized < -tolerance) or np.any(normalized > 1.0 + tolerance):
        raise ValueError("normalized_controls must lie in [0, 1]")
    normalized = np.clip(normalized, 0.0, 1.0)
    theta = assets.theta_lower + assets.theta_span * normalized
    influent, objective_weights, quality_weights = case.validated(assets)
    root_started = perf_counter()
    closure = solve_shared_unit_closure(
        assets.models,
        theta,
        influent,
        assets.common_response_scale,
        layout=assets.layout,
    )
    root_seconds = perf_counter() - root_started
    if not closure.accepted or closure.raw is None:
        return SharedUnitEvaluation(
            available=False,
            reason=f"recycle_closure_unavailable:{closure.diagnostics.reason}",
            case_id=case.case_id,
            normalized_controls=normalized,
            theta=theta,
            closure=closure,
            projection=None,
            raw=None,
            projected=None,
            objective=None,
            objective_components=None,
            engineering_rows=None,
            engineering_quantities=None,
            trust=None,
            trust_rows=None,
            feasible=False,
            maximum_upper_residual=np.inf,
            elapsed_seconds=perf_counter() - started,
            root_seconds=root_seconds,
            projection_seconds=0.0,
        )
    projection_started = perf_counter()
    projection = project_shared_unit_raw(
        closure.raw,
        theta,
        influent,
        assets.common_response_scale,
        assets.row_scales,
        layout=assets.layout,
        invariant_operator=assets.invariant_operator,
        tss_weights=assets.tss_weights,
        clarifier_volume_m3=assets.engineering.clarifier_volume_m3,
        raise_on_failure=False,
    )
    projection_seconds = perf_counter() - projection_started
    projected = np.asarray(projection.state, dtype=np.float64)
    if not projection.accepted or not np.all(np.isfinite(projected)):
        return SharedUnitEvaluation(
            available=False,
            reason="projection_unavailable",
            case_id=case.case_id,
            normalized_controls=normalized,
            theta=theta,
            closure=closure,
            projection=projection,
            raw=closure.raw,
            projected=projected if np.all(np.isfinite(projected)) else None,
            objective=None,
            objective_components=None,
            engineering_rows=None,
            engineering_quantities=None,
            trust=None,
            trust_rows=None,
            feasible=False,
            maximum_upper_residual=np.inf,
            elapsed_seconds=perf_counter() - started,
            root_seconds=root_seconds,
            projection_seconds=projection_seconds,
        )
    trust = evaluate_shared_unit_trust(
        assets.models,
        assets.leverage,
        assets.trust_limits,
        theta,
        influent,
        closure,
        projected,
        assets.common_response_scale,
        split_rows=assets.split_rows,
        reactor_rows=assets.reactor_rows,
    )
    _, trust_rows = trust.constraint_rows(assets.trust_limits)
    engineering_rows, engineering_quantities = _engineering_values(
        assets, theta, projected
    )
    objective, components = _objective_values(
        assets,
        theta,
        projected,
        objective_weights,
        quality_weights,
    )
    arrays_finite = all(
        np.all(np.isfinite(value))
        for value in (
            projected,
            trust_rows,
            engineering_rows,
            engineering_quantities,
            objective,
            components,
        )
    )
    maximum = max(
        _maximum_positive(engineering_rows), _maximum_positive(trust_rows)
    )
    feasible = bool(arrays_finite and maximum <= 1.0e-6)
    return SharedUnitEvaluation(
        available=arrays_finite,
        reason="available" if arrays_finite else "nonfinite_upper_evaluation",
        case_id=case.case_id,
        normalized_controls=normalized,
        theta=theta,
        closure=closure,
        projection=projection,
        raw=closure.raw,
        projected=projected,
        objective=objective if arrays_finite else None,
        objective_components=components if arrays_finite else None,
        engineering_rows=engineering_rows if arrays_finite else None,
        engineering_quantities=engineering_quantities if arrays_finite else None,
        trust=trust if arrays_finite else None,
        trust_rows=trust_rows if arrays_finite else None,
        feasible=feasible,
        maximum_upper_residual=maximum if arrays_finite else np.inf,
        elapsed_seconds=perf_counter() - started,
        root_seconds=root_seconds,
        projection_seconds=projection_seconds,
    )


_SHARED_HOLDOUT_CONTEXT: tuple[
    SharedUnitAssets,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
] | None = None


def _initialize_shared_holdout_worker(
    assets: SharedUnitAssets,
    theta: np.ndarray,
    feed: np.ndarray,
    truth: np.ndarray,
    normalized: np.ndarray,
) -> None:
    global _SHARED_HOLDOUT_CONTEXT
    _SHARED_HOLDOUT_CONTEXT = (
        assets,
        np.asarray(theta, dtype=np.float64),
        np.asarray(feed, dtype=np.float64),
        np.asarray(truth, dtype=np.float64),
        np.asarray(normalized, dtype=np.float64),
    )


def _shared_holdout_batch(bounds: tuple[int, int]) -> Mapping[str, np.ndarray]:
    if _SHARED_HOLDOUT_CONTEXT is None:
        raise RuntimeError("shared holdout worker was not initialized")
    assets, theta, feed, truth, normalized = _SHARED_HOLDOUT_CONTEXT
    start, stop = bounds
    count = stop - start
    raw = np.full((count, assets.layout.state_size), np.nan, dtype=np.float64)
    projected = np.full_like(raw, np.nan)
    projected_targets = np.full_like(raw, np.nan)
    available = np.zeros(count, dtype=bool)
    target_accepted = np.zeros(count, dtype=bool)
    target_elapsed_ns = np.zeros(count, dtype=np.int64)
    evaluation_json: list[str] = []
    target_diagnostics_json: list[str] = []
    for local, row in enumerate(range(start, stop)):
        evaluation = evaluate_shared_unit(
            assets,
            SharedUnitCase(influent=feed[row], case_id=f"test_{row:06d}"),
            normalized[row],
        )
        valid = bool(
            evaluation.available
            and evaluation.raw is not None
            and evaluation.projected is not None
            and np.all(np.isfinite(evaluation.raw))
            and np.all(np.isfinite(evaluation.projected))
        )
        available[local] = valid
        if valid:
            raw[local] = np.asarray(evaluation.raw, dtype=np.float64)
            projected[local] = np.asarray(evaluation.projected, dtype=np.float64)
        evaluation_json.append(json.dumps(evaluation.as_dict(include_arrays=False)))

        target_started = perf_counter_ns()
        target_projection = project_shared_unit_raw(
            truth[row],
            theta[row],
            feed[row],
            assets.common_response_scale,
            assets.row_scales,
            layout=assets.layout,
            invariant_operator=assets.invariant_operator,
            tss_weights=assets.tss_weights,
            clarifier_volume_m3=assets.engineering.clarifier_volume_m3,
            raise_on_failure=False,
        )
        target_elapsed_ns[local] = perf_counter_ns() - target_started
        target_valid = bool(
            target_projection.accepted
            and np.all(np.isfinite(target_projection.state))
        )
        target_accepted[local] = target_valid
        if target_valid:
            projected_targets[local] = target_projection.state
        target_diagnostics_json.append(json.dumps({
            "accepted": bool(target_projection.accepted),
            "diagnostics": target_projection.diagnostics.as_dict(),
        }))
    return {
        "raw": raw,
        "projected": projected,
        "projected_targets": projected_targets,
        "available": available,
        "target_accepted": target_accepted,
        "target_elapsed_ns": target_elapsed_ns,
        "evaluation_json": np.asarray(evaluation_json),
        "target_diagnostics_json": np.asarray(target_diagnostics_json),
    }


def _validate_shared_holdout_batch(
    start: int, stop: int, payload: Mapping[str, np.ndarray],
) -> None:
    count = stop - start
    raw = np.asarray(payload["raw"])
    projected = np.asarray(payload["projected"])
    projected_targets = np.asarray(payload["projected_targets"])
    available = np.asarray(payload["available"])
    target_accepted = np.asarray(payload["target_accepted"])
    elapsed = np.asarray(payload["target_elapsed_ns"])
    evaluation_json = np.asarray(payload["evaluation_json"])
    target_json = np.asarray(payload["target_diagnostics_json"])
    if (
        raw.ndim != 2
        or raw.shape[0] != count
        or projected.shape != raw.shape
        or projected_targets.shape != raw.shape
        or available.shape != (count,)
        or available.dtype.kind != "b"
        or target_accepted.shape != (count,)
        or target_accepted.dtype.kind != "b"
        or elapsed.shape != (count,)
        or elapsed.dtype.kind not in "iu"
        or np.any(elapsed < 0)
        or evaluation_json.shape != (count,)
        or target_json.shape != (count,)
    ):
        raise ValueError("shared holdout batch payload has invalid dimensions")
    if (
        np.any(np.all(np.isfinite(raw), axis=1) != available)
        or np.any(np.all(np.isfinite(projected), axis=1) != available)
        or np.any(np.all(np.isfinite(projected_targets), axis=1) != target_accepted)
    ):
        raise ValueError("shared holdout batch finite-state contract failed")
    for local, value in enumerate(evaluation_json.tolist()):
        record = json.loads(str(value))
        expected_case = f"test_{start + local:06d}"
        if (
            bool(record.get("available")) != bool(available[local])
            or record.get("case_id") != expected_case
            or bool(record.get("closure", {}).get("accepted"))
            != bool(record.get("closure", {}).get("diagnostics", {}).get("accepted"))
        ):
            raise ValueError("shared holdout availability record is inconsistent")
    for local, value in enumerate(target_json.tolist()):
        record = json.loads(str(value))
        if bool(record.get("accepted")) != bool(target_accepted[local]):
            raise ValueError("shared holdout target record is inconsistent")


def evaluate_shared_unit_holdout_batches(
    assets: SharedUnitAssets,
    decisions: npt.ArrayLike,
    influents: npt.ArrayLike,
    mechanistic: npt.ArrayLike,
    *,
    parallel_workers: int = 1,
    batch_size: int = 64,
    checkpoint_directory: Path | None = None,
    checkpoint_contract: str | None = None,
    progress: Callable[[BatchProgress], None] | None = None,
) -> list[dict[str, np.ndarray]]:
    """Evaluate route U and both holdout QPs in resumable process batches."""

    theta = _finite_matrix(decisions, 7, "holdout decisions")
    feed = _finite_matrix(
        influents,
        assets.layout.component_count,
        "holdout influents",
        rows=len(theta),
    )
    truth = _finite_matrix(
        mechanistic,
        assets.layout.state_size,
        "holdout mechanistic targets",
        rows=len(theta),
    )
    normalized = (theta - assets.theta_lower) / assets.theta_span
    if checkpoint_directory is not None and not checkpoint_contract:
        raise ValueError(
            "checkpoint_contract is required when shared holdout checkpoints are "
            "enabled"
        )
    return run_resumable_batches(
        stage="shared_unit_holdout_projection_audit",
        row_count=len(theta),
        batch_size=batch_size,
        parallel_workers=parallel_workers,
        checkpoint_directory=checkpoint_directory,
        contract_digest=checkpoint_contract or "unpersisted",
        payload_names=(
            "raw",
            "projected",
            "projected_targets",
            "available",
            "target_accepted",
            "target_elapsed_ns",
            "evaluation_json",
            "target_diagnostics_json",
        ),
        worker=_shared_holdout_batch,
        validate=_validate_shared_holdout_batch,
        initializer=_initialize_shared_holdout_worker,
        initargs=(assets, theta, feed, truth, normalized),
        progress=progress,
    )


@dataclass(frozen=True)
class SharedUnitOptimizationSettings:
    maximum_iterations: int = 250
    maximum_function_evaluations: int = 250
    initial_trust_region_radius: float = 0.25
    final_trust_region_radius: float = 1.0e-6
    feasibility_tolerance: float = 1.0e-6
    poll_radii: tuple[float, float] = (1.0e-3, 1.0e-4)
    absolute_decrease_tolerance: float = 1.0e-8
    relative_decrease_tolerance: float = 1.0e-8
    direction_rank_tolerance: float = 1.0e-10
    maximum_poll_evaluations: int = 10_000
    acceleration_growth_factor: float = 2.0
    maximum_acceleration_probes: int = 16

    def __post_init__(self) -> None:
        if self.maximum_iterations < 1 or self.maximum_function_evaluations < 1:
            raise ValueError("COBYQA budgets must be positive")
        if self.maximum_poll_evaluations < 1 or self.maximum_acceleration_probes < 0:
            raise ValueError("poll budgets must be nonnegative and finite")
        positive = (
            self.initial_trust_region_radius,
            self.final_trust_region_radius,
            self.feasibility_tolerance,
            *self.poll_radii,
            self.absolute_decrease_tolerance,
            self.relative_decrease_tolerance,
            self.direction_rank_tolerance,
            self.acceleration_growth_factor,
        )
        if not all(np.isfinite(item) and item > 0.0 for item in positive):
            raise ValueError("optimization tolerances and radii must be positive and finite")
        if self.initial_trust_region_radius <= self.final_trust_region_radius:
            raise ValueError("initial trust-region radius must exceed final radius")
        if len(self.poll_radii) != 2 or self.poll_radii[0] <= self.poll_radii[1]:
            raise ValueError("poll_radii must contain descending coarse and fine radii")
        if self.acceleration_growth_factor <= 1.0:
            raise ValueError("acceleration_growth_factor must exceed one")


@dataclass(frozen=True)
class SharedUnitPollLevel:
    radius: float
    direction_count: int
    attempted_count: int
    unavailable_count: int
    feasible_nonzero_count: int
    feasible_direction_rank: int
    complete: bool
    no_descent: bool
    accepted_move: bool
    center_objective: float
    best_trial_objective: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "radius": float(self.radius),
            "direction_count": int(self.direction_count),
            "attempted_count": int(self.attempted_count),
            "unavailable_count": int(self.unavailable_count),
            "feasible_nonzero_count": int(self.feasible_nonzero_count),
            "feasible_direction_rank": int(self.feasible_direction_rank),
            "complete": bool(self.complete),
            "no_descent": bool(self.no_descent),
            "accepted_move": bool(self.accepted_move),
            "center_objective": float(self.center_objective),
            "best_trial_objective": _optional_float(self.best_trial_objective),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SharedUnitPollLevel":
        return cls(
            radius=float(value["radius"]),
            direction_count=int(value["direction_count"]),
            attempted_count=int(value["attempted_count"]),
            unavailable_count=int(value["unavailable_count"]),
            feasible_nonzero_count=int(value["feasible_nonzero_count"]),
            feasible_direction_rank=int(value["feasible_direction_rank"]),
            complete=bool(value["complete"]),
            no_descent=bool(value["no_descent"]),
            accepted_move=bool(value["accepted_move"]),
            center_objective=float(value["center_objective"]),
            best_trial_objective=(
                None
                if value.get("best_trial_objective") is None
                else float(value["best_trial_objective"])
            ),
        )


@dataclass(frozen=True)
class SharedUnitOptimizationResult:
    route: str
    case_id: str
    status: str
    classification: str
    locally_converged: bool
    stationarity_resolved: bool
    selected: SharedUnitEvaluation | None
    cobyqa_success: bool
    cobyqa_status: str
    cobyqa_iterations: int
    cobyqa_evaluations: int
    distinct_evaluations: int
    failed_evaluations: int
    root_attempts: int
    failed_roots: int
    projection_solves: int
    poll_evaluations: int
    poll_levels: tuple[SharedUnitPollLevel, ...]
    elapsed_seconds: float
    failed_closures: int = 0
    root_seconds: float = 0.0
    projection_seconds: float = 0.0
    evaluation_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "case_id": self.case_id,
            "status": self.status,
            "classification": self.classification,
            "locally_converged": bool(self.locally_converged),
            "stationarity_resolved": bool(self.stationarity_resolved),
            "selected": None if self.selected is None else self.selected.as_dict(),
            "cobyqa_success": bool(self.cobyqa_success),
            "cobyqa_status": self.cobyqa_status,
            "cobyqa_iterations": int(self.cobyqa_iterations),
            "cobyqa_evaluations": int(self.cobyqa_evaluations),
            "distinct_evaluations": int(self.distinct_evaluations),
            "failed_evaluations": int(self.failed_evaluations),
            "root_attempts": int(self.root_attempts),
            "failed_roots": int(self.failed_roots),
            "failed_closures": int(self.failed_closures),
            "projection_solves": int(self.projection_solves),
            "poll_evaluations": int(self.poll_evaluations),
            "poll_levels": [item.as_dict() for item in self.poll_levels],
            "elapsed_seconds": float(self.elapsed_seconds),
            "root_seconds": float(self.root_seconds),
            "projection_seconds": float(self.projection_seconds),
            "evaluation_seconds": float(self.evaluation_seconds),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SharedUnitOptimizationResult":
        selected = value.get("selected")
        result = cls(
            route=str(value["route"]),
            case_id=str(value["case_id"]),
            status=str(value["status"]),
            classification=str(value["classification"]),
            locally_converged=bool(value["locally_converged"]),
            stationarity_resolved=bool(value["stationarity_resolved"]),
            selected=(
                None if selected is None else SharedUnitEvaluation.from_dict(selected)
            ),
            cobyqa_success=bool(value["cobyqa_success"]),
            cobyqa_status=str(value["cobyqa_status"]),
            cobyqa_iterations=int(value["cobyqa_iterations"]),
            cobyqa_evaluations=int(value["cobyqa_evaluations"]),
            distinct_evaluations=int(value["distinct_evaluations"]),
            failed_evaluations=int(value["failed_evaluations"]),
            root_attempts=int(value["root_attempts"]),
            failed_roots=int(value["failed_roots"]),
            projection_solves=int(value["projection_solves"]),
            poll_evaluations=int(value["poll_evaluations"]),
            poll_levels=tuple(
                SharedUnitPollLevel.from_dict(item)
                for item in value.get("poll_levels", ())
            ),
            elapsed_seconds=float(value["elapsed_seconds"]),
            failed_closures=int(value.get("failed_closures", 0)),
            root_seconds=float(value.get("root_seconds", 0.0)),
            projection_seconds=float(value.get("projection_seconds", 0.0)),
            evaluation_seconds=float(value.get("evaluation_seconds", 0.0)),
        )
        if result.route != "shared_unit":
            raise ValueError("route-U checkpoint must use route='shared_unit'")
        return result


def _poll_directions(*, fine: bool) -> FloatArray:
    axes = np.concatenate((np.eye(7), -np.eye(7)), axis=0)
    if not fine:
        return axes
    pairwise: list[FloatArray] = []
    for left in range(7):
        for right in range(left + 1, 7):
            for left_sign in (-1.0, 1.0):
                for right_sign in (-1.0, 1.0):
                    direction = np.zeros(7)
                    direction[left] = left_sign / np.sqrt(2.0)
                    direction[right] = right_sign / np.sqrt(2.0)
                    pairwise.append(direction)
    simplex = linalg.helmert(8, full=False).T
    simplex /= np.linalg.norm(simplex, axis=1, keepdims=True)
    result = np.concatenate((axes, np.asarray(pairwise), simplex), axis=0)
    if result.shape != (106, 7):
        raise AssertionError("fine route-U poll must contain 106 directions")
    return result


def _sufficient_decrease(
    trial: SharedUnitEvaluation,
    center: SharedUnitEvaluation,
    settings: SharedUnitOptimizationSettings,
) -> bool:
    if not trial.feasible or trial.objective is None or center.objective is None:
        return False
    threshold = max(
        settings.absolute_decrease_tolerance,
        settings.relative_decrease_tolerance * abs(center.objective),
    )
    return bool(trial.objective < center.objective - threshold)


def _best_evaluation(
    values: Sequence[SharedUnitEvaluation],
    *,
    require_feasible: bool,
) -> SharedUnitEvaluation | None:
    pool = [
        item
        for item in values
        if item.available
        and item.objective is not None
        and (item.feasible or not require_feasible)
    ]
    if not pool:
        return None
    if require_feasible:
        best = min(float(item.objective) for item in pool if item.objective is not None)
        tie = 1.0e-10 * max(1.0, abs(best))
        return min(
            (item for item in pool if float(item.objective) <= best + tie),
            key=lambda item: tuple(item.normalized_controls.tolist()),
        )
    return min(
        pool,
        key=lambda item: (
            item.maximum_upper_residual,
            float(item.objective),
            *item.normalized_controls.tolist(),
        ),
    )


def optimize_shared_unit_case(
    assets: SharedUnitAssets,
    case: SharedUnitCase,
    *,
    settings: SharedUnitOptimizationSettings | None = None,
    normalized_start: npt.ArrayLike | None = None,
) -> SharedUnitOptimizationResult:
    """Run value-only COBYQA followed by the declared deterministic polls.

    Every distinct normalized control vector receives two new recycle solves
    and a newly initialized projection QP.  Only byte-identical normalized
    controls share a cached evaluation.  The method never assigns a strong KKT
    label because it intentionally does not use derivative information.
    """

    settings = settings or SharedUnitOptimizationSettings()
    started = perf_counter()
    start = (
        np.full(7, 0.5, dtype=np.float64)
        if normalized_start is None
        else _finite_vector(normalized_start, 7, "normalized_start")
    )
    if np.any(start < 0.0) or np.any(start > 1.0):
        raise ValueError("normalized_start must lie in [0, 1]")
    cache: dict[bytes, SharedUnitEvaluation] = {}
    all_evaluations: list[SharedUnitEvaluation] = []
    failed_evaluations = 0
    failed_roots = 0
    failed_closures = 0
    projection_solves = 0

    def evaluate(value: npt.ArrayLike, *, force_cold: bool = False) -> SharedUnitEvaluation:
        nonlocal failed_evaluations, failed_roots, failed_closures, projection_solves
        normalized = np.clip(
            _finite_vector(value, 7, "route-U trial controls"), 0.0, 1.0
        )
        key = np.ascontiguousarray(normalized, dtype=np.float64).tobytes()
        if not force_cold and key in cache:
            return cache[key]
        item = evaluate_shared_unit(assets, case, normalized)
        failed_roots += sum(
            not _root_attempt_accepted(attempt, assets.layout.component_count)
            for attempt in (
                item.closure.diagnostics.attempt_1,
                item.closure.diagnostics.attempt_2,
            )
        )
        if not item.closure.accepted:
            failed_closures += 1
        if item.projection is not None:
            projection_solves += 1
        if not item.available:
            failed_evaluations += 1
        all_evaluations.append(item)
        if not force_cold:
            cache[key] = item
        return item

    def evaluation_times() -> tuple[float, float, float]:
        return (
            float(sum(item.root_seconds for item in all_evaluations)),
            float(sum(item.projection_seconds for item in all_evaluations)),
            float(sum(item.elapsed_seconds for item in all_evaluations)),
        )

    upper_count = len(assets.engineering_row_names) + len(assets.trust_row_names)

    def objective(value: npt.ArrayLike) -> float:
        item = evaluate(value)
        if item.available and item.objective is not None:
            return float(item.objective)
        normalized = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
        return float(1.0e6 + np.linalg.norm(normalized - start) ** 2)

    def upper(value: npt.ArrayLike) -> FloatArray:
        item = evaluate(value)
        if (
            item.available
            and item.engineering_rows is not None
            and item.trust_rows is not None
        ):
            rows = np.concatenate((item.engineering_rows, item.trust_rows))
            if rows.shape != (upper_count,):
                raise AssertionError("route-U upper-row serialization changed")
            return rows
        return np.ones(upper_count, dtype=np.float64)

    cobyqa_success = False
    cobyqa_status = "not_started"
    cobyqa_iterations = 0
    cobyqa_endpoint = start.copy()
    try:
        optimized = minimize(
            objective,
            start,
            method="COBYQA",
            bounds=Bounds(np.zeros(7), np.ones(7)),
            constraints=NonlinearConstraint(
                upper,
                np.full(upper_count, -np.inf),
                np.zeros(upper_count),
            ),
            options={
                "maxiter": settings.maximum_iterations,
                "maxfev": settings.maximum_function_evaluations,
                "initial_tr_radius": settings.initial_trust_region_radius,
                "final_tr_radius": settings.final_trust_region_radius,
                "feasibility_tol": settings.feasibility_tolerance,
                "scale": False,
                "disp": False,
            },
        )
        cobyqa_endpoint = np.clip(
            _finite_vector(optimized.x, 7, "COBYQA endpoint"), 0.0, 1.0
        )
        cobyqa_success = bool(optimized.success)
        cobyqa_status = str(optimized.message)
        cobyqa_iterations = int(optimized.nit)
    except Exception as exc:
        cobyqa_status = f"{type(exc).__name__}: {exc}"
    evaluate(cobyqa_endpoint)
    cobyqa_evaluations = len(cache)
    selected = _best_evaluation(tuple(cache.values()), require_feasible=True)
    if selected is None:
        selected = _best_evaluation(tuple(cache.values()), require_feasible=False)
    if selected is None:
        root_seconds, projection_seconds, evaluation_seconds = evaluation_times()
        return SharedUnitOptimizationResult(
            route="shared_unit",
            case_id=case.case_id,
            status="no_available_candidate",
            classification="route_u_unavailable",
            locally_converged=False,
            stationarity_resolved=False,
            selected=None,
            cobyqa_success=cobyqa_success,
            cobyqa_status=cobyqa_status,
            cobyqa_iterations=cobyqa_iterations,
            cobyqa_evaluations=cobyqa_evaluations,
            distinct_evaluations=len(cache),
            failed_evaluations=failed_evaluations,
            root_attempts=2 * len(all_evaluations),
            failed_roots=failed_roots,
            failed_closures=failed_closures,
            projection_solves=projection_solves,
            poll_evaluations=0,
            poll_levels=(),
            elapsed_seconds=perf_counter() - started,
            root_seconds=root_seconds,
            projection_seconds=projection_seconds,
            evaluation_seconds=evaluation_seconds,
        )
    # Required independent replay of the primary selected point.
    replay = evaluate(selected.normalized_controls, force_cold=True)
    selected = replay
    if not replay.available:
        root_seconds, projection_seconds, evaluation_seconds = evaluation_times()
        return SharedUnitOptimizationResult(
            route="shared_unit",
            case_id=case.case_id,
            status="selected_replay_failed",
            classification="primary_selected_replay_failed",
            locally_converged=False,
            stationarity_resolved=False,
            selected=replay,
            cobyqa_success=cobyqa_success,
            cobyqa_status=cobyqa_status,
            cobyqa_iterations=cobyqa_iterations,
            cobyqa_evaluations=cobyqa_evaluations,
            distinct_evaluations=len(cache),
            failed_evaluations=failed_evaluations,
            root_attempts=2 * len(all_evaluations),
            failed_roots=failed_roots,
            failed_closures=failed_closures,
            projection_solves=projection_solves,
            poll_evaluations=0,
            poll_levels=(),
            elapsed_seconds=perf_counter() - started,
            root_seconds=root_seconds,
            projection_seconds=projection_seconds,
            evaluation_seconds=evaluation_seconds,
        )
    poll_levels: list[SharedUnitPollLevel] = []
    poll_evaluations = 0
    locally_converged = False
    poll_status = "poll_not_started"
    if selected.feasible and selected.objective is not None:
        center = selected
        level_index = 0
        while poll_evaluations < settings.maximum_poll_evaluations:
            radius = settings.poll_radii[level_index]
            directions = _poll_directions(fine=level_index == 1)
            seen: set[bytes] = set()
            trials: list[tuple[FloatArray, SharedUnitEvaluation]] = []
            unavailable = 0
            exhausted = False
            for direction in directions:
                trial_controls = np.clip(
                    center.normalized_controls + radius * direction, 0.0, 1.0
                )
                displacement = trial_controls - center.normalized_controls
                if not np.any(displacement):
                    continue
                key = np.ascontiguousarray(trial_controls, dtype=np.float64).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                if poll_evaluations >= settings.maximum_poll_evaluations:
                    exhausted = True
                    break
                before = len(cache)
                item = evaluate(trial_controls)
                if len(cache) > before:
                    poll_evaluations += 1
                if not item.available:
                    unavailable += 1
                trials.append((displacement, item))
            feasible_displacements = np.asarray(
                [
                    displacement
                    for displacement, item in trials
                    if item.feasible and np.linalg.norm(displacement) > 0.0
                ],
                dtype=np.float64,
            )
            rank = (
                int(
                    np.linalg.matrix_rank(
                        feasible_displacements,
                        tol=settings.direction_rank_tolerance,
                    )
                )
                if feasible_displacements.size
                else 0
            )
            improving = [
                item for _, item in trials if _sufficient_decrease(item, center, settings)
            ]
            best_trial = _best_evaluation(improving, require_feasible=True)
            complete = bool(
                not exhausted
                and unavailable == 0
                and feasible_displacements.shape[0] >= 1
            )
            accepted_move = best_trial is not None
            poll_levels.append(
                SharedUnitPollLevel(
                    radius=radius,
                    direction_count=len(directions),
                    attempted_count=len(trials),
                    unavailable_count=unavailable,
                    feasible_nonzero_count=(
                        int(feasible_displacements.shape[0])
                        if feasible_displacements.ndim == 2
                        else 0
                    ),
                    feasible_direction_rank=rank,
                    complete=complete,
                    no_descent=bool(complete and best_trial is None),
                    accepted_move=accepted_move,
                    center_objective=float(center.objective),
                    best_trial_objective=(
                        None if best_trial is None else float(best_trial.objective)
                    ),
                )
            )
            if exhausted:
                poll_status = "poll_budget_exhausted"
                break
            if best_trial is not None:
                base = center.normalized_controls.copy()
                step = best_trial.normalized_controls - base
                center = best_trial
                multiplier = settings.acceleration_growth_factor
                for _ in range(settings.maximum_acceleration_probes):
                    if poll_evaluations >= settings.maximum_poll_evaluations:
                        break
                    accelerated_controls = np.clip(base + multiplier * step, 0.0, 1.0)
                    if np.array_equal(accelerated_controls, center.normalized_controls):
                        break
                    before = len(cache)
                    accelerated = evaluate(accelerated_controls)
                    if len(cache) > before:
                        poll_evaluations += 1
                    if not _sufficient_decrease(accelerated, center, settings):
                        break
                    center = accelerated
                    multiplier *= settings.acceleration_growth_factor
                level_index = 0
                continue
            if not complete:
                poll_status = "poll_incomplete"
                break
            if level_index == 0:
                level_index = 1
                continue
            locally_converged = True
            poll_status = "two_scale_no_descent_poll_completed"
            selected = center
            break
        else:  # pragma: no cover - loop exits through explicit budget checks
            poll_status = "poll_budget_exhausted"
    else:
        poll_status = "no_feasible_primary_candidate"
    # A fresh final replay is mandatory and cannot inherit a root or QP.
    final_replay = evaluate(selected.normalized_controls, force_cold=True)
    if final_replay.available:
        selected = final_replay
    else:
        selected = final_replay
        locally_converged = False
        poll_status = "final_selected_replay_failed"
    if not selected.available:
        status = "selected_replay_failed"
        classification = poll_status
    elif locally_converged and selected.feasible:
        status = "validated_finite_resolution_candidate"
        classification = "finite_resolution_feasible_poll_converged_stationarity_unresolved"
    elif selected.feasible:
        status = "validated_feasible_incumbent_stationarity_unresolved"
        classification = poll_status
    else:
        status = "best_available_candidate_feasibility_failed"
        classification = poll_status
    root_seconds, projection_seconds, evaluation_seconds = evaluation_times()
    return SharedUnitOptimizationResult(
        route="shared_unit",
        case_id=case.case_id,
        status=status,
        classification=classification,
        locally_converged=bool(locally_converged and selected.feasible),
        stationarity_resolved=False,
        selected=selected,
        cobyqa_success=cobyqa_success,
        cobyqa_status=cobyqa_status,
        cobyqa_iterations=cobyqa_iterations,
        cobyqa_evaluations=cobyqa_evaluations,
        distinct_evaluations=len(cache),
        failed_evaluations=failed_evaluations,
        root_attempts=2 * len(all_evaluations),
        failed_roots=failed_roots,
        failed_closures=failed_closures,
        projection_solves=projection_solves,
        poll_evaluations=poll_evaluations,
        poll_levels=tuple(poll_levels),
        elapsed_seconds=perf_counter() - started,
        root_seconds=root_seconds,
        projection_seconds=projection_seconds,
        evaluation_seconds=evaluation_seconds,
    )


__all__ = [
    "ROOT_CONDITION_EPSILON_LIMIT",
    "ROOT_MAXIMUM_EVALUATIONS",
    "ROOT_MIXER_AGREEMENT_TOLERANCE",
    "ROOT_RESIDUAL_TOLERANCE",
    "ROOT_RESPONSE_AGREEMENT_TOLERANCE",
    "ROOT_SOLVER_TOLERANCE",
    "SharedUnitAssets",
    "SharedUnitCase",
    "SharedUnitClosureDiagnostics",
    "SharedUnitClosureResult",
    "SharedUnitError",
    "SharedUnitEvaluation",
    "SharedUnitFitResult",
    "SharedUnitLeverageContract",
    "SharedUnitModels",
    "SharedUnitOptimizationResult",
    "SharedUnitOptimizationSettings",
    "SharedUnitPollLevel",
    "SharedUnitRidgeScore",
    "SharedUnitRootAttempt",
    "SharedUnitTrainingData",
    "SharedUnitTrustCalibration",
    "SharedUnitTrustLimits",
    "SharedUnitTrustValues",
    "build_shared_unit_assets",
    "calibrate_shared_unit_trust",
    "cross_validate_shared_unit_models",
    "evaluate_shared_unit",
    "evaluate_shared_unit_holdout_batches",
    "evaluate_shared_unit_trust",
    "extract_shared_unit_training",
    "fit_shared_unit_leverage",
    "optimize_shared_unit_case",
    "project_shared_unit_raw",
    "quadratic_prediction_jacobian",
    "solve_shared_unit_closure",
]
