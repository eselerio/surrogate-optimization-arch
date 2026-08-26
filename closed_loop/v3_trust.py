"""Calibration of the reduced-response manuscript-v3 trust diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import casadi as ca
import numpy as np
import numpy.typing as npt

from .model import INVARIANT_MATRIX, N_COMPONENTS, N_STAGES, TSS_VECTOR
from .projection import (
    LogOverflowTSSClosure,
    NetworkLayout,
    PhysicalProjector,
    QuadraticSurrogate,
    build_network_operators,
    fit_network_row_scales,
)
from .v3_parallel import BatchProgress, run_resumable_batches
from .v3_smooth import DirectAssets, _smooth_reactor_residual
from .v3_surrogate_nlp import TrustDiagnosticCallbacks


FloatArray = npt.NDArray[np.float64]


_TRUST_PROJECTION_CONTEXT: tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    NetworkLayout,
    PhysicalProjector,
    float,
    np.ndarray | None,
] | None = None


def _initialize_trust_projection_worker(
    raw: np.ndarray,
    theta: np.ndarray,
    feed: np.ndarray,
    layout: NetworkLayout,
    state_scale: np.ndarray,
    equality_scale: np.ndarray,
    inequality_scale: np.ndarray,
    clarifier_volume_m3: float,
    overflow_tss_closure: np.ndarray | None,
) -> None:
    global _TRUST_PROJECTION_CONTEXT
    _TRUST_PROJECTION_CONTEXT = (
        np.asarray(raw, dtype=np.float64),
        np.asarray(theta, dtype=np.float64),
        np.asarray(feed, dtype=np.float64),
        layout,
        PhysicalProjector(
            state_scale,
            equality_scale,
            inequality_scale,
            absolute_tolerance=1.0e-12,
            relative_tolerance=1.0e-12,
        ),
        float(clarifier_volume_m3),
        (
            None
            if overflow_tss_closure is None
            else np.asarray(overflow_tss_closure, dtype=np.float64)
        ),
    )


def _trust_projection_batch(bounds: tuple[int, int]) -> Mapping[str, np.ndarray]:
    if _TRUST_PROJECTION_CONTEXT is None:
        raise RuntimeError("trust-projection worker was not initialized")
    raw, theta, feed, layout, projector, clarifier_volume_m3, overflow_tss_closure = (
        _TRUST_PROJECTION_CONTEXT
    )
    start, stop = bounds
    projected = np.empty((stop - start, layout.state_size), dtype=np.float64)
    accepted = np.empty(stop - start, dtype=bool)
    for local, row in enumerate(range(start, stop)):
        closure_value = (
            None
            if overflow_tss_closure is None
            else float(overflow_tss_closure[row])
        )
        operators = build_network_operators(
            feed[row],
            internal_recycle=float(theta[row, 4]),
            return_recycle=float(theta[row, 5]),
            waste_fraction=float(theta[row, 6]),
            invariant_operator=INVARIANT_MATRIX,
            tss_weights=TSS_VECTOR,
            layout=layout,
            clarifier_volume_m3=clarifier_volume_m3,
            overflow_tss_closure=closure_value,
        )
        result = projector.project(
            raw[row],
            operators.equality_matrix,
            operators.equality_rhs,
            operators.inequality_matrix,
            raise_on_failure=False,
        )
        state = np.asarray(result.state, dtype=np.float64)
        if state.shape != (layout.state_size,) or not np.all(np.isfinite(state)):
            raise RuntimeError(
                f"development OOF projection returned an invalid state at row {row}"
            )
        projected[local] = state
        accepted[local] = bool(result.accepted)
    return {"projected": projected, "accepted": accepted}


def _validate_trust_projection_batch(
    start: int, stop: int, payload: Mapping[str, np.ndarray],
) -> None:
    count = stop - start
    projected = np.asarray(payload["projected"])
    accepted = np.asarray(payload["accepted"])
    if (
        projected.ndim != 2
        or projected.shape[0] != count
        or not np.all(np.isfinite(projected))
        or accepted.shape != (count,)
        or accepted.dtype.kind != "b"
    ):
        raise ValueError("trust-projection batch payload is invalid")


@dataclass(frozen=True)
class TrustCalibration:
    callbacks: TrustDiagnosticCallbacks
    correction_limit: float
    split_limit: float
    reactor_limit: float
    split_scale: FloatArray
    development_values: FloatArray
    out_of_fold_projected: FloatArray
    out_of_fold_projection_accepted: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class ParticulateSplitRows:
    """Spawn-safe callable for the frozen particulate split residual rows."""

    layout: NetworkLayout
    particulate: tuple[int, ...]
    split_scale: FloatArray
    tss_weights: FloatArray

    def __post_init__(self) -> None:
        scale = np.asarray(self.split_scale, dtype=np.float64).reshape(-1)
        weights = np.asarray(self.tss_weights, dtype=np.float64).reshape(-1)
        if (
            scale.shape != (len(self.particulate),)
            or weights.shape != (self.layout.component_count,)
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
            or not np.all(np.isfinite(weights))
        ):
            raise ValueError("particulate split callback arrays are invalid")
        object.__setattr__(self, "split_scale", scale.copy())
        object.__setattr__(self, "tss_weights", weights.copy())

    def __call__(
        self, _theta: Any, _raw: Any, response: Any, _influent: Any,
    ) -> Any:
        final = response[self.layout.reactor_slice(N_STAGES - 1)]
        underflow = response[self.layout.underflow_flow_slice]
        indices = list(self.particulate)
        lhs = _dot(self.tss_weights, final) * underflow[indices]
        rhs = _dot(self.tss_weights, underflow) * final[indices]
        scale = (
            ca.DM(self.split_scale)
            if isinstance(response, (ca.MX, ca.SX, ca.DM))
            else self.split_scale
        )
        return (lhs - rhs) / scale


@dataclass(frozen=True)
class SmoothReactorRows:
    """Spawn-safe callable for frozen smooth-reactor residual rows."""

    direct_assets: DirectAssets
    epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("smooth-reactor callback epsilon must be positive")

    def __call__(
        self, theta: Any, _raw: Any, response: Any, _influent: Any,
    ) -> Any:
        residual = _smooth_reactor_residual(
            theta, response, self.direct_assets, self.epsilon,
        )
        reactor_scale = np.asarray(
            self.direct_assets.balance_scale[: N_STAGES * N_COMPONENTS],
            dtype=np.float64,
        )
        scale = (
            ca.DM(reactor_scale)
            if isinstance(response, (ca.MX, ca.SX, ca.DM))
            else reactor_scale
        )
        return residual / scale


def nearest_rank_95(values: npt.ArrayLike) -> float:
    data = np.sort(np.asarray(values, dtype=float).reshape(-1))
    if not len(data) or not np.all(np.isfinite(data)):
        raise ValueError("the trust sample must be nonempty and finite")
    return float(data[math.ceil(0.95 * len(data)) - 1])


def _dot(left: npt.ArrayLike, right: Any) -> Any:
    return ca.dot(ca.DM(np.asarray(left, dtype=float)), right) if isinstance(
        right, (ca.MX, ca.SX, ca.DM)
    ) else float(np.asarray(left, dtype=float) @ np.asarray(right, dtype=float))


def calibrate_trust_diagnostics(
    model: QuadraticSurrogate,
    decisions: npt.ArrayLike,
    influents: npt.ArrayLike,
    targets: npt.ArrayLike,
    out_of_fold_raw: npt.ArrayLike,
    direct_assets: DirectAssets,
    *,
    overflow_closure: LogOverflowTSSClosure | None = None,
    out_of_fold_overflow_tss: npt.ArrayLike | None = None,
    layout: NetworkLayout,
    parallel_workers: int = 1,
    batch_size: int = 64,
    checkpoint_directory: Path | None = None,
    checkpoint_contract: str | None = None,
    progress: Callable[[BatchProgress], None] | None = None,
) -> TrustCalibration:
    """Project OOF predictions and freeze three nearest-rank RMS limits.

    Projection rows are independent.  When a checkpoint directory is supplied,
    fixed contiguous batches are source/input-bound and can be reused after an
    interruption.  The row mathematics and cold-QP acceptance audit are the
    same in serial and process-parallel execution.
    """

    theta = np.asarray(decisions, dtype=float)
    feed = np.asarray(influents, dtype=float)
    truth = np.asarray(targets, dtype=float)
    raw = np.asarray(out_of_fold_raw, dtype=float)
    if raw.shape != truth.shape or truth.shape != (len(theta), layout.state_size):
        raise ValueError("OOF predictions and development targets have inconsistent shapes")
    if direct_assets.clarifier.layer_count != layout.layer_count:
        raise ValueError("direct and surrogate Clarifier layer geometries must match")
    if overflow_closure is None:
        closure_prediction = None
        if out_of_fold_overflow_tss is not None:
            raise ValueError(
                "out_of_fold_overflow_tss requires an overflow closure model"
            )
    else:
        if out_of_fold_overflow_tss is None:
            raise ValueError(
                "the production overflow closure requires out-of-fold predictions"
            )
        closure_prediction = np.asarray(out_of_fold_overflow_tss, dtype=np.float64)
        if (
            closure_prediction.shape != (len(theta),)
            or not np.all(np.isfinite(closure_prediction))
            or np.any(closure_prediction <= 0.0)
        ):
            raise ValueError("out-of-fold overflow-TSS predictions are invalid")

    row_scales = fit_network_row_scales(
        truth, feed,
        internal_recycle=theta[:, 4], return_recycle=theta[:, 5],
        waste_fraction=theta[:, 6], invariant_operator=INVARIANT_MATRIX,
        tss_weights=TSS_VECTOR, layout=layout,
        clarifier_volume_m3=(
            direct_assets.clarifier.layer_volume * direct_assets.clarifier.layer_count
        ),
        minimum_scale=1.0,
        overflow_tss_closure=closure_prediction,
    )
    if checkpoint_directory is not None and not checkpoint_contract:
        raise ValueError(
            "checkpoint_contract is required when trust checkpoints are enabled"
        )
    clarifier_volume_m3 = (
        direct_assets.clarifier.layer_volume * direct_assets.clarifier.layer_count
    )
    batches = run_resumable_batches(
        stage="whole_system_development_projection",
        row_count=len(theta),
        batch_size=batch_size,
        parallel_workers=parallel_workers,
        checkpoint_directory=checkpoint_directory,
        contract_digest=checkpoint_contract or "unpersisted",
        payload_names=("projected", "accepted"),
        worker=_trust_projection_batch,
        validate=_validate_trust_projection_batch,
        initializer=_initialize_trust_projection_worker,
        initargs=(
            raw,
            theta,
            feed,
            layout,
            model.response_scale,
            row_scales.equality,
            row_scales.inequality,
            clarifier_volume_m3,
            closure_prediction,
        ),
        progress=progress,
    )
    projected = np.vstack([batch["projected"] for batch in batches])
    projection_accepted = np.concatenate(
        [batch["accepted"].astype(bool, copy=False) for batch in batches]
    )

    particulate = np.flatnonzero(TSS_VECTOR > 0.0)
    final_truth = truth[:, layout.reactor_slice(N_STAGES - 1)]
    underflow_truth = truth[:, layout.underflow_flow_slice]
    feed_tss_truth = final_truth @ TSS_VECTOR
    underflow_tss_truth = underflow_truth @ TSS_VECTOR
    term_1 = feed_tss_truth[:, None] * underflow_truth[:, particulate]
    term_2 = underflow_tss_truth[:, None] * final_truth[:, particulate]
    split_scale = np.maximum(1.0, np.sqrt(np.mean((term_1**2 + term_2**2) / 2.0, axis=0)))

    split_rows = ParticulateSplitRows(
        layout=layout,
        particulate=tuple(map(int, particulate)),
        split_scale=split_scale,
        tss_weights=TSS_VECTOR,
    )
    reactor_rows = SmoothReactorRows(direct_assets=direct_assets)

    values = np.empty((len(theta), 3), dtype=float)
    for row in range(len(theta)):
        correction = (projected[row] - raw[row]) / model.response_scale
        split = np.asarray(split_rows(theta[row], raw[row], projected[row], feed[row]), dtype=float)
        reactor = np.asarray(reactor_rows(theta[row], raw[row], projected[row], feed[row]), dtype=float)
        values[row] = (
            np.sqrt(np.mean(correction**2)),
            np.sqrt(np.mean(split**2)),
            np.sqrt(np.mean(reactor**2)),
        )
    limits = [nearest_rank_95(values[:, column]) for column in range(3)]
    return TrustCalibration(
        callbacks=TrustDiagnosticCallbacks(
            split_rows=split_rows, reactor_rows=reactor_rows,
        ),
        correction_limit=limits[0], split_limit=limits[1],
        reactor_limit=limits[2],
        split_scale=split_scale, development_values=values,
        out_of_fold_projected=projected,
        out_of_fold_projection_accepted=projection_accepted,
    )


__all__ = [
    "ParticulateSplitRows",
    "SmoothReactorRows",
    "TrustCalibration",
    "calibrate_trust_diagnostics",
    "nearest_rank_95",
]
