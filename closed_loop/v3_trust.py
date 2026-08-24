"""Calibration of the reduced-response manuscript-v3 trust diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import casadi as ca
import numpy as np
import numpy.typing as npt

from .model import INVARIANT_MATRIX, N_COMPONENTS, N_STAGES, TSS_VECTOR
from .projection import (
    NetworkLayout,
    PhysicalProjector,
    QuadraticSurrogate,
    build_network_operators,
    fit_network_row_scales,
)
from .v3_smooth import DirectAssets, _smooth_reactor_residual
from .v3_surrogate_nlp import TrustDiagnosticCallbacks


FloatArray = npt.NDArray[np.float64]


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
    layout: NetworkLayout,
) -> TrustCalibration:
    """Project OOF predictions and freeze three nearest-rank RMS limits."""

    theta = np.asarray(decisions, dtype=float)
    feed = np.asarray(influents, dtype=float)
    truth = np.asarray(targets, dtype=float)
    raw = np.asarray(out_of_fold_raw, dtype=float)
    if raw.shape != truth.shape or truth.shape != (len(theta), layout.state_size):
        raise ValueError("OOF predictions and development targets have inconsistent shapes")
    if direct_assets.clarifier.layer_count != layout.layer_count:
        raise ValueError("direct and surrogate Clarifier layer geometries must match")

    row_scales = fit_network_row_scales(
        truth, feed,
        internal_recycle=theta[:, 4], return_recycle=theta[:, 5],
        waste_fraction=theta[:, 6], invariant_operator=INVARIANT_MATRIX,
        tss_weights=TSS_VECTOR, layout=layout,
        clarifier_volume_m3=(
            direct_assets.clarifier.layer_volume * direct_assets.clarifier.layer_count
        ),
        minimum_scale=1.0,
    )
    projector = PhysicalProjector(
        model.response_scale, row_scales.equality, row_scales.inequality,
        absolute_tolerance=1.0e-12, relative_tolerance=1.0e-12,
    )
    projected = np.empty_like(raw)
    projection_accepted = np.empty(len(theta), dtype=bool)
    for row in range(len(theta)):
        operators = build_network_operators(
            feed[row], internal_recycle=float(theta[row, 4]),
            return_recycle=float(theta[row, 5]), waste_fraction=float(theta[row, 6]),
            invariant_operator=INVARIANT_MATRIX, tss_weights=TSS_VECTOR, layout=layout,
            clarifier_volume_m3=(
                direct_assets.clarifier.layer_volume
                * direct_assets.clarifier.layer_count
            ),
        )
        result = projector.project(
            raw[row], operators.equality_matrix, operators.equality_rhs,
            operators.inequality_matrix, raise_on_failure=False,
        )
        state = np.asarray(result.state, dtype=float)
        if state.shape != (layout.state_size,) or not np.all(np.isfinite(state)):
            raise RuntimeError(
                f"development OOF projection returned an invalid state at row {row}"
            )
        projected[row] = state
        projection_accepted[row] = bool(result.accepted)

    particulate = np.flatnonzero(TSS_VECTOR > 0.0)
    final_truth = truth[:, layout.reactor_slice(N_STAGES - 1)]
    underflow_truth = truth[:, layout.underflow_flow_slice]
    feed_tss_truth = final_truth @ TSS_VECTOR
    underflow_tss_truth = underflow_truth @ TSS_VECTOR
    term_1 = feed_tss_truth[:, None] * underflow_truth[:, particulate]
    term_2 = underflow_tss_truth[:, None] * final_truth[:, particulate]
    split_scale = np.maximum(1.0, np.sqrt(np.mean((term_1**2 + term_2**2) / 2.0, axis=0)))

    def split_rows(_theta: Any, _raw: Any, response: Any, _influent: Any) -> Any:
        final = response[layout.reactor_slice(N_STAGES - 1)]
        underflow = response[layout.underflow_flow_slice]
        lhs = _dot(TSS_VECTOR, final) * underflow[particulate.tolist()]
        rhs = _dot(TSS_VECTOR, underflow) * final[particulate.tolist()]
        scale = ca.DM(split_scale) if isinstance(response, (ca.MX, ca.SX, ca.DM)) else split_scale
        return (lhs - rhs) / scale

    def reactor_rows(theta_v: Any, _raw: Any, response: Any, _feed_v: Any) -> Any:
        residual = _smooth_reactor_residual(
            theta_v, response, direct_assets, 1.0e-8,
        )
        reactor_scale = np.asarray(
            direct_assets.balance_scale[: N_STAGES * N_COMPONENTS], dtype=float
        )
        scale = ca.DM(reactor_scale) if isinstance(
            response, (ca.MX, ca.SX, ca.DM)
        ) else reactor_scale
        return residual / scale

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


__all__ = ["TrustCalibration", "calibrate_trust_diagnostics", "nearest_rank_95"]
