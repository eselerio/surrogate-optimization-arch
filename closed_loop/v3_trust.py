"""Calibration of the five manuscript-v3 surrogate trust diagnostics."""

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
from .v3_smooth import DirectAssets, _smooth_response_residual
from .v3_surrogate_nlp import TrustDiagnosticCallbacks


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TrustCalibration:
    callbacks: TrustDiagnosticCallbacks
    correction_limit: float
    split_limit: float
    reactor_limit: float
    flux_limit: float
    split_scale: FloatArray
    development_values: FloatArray
    out_of_fold_projected: FloatArray


def nearest_rank_95(values: npt.ArrayLike) -> float:
    data = np.sort(np.asarray(values, dtype=float).reshape(-1))
    if not len(data) or not np.all(np.isfinite(data)):
        raise ValueError("the trust sample must be nonempty and finite")
    return float(data[math.ceil(0.95 * len(data)) - 1])


def _reduced_state(response: Any, layout: NetworkLayout) -> Any:
    blocks = [response[layout.reactor_slice(i)] for i in range(N_STAGES)]
    blocks.append(response[layout.layer_slice])
    if isinstance(response, (ca.MX, ca.SX, ca.DM)):
        return ca.vertcat(*blocks)
    return np.concatenate([np.asarray(item, dtype=float) for item in blocks])


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
    """Project OOF predictions and freeze all four nearest-rank limits."""

    theta = np.asarray(decisions, dtype=float)
    feed = np.asarray(influents, dtype=float)
    truth = np.asarray(targets, dtype=float)
    raw = np.asarray(out_of_fold_raw, dtype=float)
    if raw.shape != truth.shape or truth.shape != (len(theta), layout.state_size):
        raise ValueError("OOF predictions and development targets have inconsistent shapes")

    row_scales = fit_network_row_scales(
        truth, feed,
        internal_recycle=theta[:, 4], return_recycle=theta[:, 5],
        waste_fraction=theta[:, 6], invariant_operator=INVARIANT_MATRIX,
        tss_weights=TSS_VECTOR, layout=layout, minimum_scale=1.0,
    )
    projector = PhysicalProjector(
        model.response_scale, row_scales.equality, row_scales.inequality,
        absolute_tolerance=1.0e-12, relative_tolerance=1.0e-12,
    )
    projected = np.empty_like(raw)
    for row in range(len(theta)):
        operators = build_network_operators(
            feed[row], internal_recycle=float(theta[row, 4]),
            return_recycle=float(theta[row, 5]), waste_fraction=float(theta[row, 6]),
            invariant_operator=INVARIANT_MATRIX, tss_weights=TSS_VECTOR, layout=layout,
        )
        result = projector.project(
            raw[row], operators.equality_matrix, operators.equality_rhs,
            operators.inequality_matrix, raise_on_failure=False,
        )
        if not result.accepted:
            raise RuntimeError(f"development OOF projection failed at row {row}")
        projected[row] = result.state

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

    def balance_rows(_theta: Any, _raw: Any, response: Any, case_feed: Any) -> Any:
        state = _reduced_state(response, layout)
        final = response[layout.reactor_slice(N_STAGES - 1)]
        feed_tss = _dot(TSS_VECTOR, final)
        _, residual = _smooth_response_residual(
            _theta, case_feed, state, feed_tss, direct_assets, 1.0e-8, 1.0,
        )
        scale = ca.DM(direct_assets.balance_scale) if isinstance(
            response, (ca.MX, ca.SX, ca.DM)
        ) else direct_assets.balance_scale
        return residual / scale

    def reactor_rows(theta_v: Any, raw_v: Any, response: Any, feed_v: Any) -> Any:
        return balance_rows(theta_v, raw_v, response, feed_v)[: N_STAGES * N_COMPONENTS]

    def flux_rows(theta_v: Any, raw_v: Any, response: Any, feed_v: Any) -> Any:
        return balance_rows(theta_v, raw_v, response, feed_v)[N_STAGES * N_COMPONENTS :]

    values = np.empty((len(theta), 4), dtype=float)
    for row in range(len(theta)):
        correction = (projected[row] - raw[row]) / model.response_scale
        split = np.asarray(split_rows(theta[row], raw[row], projected[row], feed[row]), dtype=float)
        reactor = np.asarray(reactor_rows(theta[row], raw[row], projected[row], feed[row]), dtype=float)
        flux = np.asarray(flux_rows(theta[row], raw[row], projected[row], feed[row]), dtype=float)
        values[row] = (
            np.sqrt(np.mean(correction**2)),
            np.sqrt(np.mean(split**2)),
            np.sqrt(np.mean(reactor**2)),
            np.sqrt(np.mean(flux**2)),
        )
    limits = [nearest_rank_95(values[:, column]) for column in range(4)]
    return TrustCalibration(
        callbacks=TrustDiagnosticCallbacks(
            split_rows=split_rows, reactor_rows=reactor_rows, flux_rows=flux_rows,
        ),
        correction_limit=limits[0], split_limit=limits[1],
        reactor_limit=limits[2], flux_limit=limits[3],
        split_scale=split_scale, development_values=values,
        out_of_fold_projected=projected,
    )


__all__ = ["TrustCalibration", "calibrate_trust_diagnostics", "nearest_rank_95"]
