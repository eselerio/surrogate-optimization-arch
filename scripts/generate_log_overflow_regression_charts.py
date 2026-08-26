"""Chart a log-regression closure for Extended-ICSOR overflow TSS.

This is a development-fitted sensitivity analysis. It does not overwrite the
production surrogate, archived holdout predictions, or optimization results.
No non-settleable-fraction floor is imposed by this analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import osqp
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from closed_loop.manuscript_v3 import reduce_mechanistic_responses
from closed_loop.model import (
    COMPOSITE_MATRIX,
    INVARIANT_MATRIX,
    NOMINAL_INFLUENT,
    TSS_VECTOR,
)
from closed_loop.projection import (
    NetworkLayout,
    build_network_operators,
    fit_network_row_scales,
)
from scripts.generate_composite_two_route_charts import (
    COMPOSITES,
    LOCATION_LABELS,
    coordinate_normalized_score,
    finite_score,
    response_composites,
    style,
)


BASELINE = "#147D92"
CORRECTED = "#D97904"
MECHANISTIC = "#343A40"
GRID = "#D8DEE4"
LOG_RIDGE_GRID = np.logspace(-4, 4, 17)
LOG_REGRESSION_FOLD_SEED = 20260826


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_log_regression(*, alpha: float | None = None):
    """Return the deterministic standardized quadratic log-TSS estimator."""

    estimator = RidgeCV(alphas=LOG_RIDGE_GRID) if alpha is None else Ridge(alpha=alpha)
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        estimator,
    )


def fit_log_overflow_regression(
    development_decisions: np.ndarray,
    development_influents: np.ndarray,
    development_responses: np.ndarray,
    layout: NetworkLayout,
) -> tuple[object, np.ndarray, float]:
    """Fit log(X_E) on development rows and return grouped-style OOF predictions."""

    q_effluent = 1.0 - development_decisions[:, 6]
    overflow_tss = (
        development_responses[:, layout.overflow_flow_slice] @ TSS_VECTOR
    ) / q_effluent
    if np.any(overflow_tss <= 0.0):
        raise ValueError("development overflow TSS must be strictly positive for log regression")
    predictors = np.column_stack((development_decisions, development_influents))
    log_target = np.log(overflow_tss)
    selector = make_log_regression()
    selector.fit(predictors, log_target)
    alpha = float(selector[-1].alpha_)
    folds = KFold(n_splits=5, shuffle=True, random_state=LOG_REGRESSION_FOLD_SEED)
    out_of_fold_log = cross_val_predict(
        make_log_regression(alpha=alpha), predictors, log_target, cv=folds, n_jobs=1,
    )
    final = make_log_regression(alpha=alpha)
    final.fit(predictors, log_target)
    return final, np.exp(out_of_fold_log), alpha


def predict_log_overflow(
    model: object,
    decisions: np.ndarray,
    influents: np.ndarray,
) -> np.ndarray:
    decision_matrix = np.asarray(decisions, dtype=float)
    influent_matrix = np.asarray(influents, dtype=float)
    single = decision_matrix.ndim == 1
    if single:
        decision_matrix = decision_matrix[None, :]
        influent_matrix = influent_matrix[None, :]
    prediction = np.exp(model.predict(np.column_stack((decision_matrix, influent_matrix))))
    if not np.all(np.isfinite(prediction)) or np.any(prediction <= 0.0):
        raise FloatingPointError("log-regression overflow prediction must be finite and positive")
    return prediction[0] if single else prediction


def overflow_equality_row(layout: NetworkLayout) -> np.ndarray:
    """Return the linear row t_X' g_E = q_E X_E,log."""

    row = np.zeros(layout.state_size, dtype=float)
    row[layout.overflow_flow_slice] = TSS_VECTOR
    return row


def fit_log_equality_scale(
    development: np.ndarray,
    decisions: np.ndarray,
    log_predictions: np.ndarray,
    layout: NetworkLayout,
) -> float:
    overflow_tss_flow = development[:, layout.overflow_flow_slice] @ TSS_VECTOR
    predicted_flow = (1.0 - decisions[:, 6]) * log_predictions
    return max(
        1.0,
        float(np.sqrt(np.mean((predicted_flow**2 + overflow_tss_flow**2) / 2.0))),
    )


def log_regression_project(
    raw: np.ndarray,
    theta: np.ndarray,
    influent: np.ndarray,
    predicted_overflow_tss: float,
    *,
    layout: NetworkLayout,
    state_scale: np.ndarray,
    equality_scale: np.ndarray,
    inequality_scale: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Solve the projection QP with the log-regression overflow closure."""

    operators = build_network_operators(
        influent,
        internal_recycle=float(theta[4]),
        return_recycle=float(theta[5]),
        waste_fraction=float(theta[6]),
        invariant_operator=INVARIANT_MATRIX,
        tss_weights=TSS_VECTOR,
        layout=layout,
    )
    equality = np.vstack((operators.equality_matrix, overflow_equality_row(layout)))
    equality_rhs_physical = np.concatenate((
        operators.equality_rhs,
        [(1.0 - float(theta[6])) * float(predicted_overflow_tss)],
    ))
    scaled_equality = (
        equality
        * state_scale[None, :]
        / equality_scale[:, None]
    )
    equality_rhs = (
        equality_rhs_physical - equality @ raw
    ) / equality_scale
    physical_scaled_inequality = operators.inequality_matrix / inequality_scale[:, None]
    scaled_network_inequality = physical_scaled_inequality * state_scale[None, :]
    network_rhs = -(physical_scaled_inequality @ raw)
    scaled_inequality = np.vstack((-np.eye(layout.state_size), scaled_network_inequality))
    inequality_rhs = np.concatenate((raw / state_scale, network_rhs))
    constraint = sparse.csc_matrix(np.vstack((scaled_equality, scaled_inequality)))
    lower = np.concatenate((
        equality_rhs,
        np.full(inequality_rhs.size, -np.inf, dtype=float),
    ))
    upper = np.concatenate((equality_rhs, inequality_rhs))
    attempts = (
        {"eps_abs": 1.0e-8, "eps_rel": 1.0e-8, "max_iter": 100_000},
        {
            "eps_abs": 1.0e-10, "eps_rel": 1.0e-10,
            "check_termination": 1, "polish_refine_iter": 20,
            "max_iter": 200_000,
        },
        {
            "eps_abs": 1.0e-8, "eps_rel": 1.0e-8,
            "rho": 0.01, "adaptive_rho": False, "max_iter": 200_000,
        },
        {
            "eps_abs": 1.0e-8, "eps_rel": 1.0e-8,
            "rho": 10.0, "adaptive_rho": False, "max_iter": 200_000,
        },
    )
    result = None
    state = None
    equality_residual = inequality_residual = nonnegative_residual = np.inf
    candidate_equality_residual = candidate_inequality_residual = np.inf
    candidate_nonnegative_residual = np.inf
    attempt_count = 0
    for attempt_count, options in enumerate(attempts, 1):
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.eye(layout.state_size, format="csc", dtype=float),
            q=np.zeros(layout.state_size, dtype=float),
            A=constraint,
            l=lower,
            u=upper,
            polishing=True,
            verbose=False,
            **options,
        )
        candidate = solver.solve(raise_error=False)
        if candidate.x is not None and str(candidate.info.status).lower().startswith("solved"):
            candidate_state = raw + state_scale * np.asarray(candidate.x, dtype=float)
            candidate_equality_residual = float(np.max(np.abs(
                (equality @ candidate_state - equality_rhs_physical)
                / equality_scale
            )))
            candidate_inequality_residual = float(np.max(np.maximum(
                (operators.inequality_matrix @ candidate_state) / inequality_scale, 0.0,
            )))
            candidate_nonnegative_residual = float(np.max(np.maximum(
                -candidate_state / state_scale, 0.0,
            )))
            if max(
                candidate_equality_residual,
                candidate_inequality_residual,
                candidate_nonnegative_residual,
            ) <= 1.0e-7:
                result = candidate
                state = candidate_state
                equality_residual = candidate_equality_residual
                inequality_residual = candidate_inequality_residual
                nonnegative_residual = candidate_nonnegative_residual
                break
    if result is None:
        raise RuntimeError(
            "corrected projection failed after deterministic cold retries: "
            f"status={candidate.info.status}; eq={candidate_equality_residual:.3e}; "
            f"ineq={candidate_inequality_residual:.3e}; "
            f"nonnegative={candidate_nonnegative_residual:.3e}"
        )
    assert state is not None
    return state, {
        "status": str(result.info.status),
        "iterations": int(result.info.iter),
        "solver_attempts": attempt_count,
        "equality_residual": equality_residual,
        "inequality_residual": inequality_residual,
        "nonnegative_residual": nonnegative_residual,
        "overflow_closure_residual_mg_L": float(
            candidate_state[layout.overflow_flow_slice] @ TSS_VECTOR
            / (1.0 - float(theta[6]))
            - predicted_overflow_tss
        ),
    }


def tss_values(response: np.ndarray, decisions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feed = response[:, 100:120] @ TSS_VECTOR
    effluent = (response[:, 120:140] @ TSS_VECTOR) / (1.0 - decisions[:, 6])
    return feed, effluent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = (args.output or run / "report/figures/settling_floor_correction").resolve()
    output.mkdir(parents=True, exist_ok=True)
    style()

    with np.load(run / "datasets/effective_design.npz", allow_pickle=False) as stored:
        development_decisions = np.asarray(stored["development_decisions"], dtype=float)
        development_influents = np.asarray(stored["development_influents"], dtype=float)
        test_decisions = np.asarray(stored["test_decisions"], dtype=float)
        test_influents = np.asarray(stored["test_influents"], dtype=float)
        robustness_influents = np.asarray(stored["robustness_influents"], dtype=float)
    with np.load(run / "datasets/development/mechanistic_accepted_v3.npz", allow_pickle=False) as stored:
        development = reduce_mechanistic_responses(np.asarray(stored["targets"], dtype=float), 10)
    with np.load(run / "predictions/post_selection_holdout.npz", allow_pickle=False) as stored:
        raw = np.asarray(stored["raw"], dtype=float)
        baseline = np.asarray(stored["projected"], dtype=float)
        mechanistic = np.asarray(stored["mechanistic"], dtype=float)
    with np.load(run / "models/ridge_surrogate.npz", allow_pickle=False) as stored:
        state_scale = np.asarray(stored["response_scale"], dtype=float)

    layout = NetworkLayout(layer_count=10)
    log_model, development_oof_tss, selected_alpha = fit_log_overflow_regression(
        development_decisions,
        development_influents,
        development,
        layout,
    )
    holdout_log_tss = predict_log_overflow(log_model, test_decisions, test_influents)
    row_scales = fit_network_row_scales(
        development,
        development_influents,
        internal_recycle=development_decisions[:, 4],
        return_recycle=development_decisions[:, 5],
        waste_fraction=development_decisions[:, 6],
        invariant_operator=INVARIANT_MATRIX,
        tss_weights=TSS_VECTOR,
        layout=layout,
        minimum_scale=1.0,
    )
    added_scale = fit_log_equality_scale(
        development, development_decisions, development_oof_tss, layout,
    )
    equality_scale = np.concatenate((row_scales.equality, [added_scale]))
    inequality_scale = row_scales.inequality

    checkpoint = output / "log_regression_corrected_holdout_checkpoint.npz"
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(raw).view(np.uint8))
    digest.update(np.ascontiguousarray(holdout_log_tss).view(np.uint8))
    contract_digest = digest.hexdigest()
    corrected = np.empty_like(raw)
    audits: list[dict[str, float | int | str]] = []
    start_row = 0
    if checkpoint.is_file():
        with np.load(checkpoint, allow_pickle=False) as stored:
            stored_corrected = np.asarray(stored["corrected"], dtype=float)
            stored_digest = str(stored["contract_digest"].item())
            stored_audits = [json.loads(str(item)) for item in stored["audits_json"]]
        if (
            stored_digest == contract_digest
            and stored_corrected.ndim == 2
            and stored_corrected.shape[1] == raw.shape[1]
            and len(stored_corrected) == len(stored_audits)
            and len(stored_corrected) <= len(raw)
        ):
            start_row = len(stored_corrected)
            corrected[:start_row] = stored_corrected
            audits.extend(stored_audits)
            print(f"resumed corrected holdout projections: {start_row}/{len(raw)}", flush=True)
    for row in range(start_row, len(raw)):
        try:
            corrected[row], audit = log_regression_project(
                raw[row], test_decisions[row], test_influents[row], holdout_log_tss[row],
                layout=layout,
                state_scale=state_scale,
                equality_scale=equality_scale,
                inequality_scale=inequality_scale,
            )
        except RuntimeError as error:
            raise RuntimeError(f"corrected holdout row {row} failed") from error
        audits.append({"row": row, **audit})
        if (row + 1) % 250 == 0:
            print(f"corrected holdout projections: {row + 1}/{len(raw)}", flush=True)
            temporary = output / "log_regression_corrected_holdout_checkpoint.tmp.npz"
            np.savez_compressed(
                temporary,
                corrected=corrected[:row + 1],
                audits_json=np.asarray([json.dumps(item, sort_keys=True) for item in audits]),
                contract_digest=np.asarray(contract_digest),
            )
            os.replace(temporary, checkpoint)

    temporary = output / "log_regression_corrected_holdout_checkpoint.tmp.npz"
    np.savez_compressed(
        temporary,
        corrected=corrected,
        audits_json=np.asarray([json.dumps(item, sort_keys=True) for item in audits]),
        contract_digest=np.asarray(contract_digest),
    )
    os.replace(temporary, checkpoint)
    np.savez_compressed(
        output / "log_regression_diagnostics.npz",
        selected_ridge_alpha=np.asarray(selected_alpha),
        development_oof_tss=development_oof_tss,
        holdout_predicted_tss=holdout_log_tss,
    )

    truth_composite = response_composites(mechanistic, test_decisions)
    baseline_composite = response_composites(baseline, test_decisions)
    corrected_composite = response_composites(corrected, test_decisions)
    composite_sets = {"Baseline": baseline_composite, "Log-regression closure": corrected_composite}
    summary_rows: list[dict[str, object]] = []

    # Figure 1: overflow-TSS parity before and after correction.
    truth_tss = truth_composite[:, 6, 3]
    parity_predictions = (baseline_composite[:, 6, 3], corrected_composite[:, 6, 3])
    all_positive = np.concatenate((truth_tss, *parity_predictions))
    limits = (float(np.min(all_positive)) / 1.15, float(np.max(all_positive)) * 1.15)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    density = []
    for axis, label, prediction in zip(axes, composite_sets, parity_predictions, strict=True):
        nrmse, nmae, r2 = finite_score(truth_tss, prediction)
        density.append(axis.hexbin(
            truth_tss, prediction, gridsize=45, mincnt=1, cmap="viridis",
            xscale="log", yscale="log",
        ))
        axis.plot(limits, limits, "--", lw=1.3, color="#dc2626", label="Perfect match")
        axis.set(
            xlim=limits, ylim=limits,
            xlabel="Mechanistic overflow TSS (mg/L)",
            ylabel=f"{label} overflow TSS (mg/L)",
            title=f"{label}: R²={r2:.3f}; nRMSE={nrmse:.3f}",
        )
        axis.set_aspect("equal", adjustable="box")
        summary_rows.extend((
            {"scope": "holdout_overflow_tss", "method": label, "metric": "r2", "value": r2},
            {"scope": "holdout_overflow_tss", "method": label, "metric": "nrmse", "value": nrmse},
            {"scope": "holdout_overflow_tss", "method": label, "metric": "nmae", "value": nmae},
        ))
    maximum = max(2.0, *(float(np.max(item.get_array())) for item in density))
    norm = LogNorm(vmin=1.0, vmax=maximum)
    for item in density:
        item.set_norm(norm)
        item.set_clim(1.0, maximum)
    colorbar_axis = fig.add_axes((0.91, 0.17, 0.018, 0.64))
    colorbar = fig.colorbar(density[0], cax=colorbar_axis)
    colorbar.set_label("Holdout rows per hexagon (log scale)")
    fig.suptitle("Effect of the log-regression closure on overflow-TSS parity", fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.87, bottom=0.12, top=0.86, wspace=0.34)
    save(fig, output, "01_overflow_tss_parity_before_after")

    # Figure 2: headline overflow-TSS performance and low-tail behavior.
    low_mask = truth_tss <= 10.0
    metrics = {label: finite_score(truth_tss, values) for label, values in zip(composite_sets, parity_predictions, strict=True)}
    feed = {}
    effluent = {}
    for label, values in (("Mechanistic", mechanistic), ("Baseline", baseline), ("Log-regression closure", corrected)):
        feed[label], effluent[label] = tss_values(values, test_decisions)
    absolute_mae = {
        label: float(np.mean(np.abs(values - truth_tss)))
        for label, values in zip(composite_sets, parity_predictions, strict=True)
    }
    low_mae = {
        label: float(np.mean(np.abs(values[low_mask] - truth_tss[low_mask])))
        for label, values in zip(composite_sets, parity_predictions, strict=True)
    }
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.0))
    for axis, index, name, better in (
        (axes[0], 0, "Overflow-TSS nRMSE", "lower is better"),
        (axes[1], 2, "Overflow-TSS R²", "higher is better"),
    ):
        vals = [metrics[label][index] for label in composite_sets]
        bars = axis.bar(composite_sets.keys(), vals, color=(BASELINE, CORRECTED))
        axis.set_title(f"{name}\n({better})")
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    bars = axes[2].bar(absolute_mae.keys(), absolute_mae.values(), color=(BASELINE, CORRECTED))
    axes[2].set_title("Overflow-TSS MAE\n(lower is better)")
    axes[2].set_ylabel("MAE (mg/L)")
    axes[2].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    bars = axes[3].bar(low_mae.keys(), low_mae.values(), color=(BASELINE, CORRECTED))
    axes[3].set_title(f"MAE where exact TSS <= 10 mg/L\n(n={int(np.sum(low_mask)):,}; lower is better)")
    axes[3].set_ylabel("MAE (mg/L)")
    axes[3].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=18)
    fig.suptitle("Overflow-TSS performance change with the log-regression closure", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, output, "02_overflow_tss_performance_change")
    for label in composite_sets:
        summary_rows.extend((
            {"scope": "holdout_overflow_tss_low", "method": label, "metric": "mae_exact_le_10_mg_L", "value": low_mae[label]},
            {"scope": "holdout_overflow_tss", "method": label, "metric": "mae_mg_L", "value": absolute_mae[label]},
            {"scope": "holdout_overflow_tss", "method": label, "metric": "minimum_prediction_mg_L", "value": float(np.min(effluent[label]))},
        ))

    # Figure 3: Q3-compatible all-location composite performance.
    all_scores: dict[str, list[tuple[float, float, float]]] = {}
    for label, values in composite_sets.items():
        all_scores[label] = [
            coordinate_normalized_score(truth_composite[:, :, index], values[:, :, index])
            for index in range(len(COMPOSITES))
        ]
    x = np.arange(len(COMPOSITES)); width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    for axis, metric_index, ylabel, title in (
        (axes[0, 0], 0, "nRMSE", "Location-normalized RMSE"),
        (axes[0, 1], 1, "nMAE", "Location-normalized MAE"),
        (axes[1, 0], 2, "Mean location R²", "Mean location coefficient of determination"),
    ):
        for offset, (label, color) in zip((-width / 2, width / 2), (("Baseline", BASELINE), ("Log-regression closure", CORRECTED)), strict=True):
            axis.bar(x + offset, [score[metric_index] for score in all_scores[label]], width, color=color, label=label)
        axis.set(xticks=x, xticklabels=COMPOSITES, ylabel=ylabel, title=title)
    changes = 100.0 * (
        np.asarray([score[0] for score in all_scores["Log-regression closure"]])
        / np.asarray([score[0] for score in all_scores["Baseline"]]) - 1.0
    )
    bars = axes[1, 1].bar(COMPOSITES, changes, color=np.where(changes <= 0.0, CORRECTED, "#B91C1C"))
    axes[1, 1].axhline(0.0, color=MECHANISTIC, lw=0.8)
    axes[1, 1].set(title="nRMSE change (negative improves)", ylabel="Change (%)")
    axes[1, 1].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    axes[0, 0].legend(ncol=2)
    fig.suptitle("All-location composite performance with the log-regression closure", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, output, "03_all_location_composite_performance_change")
    for label, scores in all_scores.items():
        for composite, (nrmse, nmae, mean_r2) in zip(COMPOSITES, scores, strict=True):
            summary_rows.extend((
                {"scope": f"all_locations_{composite.lower()}", "method": label, "metric": "nrmse", "value": nrmse},
                {"scope": f"all_locations_{composite.lower()}", "method": label, "metric": "nmae", "value": nmae},
                {"scope": f"all_locations_{composite.lower()}", "method": label, "metric": "mean_location_r2", "value": mean_r2},
            ))

    # Figure 4: expose intended and collateral changes by location and composite.
    matrices: dict[str, np.ndarray] = {}
    for label, values in composite_sets.items():
        matrix = np.empty((len(LOCATION_LABELS), len(COMPOSITES)))
        for i in range(len(LOCATION_LABELS)):
            for j in range(len(COMPOSITES)):
                matrix[i, j] = finite_score(
                    truth_composite[:, i, j], values[:, i, j],
                )[0]
        matrices[label] = matrix
    delta = 100.0 * (matrices["Log-regression closure"] / matrices["Baseline"] - 1.0)
    common_max = float(max(np.max(matrices["Baseline"]), np.max(matrices["Log-regression closure"])))
    delta_limit = float(np.max(np.abs(delta)))
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.3))
    for axis, label in zip(axes[:2], ("Baseline", "Log-regression closure"), strict=True):
        image = axis.imshow(matrices[label], aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=common_max)
        fig.colorbar(image, ax=axis, pad=0.02, label="nRMSE")
        axis.set_title(label)
        axis.set(yticks=np.arange(8), yticklabels=LOCATION_LABELS, xticks=np.arange(4), xticklabels=COMPOSITES)
    image = axes[2].imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-delta_limit, vmax=delta_limit)
    fig.colorbar(image, ax=axes[2], pad=0.02, label="nRMSE change (%)")
    axes[2].set_title("Correction change (negative improves)")
    axes[2].set(yticks=np.arange(8), yticklabels=LOCATION_LABELS, xticks=np.arange(4), xticklabels=COMPOSITES)
    fig.suptitle("Location-level effect of the log-regression closure", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, output, "04_location_composite_nrmse_change")

    # Figure 5: empirical overflow-to-feed ratio distribution.
    ratios = {
        label: effluent[label] / feed[label]
        for label in ("Mechanistic", "Baseline", "Log-regression closure")
    }
    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    for label, color in (("Mechanistic", MECHANISTIC), ("Baseline", BASELINE), ("Log-regression closure", CORRECTED)):
        ordered = np.sort(ratios[label])
        probability = np.arange(1, len(ordered) + 1) / len(ordered)
        axis.plot(ordered, probability, color=color, lw=2.0, label=label)
    axis.set_xscale("log")
    axis.set(
        xlabel="Overflow TSS / feed TSS",
        ylabel="Cumulative fraction of holdout rows",
        title="Overflow-to-feed TSS ratio distribution",
        xlim=(max(1.0e-3, min(np.min(value) for value in ratios.values()) / 1.3), max(np.max(value) for value in ratios.values()) * 1.2),
        ylim=(0.0, 1.01),
    )
    axis.legend()
    fig.tight_layout()
    save(fig, output, "05_overflow_to_feed_tss_ratio_distribution")

    # Figure 6: fixed-decision diagnostic on the archived surrogate optima.
    cases = ["nominal", *[f"robustness_{i:02d}" for i in range(1, 11)]]
    case_labels = ["N", *[f"R{i}" for i in range(1, 11)]]
    case_influents = [NOMINAL_INFLUENT, *list(robustness_influents)]
    selected_rows: list[dict[str, object]] = []
    for case, label, influent in zip(cases, case_labels, case_influents, strict=True):
        with np.load(run / "optimization" / case / "surrogate_casewise_reference.npz", allow_pickle=False) as stored:
            theta = np.asarray(stored["theta"], dtype=float)
            case_raw = np.asarray(stored["raw"], dtype=float)
            case_baseline = np.asarray(stored["projected"], dtype=float)
            exact = np.asarray(stored["exact_reference"], dtype=float)
        predicted_tss = float(predict_log_overflow(
            log_model, theta, np.asarray(influent, dtype=float),
        ))
        case_corrected, _ = log_regression_project(
            case_raw, theta, np.asarray(influent, dtype=float), predicted_tss,
            layout=layout,
            state_scale=state_scale,
            equality_scale=equality_scale,
            inequality_scale=inequality_scale,
        )
        q_e = 1.0 - theta[6]
        selected_rows.append({
            "case": case, "label": label,
            "mechanistic": float(TSS_VECTOR @ exact[120:140] / q_e),
            "baseline": float(TSS_VECTOR @ case_baseline[120:140] / q_e),
            "log_regression": float(TSS_VECTOR @ case_corrected[120:140] / q_e),
        })
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(output / "selected_decision_tss.csv", index=False)
    x = np.arange(len(selected)); width = 0.25
    fig, axis = plt.subplots(figsize=(12.0, 5.2))
    axis.bar(x - width, selected["baseline"], width, color=BASELINE, label="Baseline projected")
    axis.bar(x, selected["log_regression"], width, color=CORRECTED, label="Log-regression closure")
    axis.bar(x + width, selected["mechanistic"], width, color=MECHANISTIC, label="Exact mechanistic replay")
    axis.set(
        xticks=x, xticklabels=selected["label"],
        ylabel="Overflow TSS (mg/L)",
        title="Archived surrogate-selected decisions: correction effect without reoptimization",
    )
    axis.legend(ncol=3)
    fig.tight_layout()
    save(fig, output, "06_selected_decision_tss_fixed_decisions")

    audit_frame = pd.DataFrame(audits)
    audit_frame.to_csv(output / "projection_audit.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "performance_summary.csv", index=False)
    stems = (
        "01_overflow_tss_parity_before_after",
        "02_overflow_tss_performance_change",
        "03_all_location_composite_performance_change",
        "04_location_composite_nrmse_change",
        "05_overflow_to_feed_tss_ratio_distribution",
        "06_selected_decision_tss_fixed_decisions",
    )
    pd.DataFrame([
        {"chart": index, "png": f"{stem}.png", "svg": f"{stem}.svg"}
        for index, stem in enumerate(stems, 1)
    ]).to_csv(output / "chart_index.csv", index=False)
    (output / "log_regression_definition.json").write_text(json.dumps({
        "analysis_type": "development-fitted log-regression closure, holdout assessment, and fixed-decision diagnostic; no production-surrogate refit or reoptimization",
        "regression_target": "log(overflow TSS in mg/L)",
        "regression_features": "standardized controls and influent; complete second-order polynomial",
        "ridge_selection": "development-only RidgeCV over a fixed logarithmic grid",
        "selected_ridge_alpha": selected_alpha,
        "projection_closure": "t_X' g_E = q_E * exp(log-regression prediction)",
        "nonsettleable_fraction_constraint": "not used",
        "projection_objective": "minimum standardized Euclidean displacement from the same raw Extended-ICSOR prediction",
        "holdout_rows": len(raw),
        "corrected_projection_failures": 0,
        "maximum_scaled_equality_residual": float(audit_frame["equality_residual"].max()),
        "maximum_scaled_inequality_residual": float(audit_frame["inequality_residual"].max()),
        "maximum_scaled_nonnegativity_residual": float(audit_frame["nonnegative_residual"].max()),
        "maximum_absolute_overflow_closure_residual_mg_L": float(audit_frame["overflow_closure_residual_mg_L"].abs().max()),
    }, indent=2), encoding="utf-8")
    print(output)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
