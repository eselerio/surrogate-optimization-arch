"""Create untouched-test charts for surrogate versus mechanistic model fidelity."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"raw": "#d97706", "projected": "#2563eb", "truth": "#111827"}


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 220, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
        "axes.titleweight": "bold", "figure.facecolor": "white",
    })


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def coordinate_metrics(truth: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    err = pred - truth
    rmse = np.sqrt(np.mean(err**2, axis=0))
    scale = np.ptp(truth, axis=0)
    nrmse = np.divide(rmse, scale, out=np.full_like(rmse, np.nan), where=scale > 0)
    ss_res = np.sum(err**2, axis=0)
    ss_tot = np.sum((truth - truth.mean(axis=0))**2, axis=0)
    r2 = 1 - np.divide(ss_res, ss_tot, out=np.full_like(ss_res, np.nan), where=ss_tot > 0)
    return rmse, nrmse, r2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = (args.output or run / "report" / "surrogate_emulation_quality").resolve()
    output.mkdir(parents=True, exist_ok=True)
    style()

    metric_path = run / "metrics" / "untouched_prediction_metrics.csv"
    stored = pd.read_csv(metric_path)
    coordinate_rows = stored.query(
        "method == 'projected' and block == 'complete_response' and coordinate != 'ALL'"
    ).reset_index(drop=True)
    names = coordinate_rows["coordinate"].tolist()
    if len(names) != 170:
        raise ValueError(f"Expected 170 complete-response coordinates, found {len(names)}")

    with np.load(run / "predictions" / "untouched_test.npz") as arrays:
        truth = arrays["mechanistic"]
        raw = arrays["raw"]
        projected = arrays["projected"]
    if truth.shape != projected.shape or truth.shape[1] != len(names):
        raise ValueError("Prediction arrays and metric coordinate names do not align")

    methods = {"Raw ridge": raw, "Projected surrogate": projected}
    summary = []
    for label, pred in methods.items():
        _, nrmse, r2 = coordinate_metrics(truth, pred)
        summary.append({
            "method": label, "test_cases": truth.shape[0], "coordinates": truth.shape[1],
            "median_r2": np.nanmedian(r2), "mean_r2": np.nanmean(r2),
            "r2_ge_0_90_fraction": np.nanmean(r2 >= .90),
            "r2_ge_0_75_fraction": np.nanmean(r2 >= .75),
            "median_nrmse": np.nanmedian(nrmse), "p90_nrmse": np.nanpercentile(nrmse, 90),
        })
    pd.DataFrame(summary).to_csv(output / "fidelity_summary.csv", index=False)

    # 1. Dimensionless parity over all 170,000 held-out values.
    mean = truth.mean(axis=0)
    std = truth.std(axis=0)
    valid = std > 1e-14
    tz = ((truth[:, valid] - mean[valid]) / std[valid]).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True, sharey=True, constrained_layout=True)
    lim = (-4, 4)
    for ax, (label, pred) in zip(axes, methods.items()):
        pz = ((pred[:, valid] - mean[valid]) / std[valid]).ravel()
        hb = ax.hexbin(tz, pz, gridsize=75, bins="log", mincnt=1, cmap="viridis", extent=(*lim, *lim))
        ax.plot(lim, lim, color="#ef4444", lw=1.2, ls="--", label="Perfect agreement")
        ax.set(xlim=lim, ylim=lim, xlabel="Mechanistic model (standardized)", title=label)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left", frameon=False)
        fig.colorbar(hb, ax=ax, label="log count")
    axes[0].set_ylabel("Surrogate prediction (standardized)")
    fig.suptitle(f"Untouched-test parity across {truth.shape[1]} outputs × {truth.shape[0]:,} cases", fontsize=13, fontweight="bold")
    save(fig, output, "01_all_output_parity")

    # 2. Coordinate-level R² and normalized error, sorted to reveal the tail.
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), constrained_layout=True)
    projected_rmse, projected_nrmse, projected_r2 = coordinate_metrics(truth, projected)
    raw_rmse, raw_nrmse, raw_r2 = coordinate_metrics(truth, raw)
    order = np.argsort(projected_r2)[::-1]
    x = np.arange(len(names))
    axes[0].plot(x, raw_r2[order], color=COLORS["raw"], lw=1.1, label="Raw ridge")
    axes[0].plot(x, projected_r2[order], color=COLORS["projected"], lw=1.4, label="Projected surrogate")
    axes[0].axhline(.90, color="#16a34a", ls="--", lw=1, label="R² = 0.90")
    axes[0].axhline(.75, color="#64748b", ls=":", lw=1, label="R² = 0.75")
    axes[0].set(ylabel="R²", title="A. Explained variance by output coordinate", ylim=(min(-.1, np.nanmin(projected_r2)), 1.02))
    axes[0].legend(ncol=4, frameon=False, loc="lower left")
    axes[1].semilogy(x, raw_nrmse[order], color=COLORS["raw"], lw=1.1, label="Raw ridge")
    axes[1].semilogy(x, projected_nrmse[order], color=COLORS["projected"], lw=1.4, label="Projected surrogate")
    axes[1].set(xlabel="Output coordinates (sorted by projected R²)", ylabel="NRMSE (RMSE / test range)", title="B. Range-normalized prediction error")
    fig.suptitle("Accuracy distribution and weak-output tail", fontsize=13, fontweight="bold")
    save(fig, output, "02_coordinate_accuracy")

    # 3. Parity for representative decision-relevant and difficult outputs.
    preferred = ["overflow_flow:S_NH4", "overflow_flow:S_PO4", "overflow_flow:X_S",
                 "underflow_flow:X_H", "clarifier_layer_5:TSS", "clarifier_layer_10:TSS"]
    selected = [names.index(n) for n in preferred if n in names]
    if len(selected) < 6:
        selected = list(np.argsort(projected_r2)[[0, 20, 60, 100, 140, -1]])
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.7), constrained_layout=True)
    for ax, j in zip(axes.flat, selected):
        lo = min(truth[:, j].min(), projected[:, j].min())
        hi = max(truth[:, j].max(), projected[:, j].max())
        ax.scatter(truth[:, j], projected[:, j], s=8, alpha=.35, color=COLORS["projected"], edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color="#ef4444", ls="--", lw=1)
        ax.set_title(names[j].replace("_flow", "").replace(":", " · "))
        ax.text(.03, .95, f"R² = {projected_r2[j]:.3f}\nNRMSE = {projected_nrmse[j]:.3f}", transform=ax.transAxes, va="top",
                bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 2})
        ax.set_xlabel("Mechanistic")
        ax.set_ylabel("Projected surrogate")
    fig.suptitle("Representative untouched-test output parity", fontsize=13, fontweight="bold")
    save(fig, output, "03_representative_outputs")

    detail = pd.DataFrame({"coordinate": names, "raw_r2": raw_r2, "projected_r2": projected_r2,
                           "raw_nrmse": raw_nrmse, "projected_nrmse": projected_nrmse,
                           "projected_rmse": projected_rmse})
    detail.sort_values("projected_r2").to_csv(output / "coordinate_fidelity.csv", index=False)
    print(output)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
