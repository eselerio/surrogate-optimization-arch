"""Create post-selection-holdout charts for reduced-surrogate fidelity."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "raw": "#d97706", "projected": "#2563eb", "truth": "#111827",
    "shared_raw": "#65a30d", "shared_projected": "#16a34a",
}
REDUCED_RESPONSE_COUNT = 161


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

    metric_path = run / "metrics" / "post_selection_prediction_metrics.csv"
    stored = pd.read_csv(metric_path)
    coordinate_rows = stored.query(
        "method == 'projected' and block == 'complete_response' and coordinate != 'ALL'"
    ).reset_index(drop=True)
    names = coordinate_rows["coordinate"].tolist()
    if len(names) != REDUCED_RESPONSE_COUNT:
        raise ValueError(
            f"Expected {REDUCED_RESPONSE_COUNT} reduced-response coordinates, "
            f"found {len(names)}"
        )

    with np.load(run / "predictions" / "post_selection_holdout.npz") as arrays:
        truth = arrays["mechanistic"]
        raw = arrays["raw"]
        projected = arrays["projected"]
    if (
        truth.ndim != 2
        or truth.shape != raw.shape
        or truth.shape != projected.shape
        or truth.shape[1] != REDUCED_RESPONSE_COUNT
        or truth.shape[1] != len(names)
    ):
        raise ValueError("Prediction arrays and metric coordinate names do not align")

    methods = {"Raw ridge": raw, "Projected surrogate": projected}
    shared_prediction_path = run / "predictions" / "shared_unit_post_selection_holdout.npz"
    shared_raw: np.ndarray | None = None
    shared_projected: np.ndarray | None = None
    if shared_prediction_path.is_file():
        with np.load(shared_prediction_path, allow_pickle=False) as arrays:
            shared_truth = np.asarray(
                arrays["mechanistic"] if "mechanistic" in arrays.files
                else arrays["truth"] if "truth" in arrays.files else truth,
                dtype=float,
            )
            raw_key = "raw_predictions" if "raw_predictions" in arrays.files else "raw"
            projected_key = (
                "projected_predictions"
                if "projected_predictions" in arrays.files else "projected"
            )
            shared_raw = np.asarray(arrays[raw_key], dtype=float)
            shared_projected = np.asarray(arrays[projected_key], dtype=float)
            available = np.asarray(
                arrays["available"] if "available" in arrays.files
                else np.ones(len(shared_raw), dtype=bool),
                dtype=bool,
            )
        if shared_truth.shape != truth.shape:
            raise ValueError("shared-unit and system-surrogate holdouts do not align")
        if shared_raw.shape != truth.shape or shared_projected.shape != truth.shape:
            raise ValueError("shared-unit holdout predictions do not use the reduced schema")
        if available.shape != (len(truth),):
            raise ValueError("shared-unit holdout availability mask does not align")
        available &= np.all(np.isfinite(shared_raw), axis=1)
        available &= np.all(np.isfinite(shared_projected), axis=1)
        if not np.any(available):
            raise ValueError("shared-unit holdout contains no available finite rows")
        truth = truth[available]
        raw = raw[available]
        projected = projected[available]
        shared_truth = shared_truth[available]
        shared_raw = shared_raw[available]
        shared_projected = shared_projected[available]
        if not np.allclose(shared_truth, truth):
            raise ValueError("shared-unit and system-surrogate holdout targets differ")
        methods["Raw shared-unit"] = shared_raw
        methods["Projected shared-unit"] = shared_projected
    summary = []
    for label, pred in methods.items():
        _, nrmse, r2 = coordinate_metrics(truth, pred)
        summary.append({
            "method": label, "holdout_cases": truth.shape[0], "coordinates": truth.shape[1],
            "median_r2": np.nanmedian(r2), "mean_r2": np.nanmean(r2),
            "r2_ge_0_90_fraction": np.nanmean(r2 >= .90),
            "r2_ge_0_75_fraction": np.nanmean(r2 >= .75),
            "median_nrmse": np.nanmedian(nrmse), "p90_nrmse": np.nanpercentile(nrmse, 90),
        })
    pd.DataFrame(summary).to_csv(output / "fidelity_summary.csv", index=False)

    # 1. Dimensionless parity over all reduced-response holdout values.
    mean = truth.mean(axis=0)
    std = truth.std(axis=0)
    valid = std > 1e-14
    tz = ((truth[:, valid] - mean[valid]) / std[valid]).ravel()
    fig, axes = plt.subplots(
        1, len(methods), figsize=(5.4 * len(methods), 4.8),
        sharex=True, sharey=True, constrained_layout=True, squeeze=False,
    )
    lim = (-4, 4)
    for ax, (label, pred) in zip(axes.flat, methods.items(), strict=True):
        pz = ((pred[:, valid] - mean[valid]) / std[valid]).ravel()
        hb = ax.hexbin(tz, pz, gridsize=75, bins="log", mincnt=1, cmap="viridis", extent=(*lim, *lim))
        ax.plot(lim, lim, color="#ef4444", lw=1.2, ls="--", label="Perfect agreement")
        ax.set(xlim=lim, ylim=lim, xlabel="Mechanistic model (standardized)", title=label)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left", frameon=False)
        fig.colorbar(hb, ax=ax, label="log count")
    axes.flat[0].set_ylabel("Surrogate prediction (standardized)")
    fig.suptitle(f"Post-selection-holdout parity across {truth.shape[1]} outputs × {truth.shape[0]:,} cases", fontsize=13, fontweight="bold")
    save(fig, output, "01_all_output_parity")

    # 2. Coordinate-level R² and normalized error, sorted to reveal the tail.
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), constrained_layout=True)
    projected_rmse, projected_nrmse, projected_r2 = coordinate_metrics(truth, projected)
    raw_rmse, raw_nrmse, raw_r2 = coordinate_metrics(truth, raw)
    order = np.argsort(projected_r2)[::-1]
    x = np.arange(len(names))
    axes[0].plot(x, raw_r2[order], color=COLORS["raw"], lw=1.1, label="Raw system")
    axes[0].plot(x, projected_r2[order], color=COLORS["projected"], lw=1.4, label="Projected system")
    axes[0].axhline(.90, color="#16a34a", ls="--", lw=1, label="R² = 0.90")
    axes[0].axhline(.75, color="#64748b", ls=":", lw=1, label="R² = 0.75")
    axes[0].set(ylabel="R²", title="A. Explained variance by output coordinate", ylim=(min(-.1, np.nanmin(projected_r2)), 1.02))
    axes[0].legend(ncol=4, frameon=False, loc="lower left")
    axes[1].semilogy(x, raw_nrmse[order], color=COLORS["raw"], lw=1.1, label="Raw system")
    axes[1].semilogy(x, projected_nrmse[order], color=COLORS["projected"], lw=1.4, label="Projected system")
    if shared_raw is not None and shared_projected is not None:
        _, shared_raw_nrmse, shared_raw_r2 = coordinate_metrics(truth, shared_raw)
        _, shared_projected_nrmse, shared_projected_r2 = coordinate_metrics(
            truth, shared_projected,
        )
        axes[0].plot(x, shared_raw_r2[order], color=COLORS["shared_raw"], lw=1.0, ls=":", label="Raw shared-unit")
        axes[0].plot(x, shared_projected_r2[order], color=COLORS["shared_projected"], lw=1.3, label="Projected shared-unit")
        axes[1].semilogy(x, shared_raw_nrmse[order], color=COLORS["shared_raw"], lw=1.0, ls=":", label="Raw shared-unit")
        axes[1].semilogy(x, shared_projected_nrmse[order], color=COLORS["shared_projected"], lw=1.3, label="Projected shared-unit")
    axes[0].legend(ncol=3, frameon=False, loc="lower left")
    axes[1].legend(ncol=2, frameon=False)
    axes[1].set(xlabel="Output coordinates (sorted by projected R²)", ylabel="NRMSE (RMSE / holdout range)", title="B. Range-normalized prediction error")
    fig.suptitle("Accuracy distribution and weak-output tail", fontsize=13, fontweight="bold")
    save(fig, output, "02_coordinate_accuracy")

    # 3. Parity for representative decision-relevant and difficult outputs.
    preferred = ["overflow_flow:S_NH4", "overflow_flow:S_PO4", "overflow_flow:X_S",
                 "underflow_flow:X_H", "reactor_5:X_H", "clarifier_inventory:TSS_mass"]
    selected = [names.index(n) for n in preferred if n in names]
    if len(selected) < 6:
        selected = list(np.argsort(projected_r2)[[0, 20, 60, 100, 140, -1]])
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.7), constrained_layout=True)
    for ax, j in zip(axes.flat, selected):
        lo = min(truth[:, j].min(), projected[:, j].min())
        hi = max(truth[:, j].max(), projected[:, j].max())
        ax.scatter(truth[:, j], projected[:, j], s=8, alpha=.35, color=COLORS["projected"], edgecolors="none")
        if shared_projected is not None:
            ax.scatter(
                truth[:, j], shared_projected[:, j], s=7, alpha=.25,
                color=COLORS["shared_projected"], edgecolors="none",
            )
        ax.plot([lo, hi], [lo, hi], color="#ef4444", ls="--", lw=1)
        ax.set_title(names[j].replace("_flow", "").replace(":", " · "))
        ax.text(.03, .95, f"R² = {projected_r2[j]:.3f}\nNRMSE = {projected_nrmse[j]:.3f}", transform=ax.transAxes, va="top",
                bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 2})
        ax.set_xlabel("Mechanistic")
        ax.set_ylabel("Projected surrogate")
    fig.suptitle("Representative post-selection-holdout output parity", fontsize=13, fontweight="bold")
    save(fig, output, "03_representative_outputs")

    detail = pd.DataFrame({"coordinate": names, "raw_r2": raw_r2, "projected_r2": projected_r2,
                           "raw_nrmse": raw_nrmse, "projected_nrmse": projected_nrmse,
                           "projected_rmse": projected_rmse})
    if shared_raw is not None and shared_projected is not None:
        shared_projected_rmse, shared_projected_nrmse, shared_projected_r2 = coordinate_metrics(
            truth, shared_projected,
        )
        _, shared_raw_nrmse, shared_raw_r2 = coordinate_metrics(truth, shared_raw)
        detail = detail.assign(
            shared_unit_raw_r2=shared_raw_r2,
            shared_unit_projected_r2=shared_projected_r2,
            shared_unit_raw_nrmse=shared_raw_nrmse,
            shared_unit_projected_nrmse=shared_projected_nrmse,
            shared_unit_projected_rmse=shared_projected_rmse,
        )
    detail.sort_values("projected_r2").to_csv(output / "coordinate_fidelity.csv", index=False)
    print(output)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
