"""Compare retained surrogate and smooth-NLP optimization decisions on exact replays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closed_loop.manuscript_v3 import clarifier_for_layers
from closed_loop.v3_reporting import CONTROL_NAMES
from closed_loop.v3_smooth import (
    DEFAULT_OBJECTIVE_WEIGHTS, engineering_quantities, fit_direct_assets,
    objective_components,
)


BLUE, ORANGE, RED = "#2563eb", "#d97706", "#dc2626"


def selected_payload(case: Path, route: str) -> dict:
    data = json.loads((case / f"{route}.json").read_text())
    start = data["starts"][data["selected_start"]]
    return start.get("final", start)


def setup() -> None:
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 220, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "grid.linewidth": .6})


def save(fig: plt.Figure, out: Path, name: str) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(out / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("optimization", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.optimization.resolve()
    run = root.parent
    out = (args.output or run / "report" / "optimization_route_comparison").resolve()
    out.mkdir(parents=True, exist_ok=True)
    setup()

    # Objective scales are refit exactly as in the archived run, from the 4,000 development rows.
    with np.load(run / "datasets" / "effective_design.npz") as design:
        with np.load(run / "datasets" / "development" / "mechanistic_accepted_v3.npz") as generated:
            targets = generated["targets"]
        assets = fit_direct_assets(design["development_decisions"], design["development_influents"], targets,
                                   clarifier=clarifier_for_layers(10))
    rows, excluded = [], []
    for case in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [case / f"{route}_selected.npz" for route in ("surrogate", "direct")]
        if not all(path.exists() for path in files):
            excluded.append({"case": case.name, "reason": "one selected decision is unavailable"})
            continue
        values = {}
        valid = True
        for route, path in zip(("surrogate", "direct"), files, strict=True):
            with np.load(path) as data:
                theta = data["theta"].astype(float)
                reference = data["reference"].astype(float)
            if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(reference))):
                valid = False
                break
            components = objective_components(theta, reference, assets)
            values[route] = {"theta": theta, "reference": reference,
                             "components": components,
                             "objective": float(DEFAULT_OBJECTIVE_WEIGHTS @ components),
                             "engineering": engineering_quantities(theta, reference, assets)}
        if not valid:
            excluded.append({"case": case.name, "reason": "non-finite exact replay"})
            continue
        s, d = values["surrogate"], values["direct"]
        row = {"case": case.name, "J_surrogate_decision_exact": s["objective"],
               "J_direct_decision_exact": d["objective"],
               "delta_J_surrogate_minus_direct": s["objective"] - d["objective"],
               "relative_penalty_percent": 100 * (s["objective"] - d["objective"]) / d["objective"]}
        row["surrogate_economic_burden"] = float(DEFAULT_OBJECTIVE_WEIGHTS[1:] @ s["components"][1:])
        row["direct_economic_burden"] = float(DEFAULT_OBJECTIVE_WEIGHTS[1:] @ d["components"][1:])
        row["delta_economic_burden"] = row["surrogate_economic_burden"] - row["direct_economic_burden"]
        for i, name in enumerate(CONTROL_NAMES):
            row[f"surrogate_{name}"] = s["theta"][i]
            row[f"direct_{name}"] = d["theta"][i]
            span = max(abs(s["theta"][i]), abs(d["theta"][i]), 1.0)
            row[f"delta_normalized_{name}"] = (s["theta"][i] - d["theta"][i]) / span
        for i, name in enumerate(("quality", "hrt", "aeration", "internal_recycle", "return_sludge", "wasting")):
            row[f"surrogate_component_{name}"] = s["components"][i]
            row[f"direct_component_{name}"] = d["components"][i]
        for name in ("effluent_cod", "effluent_tn", "effluent_tp", "effluent_tss", "srt_d"):
            row[f"surrogate_{name}"] = s["engineering"][name]
            row[f"direct_{name}"] = d["engineering"][name]
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("case").reset_index(drop=True)
    frame.to_csv(out / "exact_replay_route_comparison.csv", index=False)
    pd.DataFrame(excluded).to_csv(out / "excluded_cases.csv", index=False)
    if frame.empty:
        raise RuntimeError("No cases with both selected decisions and finite exact replays")

    # Exact objective, evaluated by the shared smooth-NLP objective on exact mechanistic replays.
    x = np.arange(len(frame)); labels = frame["case"].str.replace("robustness_", "R", regex=False).str.replace("nominal", "Nominal", regex=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1]})
    axes[0].plot(x, frame["J_surrogate_decision_exact"], "o-", color=ORANGE, label="Surrogate decision")
    axes[0].plot(x, frame["J_direct_decision_exact"], "o-", color=BLUE, label="Smooth mechanistic-NLP decision")
    for i, row in frame.iterrows(): axes[0].plot([i, i], [row["J_surrogate_decision_exact"], row["J_direct_decision_exact"]], color="#94a3b8", lw=.8)
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Exact-replay objective (lower is better)", title="A. Decision quality on the common mechanistic basis")
    axes[0].legend(frameon=False)
    colors = np.where(frame["delta_J_surrogate_minus_direct"] <= 0, ORANGE, BLUE)
    axes[1].barh(x, frame["relative_penalty_percent"], color=colors)
    axes[1].axvline(0, color="#111827", lw=.8)
    axes[1].set(yticks=x, yticklabels=labels, xlabel="Surrogate decision penalty vs. smooth NLP (%)", title="B. Positive means smooth NLP is better")
    fig.suptitle("Optimization-route comparison on exact mechanistic replays", fontsize=13, fontweight="bold")
    save(fig, out, "01_exact_objective_comparison")

    # Difference in actual operating choices, using each variable's permitted range.
    from closed_loop.manuscript_v3 import DECISION_LOWER, DECISION_UPPER
    delta = np.column_stack([(frame[f"surrogate_{name}"] - frame[f"direct_{name}"]) / (DECISION_UPPER[i] - DECISION_LOWER[i]) for i, name in enumerate(CONTROL_NAMES)])
    fig, ax = plt.subplots(figsize=(10.5, 5.3), constrained_layout=True)
    vmax = max(.05, float(np.nanmax(np.abs(delta))))
    mesh = ax.imshow(delta, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set(xticks=np.arange(7), xticklabels=CONTROL_NAMES, yticks=x, yticklabels=labels,
           title="Decision difference: surrogate minus smooth mechanistic NLP", xlabel="Control", ylabel="Case")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]): ax.text(j, i, f"{delta[i,j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(mesh, ax=ax, label="Fraction of allowed control range")
    save(fig, out, "02_decision_difference_heatmap")

    # Outcomes generated by those decisions, re-evaluated exactly.
    outcomes = [("effluent_cod", "Effluent COD"), ("effluent_tn", "Effluent TN"), ("effluent_tp", "Effluent TP"), ("effluent_tss", "Effluent TSS")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (field, label) in zip(axes.flat, outcomes, strict=True):
        ax.plot(x, frame[f"surrogate_{field}"], "o-", color=ORANGE, label="Surrogate decision")
        ax.plot(x, frame[f"direct_{field}"], "o-", color=BLUE, label="Smooth NLP decision")
        ax.set(xticks=x, xticklabels=labels, title=label, ylabel="Exact mechanistic replay")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Process outcomes resulting from the two decisions", fontsize=13, fontweight="bold")
    save(fig, out, "03_exact_process_outcomes")

    # Economic / operating terms: HRT is a capacity proxy; the remaining terms are resource proxies.
    resource_names = ("HRT / capacity", "Aeration", "Internal recycle", "Return sludge", "Wasting")
    component_keys = ("hrt", "aeration", "internal_recycle", "return_sludge", "wasting")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1]})
    axes[0].plot(x, frame["surrogate_economic_burden"], "o-", color=ORANGE, label="Surrogate decision")
    axes[0].plot(x, frame["direct_economic_burden"], "o-", color=BLUE, label="Smooth NLP decision")
    for i, row in frame.iterrows(): axes[0].plot([i, i], [row["surrogate_economic_burden"], row["direct_economic_burden"]], color="#94a3b8", lw=.8)
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Weighted resource / capacity burden", title="A. Non-quality objective contribution (lower is better)")
    axes[0].legend(frameon=False)
    differences = np.array([DEFAULT_OBJECTIVE_WEIGHTS[i + 1] * (frame[f"surrogate_component_{key}"] - frame[f"direct_component_{key}"]) for i, key in enumerate(component_keys)])
    axes[1].boxplot(differences.T, tick_labels=resource_names, showfliers=False)
    axes[1].axhline(0, color="#111827", lw=.8)
    axes[1].set(ylabel="Weighted difference: surrogate − smooth NLP", title="B. Paired resource-term differences")
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("Economic and operating-burden comparison on exact replays", fontsize=13, fontweight="bold")
    save(fig, out, "04_economic_operating_burden")

    # All six objective components, including water quality, on their native normalized scales.
    all_names = ("Water quality", "HRT / capacity", "Aeration", "Internal recycle", "Return sludge", "Wasting")
    all_keys = ("quality", "hrt", "aeration", "internal_recycle", "return_sludge", "wasting")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
    for ax, name, key, weight in zip(axes.flat, all_names, all_keys, DEFAULT_OBJECTIVE_WEIGHTS, strict=True):
        s_col, d_col = f"surrogate_component_{key}", f"direct_component_{key}"
        ax.plot(x, frame[s_col], "o-", color=ORANGE, label="Surrogate decision")
        ax.plot(x, frame[d_col], "o-", color=BLUE, label="Smooth NLP decision")
        for i, row in frame.iterrows():
            ax.plot([i, i], [row[s_col], row[d_col]], color="#94a3b8", lw=.7)
        median_delta = float((frame[s_col] - frame[d_col]).median())
        ax.set(xticks=x, xticklabels=labels, title=f"{name} (weight {weight:.2f})", ylabel="Normalized component")
        ax.text(.02, .95, f"median ΔS−M = {median_delta:+.3f}", transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 2})
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("All objective components on exact mechanistic replays", fontsize=13, fontweight="bold")
    save(fig, out, "05_all_objective_components")

    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    weighted_deltas = np.array([
        DEFAULT_OBJECTIVE_WEIGHTS[i] * (frame[f"surrogate_component_{key}"] - frame[f"direct_component_{key}"])
        for i, key in enumerate(all_keys)
    ])
    ax.boxplot(weighted_deltas.T, tick_labels=all_names, showfliers=False)
    ax.axhline(0, color="#111827", lw=.8)
    ax.set(ylabel="Weighted difference: surrogate − smooth NLP", title="How each objective component contributes to the total-objective gap")
    ax.tick_params(axis="x", rotation=22)
    save(fig, out, "06_weighted_component_differences")

    wins = int((frame["delta_J_surrogate_minus_direct"] < 0).sum())
    summary = pd.DataFrame([{"comparable_cases": len(frame), "surrogate_wins": wins,
        "smooth_nlp_wins": len(frame)-wins, "median_surrogate_penalty_percent": frame["relative_penalty_percent"].median(),
        "mean_surrogate_penalty_percent": frame["relative_penalty_percent"].mean(),
        "surrogate_lower_economic_burden_cases": int((frame["delta_economic_burden"] < 0).sum()),
        "median_surrogate_economic_burden_difference": frame["delta_economic_burden"].median(),
        "excluded_cases": len(excluded)}])
    summary.to_csv(out / "comparison_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
