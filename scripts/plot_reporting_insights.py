"""Build decision-oriented charts from the article-v3 reporting tables."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closed_loop.model import COMPOSITE_MATRIX
from closed_loop.v3_smooth import DEFAULT_OBJECTIVE_WEIGHTS, QUALITY_WEIGHTS


BLUE, ORANGE, GREEN, RED = "#2563eb", "#d97706", "#16a34a", "#dc2626"
SHARED_RESPONSE_COUNT = 160
REDUCED_RESPONSE_COUNT = 161
CLARIFIER_VOLUME_M3 = 6_000.0


def style() -> None:
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 220, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "grid.linewidth": .6})


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    for extension in ("png", "svg"):
        fig.savefig(output / f"{stem}.{extension}", bbox_inches="tight")
    plt.close(fig)


def score(truth: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    r2 = 1 - float(np.sum((pred - truth) ** 2)) / ss_tot
    nrmse = float(np.sqrt(np.mean((pred - truth) ** 2)) / np.ptp(truth))
    return r2, nrmse


def reduced_response(values: np.ndarray) -> np.ndarray:
    """Return the 161-coordinate response, accepting full mechanistic rows."""

    response = np.asarray(values, dtype=float)
    if response.ndim != 2:
        raise ValueError("response arrays must be two-dimensional")
    if response.shape[1] == REDUCED_RESPONSE_COUNT:
        return response
    layer_count = response.shape[1] - SHARED_RESPONSE_COUNT
    if layer_count < 3:
        raise ValueError(
            "response arrays must contain 161 reduced coordinates or a full "
            "mechanistic response with at least three Clarifier layers"
        )
    inventory = (
        CLARIFIER_VOLUME_M3 / layer_count
        * np.sum(response[:, SHARED_RESPONSE_COUNT:], axis=1)
    )
    return np.column_stack((response[:, :SHARED_RESPONSE_COUNT], inventory))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve(); tables = run / "report" / "tables"
    output = (args.output or run / "report" / "decision_insights").resolve()
    output.mkdir(parents=True, exist_ok=True); style()

    # 1. Timing: distinguish the primary solve from the end-to-end route cost.
    timing = pd.read_csv(tables / "timing_summary.csv")
    wanted = ["surrogate_primary_optimization", "surrogate_complete_optimization",
              "direct_primary_optimization", "direct_complete_optimization", "surrogate_exact_reference", "direct_exact_reference"]
    t = timing.set_index("category").loc[wanted].reset_index()
    labels = ["Surrogate\nprimary", "Surrogate\ncomplete", "Smooth NLP\nprimary", "Smooth NLP\ncomplete", "Surrogate\nreference replay", "Smooth NLP\nreference replay"]
    colors = [ORANGE, ORANGE, BLUE, BLUE, "#94a3b8", "#94a3b8"]
    fig, ax = plt.subplots(figsize=(10.8, 5.2), constrained_layout=True)
    bars = ax.bar(np.arange(len(t)), t["median"], yerr=[t["median"] - (t["median"] - t["iqr"] / 2).clip(lower=0), t["iqr"] / 2], color=colors, capsize=3)
    for bar, row in zip(bars, t.itertuples(), strict=True): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.35, f"{row.median:.1f}s", ha="center", va="bottom")
    ax.set(xticks=np.arange(len(t)), xticklabels=labels, ylabel="Median seconds per robustness case", title="Computational duration by optimization route and validation step")
    ax.text(.01, .97, "Primary optimization: surrogate is faster; complete route includes surrogate local certification.", transform=ax.transAxes, va="top", fontsize=8)
    save(fig, output, "01_computational_timing")

    # Derive the objective's effluent composites directly from the shared
    # response coordinates; raw/projected rows intentionally have no layers.
    with np.load(run / "datasets" / "effective_design.npz") as design, np.load(run / "datasets" / "development" / "mechanistic_accepted_v3.npz") as development, np.load(run / "predictions" / "post_selection_holdout.npz") as test:
        theta = design["test_decisions"]
        responses = {
            "Mechanistic": reduced_response(test["mechanistic"]),
            "Raw surrogate": reduced_response(test["raw"]),
            "Projected surrogate": reduced_response(test["projected"]),
        }
        development_theta = design["development_decisions"]
        development_response = reduced_response(development["targets"])
    development_effluent = (
        development_response[:, 120:140]
        / (1.0 - development_theta[:, 6])[:, None]
    )
    quality_scale = np.std(
        development_effluent @ COMPOSITE_MATRIX.T, axis=0, ddof=0,
    )
    if np.any(quality_scale <= 0.0):
        raise ValueError("development effluent-composite scales must be positive")
    composite_names = ["COD", "TN", "TP", "TSS", "Quality objective"]
    composites: dict[str, np.ndarray] = {}
    for label, response in responses.items():
        effluent = response[:, 120:140] / (1.0 - theta[:, 6])[:, None]
        effluent_composites = effluent @ COMPOSITE_MATRIX.T
        values = np.column_stack((
            effluent_composites,
            (effluent_composites / quality_scale) @ QUALITY_WEIGHTS,
        ))
        composites[label] = values
    records = []
    for method in ("Raw surrogate", "Projected surrogate"):
        for j, name in enumerate(composite_names):
            r2, nrmse = score(composites["Mechanistic"][:, j], composites[method][:, j])
            records.append({"method": method, "composite": name, "r2": r2, "nrmse": nrmse})
    composite_metrics = pd.DataFrame(records); composite_metrics.to_csv(output / "effluent_composite_fidelity.csv", index=False)

    # 2. Composite fidelity, including parity for the deployed projected surrogate.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), constrained_layout=True)
    pivot = composite_metrics.pivot(index="composite", columns="method", values="r2").reindex(composite_names)
    x = np.arange(len(composite_names)); width = .36
    axes[0].bar(x-width/2, pivot["Raw surrogate"], width, color=ORANGE, label="Raw")
    axes[0].bar(x+width/2, pivot["Projected surrogate"], width, color=BLUE, label="Projected")
    axes[0].axhline(.9, color=GREEN, ls="--", lw=1, label="R² = 0.90")
    axes[0].set(xticks=x, xticklabels=composite_names, ylabel="R²", ylim=(-.05, 1.05), title="A. Derived effluent-composite fidelity")
    axes[0].legend(frameon=False)
    q_truth, q_pred = composites["Mechanistic"][:, 4], composites["Projected surrogate"][:, 4]
    lo, hi = min(q_truth.min(), q_pred.min()), max(q_truth.max(), q_pred.max())
    axes[1].hexbin(q_truth, q_pred, gridsize=48, bins="log", mincnt=1, cmap="viridis")
    axes[1].plot([lo, hi], [lo, hi], ls="--", color=RED, lw=1)
    q_r2, q_nrmse = score(q_truth, q_pred)
    axes[1].set(xlabel="Mechanistic quality component", ylabel="Projected surrogate quality component", title=f"B. Quality-component parity (R²={q_r2:.3f}; NRMSE={q_nrmse:.3f})")
    save(fig, output, "02_effluent_composite_fidelity")

    # 3. Per-plant-block response fidelity from the reporting metrics.
    metrics = pd.read_csv(tables / "prediction_metrics.csv")
    blocks = metrics[(metrics.coordinate == "ALL") & (metrics.block != "complete_response") & metrics.method.isin(["raw", "projected"])].copy()
    block_order = ["mixer", *[f"reactor_{i}" for i in range(1, 6)], "clarifier_overflow", "clarifier_underflow", "clarifier_inventory"]
    blocks["block"] = pd.Categorical(blocks["block"], block_order, ordered=True); blocks = blocks.sort_values("block")
    label_map = {
        "clarifier_overflow": "overflow",
        "clarifier_underflow": "underflow",
        "clarifier_inventory": "inventory",
    }
    labels = [
        label_map.get(str(value), str(value).replace("reactor_", "R"))
        for value in blocks[blocks.method == "projected"].block
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), constrained_layout=True, sharex=True)
    for method, color, label in (("raw", ORANGE, "Raw"), ("projected", BLUE, "Projected")):
        data = blocks[blocks.method == method]
        axes[0].plot(np.arange(len(data)), data.r2_mean, "o-", color=color, label=label)
        axes[1].plot(np.arange(len(data)), data.nrmse, "o-", color=color, label=label)
    axes[0].axhline(.9, color=GREEN, ls="--", lw=1); axes[0].set(ylabel="Block-average R²", title="A. Fidelity by plant component")
    axes[0].legend(frameon=False); axes[1].set(yscale="log", ylabel="Block NRMSE", title="B. Range-normalized block error", xticks=np.arange(len(labels)), xticklabels=labels)
    axes[1].tick_params(axis="x", rotation=35)
    save(fig, output, "03_fidelity_by_plant_component")

    # 4/5. Optimization economic burden vs. exact-replay quality component.
    comparison = pd.read_csv(run / "report" / "optimization_route_comparison" / "exact_replay_route_comparison.csv")
    economic_s = (DEFAULT_OBJECTIVE_WEIGHTS[1] * comparison.surrogate_component_hrt + DEFAULT_OBJECTIVE_WEIGHTS[2] * comparison.surrogate_component_aeration + DEFAULT_OBJECTIVE_WEIGHTS[3] * comparison.surrogate_component_internal_recycle + DEFAULT_OBJECTIVE_WEIGHTS[4] * comparison.surrogate_component_return_sludge + DEFAULT_OBJECTIVE_WEIGHTS[5] * comparison.surrogate_component_wasting)
    economic_d = (DEFAULT_OBJECTIVE_WEIGHTS[1] * comparison.direct_component_hrt + DEFAULT_OBJECTIVE_WEIGHTS[2] * comparison.direct_component_aeration + DEFAULT_OBJECTIVE_WEIGHTS[3] * comparison.direct_component_internal_recycle + DEFAULT_OBJECTIVE_WEIGHTS[4] * comparison.direct_component_return_sludge + DEFAULT_OBJECTIVE_WEIGHTS[5] * comparison.direct_component_wasting)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
    for ax, s, d, title in ((axes[0], economic_s, economic_d, "Economic / operating burden"), (axes[1], comparison.surrogate_component_quality, comparison.direct_component_quality, "Effluent-quality objective component")):
        for i in range(len(comparison)):
            ax.plot([0, 1], [s.iloc[i], d.iloc[i]], color="#94a3b8", lw=.8)
        ax.scatter(np.zeros(len(s)), s, color=ORANGE, label="Surrogate decision", zorder=3)
        ax.scatter(np.ones(len(d)), d, color=BLUE, label="Smooth NLP decision", zorder=3)
        ax.set(xlim=(-.35, 1.35), xticks=[0, 1], xticklabels=["Surrogate", "Smooth NLP"], ylabel="Lower is better", title=title)
    axes[0].legend(frameon=False)
    fig.suptitle("Which optimization decision is preferable? Exact-replay paired comparison", fontsize=13, fontweight="bold")
    save(fig, output, "04_optimization_economics_and_quality")

    # Pairwise Pareto view makes the quality-versus-operating trade-off explicit.
    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    for i, row in comparison.iterrows():
        ax.plot([economic_s.iloc[i], economic_d.iloc[i]], [row.surrogate_component_quality, row.direct_component_quality], color="#94a3b8", lw=.8)
    ax.scatter(economic_s, comparison.surrogate_component_quality, color=ORANGE, label="Surrogate decision")
    ax.scatter(economic_d, comparison.direct_component_quality, color=BLUE, label="Smooth NLP decision")
    ax.set(xlabel="Weighted economic / operating burden (lower is better)", ylabel="Quality component (lower is better)", title="Optimization trade-off frontier across cases")
    ax.legend(frameon=False)
    save(fig, output, "05_optimization_tradeoff_frontier")

    summary = pd.DataFrame([{"projected_quality_component_r2": q_r2, "projected_quality_component_nrmse": q_nrmse,
        "surrogate_primary_median_seconds": float(t.loc[0, "median"]), "direct_primary_median_seconds": float(t.loc[2, "median"]),
        "surrogate_complete_median_seconds": float(t.loc[1, "median"]), "direct_complete_median_seconds": float(t.loc[3, "median"]),
        "surrogate_lower_economic_burden_cases": int((economic_s < economic_d).sum()),
        "surrogate_lower_quality_component_cases": int((comparison.surrogate_component_quality < comparison.direct_component_quality).sum())}])
    summary.to_csv(output / "insight_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
