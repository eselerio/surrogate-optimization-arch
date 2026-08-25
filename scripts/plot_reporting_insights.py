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
ROUTE_COLORS = {"surrogate": ORANGE, "shared_unit": GREEN, "direct": BLUE}
ROUTE_LABELS = {
    "surrogate": "System surrogate", "shared_unit": "Shared-unit surrogate",
    "direct": "Smooth NLP",
}
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
    routes = ["surrogate", "direct"]
    if timing["category"].astype(str).str.startswith("shared_unit_").any():
        routes = ["surrogate", "shared_unit", "direct"]
    wanted = [
        f"{route}_{phase}"
        for route in routes
        for phase in ("primary_optimization", "complete_optimization", "exact_reference")
    ]
    available = timing.set_index("category")
    wanted = [category for category in wanted if category in available.index]
    t = available.loc[wanted].reset_index()
    phase_label = {
        "primary_optimization": "primary", "complete_optimization": "complete",
        "exact_reference": "reference replay",
    }
    labels, colors = [], []
    for category in wanted:
        route = next(item for item in routes if category.startswith(f"{item}_"))
        phase = category.removeprefix(f"{route}_")
        labels.append(f"{ROUTE_LABELS[route]}\n{phase_label[phase]}")
        colors.append(ROUTE_COLORS[route] if phase != "exact_reference" else "#94a3b8")
    fig, ax = plt.subplots(figsize=(10.8, 5.2), constrained_layout=True)
    bars = ax.bar(np.arange(len(t)), t["median"], yerr=[t["median"] - (t["median"] - t["iqr"] / 2).clip(lower=0), t["iqr"] / 2], color=colors, capsize=3)
    for bar, row in zip(bars, t.itertuples(), strict=True): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.35, f"{row.median:.1f}s", ha="center", va="bottom")
    ax.set(xticks=np.arange(len(t)), xticklabels=labels, ylabel="Median seconds per robustness case", title="Computational duration by optimization route and validation step")
    ax.text(.01, .97, "Primary, complete-route, and exact-replay costs are shown separately.", transform=ax.transAxes, va="top", fontsize=8)
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
    response_theta = {label: theta for label in responses}
    fidelity_pairs = [
        ("Raw surrogate", "Mechanistic"),
        ("Projected surrogate", "Mechanistic"),
    ]
    shared_path = run / "predictions" / "shared_unit_post_selection_holdout.npz"
    if shared_path.is_file():
        with np.load(shared_path, allow_pickle=False) as shared:
            raw_key = "raw_predictions" if "raw_predictions" in shared.files else "raw"
            projected_key = (
                "projected_predictions"
                if "projected_predictions" in shared.files else "projected"
            )
            shared_raw = np.asarray(shared[raw_key], dtype=float)
            shared_projected = np.asarray(shared[projected_key], dtype=float)
            available = np.asarray(
                shared["available"] if "available" in shared.files
                else np.ones(len(shared_raw), dtype=bool), dtype=bool,
            )
            shared_theta = np.asarray(
                shared["decisions"] if "decisions" in shared.files else theta,
                dtype=float,
            )
        if available.shape != (len(theta),):
            raise ValueError("shared-unit holdout availability mask does not align")
        available &= np.all(np.isfinite(shared_raw), axis=1)
        available &= np.all(np.isfinite(shared_projected), axis=1)
        if np.any(available):
            responses["Mechanistic shared-unit"] = responses["Mechanistic"][available]
            responses["Raw shared-unit"] = reduced_response(shared_raw[available])
            responses["Projected shared-unit"] = reduced_response(shared_projected[available])
            response_theta["Mechanistic shared-unit"] = shared_theta[available]
            response_theta["Raw shared-unit"] = shared_theta[available]
            response_theta["Projected shared-unit"] = shared_theta[available]
            fidelity_pairs.extend([
                ("Raw shared-unit", "Mechanistic shared-unit"),
                ("Projected shared-unit", "Mechanistic shared-unit"),
            ])
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
        method_theta = response_theta[label]
        effluent = response[:, 120:140] / (1.0 - method_theta[:, 6])[:, None]
        effluent_composites = effluent @ COMPOSITE_MATRIX.T
        values = np.column_stack((
            effluent_composites,
            (effluent_composites / quality_scale) @ QUALITY_WEIGHTS,
        ))
        composites[label] = values
    records = []
    for method, truth_method in fidelity_pairs:
        for j, name in enumerate(composite_names):
            r2, nrmse = score(composites[truth_method][:, j], composites[method][:, j])
            records.append({"method": method, "composite": name, "r2": r2, "nrmse": nrmse})
    composite_metrics = pd.DataFrame(records); composite_metrics.to_csv(output / "effluent_composite_fidelity.csv", index=False)

    # 2. Composite fidelity, including parity for the deployed projected surrogate.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), constrained_layout=True)
    pivot = composite_metrics.pivot(index="composite", columns="method", values="r2").reindex(composite_names)
    x = np.arange(len(composite_names)); width = .8 / len(pivot.columns)
    metric_colors = {
        "Raw surrogate": ORANGE, "Projected surrogate": BLUE,
        "Raw shared-unit": "#65a30d", "Projected shared-unit": GREEN,
    }
    offsets = (np.arange(len(pivot.columns)) - (len(pivot.columns) - 1) / 2) * width
    for offset, method in zip(offsets, pivot.columns, strict=True):
        axes[0].bar(x + offset, pivot[method], width, color=metric_colors[method], label=method)
    axes[0].axhline(.9, color=GREEN, ls="--", lw=1, label="R² = 0.90")
    axes[0].set(xticks=x, xticklabels=composite_names, ylabel="R²", ylim=(-.05, 1.05), title="A. Derived effluent-composite fidelity")
    axes[0].legend(frameon=False)
    q_truth, q_pred = composites["Mechanistic"][:, 4], composites["Projected surrogate"][:, 4]
    lo, hi = min(q_truth.min(), q_pred.min()), max(q_truth.max(), q_pred.max())
    axes[1].hexbin(q_truth, q_pred, gridsize=48, bins="log", mincnt=1, cmap="viridis")
    if "Projected shared-unit" in composites:
        shared_truth = composites["Mechanistic shared-unit"][:, 4]
        shared_pred = composites["Projected shared-unit"][:, 4]
        axes[1].scatter(
            shared_truth, shared_pred, s=7, alpha=.2, color=GREEN,
            label="Shared-unit projected",
        )
        axes[1].legend(frameon=False)
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

    # 4/5. Optimization burden and quality for every available route. Missing
    # route artifacts remain NaN and therefore do not invalidate other pairs.
    comparison = pd.read_csv(
        run / "report" / "optimization_route_comparison"
        / "exact_replay_route_comparison.csv"
    )
    comparison_routes = [
        route for route in routes
        if f"{route}_component_quality" in comparison.columns
    ]
    economic: dict[str, pd.Series] = {}
    for route in comparison_routes:
        economic[route] = sum(
            DEFAULT_OBJECTIVE_WEIGHTS[index]
            * comparison[f"{route}_component_{name}"]
            for index, name in enumerate(
                ("hrt", "aeration", "internal_recycle", "return_sludge", "wasting"),
                start=1,
            )
        )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
    positions = np.arange(len(comparison_routes))
    quantities = (
        (economic, "Economic / operating burden"),
        ({route: comparison[f"{route}_component_quality"] for route in comparison_routes},
         "Effluent-quality objective component"),
    )
    for ax, (values_by_route, title) in zip(axes, quantities, strict=True):
        for row_index in range(len(comparison)):
            values = [values_by_route[route].iloc[row_index] for route in comparison_routes]
            ax.plot(positions, values, color="#94a3b8", lw=.7)
        for position, route in enumerate(comparison_routes):
            ax.scatter(
                np.full(len(comparison), position), values_by_route[route],
                color=ROUTE_COLORS[route], label=ROUTE_LABELS[route], zorder=3,
            )
        ax.set(xticks=positions, xticklabels=[ROUTE_LABELS[item] for item in comparison_routes],
               ylabel="Lower is better", title=title)
    axes[0].legend(frameon=False)
    fig.suptitle("Exact-replay comparison of selected optimization decisions", fontsize=13, fontweight="bold")
    save(fig, output, "04_optimization_economics_and_quality")

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    for row_index in range(len(comparison)):
        ax.plot(
            [economic[route].iloc[row_index] for route in comparison_routes],
            [comparison[f"{route}_component_quality"].iloc[row_index] for route in comparison_routes],
            color="#94a3b8", lw=.7,
        )
    for route in comparison_routes:
        ax.scatter(
            economic[route], comparison[f"{route}_component_quality"],
            color=ROUTE_COLORS[route], label=ROUTE_LABELS[route],
        )
    ax.set(xlabel="Weighted economic / operating burden (lower is better)", ylabel="Quality component (lower is better)", title="Optimization trade-off frontier across cases")
    ax.legend(frameon=False)
    save(fig, output, "05_optimization_tradeoff_frontier")

    timing_lookup = timing.set_index("category")["median"].to_dict()
    summary_row: dict[str, object] = {
        "projected_quality_component_r2": q_r2,
        "projected_quality_component_nrmse": q_nrmse,
    }
    for route in routes:
        summary_row[f"{route}_primary_median_seconds"] = timing_lookup.get(
            f"{route}_primary_optimization", np.nan,
        )
        summary_row[f"{route}_complete_median_seconds"] = timing_lookup.get(
            f"{route}_complete_optimization", np.nan,
        )
    for left, right, symbol in (
        ("surrogate", "shared_unit", "S_U"),
        ("surrogate", "direct", "S_M"),
        ("shared_unit", "direct", "U_M"),
    ):
        if left in economic and right in economic:
            valid = economic[left].notna() & economic[right].notna()
            summary_row[f"comparable_cases_{symbol}"] = int(valid.sum())
            summary_row[f"left_lower_economic_cases_{symbol}"] = int(
                (economic[left][valid] < economic[right][valid]).sum()
            )
            left_quality = comparison[f"{left}_component_quality"]
            right_quality = comparison[f"{right}_component_quality"]
            summary_row[f"left_lower_quality_cases_{symbol}"] = int(
                (left_quality[valid] < right_quality[valid]).sum()
            )
    if "surrogate" in economic and "direct" in economic:
        valid = economic["surrogate"].notna() & economic["direct"].notna()
        summary_row["surrogate_lower_economic_burden_cases"] = int(
            (economic["surrogate"][valid] < economic["direct"][valid]).sum()
        )
        summary_row["surrogate_lower_quality_component_cases"] = int((
            comparison.loc[valid, "surrogate_component_quality"]
            < comparison.loc[valid, "direct_component_quality"]
        ).sum())
    summary = pd.DataFrame([summary_row])
    summary.to_csv(output / "insight_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
