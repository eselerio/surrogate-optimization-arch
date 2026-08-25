"""Generate the twelve requested article-v3 result-comparison charts."""

from __future__ import annotations

from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from closed_loop.model import COMPONENTS, COMPOSITE_MATRIX, TSS_VECTOR


RUN = ROOT / "results" / "article_v3" / "article_full_50000_reduced_001"
TABLES = RUN / "report" / "tables"
OUT = RUN / "report" / "figures" / "requested_comparisons"
OUT.mkdir(parents=True, exist_ok=True)

RAW = "#D97904"
PROJECTED = "#147D92"
SURROGATE = "#147D92"
SHARED_UNIT = "#2E8B57"
DIRECT = "#7A5195"
REFERENCE = "#343A40"
BAD = "#F6D7D7"
GRID = "#D8DEE4"
ROUTE_ORDER = ("surrogate", "shared_unit", "direct")
ROUTE_COLOR = {
    "surrogate": SURROGATE, "shared_unit": SHARED_UNIT, "direct": DIRECT,
}
ROUTE_LABEL = {
    "surrogate": "System surrogate", "shared_unit": "Shared-unit surrogate",
    "direct": "Smooth NLP",
}
ROUTE_MARKER = {"surrogate": "o", "shared_unit": "^", "direct": "s"}

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 240,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.alpha": 0.55,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def annotate_bars(ax: plt.Axes, bars, fmt: str = ".3f") -> None:
    for bar in bars:
        value = bar.get_height()
        if np.isfinite(value):
            ax.annotate(
                format(value, fmt),
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, rotation=0,
            )


def shade_ineligible(ax: plt.Axes, eligible: np.ndarray) -> None:
    for index, valid in enumerate(eligible):
        if not bool(valid):
            ax.axvspan(index - 0.5, index + 0.5, color=BAD, alpha=0.55, zorder=0)


def route_parity_chart(route: str, prediction_method: str, stem: str, title: str) -> dict[str, float]:
    subset = quality[
        (quality["decision_route"] == route)
        & (quality["response_method"].isin([prediction_method, "reference"]))
        & quality["available"].astype(bool)
    ]
    pivot = subset.pivot(index="case", columns="response_method", values=composites)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    errors: list[float] = []
    r2_values: list[float] = []
    for ax, component in zip(axes.flat, composites, strict=True):
        x = pivot[(component, "reference")].to_numpy(float)
        y = pivot[(component, prediction_method)].to_numpy(float)
        limits = [min(x.min(), y.min()), max(x.max(), y.max())]
        pad = max(1.0e-9, 0.06 * (limits[1] - limits[0]))
        limits = [limits[0] - pad, limits[1] + pad]
        ax.plot(limits, limits, color=REFERENCE, lw=1.2, ls="--", label="perfect match")
        ax.scatter(x, y, s=42, color=ROUTE_COLOR[route],
                   edgecolor="white", linewidth=0.6, zorder=3)
        for label, xx, yy in zip([case_label[c] for c in pivot.index], x, y, strict=True):
            ax.annotate(label, (xx, yy), xytext=(3, 2), textcoords="offset points", fontsize=6)
        abs_percent = np.abs(y - x) / np.maximum(np.abs(x), 1.0e-12) * 100.0
        r2 = 1.0 - np.sum((y - x) ** 2) / np.sum((x - np.mean(x)) ** 2)
        errors.extend(abs_percent.tolist())
        r2_values.append(float(r2))
        ax.set_xlim(limits); ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{component}: median |error| = {np.median(abs_percent):.1f}%")
        ax.set_xlabel("Exact mechanistic replay")
        ax.set_ylabel(f"{prediction_method.capitalize()} prediction")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94))
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.text(0.5, 0.01, "Each point is the nominal or one robustness case; closer to the dashed line is better.", ha="center")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    save(fig, stem)
    return {
        "median_absolute_percent_error": float(np.median(errors)),
        "mean_absolute_percent_error": float(np.mean(errors)),
        "mean_component_r2": float(np.mean(r2_values)),
    }


prediction = pd.read_csv(TABLES / "prediction_metrics.csv")
quality = pd.read_csv(TABLES / "selected_quality.csv")
controls = pd.read_csv(TABLES / "scenario_controls.csv")
comparison = pd.read_csv(TABLES / "scenario_comparison.csv")
timing = pd.read_csv(RUN / "metrics" / "robustness_case_timing.csv")

present_routes = set(quality.get("decision_route", pd.Series(dtype=str)).astype(str))
present_routes.update(controls.get("route", pd.Series(dtype=str)).astype(str))
routes = tuple(
    route for route in ROUTE_ORDER
    if route in present_routes or any(
        (RUN / "optimization").glob(f"*/{route}_selected.npz")
    )
)
if "shared_unit" not in routes:
    routes = ("surrogate", "direct")

robust_cases = [f"robustness_{index:02d}" for index in range(1, 11)]
all_cases = ["nominal", *robust_cases]
case_label = {"nominal": "N", **{case: f"R{i}" for i, case in enumerate(robust_cases, 1)}}
composites = ["COD", "TN", "TP", "TSS"]
summary_rows: list[dict[str, object]] = []

# 1. Complete-response composite prediction.
overall = prediction[
    prediction["method"].isin(["raw", "projected"])
    & prediction["block"].eq("complete_response")
    & prediction["coordinate"].eq("ALL")
].set_index("method")
metrics = [("nrmse", "Normalized RMSE", True), ("nmae", "Normalized MAE", True), ("r2_mean", "Mean $R^2$", False)]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
for ax, (column, label, lower_better) in zip(axes, metrics, strict=True):
    values = overall.loc[["raw", "projected"], column].to_numpy(float)
    bars = ax.bar(["Raw", "Projected"], values, color=[RAW, PROJECTED], width=0.62)
    annotate_bars(ax, bars)
    ax.set_title(label + (" (lower is better)" if lower_better else " (higher is better)"))
    ax.set_ylabel(label)
    ax.set_ylim(0, max(values) * 1.22)
fig.suptitle("Q1. Reduced-response prediction on the 3,343-row post-selection holdout", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.92))
save(fig, "q01_holdout_composite_accuracy")
q1_improvement = 100.0 * (overall.loc["raw", "nrmse"] - overall.loc["projected", "nrmse"]) / overall.loc["raw", "nrmse"]
summary_rows.append({"question": 1, "metric": "projection_nrmse_improvement_percent", "value": q1_improvement})

# 2. Block-level reduced-response prediction. The Clarifier contributes outlet
# component flows and one inventory coordinate, never a surrogate layer profile.
stage_blocks = [
    "mixer",
    *[f"reactor_{i}" for i in range(1, 6)],
    "clarifier_overflow",
    "clarifier_underflow",
    "clarifier_inventory",
]
stage_names = [
    "Mixer",
    *[f"Reactor {i}" for i in range(1, 6)],
    "Overflow",
    "Underflow",
    "Inventory",
]
stage = prediction[
    prediction["method"].isin(["raw", "projected"])
    & prediction["block"].isin(stage_blocks)
    & prediction["coordinate"].eq("ALL")
].pivot(index="block", columns="method", values="nrmse").loc[stage_blocks]
x = np.arange(len(stage))
fig, (ax, delta_ax) = plt.subplots(2, 1, figsize=(10.5, 7), gridspec_kw={"height_ratios": [2, 1]})
width = 0.36
ax.bar(x - width / 2, stage["raw"], width, color=RAW, label="Raw")
ax.bar(x + width / 2, stage["projected"], width, color=PROJECTED, label="Projected")
ax.set_xticks(x, stage_names); ax.set_ylabel("Normalized RMSE")
ax.set_title("Response-block accuracy (lower is better)"); ax.legend(ncol=2)
change = 100.0 * (stage["projected"] - stage["raw"]) / stage["raw"]
delta_ax.bar(x, change, color=np.where(change <= 0, PROJECTED, "#C44E52"))
delta_ax.axhline(0, color=REFERENCE, lw=0.8)
delta_ax.set_xticks(x, stage_names); delta_ax.set_ylabel("Projection change (%)")
delta_ax.set_title("Negative values mean projection improved accuracy")
fig.suptitle("Q2. Prediction accuracy by reduced-response block", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "q02_holdout_accuracy_by_response_block")
summary_rows.append({"question": 2, "metric": "stages_improved_by_projection", "value": int((change < 0).sum())})

# 3. Component-level prediction aggregated over mixer/reactors/outlets.
process_blocks = ["mixer", *[f"reactor_{i}" for i in range(1, 6)], "clarifier_overflow", "clarifier_underflow"]
coordinate_rows = prediction[
    prediction["method"].isin(["raw", "projected"])
    & prediction["block"].eq("complete_response")
    & prediction["coordinate"].ne("ALL")
].copy()
coordinate_rows["component"] = coordinate_rows["coordinate"].str.split(":").str[-1]
coordinate_rows["stage_prefix"] = coordinate_rows["coordinate"].str.split(":").str[0]
allowed_prefixes = ["mixer", *[f"reactor_{i}" for i in range(1, 6)], "overflow_flow", "underflow_flow"]
coordinate_rows = coordinate_rows[coordinate_rows["stage_prefix"].isin(allowed_prefixes)]
component_metric = (
    coordinate_rows.groupby(["method", "component"])["nrmse"]
    .apply(lambda values: float(np.sqrt(np.mean(np.square(values)))))
    .unstack("method").loc[list(COMPONENTS)]
)
y = np.arange(len(component_metric))
fig, ax = plt.subplots(figsize=(9.5, 9.2))
ax.barh(y + 0.19, component_metric["raw"], 0.38, color=RAW, label="Raw")
ax.barh(y - 0.19, component_metric["projected"], 0.38, color=PROJECTED, label="Projected")
ax.set_yticks(y, component_metric.index); ax.invert_yaxis()
ax.set_xlabel("Stage-aggregated normalized RMSE (lower is better)")
ax.set_title("Q3. Component prediction across mixer, reactors, and Clarifier outlets")
ax.legend(ncol=2)
fig.tight_layout()
save(fig, "q03_holdout_component_accuracy")
summary_rows.append({"question": 3, "metric": "components_improved_by_projection", "value": int((component_metric["projected"] < component_metric["raw"]).sum())})

# 4. Component-level prediction by stage.
stage_prefixes = [*[f"reactor_{i}" for i in range(1, 6)], "overflow_flow", "underflow_flow"]
stage_labels = [*[f"R{i}" for i in range(1, 6)], "Overflow", "Underflow"]
stage_component = coordinate_rows[coordinate_rows["stage_prefix"].isin(stage_prefixes)]
raw_matrix = stage_component[stage_component.method.eq("raw")].pivot(index="stage_prefix", columns="component", values="nrmse").loc[stage_prefixes, list(COMPONENTS)]
projected_matrix = stage_component[stage_component.method.eq("projected")].pivot(index="stage_prefix", columns="component", values="nrmse").loc[stage_prefixes, list(COMPONENTS)]
delta_matrix = 100.0 * (projected_matrix - raw_matrix) / raw_matrix
fig, axes = plt.subplots(3, 1, figsize=(16, 9.5), gridspec_kw={"height_ratios": [1, 1, 1.1]})
for ax, matrix, title, cmap, vmin, vmax in (
    (axes[0], raw_matrix, "Raw normalized RMSE", "YlOrRd", 0, float(max(raw_matrix.max().max(), projected_matrix.max().max()))),
    (axes[1], projected_matrix, "Projected normalized RMSE", "YlOrRd", 0, float(max(raw_matrix.max().max(), projected_matrix.max().max()))),
):
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, pad=0.01, label="nRMSE")
    ax.set_title(title); ax.set_yticks(range(len(stage_labels)), stage_labels)
    ax.set_xticks(range(len(COMPONENTS)), COMPONENTS, rotation=55, ha="right")
limit = float(np.nanmax(np.abs(delta_matrix.to_numpy())))
im = axes[2].imshow(delta_matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
fig.colorbar(im, ax=axes[2], pad=0.01, label="Projection change (%)")
axes[2].set_title("Projection change: blue/negative improves; red/positive worsens")
axes[2].set_yticks(range(len(stage_labels)), stage_labels)
axes[2].set_xticks(range(len(COMPONENTS)), COMPONENTS, rotation=55, ha="right")
fig.suptitle("Q4. Component prediction accuracy by reactor and Clarifier outlet", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "q04_holdout_component_accuracy_by_stage")
summary_rows.append({"question": 4, "metric": "stage_component_cells_improved", "value": int((delta_matrix.to_numpy() < 0).sum())})

# 5 and 6. Optimizer-native effluent composites versus exact replay.
q5 = route_parity_chart(
    "surrogate", "projected", "q05_surrogate_effluent_prediction_vs_mechanistic",
    "Q5. Surrogate-optimization effluent prediction vs exact mechanistic replay",
)
q6 = route_parity_chart(
    "direct", "smooth", "q06_smooth_nlp_effluent_prediction_vs_mechanistic",
    "Q6. Smooth-NLP effluent prediction vs exact mechanistic replay",
)
for key, value in q5.items(): summary_rows.append({"question": 5, "metric": key, "value": value})
for key, value in q6.items(): summary_rows.append({"question": 6, "metric": key, "value": value})
if (
    "shared_unit" in routes
    and not quality[
        quality["decision_route"].eq("shared_unit")
        & quality["response_method"].eq("projected")
    ].empty
):
    shared_parity = route_parity_chart(
        "shared_unit", "projected",
        "shared_unit_effluent_prediction_vs_mechanistic",
        "Shared-unit optimization: projected response vs exact mechanistic replay",
    )
    for key, value in shared_parity.items():
        summary_rows.append({"question": "6U", "metric": key, "value": value})

# Exact-reference components for questions 7--10.
with np.load(RUN / "datasets" / "effective_design.npz", allow_pickle=False) as design_npz:
    development_decisions = np.asarray(design_npz["development_decisions"], dtype=float)
with np.load(RUN / "datasets" / "development" / "mechanistic_accepted_v3.npz", allow_pickle=False) as target_npz:
    development_targets = np.asarray(target_npz["targets"], dtype=float)
q_e = 1.0 - development_decisions[:, 6]
development_effluent = development_targets[:, 120:140] / q_e[:, None]
quality_scale = np.std(development_effluent @ COMPOSITE_MATRIX.T, axis=0, ddof=0)
weights = np.asarray([0.50, 0.15, 0.20, 0.05, 0.05, 0.05])
records: list[dict[str, object]] = []
for case in all_cases:
    for route in routes:
        case_directory = RUN / "optimization" / case
        path = case_directory / f"{route}_casewise_reference.npz"
        if not path.is_file():
            path = case_directory / f"{route}_selected.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as stored:
            theta_key = "controls" if "controls" in stored.files else "theta"
            response_key = next((key for key in (
                "reference_response", "exact_reference", "reference",
            ) if key in stored.files), None)
            if theta_key not in stored.files or response_key is None:
                continue
            theta = np.asarray(stored[theta_key], dtype=float)
            response = np.asarray(stored[response_key], dtype=float)
        if theta.shape != (7,) or response.shape[0] < 160 or not (
            np.all(np.isfinite(theta)) and np.all(np.isfinite(response))
        ):
            continue
        effluent = response[120:140] / (1.0 - theta[6])
        comp = COMPOSITE_MATRIX @ effluent
        underflow = response[140:160] / (theta[5] + theta[6])
        underflow_tss = float(TSS_VECTOR @ underflow)
        objective_parts = np.asarray([
            np.mean(comp / quality_scale), (theta[0] - 6.0) / 30.0,
            theta[0] * np.sum(theta[1:4]) / (36.0 * 3.0), theta[4] / 4.0,
            (theta[5] - 0.25), theta[6] * underflow_tss / (0.05 * 15_000.0),
        ])
        records.append({
            "case": case, "route": route,
            **dict(zip(composites, comp, strict=True)),
            **dict(zip(
                ("quality_component", "hrt_component", "aeration_component",
                 "internal_recycle_component", "return_sludge_component", "wasting_component"),
                objective_parts, strict=True,
            )),
            "economic_contribution": float(weights[1:] @ objective_parts[1:]),
            "exact_objective": float(weights @ objective_parts),
        })
exact = pd.DataFrame(records)
labels = [case_label[case] for case in robust_cases]
x = np.arange(len(robust_cases))
route_width = 0.78 / len(routes)
offsets = {
    route: (index - (len(routes) - 1) / 2) * route_width
    for index, route in enumerate(routes)
}
pair_specs = [
    ("surrogate", "shared_unit", "S_U"),
    ("surrogate", "direct", "S_M"),
    ("shared_unit", "direct", "U_M"),
]
pair_specs = [item for item in pair_specs if item[0] in routes and item[1] in routes]
comparison_index = comparison.set_index("case")


def pair_eligible(left: str, right: str, symbol: str) -> pd.Series:
    column = f"comparison_eligible_{symbol}"
    if column not in comparison_index and symbol == "S_M":
        column = "comparison_eligible"
    if column in comparison_index:
        return comparison_index[column].reindex(robust_cases).fillna(False).astype(bool)
    available = exact.pivot_table(index="case", columns="route", values="exact_objective", aggfunc="first")
    return available.get(left, pd.Series(dtype=float)).reindex(robust_cases).notna() & available.get(right, pd.Series(dtype=float)).reindex(robust_cases).notna()


pair_masks = {
    symbol: pair_eligible(left, right, symbol)
    for left, right, symbol in pair_specs
}
eligible_any = np.logical_or.reduce(
    [mask.to_numpy(bool) for mask in pair_masks.values()]
) if pair_masks else np.zeros(len(robust_cases), dtype=bool)


def route_pivot(column: str) -> pd.DataFrame:
    return exact[exact.case.isin(robust_cases)].pivot(
        index="case", columns="route", values=column,
    ).reindex(index=robust_cases, columns=routes)


def grouped_exact_bars(column: str, stem: str, title: str, ylabel: str, question: int) -> None:
    pivot = route_pivot(column)
    fig, ax = plt.subplots(figsize=(12, 5.4))
    shade_ineligible(ax, eligible_any)
    handles = []
    for route in routes:
        bars = ax.bar(
            x + offsets[route], pivot[route], route_width,
            color=ROUTE_COLOR[route], label=ROUTE_LABEL[route],
        )
        handles.append(bars)
    ax.set_xticks(x, labels); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(handles=[*handles, Patch(facecolor=BAD, alpha=0.55, label="No eligible pair")], ncol=len(routes) + 1)
    fig.text(0.5, 0.01, "All available bars use exact nonsmooth mechanistic replay; lower is better.", ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, stem)
    for left, right, symbol in pair_specs:
        valid = pair_masks[symbol] & pivot[left].notna() & pivot[right].notna()
        summary_rows.extend([
            {"question": question, "metric": f"eligible_cases_{symbol}", "value": int(valid.sum())},
            {"question": question, "metric": f"{left}_lower_count_vs_{right}", "value": int((pivot.loc[valid, left] < pivot.loc[valid, right]).sum())},
            {"question": question, "metric": f"{right}_lower_count_vs_{left}", "value": int((pivot.loc[valid, right] < pivot.loc[valid, left]).sum())},
        ])
        if symbol == "S_M":
            summary_rows.extend([
                {"question": question, "metric": "eligible_cases", "value": int(valid.sum())},
                {"question": question, "metric": "surrogate_lower_count", "value": int((pivot.loc[valid, left] < pivot.loc[valid, right]).sum())},
                {"question": question, "metric": "direct_lower_count", "value": int((pivot.loc[valid, right] < pivot.loc[valid, left]).sum())},
            ])


grouped_exact_bars("exact_objective", "q07_exact_optimal_objective", "Q7. Exact objective at selected decisions across robustness cases", "Exact total objective", 7)
grouped_exact_bars("quality_component", "q08_exact_water_quality_component", "Q8. Exact normalized water-quality component across robustness cases", "Normalized water-quality component", 8)

# 9. Exact effluent quality from every selected decision.
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, component in zip(axes.flat, composites, strict=True):
    pivot = route_pivot(component)
    shade_ineligible(ax, eligible_any)
    for route in routes:
        ax.plot(x, pivot[route], marker=ROUTE_MARKER[route], color=ROUTE_COLOR[route], label=ROUTE_LABEL[route])
    ax.set_xticks(x, labels); ax.set_title(component); ax.set_ylabel("Effluent composite")
handles, legend_labels = axes.flat[0].get_legend_handles_labels()
handles.append(Patch(facecolor=BAD, alpha=0.55)); legend_labels.append("No eligible pair")
fig.legend(handles, legend_labels, loc="upper center", ncol=len(routes) + 1, bbox_to_anchor=(0.5, 0.95))
fig.suptitle("Q9. Exact effluent quality yielded by selected decisions", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.91)); save(fig, "q09_exact_effluent_composites")
for left, right, symbol in pair_specs:
    component_pivots = [route_pivot(component) for component in composites]
    mask = pair_masks[symbol].to_numpy(bool)
    relative = np.column_stack([
        np.abs(pivot[left] - pivot[right]) / np.maximum(np.abs(pivot[right]), 1.0e-12)
        for pivot in component_pivots
    ])
    summary_rows.append({
        "question": 9, "metric": f"median_absolute_route_difference_percent_{symbol}",
        "value": float(100 * np.nanmedian(relative[mask])) if np.any(mask) else np.nan,
    })
    if symbol == "S_M":
        summary_rows.append({
            "question": 9, "metric": "median_absolute_route_difference_percent",
            "value": float(100 * np.nanmedian(relative[mask])) if np.any(mask) else np.nan,
        })

# 10. Weighted economic contribution and its composition.
economic_columns = ["hrt_component", "aeration_component", "internal_recycle_component", "return_sludge_component", "wasting_component"]
economic_names = ["HRT", "Aeration", "Internal recycle", "Return sludge", "Wasting"]
economic_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
hatches = {"surrogate": "", "shared_unit": "..", "direct": "///"}
fig, ax = plt.subplots(figsize=(13, 6.2)); shade_ineligible(ax, eligible_any)
for route in routes:
    route_data = exact[exact.route.eq(route) & exact.case.isin(robust_cases)].set_index("case").reindex(robust_cases)
    bottom = np.zeros(len(robust_cases))
    for column, name, color, weight in zip(economic_columns, economic_names, economic_colors, weights[1:], strict=True):
        values = weight * route_data[column].to_numpy(float)
        ax.bar(x + offsets[route], values, route_width, bottom=bottom, color=color,
               edgecolor="white", linewidth=0.3, hatch=hatches[route],
               label=name if route == routes[0] else None)
        bottom += values
ax.set_xticks(x, labels); ax.set_ylabel("Weighted economic/resource contribution")
ax.set_title("Q10. Exact economic/resource objective contribution across robustness cases")
component_handles = [Patch(facecolor=color, label=name) for color, name in zip(economic_colors, economic_names, strict=True)]
route_handles = [Patch(facecolor="white", edgecolor="black", hatch=hatches[route], label=ROUTE_LABEL[route]) for route in routes]
ax.legend(handles=[*component_handles, *route_handles, Patch(facecolor=BAD, alpha=0.55, label="No eligible pair")], ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(rect=(0, 0.12, 1, 1)); save(fig, "q10_exact_economic_component")
economic = route_pivot("economic_contribution")
for left, right, symbol in pair_specs:
    valid = pair_masks[symbol] & economic[left].notna() & economic[right].notna()
    summary_rows.append({"question": 10, "metric": f"{left}_lower_count_vs_{right}", "value": int((economic.loc[valid, left] < economic.loc[valid, right]).sum())})
    if symbol == "S_M":
        summary_rows.extend([
            {"question": 10, "metric": "surrogate_lower_count", "value": int((economic.loc[valid, left] < economic.loc[valid, right]).sum())},
            {"question": 10, "metric": "direct_lower_count", "value": int((economic.loc[valid, right] < economic.loc[valid, left]).sum())},
        ])

# 11. Optimal controls.
control_columns = ["H", "a_3", "a_4", "a_5", "r_I", "r_R", "w"]
control_titles = ["HRT H", "Aeration a3", "Aeration a4", "Aeration a5", "Internal recycle rI", "Return sludge rR", "Waste fraction w"]
fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
robust_controls = controls.set_index(["case", "route"])
for ax, column, title in zip(axes.flat, control_columns, control_titles, strict=False):
    shade_ineligible(ax, eligible_any)
    for route in routes:
        values = np.asarray([
            robust_controls.loc[(case, route), column]
            if (case, route) in robust_controls.index else np.nan
            for case in robust_cases
        ], float)
        ax.plot(x, values, marker=ROUTE_MARKER[route], color=ROUTE_COLOR[route], label=ROUTE_LABEL[route])
    ax.set_title(title); ax.set_xticks(x, labels)
axes.flat[-1].axis("off")
handles, legend_labels = axes.flat[0].get_legend_handles_labels()
handles.append(Patch(facecolor=BAD, alpha=0.55)); legend_labels.append("No eligible pair")
fig.legend(handles, legend_labels, loc="lower right", bbox_to_anchor=(0.93, 0.08))
fig.suptitle("Q11. Selected operating decisions across robustness cases", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.96)); save(fig, "q11_optimal_operating_values")
bounds_low = np.asarray([6, 0, 0, 0, 0, 0.25, 0.005])
bounds_high = np.asarray([36, 1, 1, 1, 4, 1.25, 0.05])
for left, right, symbol in pair_specs:
    left_values = np.vstack([[robust_controls.loc[(case, left), col] if (case, left) in robust_controls.index else np.nan for col in control_columns] for case in robust_cases])
    right_values = np.vstack([[robust_controls.loc[(case, right), col] if (case, right) in robust_controls.index else np.nan for col in control_columns] for case in robust_cases])
    normalized_rms = np.sqrt(np.mean(((left_values - right_values) / (bounds_high - bounds_low)) ** 2, axis=1))
    valid = pair_masks[symbol].to_numpy(bool) & np.isfinite(normalized_rms)
    summary_rows.append({"question": 11, "metric": f"median_normalized_control_rms_difference_{symbol}", "value": float(np.median(normalized_rms[valid])) if np.any(valid) else np.nan})
    if symbol == "S_M":
        summary_rows.append({"question": 11, "metric": "median_normalized_control_rms_difference", "value": float(np.median(normalized_rms[valid])) if np.any(valid) else np.nan})

# 12. Primary optimization time only.
time_pivot = timing.pivot(index="case", columns="route", values="primary_optimization_seconds").reindex(index=robust_cases, columns=routes)
fig, ax = plt.subplots(figsize=(12, 5.7)); bar_groups = []
for route in routes:
    bars = ax.bar(x + offsets[route], time_pivot[route], route_width, color=ROUTE_COLOR[route], label=f"{ROUTE_LABEL[route]} primary")
    bar_groups.append(bars)
ax.set_yscale("log"); ax.set_xticks(x, labels); ax.set_ylabel("Primary optimization time (s, log scale)")
ax.set_title("Q12. Primary optimization time across robustness cases"); ax.legend(ncol=len(routes))
for bars in bar_groups:
    for bar in bars:
        if np.isfinite(bar.get_height()):
            ax.annotate(f"{bar.get_height():.1f}", (bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6)
fig.text(0.5, 0.01, "Certification, recovery, and exact-replay times are excluded.", ha="center")
fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, "q12_primary_optimization_time")
for route in routes:
    summary_rows.append({"question": 12, "metric": f"{route}_mean_seconds", "value": float(time_pivot[route].mean())})
for left, right, symbol in pair_specs:
    valid = time_pivot[left].notna() & time_pivot[right].notna()
    summary_rows.append({"question": 12, "metric": f"{left}_faster_case_count_vs_{right}", "value": int((time_pivot.loc[valid, left] < time_pivot.loc[valid, right]).sum())})
    if symbol == "S_M":
        summary_rows.append({"question": 12, "metric": "surrogate_faster_case_count", "value": int((time_pivot.loc[valid, left] < time_pivot.loc[valid, right]).sum())})

pd.DataFrame(summary_rows).to_csv(OUT / "chart_summary.csv", index=False)
with (OUT / "chart_index.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["question", "png", "svg"])
    for question in range(1, 13):
        matches = sorted(OUT.glob(f"q{question:02d}_*.png"))
        if len(matches) != 1:
            raise RuntimeError(f"question {question} produced {len(matches)} PNG files")
        writer.writerow([question, matches[0].name, matches[0].with_suffix(".svg").name])

print(OUT)
print(pd.DataFrame(summary_rows).to_string(index=False))
