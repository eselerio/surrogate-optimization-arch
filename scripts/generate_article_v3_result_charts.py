"""Generate the article-v3 result-comparison charts, including objective values."""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from closed_loop.model import COMPONENTS, COMPOSITE_MATRIX, NOMINAL_INFLUENT, TSS_VECTOR


parser = argparse.ArgumentParser(
    description="Generate the twelve article-v3 comparison charts for one completed run."
)
parser.add_argument(
    "--run-id",
    default="article_full_50000_three_route_001",
    help="Run directory name below results/article_v3 (default: %(default)s).",
)
parser.add_argument(
    "--routes",
    default="surrogate,shared_unit,direct",
    help="Comma-separated routes to include, in display order (default: %(default)s).",
)
parser.add_argument(
    "--figure-subdirectory",
    default="requested_comparisons",
    help="Output directory below report/figures (default: %(default)s).",
)
arguments = parser.parse_args()

requested_routes = tuple(route.strip() for route in arguments.routes.split(",") if route.strip())
if not requested_routes or len(set(requested_routes)) != len(requested_routes) or any(
    route not in ("surrogate", "shared_unit", "direct") for route in requested_routes
):
    raise ValueError("--routes must be a nonempty, nonrepeating subset of (surrogate, shared_unit, direct)")
output_subdirectory = Path(arguments.figure_subdirectory)
if output_subdirectory.is_absolute() or ".." in output_subdirectory.parts:
    raise ValueError("--figure-subdirectory must be a relative path below report/figures")

RUN = ROOT / "results" / "article_v3" / arguments.run_id
TABLES = RUN / "report" / "tables"
OUT = RUN / "report" / "figures" / output_subdirectory
if not TABLES.is_dir():
    raise FileNotFoundError(f"Completed report tables were not found: {TABLES}")
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


def route_removal_parity_chart(route: str, prediction_method: str, stem: str, title: str) -> dict[str, float]:
    """Compare predicted and exact composite removals on the selected cases."""
    subset = quality[
        (quality["decision_route"] == route)
        & (quality["response_method"].isin([prediction_method, "reference"]))
        & quality["available"].astype(bool)
    ]
    pivot = subset.pivot(index="case", columns="response_method", values=composites)
    influent = influent_composites.reindex(pivot.index)
    if influent.isna().any().any() or np.any(influent.to_numpy(float) <= 0.0):
        raise RuntimeError("positive composite influent concentrations are required for removal plots")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    errors: list[float] = []
    r2_values: list[float] = []
    for ax, component in zip(axes.flat, composites, strict=True):
        feed = influent[component].to_numpy(float)
        exact_removal = 100.0 * (1.0 - pivot[(component, "reference")].to_numpy(float) / feed)
        predicted_removal = 100.0 * (1.0 - pivot[(component, prediction_method)].to_numpy(float) / feed)
        limits = [min(exact_removal.min(), predicted_removal.min()), max(exact_removal.max(), predicted_removal.max())]
        pad = max(1.0, 0.06 * (limits[1] - limits[0]))
        limits = [limits[0] - pad, limits[1] + pad]
        ax.plot(limits, limits, color=REFERENCE, lw=1.2, ls="--", label="perfect match")
        ax.scatter(exact_removal, predicted_removal, s=42, color=ROUTE_COLOR[route],
                   edgecolor="white", linewidth=0.6, zorder=3)
        for label, xx, yy in zip([case_label[c] for c in pivot.index], exact_removal, predicted_removal, strict=True):
            ax.annotate(label, (xx, yy), xytext=(3, 2), textcoords="offset points", fontsize=6)
        absolute_error = np.abs(predicted_removal - exact_removal)
        denominator = np.sum((exact_removal - np.mean(exact_removal)) ** 2)
        r2 = np.nan if denominator <= 1.0e-12 else 1.0 - np.sum((predicted_removal - exact_removal) ** 2) / denominator
        errors.extend(absolute_error.tolist())
        r2_values.append(float(r2))
        ax.set_xlim(limits); ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{component}: median |error| = {np.median(absolute_error):.1f} pp")
        ax.set_xlabel("Exact mechanistic removal (%)")
        ax.set_ylabel(f"{prediction_method.capitalize()} removal (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94))
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.text(0.5, 0.01, "Removal = 100 × (influent − effluent) / influent; each point is nominal or one robustness case.", ha="center")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    save(fig, stem)
    return {
        "median_absolute_removal_error_percentage_points": float(np.median(errors)),
        "mean_absolute_removal_error_percentage_points": float(np.mean(errors)),
        "mean_component_r2": float(np.nanmean(r2_values)),
    }


prediction = pd.read_csv(TABLES / "prediction_metrics.csv")
quality = pd.read_csv(TABLES / "selected_quality.csv")
controls = pd.read_csv(TABLES / "scenario_controls.csv")
comparison = pd.read_csv(TABLES / "scenario_comparison.csv")
nominal_comparison = pd.read_csv(TABLES / "nominal_comparison.csv")
timing = pd.read_csv(RUN / "metrics" / "robustness_case_timing.csv")
profiles = pd.read_csv(TABLES / "process_profiles.csv")

present_routes = set(quality.get("decision_route", pd.Series(dtype=str)).astype(str))
present_routes.update(controls.get("route", pd.Series(dtype=str)).astype(str))
routes = tuple(
    route for route in requested_routes
    if route in present_routes or any((RUN / "optimization").glob(f"*/{route}_selected.npz"))
)
missing_routes = tuple(route for route in requested_routes if route not in routes)
if missing_routes:
    raise RuntimeError(f"requested routes are unavailable: {missing_routes}")

robust_cases = [f"robustness_{index:02d}" for index in range(1, 11)]
all_cases = ["nominal", *robust_cases]
case_label = {"nominal": "N", **{case: f"R{i}" for i, case in enumerate(robust_cases, 1)}}
composites = ["COD", "TN", "TP", "TSS"]
with np.load(RUN / "datasets" / "effective_design.npz", allow_pickle=False) as design_for_removal:
    robustness_influents = np.asarray(design_for_removal["robustness_influents"], dtype=float)
if robustness_influents.shape != (len(robust_cases), len(COMPONENTS)):
    raise RuntimeError("robustness influents have an unexpected shape")
influent_composites = pd.DataFrame(
    np.vstack((np.asarray(NOMINAL_INFLUENT, dtype=float), robustness_influents)) @ COMPOSITE_MATRIX.T,
    index=all_cases,
    columns=composites,
)
summary_rows: list[dict[str, object]] = []

# 1--4. Holdout emulation diagnostics.  The shared-unit and whole-system
# surrogates have distinct audited holdouts, so they are plotted side by side
# rather than pooled into a misleading single raw/projected metric.
prediction_routes = tuple(
    route for route in ("surrogate", "shared_unit")
    if route in routes and route in set(prediction["route"].astype(str))
)
prediction_columns = tuple(
    (route, method) for route in prediction_routes for method in ("raw", "projected")
)
prediction_labels = {
    ("surrogate", "raw"): "System\nraw",
    ("surrogate", "projected"): "System\nprojected",
    ("shared_unit", "raw"): "Shared-unit\nraw",
    ("shared_unit", "projected"): "Shared-unit\nprojected",
}
prediction_colors = {
    ("surrogate", "raw"): RAW,
    ("surrogate", "projected"): SURROGATE,
    ("shared_unit", "raw"): "#E6A44A",
    ("shared_unit", "projected"): SHARED_UNIT,
}
prediction_hatches = {
    ("surrogate", "raw"): "",
    ("surrogate", "projected"): "",
    ("shared_unit", "raw"): "..",
    ("shared_unit", "projected"): "..",
}

# 1. Complete-response composite prediction.
overall = prediction[
    prediction["method"].isin(["raw", "projected"])
    & prediction["block"].eq("complete_response")
    & prediction["coordinate"].eq("ALL")
].set_index(["route", "method"])
metrics = [("nrmse", "Normalized RMSE", True), ("nmae", "Normalized MAE", True), ("r2_mean", "Mean $R^2$", False)]
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
for ax, (column, label, lower_better) in zip(axes, metrics, strict=True):
    values = overall.loc[list(prediction_columns), column].to_numpy(float)
    bars = ax.bar(
        [prediction_labels[key] for key in prediction_columns], values,
        color=[prediction_colors[key] for key in prediction_columns], width=0.68,
    )
    for bar, key in zip(bars, prediction_columns, strict=True):
        bar.set_hatch(prediction_hatches[key])
    if column != "r2_mean":
        annotate_bars(ax, bars)
    ax.set_title(label + (" (lower is better)" if lower_better else " (higher is better)"))
    ax.set_ylabel(label)
    if column in {"nrmse", "nmae"}:
        # The shared-unit gate failure produces errors orders of magnitude
        # larger than the whole-system route; log axes preserve both signals.
        ax.set_yscale("log")
        ax.set_ylim(float(np.nanmin(values[values > 0])) * 0.55, float(np.nanmax(values)) * 1.8)
    else:
        ax.set_yscale("symlog", linthresh=1.0)
        lower, upper = min(0.0, float(np.nanmin(values))), max(0.0, float(np.nanmax(values)))
        padding = max(1.0, 0.12 * (upper - lower))
        ax.set_ylim(lower - padding, upper + padding)
if "shared_unit" in prediction_routes:
    q1_title = "Q1. Complete-response prediction on the route-specific post-selection holdouts"
    q1_note = "The whole-system and shared-unit routes use separate audited holdouts; compare raw-to-projected change within each route."
else:
    q1_title = "Q1. Whole-system surrogate prediction on the post-selection holdout"
    q1_note = "Raw and projected predictions use the same 3,343-row post-selection holdout."
fig.suptitle(q1_title, fontsize=14)
fig.text(0.5, 0.01, q1_note, ha="center")
fig.tight_layout(rect=(0, 0.05, 1, 0.92))
save(fig, "q01_holdout_composite_accuracy")
for route in prediction_routes:
    q1_improvement = 100.0 * (
        overall.loc[(route, "raw"), "nrmse"] - overall.loc[(route, "projected"), "nrmse"]
    ) / overall.loc[(route, "raw"), "nrmse"]
    summary_rows.append({"question": 1, "metric": f"{route}_projection_nrmse_improvement_percent", "value": q1_improvement})
    if route == "surrogate":
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
].pivot(index="block", columns=["route", "method"], values="nrmse").reindex(
    index=stage_blocks, columns=list(prediction_columns),
)
x = np.arange(len(stage))
fig, (ax, delta_ax) = plt.subplots(2, 1, figsize=(12.5, 7.5), gridspec_kw={"height_ratios": [2, 1]})
width = 0.78 / len(prediction_columns)
for index, key in enumerate(prediction_columns):
    offset = (index - (len(prediction_columns) - 1) / 2) * width
    bars = ax.bar(x + offset, stage[key], width, color=prediction_colors[key], label=prediction_labels[key])
    for bar in bars:
        bar.set_hatch(prediction_hatches[key])
ax.set_xticks(x, stage_names); ax.set_ylabel("Normalized RMSE")
ax.set_title("Response-block accuracy (lower is better)"); ax.legend(ncol=len(prediction_columns))
stage_values = stage.to_numpy(float)
ax.set_yscale("log")
ax.set_ylim(float(np.nanmin(stage_values[stage_values > 0])) * 0.55, float(np.nanmax(stage_values)) * 1.8)
changes: dict[str, pd.Series] = {}
for index, route in enumerate(prediction_routes):
    change = 100.0 * (stage[(route, "projected")] - stage[(route, "raw")]) / stage[(route, "raw")]
    changes[route] = change
    offset = (index - (len(prediction_routes) - 1) / 2) * 0.36
    bars = delta_ax.bar(x + offset, change, 0.34, color=prediction_colors[(route, "projected")], label=ROUTE_LABEL[route])
    for bar in bars:
        bar.set_hatch(prediction_hatches[(route, "projected")])
delta_ax.axhline(0, color=REFERENCE, lw=0.8)
delta_ax.set_xticks(x, stage_names); delta_ax.set_ylabel("Projection change (%)")
delta_ax.set_title("Negative values mean projection improved accuracy"); delta_ax.legend(ncol=len(prediction_routes))
fig.suptitle("Q2. Prediction accuracy by reduced-response block", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "q02_holdout_accuracy_by_response_block")
for route, change in changes.items():
    count = int((change < 0).sum())
    summary_rows.append({"question": 2, "metric": f"{route}_stages_improved_by_projection", "value": count})
    if route == "surrogate":
        summary_rows.append({"question": 2, "metric": "stages_improved_by_projection", "value": count})

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
    coordinate_rows.groupby(["route", "method", "component"])["nrmse"]
    .apply(lambda values: float(np.sqrt(np.mean(np.square(values)))))
    .reindex(pd.MultiIndex.from_product([prediction_routes, ("raw", "projected"), list(COMPONENTS)]))
    .unstack([0, 1]).reindex(list(COMPONENTS))
)
y = np.arange(len(component_metric))
fig, ax = plt.subplots(figsize=(11, 9.2))
height = 0.78 / len(prediction_columns)
for index, key in enumerate(prediction_columns):
    offset = (index - (len(prediction_columns) - 1) / 2) * height
    bars = ax.barh(y + offset, component_metric[key], height, color=prediction_colors[key], label=prediction_labels[key])
    for bar in bars:
        bar.set_hatch(prediction_hatches[key])
ax.set_yticks(y, component_metric.index); ax.invert_yaxis()
ax.set_xlabel("Stage-aggregated normalized RMSE (lower is better)")
ax.set_title("Q3. Component prediction across mixer, reactors, and Clarifier outlets")
ax.legend(ncol=2)
component_values = component_metric.to_numpy(float)
ax.set_xscale("log")
ax.set_xlim(float(np.nanmin(component_values[component_values > 0])) * 0.55, float(np.nanmax(component_values)) * 1.8)
fig.tight_layout()
save(fig, "q03_holdout_component_accuracy")
for route in prediction_routes:
    count = int((component_metric[(route, "projected")] < component_metric[(route, "raw")]).sum())
    summary_rows.append({"question": 3, "metric": f"{route}_components_improved_by_projection", "value": count})
    if route == "surrogate":
        summary_rows.append({"question": 3, "metric": "components_improved_by_projection", "value": count})

# 4. Component-level prediction by stage.
stage_prefixes = [*[f"reactor_{i}" for i in range(1, 6)], "overflow_flow", "underflow_flow"]
stage_labels = [*[f"R{i}" for i in range(1, 6)], "Overflow", "Underflow"]
stage_component = coordinate_rows[coordinate_rows["stage_prefix"].isin(stage_prefixes)]
fig, axes = plt.subplots(len(prediction_routes), 3, figsize=(16, 5.2 * len(prediction_routes)), squeeze=False)
for row, route in enumerate(prediction_routes):
    raw_matrix = stage_component[
        stage_component.route.eq(route) & stage_component.method.eq("raw")
    ].pivot(index="stage_prefix", columns="component", values="nrmse").reindex(index=stage_prefixes, columns=list(COMPONENTS))
    projected_matrix = stage_component[
        stage_component.route.eq(route) & stage_component.method.eq("projected")
    ].pivot(index="stage_prefix", columns="component", values="nrmse").reindex(index=stage_prefixes, columns=list(COMPONENTS))
    delta_matrix = 100.0 * (projected_matrix - raw_matrix) / raw_matrix
    maximum = float(np.nanmax([raw_matrix.to_numpy(), projected_matrix.to_numpy()]))
    limit = float(np.nanmax(np.abs(delta_matrix.to_numpy())))
    for ax, matrix, title, cmap, vmin, vmax, colorbar_label in (
        (axes[row, 0], raw_matrix, "Raw nRMSE", "YlOrRd", 0, maximum, "nRMSE"),
        (axes[row, 1], projected_matrix, "Projected nRMSE", "YlOrRd", 0, maximum, "nRMSE"),
        (axes[row, 2], delta_matrix, "Projection change", "RdBu_r", -limit, limit, "%"),
    ):
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, pad=0.01, label=colorbar_label)
        ax.set_title(f"{ROUTE_LABEL[route]}: {title}")
        ax.set_yticks(range(len(stage_labels)), stage_labels)
        ax.set_xticks(range(len(COMPONENTS)), COMPONENTS, rotation=55, ha="right")
    count = int((delta_matrix.to_numpy() < 0).sum())
    summary_rows.append({"question": 4, "metric": f"{route}_stage_component_cells_improved", "value": count})
    if route == "surrogate":
        summary_rows.append({"question": 4, "metric": "stage_component_cells_improved", "value": count})
fig.suptitle("Q4. Component prediction accuracy by reactor and Clarifier outlet", fontsize=14)
fig.text(0.5, 0.01, "Blue projection-change cells improve nRMSE; each route uses its own nRMSE color scale.", ha="center")
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
save(fig, "q04_holdout_component_accuracy_by_stage")

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
q5_removal = route_removal_parity_chart(
    "surrogate", "projected", "q05_surrogate_percent_removal_vs_mechanistic",
    "System-surrogate removal prediction vs exact mechanistic replay",
)
q6_removal = route_removal_parity_chart(
    "direct", "smooth", "q06_smooth_nlp_percent_removal_vs_mechanistic",
    "Smooth-NLP removal prediction vs exact mechanistic replay",
)
for key, value in q5_removal.items(): summary_rows.append({"question": "5R", "metric": key, "value": value})
for key, value in q6_removal.items(): summary_rows.append({"question": "6R", "metric": key, "value": value})
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
    shared_removal_parity = route_removal_parity_chart(
        "shared_unit", "projected", "shared_unit_percent_removal_vs_mechanistic",
        "Shared-unit surrogate removal prediction vs exact mechanistic replay",
    )
    for key, value in shared_removal_parity.items():
        summary_rows.append({"question": "6UR", "metric": key, "value": value})

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
all_three_eligible = (
    comparison_index["all_three_comparison_eligible"].reindex(robust_cases).fillna(False).astype(bool)
    if "all_three_comparison_eligible" in comparison_index
    else pd.Series(eligible_any, index=robust_cases)
)
if len(routes) == 3:
    comparison_eligible = all_three_eligible
    comparison_scope_label = "all-three comparison"
    nominal_eligibility_column = "all_three_comparison_eligible"
elif len(pair_specs) == 1:
    left_route, right_route, pair_symbol = pair_specs[0]
    comparison_eligible = pair_masks[pair_symbol]
    comparison_scope_label = f"{ROUTE_LABEL[left_route]}–{ROUTE_LABEL[right_route]} comparison"
    nominal_eligibility_column = f"comparison_eligible_{pair_symbol}"
else:
    raise RuntimeError("the selected route set does not define one comparison scope")


def route_pivot(column: str) -> pd.DataFrame:
    return exact[exact.case.isin(robust_cases)].pivot(
        index="case", columns="route", values=column,
    ).reindex(index=robust_cases, columns=routes)


def grouped_exact_bars(column: str, stem: str, title: str, ylabel: str, question: int) -> None:
    pivot = route_pivot(column)
    fig, ax = plt.subplots(figsize=(12, 5.4))
    shade_ineligible(ax, comparison_eligible)
    handles = []
    for route in routes:
        bars = ax.bar(
            x + offsets[route], pivot[route], route_width,
            color=ROUTE_COLOR[route], label=ROUTE_LABEL[route],
        )
        handles.append(bars)
    ax.set_xticks(x, labels); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(handles=[*handles, Patch(facecolor=BAD, alpha=0.55, label=f"Not eligible for {comparison_scope_label}")], ncol=len(routes) + 1)
    fig.text(0.5, 0.01, f"All available bars use exact nonsmooth mechanistic replay; pink cases are not eligible for the {comparison_scope_label}.", ha="center")
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

# 13. Dedicated objective-value comparison, retaining the nominal case as well
# as the robustness cases used in Q7.
all_case_x = np.arange(len(all_cases))
all_case_labels = [case_label[case] for case in all_cases]
all_case_objective = exact.pivot(index="case", columns="route", values="exact_objective").reindex(index=all_cases, columns=routes)
nominal_comparison_eligible = nominal_comparison.set_index("case")[nominal_eligibility_column]
all_case_eligible = pd.concat([
    nominal_comparison_eligible.reindex(["nominal"]),
    comparison_eligible,
]).reindex(all_cases).fillna(False).astype(bool)
fig, ax = plt.subplots(figsize=(13, 5.8))
shade_ineligible(ax, all_case_eligible)
for route in routes:
    ax.bar(
        all_case_x + offsets[route], all_case_objective[route], route_width,
        color=ROUTE_COLOR[route], label=ROUTE_LABEL[route],
    )
ax.set_xticks(all_case_x, all_case_labels)
ax.set_ylabel("Exact total objective")
ax.set_title("Q13. Objective-value comparison at the three routes' selected decisions")
ax.legend(handles=[
    *[Patch(facecolor=ROUTE_COLOR[route], label=ROUTE_LABEL[route]) for route in routes],
    Patch(facecolor=BAD, alpha=0.55, label=f"Not eligible for {comparison_scope_label}"),
], ncol=len(routes) + 1)
fig.text(0.5, 0.01, f"Values use exact nonsmooth mechanistic replay; lower is better. Pink cases are not eligible for the {comparison_scope_label}.", ha="center")
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "q13_exact_objective_value_comparison")
for route in routes:
    summary_rows.append({
        "question": 13,
        "metric": f"{route}_nominal_exact_objective",
        "value": float(all_case_objective.loc["nominal", route]),
    })

grouped_exact_bars("quality_component", "q08_exact_water_quality_component", "Q8. Exact normalized water-quality component across robustness cases", "Normalized water-quality component", 8)

# 9. Exact effluent quality from every selected decision.
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, component in zip(axes.flat, composites, strict=True):
    pivot = route_pivot(component)
    shade_ineligible(ax, comparison_eligible)
    for route in routes:
        ax.plot(x, pivot[route], marker=ROUTE_MARKER[route], color=ROUTE_COLOR[route], label=ROUTE_LABEL[route])
    ax.set_xticks(x, labels); ax.set_title(component); ax.set_ylabel("Effluent composite")
handles, legend_labels = axes.flat[0].get_legend_handles_labels()
handles.append(Patch(facecolor=BAD, alpha=0.55)); legend_labels.append(f"Not eligible for {comparison_scope_label}")
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
fig, ax = plt.subplots(figsize=(13, 6.2)); shade_ineligible(ax, comparison_eligible)
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
ax.legend(handles=[*component_handles, *route_handles, Patch(facecolor=BAD, alpha=0.55, label=f"Not eligible for {comparison_scope_label}")], ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
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
    shade_ineligible(ax, comparison_eligible)
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
handles.append(Patch(facecolor=BAD, alpha=0.55)); legend_labels.append(f"Not eligible for {comparison_scope_label}")
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

# 14--17. Exact-replay evolution through the main liquid-treatment train.
# The clarifier underflow is a side stream, so the serial path terminates at
# the clarifier overflow (the treated effluent).
main_profile_locations = [
    "mixer", *[f"reactor_{index}" for index in range(1, 6)], "clarifier_overflow",
]
main_profile_labels = ["Influent", "Mixer", *[f"R{index}" for index in range(1, 6)], "Effluent"]
case_colors = {
    "nominal": "#111827",
    **{
        case: plt.get_cmap("tab20")(index)
        for index, case in enumerate(robust_cases)
    },
}


def main_train_profile_chart(component: str, question: int, stem: str) -> None:
    figure, axes = plt.subplots(len(routes), 1, figsize=(12, 3.5 * len(routes)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, route in zip(axes, routes, strict=True):
        for case in all_cases:
            reference_path = RUN / "optimization" / case / f"{route}_casewise_reference.npz"
            with np.load(reference_path, allow_pickle=False) as stored:
                full_response = np.asarray(stored["exact_reference_full"], dtype=float)
            if full_response.shape != (170,):
                raise RuntimeError(f"unexpected exact-response shape for {route}/{case}")
            liquid_path = np.vstack((
                full_response[0:20],
                *[full_response[20 * index:20 * (index + 1)] for index in range(1, 6)],
                full_response[120:140],
            ))
            values = np.concatenate((
                [float(influent_composites.loc[case, component])],
                liquid_path @ COMPOSITE_MATRIX[composites.index(component)],
            ))
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise RuntimeError(f"nonpositive or unavailable {component} profile for {route}/{case}")
            qualified = bool(all_case_eligible.loc[case])
            axis.plot(
                range(len(main_profile_labels)), values,
                color=case_colors[case], lw=2.2 if case == "nominal" else 1.25,
                ls="-" if qualified else "--", marker="o", ms=4.2,
                alpha=1.0 if qualified else 0.72,
            )
        axis.set_yscale("log")
        axis.set_ylabel(f"{component} (mg/L, log scale)")
        axis.set_title(ROUTE_LABEL[route])
        axis.set_xticks(range(len(main_profile_labels)), main_profile_labels)
    handles = [
        Line2D([0], [0], color=case_colors[case], lw=2.2 if case == "nominal" else 1.5,
               ls="-" if bool(all_case_eligible.loc[case]) else "--", marker="o",
               label=case_label[case])
        for case in all_cases
    ]
    figure.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.005), title="Case")
    figure.suptitle(f"Q{question}. {component} evolution through the main treatment train", fontsize=14)
    figure.text(
        0.5, 0.13,
        f"Lines are exact mechanistic replays at each route's selected decision. Dashed cases are not eligible for the {comparison_scope_label}.",
        ha="center",
    )
    figure.tight_layout(rect=(0, 0.18, 1, 0.96))
    save(figure, stem)


for question, component, stem in (
    (14, "COD", "q14_cod_main_treatment_train_profiles"),
    (15, "TN", "q15_tn_main_treatment_train_profiles"),
    (16, "TP", "q16_tp_main_treatment_train_profiles"),
    (17, "TSS", "q17_tss_main_treatment_train_profiles"),
):
    main_train_profile_chart(component, question, stem)

pd.DataFrame(summary_rows).to_csv(OUT / "chart_summary.csv", index=False)
with (OUT / "chart_index.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["question", "png", "svg"])
    primary_stems = {
        1: "q01_holdout_composite_accuracy",
        2: "q02_holdout_accuracy_by_response_block",
        3: "q03_holdout_component_accuracy",
        4: "q04_holdout_component_accuracy_by_stage",
        5: "q05_surrogate_effluent_prediction_vs_mechanistic",
        6: "q06_smooth_nlp_effluent_prediction_vs_mechanistic",
        7: "q07_exact_optimal_objective",
        8: "q08_exact_water_quality_component",
        9: "q09_exact_effluent_composites",
        10: "q10_exact_economic_component",
        11: "q11_optimal_operating_values",
        12: "q12_primary_optimization_time",
        13: "q13_exact_objective_value_comparison",
        14: "q14_cod_main_treatment_train_profiles",
        15: "q15_tn_main_treatment_train_profiles",
        16: "q16_tp_main_treatment_train_profiles",
        17: "q17_tss_main_treatment_train_profiles",
    }
    for question, stem in primary_stems.items():
        primary_png = OUT / f"{stem}.png"
        if not primary_png.is_file():
            raise RuntimeError(f"question {question} did not produce {primary_png.name}")
        writer.writerow([question, primary_png.name, primary_png.with_suffix(".svg").name])
    shared_parity_png = OUT / "shared_unit_effluent_prediction_vs_mechanistic.png"
    if shared_parity_png.is_file():
        writer.writerow(["6U", shared_parity_png.name, shared_parity_png.with_suffix(".svg").name])
    for label, stem in (
        ("5R", "q05_surrogate_percent_removal_vs_mechanistic"),
        ("6R", "q06_smooth_nlp_percent_removal_vs_mechanistic"),
        ("6UR", "shared_unit_percent_removal_vs_mechanistic"),
    ):
        removal_png = OUT / f"{stem}.png"
        if removal_png.is_file():
            writer.writerow([label, removal_png.name, removal_png.with_suffix(".svg").name])

print(OUT)
print(pd.DataFrame(summary_rows).to_string(index=False))
