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
DIRECT = "#7A5195"
REFERENCE = "#343A40"
BAD = "#F6D7D7"
GRID = "#D8DEE4"

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
        ax.scatter(x, y, s=42, color=SURROGATE if route == "surrogate" else DIRECT,
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
    for route in ("surrogate", "direct"):
        path = RUN / "optimization" / case / f"{route}_casewise_reference.npz"
        with np.load(path, allow_pickle=False) as stored:
            theta = np.asarray(stored["theta"], dtype=float)
            response = np.asarray(stored["exact_reference"], dtype=float)
        effluent = response[120:140] / (1.0 - theta[6])
        comp = COMPOSITE_MATRIX @ effluent
        underflow = response[140:160] / (theta[5] + theta[6])
        underflow_tss = float(TSS_VECTOR @ underflow)
        objective_parts = np.asarray([
            np.mean(comp / quality_scale),
            (theta[0] - 6.0) / 30.0,
            theta[0] * np.sum(theta[1:4]) / (36.0 * 3.0),
            theta[4] / 4.0,
            (theta[5] - 0.25) / 1.0,
            theta[6] * underflow_tss / (0.05 * 15_000.0),
        ])
        records.append({
            "case": case, "route": route,
            **dict(zip(composites, comp, strict=True)),
            "quality_component": objective_parts[0],
            "hrt_component": objective_parts[1],
            "aeration_component": objective_parts[2],
            "internal_recycle_component": objective_parts[3],
            "return_sludge_component": objective_parts[4],
            "wasting_component": objective_parts[5],
            "economic_contribution": float(weights[1:] @ objective_parts[1:]),
            "exact_objective": float(weights @ objective_parts),
        })
exact = pd.DataFrame(records)
eligible_map = comparison.set_index("case")["comparison_eligible"].astype(bool)
eligible = np.asarray([eligible_map.loc[case] for case in robust_cases], dtype=bool)
labels = [case_label[case] for case in robust_cases]
x = np.arange(10)
width = 0.37

def paired_exact_bars(column: str, stem: str, title: str, ylabel: str, question: int) -> None:
    pivot = exact[exact.case.isin(robust_cases)].pivot(index="case", columns="route", values=column).loc[robust_cases]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    shade_ineligible(ax, eligible)
    b1 = ax.bar(x - width / 2, pivot["surrogate"], width, color=SURROGATE, label="Surrogate decision")
    b2 = ax.bar(x + width / 2, pivot["direct"], width, color=DIRECT, label="Smooth-NLP decision")
    ax.set_xticks(x, labels); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(handles=[b1, b2, Patch(facecolor=BAD, alpha=0.55, label="Paired comparison ineligible")], ncol=3)
    fig.text(0.5, 0.01, "Both bars use exact nonsmooth mechanistic replay; lower is better.", ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, stem)
    valid = pivot.loc[np.asarray(robust_cases)[eligible]]
    summary_rows.extend([
        {"question": question, "metric": "eligible_cases", "value": int(eligible.sum())},
        {"question": question, "metric": "surrogate_lower_count", "value": int((valid.surrogate < valid.direct).sum())},
        {"question": question, "metric": "direct_lower_count", "value": int((valid.direct < valid.surrogate).sum())},
    ])

paired_exact_bars("exact_objective", "q07_exact_optimal_objective", "Q7. Exact objective at selected decisions across robustness cases", "Exact total objective", 7)
paired_exact_bars("quality_component", "q08_exact_water_quality_component", "Q8. Exact normalized water-quality component across robustness cases", "Normalized water-quality component", 8)

# 9. Exact effluent quality from both selected decisions.
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, component in zip(axes.flat, composites, strict=True):
    pivot = exact[exact.case.isin(robust_cases)].pivot(index="case", columns="route", values=component).loc[robust_cases]
    shade_ineligible(ax, eligible)
    ax.plot(x, pivot["surrogate"], marker="o", color=SURROGATE, label="Surrogate decision")
    ax.plot(x, pivot["direct"], marker="s", color=DIRECT, label="Smooth-NLP decision")
    ax.set_xticks(x, labels); ax.set_title(component); ax.set_ylabel("Effluent composite")
handles, legend_labels = axes.flat[0].get_legend_handles_labels()
handles.append(Patch(facecolor=BAD, alpha=0.55)); legend_labels.append("Paired comparison ineligible")
fig.legend(handles, legend_labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.95))
fig.suptitle("Q9. Exact effluent quality yielded by the two selected decisions", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.91))
save(fig, "q09_exact_effluent_composites")
eligible_exact = exact[exact.case.isin(np.asarray(robust_cases)[eligible])]
pair = eligible_exact.pivot(index="case", columns="route", values=composites)
relative = np.abs(pair.xs("surrogate", axis=1, level="route") - pair.xs("direct", axis=1, level="route")) / np.maximum(np.abs(pair.xs("direct", axis=1, level="route")), 1.0e-12)
summary_rows.append({"question": 9, "metric": "median_absolute_route_difference_percent", "value": float(100 * np.median(relative.to_numpy()))})

# 10. Weighted economic contribution and its composition.
economic_columns = ["hrt_component", "aeration_component", "internal_recycle_component", "return_sludge_component", "wasting_component"]
economic_names = ["HRT", "Aeration", "Internal recycle", "Return sludge", "Wasting"]
economic_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
fig, ax = plt.subplots(figsize=(13, 6.2))
shade_ineligible(ax, eligible)
for route, offset, hatch in (("surrogate", -width / 2, ""), ("direct", width / 2, "///")):
    route_data = exact[(exact.route == route) & exact.case.isin(robust_cases)].set_index("case").loc[robust_cases]
    bottom = np.zeros(10)
    for column, name, color, weight in zip(economic_columns, economic_names, economic_colors, weights[1:], strict=True):
        values = weight * route_data[column].to_numpy(float)
        ax.bar(x + offset, values, width, bottom=bottom, color=color, edgecolor="white", linewidth=0.3,
               hatch=hatch, label=name if route == "surrogate" else None)
        bottom += values
ax.set_xticks(x, labels); ax.set_ylabel("Weighted economic/resource contribution")
ax.set_title("Q10. Exact economic/resource objective contribution across robustness cases")
component_handles = [Patch(facecolor=color, label=name) for color, name in zip(economic_colors, economic_names, strict=True)]
route_handles = [Patch(facecolor="white", edgecolor="black", label="Surrogate: left/solid"), Patch(facecolor="white", edgecolor="black", hatch="///", label="Smooth NLP: right/hatched"), Patch(facecolor=BAD, alpha=0.55, label="Ineligible")]
ax.legend(handles=[*component_handles, *route_handles], ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(rect=(0, 0.12, 1, 1))
save(fig, "q10_exact_economic_component")
valid_economic = exact[exact.case.isin(np.asarray(robust_cases)[eligible])].pivot(index="case", columns="route", values="economic_contribution")
summary_rows.extend([
    {"question": 10, "metric": "surrogate_lower_count", "value": int((valid_economic.surrogate < valid_economic.direct).sum())},
    {"question": 10, "metric": "direct_lower_count", "value": int((valid_economic.direct < valid_economic.surrogate).sum())},
])

# 11. Optimal controls.
control_columns = ["H", "a_3", "a_4", "a_5", "r_I", "r_R", "w"]
control_titles = ["HRT H", "Aeration a3", "Aeration a4", "Aeration a5", "Internal recycle rI", "Return sludge rR", "Waste fraction w"]
fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
robust_controls = controls.set_index(["case", "route"])
for ax, column, title in zip(axes.flat, control_columns, control_titles, strict=False):
    shade_ineligible(ax, eligible)
    s = np.asarray([robust_controls.loc[(case, "surrogate"), column] for case in robust_cases], float)
    d = np.asarray([robust_controls.loc[(case, "direct"), column] for case in robust_cases], float)
    ax.plot(x, s, marker="o", color=SURROGATE, label="Surrogate")
    ax.plot(x, d, marker="s", color=DIRECT, label="Smooth NLP")
    ax.set_title(title); ax.set_xticks(x, labels)
axes.flat[-1].axis("off")
handles, legend_labels = axes.flat[0].get_legend_handles_labels()
handles.append(Patch(facecolor=BAD, alpha=0.55)); legend_labels.append("Paired comparison ineligible")
fig.legend(handles, legend_labels, loc="lower right", bbox_to_anchor=(0.93, 0.08))
fig.suptitle("Q11. Selected operating decisions across robustness cases", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "q11_optimal_operating_values")
bounds_low = np.asarray([6, 0, 0, 0, 0, 0.25, 0.005])
bounds_high = np.asarray([36, 1, 1, 1, 4, 1.25, 0.05])
control_pivot_s = np.vstack([[robust_controls.loc[(case, "surrogate"), col] for col in control_columns] for case in robust_cases])
control_pivot_d = np.vstack([[robust_controls.loc[(case, "direct"), col] for col in control_columns] for case in robust_cases])
normalized_rms = np.sqrt(np.mean(((control_pivot_s - control_pivot_d) / (bounds_high - bounds_low)) ** 2, axis=1))
summary_rows.append({"question": 11, "metric": "median_normalized_control_rms_difference", "value": float(np.median(normalized_rms[eligible]))})

# 12. Primary optimization time only.
time_pivot = timing.pivot(index="case", columns="route", values="primary_optimization_seconds").loc[robust_cases]
fig, ax = plt.subplots(figsize=(12, 5.7))
b1 = ax.bar(x - width / 2, time_pivot["surrogate"], width, color=SURROGATE, label="Surrogate primary")
b2 = ax.bar(x + width / 2, time_pivot["direct"], width, color=DIRECT, label="Smooth-NLP primary")
ax.set_yscale("log")
ax.set_xticks(x, labels); ax.set_ylabel("Primary optimization time (s, log scale)")
ax.set_title("Q12. Primary optimization time across robustness cases")
ax.legend(ncol=2)
for bars in (b1, b2):
    for bar in bars:
        ax.annotate(f"{bar.get_height():.1f}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=6)
fig.text(0.5, 0.01, "Certification, recovery, and exact-replay times are excluded.", ha="center")
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "q12_primary_optimization_time")
summary_rows.extend([
    {"question": 12, "metric": "surrogate_mean_seconds", "value": float(time_pivot.surrogate.mean())},
    {"question": 12, "metric": "direct_mean_seconds", "value": float(time_pivot.direct.mean())},
    {"question": 12, "metric": "surrogate_faster_case_count", "value": int((time_pivot.surrogate < time_pivot.direct).sum())},
])

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
