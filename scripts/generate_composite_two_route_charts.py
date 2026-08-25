"""Generate the two-route article chart package using composite prediction metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from closed_loop.model import COMPOSITE_MATRIX, NOMINAL_INFLUENT, TSS_VECTOR


EXTENDED = "#147D92"
DIRECT = "#7A5195"
RAW = "#D97904"
REFERENCE = "#343A40"
BAD = "#F6D7D7"
GRID = "#D8DEE4"
ROUTES = ("surrogate", "direct")
ROUTE_LABEL = {"surrogate": "Extended ICSOR", "direct": "Smooth NLP"}
ROUTE_COLOR = {"surrogate": EXTENDED, "direct": DIRECT}
ROUTE_MARKER = {"surrogate": "o", "direct": "s"}
COMPOSITES = ("COD", "TN", "TP", "TSS")
LOCATIONS = ("mixer", "reactor_1", "reactor_2", "reactor_3", "reactor_4", "reactor_5", "overflow", "underflow")
LOCATION_LABELS = ("Mixer", "R1", "R2", "R3", "R4", "R5", "Overflow", "Underflow")


def style() -> None:
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


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def finite_score(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    scale = float(np.ptp(truth))
    if truth.shape != prediction.shape or not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("prediction score received incompatible or non-finite arrays")
    if scale <= 0.0:
        raise ValueError("prediction score requires a nonzero truth range")
    error = prediction - truth
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = 1.0 - float(np.sum(error**2)) / denominator if denominator > 0.0 else np.nan
    return (
        float(np.sqrt(np.mean(error**2)) / scale),
        float(np.mean(np.abs(error)) / scale),
        r2,
    )


def response_composites(response: np.ndarray, decisions: np.ndarray) -> np.ndarray:
    """Return samples x locations x COD/TN/TP/TSS concentrations."""

    values = np.asarray(response, dtype=float)
    theta = np.asarray(decisions, dtype=float)
    if values.ndim != 2 or values.shape[1] < 160 or theta.shape != (len(values), 7):
        raise ValueError("unexpected response or decision shape")
    blocks = [values[:, 0:20], *[values[:, 20 * i:20 * (i + 1)] for i in range(1, 6)]]
    overflow = values[:, 120:140] / (1.0 - theta[:, 6])[:, None]
    underflow = values[:, 140:160] / (theta[:, 5] + theta[:, 6])[:, None]
    blocks.extend((overflow, underflow))
    return np.stack([block @ COMPOSITE_MATRIX.T for block in blocks], axis=1)


def route_parity(
    quality: pd.DataFrame,
    *,
    route: str,
    method: str,
    cases: list[str],
    case_labels: dict[str, str],
    output: Path,
    stem: str,
    title: str,
) -> dict[str, float]:
    subset = quality[
        quality["decision_route"].eq(route)
        & quality["response_method"].isin((method, "reference"))
        & quality["available"].astype(str).str.lower().eq("true")
    ]
    pivot = subset.pivot(index="case", columns="response_method", values=list(COMPOSITES)).reindex(cases).dropna()
    if pivot.empty:
        raise RuntimeError(f"no complete {route}/{method} quality pairs")
    errors: list[float] = []
    r2_values: list[float] = []
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    for axis, component in zip(axes.flat, COMPOSITES, strict=True):
        x = pivot[(component, "reference")].to_numpy(float)
        y = pivot[(component, method)].to_numpy(float)
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        pad = max(1.0e-9, 0.06 * (high - low))
        limits = (low - pad, high + pad)
        axis.plot(limits, limits, color=REFERENCE, lw=1.2, ls="--", label="Perfect match")
        axis.scatter(x, y, s=42, color=ROUTE_COLOR[route], edgecolor="white", linewidth=0.6, zorder=3)
        for case, xx, yy in zip(pivot.index, x, y, strict=True):
            axis.annotate(case_labels[case], (xx, yy), xytext=(3, 2), textcoords="offset points", fontsize=6)
        percentage = np.abs(y - x) / np.maximum(np.abs(x), 1.0e-12) * 100.0
        denominator = float(np.sum((x - np.mean(x)) ** 2))
        r2 = 1.0 - float(np.sum((y - x) ** 2)) / denominator if denominator > 0.0 else np.nan
        errors.extend(percentage.tolist())
        r2_values.append(r2)
        axis.set(xlim=limits, ylim=limits, xlabel="Exact mechanistic replay", ylabel=f"{method.capitalize()} prediction")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{component}: median |error| = {np.median(percentage):.1f}%")
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.94))
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    save(fig, output, stem)
    return {
        "median_absolute_percent_error": float(np.median(errors)),
        "mean_absolute_percent_error": float(np.mean(errors)),
        "mean_component_r2": float(np.nanmean(r2_values)),
    }


def removal_parity(
    quality: pd.DataFrame,
    influent: pd.DataFrame,
    *,
    route: str,
    method: str,
    cases: list[str],
    case_labels: dict[str, str],
    output: Path,
    stem: str,
    title: str,
) -> dict[str, float]:
    subset = quality[
        quality["decision_route"].eq(route)
        & quality["response_method"].isin((method, "reference"))
        & quality["available"].astype(str).str.lower().eq("true")
    ]
    pivot = subset.pivot(index="case", columns="response_method", values=list(COMPOSITES)).reindex(cases).dropna()
    errors: list[float] = []
    r2_values: list[float] = []
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    for axis, component in zip(axes.flat, COMPOSITES, strict=True):
        feed = influent.loc[pivot.index, component].to_numpy(float)
        reference = 100.0 * (feed - pivot[(component, "reference")].to_numpy(float)) / feed
        predicted = 100.0 * (feed - pivot[(component, method)].to_numpy(float)) / feed
        low, high = min(reference.min(), predicted.min()), max(reference.max(), predicted.max())
        pad = max(0.2, 0.06 * (high - low))
        limits = (low - pad, high + pad)
        axis.plot(limits, limits, color=REFERENCE, lw=1.2, ls="--", label="Perfect match")
        axis.scatter(reference, predicted, s=42, color=ROUTE_COLOR[route], edgecolor="white", linewidth=0.6, zorder=3)
        for case, xx, yy in zip(pivot.index, reference, predicted, strict=True):
            axis.annotate(case_labels[case], (xx, yy), xytext=(3, 2), textcoords="offset points", fontsize=6)
        absolute = np.abs(predicted - reference)
        denominator = float(np.sum((reference - np.mean(reference)) ** 2))
        r2_values.append(1.0 - float(np.sum((predicted - reference) ** 2)) / denominator if denominator > 0.0 else np.nan)
        errors.extend(absolute.tolist())
        axis.set(xlim=limits, ylim=limits, xlabel="Exact removal (%)", ylabel=f"{method.capitalize()} removal (%)")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{component}: median |error| = {np.median(absolute):.2f} percentage points")
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.94))
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    save(fig, output, stem)
    return {
        "median_absolute_removal_error_percentage_points": float(np.median(errors)),
        "mean_absolute_removal_error_percentage_points": float(np.mean(errors)),
        "mean_component_r2": float(np.nanmean(r2_values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    tables = run / "report" / "tables"
    output = (args.output or run / "report/figures/system_surrogate_vs_smooth_nlp").resolve()
    output.mkdir(parents=True, exist_ok=True)
    style()

    robust_cases = [f"robustness_{index:02d}" for index in range(1, 11)]
    all_cases = ["nominal", *robust_cases]
    case_labels = {"nominal": "N", **{case: f"R{i}" for i, case in enumerate(robust_cases, 1)}}
    summary: list[dict[str, object]] = []

    with np.load(run / "datasets/effective_design.npz", allow_pickle=False) as stored:
        development_decisions = np.asarray(stored["development_decisions"], dtype=float)
        test_decisions = np.asarray(stored["test_decisions"], dtype=float)
        robustness_influents = np.asarray(stored["robustness_influents"], dtype=float)
    with np.load(run / "predictions/post_selection_holdout.npz", allow_pickle=False) as stored:
        holdout = {name: np.asarray(stored[name], dtype=float) for name in ("mechanistic", "raw", "projected")}
    composite_predictions = {
        name: response_composites(values, test_decisions) for name, values in holdout.items()
    }

    # Q1--Q4: every performance statement is derived in COD/TN/TP/TSS space.
    metric_rows: list[dict[str, object]] = []
    scales = np.ptp(composite_predictions["mechanistic"], axis=0)
    if np.any(scales <= 0.0):
        raise RuntimeError("holdout composite truth contains a zero-range location/quantity")
    for method in ("raw", "projected"):
        normalized_error = (
            composite_predictions[method] - composite_predictions["mechanistic"]
        ) / scales[None, :, :]
        per_coordinate_r2 = []
        for location in range(len(LOCATIONS)):
            for component in range(len(COMPOSITES)):
                _, _, r2 = finite_score(
                    composite_predictions["mechanistic"][:, location, component],
                    composite_predictions[method][:, location, component],
                )
                per_coordinate_r2.append(r2)
        metric_rows.append({
            "method": method,
            "nrmse": float(np.sqrt(np.mean(normalized_error**2))),
            "nmae": float(np.mean(np.abs(normalized_error))),
            "r2_mean": float(np.nanmean(per_coordinate_r2)),
        })
    overall = pd.DataFrame(metric_rows).set_index("method")
    overall.reset_index().to_csv(output / "holdout_composite_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for axis, (column, label, lower) in zip(
        axes,
        (("nrmse", "Normalized RMSE", True), ("nmae", "Normalized MAE", True), ("r2_mean", "Mean R²", False)),
        strict=True,
    ):
        values = overall.loc[["raw", "projected"], column].to_numpy(float)
        bars = axis.bar(("Raw", "Projected"), values, color=(RAW, EXTENDED), width=0.62)
        for bar in bars:
            axis.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width()/2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
        axis.set_title(f"{label} ({'lower' if lower else 'higher'} is better)")
        axis.set_ylabel(label)
    fig.suptitle("Q1. Holdout accuracy across COD, TN, TP, and TSS at all system locations", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, output, "q01_holdout_composite_accuracy")
    improvement = 100.0 * (overall.loc["raw", "nrmse"] - overall.loc["projected", "nrmse"]) / overall.loc["raw", "nrmse"]
    summary.append({"question": 1, "metric": "composite_projection_nrmse_improvement_percent", "value": improvement})

    location_nrmse = {method: np.sqrt(np.mean(((composite_predictions[method] - composite_predictions["mechanistic"]) / scales[None, :, :]) ** 2, axis=(0, 2))) for method in ("raw", "projected")}
    x = np.arange(len(LOCATIONS)); width = 0.36
    fig, axis = plt.subplots(figsize=(11, 5.2))
    axis.bar(x-width/2, location_nrmse["raw"], width, color=RAW, label="Raw")
    axis.bar(x+width/2, location_nrmse["projected"], width, color=EXTENDED, label="Projected")
    axis.set(xticks=x, xticklabels=LOCATION_LABELS, ylabel="Composite nRMSE", title="Q2. Composite prediction accuracy by system location")
    axis.legend(ncol=2)
    fig.tight_layout(); save(fig, output, "q02_holdout_accuracy_by_response_block")
    summary.append({"question": 2, "metric": "locations_improved_by_projection", "value": int(np.sum(location_nrmse["projected"] < location_nrmse["raw"]))})

    component_scores: dict[str, list[tuple[float, float, float]]] = {"raw": [], "projected": []}
    for method in component_scores:
        for component in range(len(COMPOSITES)):
            component_scores[method].append(finite_score(
                composite_predictions["mechanistic"][:, :, component].ravel(),
                composite_predictions[method][:, :, component].ravel(),
            ))
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5))
    for axis, metric_index, ylabel, title in (
        (axes[0, 0], 0, "nRMSE", "Range-normalized RMSE"),
        (axes[0, 1], 1, "nMAE", "Range-normalized MAE"),
        (axes[1, 0], 2, "R²", "Coefficient of determination"),
    ):
        raw_values = [row[metric_index] for row in component_scores["raw"]]
        projected_values = [row[metric_index] for row in component_scores["projected"]]
        axis.bar(np.arange(4)-width/2, raw_values, width, color=RAW, label="Raw")
        axis.bar(np.arange(4)+width/2, projected_values, width, color=EXTENDED, label="Projected")
        axis.set(xticks=np.arange(4), xticklabels=COMPOSITES, ylabel=ylabel, title=title)
    axes[1, 1].axis("off")
    axes[0, 0].legend(ncol=2)
    fig.suptitle("Q3. Holdout prediction accuracy by composite quantity", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, output, "q03_holdout_component_accuracy")
    summary.append({"question": 3, "metric": "composites_improved_by_projection", "value": int(sum(component_scores["projected"][i][0] < component_scores["raw"][i][0] for i in range(4)))})

    matrices = {}
    for method in ("raw", "projected"):
        matrix = np.empty((len(LOCATIONS), len(COMPOSITES)))
        for i in range(len(LOCATIONS)):
            for j in range(len(COMPOSITES)):
                matrix[i, j] = finite_score(composite_predictions["mechanistic"][:, i, j], composite_predictions[method][:, i, j])[0]
        matrices[method] = matrix
    delta = 100.0 * (matrices["projected"] - matrices["raw"]) / matrices["raw"]
    limit = float(np.nanmax(np.abs(delta)))
    common_max = float(max(np.nanmax(matrices["raw"]), np.nanmax(matrices["projected"])))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.3))
    for axis, matrix, title in ((axes[0], matrices["raw"], "Raw composite nRMSE"), (axes[1], matrices["projected"], "Projected composite nRMSE")):
        image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=common_max)
        fig.colorbar(image, ax=axis, pad=0.02)
        axis.set_title(title)
        axis.set(yticks=np.arange(len(LOCATIONS)), yticklabels=LOCATION_LABELS, xticks=np.arange(4), xticklabels=COMPOSITES)
    image = axes[2].imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    fig.colorbar(image, ax=axes[2], pad=0.02, label="Change (%)")
    axes[2].set_title("Projection change (negative improves)")
    axes[2].set(yticks=np.arange(len(LOCATIONS)), yticklabels=LOCATION_LABELS, xticks=np.arange(4), xticklabels=COMPOSITES)
    fig.suptitle("Q4. Composite prediction accuracy by treatment location", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); save(fig, output, "q04_holdout_component_accuracy_by_stage")
    summary.append({"question": 4, "metric": "location_composite_cells_improved", "value": int(np.sum(delta < 0.0))})

    quality_path = tables / "selected_quality.csv"
    if quality_path.is_file():
        quality = pd.read_csv(quality_path)
    else:
        quality_rows: list[dict[str, object]] = []
        for case in all_cases:
            for route in ROUTES:
                with np.load(run / "optimization" / case / f"{route}_casewise_reference.npz", allow_pickle=False) as stored:
                    theta = np.asarray(stored["theta"], dtype=float)
                    method_arrays = (
                        (("raw", np.asarray(stored["raw"], dtype=float)), ("projected", np.asarray(stored["projected"], dtype=float)))
                        if route == "surrogate"
                        else (("smooth", np.asarray(stored["optimizer_native"], dtype=float)),)
                    )
                    method_arrays = (*method_arrays, ("reference", np.asarray(stored["exact_reference"], dtype=float)))
                for method, response in method_arrays:
                    effluent = response[120:140] / (1.0 - theta[6])
                    values = COMPOSITE_MATRIX @ effluent
                    quality_rows.append({
                        "case": case, "decision_route": route,
                        "response_method": method, "available": True,
                        **dict(zip(COMPOSITES, values, strict=True)),
                        "objective": np.nan,
                    })
        quality = pd.DataFrame(quality_rows)
    for route, method, question, stem, title in (
        ("surrogate", "projected", 5, "q05_surrogate_effluent_prediction_vs_mechanistic", "Q5. Extended-ICSOR effluent prediction vs exact mechanistic replay"),
        ("direct", "smooth", 6, "q06_smooth_nlp_effluent_prediction_vs_mechanistic", "Q6. Smooth-NLP effluent prediction vs exact mechanistic replay"),
    ):
        metrics = route_parity(quality, route=route, method=method, cases=all_cases, case_labels=case_labels, output=output, stem=stem, title=title)
        summary.extend({"question": question, "metric": key, "value": value} for key, value in metrics.items())

    influent_values = np.vstack((NOMINAL_INFLUENT, robustness_influents)) @ COMPOSITE_MATRIX.T
    influent = pd.DataFrame(influent_values, index=all_cases, columns=COMPOSITES)
    for route, method, question, stem, title in (
        ("surrogate", "projected", "5R", "q05_surrogate_percent_removal_vs_mechanistic", "Extended-ICSOR removal prediction vs exact mechanistic replay"),
        ("direct", "smooth", "6R", "q06_smooth_nlp_percent_removal_vs_mechanistic", "Smooth-NLP removal prediction vs exact mechanistic replay"),
    ):
        metrics = removal_parity(quality, influent, route=route, method=method, cases=all_cases, case_labels=case_labels, output=output, stem=stem, title=title)
        summary.extend({"question": question, "metric": key, "value": value} for key, value in metrics.items())

    with np.load(run / "datasets/development/mechanistic_accepted_v3.npz", allow_pickle=False) as stored:
        development_targets = np.asarray(stored["targets"], dtype=float)
    development_effluent = development_targets[:, 120:140] / (1.0 - development_decisions[:, 6])[:, None]
    quality_scale = np.std(development_effluent @ COMPOSITE_MATRIX.T, axis=0, ddof=0)
    weights = np.asarray((0.50, 0.15, 0.20, 0.05, 0.05, 0.05))
    exact_rows: list[dict[str, object]] = []
    for case in all_cases:
        for route in ROUTES:
            with np.load(run / "optimization" / case / f"{route}_casewise_reference.npz", allow_pickle=False) as stored:
                theta = np.asarray(stored["theta"], dtype=float)
                response = np.asarray(stored["exact_reference"], dtype=float)
            effluent = response[120:140] / (1.0 - theta[6])
            composites = COMPOSITE_MATRIX @ effluent
            underflow = response[140:160] / (theta[5] + theta[6])
            parts = np.asarray((
                np.mean(composites / quality_scale),
                (theta[0] - 6.0) / 30.0,
                theta[0] * np.sum(theta[1:4]) / 108.0,
                theta[4] / 4.0,
                (theta[5] - 0.25) / 1.0,
                theta[6] * float(TSS_VECTOR @ underflow) / 750.0,
            ))
            exact_rows.append({"case": case, "route": route, **dict(zip(COMPOSITES, composites, strict=True)), "quality": parts[0], "hrt": parts[1], "aeration": parts[2], "internal_recycle": parts[3], "return_sludge": parts[4], "wasting": parts[5], "economic": float(weights[1:] @ parts[1:]), "objective": float(weights @ parts)})
    exact = pd.DataFrame(exact_rows)
    comparison_path = tables / "scenario_comparison.csv"
    if not comparison_path.is_file():
        comparison_path = run / "metrics/case_common_reference_comparison.csv"
    comparison = pd.read_csv(comparison_path).set_index("case")
    eligibility = comparison["comparison_eligible"].astype(str).str.lower().eq("true").reindex(robust_cases).fillna(False)
    eligible = eligibility.to_numpy(bool)
    x = np.arange(len(robust_cases)); width = 0.37

    def shade(axis: plt.Axes) -> None:
        for index, valid in enumerate(eligible):
            if not valid:
                axis.axvspan(index - 0.5, index + 0.5, color=BAD, alpha=0.55, zorder=0)

    def paired_bars(column: str, stem: str, title: str, ylabel: str, question: int) -> None:
        pivot = exact[exact.case.isin(robust_cases)].pivot(index="case", columns="route", values=column).reindex(robust_cases)
        fig, axis = plt.subplots(figsize=(12, 5.4)); shade(axis)
        axis.bar(x-width/2, pivot["surrogate"], width, color=EXTENDED, label=ROUTE_LABEL["surrogate"])
        axis.bar(x+width/2, pivot["direct"], width, color=DIRECT, label=ROUTE_LABEL["direct"])
        axis.set(xticks=x, xticklabels=[case_labels[c] for c in robust_cases], ylabel=ylabel, title=title)
        axis.legend(handles=(Patch(facecolor=EXTENDED, label=ROUTE_LABEL["surrogate"]), Patch(facecolor=DIRECT, label=ROUTE_LABEL["direct"]), Patch(facecolor=BAD, alpha=.55, label="Comparison ineligible")), ncol=3)
        fig.tight_layout(); save(fig, output, stem)
        valid = pivot.loc[eligibility]
        summary.extend((
            {"question": question, "metric": "eligible_cases", "value": int(eligible.sum())},
            {"question": question, "metric": "extended_icsor_lower_count", "value": int((valid.surrogate < valid.direct).sum())},
            {"question": question, "metric": "smooth_nlp_lower_count", "value": int((valid.direct < valid.surrogate).sum())},
        ))

    paired_bars("objective", "q07_exact_optimal_objective", "Q7. Exact objective at selected decisions across robustness cases", "Exact total objective", 7)
    paired_bars("quality", "q08_exact_water_quality_component", "Q8. Exact normalized water-quality component across robustness cases", "Normalized water-quality component", 8)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, component in zip(axes.flat, COMPOSITES, strict=True):
        pivot = exact[exact.case.isin(robust_cases)].pivot(index="case", columns="route", values=component).reindex(robust_cases)
        shade(axis)
        axis.plot(x, pivot["surrogate"], marker="o", color=EXTENDED, label=ROUTE_LABEL["surrogate"])
        axis.plot(x, pivot["direct"], marker="s", color=DIRECT, label=ROUTE_LABEL["direct"])
        axis.set(xticks=x, xticklabels=[case_labels[c] for c in robust_cases], ylabel=f"{component} (mg/L)", title=component)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(.5, .95))
    fig.suptitle("Q9. Exact effluent composites at the selected decisions", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .91)); save(fig, output, "q09_exact_effluent_composites")
    pair = exact[exact.case.isin(np.asarray(robust_cases)[eligible])].pivot(index="case", columns="route", values=list(COMPOSITES))
    relative = np.abs(pair.xs("surrogate", axis=1, level="route") - pair.xs("direct", axis=1, level="route")) / np.maximum(np.abs(pair.xs("direct", axis=1, level="route")), 1e-12)
    summary.append({"question": 9, "metric": "median_absolute_route_difference_percent", "value": float(100*np.median(relative.to_numpy())) if not relative.empty else np.nan})

    economic_columns = ("hrt", "aeration", "internal_recycle", "return_sludge", "wasting")
    economic_labels = ("HRT", "Aeration", "Internal recycle", "Return sludge", "Wasting")
    economic_colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2")
    fig, axis = plt.subplots(figsize=(13, 6.2)); shade(axis)
    for route, offset, hatch in (("surrogate", -width/2, ""), ("direct", width/2, "///")):
        data = exact[exact.route.eq(route) & exact.case.isin(robust_cases)].set_index("case").reindex(robust_cases)
        bottom = np.zeros(10)
        for column, label, color, weight in zip(economic_columns, economic_labels, economic_colors, weights[1:], strict=True):
            values = weight * data[column].to_numpy(float)
            axis.bar(x+offset, values, width, bottom=bottom, color=color, edgecolor="white", linewidth=.3, hatch=hatch, label=label if route == "surrogate" else None)
            bottom += values
    axis.set(xticks=x, xticklabels=[case_labels[c] for c in robust_cases], ylabel="Weighted economic/resource contribution", title="Q10. Exact economic/resource contribution")
    axis.legend(handles=[*[Patch(facecolor=c, label=l) for c, l in zip(economic_colors, economic_labels, strict=True)], Patch(facecolor="white", edgecolor="black", label="Extended ICSOR: left/solid"), Patch(facecolor="white", edgecolor="black", hatch="///", label="Smooth NLP: right/hatched")], ncol=4, loc="upper center", bbox_to_anchor=(.5, -.12))
    fig.tight_layout(rect=(0, .12, 1, 1)); save(fig, output, "q10_exact_economic_component")

    controls_path = tables / "scenario_controls.csv"
    if controls_path.is_file():
        controls = pd.read_csv(controls_path).set_index(["case", "route"])
    else:
        control_rows: list[dict[str, object]] = []
        for case in robust_cases:
            for route in ROUTES:
                with np.load(run / "optimization" / case / f"{route}_casewise_reference.npz", allow_pickle=False) as stored:
                    theta = np.asarray(stored["theta"], dtype=float)
                control_rows.append({
                    "case": case, "route": route,
                    **dict(zip(("H", "a_3", "a_4", "a_5", "r_I", "r_R", "w"), theta, strict=True)),
                })
        controls = pd.DataFrame(control_rows).set_index(["case", "route"])
    control_columns = ("H", "a_3", "a_4", "a_5", "r_I", "r_R", "w")
    control_titles = ("HRT H", "Aeration a3", "Aeration a4", "Aeration a5", "Internal recycle rI", "Return sludge rR", "Waste fraction w")
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    for axis, column, title in zip(axes.flat, control_columns, control_titles, strict=False):
        shade(axis)
        for route in ROUTES:
            values = np.asarray([controls.loc[(case, route), column] for case in robust_cases], float)
            axis.plot(x, values, marker=ROUTE_MARKER[route], color=ROUTE_COLOR[route], label=ROUTE_LABEL[route])
        axis.set_title(title); axis.set_xticks(x, [case_labels[c] for c in robust_cases])
    axes.flat[-1].axis("off")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower right", bbox_to_anchor=(.93, .08))
    fig.suptitle("Q11. Selected operating decisions across robustness cases", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96)); save(fig, output, "q11_optimal_operating_values")

    timing = pd.read_csv(run / "metrics/robustness_case_timing.csv").pivot(index="case", columns="route", values="primary_optimization_seconds").reindex(index=robust_cases, columns=ROUTES)
    fig, axis = plt.subplots(figsize=(12, 5.7))
    axis.bar(x-width/2, timing["surrogate"], width, color=EXTENDED, label=f"{ROUTE_LABEL['surrogate']} primary")
    axis.bar(x+width/2, timing["direct"], width, color=DIRECT, label=f"{ROUTE_LABEL['direct']} primary")
    axis.set_yscale("log"); axis.set(xticks=x, xticklabels=[case_labels[c] for c in robust_cases], ylabel="Primary optimization time (s, log scale)", title="Q12. Primary optimization time across robustness cases")
    axis.legend(ncol=2); fig.tight_layout(); save(fig, output, "q12_primary_optimization_time")
    summary.extend((
        {"question": 12, "metric": "extended_icsor_mean_seconds", "value": float(timing.surrogate.mean())},
        {"question": 12, "metric": "smooth_nlp_mean_seconds", "value": float(timing.direct.mean())},
        {"question": 12, "metric": "extended_icsor_faster_case_count", "value": int((timing.surrogate < timing.direct).sum())},
    ))

    all_objectives = exact.pivot(index="case", columns="route", values="objective").reindex(all_cases)
    all_x = np.arange(len(all_cases))
    fig, axis = plt.subplots(figsize=(12.5, 5.7))
    axis.bar(all_x-width/2, all_objectives.surrogate, width, color=EXTENDED, label=ROUTE_LABEL["surrogate"])
    axis.bar(all_x+width/2, all_objectives.direct, width, color=DIRECT, label=ROUTE_LABEL["direct"])
    for index, valid in enumerate((True, *eligible)):
        if not valid:
            axis.axvspan(index-.5, index+.5, color=BAD, alpha=.55, zorder=0)
    axis.set(xticks=all_x, xticklabels=[case_labels[c] for c in all_cases], ylabel="Exact total objective", title="Q13. Exact objective value at both routes' selected decisions")
    axis.legend(ncol=2); fig.tight_layout(); save(fig, output, "q13_exact_objective_value_comparison")
    summary.extend({"question": 13, "metric": f"{route}_nominal_exact_objective", "value": float(all_objectives.loc["nominal", route])} for route in ROUTES)

    profile_labels = ("Influent", "Mixer", "R1", "R2", "R3", "R4", "R5", "Effluent")
    case_colors = {"nominal": "#111827", **{case: plt.get_cmap("tab20")(index) for index, case in enumerate(robust_cases)}}
    all_eligible = pd.Series((True, *eligible), index=all_cases)
    for question, component, stem in (
        (14, "COD", "q14_cod_main_treatment_train_profiles"),
        (15, "TN", "q15_tn_main_treatment_train_profiles"),
        (16, "TP", "q16_tp_main_treatment_train_profiles"),
        (17, "TSS", "q17_tss_main_treatment_train_profiles"),
    ):
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        component_index = COMPOSITES.index(component)
        for axis, route in zip(axes, ROUTES, strict=True):
            for case in all_cases:
                with np.load(run / "optimization" / case / f"{route}_casewise_reference.npz", allow_pickle=False) as stored:
                    full = np.asarray(stored["exact_reference_full"], dtype=float)
                    theta = np.asarray(stored["theta"], dtype=float)
                liquid = [full[0:20], *[full[20*i:20*(i+1)] for i in range(1, 6)], full[120:140] / (1.0-theta[6])]
                values = np.asarray([influent.loc[case, component], *[float(COMPOSITE_MATRIX[component_index] @ block) for block in liquid]])
                axis.plot(np.arange(len(profile_labels)), values, color=case_colors[case], lw=2.2 if case == "nominal" else 1.25, ls="-" if all_eligible.loc[case] else "--", marker="o", ms=4.2, alpha=1.0 if all_eligible.loc[case] else .72)
            axis.set_yscale("log"); axis.set_ylabel(f"{component} (mg/L, log scale)"); axis.set_title(ROUTE_LABEL[route]); axis.set_xticks(np.arange(len(profile_labels)), profile_labels)
        handles = [Line2D((0,), (0,), color=case_colors[case], lw=2.2 if case == "nominal" else 1.5, ls="-" if all_eligible.loc[case] else "--", marker="o", label=case_labels[case]) for case in all_cases]
        fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(.5, .005), title="Case")
        fig.suptitle(f"Q{question}. {component} evolution through the main treatment train", fontsize=14)
        fig.tight_layout(rect=(0, .14, 1, .96)); save(fig, output, stem)

    pd.DataFrame(summary).to_csv(output / "chart_summary.csv", index=False)
    stems = {
        1: "q01_holdout_composite_accuracy", 2: "q02_holdout_accuracy_by_response_block",
        3: "q03_holdout_component_accuracy", 4: "q04_holdout_component_accuracy_by_stage",
        5: "q05_surrogate_effluent_prediction_vs_mechanistic", 6: "q06_smooth_nlp_effluent_prediction_vs_mechanistic",
        7: "q07_exact_optimal_objective", 8: "q08_exact_water_quality_component",
        9: "q09_exact_effluent_composites", 10: "q10_exact_economic_component",
        11: "q11_optimal_operating_values", 12: "q12_primary_optimization_time",
        13: "q13_exact_objective_value_comparison", 14: "q14_cod_main_treatment_train_profiles",
        15: "q15_tn_main_treatment_train_profiles", 16: "q16_tp_main_treatment_train_profiles",
        17: "q17_tss_main_treatment_train_profiles", "5R": "q05_surrogate_percent_removal_vs_mechanistic",
        "6R": "q06_smooth_nlp_percent_removal_vs_mechanistic",
    }
    pd.DataFrame([
        {"question": question, "png": f"{stem}.png", "svg": f"{stem}.svg"}
        for question, stem in stems.items()
    ]).to_csv(output / "chart_index.csv", index=False)
    print(output)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
