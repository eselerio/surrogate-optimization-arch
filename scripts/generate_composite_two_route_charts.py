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
from matplotlib.colors import LogNorm
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
SAVE_STEMS: set[str] | None = None

# The retained article package is numbered in Results-and-Discussion order,
# rather than by the historical analytical-question IDs.  The source stems
# let --existing-only migrate an earlier version without recreating removed
# charts.
ARTICLE_FIGURE_STEMS = {
    "q01_holdout_accuracy_overview": {
        "q01_q02_q03_holdout_accuracy", "q01_holdout_composite_accuracy",
        "q02_holdout_accuracy_by_response_block", "q03_holdout_component_accuracy",
    },
    "q02_holdout_accuracy_by_location": {"q04_holdout_component_accuracy_by_stage"},
    "q03_holdout_parity_all_locations": {"q18_holdout_effluent_composite_parity"},
    "q04_effluent_and_removal_parity": {
        "q05_effluent_and_removal_parity", "q05_surrogate_effluent_prediction_vs_mechanistic",
        "q05_surrogate_percent_removal_vs_mechanistic",
    },
    "q05_effluent_and_operating_values": {
        "q09_q11_effluent_and_operating_values", "q09_exact_effluent_composites",
        "q11_optimal_operating_values",
    },
    "q06_treatment_train_profiles": {
        "q14_cod_tn_tp_tss_main_treatment_train_profiles", "q14_cod_main_treatment_train_profiles",
        "q15_tn_main_treatment_train_profiles", "q15_tn_tp_tss_main_treatment_train_profiles",
        "q16_tp_main_treatment_train_profiles", "q17_tss_main_treatment_train_profiles",
    },
    "q07_objective_quality_economic": {
        "q07_q08_q10_objective_quality_economic", "q07_exact_optimal_objective",
        "q08_exact_water_quality_component", "q10_exact_economic_component",
    },
    "q08_optimization_time": {"q12_primary_optimization_time"},
}


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
    # Captions belong in the companion README, not inside publication figures.
    # Clear figure- and axes-level titles immediately before every export so a
    # future plotting change cannot accidentally reintroduce a title.
    if fig._suptitle is not None:
        fig._suptitle.remove()
    for axis in fig.axes:
        axis.set_title("")
    if SAVE_STEMS is None or stem in SAVE_STEMS:
        fig.savefig(output / f"{stem}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_legacy_readme(output: Path) -> None:
    """Document the chart package in place of embedded figure titles."""

    output.joinpath("README.md").write_text(
        """# Article-v3 figure package

These untitled figures are generated only from the parent result run. Each
figure is retained as a PNG. `chart_index.csv` maps the question identifiers to
file names; `chart_summary.csv` contains the principal numerical comparisons.

## Common conventions

- **Extended ICSOR** is the `surrogate` route; **Smooth NLP** is the `direct`
  route. `N` denotes the nominal scenario and `S1`-`S10` denote influent
  scenarios 1-10.
- Water-quality quantities are COD, TN, TP, and TSS concentrations in mg/L.
  "Exact mechanistic replay" means the mechanistic model evaluated at the
  selected route decision.
- Holdout errors are coordinate-normalized: each location/composite error is
  divided by that location/composite's mechanistic holdout range. nRMSE and
  nMAE are lower-is-better; mean location R^2 is higher-is-better.
- The nominal scenario and all influent scenarios are plotted and included in paired
  comparisons.
- All figure titles are deliberately omitted. Panel legends, axis labels, and
  this README provide the interpretation.

## Figures and target-run sources

- **Q1/Q2/Q3** `q01_q02_q03_holdout_accuracy`: combined holdout accuracy:
  aggregate nRMSE, nMAE, and mean location R^2; location-level nRMSE, nMAE,
  and mean location R^2; and nRMSE, nMAE, and mean location R^2 for COD, TN,
  TP, and TSS.
  Sources: `predictions/post_selection_holdout.npz` and
  `datasets/effective_design.npz`.
- **Q4** `q04_holdout_component_accuracy_by_stage`: location-by-composite raw
  nRMSE, projected nRMSE, and percent nRMSE change (negative is improvement).
  Same holdout sources as Q1.
- **Q5/Q5R** `q05_effluent_and_removal_parity`: combined Extended-ICSOR
  effluent-concentration and removal parity against exact replay. Teal dots are
  concentration parity; orange dots are removal parity. Source: `report/tables/selected_quality.csv`
  (or the casewise-reference files when that table is unavailable).
- **Q6/Q6R** Smooth-NLP selected-scenario concentration and removal parity;
  differences are percentage points. Sources: selected-quality data plus
  `datasets/effective_design.npz` and the nominal influent contract.
- **Q7/Q8/Q10** `q07_q08_q10_objective_quality_economic`: one horizontal
  figure of exact total objective, its normalized water-quality component, and
  weighted HRT, aeration, recycle, return-sludge, and wasting contributions for
  the nominal and every influent scenario. Source:
  `optimization/<case>/*_casewise_reference.npz` and development targets.
- **Q9/Q11** `q09_q11_effluent_and_operating_values`: exact-replay effluent
  COD, TN, TP, and TSS alongside selected HRT, aeration, recycle, return-sludge,
  and wasting controls for the nominal and every influent scenario. Source:
  `report/tables/scenario_controls.csv` (or casewise-reference `theta`).
- **Q12** `q12_primary_optimization_time`: optimization time only for the nominal
  and every influent scenario, shown on a logarithmic seconds axis. Sources:
  `metrics/robustness_case_timing.csv` and the nominal candidate evaluation.
- **Q13** `q13_exact_objective_value_comparison`: exact total objective for the
  nominal scenario and all influent scenarios. Source: casewise-reference files.
- **Q14** `q14_cod_tn_tp_tss_main_treatment_train_profiles`: combined COD,
  TN, TP, and TSS treatment-train profiles. Each follows influent → mixer →
  reactor R1–R5 → clarifier effluent; y-axes are logarithmic. Source:
  `exact_reference_full`, `theta`, and influents in casewise-reference files
  and `datasets/effective_design.npz`.
- **Q18** `q18_holdout_effluent_composite_parity`: projected Extended ICSOR
  parity for all holdout rows and all eight locations. Hexagon color is the
  number of location-observations; location medians are marked. Sources are the
  post-selection holdout predictions and test decisions.

Composite calculations use the repository's authoritative `COMPOSITE_MATRIX`;
overflow and underflow component flows are converted to concentrations before
composites are calculated. No source data from another run are used.
""",
        encoding="utf-8",
    )


def write_readme(output: Path) -> None:
    """Write the README for the retained, sequential article figure package."""

    output.joinpath("README.md").write_text(
        """# Article-v3 figure package

These untitled PNG figures are generated only from the parent result run.
`chart_index.csv` maps the Results sequence number, historical source
questions, and presentation role to each file. `chart_summary.csv` contains
the principal numerical comparisons.

## Common conventions

- **Extended ICSOR** is the `surrogate` route; **Smooth NLP** is the `direct`
  route. `N` is the nominal scenario; `S1`--`S10` are influent scenarios.
- Water-quality quantities are COD, TN, TP, and TSS concentrations in mg/L.
  Exact mechanistic replay evaluates the mechanistic model at a selected
  route decision. nRMSE and nMAE are lower-is-better; mean location R^2 is
  higher-is-better.
- The nominal and all influent scenarios appear in paired comparisons. Figure
  titles are deliberately omitted; axes, legends, and this README interpret
  the charts.

## Results and discussion presentation plan

The sequence moves from evidence to interpretation: (1) holdout validation,
(2) exact selected-decision replay, (3) operating and treatment-train behavior,
and (4) objective trade-offs and primary search time. Complementary views of
the same question are adjacent.

## Figures and target-run sources

- **Results sequence 1** `q01_holdout_accuracy_overview.png` (Q1/Q2/Q3):
  aggregate, location, and composite holdout accuracy. Sources:
  `predictions/post_selection_holdout.npz`, `datasets/effective_design.npz`.
- **Results sequence 2** `q02_holdout_accuracy_by_location.png` (Q4): raw and
  projected nRMSE by location and composite, plus percentage change.
- **Results sequence 3** `q03_holdout_parity_all_locations.png` (Q18):
  all-location projected-holdout parity and named location medians.
- **Results sequence 4** `q04_effluent_and_removal_parity.png` (Q5/Q5R):
  selected Extended-ICSOR concentration and removal parity against replay.
- **Results sequence 5** `q05_effluent_and_operating_values.png` (Q9/Q11):
  exact effluent outcomes and operating controls for both routes.
- **Results sequence 6** `q06_treatment_train_profiles.png` (Q14): COD, TN,
  TP, and TSS treatment-train profiles for both routes.
- **Results sequence 7** `q07_objective_quality_economic.png` (Q7/Q8/Q10):
  exact total objective, quality contribution, and resource contributions.
- **Results sequence 8** `q08_optimization_time.png` (Q12): primary search
  time for nominal and all influent scenarios.
""",
        encoding="utf-8",
    )


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


def coordinate_normalized_score(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[float, float, float]:
    """Score sample-by-coordinate data with every coordinate weighted equally."""

    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if (
        truth.ndim != 2
        or truth.shape != prediction.shape
        or not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(prediction))
    ):
        raise ValueError("coordinate score received incompatible or non-finite arrays")
    scales = np.ptp(truth, axis=0)
    if np.any(scales <= 0.0):
        raise ValueError("coordinate score requires a nonzero truth range per coordinate")
    normalized_error = (prediction - truth) / scales[None, :]
    coordinate_r2 = [
        finite_score(truth[:, index], prediction[:, index])[2]
        for index in range(truth.shape[1])
    ]
    return (
        float(np.sqrt(np.mean(normalized_error**2))),
        float(np.mean(np.abs(normalized_error))),
        float(np.nanmean(coordinate_r2)),
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
        axis.set(
            xlim=limits, ylim=limits,
            xlabel="Mechanistic (mg/L)",
            ylabel=f"{ROUTE_LABEL[route]} prediction (mg/L)",
        )
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
        axis.set(
            xlim=limits, ylim=limits,
            xlabel="Mechanistic removal (%)",
            ylabel=f"{ROUTE_LABEL[route]} removal (%)",
        )
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


def combined_q05_parity(
    quality: pd.DataFrame,
    influent: pd.DataFrame,
    *,
    cases: list[str],
    case_labels: dict[str, str],
    output: Path,
) -> None:
    """Combine Q5 concentration and removal parity with distinct dot colors."""

    route = "surrogate"
    method = "projected"
    subset = quality[
        quality["decision_route"].eq(route)
        & quality["response_method"].isin((method, "reference"))
        & quality["available"].astype(str).str.lower().eq("true")
    ]
    pivot = subset.pivot(index="case", columns="response_method", values=list(COMPOSITES)).reindex(cases).dropna()
    if pivot.empty:
        raise RuntimeError("no complete Extended-ICSOR Q5 quality pairs")

    concentration_color = "#147D92"
    removal_color = "#D97904"
    fig, axes = plt.subplots(2, len(COMPOSITES), figsize=(19.5, 9.2), squeeze=False)
    for column, component in enumerate(COMPOSITES):
        reference = pivot[(component, "reference")].to_numpy(float)
        predicted = pivot[(component, method)].to_numpy(float)
        low, high = min(reference.min(), predicted.min()), max(reference.max(), predicted.max())
        pad = max(1.0e-9, 0.06 * (high - low))
        limits = (low - pad, high + pad)
        axis = axes[0, column]
        axis.plot(limits, limits, color=REFERENCE, lw=1.2, ls="--")
        axis.scatter(reference, predicted, s=42, color=concentration_color, edgecolor="white", linewidth=0.6, zorder=3)
        for case, xx, yy in zip(pivot.index, reference, predicted, strict=True):
            axis.annotate(case_labels[case], (xx, yy), xytext=(3, 2), textcoords="offset points", fontsize=6)
        axis.set(
            xlim=limits, ylim=limits,
            xlabel="Mechanistic (mg/L)",
            ylabel=f"{component} - Extended ICSOR prediction (mg/L)",
        )
        axis.set_aspect("equal", adjustable="box")

        feed = influent.loc[pivot.index, component].to_numpy(float)
        removal_reference = 100.0 * (feed - reference) / feed
        removal_predicted = 100.0 * (feed - predicted) / feed
        low, high = min(removal_reference.min(), removal_predicted.min()), max(removal_reference.max(), removal_predicted.max())
        pad = max(0.2, 0.06 * (high - low))
        limits = (low - pad, high + pad)
        axis = axes[1, column]
        axis.plot(limits, limits, color=REFERENCE, lw=1.2, ls="--")
        axis.scatter(removal_reference, removal_predicted, s=42, color=removal_color, edgecolor="white", linewidth=0.6, zorder=3)
        for case, xx, yy in zip(pivot.index, removal_reference, removal_predicted, strict=True):
            axis.annotate(case_labels[case], (xx, yy), xytext=(3, 2), textcoords="offset points", fontsize=6)
        axis.set(
            xlim=limits, ylim=limits,
            xlabel="Mechanistic removal (%)",
            ylabel=f"{component} - Extended ICSOR removal (%)",
        )
        axis.set_aspect("equal", adjustable="box")
    fig.legend(
        handles=[
            Line2D((0,), (0,), color=REFERENCE, lw=1.2, ls="--", label="Perfect match"),
            Line2D((0,), (0,), marker="o", color="none", markerfacecolor=concentration_color, markeredgecolor="white", markersize=7, label="Effluent concentration parity"),
            Line2D((0,), (0,), marker="o", color="none", markerfacecolor=removal_color, markeredgecolor="white", markersize=7, label="Removal parity"),
        ],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, output, "q04_effluent_and_removal_parity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--eligibility-file",
        type=Path,
        help="Optional re-adjudicated comparison CSV; the scientific run is not edited.",
    )
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Regenerate only PNG figures that already exist in the output directory.",
    )
    args = parser.parse_args()
    run = args.run.resolve()
    tables = run / "report" / "tables"
    output = (args.output or run / "report/figures/system_surrogate_vs_smooth_nlp").resolve()
    output.mkdir(parents=True, exist_ok=True)
    global SAVE_STEMS
    if args.existing_only:
        SAVE_STEMS = {path.stem for path in output.glob("*.png")}
        for article_stem, prior_stems in ARTICLE_FIGURE_STEMS.items():
            if SAVE_STEMS.intersection(prior_stems):
                SAVE_STEMS.add(article_stem)
            SAVE_STEMS.difference_update(prior_stems)
    else:
        # The article package is deliberately curated to the eight figures in
        # Results-and-Discussion order; removed legacy charts remain absent.
        SAVE_STEMS = set(ARTICLE_FIGURE_STEMS)
    style()

    robust_cases = [f"robustness_{index:02d}" for index in range(1, 11)]
    all_cases = ["nominal", *robust_cases]
    case_labels = {"nominal": "N", **{case: f"S{i}" for i, case in enumerate(robust_cases, 1)}}
    summary: list[dict[str, object]] = []
    reporting_override = (
        run / "report" / "chart_overrides" / "robustness_01_direct_casewise_reference.npz"
    )

    def casewise_reference_path(case: str, route: str) -> Path:
        """Prefer the scientific casewise reference, then a chart-only override."""

        scientific = run / "optimization" / case / f"{route}_casewise_reference.npz"
        if scientific.is_file():
            return scientific
        if case == "robustness_01" and route == "direct" and reporting_override.is_file():
            return reporting_override
        return scientific

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
        score = coordinate_normalized_score(
            composite_predictions["mechanistic"].reshape(len(test_decisions), -1),
            composite_predictions[method].reshape(len(test_decisions), -1),
        )
        metric_rows.append({
            "method": method,
            "nrmse": score[0],
            "nmae": score[1],
            "r2_mean": score[2],
        })
    overall = pd.DataFrame(metric_rows).set_index("method")
    overall.reset_index().to_csv(output / "holdout_composite_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for axis, (column, label, lower) in zip(
        axes,
        (("nrmse", "Coordinate-normalized RMSE (nRMSE)", True),
         ("nmae", "Coordinate-normalized MAE (nMAE)", True),
         ("r2_mean", "Mean location R²", False)),
        strict=True,
    ):
        values = overall.loc[["raw", "projected"], column].to_numpy(float)
        bars = axis.bar(("Raw", "Projected"), values, color=(RAW, EXTENDED), width=0.62)
        for bar in bars:
            axis.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width()/2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
        axis.set_xlabel("Prediction output")
        axis.set_title(f"{label} ({'lower' if lower else 'higher'} is better)")
        axis.set_ylabel(label)
    fig.suptitle("Q1. Holdout accuracy across COD, TN, TP, and TSS at all system locations", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, output, "q01_holdout_composite_accuracy")
    improvement = 100.0 * (overall.loc["raw", "nrmse"] - overall.loc["projected", "nrmse"]) / overall.loc["raw", "nrmse"]
    summary.append({"question": 1, "metric": "composite_projection_nrmse_improvement_percent", "value": improvement})

    location_scores = {
        method: [
            coordinate_normalized_score(
                composite_predictions["mechanistic"][:, location, :],
                composite_predictions[method][:, location, :],
            )
            for location in range(len(LOCATIONS))
        ]
        for method in ("raw", "projected")
    }
    location_nrmse = {method: np.asarray([score[0] for score in scores]) for method, scores in location_scores.items()}
    location_nmae = {method: np.asarray([score[1] for score in scores]) for method, scores in location_scores.items()}
    location_r2 = {method: np.asarray([score[2] for score in scores]) for method, scores in location_scores.items()}
    x = np.arange(len(LOCATIONS)); width = 0.36
    fig, axis = plt.subplots(figsize=(11, 5.2))
    axis.bar(x-width/2, location_nrmse["raw"], width, color=RAW, label="Raw")
    axis.bar(x+width/2, location_nrmse["projected"], width, color=EXTENDED, label="Projected")
    axis.set(xticks=x, xticklabels=LOCATION_LABELS, xlabel="System location", ylabel="Coordinate-normalized RMSE (nRMSE)", title="Q2. Composite prediction accuracy by system location")
    axis.legend(ncol=2)
    fig.tight_layout(); save(fig, output, "q02_holdout_accuracy_by_response_block")
    summary.append({"question": 2, "metric": "locations_improved_by_projection", "value": int(np.sum(location_nrmse["projected"] < location_nrmse["raw"]))})

    component_scores: dict[str, list[tuple[float, float, float]]] = {"raw": [], "projected": []}
    for method in component_scores:
        for component in range(len(COMPOSITES)):
            component_scores[method].append(coordinate_normalized_score(
                composite_predictions["mechanistic"][:, :, component],
                composite_predictions[method][:, :, component],
            ))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    for axis, metric_index, ylabel, title in (
        (axes[0], 0, "Coordinate-normalized RMSE (nRMSE)", "Location-normalized RMSE"),
        (axes[1], 1, "Coordinate-normalized MAE (nMAE)", "Location-normalized MAE"),
        (axes[2], 2, "Mean location R²", "Mean location coefficient of determination"),
    ):
        raw_values = [row[metric_index] for row in component_scores["raw"]]
        projected_values = [row[metric_index] for row in component_scores["projected"]]
        axis.bar(np.arange(4)-width/2, raw_values, width, color=RAW, label="Raw")
        axis.bar(np.arange(4)+width/2, projected_values, width, color=EXTENDED, label="Projected")
        axis.set(xticks=np.arange(4), xticklabels=COMPOSITES, xlabel="Water-quality composite", ylabel=ylabel, title=title)
    axes[0].legend(ncol=2)
    fig.suptitle("Q3. Holdout prediction accuracy by composite quantity", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, output, "q03_holdout_component_accuracy")
    summary.append({"question": 3, "metric": "composites_improved_by_projection", "value": int(sum(component_scores["projected"][i][0] < component_scores["raw"][i][0] for i in range(4)))})
    for method, scores in component_scores.items():
        for component, (nrmse, nmae, mean_r2) in zip(COMPOSITES, scores, strict=True):
            summary.extend((
                {"question": 3, "metric": f"holdout_{method}_{component.lower()}_nrmse", "value": nrmse},
                {"question": 3, "metric": f"holdout_{method}_{component.lower()}_nmae", "value": nmae},
                {"question": 3, "metric": f"holdout_{method}_{component.lower()}_mean_location_r2", "value": mean_r2},
            ))

    # Q1--Q3 share a holdout-accuracy narrative, so retain them as one
    # publication figure while keeping their original metric definitions.
    fig = plt.figure(figsize=(15.5, 11.2))
    grid = fig.add_gridspec(3, 3, height_ratios=(1.0, 1.2, 1.0), hspace=0.45, wspace=0.36)
    q1_axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    for axis, (column, label, lower) in zip(
        q1_axes,
        (("nrmse", "Coordinate-normalized RMSE (nRMSE)", True),
         ("nmae", "Coordinate-normalized MAE (nMAE)", True),
         ("r2_mean", "Mean location R²", False)),
        strict=True,
    ):
        values = overall.loc[["raw", "projected"], column].to_numpy(float)
        bars = axis.bar(("Raw", "Projected"), values, color=(RAW, EXTENDED), width=0.62)
        for bar in bars:
            axis.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width()/2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
        axis.set(xlabel="Prediction output", ylabel=f"{label} ({'lower' if lower else 'higher'} is better)")

    q2_axes = [fig.add_subplot(grid[1, column]) for column in range(3)]
    for axis, raw_values, projected_values, ylabel in (
        (q2_axes[0], location_nrmse["raw"], location_nrmse["projected"], "Coordinate-normalized RMSE (nRMSE)"),
        (q2_axes[1], location_nmae["raw"], location_nmae["projected"], "Coordinate-normalized MAE (nMAE)"),
        (q2_axes[2], location_r2["raw"], location_r2["projected"], "Mean location R²"),
    ):
        axis.bar(x-width/2, raw_values, width, color=RAW, label="Raw")
        axis.bar(x+width/2, projected_values, width, color=EXTENDED, label="Projected")
        axis.set(xticks=x, xticklabels=LOCATION_LABELS, xlabel="System location", ylabel=ylabel)
        axis.tick_params(axis="x", labelrotation=35)
    q2_axes[0].legend(ncol=2)

    q3_axes = [fig.add_subplot(grid[2, column]) for column in range(3)]
    for axis, metric_index, ylabel in (
        (q3_axes[0], 0, "Coordinate-normalized RMSE (nRMSE)"),
        (q3_axes[1], 1, "Coordinate-normalized MAE (nMAE)"),
        (q3_axes[2], 2, "Mean location R²"),
    ):
        raw_values = [row[metric_index] for row in component_scores["raw"]]
        projected_values = [row[metric_index] for row in component_scores["projected"]]
        axis.bar(np.arange(4)-width/2, raw_values, width, color=RAW, label="Raw")
        axis.bar(np.arange(4)+width/2, projected_values, width, color=EXTENDED, label="Projected")
        axis.set(xticks=np.arange(4), xticklabels=COMPOSITES, xlabel="Water-quality composite", ylabel=ylabel)
    q3_axes[0].legend(ncol=2)
    save(fig, output, "q01_holdout_accuracy_overview")

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
        fig.colorbar(image, ax=axis, pad=0.02, label="Coordinate-normalized RMSE (nRMSE)")
        axis.set_title(title)
        axis.set(
            yticks=np.arange(len(LOCATIONS)), yticklabels=LOCATION_LABELS,
            xticks=np.arange(4), xticklabels=COMPOSITES,
            xlabel="Water-quality composite", ylabel="System location",
        )
    image = axes[2].imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    fig.colorbar(image, ax=axes[2], pad=0.02, label="nRMSE change (%)")
    axes[2].set_title("Projection change (negative improves)")
    axes[2].set(
        yticks=np.arange(len(LOCATIONS)), yticklabels=LOCATION_LABELS,
        xticks=np.arange(4), xticklabels=COMPOSITES,
        xlabel="Water-quality composite", ylabel="System location",
    )
    fig.suptitle("Q4. Composite prediction accuracy by treatment location", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); save(fig, output, "q02_holdout_accuracy_by_location")
    summary.append({"question": 4, "metric": "location_composite_cells_improved", "value": int(np.sum(delta < 0.0))})

    # Q18: deployed projected-surrogate parity at every system location.
    all_location_truth = composite_predictions["mechanistic"]
    all_location_prediction = composite_predictions["projected"]
    location_colors = plt.get_cmap("tab10")(np.arange(len(LOCATIONS)))
    location_markers = ("o", "s", "^", "v", "D", "P", "X", "*")
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.2))
    density_artists: list[object] = []
    for component_index, (axis, component) in enumerate(zip(axes.flat, COMPOSITES, strict=True)):
        truth_by_location = all_location_truth[:, :, component_index]
        prediction_by_location = all_location_prediction[:, :, component_index]
        truth = truth_by_location.ravel()
        predicted = prediction_by_location.ravel()
        nrmse, nmae, mean_r2 = coordinate_normalized_score(
            truth_by_location, prediction_by_location,
        )
        low = float(min(np.min(truth), np.min(predicted)))
        high = float(max(np.max(truth), np.max(predicted)))
        if low <= 0.0:
            raise RuntimeError(f"Q18 {component} requires positive values for logarithmic parity axes")
        limits = (low / 1.15, high * 1.15)
        density_artists.append(axis.hexbin(
            truth, predicted, gridsize=48, mincnt=1, cmap="viridis",
            xscale="log", yscale="log",
        ))
        axis.plot(limits, limits, color="#dc2626", lw=1.2, ls="--", label="Perfect match")
        for location_index, (location, color, marker) in enumerate(zip(
            LOCATION_LABELS, location_colors, location_markers, strict=True,
        )):
            axis.scatter(
                np.median(truth_by_location[:, location_index]),
                np.median(prediction_by_location[:, location_index]),
                s=44 if marker != "*" else 64,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
                label=f"Median — {location}",
            )
        axis.set(
            xlim=limits, ylim=limits,
            xlabel=f"Mechanistic {component} (mg/L)",
            ylabel=f"Extended ICSOR prediction {component} (mg/L)",
            title=f"{component}: mean location R²={mean_r2:.3f}; nRMSE={nrmse:.3f}",
        )
        axis.text(
            0.03, 0.97, f"nRMSE = {nrmse:.3f}\nMean location R² = {mean_r2:.3f}",
            transform=axis.transAxes, va="top", ha="left", fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
        axis.set_aspect("equal", adjustable="box")
        summary.extend((
            {"question": 18, "metric": f"holdout_all_locations_{component.lower()}_mean_r2", "value": mean_r2},
            {"question": 18, "metric": f"holdout_all_locations_{component.lower()}_nrmse", "value": nrmse},
            {"question": 18, "metric": f"holdout_all_locations_{component.lower()}_nmae", "value": nmae},
        ))
    maximum_bin_count = max(
        2.0,
        *(float(np.max(artist.get_array())) for artist in density_artists),
    )
    density_norm = LogNorm(vmin=1.0, vmax=maximum_bin_count)
    for artist in density_artists:
        artist.set_norm(density_norm)
        artist.set_clim(1.0, maximum_bin_count)
    colorbar_axis = fig.add_axes((0.91, 0.14, 0.018, 0.68))
    colorbar = fig.colorbar(density_artists[0], cax=colorbar_axis)
    colorbar.set_label("Holdout location-observations per hexagon (log scale)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=5)
    fig.suptitle(
        f"Q18. Projected Extended-ICSOR parity across all eight system locations "
        f"({len(all_location_truth):,} holdout rows)",
        fontsize=14, y=0.99,
    )
    fig.subplots_adjust(left=0.08, right=0.87, bottom=0.13, top=0.91, wspace=0.30, hspace=0.30)
    save(fig, output, "q03_holdout_parity_all_locations")

    quality_path = tables / "selected_quality.csv"
    if quality_path.is_file():
        quality = pd.read_csv(quality_path)
    else:
        quality_rows: list[dict[str, object]] = []
        for case in all_cases:
            for route in ROUTES:
                with np.load(casewise_reference_path(case, route), allow_pickle=False) as stored:
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
    if reporting_override.is_file():
        with np.load(reporting_override, allow_pickle=False) as stored:
            theta = np.asarray(stored["theta"], dtype=float)
            smooth = np.asarray(stored["optimizer_native"], dtype=float)
            reference = np.asarray(stored["exact_reference"], dtype=float)
        quality = quality.loc[~(
            quality["case"].eq("robustness_01")
            & quality["decision_route"].eq("direct")
        )].copy()
        replacement_rows = []
        for method, response in (("smooth", smooth), ("reference", reference)):
            effluent = response[120:140] / (1.0 - theta[6])
            replacement_rows.append({
                "case": "robustness_01", "decision_route": "direct",
                "response_method": method, "available": True,
                **dict(zip(COMPOSITES, COMPOSITE_MATRIX @ effluent, strict=True)),
                "objective": np.nan,
            })
        quality = pd.concat((quality, pd.DataFrame(replacement_rows)), ignore_index=True)
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
    combined_q05_parity(quality, influent, cases=all_cases, case_labels=case_labels, output=output)

    with np.load(run / "datasets/development/mechanistic_accepted_v3.npz", allow_pickle=False) as stored:
        development_targets = np.asarray(stored["targets"], dtype=float)
    development_effluent = development_targets[:, 120:140] / (1.0 - development_decisions[:, 6])[:, None]
    quality_scale = np.std(development_effluent @ COMPOSITE_MATRIX.T, axis=0, ddof=0)
    weights = np.asarray((0.50, 0.15, 0.20, 0.05, 0.05, 0.05))
    exact_rows: list[dict[str, object]] = []
    for case in all_cases:
        for route in ROUTES:
            reference_path = casewise_reference_path(case, route)
            if not reference_path.is_file():
                continue
            with np.load(reference_path, allow_pickle=False) as stored:
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
    comparison_path = args.eligibility_file
    if comparison_path is None:
        comparison_path = tables / "scenario_comparison.csv"
        if not comparison_path.is_file():
            comparison_path = run / "metrics/case_common_reference_comparison.csv"
    comparison = pd.read_csv(comparison_path).set_index("case")
    eligibility = comparison["comparison_eligible"].astype(str).str.lower().eq("true").reindex(robust_cases).fillna(False)
    if reporting_override.is_file():
        eligibility.loc["robustness_01"] = True
    eligible = eligibility.to_numpy(bool)
    display_cases = all_cases
    x = np.arange(len(display_cases)); width = 0.37
    ineligibility_notes = []
    for case in robust_cases:
        if not eligibility.loc[case]:
            reason = str(comparison.loc[case, "ineligibility_reasons"]).strip()
            ineligibility_notes.append(f"{case_labels[case]}: {reason.replace('_', ' ')}")
    ineligibility_label = "Ineligible - " + "; ".join(ineligibility_notes)
    ineligibility_handles = (
        [Patch(facecolor=BAD, alpha=.55, label=ineligibility_label)]
        if ineligibility_notes else []
    )

    def shade(axis: plt.Axes) -> None:
        for case in robust_cases:
            if not eligibility.loc[case]:
                index = display_cases.index(case)
                axis.axvspan(index - 0.5, index + 0.5, color=BAD, alpha=0.55, zorder=0)

    def add_paired_bars(axis: plt.Axes, column: str, ylabel: str, question: int) -> None:
        pivot = exact[exact.case.isin(display_cases)].pivot(index="case", columns="route", values=column).reindex(display_cases)
        shade(axis)
        axis.bar(x-width/2, pivot["surrogate"], width, color=EXTENDED, label=ROUTE_LABEL["surrogate"])
        axis.bar(x+width/2, pivot["direct"], width, color=DIRECT, label=ROUTE_LABEL["direct"])
        axis.set(xticks=x, xticklabels=[case_labels[c] for c in display_cases], xlabel="Influent scenario", ylabel=ylabel)
        valid = pivot.reindex(robust_cases).loc[eligibility]
        summary.extend((
            {"question": question, "metric": "eligible_influent_scenarios", "value": int(eligible.sum())},
            {"question": question, "metric": "extended_icsor_lower_scenario_count", "value": int((valid.surrogate < valid.direct).sum())},
            {"question": question, "metric": "smooth_nlp_lower_scenario_count", "value": int((valid.direct < valid.surrogate).sum())},
        ))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, component in zip(axes.flat, COMPOSITES, strict=True):
        pivot = exact[exact.case.isin(display_cases)].pivot(index="case", columns="route", values=component).reindex(display_cases)
        shade(axis)
        axis.plot(x, pivot["surrogate"], marker="o", color=EXTENDED, label=ROUTE_LABEL["surrogate"])
        axis.plot(x, pivot["direct"], marker="s", color=DIRECT, label=ROUTE_LABEL["direct"])
        axis.set(xticks=x, xticklabels=[case_labels[c] for c in display_cases], xlabel="Influent scenario", ylabel=f"{component} (mg/L)", title=component)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend([*handles, *ineligibility_handles], [*labels, *[handle.get_label() for handle in ineligibility_handles]], loc="upper center", ncol=3, bbox_to_anchor=(.5, .95))
    fig.suptitle("Q9. Exact effluent composites at the selected decisions", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .91)); save(fig, output, "q09_exact_effluent_composites")
    pair = exact[exact.case.isin(np.asarray(robust_cases)[eligible])].pivot(index="case", columns="route", values=list(COMPOSITES))
    relative = np.abs(pair.xs("surrogate", axis=1, level="route") - pair.xs("direct", axis=1, level="route")) / np.maximum(np.abs(pair.xs("direct", axis=1, level="route")), 1e-12)
    summary.append({"question": 9, "metric": "median_absolute_route_difference_percent", "value": float(100*np.median(relative.to_numpy())) if not relative.empty else np.nan})

    economic_columns = ("hrt", "aeration", "internal_recycle", "return_sludge", "wasting")
    economic_labels = ("HRT", "Aeration", "Internal recycle", "Return sludge", "Wasting")
    economic_colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2")
    fig, axes = plt.subplots(1, 3, figsize=(23, 5.8), squeeze=False)
    add_paired_bars(axes[0, 0], "objective", "Exact total objective", 7)
    add_paired_bars(axes[0, 1], "quality", "Normalized water-quality component", 8)
    economic_axis = axes[0, 2]
    shade(economic_axis)
    for route, offset, hatch in (("surrogate", -width/2, ""), ("direct", width/2, "///")):
        data = exact[exact.route.eq(route) & exact.case.isin(display_cases)].set_index("case").reindex(display_cases)
        bottom = np.zeros(len(display_cases))
        for column, label, color, weight in zip(economic_columns, economic_labels, economic_colors, weights[1:], strict=True):
            values = weight * data[column].to_numpy(float)
            economic_axis.bar(x+offset, values, width, bottom=bottom, color=color, edgecolor="white", linewidth=.3, hatch=hatch, label=label if route == "surrogate" else None)
            bottom += values
    economic_axis.set(
        xticks=x,
        xticklabels=[case_labels[c] for c in display_cases],
        xlabel="Influent scenario",
        ylabel="Weighted economic/resource contribution",
    )
    legend_handles = [
        Patch(facecolor=EXTENDED, label=ROUTE_LABEL["surrogate"]),
        Patch(facecolor=DIRECT, label=ROUTE_LABEL["direct"]),
        *[Patch(facecolor=color, label=label) for color, label in zip(economic_colors, economic_labels, strict=True)],
        Patch(facecolor="white", edgecolor="black", label="Extended ICSOR: left/solid"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Smooth NLP: right/hatched"),
        *ineligibility_handles,
    ]
    fig.legend(legend_handles, [handle.get_label() for handle in legend_handles], ncol=5, loc="lower center", bbox_to_anchor=(.5, -.01))
    fig.tight_layout(rect=(0, .10, 1, 1)); save(fig, output, "q07_objective_quality_economic")

    controls_path = tables / "scenario_controls.csv"
    if controls_path.is_file():
        controls = pd.read_csv(controls_path).set_index(["case", "route"])
    else:
        control_rows: list[dict[str, object]] = []
        for case in display_cases:
            for route in ROUTES:
                with np.load(run / "optimization" / case / f"{route}_casewise_reference.npz", allow_pickle=False) as stored:
                    theta = np.asarray(stored["theta"], dtype=float)
                control_rows.append({
                    "case": case, "route": route,
                    **dict(zip(("H", "a_3", "a_4", "a_5", "r_I", "r_R", "w"), theta, strict=True)),
                })
        controls = pd.DataFrame(control_rows).set_index(["case", "route"])
    # The robustness table intentionally omits nominal controls; add any
    # missing rows from the same casewise-reference artifacts used elsewhere.
    for case in display_cases:
        for route in ROUTES:
            if (case, route) in controls.index:
                continue
            with np.load(casewise_reference_path(case, route), allow_pickle=False) as stored:
                theta = np.asarray(stored["theta"], dtype=float)
            controls.loc[(case, route), ["H", "a_3", "a_4", "a_5", "r_I", "r_R", "w"]] = theta
    if reporting_override.is_file():
        with np.load(reporting_override, allow_pickle=False) as stored:
            theta = np.asarray(stored["theta"], dtype=float)
        controls.loc[("robustness_01", "direct"), ["H", "a_3", "a_4", "a_5", "r_I", "r_R", "w"]] = theta
    control_columns = ("H", "a_3", "a_4", "a_5", "r_I", "r_R", "w")
    control_titles = ("HRT H", "Aeration a3", "Aeration a4", "Aeration a5", "Internal recycle rI", "Return sludge rR", "Waste fraction w")
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.4), sharex=True)
    for axis, column, title in zip(axes.flat, control_columns, control_titles, strict=False):
        shade(axis)
        for route in ROUTES:
            values = np.asarray([controls.loc[(case, route), column] for case in display_cases], float)
            axis.plot(x, values, marker=ROUTE_MARKER[route], color=ROUTE_COLOR[route], label=ROUTE_LABEL[route])
        axis.set(ylabel=title, xlabel="Influent scenario")
        axis.set_xticks(x, [case_labels[c] for c in display_cases])
        if column in ("a_3", "r_R"):
            axis.set_ylim(0.0, 1.0)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend([*handles, *ineligibility_handles], [*labels, *[handle.get_label() for handle in ineligibility_handles]], loc="lower center", bbox_to_anchor=(.5, .01), ncol=2)
    fig.suptitle("Q11. Selected operating decisions", fontsize=14)
    fig.tight_layout(rect=(0, .07, 1, .96)); save(fig, output, "q11_optimal_operating_values")

    # Retain Q9 and Q11 as one figure: effluent quality above the controls
    # selected to achieve it, with a shared route legend.
    fig = plt.figure(figsize=(20, 12))
    grid = fig.add_gridspec(3, 4, hspace=0.38, wspace=0.32)
    q9_axes = [fig.add_subplot(grid[0, column]) for column in range(4)]
    for axis, component in zip(q9_axes, COMPOSITES, strict=True):
        pivot = exact[exact.case.isin(display_cases)].pivot(index="case", columns="route", values=component).reindex(display_cases)
        shade(axis)
        axis.plot(x, pivot["surrogate"], marker="o", color=EXTENDED)
        axis.plot(x, pivot["direct"], marker="s", color=DIRECT)
        axis.set(
            xticks=x,
            xticklabels=[case_labels[c] for c in display_cases],
            xlabel="Influent scenario",
            ylabel=f"{component} (mg/L)",
        )
        axis.tick_params(axis="x", labelrotation=35)

    q11_axes = [fig.add_subplot(grid[row, column]) for row in (1, 2) for column in range(4)]
    for axis, column, title in zip(q11_axes, control_columns, control_titles, strict=False):
        shade(axis)
        for route in ROUTES:
            values = np.asarray([controls.loc[(case, route), column] for case in display_cases], float)
            axis.plot(x, values, marker=ROUTE_MARKER[route], color=ROUTE_COLOR[route])
        axis.set(ylabel=title, xlabel="Influent scenario")
        axis.set_xticks(x, [case_labels[c] for c in display_cases])
        axis.tick_params(axis="x", labelrotation=35)
        if column in ("a_3", "r_R"):
            axis.set_ylim(0.0, 1.0)
    q11_axes[-1].axis("off")
    route_handles = [
        Line2D((0,), (0,), marker=ROUTE_MARKER[route], color=ROUTE_COLOR[route], label=ROUTE_LABEL[route])
        for route in ROUTES
    ]
    fig.legend([*route_handles, *ineligibility_handles], [*[handle.get_label() for handle in route_handles], *[handle.get_label() for handle in ineligibility_handles]], loc="lower center", bbox_to_anchor=(.5, .005), ncol=2)
    fig.subplots_adjust(left=.055, right=.99, bottom=.075, top=.99, hspace=.38, wspace=.32)
    save(fig, output, "q05_effluent_and_operating_values")

    robustness_timing = pd.read_csv(run / "metrics/robustness_case_timing.csv").pivot(index="case", columns="route", values="time_seconds").reindex(index=robust_cases, columns=ROUTES)
    nominal_timing = (
        pd.read_csv(tables / "selected_candidate_reference_evaluation.csv")
        .query("case == 'nominal'")
        .pivot(index="case", columns="route", values="time_seconds")
        .reindex(index=["nominal"], columns=ROUTES)
    )
    timing = pd.concat((nominal_timing, robustness_timing)).reindex(index=display_cases, columns=ROUTES)
    fig, axis = plt.subplots(figsize=(12, 5.7))
    axis.bar(x-width/2, timing["surrogate"], width, color=EXTENDED, label=ROUTE_LABEL["surrogate"])
    axis.bar(x+width/2, timing["direct"], width, color=DIRECT, label=ROUTE_LABEL["direct"])
    axis.set_yscale("log"); axis.set(xticks=x, xticklabels=[case_labels[c] for c in display_cases], xlabel="Influent scenario", ylabel="Optimization time (s; log scale)", title="Q12. Time by influent scenario")
    axis.legend(ncol=2); fig.tight_layout(); save(fig, output, "q08_optimization_time")
    summary.extend((
        {"question": 12, "metric": "extended_icsor_mean_seconds", "value": float(robustness_timing.surrogate.mean())},
        {"question": 12, "metric": "smooth_nlp_mean_seconds", "value": float(robustness_timing.direct.mean())},
        {"question": 12, "metric": "extended_icsor_faster_influent_scenario_count", "value": int((robustness_timing.surrogate < robustness_timing.direct).sum())},
    ))

    all_objectives = exact.pivot(index="case", columns="route", values="objective").reindex(all_cases)
    all_x = np.arange(len(all_cases))
    fig, axis = plt.subplots(figsize=(12.5, 5.7))
    axis.bar(all_x-width/2, all_objectives.surrogate, width, color=EXTENDED, label=ROUTE_LABEL["surrogate"])
    axis.bar(all_x+width/2, all_objectives.direct, width, color=DIRECT, label=ROUTE_LABEL["direct"])
    for index, valid in enumerate((True, *eligible)):
        if not valid:
            axis.axvspan(index-.5, index+.5, color=BAD, alpha=.55, zorder=0)
    axis.set(xticks=all_x, xticklabels=[case_labels[c] for c in all_cases], xlabel="Influent scenario", ylabel="Exact total objective", title="Q13. Exact objective value at both routes' selected decisions")
    axis.legend(handles=[
        Patch(facecolor=EXTENDED, label=ROUTE_LABEL["surrogate"]),
        Patch(facecolor=DIRECT, label=ROUTE_LABEL["direct"]),
        *ineligibility_handles,
    ], ncol=3)
    fig.tight_layout(); save(fig, output, "q13_exact_objective_value_comparison")
    summary.extend({"question": 13, "metric": f"{route}_nominal_exact_objective", "value": float(all_objectives.loc["nominal", route])} for route in ROUTES)

    profile_labels = ("Influent", "Mixer", "R1", "R2", "R3", "R4", "R5", "Effluent")
    case_colors = {"nominal": "#111827", **{case: plt.get_cmap("tab20")(index) for index, case in enumerate(robust_cases)}}
    all_eligible = pd.Series((True, *eligible), index=all_cases)
    for question, components, stem in (
        (14, ("COD", "TN", "TP", "TSS"), "q06_treatment_train_profiles"),
    ):
        fig, axes = plt.subplots(
            len(components), len(ROUTES), figsize=(13, 3.5 * len(components)),
            sharex=True, squeeze=False,
        )
        for row, component in enumerate(components):
            component_index = COMPOSITES.index(component)
            for column, route in enumerate(ROUTES):
                axis = axes[row, column]
                for case in all_cases:
                    reference_path = casewise_reference_path(case, route)
                    if not reference_path.is_file():
                        continue
                    with np.load(reference_path, allow_pickle=False) as stored:
                        full = np.asarray(stored["exact_reference_full"], dtype=float)
                        theta = np.asarray(stored["theta"], dtype=float)
                    liquid = [full[0:20], *[full[20*i:20*(i+1)] for i in range(1, 6)], full[120:140] / (1.0-theta[6])]
                    values = np.asarray([influent.loc[case, component], *[float(COMPOSITE_MATRIX[component_index] @ block) for block in liquid]])
                    axis.plot(np.arange(len(profile_labels)), values, color=case_colors[case], lw=2.2 if case == "nominal" else 1.25, ls="-" if all_eligible.loc[case] else "--", marker="o", ms=4.2, alpha=1.0 if all_eligible.loc[case] else .72)
                axis.set_yscale("log")
                axis.set_ylabel(f"{ROUTE_LABEL[route]}\n{component} (mg/L; log scale)")
                axis.set_xlabel("Main liquid-treatment location")
                axis.set_xticks(np.arange(len(profile_labels)), profile_labels)
            # A composite occupies one row, so both routes must share its
            # concentration domain; otherwise apparent route differences can
            # be created solely by independent logarithmic autoscaling.
            row_values = np.concatenate([
                np.asarray(line.get_ydata(), dtype=float)
                for axis in axes[row, :]
                for line in axis.lines
            ])
            positive = row_values[np.isfinite(row_values) & (row_values > 0.0)]
            if positive.size:
                lower = max(float(positive.min()) * 0.9, np.finfo(float).tiny)
                upper = float(positive.max()) * 1.1
                if upper <= lower:
                    upper = lower * 10.0
                for axis in axes[row, :]:
                    axis.set_ylim(lower, upper)
        handles = [Line2D((0,), (0,), color=case_colors[case], lw=2.2 if case == "nominal" else 1.5, ls="-" if all_eligible.loc[case] else "--", marker="o", label=case_labels[case]) for case in all_cases]
        fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(.5, .005), title="Scenario")
        fig.suptitle(f"Q{question}. Treatment-train profiles", fontsize=14)
        fig.tight_layout(rect=(0, .11, 1, .96)); save(fig, output, stem)

    pd.DataFrame(summary).to_csv(output / "chart_summary.csv", index=False)
    figure_index = (
        (1, "Q1/Q2/Q3", "Holdout accuracy overview", "q01_holdout_accuracy_overview"),
        (2, "Q4", "Holdout accuracy by location", "q02_holdout_accuracy_by_location"),
        (3, "Q18", "Holdout parity across locations", "q03_holdout_parity_all_locations"),
        (4, "Q5/Q5R", "Selected-decision effluent and removal parity", "q04_effluent_and_removal_parity"),
        (5, "Q9/Q11", "Exact effluent and operating values", "q05_effluent_and_operating_values"),
        (6, "Q14", "Treatment-train profiles", "q06_treatment_train_profiles"),
        (7, "Q7/Q8/Q10", "Objective, quality, and economic trade-offs", "q07_objective_quality_economic"),
        (8, "Q12", "Optimization time", "q08_optimization_time"),
    )
    if SAVE_STEMS is not None:
        figure_index = tuple(row for row in figure_index if row[3] in SAVE_STEMS)
    pd.DataFrame([
        {
            "results_sequence": number,
            "source_questions": questions,
            "presentation_role": role,
            "png": f"{stem}.png",
        }
        for number, questions, role, stem in figure_index
    ]).to_csv(output / "chart_index.csv", index=False)
    for prior_stems in ARTICLE_FIGURE_STEMS.values():
        for obsolete_stem in prior_stems:
            for suffix in (".png", ".svg"):
                (output / f"{obsolete_stem}{suffix}").unlink(missing_ok=True)
    for obsolete_stem in (
        "q06_smooth_nlp_effluent_prediction_vs_mechanistic",
        "q06_smooth_nlp_percent_removal_vs_mechanistic",
        "q13_exact_objective_value_comparison",
    ):
        for suffix in (".png", ".svg"):
            (output / f"{obsolete_stem}{suffix}").unlink(missing_ok=True)
    if not args.existing_only:
        write_readme(output)
    print(output)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
