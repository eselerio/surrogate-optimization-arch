"""Compare optimization routes on the common exact-mechanistic replay basis."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closed_loop.manuscript_v3 import DECISION_LOWER, DECISION_UPPER, clarifier_for_layers
from closed_loop.v3_reporting import CONTROL_NAMES
from closed_loop.v3_smooth import (
    DEFAULT_OBJECTIVE_WEIGHTS, engineering_quantities, fit_direct_assets,
    objective_components,
)


ROUTE_ORDER = ("surrogate", "shared_unit", "direct")
ROUTE_LABEL = {
    "surrogate": "System surrogate",
    "shared_unit": "Shared-unit surrogate",
    "direct": "Smooth mechanistic NLP",
}
ROUTE_COLOR = {
    "surrogate": "#d97706",
    "shared_unit": "#16a34a",
    "direct": "#2563eb",
}
ROUTE_MARKER = {"surrogate": "o", "shared_unit": "^", "direct": "s"}
PAIR_ORDER = (
    ("surrogate", "shared_unit", "S-U"),
    ("surrogate", "direct", "S-M"),
    ("shared_unit", "direct", "U-M"),
)


def setup() -> None:
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 220, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "grid.linewidth": .6})


def save(fig: plt.Figure, out: Path, name: str) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(out / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def _routes(root: Path) -> tuple[str, ...]:
    shared = (
        any(root.glob("*/shared_unit_casewise_reference.npz"))
        or any(root.glob("*/shared_unit_selected.npz"))
        or any(root.glob("*/shared_unit.json"))
    )
    return ROUTE_ORDER if shared else ("surrogate", "direct")


def _route_reference_artifact(case: Path, route: str) -> Path | None:
    """Return the current common-reference artifact, with legacy fallback."""

    for suffix in ("casewise_reference", "selected"):
        path = case / f"{route}_{suffix}.npz"
        if path.is_file():
            return path
    return None


def _reference(data: np.lib.npyio.NpzFile) -> np.ndarray | None:
    for key in ("exact_reference", "reference"):
        if key in data.files:
            return np.asarray(data[key], dtype=float)
    return None


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
    routes = _routes(root)

    with np.load(run / "datasets" / "effective_design.npz") as design:
        with np.load(run / "datasets" / "development" / "mechanistic_accepted_v3.npz") as generated:
            targets = generated["targets"]
        assets = fit_direct_assets(
            design["development_decisions"], design["development_influents"], targets,
            clarifier=clarifier_for_layers(10),
        )

    rows: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    component_names = (
        "quality", "hrt", "aeration", "internal_recycle", "return_sludge", "wasting",
    )
    engineering_names = ("effluent_cod", "effluent_tn", "effluent_tp", "effluent_tss", "srt_d")
    for case in sorted(path for path in root.iterdir() if path.is_dir()):
        values: dict[str, dict[str, object]] = {}
        reasons: dict[str, str] = {}
        for route in routes:
            path = _route_reference_artifact(case, route)
            if path is None:
                reasons[route] = "selected decision unavailable"
                continue
            with np.load(path, allow_pickle=False) as data:
                theta = np.asarray(data["theta"], dtype=float) if "theta" in data.files else None
                reference = _reference(data)
            if theta is None or reference is None or not (
                np.all(np.isfinite(theta)) and np.all(np.isfinite(reference))
            ):
                reasons[route] = "finite exact replay unavailable"
                continue
            components = objective_components(theta, reference, assets)
            values[route] = {
                "theta": theta,
                "components": components,
                "objective": float(DEFAULT_OBJECTIVE_WEIGHTS @ components),
                "engineering": engineering_quantities(theta, reference, assets),
            }

        row: dict[str, object] = {"case": case.name}
        for route in routes:
            available = route in values
            row[f"{route}_available"] = available
            row[f"{route}_ineligibility_reason"] = reasons.get(route)
            if not available:
                continue
            item = values[route]
            row[f"J_{route}_decision_exact"] = item["objective"]
            components = np.asarray(item["components"], dtype=float)
            row[f"{route}_economic_burden"] = float(
                DEFAULT_OBJECTIVE_WEIGHTS[1:] @ components[1:]
            )
            for index, name in enumerate(CONTROL_NAMES):
                row[f"{route}_{name}"] = np.asarray(item["theta"])[index]
            for index, name in enumerate(component_names):
                row[f"{route}_component_{name}"] = components[index]
            for name in engineering_names:
                row[f"{route}_{name}"] = item["engineering"][name]

        for left, right, symbol in PAIR_ORDER:
            if left not in routes or right not in routes:
                continue
            key = symbol.replace("-", "_")
            eligible = left in values and right in values
            row[f"comparison_eligible_{key}"] = eligible
            row[f"ineligibility_reason_{key}"] = (
                None if eligible else "; ".join(
                    f"{route}: {reasons.get(route, 'unavailable')}"
                    for route in (left, right) if route not in values
                )
            )
            if not eligible:
                continue
            delta = float(values[left]["objective"]) - float(values[right]["objective"])
            row[f"delta_J_{symbol.replace('-', '_minus_')}"] = delta
            denominator = float(values[right]["objective"])
            row[f"relative_penalty_percent_{symbol.replace('-', '_vs_')}"] = (
                100.0 * delta / denominator if denominator != 0.0 else np.nan
            )
            theta_left = np.asarray(values[left]["theta"], dtype=float)
            theta_right = np.asarray(values[right]["theta"], dtype=float)
            for index, name in enumerate(CONTROL_NAMES):
                row[f"delta_normalized_{key}_{name}"] = (
                    (theta_left[index] - theta_right[index])
                    / (DECISION_UPPER[index] - DECISION_LOWER[index])
                )

        # Preserve the established S-M columns consumed by older analyses.
        row["delta_J_surrogate_minus_direct"] = row.get("delta_J_S_minus_M", np.nan)
        row["relative_penalty_percent"] = row.get("relative_penalty_percent_S_vs_M", np.nan)
        for name in CONTROL_NAMES:
            row[f"delta_normalized_{name}"] = row.get(
                f"delta_normalized_S_M_{name}", np.nan,
            )
        if "surrogate_economic_burden" in row and "direct_economic_burden" in row:
            row["delta_economic_burden"] = (
                float(row["surrogate_economic_burden"])
                - float(row["direct_economic_burden"])
            )
        rows.append(row)
        for route, reason in reasons.items():
            excluded.append({"case": case.name, "route": route, "reason": reason})

    frame = pd.DataFrame(rows).sort_values("case").reset_index(drop=True)
    frame.to_csv(out / "exact_replay_route_comparison.csv", index=False)
    pd.DataFrame(excluded, columns=("case", "route", "reason")).to_csv(
        out / "excluded_cases.csv", index=False,
    )
    if frame.empty or not any(
        isinstance(frame.get(f"{route}_available"), pd.Series)
        and frame[f"{route}_available"].any() for route in routes
    ):
        raise RuntimeError("No selected decision has a finite exact replay")

    x = np.arange(len(frame))
    labels = frame["case"].str.replace("robustness_", "R", regex=False).str.replace("nominal", "Nominal", regex=False)
    pairs = [(left, right, symbol) for left, right, symbol in PAIR_ORDER if left in routes and right in routes]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), constrained_layout=True)
    for route in routes:
        column = f"J_{route}_decision_exact"
        if column in frame:
            axes[0].plot(x, frame[column], marker=ROUTE_MARKER[route], color=ROUTE_COLOR[route], label=ROUTE_LABEL[route])
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Exact-replay objective (lower is better)", title="A. Common-reference decision quality")
    axes[0].legend(frameon=False)
    offsets = np.linspace(-.25, .25, len(pairs)) if len(pairs) > 1 else [0.0]
    for offset, (_, _, symbol) in zip(offsets, pairs, strict=True):
        column = f"delta_J_{symbol.replace('-', '_minus_')}"
        axes[1].bar(x + offset, frame.get(column, np.nan), width=.7 / len(pairs), label=symbol)
    axes[1].axhline(0, color="#111827", lw=.8)
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Left-route objective minus right-route objective", title="B. Pairwise exact-objective differences")
    axes[1].legend(frameon=False)
    fig.suptitle("Optimization-route comparison on exact mechanistic replays", fontsize=13, fontweight="bold")
    save(fig, out, "01_exact_objective_comparison")

    fig, axes = plt.subplots(len(pairs), 1, figsize=(10.5, 4.2 * len(pairs)), constrained_layout=True, squeeze=False)
    for ax, (left, right, symbol) in zip(axes.flat, pairs, strict=True):
        columns = []
        key = symbol.replace("-", "_")
        for name in CONTROL_NAMES:
            value = frame.get(f"delta_normalized_{key}_{name}")
            columns.append(
                pd.to_numeric(value, errors="coerce").to_numpy(float)
                if isinstance(value, pd.Series) else np.full(len(frame), np.nan)
            )
        delta = np.column_stack(columns)
        vmax = max(.05, float(np.nanmax(np.abs(delta))) if np.isfinite(delta).any() else .05)
        mesh = ax.imshow(delta, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set(xticks=np.arange(len(CONTROL_NAMES)), xticklabels=CONTROL_NAMES,
               yticks=x, yticklabels=labels, title=f"{symbol}: {ROUTE_LABEL[left]} minus {ROUTE_LABEL[right]}")
        fig.colorbar(mesh, ax=ax, label="Fraction of allowed control range")
    save(fig, out, "02_decision_difference_heatmap")

    outcomes = (("effluent_cod", "Effluent COD"), ("effluent_tn", "Effluent TN"),
                ("effluent_tp", "Effluent TP"), ("effluent_tss", "Effluent TSS"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (field, label) in zip(axes.flat, outcomes, strict=True):
        for route in routes:
            column = f"{route}_{field}"
            if column in frame:
                ax.plot(x, frame[column], marker=ROUTE_MARKER[route], color=ROUTE_COLOR[route], label=ROUTE_LABEL[route])
        ax.set(xticks=x, xticklabels=labels, title=label, ylabel="Exact mechanistic replay")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Process outcomes resulting from selected decisions", fontsize=13, fontweight="bold")
    save(fig, out, "03_exact_process_outcomes")

    component_keys = ("quality", "hrt", "aeration", "internal_recycle", "return_sludge", "wasting")
    component_labels = ("Water quality", "HRT / capacity", "Aeration", "Internal recycle", "Return sludge", "Wasting")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
    for ax, name, key, weight in zip(axes.flat, component_labels, component_keys, DEFAULT_OBJECTIVE_WEIGHTS, strict=True):
        for route in routes:
            column = f"{route}_component_{key}"
            if column in frame:
                ax.plot(x, frame[column], marker=ROUTE_MARKER[route], color=ROUTE_COLOR[route], label=ROUTE_LABEL[route])
        ax.set(xticks=x, xticklabels=labels, title=f"{name} (weight {weight:.2f})", ylabel="Normalized component")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("All objective components on exact mechanistic replays", fontsize=13, fontweight="bold")
    save(fig, out, "04_economic_operating_burden")

    # Keep the historical filename as a second rendering of the component panel.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
    for ax, name, key in zip(axes.flat, component_labels, component_keys, strict=True):
        for route in routes:
            column = f"{route}_component_{key}"
            if column in frame:
                ax.plot(x, frame[column], marker=ROUTE_MARKER[route], color=ROUTE_COLOR[route], label=ROUTE_LABEL[route])
        ax.set(xticks=x, xticklabels=labels, title=name)
    axes[0, 0].legend(frameon=False, fontsize=8)
    save(fig, out, "05_all_objective_components")

    weighted, tick_labels = [], []
    for left, right, symbol in pairs:
        for index, key in enumerate(component_keys):
            left_col, right_col = f"{left}_component_{key}", f"{right}_component_{key}"
            if left_col in frame and right_col in frame:
                values = DEFAULT_OBJECTIVE_WEIGHTS[index] * (frame[left_col] - frame[right_col])
                finite = values[np.isfinite(values)].to_numpy(float)
                if finite.size:
                    weighted.append(finite)
                    tick_labels.append(f"{symbol}\n{component_labels[index]}")
    fig, ax = plt.subplots(figsize=(max(10.5, len(weighted) * .65), 5.2), constrained_layout=True)
    if weighted:
        ax.boxplot(weighted, tick_labels=tick_labels, showfliers=False)
    ax.axhline(0, color="#111827", lw=.8)
    ax.set(ylabel="Weighted left-route minus right-route difference", title="Objective-component contributions to pairwise gaps")
    ax.tick_params(axis="x", rotation=30)
    save(fig, out, "06_weighted_component_differences")

    summary: dict[str, object] = {"cases": len(frame), "excluded_route_cases": len(excluded)}
    for _, _, symbol in pairs:
        key = symbol.replace("-", "_")
        eligibility = frame.get(f"comparison_eligible_{key}", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
        delta_value = frame.get(f"delta_J_{symbol.replace('-', '_minus_')}")
        delta = pd.to_numeric(delta_value, errors="coerce") if isinstance(delta_value, pd.Series) else pd.Series(np.nan, index=frame.index)
        summary[f"comparable_cases_{key}"] = int(eligibility.sum())
        summary[f"left_route_wins_{key}"] = int((delta[eligibility] < 0).sum())
        summary[f"right_route_wins_{key}"] = int((delta[eligibility] > 0).sum())
    if "S_M" in {symbol.replace("-", "_") for _, _, symbol in pairs}:
        sm_eligible = frame["comparison_eligible_S_M"].fillna(False).astype(bool)
        sm_delta = pd.to_numeric(frame["delta_J_S_minus_M"], errors="coerce")
        penalty = pd.to_numeric(frame["relative_penalty_percent"], errors="coerce")
        economic_value = frame.get("delta_economic_burden")
        economic_delta = (
            pd.to_numeric(economic_value, errors="coerce")
            if isinstance(economic_value, pd.Series)
            else pd.Series(np.nan, index=frame.index)
        )
        summary.update({
            "comparable_cases": int(sm_eligible.sum()),
            "surrogate_wins": int((sm_delta[sm_eligible] < 0).sum()),
            "smooth_nlp_wins": int((sm_delta[sm_eligible] > 0).sum()),
            "median_surrogate_penalty_percent": float(penalty[sm_eligible].median()),
            "mean_surrogate_penalty_percent": float(penalty[sm_eligible].mean()),
            "surrogate_lower_economic_burden_cases": int(
                (economic_delta[sm_eligible] < 0).sum()
            ),
            "median_surrogate_economic_burden_difference": float(
                economic_delta[sm_eligible].median()
            ),
            "excluded_cases": int((~sm_eligible).sum()),
        })
    pd.DataFrame([summary]).to_csv(out / "comparison_summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
