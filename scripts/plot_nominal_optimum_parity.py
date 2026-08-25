"""Plot nominal reduced-surrogate and full mechanistic replay parity."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "results" / "article_v3" / "article_full_50000_reduced_001"
SHARED_RESPONSE_COUNT = 160
REDUCED_RESPONSE_COUNT = 161
CLARIFIER_VOLUME_M3 = 6_000.0


def reduced_response(response: np.ndarray) -> np.ndarray:
    """Map a full layer-resolved response to the 161-coordinate response."""

    values = np.asarray(response, dtype=float).reshape(-1)
    if values.size == REDUCED_RESPONSE_COUNT:
        return values
    layer_count = values.size - SHARED_RESPONSE_COUNT
    if layer_count < 3:
        raise ValueError(
            "expected a 161-coordinate reduced response or a full response "
            "with at least three Clarifier layers"
        )
    inventory = (
        CLARIFIER_VOLUME_M3 / layer_count
        * float(np.sum(values[SHARED_RESPONSE_COUNT:]))
    )
    return np.concatenate((values[:SHARED_RESPONSE_COUNT], [inventory]))


def parity_axis(ax: plt.Axes, truth: np.ndarray, prediction: np.ndarray, groups: list[tuple[str, slice, str]]) -> None:
    for label, index, color in groups:
        ax.scatter(
            truth[index], prediction[index], s=30, alpha=0.8, label=label,
            color=color, edgecolors="white", linewidths=0.35,
        )
    lower = min(float(np.min(prediction)), float(np.min(truth)), -30.0)
    upper = max(float(np.max(prediction)), float(np.max(truth))) * 1.25
    line = np.linspace(lower, upper, 300)
    ax.plot(line, line, "k--", lw=1.1, label="identity")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Exact BDF state / flow coordinate")
    ax.set_ylabel("Raw ICSOR prediction")
    ax.grid(True, which="both", alpha=0.22)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, nargs="?", default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = (args.output or run / "report" / "figures").resolve()
    with np.load(
        run / "optimization" / "nominal" / "surrogate_casewise_reference.npz",
        allow_pickle=False,
    ) as stored:
        exact_response = np.asarray(stored["exact_reference"], dtype=float)
        predicted = np.asarray(stored["raw"], dtype=float)
    with np.load(
        run / "optimization" / "nominal" / "direct_casewise_reference.npz",
        allow_pickle=False,
    ) as stored:
        direct_reduced = np.asarray(stored["optimizer_native"], dtype=float)
        direct_exact = np.asarray(stored["exact_reference"], dtype=float)
        direct_full = np.asarray(stored["optimizer_native_full"], dtype=float)
        direct_exact_full = np.asarray(stored["exact_reference_full"], dtype=float)
    shared_path = (
        run / "optimization" / "nominal" / "shared_unit_casewise_reference.npz"
    )
    shared_prediction: np.ndarray | None = None
    shared_exact: np.ndarray | None = None
    if shared_path.is_file():
        with np.load(shared_path, allow_pickle=False) as stored:
            reference_key = next(
                (key for key in ("reference_response", "exact_reference", "reference") if key in stored.files),
                None,
            )
            if reference_key is not None:
                shared_exact = reduced_response(stored[reference_key])
            for key in (
                "projected_response", "optimizer_native", "projected",
                "raw_response", "raw",
            ):
                if key in stored.files:
                    shared_prediction = reduced_response(stored[key])
                    break
    with np.load(run / "models" / "ridge_surrogate.npz", allow_pickle=False) as surrogate:
        scale = np.asarray(surrogate["response_scale"], dtype=float)
    exact = reduced_response(exact_response)
    if predicted.shape != (REDUCED_RESPONSE_COUNT,) or scale.shape != predicted.shape:
        raise ValueError("the nominal surrogate artifacts are not 161-coordinate responses")
    if exact.shape != predicted.shape or direct_reduced.shape != direct_exact.shape:
        raise ValueError("the nominal response arrays do not share the reduced schema")

    # Reduced order: mixer (20), five CSTRs (5 x 20), overflow and underflow
    # flow vectors (2 x 20), and one Clarifier-solids inventory.
    reactor_groups = [
        ("Mixer", slice(0, 20), "#4c78a8"),
        ("CSTR 1", slice(20, 40), "#f58518"),
        ("CSTR 2", slice(40, 60), "#e45756"),
        ("CSTR 3", slice(60, 80), "#72b7b2"),
        ("CSTR 4", slice(80, 100), "#54a24b"),
        ("CSTR 5", slice(100, 120), "#b279a2"),
    ]
    outlet_groups = [
        ("Overflow", slice(120, 140), "#4c78a8"),
        ("Underflow", slice(140, 160), "#f58518"),
    ]
    blocks = [("Mixer", slice(0, 20))] + [
        (f"CSTR {i}", slice(20 * i, 20 * (i + 1))) for i in range(1, 6)
    ] + [
        ("Overflow", slice(120, 140)),
        ("Underflow", slice(140, 160)),
        ("Inventory", slice(160, 161)),
    ]
    block_rmse = [
        float(np.sqrt(np.mean(((predicted[index] - exact[index]) / scale[index]) ** 2)))
        for _, index in blocks
    ]
    nlp_block_rmse = [
        float(np.sqrt(np.mean(((direct_reduced[index] - direct_exact[index]) / scale[index]) ** 2)))
        for _, index in blocks
    ]

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.1, 0.9))
    ax_reactor = fig.add_subplot(grid[0, 0])
    ax_outlet = fig.add_subplot(grid[0, 1])
    ax_inventory = fig.add_subplot(grid[1, 0])
    ax_error = fig.add_subplot(grid[1, 1])

    parity_axis(ax_reactor, exact, predicted, reactor_groups)
    ax_reactor.set_title("A. Mixer and reactor coordinates")
    ax_reactor.legend(loc="upper left", fontsize=8, ncol=2)

    parity_axis(ax_outlet, exact, predicted, outlet_groups)
    ax_outlet.set_title("B. Clarifier overflow and underflow coordinates")
    ax_outlet.legend(loc="upper left", fontsize=8)

    inventory = [exact[-1], predicted[-1]]
    bars = ax_inventory.bar(
        ["Exact replay", "Raw surrogate"], inventory,
        color=["#1f77b4", "#d62728"],
    )
    for bar, value in zip(bars, inventory, strict=True):
        ax_inventory.annotate(
            f"{value:,.0f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
        )
    ax_inventory.set_ylabel("Clarifier solids inventory (g TSS)")
    ax_inventory.set_title("C. Aggregate Clarifier inventory")

    colors = ["#4c78a8"] + ["#72b7b2"] * 5 + ["#f58518", "#e45756", "#b279a2"]
    ax_error.bar([name for name, _ in blocks], block_rmse, color=colors)
    ax_error.axhline(1.0, color="black", lw=1.1, ls="--", label="one development SD")
    ax_error.set_ylim(0, max(1.05, max(block_rmse) * 1.2))
    ax_error.set_ylabel("RMS error / development scale")
    ax_error.set_title("D. Error at the nominal optimum")
    ax_error.tick_params(axis="x", rotation=38)
    ax_error.grid(True, axis="y", alpha=0.25)
    ax_error.legend(fontsize=8)

    fig.suptitle(
        "Nominal optimum: raw 161-coordinate surrogate versus exact mechanistic replay\n"
        "The surrogate retains aggregate Clarifier inventory, not a layer profile.",
        fontsize=13,
        fontweight="bold",
    )
    output.mkdir(parents=True, exist_ok=True)
    surrogate_output = output / "nominal_optimum_reduced_surrogate_vs_bdf.png"
    fig.savefig(surrogate_output, dpi=220, bbox_inches="tight")
    fig.savefig(surrogate_output.with_suffix(".pdf"), bbox_inches="tight")

    # The companion figure separates the constrained NLP state from the raw
    # surrogate.  It is a smooth-NLP versus nonsmoothed-BDF consistency check,
    # not a test of raw surrogate prediction accuracy.
    fig2, (ax_parity, ax_nlp_error) = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    if direct_full.shape != direct_exact_full.shape or direct_full.size <= SHARED_RESPONSE_COUNT:
        raise ValueError("direct and exact responses must share a full layer-resolved schema")
    all_groups = reactor_groups + outlet_groups + [
        ("Clarifier layers", slice(SHARED_RESPONSE_COUNT, None), "#b279a2")
    ]
    parity_axis(ax_parity, direct_exact_full, direct_full, all_groups)
    ax_parity.set_title("A. Smooth NLP physical state versus exact BDF replay")
    ax_parity.set_ylabel("Smooth NLP physical state / flow coordinate")
    ax_parity.legend(loc="upper left", fontsize=8, ncol=2)

    location = np.arange(len(blocks))
    ax_nlp_error.bar(location, nlp_block_rmse, color="#54a24b")
    ax_nlp_error.set_yscale("log")
    ax_nlp_error.set_xticks(location, [name for name, _ in blocks], rotation=38)
    ax_nlp_error.set_ylabel("Reduced-response RMS / surrogate development scale")
    ax_nlp_error.set_title("B. Smooth-NLP-to-BDF consistency by shared block")
    ax_nlp_error.grid(True, axis="y", which="both", alpha=0.25)
    fig2.suptitle(
        "Nominal optimum: layer-resolved mechanistic consistency check\n"
        "Clarifier layers are compared only between the smooth and exact models.",
        fontsize=13,
        fontweight="bold",
    )
    direct_output = output / "nominal_optimum_smooth_nlp_vs_bdf.png"
    fig2.savefig(direct_output, dpi=220, bbox_inches="tight")
    fig2.savefig(direct_output.with_suffix(".pdf"), bbox_inches="tight")

    shared_outputs: list[Path] = []
    if shared_prediction is not None and shared_exact is not None:
        if shared_prediction.shape != shared_exact.shape or shared_prediction.shape != scale.shape:
            raise ValueError("shared-unit nominal responses do not share the reduced schema")
        shared_block_rmse = [
            float(np.sqrt(np.mean(
                ((shared_prediction[index] - shared_exact[index]) / scale[index]) ** 2
            )))
            for _, index in blocks
        ]
        fig3, (ax_shared, ax_shared_error) = plt.subplots(
            1, 2, figsize=(14, 5.8), constrained_layout=True,
        )
        parity_axis(
            ax_shared, shared_exact, shared_prediction,
            reactor_groups + outlet_groups + [("Inventory", slice(160, 161), "#b279a2")],
        )
        ax_shared.set_title("A. Shared-unit surrogate versus exact BDF replay")
        ax_shared.set_ylabel("Shared-unit prediction")
        ax_shared.legend(loc="upper left", fontsize=8, ncol=2)
        location = np.arange(len(blocks))
        ax_shared_error.bar(location, shared_block_rmse, color="#16a34a")
        ax_shared_error.axhline(1.0, color="black", lw=1.1, ls="--")
        ax_shared_error.set_xticks(location, [name for name, _ in blocks], rotation=38)
        ax_shared_error.set_ylabel("RMS error / system-surrogate development scale")
        ax_shared_error.set_title("B. Shared-unit replay error by response block")
        ax_shared_error.grid(True, axis="y", alpha=0.25)
        fig3.suptitle(
            "Nominal optimum: shared reactor/Clarifier surrogate versus exact mechanistic replay",
            fontsize=13, fontweight="bold",
        )
        shared_output = output / "nominal_optimum_shared_unit_vs_bdf.png"
        fig3.savefig(shared_output, dpi=220, bbox_inches="tight")
        fig3.savefig(shared_output.with_suffix(".pdf"), bbox_inches="tight")
        shared_outputs = [shared_output, shared_output.with_suffix(".pdf")]
        plt.close(fig3)
    plt.close(fig)
    plt.close(fig2)
    print(surrogate_output)
    print(surrogate_output.with_suffix(".pdf"))
    print(direct_output)
    print(direct_output.with_suffix(".pdf"))
    for path in shared_outputs:
        print(path)


if __name__ == "__main__":
    main()
