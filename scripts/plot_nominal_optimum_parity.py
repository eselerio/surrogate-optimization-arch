"""Plot raw ICSOR predictions against the exact BDF replay at the nominal optimum."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "closed_loop" / "verify_nlp_2000_003"
DATA = RUN / "optimization" / "cache" / "nominal" / "exact_combined.npz"
SURROGATE = RUN / "models" / "development_surrogate.npz"
OUTPUT = RUN / "figures" / "nominal_optimum_icsor_vs_bdf.png"
OUTPUT_PDF = OUTPUT.with_suffix(".pdf")
OUTPUT_NLP = RUN / "figures" / "nominal_optimum_nlp_vs_bdf.png"
OUTPUT_NLP_PDF = OUTPUT_NLP.with_suffix(".pdf")


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
    stored = np.load(DATA)
    surrogate = np.load(SURROGATE)
    exact = stored["target"]
    predicted = stored["raw_surrogate_prediction"]
    nlp_state = stored["selected_complete_state"]
    scale = surrogate["response_scale"]

    # State order: mixer (20), five CSTRs (5 x 20), overflow and underflow
    # flow vectors (2 x 20), and ten clarifier-layer TSS values.
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
        ("Layers", slice(160, 170)),
    ]
    block_rmse = [
        float(np.sqrt(np.mean(((predicted[index] - exact[index]) / scale[index]) ** 2)))
        for _, index in blocks
    ]
    nlp_block_rmse = [
        float(np.sqrt(np.mean(((nlp_state[index] - exact[index]) / scale[index]) ** 2)))
        for _, index in blocks
    ]

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.1, 0.9))
    ax_reactor = fig.add_subplot(grid[0, 0])
    ax_outlet = fig.add_subplot(grid[0, 1])
    ax_layers = fig.add_subplot(grid[1, 0])
    ax_error = fig.add_subplot(grid[1, 1])

    parity_axis(ax_reactor, exact, predicted, reactor_groups)
    ax_reactor.set_title("A. Mixer and reactor coordinates")
    ax_reactor.legend(loc="upper left", fontsize=8, ncol=2)

    parity_axis(ax_outlet, exact, predicted, outlet_groups)
    ax_outlet.set_title("B. Clarifier overflow and underflow coordinates")
    ax_outlet.legend(loc="upper left", fontsize=8)

    layers = np.arange(1, 11)
    ax_layers.plot(layers, exact[160:170], "o-", lw=2, color="#1f77b4", label="Exact BDF replay")
    ax_layers.plot(layers, predicted[160:170], "s--", lw=2, color="#d62728", label="Raw ICSOR prediction")
    ax_layers.set_yscale("log")
    ax_layers.set_xticks(layers)
    ax_layers.set_xlabel("Clarifier layer")
    ax_layers.set_ylabel("TSS (g TSS m$^{-3}$)")
    ax_layers.set_title("C. Clarifier TSS profile")
    ax_layers.grid(True, which="both", alpha=0.25)
    ax_layers.legend(fontsize=9)

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
        "Nominal optimum: raw ICSOR complete-state prediction versus exact mechanistic BDF replay\n"
        "The raw surrogate is shown; the constrained NLP reconstructs its own physical state separately.",
        fontsize=13,
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")

    # The companion figure separates the constrained NLP state from the raw
    # surrogate.  It is a smooth-NLP versus nonsmoothed-BDF consistency check,
    # not a test of raw surrogate prediction accuracy.
    fig2, (ax_parity, ax_nlp_error) = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    all_groups = reactor_groups + outlet_groups + [("Clarifier layers", slice(160, 170), "#b279a2")]
    parity_axis(ax_parity, exact, nlp_state, all_groups)
    ax_parity.set_title("A. Smooth NLP physical state versus exact BDF replay")
    ax_parity.set_ylabel("Smooth NLP physical state / flow coordinate")
    ax_parity.legend(loc="upper left", fontsize=8, ncol=2)

    location = np.arange(len(blocks))
    ax_nlp_error.bar(location, nlp_block_rmse, color="#54a24b")
    ax_nlp_error.set_yscale("log")
    ax_nlp_error.set_xticks(location, [name for name, _ in blocks], rotation=38)
    ax_nlp_error.set_ylabel("RMS error / development scale (log scale)")
    ax_nlp_error.set_title("B. Smooth-NLP-to-BDF consistency by state block")
    ax_nlp_error.grid(True, axis="y", which="both", alpha=0.25)
    fig2.suptitle(
        "Nominal optimum: mechanistic consistency check of the smooth NLP versus exact BDF replay\n"
        "This figure does not evaluate raw ICSOR surrogate prediction accuracy.",
        fontsize=13,
        fontweight="bold",
    )
    fig2.savefig(OUTPUT_NLP, dpi=220, bbox_inches="tight")
    fig2.savefig(OUTPUT_NLP_PDF, bbox_inches="tight")
    print(OUTPUT)
    print(OUTPUT_PDF)
    print(OUTPUT_NLP)
    print(OUTPUT_NLP_PDF)


if __name__ == "__main__":
    main()
