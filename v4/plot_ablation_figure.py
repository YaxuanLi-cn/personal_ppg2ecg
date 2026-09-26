"""Render the compact ablation chart used by the experiments section."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "v4" / "experiments_results.json"
OUTPUTS = (ROOT / "figs", ROOT / "v4" / "figs")


def main() -> None:
    with RESULTS.open() as handle:
        rows = json.load(handle)["ablation"]

    metric_keys = ["MAE", "RMSE", "FD", "MAE_hr_paired"]
    metric_labels = ["MAE", "RMSE", "FD", "HR-MAE"]
    baseline = rows[0]
    values = np.asarray(
        [[row[key] / baseline[key] * 100.0 for key in metric_keys] for row in rows]
    )

    # Keep the four groups close together so the bars occupy the figure rather
    # than leaving a large blank band between neighboring metrics.
    x = np.arange(len(metric_keys)) * 1.12
    width = 0.36
    colors = ["#9AA5B5", "#2F7FB6", "#D9252A"]
    labels = ["Basic baseline", "Patient-context-only", "PSPFlow"]

    fig, ax = plt.subplots(figsize=(13.6, 5.35), dpi=120)
    offsets = np.array([-width, 0.0, width])
    def format_percent(value: float) -> str:
        if abs(value - round(value)) < 1e-8:
            return f"{int(round(value))}%"
        return f"{value:.2f}".rstrip("0").rstrip(".") + "%"

    for row, color, label, offset in zip(values, colors, labels, offsets):
        ax.bar(x + offset, row, width=width, color=color, label=label, linewidth=0)
        for xpos, value in zip(x + offset, row):
            ax.text(
                xpos,
                value + 1.4,
                format_percent(value),
                ha="center",
                va="bottom",
                fontsize=17,
                color="#111111",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=21)
    ax.set_ylabel("Metric / basic baseline (%)", fontsize=20, labelpad=10)
    ax.set_ylim(0, 112)
    ax.set_xlim(x[0] - 0.62, x[-1] + 0.62)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=17, width=1.6, length=7)
    ax.tick_params(axis="x", width=1.6, length=7, pad=8)
    ax.grid(axis="y", color="#D5D9DE", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)

    # The legend stays outside the axes, but close enough to avoid a large
    # empty strip below the x-axis.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.105),
        ncol=3,
        frameon=False,
        fontsize=19,
        handlelength=1.35,
        handleheight=0.9,
        columnspacing=1.3,
        handletextpad=0.5,
        borderaxespad=0,
    )

    fig.subplots_adjust(left=0.095, right=0.995, top=0.995, bottom=0.205)
    for output_dir in OUTPUTS:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / "ablation_improvements.png",
            dpi=160,
            bbox_inches="tight",
            pad_inches=0.035,
        )
        fig.savefig(
            output_dir / "ablation_improvements.pdf",
            bbox_inches="tight",
            pad_inches=0.035,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
