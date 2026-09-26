"""Render the benchmark chart with compact two-decimal value labels."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "v4" / "experiments_results.json"
OUTPUTS = (ROOT / "figs", ROOT / "v4" / "figs")


def main() -> None:
    with RESULTS.open() as handle:
        rows = json.load(handle)["main_comparison"]

    metrics = [
        ("MAE", "MAE"),
        ("RMSE", "RMSE"),
        ("FD", "FD"),
        ("MAE_hr_paired", "HR-MAE"),
    ]
    palette = {
        "PSPFlow": "#D9252A",
        "CardioFlow": "#2F7FB6",
        "CLEP-GAN": "#8055B8",
        "CardioGAN": "#FF7F0E",
        "RDDM": "#2CA02C",
        "UniCardio": "#CB6BB2",
    }

    # Match the reference composition: four side-by-side horizontal panels.

    def format_value(value: float) -> str:
        return f"{value:.2f}"

    def format_tick(value: float, _position: int) -> str:
        if value <= 0:
            return ""
        if abs(value - round(value)) < 1e-8:
            return str(int(round(value)))
        return f"{value:.2f}".rstrip("0").rstrip(".")

    # Four simple major tick values keep the panels readable without a dense
    # run of minor labels. FD uses the requested 500--2000 scale.
    tick_values = {
        "MAE": [0.4, 0.6, 0.8, 1.0],
        "RMSE": [0.5, 1.0, 1.5, 2.0],
        "FD": [500.0, 1000.0, 1500.0, 2000.0],
        "MAE_hr_paired": [10.0, 20.0, 30.0, 40.0],
    }

    fig, axes = plt.subplots(1, 4, figsize=(18.0, 5.25), dpi=120)
    axis_maxima = {
        "MAE": 1.6,
        "RMSE": 2.1,
        "FD": 2100.0,
        "MAE_hr_paired": 50.0,
    }
    for panel_index, (ax, (key, title)) in enumerate(zip(axes, metrics)):
        ordered = sorted(rows, key=lambda row: row[key])
        names = [row["method"] for row in ordered]
        values = [row[key] for row in ordered]
        positions = list(range(len(names)))
        ax.barh(
            positions,
            values,
            height=0.78,
            color=[palette[name] for name in names],
            linewidth=0,
        )
        for position, value in zip(positions, values):
            # Point-offset annotation gives every bar the same visible gap.
            ax.annotate(
                format_value(value),
                xy=(value, position),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=12,
                clip_on=False,
            )
        ax.set_title(title, fontsize=17, pad=10)
        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontsize=11)
        ax.invert_yaxis()
        ax.set_xscale("linear")
        ax.xaxis.set_major_locator(FixedLocator(tick_values[key]))
        ax.xaxis.set_major_formatter(FuncFormatter(format_tick))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis="x", color="#D5D9DE", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0.0, axis_maxima[key])
        ax.tick_params(axis="x", labelsize=10, length=4, width=1.0, pad=4)
        ax.tick_params(axis="y", length=3, width=1.0, pad=4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        if panel_index == 0:
            ax.set_ylabel("Method", fontsize=14, labelpad=9)

    fig.suptitle("Lower is better; values are reported test metrics", fontsize=18, y=0.985)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.79, bottom=0.17, wspace=0.56)
    for output_dir in OUTPUTS:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / "benchmark_metrics.png",
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.04,
        )
        fig.savefig(
            output_dir / "benchmark_metrics.pdf",
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
