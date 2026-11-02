"""Fig. 6 — replica leave-one-sheet-out structure (frozen table; no retraining)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figure_style import WIDTH_DOUBLE_IN, save_manuscript_figure

SCENARIO_ORDER = ("normalOperation", "extremeEvents", "accidentalEvents")
SCENARIO_LABEL = {
    "normalOperation": "Normal",
    "extremeEvents": "Extreme",
    "accidentalEvents": "Accidental",
}
SCENARIO_COLOR = {
    "normalOperation": "#2563eb",
    "extremeEvents": "#ea580c",
    "accidentalEvents": "#7c3aed",
}
SCENARIO_MARKER = {
    "normalOperation": "o",
    "extremeEvents": "s",
    "accidentalEvents": "D",
}
CONSTANT_COLOR = "#9ca3af"

_SKILL_SCRIPTS = Path(__file__).resolve().parents[2] / ".cursor" / "skills" / "nature-figure" / "scripts"


def _sheet_label(group: str) -> str:
    name = str(group).split(".xlsx_", 1)[-1]
    return name.replace("_datos", "")


def _panel_letter(ax, letter: str) -> None:
    ax.text(
        0.0,
        1.02,
        f"({letter})",
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=9,
        va="bottom",
        ha="left",
    )


def plot_replica_fig6(logo_df: pd.DataFrame, out_png: Path) -> None:
    df = logo_df.copy()
    df["constant"] = df["test_dose_n_unique"] <= 1
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH_DOUBLE_IN, 3.55),
        constrained_layout=True,
    )
    ax_a, ax_b = axes

    for scen in SCENARIO_ORDER:
        sub = df[df["scenario"] == scen]
        const = sub[sub["constant"]]
        multi = sub[~sub["constant"]]
        if len(const):
            jitter = rng.uniform(-0.10, 0.10, len(const))
            ax_a.scatter(
                const["test_dose_n_unique"] + jitter,
                const["mae"],
                s=16,
                c=CONSTANT_COLOR,
                marker=SCENARIO_MARKER[scen],
                alpha=0.55,
                linewidths=0,
                zorder=2,
            )
        if len(multi):
            x_off = {"normalOperation": 0.0, "extremeEvents": -0.10, "accidentalEvents": 0.10}[scen]
            ax_a.scatter(
                multi["test_dose_n_unique"] + x_off,
                multi["mae"],
                s=46,
                c=SCENARIO_COLOR[scen],
                marker=SCENARIO_MARKER[scen],
                edgecolors="black",
                linewidths=0.4,
                zorder=3,
            )
            for _, row in multi.iterrows():
                if "pico1" in str(row["group"]) and row["mae"] > 1:
                    ax_a.annotate(
                        _sheet_label(row["group"]),
                        (row["test_dose_n_unique"] + x_off, row["mae"]),
                        textcoords="offset points",
                        xytext=(-28, 6),
                        fontsize=7,
                    )

    ax_a.set_xticks([1, 2, 4])
    ax_a.set_xlim(0.6, 4.5)
    ax_a.set_xlabel("Distinct dose levels on held-out sheet")
    ax_a.set_ylabel("Leave-one-sheet-out MAE (controller dose units)")
    _panel_letter(ax_a, "a")

    const_counts = []
    multi_counts = []
    for scen in SCENARIO_ORDER:
        sub = df[df["scenario"] == scen]
        const_counts.append(int(sub["constant"].sum()))
        multi_counts.append(int((~sub["constant"]).sum()))
    x = np.arange(len(SCENARIO_ORDER))
    ax_b.bar(x, const_counts, color=CONSTANT_COLOR, width=0.62, label="Constant dose")
    ax_b.bar(
        x,
        multi_counts,
        bottom=const_counts,
        color="#1f2937",
        width=0.62,
        label="Multi-level dose",
    )
    for i, (c, m) in enumerate(zip(const_counts, multi_counts)):
        ax_b.text(i, c / 2 if c else 0.4, str(c), ha="center", va="center", fontsize=7, color="#111827")
        if m:
            ax_b.text(i, c + m + 0.6, str(m), ha="center", va="bottom", fontsize=7)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([SCENARIO_LABEL[s] for s in SCENARIO_ORDER])
    ax_b.set_ylabel("Event sheets")
    ax_b.set_ylim(0, max(c + m for c, m in zip(const_counts, multi_counts)) + 6)
    _panel_letter(ax_b, "b")

    ax_a.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=CONSTANT_COLOR, markersize=6, label="Constant dose"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=SCENARIO_COLOR["normalOperation"],
                markeredgecolor="black",
                markersize=6,
                label="Normal, multi-level",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=SCENARIO_COLOR["extremeEvents"],
                markeredgecolor="black",
                markersize=6,
                label="Extreme, multi-level",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=SCENARIO_COLOR["accidentalEvents"],
                markeredgecolor="black",
                markersize=6,
                label="Accidental, multi-level",
            ),
        ],
        loc="upper right",
        fontsize=6.5,
        handletextpad=0.4,
        borderpad=0.2,
    )
    ax_b.legend(
        handles=[
            Patch(facecolor=CONSTANT_COLOR, label="Constant dose"),
            Patch(facecolor="#1f2937", label="Multi-level dose"),
        ],
        loc="upper right",
        fontsize=6.5,
        handletextpad=0.4,
        borderpad=0.2,
    )

    if _SKILL_SCRIPTS.is_dir():
        sys.path.insert(0, str(_SKILL_SCRIPTS))
        from audit_panel_alignment import require_matplotlib_panel_alignment

        require_matplotlib_panel_alignment(
            fig,
            json_out=str(Path(out_png).with_suffix(".alignment.json")),
            overlay_svg=str(Path(out_png).with_suffix(".alignment.svg")),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            strict=True,
        )

    save_manuscript_figure(fig, out_png)
    plt.close(fig)
