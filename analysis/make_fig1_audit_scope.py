"""Generate Fig. 1 — process chain + case-study scope (*Water Supply* vector figure)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from figure_style import save_manuscript_figure

sys_path = Path(__file__).resolve().parent
PROJECT = sys_path.parent
FIG_DIR = PROJECT / "figures"
JSON_PATH = PROJECT / "outputs" / "analysis_summary.json"

# 190 mm × 100 mm at 300 dpi ≈ 2244 × 1181 px; use inches for matplotlib
W_IN = 190 / 25.4
H_IN = 100 / 25.4

COLORS = {
    "edwards": "#2563eb",
    "coagopt": "#059669",
    "stimela": "#ea580c",
    "replica": "#7c3aed",
    "process_fill": "#f2f2f2",
    "text_muted": "#4b5563",
}


def _load_metrics() -> dict:
    with open(JSON_PATH, encoding="utf-8") as f:
        s = json.load(f)
    t1 = s["task1"]["primary_cv_rf"]
    logo = t1["logo_error_distribution"]
    boot = logo["cluster_bootstrap"]["pooled_r2"]
    rand = s["task1"]["random_5fold_rf_si"]["r2"]
    t2 = s["task2"]
    t3 = s["task3"]
    const_dose = t3["logo_sheets_constant_dose"]
    n_sheets = t3["logo_n_sheets"]
    trans = t3["scenario_transfer_primary"]
    r2_ext = trans[0]["r2"]
    r2_acc = trans[1]["r2"]
    imp_ext = trans[0]["rmse_improvement_vs_median"]
    imp_acc = trans[1]["rmse_improvement_vs_median"]
    return {
        "r2_random": rand,
        "r2_logo": t1["r2"],
        "ci_lo": boot["ci_low"],
        "ci_hi": boot["ci_high"],
        "r2_group": logo["r2_group_means_one_point_per_source_water"],
        "r2_within": logo["r2_within_group_median"],
        "n_const": const_dose,
        "n_sheets": n_sheets,
        "r2_ext": r2_ext,
        "r2_acc": r2_acc,
        "imp_ext": imp_ext,
        "imp_acc": imp_acc,
        "n_dose_levels": t2["n_unique_min_dose_levels"],
        "stimela_note": "k(dose) closed-form",
    }


def _box(ax, xy, w, h, title, subtitle=None, fc="#f2f2f2"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=0.75,
        edgecolor="black",
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=8, fontweight="bold")
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=6,
            color=COLORS["text_muted"],
        )


def main() -> None:
    m = _load_metrics()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.75,
        }
    )
    fig = plt.figure(figsize=(W_IN, H_IN))
    ax_a = fig.add_axes([0.04, 0.42, 0.92, 0.52])
    ax_b = fig.add_axes([0.04, 0.06, 0.92, 0.32])
    for ax in (ax_a, ax_b):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax_a.text(0, 1.02, "(a)", transform=ax_a.transAxes, fontsize=9, fontweight="bold", va="bottom")

    stages = [
        ("Raw water", "turb., pH, DOC/TOC"),
        ("Coagulation", "dose, coagulant"),
        ("Sedimentation", "settled turb."),
        ("Filtration", "no public labels"),
        ("Effluent", "no public labels"),
    ]
    n = len(stages)
    bw, bh, gap = 0.15, 0.35, 0.025
    x0 = 0.05
    ys = 0.55
    xs = [x0 + i * (bw + gap) for i in range(n)]
    for i, (ti, sub) in enumerate(stages):
        _box(ax_a, (xs[i], ys), bw, bh, ti, sub)
        if i < n - 1:
            ax_a.annotate(
                "",
                xy=(xs[i + 1], ys + bh / 2),
                xytext=(xs[i] + bw, ys + bh / 2),
                arrowprops=dict(arrowstyle="-|>", lw=0.75, color="black"),
            )

    # Dataset callouts
    def callout(x, y, text, color, dx=0, dy=0):
        ax_a.text(
            x,
            y,
            text,
            fontsize=6,
            color=color,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, lw=0.75),
        )
        ax_a.plot([x + dx, xs[1] + bw / 2], [y + dy, ys + bh], ls="--", lw=0.6, color=color)

    callout(
        0.22,
        0.38,
        "Edwards jar\n1,093 jars · 101 sources",
        COLORS["edwards"],
    )
    callout(
        0.38,
        0.92,
        "NAIADES CoagOpt\n333 inlets · 14 doses",
        COLORS["coagopt"],
        dy=-0.05,
    )
    callout(
        0.52,
        0.38,
        "Stimela high turb.\n1,620 runs",
        COLORS["stimela"],
    )
    coag_cx = xs[1] + bw / 2
    route_y = ys + bh + 0.14
    replica_x = 0.84
    ax_a.text(
        replica_x,
        route_y + 0.06,
        "dWTP replica\n87 sheets · 6 dose levels",
        fontsize=6,
        color=COLORS["replica"],
        ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=COLORS["replica"], lw=0.75),
    )
    ax_a.plot(
        [replica_x, coag_cx],
        [route_y + 0.02, route_y + 0.02],
        ls="--",
        lw=0.6,
        color=COLORS["replica"],
    )
    ax_a.annotate(
        "",
        xy=(coag_cx, ys + bh),
        xytext=(coag_cx, route_y + 0.02),
        arrowprops=dict(arrowstyle="-|>", lw=0.6, color=COLORS["replica"], ls="--"),
    )

    # Panel (b)
    ax_b.text(0, 1.05, "(b)", transform=ax_b.transAxes, fontsize=9, fontweight="bold", va="bottom")
    pw = 0.28
    panels = [
        (
            "A. Cross-validation protocol",
            [
                f"Random 5-fold  R² = {m['r2_random']:.2f}",
                f"LOGO (101 waters)  R² = {m['r2_logo']:.2f}",
                f"  95% CI [{m['ci_lo']:.2f}, {m['ci_hi']:.2f}]",
                f"Group-mean R² = {m['r2_group']:.2f}",
                f"Within-group median R² = {m['r2_within']:.2f}",
            ],
        ),
        (
            "B. Label structure",
            [
                f"CoagOpt: {int(m['n_dose_levels'])}-level turb. lookup",
                "Stimela: y = x · k(dose)",
                f"Replica: {m['n_const']}/{m['n_sheets']} sheets",
                "  constant dose per sheet",
            ],
        ),
        (
            "C. Supported claims",
            [
                "✓ In-batch forward interpretation",
                "✗ New source-water extrapolation",
                "✗ CoagOpt inverse ML",
                "✗ Cross-scenario transfer",
            ],
        ),
    ]
    for i, (title, lines) in enumerate(panels):
        x = 0.04 + i * (pw + 0.04)
        _box(ax_b, (x, 0.12), pw, 0.78, "", fc="white")
        ax_b.text(x + pw / 2, 0.86, title, fontsize=7, fontweight="bold", ha="center", va="top")
        yl = 0.64
        for line in lines:
            color = "black"
            if line.strip().startswith("✓"):
                color = "#15803d"
            elif line.strip().startswith("✗"):
                color = "#b91c1c"
            ax_b.text(
                x + 0.03,
                yl,
                line,
                fontsize=6,
                va="top",
                ha="left",
                color=color,
                family="monospace",
            )
            yl -= 0.12

    cap = (
        "Fig. 1. Scope of the case study. (a) Coagulation–sedimentation chain with "
        "four datasets mapped to observable stages (dashed); filtration and effluent stages lack "
        "labels in the public benchmarks used here. (b) Three lenses: grouped CV, label "
        "identifiability (Stimela: flocculated turbidity = inlet turbidity × k(dose)), and "
        "supported vs unsupported ML claims."
    )
    fig.text(0.5, 0.01, cap, ha="center", fontsize=6.5, wrap=True)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_manuscript_figure(fig, FIG_DIR / "fig1_audit_scope.png")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'fig1_audit_scope.png'}")


if __name__ == "__main__":
    main()
