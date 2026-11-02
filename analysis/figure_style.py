"""Shared IWA *Water Supply* figure export: print-width figsize, 600 dpi, RGB, no alpha."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from PIL import Image

FIG_DPI = 600
WIDTH_DOUBLE_IN = 7.48  # 190 mm full page / double-column span
WIDTH_SINGLE_IN = 3.54  # 90 mm single column

FEATURE_LABEL_MAP = {
    "turbidity_ntu": "Inlet turbidity (NTU)",
    "ph": "pH",
    "doc_mg_l": r"Inlet DOC (mg L$^{-1}$)",
    "toc_in_mg_l": r"Inlet TOC (mg L$^{-1}$)",
    "uv254_in": r"UV254 (cm$^{-1}$)",
    "alkalinity_mg_caco3": r"Alkalinity (mg CaCO$_3$ L$^{-1}$)",
    "dose_mmol_l": r"Coagulant dose (mmol L$^{-1}$)",
    "settle_time_min": "Settling time (min)",
    "coagulant_code": "Coagulant type",
}


def apply_rc() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "savefig.transparent": False,
        }
    )


def labelled_features(cols: list[str]) -> list[str]:
    return [FEATURE_LABEL_MAP.get(c, c.replace("_", " ")) for c in cols]


def flatten_png_rgb(path: Path, dpi: int = FIG_DPI) -> None:
    """Force RGB (no alpha) so typesetters do not paint transparency as black."""
    im = Image.open(path)
    if im.mode == "P":
        im = im.convert("RGBA")
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        alpha = im.split()[-1]
        bg.paste(im.convert("RGBA"), mask=alpha)
        bg.save(path, dpi=(dpi, dpi))
    elif im.mode != "RGB":
        im.convert("RGB").save(path, dpi=(dpi, dpi))
    else:
        im.save(path, dpi=(dpi, dpi))


def save_manuscript_figure(fig, path, *, dpi: int = FIG_DPI, also_pdf: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor("white")
    for ax in fig.get_axes():
        ax.set_facecolor("white")
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    if also_pdf:
        fig.savefig(
            path.with_suffix(".pdf"),
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            transparent=False,
        )
    if path.suffix.lower() == ".png":
        flatten_png_rgb(path, dpi=dpi)
