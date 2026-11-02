"""Re-export manuscript figures at 600 dpi RGB without retraining the frozen analysis.

Edwards LOGO predictions are recomputed only if the cached OOF file is missing
(same seed 42 / 300-tree RF as the frozen pipeline).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    EDWARDS_PRIMARY_CV_PROTOCOL,
    EDWARDS_RF_N_ESTIMATORS,
    EDWARDS_SI_CV_PROTOCOL,
    EDWARDS_SOURCE_WATER_COLS,
    EDWARDS_SOURCE_WATER_ROUND,
    FIGURES_DIR,
    OUTPUT_ROOT,
    PROCESSED_DIR,
    RANDOM_STATE,
    TABLES_DIR,
    TURB_FLOC_THRESHOLD,
)
from figure_style import (  # noqa: E402
    WIDTH_DOUBLE_IN,
    WIDTH_SINGLE_IN,
    apply_rc,
    labelled_features,
    save_manuscript_figure,
)
from make_fig1_audit_scope import main as make_fig1  # noqa: E402


def _metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
    if len(y_true) >= 2 and np.var(y_true) > 1e-12:
        out["r2"] = float(r2_score(y_true, y_pred))
    else:
        out["r2"] = float("nan")
    return out


def _leave_one_group_cv_predictions(model_factory, X, y, groups):
    groups = np.asarray(groups)
    logo = LeaveOneGroupOut()
    y_pred = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in logo.split(X, y, groups):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])
    return y_pred


def _water_group_id(ed_model: pd.DataFrame, cols: list[str], prec: dict[str, int]) -> pd.Series:
    parts = [ed_model[c].round(prec[c]).astype(str) for c in cols]
    return parts[0].str.cat(parts[1:], sep="|")


def _make_water_groups(ed_model: pd.DataFrame) -> dict[str, pd.Series]:
    cols6 = EDWARDS_SOURCE_WATER_COLS
    prec6 = EDWARDS_SOURCE_WATER_ROUND
    gid6 = _water_group_id(ed_model, cols6, prec6)
    gid6_coag = gid6 + "|coag=" + ed_model["coagulant_code"].astype(str)
    return {
        EDWARDS_PRIMARY_CV_PROTOCOL: gid6,
        EDWARDS_SI_CV_PROTOCOL: gid6_coag,
    }

OOF_PATH = PROCESSED_DIR / "edwards_logo_oof.parquet"
FEATURE_COLS = [
    "turbidity_ntu",
    "ph",
    "doc_mg_l",
    "toc_in_mg_l",
    "uv254_in",
    "alkalinity_mg_caco3",
    "dose_mmol_l",
    "settle_time_min",
    "coagulant_code",
]


def _impute_edwards(ed: pd.DataFrame) -> pd.DataFrame:
    out = ed.copy()
    for col in [
        "doc_mg_l",
        "toc_in_mg_l",
        "uv254_in",
        "alkalinity_mg_caco3",
        "settle_time_min",
    ]:
        out[col] = out[col].fillna(out[col].median())
    return out


def _edwards_logo_frame() -> pd.DataFrame:
    ed = _impute_edwards(pd.read_parquet(PROCESSED_DIR / "edwards_clean.parquet"))
    if OOF_PATH.exists():
        oof = pd.read_parquet(OOF_PATH)
        if len(oof) == len(ed) and "y_logo" in oof.columns:
            return oof
    groups = _make_water_groups(ed)[EDWARDS_PRIMARY_CV_PROTOCOL].values
    X = ed[FEATURE_COLS].values
    y = ed["settled_turbidity_ntu"].values

    def rf_factory():
        return RandomForestRegressor(
            n_estimators=EDWARDS_RF_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    y_logo = _leave_one_group_cv_predictions(rf_factory, X, y, groups)
    out = ed.copy()
    out["y_logo"] = y_logo
    out["water_group_id"] = groups
    out.to_parquet(OOF_PATH, index=False)
    return out


def plot_edwards_parity() -> None:
    df = _edwards_logo_frame()
    y = df["settled_turbidity_ntu"].values
    y_par = df["y_logo"].values
    df = df.copy()
    df["ae"] = np.abs(df["settled_turbidity_ntu"] - df["y_logo"])
    gdf = df.groupby("water_group_id", as_index=False)["ae"].mean().rename(columns={"ae": "mae"})
    m = _metrics(y, y_par)
    n_g = int(df["water_group_id"].nunique())
    mae_s = gdf["mae"]
    summary = json.loads((OUTPUT_ROOT / "task1_edwards.json").read_text(encoding="utf-8"))
    m_base = summary["primary_cv_rf"]["global_mean_baseline"]
    n_lt2 = int((mae_s < 2).sum())
    n_gt5 = int((mae_s > 5).sum())

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_DOUBLE_IN, 3.5))
    ax = axes[0]
    ax.scatter(y, y_par, alpha=0.35, s=10, c="#2563eb", edgecolors="none")
    lims = [min(y.min(), y_par.min()), max(y.max(), y_par.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("Observed settled turbidity (NTU)")
    ax.set_ylabel("Predicted settled turbidity (NTU)")
    ax.set_title(f"(a) LOGO parity ({len(y):,} jars, {n_g} waters, R²={m['r2']:.2f})")
    ax.set_aspect("equal", adjustable="box")

    axb = axes[1]
    axb.hist(mae_s, bins=20, color="#2563eb", edgecolor="white", alpha=0.85)
    axb.axvline(2.0, color="#dc2626", ls="--", lw=1)
    axb.axvline(5.0, color="#991b1b", ls=":", lw=1)
    ymax = axb.get_ylim()[1]
    axb.text(2.05, ymax * 0.92, "2 NTU", color="#dc2626", fontsize=7, va="top")
    axb.text(5.05, ymax * 0.82, "5 NTU", color="#991b1b", fontsize=7, va="top")
    axb.set_xlabel("Per-source-water MAE (NTU)")
    axb.set_ylabel("Count")
    axb.set_title(f"(b) LOGO error by source (median {mae_s.median():.2f} NTU)")
    axb.text(
        0.03,
        0.97,
        f"{n_lt2}/{len(gdf)} MAE<2 NTU\n{n_gt5}/{len(gdf)} MAE>5 NTU",
        transform=axb.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none"),
    )
    rmse_gain = 1.0 - m["rmse"] / m_base["rmse"]
    fig.text(
        0.5,
        -0.02,
        f"Global-mean baseline RMSE={m_base['rmse']:.2f} NTU → model {m['rmse']:.2f} "
        f"({rmse_gain*100:.0f}% ↓)",
        ha="center",
        fontsize=7,
        color="#4b5563",
    )
    fig.tight_layout()
    save_manuscript_figure(fig, FIGURES_DIR / "fig2_edwards_parity.png")
    plt.close(fig)


def plot_shap() -> None:
    df = _impute_edwards(pd.read_parquet(PROCESSED_DIR / "edwards_clean.parquet"))
    X = df[FEATURE_COLS].values
    y = df["settled_turbidity_ntu"].values
    model = RandomForestRegressor(
        n_estimators=EDWARDS_RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    rng = np.random.default_rng(RANDOM_STATE)
    sample_n = min(800, len(X))
    idx = rng.choice(len(X), size=sample_n, replace=False)
    shap_values = explainer.shap_values(X[idx])
    names = labelled_features(FEATURE_COLS)
    display = df.iloc[idx][FEATURE_COLS].copy()
    display.columns = names

    plt.figure(figsize=(WIDTH_DOUBLE_IN, 4.4))
    shap.summary_plot(
        shap_values,
        display,
        feature_names=names,
        show=False,
        max_display=10,
    )
    fig = plt.gcf()
    fig.tight_layout()
    save_manuscript_figure(fig, FIGURES_DIR / "fig3_shap_summary.png")
    plt.close(fig)

    # SI waterfalls keep the same display names
    base_val = float(np.asarray(explainer.expected_value).ravel()[0])
    residual = np.abs(y - model.predict(X))
    for tag, row_i in [
        ("median_residual", int(np.argsort(residual)[len(residual) // 2])),
        ("high_residual", int(np.argsort(residual)[int(0.9 * len(residual))])),
    ]:
        sv = np.asarray(explainer.shap_values(X[[row_i]])).reshape(-1)
        plt.figure(figsize=(WIDTH_DOUBLE_IN, 4.0))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv,
                base_values=base_val,
                data=X[row_i],
                feature_names=names,
            ),
            show=False,
        )
        fig = plt.gcf()
        fig.tight_layout()
        save_manuscript_figure(
            fig, FIGURES_DIR / f"fig3_shap_waterfall_{tag}.png", also_pdf=False
        )
        plt.close(fig)


def plot_coagopt() -> None:
    opt = pd.read_parquet(PROCESSED_DIR / "naiades_min_dose.parquet")
    n_feasible = int(len(opt))
    ph_levels = sorted(opt["pH"].unique())
    nest_layers = list(
        zip(
            ph_levels,
            ["o", "^", "s"],
            [90, 55, 20],
            ["#059669", "#2563eb", "#d97706"],
            [False, False, True],
        )
    )
    conduct_levels = sorted(opt["conduct"].unique())
    fig, axes = plt.subplots(
        1,
        len(conduct_levels),
        figsize=(WIDTH_DOUBLE_IN, 3.2),
        sharey=True,
        sharex=True,
    )
    if len(conduct_levels) == 1:
        axes = [axes]
    for ax, cond in zip(axes, conduct_levels):
        sub_c = opt[opt["conduct"] == cond]
        for turb, grp in sub_c.groupby("turb_in", sort=True):
            doses = grp["min_dose"].unique()
            if len(doses) != 1:
                raise ValueError(f"min_dose not unique at cond={cond}, turb={turb}")
            x, y = float(turb), float(doses[0])
            for z, (_ph, marker, size, color, filled) in enumerate(nest_layers):
                if filled:
                    ax.scatter(
                        [x],
                        [y],
                        s=size,
                        marker=marker,
                        c=color,
                        edgecolors="white",
                        linewidths=0.35,
                        zorder=2 + z,
                    )
                else:
                    ax.scatter(
                        [x],
                        [y],
                        s=size,
                        marker=marker,
                        facecolors="none",
                        edgecolors=color,
                        linewidths=1.15,
                        zorder=2 + z,
                    )
        ax.set_title(rf"Conductivity = {cond:g} mS m$^{{-1}}$")
        ax.set_xlabel("Inlet turbidity (simulation units)")
        ax.grid(True, alpha=0.25, lw=0.5)
    axes[0].set_ylabel("Minimum coagulant dose (simulation units)")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=mk,
            color="w",
            markerfacecolor=col if fill else "none",
            markeredgecolor=col,
            markeredgewidth=1.2 if not fill else 0.8,
            markersize={90: 11, 55: 8.5, 20: 5.5}[sz],
            label=f"pH = {ph:g}",
        )
        for ph, mk, sz, col, fill in nest_layers
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        title="Inlet pH (concentric markers)",
        fontsize=8,
        title_fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout()
    save_manuscript_figure(fig, FIGURES_DIR / "fig4_coagopt_min_dose_vs_turbidity.png")
    plt.close(fig)

    counts = pd.read_csv(TABLES_DIR / "table_min_dose_level_counts.csv")
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE_IN * 0.72, 3.4))
    levels = counts.iloc[:, 0].astype(float)
    ax.bar(range(len(levels)), counts.iloc[:, 1].values, color="#059669", edgecolor="white")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{v:g}" for v in levels])
    ax.set_xlabel("Minimum coagulant dose (simulation units)")
    ax.set_ylabel("Count of inlet states")
    ax.set_title(f"CoagOpt min-dose levels (n = {n_feasible} feasible states)")
    fig.tight_layout()
    save_manuscript_figure(fig, FIGURES_DIR / "fig_s_coagopt_min_dose_level_counts.png")
    plt.close(fig)


def plot_stimela() -> None:
    ratio = pd.read_csv(TABLES_DIR / "table_stimela_ratio_by_coag_dos.csv")
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE_IN, 3.6))
    ax.errorbar(
        ratio["coag_dos"],
        ratio["mean_ratio"],
        yerr=ratio["std_ratio"].fillna(0),
        fmt="o-",
        capsize=3,
        color="#2563eb",
        ecolor="#2563eb",
    )
    ax.set_title("Stimela high turbidity: dose-dependent removal ratio")
    ax.set_xlabel(r"Coagulant dose (mmol L$^{-1}$)")
    ax.set_ylabel("Flocculated / inlet turbidity ratio (-)")
    fig.tight_layout()
    save_manuscript_figure(fig, FIGURES_DIR / "fig5_stimela_ratio_by_dose.png")
    plt.close(fig)


def plot_replica() -> None:
    from plot_fig6_replica import plot_replica_fig6

    logo_df = pd.read_csv(TABLES_DIR / "table_replica_logo_by_sheet.csv")
    plot_replica_fig6(logo_df, FIGURES_DIR / "fig6_replica_logo_boxplot.png")


def main() -> None:
    apply_rc()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Fig. 2 CoagOpt + Fig. S1 …")
    plot_coagopt()
    print("Fig. 3 Stimela …")
    plot_stimela()
    print("Fig. 6 replica …")
    plot_replica()
    print("Fig. 1 schematic …")
    make_fig1()
    print("Fig. 4 Edwards LOGO (may recompute OOF) …")
    plot_edwards_parity()
    print("Fig. 5 SHAP …")
    plot_shap()
    print("Done.")


if __name__ == "__main__":
    main()
