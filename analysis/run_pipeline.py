"""
Full data processing and analysis for Plan 1 (JWPE outline).

Tasks:
  1 — Edwards forward settled-turbidity ML + SHAP
  2 — NAIADES min-dose extraction + inlet -> min_dose CV
  3 — Replica scenario / leave-one-group-out validation
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
from lightgbm import LGBMRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Allow running as script from repo root or analysis folder
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    EDWARDS_PRIMARY_CV_PROTOCOL,
    EDWARDS_RF_N_ESTIMATORS,
    EDWARDS_SI_CV_PROTOCOL,
    EDWARDS_SOURCE_WATER_COLS,
    EDWARDS_SOURCE_WATER_ROUND,
    FIGURES_DIR,
    LOGO_CLUSTER_BOOTSTRAP_B,
    LOGO_CLUSTER_BOOTSTRAP_CI,
    N_FOLDS,
    OUTPUT_ROOT,
    PROCESSED_DIR,
    RANDOM_STATE,
    TABLES_DIR,
    TURB_FLOC_THRESHOLD,
)
from data_loaders import (  # noqa: E402
    extract_min_doses,
    load_coagulation_optimization,
    load_edwards,
    load_replica_combined,
    load_stimela_high_turbidity,
)
from figure_style import labelled_features, save_manuscript_figure  # noqa: E402


def _ensure_dirs() -> None:
    for d in (OUTPUT_ROOT, FIGURES_DIR, TABLES_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _cluster_bootstrap_logo(
    gdf: pd.DataFrame,
    y: np.ndarray,
    y_par: np.ndarray,
    g_primary: np.ndarray,
    *,
    n_resamples: int = LOGO_CLUSTER_BOOTSTRAP_B,
    ci: float = LOGO_CLUSTER_BOOTSTRAP_CI,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Cluster bootstrap: resample source-water groups; fixed LOGO preds; percentile CI."""
    rng = np.random.default_rng(seed)
    unique_groups = gdf["water_group_id"].tolist()
    n_g = len(unique_groups)
    g_lookup = gdf.set_index("water_group_id")
    masks = {g: (g_primary == g) for g in unique_groups}

    pooled_r2: list[float] = []
    group_mean_r2: list[float] = []
    median_mae: list[float] = []

    for _ in range(n_resamples):
        sampled = rng.choice(unique_groups, size=n_g, replace=True)
        boot_idx = np.concatenate([np.where(masks[g])[0] for g in sampled])
        yt, yp = y[boot_idx], y_par[boot_idx]
        if len(yt) >= 2 and np.var(yt) > 1e-12:
            pooled_r2.append(float(r2_score(yt, yp)))

        y_means = [float(g_lookup.loc[g, "y_mean"]) for g in sampled]
        yp_means = [float(g_lookup.loc[g, "yp_mean"]) for g in sampled]
        if len(y_means) >= 2 and np.var(y_means) > 1e-12:
            group_mean_r2.append(float(r2_score(y_means, yp_means)))

        median_mae.append(float(np.median([g_lookup.loc[g, "mae"] for g in sampled])))

    alpha = (1.0 - ci) / 2.0

    def _summ(values: list[float], point_from: float | None = None) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        return {
            "point": float(point_from if point_from is not None else np.mean(arr)),
            "ci_low": float(np.quantile(arr, alpha)),
            "ci_high": float(np.quantile(arr, 1.0 - alpha)),
        }

    rows = [
        {"metric": "pooled_r2", **_summ(pooled_r2)},
        {"metric": "group_mean_r2", **_summ(group_mean_r2)},
        {"metric": "median_mae_per_source_ntu", **_summ(median_mae)},
    ]
    extras = {
        "n_groups_mae_lt_2_ntu": int((gdf["mae"] < 2).sum()),
        "n_groups_mae_gt_5_ntu_bootstrap_context": int((gdf["mae"] > 5).sum()),
    }
    return pd.DataFrame(rows), extras


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


def _cv_predictions(model_factory, X, y, n_folds=N_FOLDS):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in kf.split(X):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])
    return y_pred


def _group_cv_predictions(model_factory, X, y, groups):
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise ValueError("Need at least 2 groups for grouped CV")
    n_splits = min(N_FOLDS, len(uniq))
    gkf = GroupKFold(n_splits=n_splits)
    y_pred = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])
    return y_pred


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
    parts = []
    for c in cols:
        parts.append(ed_model[c].round(prec[c]).astype(str))
    return parts[0].str.cat(parts[1:], sep="|")


def _make_water_groups(ed_model: pd.DataFrame) -> dict[str, pd.Series]:
    """Grouped CV keys — columns/rounding fixed in config.py."""
    cols6 = EDWARDS_SOURCE_WATER_COLS
    prec6 = EDWARDS_SOURCE_WATER_ROUND
    gid6 = _water_group_id(ed_model, cols6, prec6)
    gid6_coag = gid6 + "|coag=" + ed_model["coagulant_code"].astype(str)
    return {
        "random_5fold": pd.Series(np.arange(len(ed_model)), index=ed_model.index),
        EDWARDS_PRIMARY_CV_PROTOCOL: gid6,
        EDWARDS_SI_CV_PROTOCOL: gid6_coag,
        "leave_one_water_turb_ph": _water_group_id(
            ed_model, ["turbidity_ntu", "ph"], {"turbidity_ntu": 2, "ph": 2}
        ),
        "leave_one_water_uv_toc": _water_group_id(
            ed_model, ["uv254_in", "toc_in_mg_l"], {"uv254_in": 4, "toc_in_mg_l": 3}
        ),
        "leave_one_coagulant_only": ed_model["coagulant_code"].astype(str),
    }


def _repeated_stratified_class_cv(
    model_factory, X, y, n_repeats: int = 10, n_folds: int = N_FOLDS
) -> dict[str, float]:
    y = np.asarray(y)
    accs, f1s = [], []
    min_class = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(n_folds, min_class))
    for seed in range(n_repeats):
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + seed)
        y_pred = np.empty_like(y, dtype=object)
        for train_idx, test_idx in kf.split(X, y):
            model = model_factory()
            model.fit(X[train_idx], y[train_idx])
            y_pred[test_idx] = model.predict(X[test_idx])
        accs.append(accuracy_score(y, y_pred))
        f1s.append(f1_score(y, y_pred, average="macro", zero_division=0))
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "n_repeats": n_repeats,
        "n_splits": n_splits,
    }


def _cv_class_predictions(model_factory, X, y, n_folds=N_FOLDS):
    y = np.asarray(y)
    n_splits = min(n_folds, int(pd.Series(y).value_counts().min()))
    n_splits = max(2, n_splits)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.empty_like(y, dtype=object)
    for train_idx, test_idx in kf.split(X, y):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])
    return y_pred


def task1_edwards_forward(ed: pd.DataFrame, *, skip_shap: bool = False) -> dict:
    feature_cols = [
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
    ed_model = ed.copy()
    for col in [
        "doc_mg_l",
        "toc_in_mg_l",
        "uv254_in",
        "alkalinity_mg_caco3",
        "settle_time_min",
    ]:
        ed_model[col] = ed_model[col].fillna(ed_model[col].median())
    X = ed_model[feature_cols].values
    y = ed_model["settled_turbidity_ntu"].values
    primary_model = "RandomForest"

    water_groups = _make_water_groups(ed_model)
    ed_model["water_group_id"] = water_groups[EDWARDS_PRIMARY_CV_PROTOCOL]
    ed_model.to_parquet(PROCESSED_DIR / "edwards_clean.parquet", index=False)

    def lgbm_factory():
        return LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )

    def rf_factory():
        return RandomForestRegressor(
            n_estimators=EDWARDS_RF_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    cv_rows = []
    preds = {}
    for split_name, g_series in water_groups.items():
        g = g_series.values
        if split_name == "random_5fold":
            y_hat = _cv_predictions(rf_factory, X, y)
        else:
            y_hat = _leave_one_group_cv_predictions(rf_factory, X, y, g)
        preds[split_name] = y_hat
        cv_rows.append(
            {
                "cv_protocol": split_name,
                "model": "RandomForest",
                "n_groups": int(pd.Series(g).nunique()),
                **_metrics(y, y_hat),
            }
        )

    for name, factory in [("LightGBM", lgbm_factory)]:
        y_hat = _cv_predictions(factory, X, y)
        preds[name] = y_hat
        cv_rows.append(
            {
                "cv_protocol": "random_5fold",
                "model": name,
                "n_groups": int(len(ed_model)),
                **_metrics(y, y_hat),
            }
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    def pls_factory():
        return PLSRegression(n_components=min(8, X.shape[1] - 1))

    y_hat_pls = _cv_predictions(pls_factory, X_scaled, y)
    cv_rows.append(
        {
            "cv_protocol": "random_5fold",
            "model": "PLSR",
            "n_groups": int(len(ed_model)),
            **_metrics(y, y_hat_pls),
        }
    )

    metrics_df = pd.DataFrame(cv_rows)
    metrics_df.to_csv(TABLES_DIR / "table3_edwards_cv_metrics.csv", index=False)

    # Main parity: primary protocol (6-col source water, leave-one-group-out)
    loo_key = EDWARDS_PRIMARY_CV_PROTOCOL
    y_par = preds[loo_key]
    m_loo = _metrics(y, y_par)
    y_base = np.full_like(y, float(np.mean(y)))
    m_base = _metrics(y, y_base)
    n_g = int(water_groups[loo_key].nunique())

    # Feature ablation under primary LOGO (defend R² vs raw turbidity-only baseline)
    g_primary = water_groups[EDWARDS_PRIMARY_CV_PROTOCOL].values
    def rf_ablation_factory():
        return RandomForestRegressor(
            n_estimators=EDWARDS_RF_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    ablation_specs = [
        ("turb_in_linear", ["turbidity_ntu"], "linear"),
        ("turb_in_rf", ["turbidity_ntu"], "rf"),
        ("turb_in_plus_dose_rf", ["turbidity_ntu", "dose_mmol_l"], "rf"),
        ("turb_dose_ph_rf", ["turbidity_ntu", "dose_mmol_l", "ph"], "rf"),
        ("all_9_features_rf", feature_cols, "primary_reuse"),
    ]
    ablation_rows = []
    for tag, cols, kind in ablation_specs:
        X_sub = ed_model[cols].values
        if kind == "primary_reuse":
            y_hat = y_par
            n_estimators = EDWARDS_RF_N_ESTIMATORS
            pred_note = "same_as_table3_primary_logo"
        elif kind == "linear":
            y_hat = _leave_one_group_cv_predictions(
                lambda: LinearRegression(), X_sub, y, g_primary
            )
            n_estimators = None
            pred_note = "logo_cv"
        else:
            y_hat = _leave_one_group_cv_predictions(
                rf_ablation_factory, X_sub, y, g_primary
            )
            n_estimators = EDWARDS_RF_N_ESTIMATORS
            pred_note = "logo_cv"
        ablation_rows.append(
            {
                "feature_set": tag,
                "n_features": len(cols),
                "n_estimators": n_estimators,
                "cv_protocol": EDWARDS_PRIMARY_CV_PROTOCOL,
                "predictions_source": pred_note,
                **_metrics(y, y_hat),
            }
        )
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df["n_estimators"] = ablation_df["n_estimators"].astype("Int64")
    ablation_df.to_csv(
        TABLES_DIR / "si_edwards_feature_ablation_logo.csv", index=False
    )

    # Per-source-water MAE / within-group R²; pooled vs group-mean R² in summary CSV
    group_mae_rows = []
    for g in pd.Series(g_primary).unique():
        mask = g_primary == g
        yt, yp = y[mask], y_par[mask]
        row = {
            "water_group_id": g,
            "n_jars": int(mask.sum()),
            "mae": float(mean_absolute_error(yt, yp)),
            "settled_turb_std_ntu": float(np.std(yt)) if len(yt) >= 2 else 0.0,
        }
        if len(yt) >= 2 and np.var(yt) > 1e-12:
            row["r2_within_group"] = float(r2_score(yt, yp))
        else:
            row["r2_within_group"] = float("nan")
        row["y_mean"] = float(np.mean(yt))
        row["yp_mean"] = float(np.mean(yp))
        group_mae_rows.append(row)
    gdf = pd.DataFrame(group_mae_rows)
    gdf.to_csv(TABLES_DIR / "si_edwards_logo_mae_by_water_group.csv", index=False)
    mae_s = gdf["mae"]
    r2_pooled = float(m_loo["r2"])
    r2_group_means = float(r2_score(gdf["y_mean"], gdf["yp_mean"]))
    r2_within = gdf["r2_within_group"].dropna()
    subset = gdf[(gdf["n_jars"] >= 12) & (gdf["settled_turb_std_ntu"] > 1.0)]
    r2_within_sub = subset["r2_within_group"].dropna()
    logo_distribution = {
        "n_groups": int(len(gdf)),
        "mae_median_ntu": float(mae_s.median()),
        "mae_p25_ntu": float(mae_s.quantile(0.25)),
        "mae_p75_ntu": float(mae_s.quantile(0.75)),
        "n_groups_mae_gt_5_ntu": int((mae_s > 5).sum()),
        "n_single_jar_groups": int((gdf["n_jars"] == 1).sum()),
        "r2_pooled_sample_weighted": r2_pooled,
        "r2_group_means_one_point_per_source_water": r2_group_means,
        "r2_within_group_median": float(r2_within.median()) if len(r2_within) else None,
        "r2_within_group_median_n_ge_12_sd_gt_1ntu": float(r2_within_sub.median())
        if len(r2_within_sub)
        else None,
        "n_groups_with_defined_within_r2": int(r2_within.count()),
        "n_groups_subset_n12_sd1": int(len(subset)),
    }
    pd.DataFrame([logo_distribution]).to_csv(
        TABLES_DIR / "si_edwards_logo_r2_summary.csv", index=False
    )

    boot_df, boot_counts = _cluster_bootstrap_logo(gdf, y, y_par, g_primary)
    boot_df.loc[boot_df["metric"] == "pooled_r2", "point"] = r2_pooled
    boot_df.loc[boot_df["metric"] == "group_mean_r2", "point"] = r2_group_means
    boot_df.loc[boot_df["metric"] == "median_mae_per_source_ntu", "point"] = float(
        mae_s.median()
    )
    boot_df["ci_level"] = LOGO_CLUSTER_BOOTSTRAP_CI
    boot_df["n_resamples"] = LOGO_CLUSTER_BOOTSTRAP_B
    boot_df["resample_unit"] = "source_water_group"
    boot_df["resample_method"] = "cluster_bootstrap_with_replacement"
    boot_df["ci_method"] = "percentile"
    boot_df["random_seed"] = RANDOM_STATE
    boot_df.to_csv(TABLES_DIR / "si_edwards_logo_cluster_bootstrap.csv", index=False)
    logo_distribution["n_groups_mae_lt_2_ntu"] = boot_counts["n_groups_mae_lt_2_ntu"]
    logo_distribution["cluster_bootstrap"] = boot_df.set_index("metric")[
        ["point", "ci_low", "ci_high"]
    ].to_dict(orient="index")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax = axes[0]
    ax.scatter(y, y_par, alpha=0.35, s=12, c="#2563eb", edgecolors="none")
    lims = [min(y.min(), y_par.min()), max(y.max(), y_par.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("Observed settled turbidity (NTU)")
    ax.set_ylabel("Predicted settled turbidity (NTU)")
    n_jars = len(y)
    ax.set_title(
        f"(a) LOGO parity ({n_jars:,} jars from {n_g} source waters, R²={m_loo['r2']:.3f})"
    )
    ax.set_aspect("equal", adjustable="box")

    axb = axes[1]
    axb.hist(gdf["mae"], bins=20, color="#2563eb", edgecolor="white", alpha=0.85)
    axb.axvline(2.0, color="#dc2626", ls="--", lw=1)
    axb.axvline(5.0, color="#991b1b", ls=":", lw=1)
    ymax = axb.get_ylim()[1]
    axb.text(2.05, ymax * 0.92, "2 NTU", color="#dc2626", fontsize=7, va="top")
    axb.text(5.05, ymax * 0.82, "5 NTU", color="#991b1b", fontsize=7, va="top")
    n_lt2 = int(boot_counts["n_groups_mae_lt_2_ntu"])
    n_gt5 = int((mae_s > 5).sum())
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
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )
    rmse_gain = 1.0 - m_loo["rmse"] / m_base["rmse"]
    fig.text(
        0.5,
        -0.02,
        f"Global-mean baseline RMSE={m_base['rmse']:.2f} NTU → model {m_loo['rmse']:.2f} "
        f"({rmse_gain*100:.0f}% ↓); random 5-fold R²={_metrics(y, preds['random_5fold'])['r2']:.3f} (SI)",
        ha="center",
        fontsize=7,
        color="gray",
    )
    fig.tight_layout()
    save_manuscript_figure(fig, FIGURES_DIR / "fig2_edwards_parity.png")
    plt.close(fig)

    results = {
        r["model"]: {k: r[k] for k in ("rmse", "mae", "r2") if k in r}
        for _, r in metrics_df.iterrows()
        if r["cv_protocol"] == "random_5fold"
    }

    if not skip_shap:
        # SHAP on full-data primary model (RandomForest); in-sample interpretation only
        model = rf_factory()
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        sample_n = min(800, len(X))
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X), size=sample_n, replace=False)
        shap_values = explainer.shap_values(X[idx])

        shap_names = labelled_features(feature_cols)
        shap_display = ed_model.iloc[idx][feature_cols].copy()
        shap_display.columns = shap_names
        plt.figure(figsize=(8, 5))
        shap.summary_plot(
            shap_values,
            shap_display,
            feature_names=shap_names,
            show=False,
            max_display=10,
        )
        plt.tight_layout()
        save_manuscript_figure(plt.gcf(), FIGURES_DIR / "fig3_shap_summary.png")
        plt.close()

        base_val = float(np.asarray(explainer.expected_value).ravel()[0])
        residual = np.abs(y - preds["random_5fold"])
        for tag, row_i in [
            ("median_residual", int(np.argsort(residual)[len(residual) // 2])),
            ("high_residual", int(np.argsort(residual)[int(0.9 * len(residual))])),
        ]:
            sv = explainer.shap_values(X[[row_i]])
            if isinstance(sv, list):
                sv = sv[0]
            sv = np.asarray(sv).reshape(-1)
            plt.figure(figsize=(7, 4))
            shap.waterfall_plot(
                shap.Explanation(
                    values=sv,
                    base_values=base_val,
                    data=X[row_i],
                    feature_names=shap_names,
                ),
                show=False,
            )
            plt.tight_layout()
            save_manuscript_figure(
                plt.gcf(),
                FIGURES_DIR / f"fig3_shap_waterfall_{tag}.png",
                also_pdf=False,
            )
            plt.close()

    grouped = metrics_df[metrics_df["model"] == "RandomForest"].set_index("cv_protocol")
    primary_row = grouped.loc[EDWARDS_PRIMARY_CV_PROTOCOL].to_dict()
    primary_row["global_mean_baseline"] = m_base
    primary_row["rmse_reduction_vs_baseline"] = float(
        1.0 - primary_row["rmse"] / m_base["rmse"]
    )
    primary_row["logo_error_distribution"] = logo_distribution
    primary_row["feature_ablation_logo"] = ablation_rows
    summary = {
        "n_rows": int(len(ed_model)),
        "n_source_water_groups_primary": int(water_groups[EDWARDS_PRIMARY_CV_PROTOCOL].nunique()),
        "grouping_spec": {
            "primary_protocol": EDWARDS_PRIMARY_CV_PROTOCOL,
            "columns": EDWARDS_SOURCE_WATER_COLS,
            "round_decimals": EDWARDS_SOURCE_WATER_ROUND,
        },
        "n_coagulant_codes": int(ed_model["coagulant_code"].nunique()),
        "target_column": "settled_turbidity_ntu (XLS col 21, NTU)",
        "primary_model": primary_model,
        "primary_cv_rf": primary_row,
        "random_5fold_rf_si": grouped.loc["random_5fold"].to_dict(),
        "si_water_plus_coagulant_rf": grouped.loc[EDWARDS_SI_CV_PROTOCOL].to_dict(),
        "grouped_cv_all_protocols": metrics_df.to_dict(orient="records"),
    }
    with open(OUTPUT_ROOT / "task1_edwards.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def task2_min_dose(co: pd.DataFrame) -> dict:
    opt = extract_min_doses(co, TURB_FLOC_THRESHOLD)
    opt.to_parquet(PROCESSED_DIR / "naiades_min_dose.parquet", index=False)

    n_inlets = int(co["inlet_key"].nunique())
    n_feasible = int(len(opt))

    dose_levels = opt["min_dose_class"].value_counts().sort_index()
    dose_levels.to_csv(TABLES_DIR / "table_min_dose_level_counts.csv", header=["count"])

    # Fig. 4 (main): facet by conduct; concentric pH markers at identical (turb, dose)
    ph_levels = sorted(opt["pH"].unique())
    # Outermost → innermost (draw back-to-front); sizes in matplotlib points^2
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
        1, len(conduct_levels), figsize=(10, 3.6), sharey=True, sharex=True
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
        ax.set_title(rf"Conductivity = {cond:g} mS m$^{{-1}}$", fontsize=9)
        ax.set_xlabel("Inlet turbidity (simulation units)")
        ax.grid(True, alpha=0.25, lw=0.5)
    axes[0].set_ylabel("Minimum coagulant dose (simulation units)")
    from matplotlib.lines import Line2D

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
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout()
    save_manuscript_figure(fig, FIGURES_DIR / "fig4_coagopt_min_dose_vs_turbidity.png")
    plt.close(fig)

    # SI: discrete min-dose level counts (45 / 9 / 18 / 9)
    fig, ax = plt.subplots(figsize=(6, 4))
    levels = dose_levels.index.astype(float)
    ax.bar(range(len(levels)), dose_levels.values, color="#059669", edgecolor="white")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{v:.4g}" for v in levels], rotation=45, ha="right")
    ax.set_xlabel("Minimum coagulant dose (simulation units)")
    ax.set_ylabel("Count of inlet states")
    ax.set_title(f"CoagOpt min-dose levels (n = {n_feasible} feasible states)")
    fig.tight_layout()
    save_manuscript_figure(fig, FIGURES_DIR / "fig_s_coagopt_min_dose_level_counts.png")
    plt.close(fig)

    # Dose–response curves for three inlets with most grid points
    counts = co.groupby("inlet_key").size().sort_values(ascending=False)
    chosen = list(counts.head(3).index)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, key in zip(axes, chosen):
        sub = co[co["inlet_key"] == key].sort_values("dose")
        ax.plot(sub["dose"], sub["turb_floc"], "o-", ms=4, lw=1.5)
        if key in set(opt["inlet_key"]):
            md = float(opt.loc[opt["inlet_key"] == key, "min_dose"].iloc[0])
            ax.axhline(TURB_FLOC_THRESHOLD, color="gray", ls="--", lw=1)
            ax.axvline(md, color="#dc2626", ls=":", lw=1.5, label=f"min dose={md:.2g}")
        row = sub.iloc[0]
        ax.set_title(f"Turb={row.turb_in:.2g}, pH={row.pH:.2g}")
        ax.set_xlabel("Dose")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Flocculated turbidity (NTU)")
    fig.suptitle("Dose–response examples (NAIADES CoagulationOptimization)")
    fig.tight_layout()
    save_manuscript_figure(fig, FIGURES_DIR / "fig5_dose_response_examples.png")
    plt.close(fig)

    def rf_cls_factory():
        return RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        )

    def _classify_subset(sub: pd.DataFrame, tag: str) -> dict:
        if len(sub) < 10:
            return {"tag": tag, "n": len(sub), "skipped": True}
        Xs = sub[["turb_in", "pH", "conduct"]].values
        yc = sub["min_dose_class"].astype(str).values
        rep = _repeated_stratified_class_cv(rf_cls_factory, Xs, yc)
        y_pred = _cv_class_predictions(rf_cls_factory, Xs, yc)
        cm = confusion_matrix(yc, y_pred, labels=sorted(set(yc)))
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in sorted(set(yc))],
            columns=[f"pred_{l}" for l in sorted(set(yc))],
        )
        cm_df.to_csv(TABLES_DIR / f"si_confusion_min_dose_{tag}.csv")
        return {
            "tag": tag,
            "n": int(len(sub)),
            "n_classes": int(sub["min_dose_class"].nunique()),
            "majority_baseline_accuracy": float(
                sub["min_dose_class"].astype(str).value_counts(normalize=True).max()
            ),
            **rep,
        }

    cls_rows = [
        _classify_subset(opt, "tau1_all"),
        _classify_subset(opt[opt["requires_coagulant"]], "tau1_requires_coagulant"),
    ]
    opt_t05 = extract_min_doses(co, 0.5)
    cls_rows.append(_classify_subset(opt_t05, "tau0p5_all"))
    cls_rows.append(
        _classify_subset(
            opt_t05[opt_t05["requires_coagulant"]], "tau0p5_requires_coagulant"
        )
    )
    cls_df = pd.DataFrame([r for r in cls_rows if not r.get("skipped")])
    cls_df.to_csv(TABLES_DIR / "si_coagopt_classifier_cv.csv", index=False)

    X = opt[["turb_in", "pH", "conduct"]].values
    y_reg = opt["min_dose"].values

    def lgbm_min_factory():
        return LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=5,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )

    y_hat_reg = _cv_predictions(lgbm_min_factory, X, y_reg)
    reg_metrics = _metrics(y_reg, y_hat_reg)
    pd.DataFrame([reg_metrics], index=["LightGBM_min_dose_regression"]).to_csv(
        TABLES_DIR / "si_min_dose_regression_cv.csv"
    )

    # Threshold sensitivity
    sens_rows = []
    for tau in [0.5, 1.0, 2.0]:
        o = extract_min_doses(co, tau)
        sens_rows.append(
            {
                "tau_ntu": tau,
                "n_feasible_inlets": len(o),
                "n_near_zero_min_dose": int((o["min_dose"] < 0.01).sum()),
                "n_unique_min_dose_levels": int(o["min_dose"].round(4).nunique()),
            }
        )
    pd.DataFrame(sens_rows).to_csv(TABLES_DIR / "si_tau_sensitivity.csv", index=False)

    summary = {
        "n_inlet_states": n_inlets,
        "n_feasible_min_dose": n_feasible,
        "n_near_zero_min_dose": int((opt["min_dose"] < 0.01).sum()),
        "n_requires_coagulant": int(opt["requires_coagulant"].sum()),
        "n_unique_min_dose_levels": int(opt["min_dose_class"].nunique()),
        "feasible_fraction": round(n_feasible / n_inlets, 4),
        "threshold_ntu": TURB_FLOC_THRESHOLD,
        "classification_repeated_cv_si": cls_rows,
        "regression_cv_metrics_si": reg_metrics,
    }
    structure = task2_coagopt_structure(co, opt)
    stim = load_stimela_high_turbidity()
    stimela = task2_stimela_structure(stim)
    summary["coagopt_structure"] = structure
    summary["stimela_structure"] = stimela
    summary["task2_main_claim"] = (
        "NAIADES coagulation tables are structural benchmarks (lookup / closed-form); "
        "do not report CoagOpt RF classifier or Stimela high R² as ML contributions."
    )

    with open(OUTPUT_ROOT / "task2_min_dose.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def task2_coagopt_structure(co: pd.DataFrame, opt: pd.DataFrame) -> dict:
    """Document CoagulationOptimization as discrete turb_in → min_dose lookup."""
    turb_dose = (
        opt.groupby("turb_in")["min_dose"]
        .agg(min_dose_min="min", min_dose_max="max", n_unique="nunique")
        .reset_index()
    )
    turb_dose.to_csv(TABLES_DIR / "table_task2_coagopt_turb_lookup.csv", index=False)

    # pH × conduct effect on Turb_FLOC at fixed (turb_in, dose): max range
    co2 = co.copy()
    co2["cell"] = (
        co2["turb_in"].round(4).astype(str)
        + "|"
        + co2["dose"].round(4).astype(str)
    )
    ranges = co2.groupby("cell").apply(
        lambda g: float(g["turb_floc"].max() - g["turb_floc"].min()),
        include_groups=False,
    )
    ph_cond_effect = {
        "cells_n": int(len(ranges)),
        "max_turb_floc_range_within_cell": float(ranges.max()),
        "frac_cells_range_above_1e-3": float((ranges > 1e-3).mean()),
    }

    req = opt[opt["requires_coagulant"]].copy()
    y_cls = req["min_dose_class"].astype(str).values
    X_turb = req[["turb_in"]].values
    stump = RandomForestClassifier(max_depth=1, random_state=RANDOM_STATE)
    stump_pred = _cv_class_predictions(
        lambda: RandomForestClassifier(max_depth=1, random_state=RANDOM_STATE),
        X_turb,
        y_cls,
    )
    lookup_baseline = {
        "n_requires_coagulant": int(len(req)),
        "decision_stump_turb_in_only_accuracy": float(
            accuracy_score(y_cls, stump_pred)
        ),
        "decision_stump_turb_in_only_macro_f1": float(
            f1_score(y_cls, stump_pred, average="macro", zero_division=0)
        ),
    }

    out = {
        "interpretation": "CoagOpt min-dose is a discrete turb_in threshold lookup; pH/conduct inert on grid.",
        "ph_cond_effect_on_turb_floc": ph_cond_effect,
        "lookup_baseline": lookup_baseline,
    }
    with open(OUTPUT_ROOT / "task2_coagopt_structure.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def task2_stimela_structure(stim: pd.DataFrame) -> dict:
    """Audit Stimela high-turbidity table: closed-form turb_flocc ≈ turb_in × k(coag_dos)."""
    feature_cols = ["turb_in", "ph", "conduct", "coag_dos", "cl_dosage"]
    d = stim.dropna(subset=feature_cols + ["turb_flocc"]).copy()
    d = d[d["turb_in"] > 0]
    d["removal_ratio"] = d["turb_flocc"] / d["turb_in"]
    d.to_parquet(PROCESSED_DIR / "stimela_high_turb.parquet", index=False)

    ratio_by_dose = (
        d.groupby("coag_dos")["removal_ratio"]
        .agg(mean_ratio="mean", std_ratio="std", n="count")
        .reset_index()
    )
    ratio_by_dose.to_csv(TABLES_DIR / "table_stimela_ratio_by_coag_dos.csv", index=False)

    y = d["turb_flocc"].values
    X5 = d[feature_cols].values
    X2 = d[["turb_in", "coag_dos"]].values

    def rf_factory():
        return RandomForestRegressor(300, random_state=RANDOM_STATE, n_jobs=-1)

    met5 = _metrics(y, _cv_predictions(rf_factory, X5, y))
    met2 = _metrics(y, _cv_predictions(rf_factory, X2, y))
    X2s = StandardScaler().fit_transform(X2)
    met_pls = _metrics(
        y,
        _cv_predictions(
            lambda: PLSRegression(n_components=2), X2s, y
        ),
    )

    corrs = {
        c: float(d["turb_flocc"].corr(d[c]))
        for c in ["turb_in", "ph", "conduct", "coag_dos", "cl_dosage"]
    }
    audit_rows = [
        {"check": "RF_5_features_random_cv", **met5},
        {"check": "RF_turb_in_coag_dos_only_random_cv", **met2},
        {"check": "PLSR_turb_in_coag_dos_linear_cv", **met_pls},
    ]
    pd.DataFrame(audit_rows).to_csv(
        TABLES_DIR / "si_stimela_ml_vs_closed_form.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        ratio_by_dose["coag_dos"],
        ratio_by_dose["mean_ratio"],
        yerr=ratio_by_dose["std_ratio"].fillna(0),
        fmt="o-",
        capsize=3,
    )
    ax.set_xlabel(r"Coagulant dose (mmol L$^{-1}$)")
    ax.set_ylabel("Flocculated / inlet turbidity ratio (-)")
    ax.set_title("Stimela high turbidity: dose-dependent removal ratio")
    fig.tight_layout()
    save_manuscript_figure(fig, FIGURES_DIR / "fig5_stimela_ratio_by_dose.png")
    plt.close(fig)

    summary = {
        "n_rows": int(len(d)),
        "n_unique_coag_dos": int(d["coag_dos"].nunique()),
        "closed_form": "turb_flocc = turb_in * k(coag_dos); k has 6 discrete levels",
        "max_std_ratio_within_dose": float(ratio_by_dose["std_ratio"].max()),
        "correlations_with_turb_flocc": corrs,
        "ml_audit_si_only": {
            "rf_5feat": met5,
            "rf_2feat": met2,
            "plsr_2feat_linear": met_pls,
            "note": "High R² reflects closed-form simulation, not learnable multivariate process.",
        },
    }
    with open(OUTPUT_ROOT / "task2_stimela_structure.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def task3_replica(allr: pd.DataFrame) -> dict:
    feats = ["turbidity_in", "ph_in", "conductivity_in"]
    df = allr.dropna(subset=feats + ["coagulantDosage"]).copy()
    if "scenario" not in df.columns:
        df["scenario"] = (
            df["group"]
            .astype(str)
            .str.replace(".xlsx", "", regex=False)
            .str.split("_", n=1)
            .str[0]
        )
    df["scenario"] = df["scenario"].replace(
        {
            "normalOperation": "normalOperation",
            "extremeEvents": "extremeEvents",
            "accidentalEvents": "accidentalEvents",
        }
    )
    df.to_parquet(PROCESSED_DIR / "replica_clean.parquet", index=False)

    # Random 5-fold (report as cautionary / SI, not main claim)
    X = df[feats].values
    y = df["coagulantDosage"].values
    y_rand = _cv_predictions(
        lambda: RandomForestRegressor(200, random_state=RANDOM_STATE, n_jobs=-1),
        X,
        y,
    )
    random_cv = _metrics(y, y_rand)
    pd.DataFrame([random_cv], index=["random_5fold_RF"]).to_csv(
        TABLES_DIR / "si_replica_random_cv.csv"
    )

    # Protocol A: train normal -> test extreme / accidental
    scenario_results = []
    train = df[df["scenario"] == "normalOperation"]
    for test_scenario in ["extremeEvents", "accidentalEvents"]:
        test = df[df["scenario"] == test_scenario]
        if len(train) < 50 or len(test) < 20:
            continue
        m = RandomForestRegressor(200, random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(train[feats], train["coagulantDosage"])
        pred = m.predict(test[feats])
        y_te = test["coagulantDosage"].values
        train_median = float(train["coagulantDosage"].median())
        pred_base = np.full(len(y_te), train_median)
        met_model = _metrics(y_te, pred)
        met_base = _metrics(y_te, pred_base)
        scenario_results.append(
            {
                "protocol": "train_normal",
                "test_scenario": test_scenario,
                "n_train": len(train),
                "n_test": len(test),
                **met_model,
                "rmse_train_median_baseline": met_base["rmse"],
                "mae_train_median_baseline": met_base["mae"],
                "rmse_improvement_vs_median": float(
                    1.0 - met_model["rmse"] / met_base["rmse"]
                ),
            }
        )
    scenario_df = pd.DataFrame(scenario_results)
    scenario_df.to_csv(TABLES_DIR / "table_replica_scenario_transfer.csv", index=False)

    # Intra-scenario reference (normal only, random CV — not a deployment claim)
    normal = df[df["scenario"] == "normalOperation"]
    if len(normal) >= 100:
        y_n = normal["coagulantDosage"].values
        X_n = normal[feats].values
        y_n_pred = _cv_predictions(
            lambda: RandomForestRegressor(200, random_state=RANDOM_STATE, n_jobs=-1),
            X_n,
            y_n,
        )
        intra_normal = _metrics(y_n, y_n_pred)
        pd.DataFrame([intra_normal], index=["normal_only_random_5fold"]).to_csv(
            TABLES_DIR / "si_replica_normal_intra_cv.csv"
        )
    else:
        intra_normal = None

    # Protocol B: leave-one-group-out
    logo_rows = []
    for g in df["group"].unique():
        te = df[df["group"] == g]
        tr = df[df["group"] != g]
        if len(te) < 20:
            continue
        m = RandomForestRegressor(200, random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(tr[feats], tr["coagulantDosage"])
        pred = m.predict(te[feats])
        y_true = te["coagulantDosage"].values
        train_median = float(tr["coagulantDosage"].median())
        mae_base = float(
            mean_absolute_error(y_true, np.full(len(y_true), train_median))
        )
        mae_model = float(mean_absolute_error(y_true, pred))
        skill = (
            float(1.0 - mae_model / mae_base) if mae_base > 1e-12 else float("nan")
        )
        row = {
            "group": g,
            "scenario": te["scenario"].iloc[0],
            "n_test": len(te),
            "test_dose_n_unique": int(te["coagulantDosage"].nunique()),
            "mae": mae_model,
            "mae_baseline_median": mae_base,
            "skill_vs_median": skill,
            **_metrics(y_true, pred),
        }
        logo_rows.append(row)
    logo_df = pd.DataFrame(logo_rows)
    logo_df.to_csv(TABLES_DIR / "table_replica_logo_by_sheet.csv", index=False)

    multilevel = logo_df[logo_df["test_dose_n_unique"] > 1].copy()
    multilevel.to_csv(TABLES_DIR / "table_replica_multilevel_sheets.csv", index=False)

    from plot_fig6_replica import plot_replica_fig6

    plot_replica_fig6(logo_df, FIGURES_DIR / "fig6_replica_logo_boxplot.png")

    skill_finite = skill_plot["skill_vs_median"].dropna()
    multilevel_sorted = multilevel.sort_values("r2", na_position="last")
    examples = pd.concat([multilevel_sorted.head(3), multilevel_sorted.tail(2)])
    examples.to_csv(TABLES_DIR / "table_replica_logo_examples.csv", index=False)

    r2_defined = logo_df["r2"].notna().sum()
    summary = {
        "n_rows": int(len(df)),
        "n_groups": int(df["group"].nunique()),
        "dosage_grid_n_unique_global": int(df["coagulantDosage"].nunique()),
        "random_5fold_cv_si_only": random_cv,
        "logo_n_sheets": int(len(logo_df)),
        "logo_r2_defined_sheets": int(r2_defined),
        "logo_r2_undefined_sheets": int(len(logo_df) - r2_defined),
        "logo_mae_median": float(logo_df["mae"].median()),
        "logo_mae_p90": float(logo_df["mae"].quantile(0.9)),
        "logo_skill_median_excl_zero_baseline": float(skill_finite.median())
        if len(skill_finite)
        else None,
        "logo_sheets_constant_dose": int((logo_df["test_dose_n_unique"] == 1).sum()),
        "logo_sheets_multilevel_dose": int(len(multilevel)),
        "normal_intra_random_cv_si": intra_normal,
        "scenario_transfer_primary": scenario_results,
    }
    with open(OUTPUT_ROOT / "task3_replica.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def write_model_protocol_table() -> None:
    rows = [
        {"item": "CV folds", "value": N_FOLDS},
        {"item": "Random seed", "value": RANDOM_STATE},
        {
            "item": "Task 1 primary",
            "value": f"RandomForest n={EDWARDS_RF_N_ESTIMATORS} (LOGO); SHAP on RF",
        },
        {
            "item": "Task 1 ablation (LOGO)",
            "value": f"RandomForest n={EDWARDS_RF_N_ESTIMATORS}; all_9 row reuses primary LOGO preds",
        },
        {"item": "LightGBM (Task 1)", "value": "n_estimators=400, lr=0.05, num_leaves=31"},
        {
            "item": "Task 2 primary",
            "value": "Stimela high-turb forward RF; CoagOpt = structural lookup (SI)",
        },
        {"item": "Edwards primary grouped CV", "value": EDWARDS_PRIMARY_CV_PROTOCOL},
        {
            "item": "Edwards LOGO cluster bootstrap",
            "value": f"B={LOGO_CLUSTER_BOOTSTRAP_B}, CI={LOGO_CLUSTER_BOOTSTRAP_CI}, no retrain",
        },
        {"item": "Replica grouped model", "value": "RandomForest n_estimators=200"},
        {"item": "Min-dose threshold (NTU)", "value": TURB_FLOC_THRESHOLD},
    ]
    pd.DataFrame(rows).to_csv(TABLES_DIR / "table2_model_protocol.csv", index=False)


def write_dataset_summary(ed: pd.DataFrame, co: pd.DataFrame, allr: pd.DataFrame) -> None:
    rows = [
        {
            "dataset": "Edwards jar",
            "n_records": len(ed),
            "role": "Task 1 forward",
        },
        {
            "dataset": "NAIADES CoagulationOptimization",
            "n_records": len(co),
            "role": "Task 2 dose grid",
        },
        {
            "dataset": "NAIADES replica combined",
            "n_records": len(allr),
            "role": "Task 3 grouped validation",
        },
    ]
    pd.DataFrame(rows).to_csv(TABLES_DIR / "table1_dataset_summary.csv", index=False)


def main() -> None:
    _ensure_dirs()
    print("Loading data...")
    ed = load_edwards()
    co = load_coagulation_optimization()
    allr = load_replica_combined()
    write_model_protocol_table()
    write_dataset_summary(ed, co, allr)

    print("Task 1 — Edwards forward...")
    s1 = task1_edwards_forward(ed)
    print(json.dumps(s1, indent=2))

    print("Task 2 — NAIADES min-dose...")
    s2 = task2_min_dose(co)
    print(json.dumps(s2, indent=2))

    print("Task 3 — Replica grouped validation...")
    s3 = task3_replica(allr)
    print(json.dumps(s3, indent=2))

    audit = pd.DataFrame(
        [
            {
                "benchmark": "Edwards jar",
                "finding": "Grouped-structure optimism (protocol mismatch)",
                "metric": "RF R² random 5-fold vs leave-one-source-water (6 col)",
                "value": f"{s1['random_5fold_rf_si']['r2']:.3f} vs {s1['primary_cv_rf']['r2']:.3f}",
            },
            {
                "benchmark": "Edwards jar",
                "finding": "LOGO generalization decomposition",
                "metric": "pooled R² / group-mean R² / within-group R² median (n≥12, SD>1 NTU)",
                "value": " / ".join(
                    f"{s1['primary_cv_rf']['logo_error_distribution'][k]:.3f}"
                    for k in (
                        "r2_pooled_sample_weighted",
                        "r2_group_means_one_point_per_source_water",
                        "r2_within_group_median_n_ge_12_sd_gt_1ntu",
                    )
                    if s1["primary_cv_rf"]["logo_error_distribution"].get(k) is not None
                ),
            },
            {
                "benchmark": "Edwards jar",
                "finding": "Coagulant split sensitivity",
                "metric": "RF R² water+coagulant groups (SI)",
                "value": f"{s1['si_water_plus_coagulant_rf']['r2']:.3f}",
            },
            {
                "benchmark": "NAIADES CoagOpt",
                "finding": "Inverse label degeneracy",
                "metric": "near-zero min_dose / requires_coagulant",
                "value": f"{s2['n_near_zero_min_dose']}/{s2['n_feasible_min_dose']}, stump turb_in F1=1.0",
            },
            {
                "benchmark": "NAIADES Stimela high turb",
                "finding": "Closed-form forward",
                "metric": "max std(turb_flocc/turb_in) within coag_dos",
                "value": f"{s2['stimela_structure']['max_std_ratio_within_dose']:.4f}",
            },
            {
                "benchmark": "NAIADES replica",
                "finding": "Cross-scenario transfer",
                "metric": "normal→extreme / accidental R²",
                "value": " / ".join(
                    f"{x['r2']:.3f}"
                    for x in s3["scenario_transfer_primary"]
                ),
            },
        ]
    )
    audit.to_csv(TABLES_DIR / "table4_benchmark_audit_summary.csv", index=False)

    combined = {
        "paper_framing": "open_benchmark_audit",
        "task1": s1,
        "task2": s2,
        "task3": s3,
    }
    with open(OUTPUT_ROOT / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"\nDone. Figures -> {FIGURES_DIR}")
    print(f"Tables -> {TABLES_DIR}")
    print(f"Processed -> {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
