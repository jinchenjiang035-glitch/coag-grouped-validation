"""Recompute Edwards LOGO R² summary (group-mean vs within-group) without full pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    EDWARDS_PRIMARY_CV_PROTOCOL,
    OUTPUT_ROOT,
    RANDOM_STATE,
    TABLES_DIR,
)
from data_loaders import load_edwards  # noqa: E402
from run_pipeline import _leave_one_group_cv_predictions, _make_water_groups, _metrics  # noqa: E402


def main() -> None:
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
    ed = load_edwards()
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
    water_groups = _make_water_groups(ed_model)
    g_primary = water_groups[EDWARDS_PRIMARY_CV_PROTOCOL].values

    def rf_factory():
        return RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        )

    y_par = _leave_one_group_cv_predictions(rf_factory, X, y, g_primary)
    m_loo = _metrics(y, y_par)

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
        "r2_pooled_sample_weighted": float(m_loo["r2"]),
        "r2_group_means_one_point_per_source_water": float(
            r2_score(gdf["y_mean"], gdf["yp_mean"])
        ),
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

    for name in ("task1_edwards.json", "analysis_summary.json"):
        path = OUTPUT_ROOT / name
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        root = data if name == "analysis_summary.json" else data
        if name == "analysis_summary.json":
            root = data["task1"]["primary_cv_rf"]
        else:
            root = data["primary_cv_rf"]
        root["logo_error_distribution"] = logo_distribution
        if name == "task1_edwards.json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        else:
            data["task1"]["primary_cv_rf"]["logo_error_distribution"] = logo_distribution
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    print(json.dumps(logo_distribution, indent=2))


if __name__ == "__main__":
    main()
