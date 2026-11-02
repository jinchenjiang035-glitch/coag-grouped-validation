"""Load and clean Edwards jar and NAIADES datasets."""

from __future__ import annotations

import pandas as pd

from config import (
    EDWARDS_XLS,
    NAIADES_ROOT,
    REPLICA_CACHE,
    STIMELA_HIGH_TURB_XLS,
    TURB_FLOC_THRESHOLD,
)

# Edwards 9X9.XLS column map (Using TOC sheet, 0-based after skiprows=10 data rows)
# Settled water quality block: col 21 = Turb (NTU), 22 = TOC (mg/L), 23 = UV-254 (1/cm)
# Raw water: col 7 = TOC, col 8 = DOC, col 9 = UV-254


def load_edwards() -> pd.DataFrame:
    """Edwards 1997 jar matrix (Using TOC sheet)."""
    raw = pd.read_excel(
        EDWARDS_XLS,
        sheet_name="Using TOC",
        header=None,
        skiprows=10,
        engine="xlrd",
    ).iloc[1:]
    raw = raw.replace(["--", "", "NA", " "], None)

    doc_mg_l = pd.to_numeric(raw[8], errors="coerce")
    toc_in_mg_l = pd.to_numeric(raw[7], errors="coerce")
    doc_mg_l = doc_mg_l.fillna(toc_in_mg_l)

    df = pd.DataFrame(
        {
            "coagulant_code": pd.to_numeric(raw[2], errors="coerce"),
            "dose_mg_l": pd.to_numeric(raw[14], errors="coerce"),
            "dose_mmol_l": pd.to_numeric(raw[15], errors="coerce"),
            "toc_in_mg_l": toc_in_mg_l,
            "doc_mg_l": doc_mg_l,
            "uv254_in": pd.to_numeric(raw[9], errors="coerce"),
            "alkalinity_mg_caco3": pd.to_numeric(raw[10], errors="coerce"),
            "turbidity_ntu": pd.to_numeric(raw[11], errors="coerce"),
            "ph": pd.to_numeric(raw[12], errors="coerce"),
            "settle_time_min": pd.to_numeric(raw[19], errors="coerce"),
            "settled_turbidity_ntu": pd.to_numeric(raw[21], errors="coerce"),
            "settled_toc_mg_l": pd.to_numeric(raw[22], errors="coerce"),
            "settled_uv254": pd.to_numeric(raw[23], errors="coerce"),
        }
    )
    df = df.dropna(
        subset=[
            "dose_mmol_l",
            "ph",
            "turbidity_ntu",
            "settled_turbidity_ntu",
            "coagulant_code",
        ]
    )
    df["coagulant_code"] = df["coagulant_code"].astype(int)
    return df.reset_index(drop=True)


def load_coagulation_optimization() -> pd.DataFrame:
    path = NAIADES_ROOT / "CoagulationOptimization.xlsx"
    co = pd.read_excel(path, engine="openpyxl")
    co.columns = ["turb_in", "pH", "conduct", "dose", "turb_floc"]
    co["inlet_key"] = (
        co.turb_in.round(4).astype(str)
        + "_"
        + co.pH.round(4).astype(str)
        + "_"
        + co.conduct.round(2).astype(str)
    )
    return co


def extract_min_doses(
    co: pd.DataFrame,
    threshold: float = TURB_FLOC_THRESHOLD,
    min_positive_dose: float = 0.01,
) -> pd.DataFrame:
    """Minimum dose meeting turbidity target; flags near-zero grid doses."""
    feasible = co[co["turb_floc"] <= threshold]
    opt = feasible.groupby("inlet_key", as_index=False).agg(
        min_dose=("dose", "min"),
        turb_in=("turb_in", "first"),
        pH=("pH", "first"),
        conduct=("conduct", "first"),
        min_turb_floc=("turb_floc", "min"),
    )
    opt["min_dose_class"] = opt["min_dose"].round(4)
    opt["requires_coagulant"] = opt["min_dose"] >= min_positive_dose
    return opt


def load_stimela_high_turbidity() -> pd.DataFrame:
    df = pd.read_excel(STIMELA_HIGH_TURB_XLS, engine="openpyxl")
    df.columns = [
        "turb_in",
        "ph",
        "conduct",
        "coag_dos",
        "cl_dosage",
        "turb_flocc",
        "turb_out",
        "cl_residual",
    ]
    return df.dropna(subset=["turb_in", "coag_dos", "turb_flocc"])


def load_replica_combined() -> pd.DataFrame:
    if REPLICA_CACHE.exists():
        return pd.read_parquet(REPLICA_CACHE)

    rows = []
    for name in [
        "normalOperation.xlsx",
        "extremeEvents.xlsx",
        "accidentalEvents.xlsx",
    ]:
        p = NAIADES_ROOT / "dWTP replica's Data for Optimization" / name
        xl = pd.ExcelFile(p, engine="openpyxl")
        scenario = name.replace(".xlsx", "")
        for sn in xl.sheet_names:
            if not (sn.startswith("pico") or sn.endswith("_datos")):
                continue
            sheet_df = pd.read_excel(p, sheet_name=sn, engine="openpyxl")
            if sheet_df.shape[1] < 11:
                continue
            sub = sheet_df.iloc[:, 1:11].copy()
            sub.columns = [
                "turbidity_in",
                "conductivity_in",
                "ph_in",
                "temperature_in",
                "turbidity_out",
                "chlorine_out",
                "temperature_out",
                "filtrationTime",
                "coagulantDosage",
                "clorineDosage",
            ]
            sub["scenario"] = scenario
            sub["sheet"] = sn
            sub["group"] = f"{scenario}_{sn}"
            rows.append(sub)
    allr = pd.concat(rows, ignore_index=True)
    allr.to_parquet(REPLICA_CACHE, index=False)
    return allr
