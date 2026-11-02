from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATA_ROOT = REPO_ROOT / "feasibility_check"
NAIADES_ROOT = (
    DATA_ROOT
    / "naiades_unzip"
    / "DynamicalTreatmentsSuggestionsModellingDataset"
)
EDWARDS_XLS = DATA_ROOT / "9X9.XLS"
REPLICA_CACHE = DATA_ROOT / "replica_combined.parquet"
STIMELA_HIGH_TURB_XLS = NAIADES_ROOT / "Stimela_Simulations_HighTurbidity.xlsx"

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"
PROCESSED_DIR = OUTPUT_ROOT / "processed"

RANDOM_STATE = 42
N_FOLDS = 5
# Edwards Task 1 primary LOGO + ablation (must match)
EDWARDS_RF_N_ESTIMATORS = 300
# Cluster bootstrap on LOGO hold-out predictions (resample source-water groups, no retrain)
LOGO_CLUSTER_BOOTSTRAP_B = 10000
LOGO_CLUSTER_BOOTSTRAP_CI = 0.95
TURB_FLOC_THRESHOLD = 1.0  # NTU

# --- Edwards grouped CV (hardcoded; do not change without re-running pipeline) ---
# Primary: same source water = 6 inlet-quality columns (NO coagulant, dose, or time).
EDWARDS_SOURCE_WATER_COLS = [
    "turbidity_ntu",
    "ph",
    "doc_mg_l",
    "toc_in_mg_l",
    "uv254_in",
    "alkalinity_mg_caco3",
]
EDWARDS_SOURCE_WATER_ROUND = {
    "turbidity_ntu": 2,
    "ph": 2,
    "doc_mg_l": 3,
    "toc_in_mg_l": 3,
    "uv254_in": 4,
    "alkalinity_mg_caco3": 1,
}
# Main text / Fig.2 / JSON primary metric
EDWARDS_PRIMARY_CV_PROTOCOL = "leave_one_source_water_6col"
# SI sensitivity only (inflated group split)
EDWARDS_SI_CV_PROTOCOL = "leave_one_source_water_6col_plus_coagulant"
