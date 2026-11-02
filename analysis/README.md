# Plan 1 analysis pipeline

**Paper framing:** `open_benchmark_audit` — see `outputs/analysis_summary.json`.

```powershell
$env:MPLBACKEND='Agg'
D:\论文1\feasibility_check\.venv\Scripts\python.exe D:\论文1\paper_coag_open\analysis\run_pipeline.py
```

## Task 1 (Edwards)

- **Main:** pooled R²~0.27; **group-mean R²~0.36**; within-group R² median ~0.08 (`si_edwards_logo_r2_summary.csv`)  
- **Uncertainty:** cluster bootstrap **B=10 000**, seed **42**, percentile CI → `si_edwards_logo_cluster_bootstrap.csv`  
- **Fast re-run:** `rerun_task1_only.py` (add `--with-shap` for Fig.3)  
- **Fig.2:** (a) LOGO parity, (b) per-source MAE histogram  
- **SI:** `si_edwards_feature_ablation_logo.csv` (RF ablation **n=300**; `all_9` row = primary LOGO preds), `si_edwards_logo_mae_by_water_group.csv`  
- **SI:** random 5-fold ~0.90; water+coagulant groups ~0.02 (grouped-structure optimism)  

## Task 2 (NAIADES) — structure, not ML leaderboard

| Source | Main output | Do **not** claim |
|--------|-------------|------------------|
| CoagOpt | `fig4_coagopt_min_dose_vs_turbidity.png` (main); `fig_s_coagopt_min_dose_level_counts.png` (SI); turb lookup table | RF classifier accuracy as contribution |
| Stimela high turb | `fig5_stimela_ratio_by_dose.png`, ratio table | R²=0.999 as learned forward process |

## Task 3 (replica)

- **Main:** `table_replica_scenario_transfer.csv`  
- **Diagnostic:** LOGO + Fig.6  

## Key table for manuscript

`tables/table4_benchmark_audit_summary.csv`
