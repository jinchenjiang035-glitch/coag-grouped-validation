# Grouped validation of open coagulation data

Analysis code and frozen outputs for:

> When does coagulant-dosing machine learning generalise? A grouped-validation case study on open NAIADES and jar-test data

Target journal: IWA *Water Supply*. Random seed **42**. Entry point: `analysis/run_pipeline.py`.

This repository does **not** redistribute the raw NAIADES or Edwards files. Download those from the sources below, then run the scripts.

## Data (do not upload copies here)

- **NAIADES** CoagOpt, Stimela, and replica workbooks: Zenodo [10.5281/zenodo.7870529](https://doi.org/10.5281/zenodo.7870529) (CC BY 4.0).
- **Edwards** jar matrix: [paleolimbot/edwards97](https://github.com/paleolimbot/edwards97) (`data-raw/9X9.XLS`, Using TOC sheet).

Place the unpacked NAIADES bundle and `9X9.XLS` in a sibling folder named `feasibility_check/` (see `analysis/config.py`).

## Reproduce frozen metrics

Use **Python 3.12.7**. Package versions match Methods §3.9:

```text
python -m pip install -r requirements.txt
python analysis/run_pipeline.py
```

## What is in this repository

| Folder | Contents |
|--------|----------|
| `analysis/` | Scripts used for the manuscript |
| `tables/` | Frozen CSV tables (main text and SI) |
| `figures/` | Manuscript Fig. 1–6 and Fig. S1 |
| `outputs/` | Frozen JSON summaries, including `analysis_summary.json` |

## Licence

MIT (code and derived tables/figures). Underlying datasets keep their original licences.
