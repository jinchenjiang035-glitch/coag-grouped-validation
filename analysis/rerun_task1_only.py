"""Re-run Edwards Task 1 only (writes task1_edwards.json; skips Tasks 2-3).

By default skips SHAP figures for speed. Pass --with-shap for Fig.3 outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loaders import load_edwards  # noqa: E402
from run_pipeline import task1_edwards_forward  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run Edwards Task 1 only.")
    parser.add_argument(
        "--with-shap",
        action="store_true",
        help="Also compute SHAP figures (Fig.3).",
    )
    args = parser.parse_args()
    task1_edwards_forward(load_edwards(), skip_shap=not args.with_shap)
    print("Task 1 done -> outputs/task1_edwards.json, tables/si_edwards_*, figures/fig2_*")


if __name__ == "__main__":
    main()
