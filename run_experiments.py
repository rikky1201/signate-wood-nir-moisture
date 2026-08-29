#!/usr/bin/env python
"""
Experiment protocol — NIR wood moisture prediction.
Follows the 4-step design document.

Usage:
    python run_experiments.py              # run all steps
    python run_experiments.py --step 2    # run only step 2
    python run_experiments.py --n-splits 5

Steps:
  1. Sanity check  : raw + NoSelection + PLS
  2. Preprocessing : compare all preprocessors (NoSelection + PLS fixed)
  3. Feature sel.  : compare NoSelection vs CARS (best prep + PLS fixed)
  4. Model         : compare PLS / Ridge / RF / GBM (best prep + best selector)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data_io import load_signate
from src.preprocessing import build_preprocessor
from src.feature_selection import NoSelection, CARSSelector
from src.models import get_model
from src.evaluation import compare_pipelines, print_summary_table

RESULTS_DIR = Path("results")

PREPROCESSING_SPECS = [
    "raw", "snv", "msc", "detrend",
    "sg1", "sg2",
    "snv+sg1", "snv+sg2",
    "msc+sg2", "detrend+sg1",
]


# ─── Step runners ────────────────────────────────────────────────────────────

def step1_sanity(X, y, groups, n_splits):
    print("\n" + "=" * 62)
    print("STEP 1  Sanity check — raw + NoSelection + PLS")
    print("=" * 62)

    configs = [{
        "name":         "raw+NoSel+PLS",
        "preprocessor": build_preprocessor("raw"),
        "selector":     NoSelection(),
        "model":        get_model("pls"),
    }]
    df = compare_pipelines(
        configs, X, y, groups, n_splits=n_splits,
        save_dir=RESULTS_DIR, filename="step1_sanity.csv",
    )
    print_summary_table(df, "Step 1 result")
    return df


def step2_preprocessing(X, y, groups, n_splits):
    print("\n" + "=" * 62)
    print("STEP 2  Preprocessing comparison  (NoSelection + PLS fixed)")
    print("=" * 62)

    configs = [
        {
            "name":         spec,
            "preprocessor": build_preprocessor(spec),
            "selector":     NoSelection(),
            "model":        get_model("pls"),
        }
        for spec in PREPROCESSING_SPECS
    ]
    df = compare_pipelines(
        configs, X, y, groups, n_splits=n_splits,
        save_dir=RESULTS_DIR, filename="step2_preprocessing.csv",
    )
    print_summary_table(df, "Step 2: Preprocessing comparison")

    best_prep = str(df.iloc[0]["name"])
    print(f"  → Best preprocessing: '{best_prep}'")
    return df, best_prep


def step3_feature_selection(X, y, groups, n_splits, best_prep):
    print("\n" + "=" * 62)
    print(f"STEP 3  Feature selection  (prep='{best_prep}' + PLS fixed)")
    print("=" * 62)

    sel_options = [
        ("NoSelection", NoSelection()),
        ("CARS",        CARSSelector(n_components=10, n_runs=50,
                                     cv=5, random_state=42, verbose=False)),
    ]

    configs = [
        {
            "name":         f"{best_prep}+{sel_name}+PLS",
            "preprocessor": build_preprocessor(best_prep),
            "selector":     sel_obj,
            "model":        get_model("pls"),
        }
        for sel_name, sel_obj in sel_options
    ]
    df = compare_pipelines(
        configs, X, y, groups, n_splits=n_splits,
        save_dir=RESULTS_DIR, filename="step3_feature_selection.csv",
    )
    print_summary_table(df, "Step 3: Feature selection comparison")

    best_row_name = str(df.iloc[0]["name"])
    name_to_sel = {
        f"{best_prep}+{n}+PLS": (n, s) for n, s in sel_options
    }
    best_sel_name, best_sel = name_to_sel[best_row_name]
    print(f"  → Best selector: '{best_sel_name}'")
    return df, best_sel_name, best_sel


def step4_models(X, y, groups, n_splits, best_prep, best_sel_name, best_sel):
    print("\n" + "=" * 62)
    print(f"STEP 4  Model comparison  "
          f"(prep='{best_prep}' + sel='{best_sel_name}')")
    print("=" * 62)

    model_keys = ["pls", "ridge", "rf", "gbm"]
    if len(y) <= 500:
        model_keys.append("gpr")
        print("  [GPR included: n_samples ≤ 500]")
    else:
        print(f"  [GPR skipped: n_samples={len(y)} > 500, O(n³) too slow]")

    configs = [
        {
            "name":         f"{best_prep}+{best_sel_name}+{mkey.upper()}",
            "preprocessor": build_preprocessor(best_prep),
            "selector":     best_sel,
            "model":        get_model(mkey),
        }
        for mkey in model_keys
    ]
    df = compare_pipelines(
        configs, X, y, groups, n_splits=n_splits,
        save_dir=RESULTS_DIR, filename="step4_models.csv",
    )
    print_summary_table(df, "Step 4: Model comparison")
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NIR wood moisture — 4-step experiment protocol"
    )
    parser.add_argument(
        "--step", default="all",
        choices=["1", "2", "3", "4", "all"],
        help="Which step(s) to run (default: all)",
    )
    parser.add_argument(
        "--n-splits", type=int, default=5,
        help="Number of GroupKFold splits (default: 5)",
    )
    args = parser.parse_args()

    print("Loading data...")
    ds = load_signate(split="train")
    X, y, groups = ds.X, ds.y, ds.groups
    print(
        f"  X: {X.shape}  "
        f"y: [{y.min():.1f}, {y.max():.1f}]  "
        f"n_species: {len(np.unique(groups))}"
    )

    RESULTS_DIR.mkdir(exist_ok=True)

    # Defaults used when skipping earlier steps
    best_prep     = "snv"
    best_sel_name = "NoSelection"
    best_sel      = NoSelection()

    step = args.step

    if step in ("1", "all"):
        step1_sanity(X, y, groups, args.n_splits)

    if step in ("2", "all"):
        _, best_prep = step2_preprocessing(X, y, groups, args.n_splits)

    if step in ("3", "all"):
        _, best_sel_name, best_sel = step3_feature_selection(
            X, y, groups, args.n_splits, best_prep
        )

    if step in ("4", "all"):
        step4_models(
            X, y, groups, args.n_splits,
            best_prep, best_sel_name, best_sel
        )

    print(f"\n=== Done. Results saved to '{RESULTS_DIR}/' ===")


if __name__ == "__main__":
    main()
