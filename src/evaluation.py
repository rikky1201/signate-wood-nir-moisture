"""
Cross-validation comparison engine for NIR spectral pipelines.

compare_pipelines  — run GroupKFold CV for multiple configs, return results table
plot_predictions   — predicted vs actual scatter plot
print_summary_table — pretty-print results DataFrame
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _build_pipeline(cfg: dict) -> Pipeline:
    """Build a 3-step Pipeline from a config dict.

    Expected keys: 'preprocessor', 'selector', 'model'
    Each value is cloned so the original objects are not mutated.
    """
    return Pipeline([
        ("prep",  clone(cfg["preprocessor"])),
        ("sel",   clone(cfg["selector"])),
        ("model", clone(cfg["model"])),
    ])


def compare_pipelines(
    configs: list,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    save_dir: str = None,
    filename: str = "results.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """Compare pipeline configs via GroupKFold CV (leak-free).

    Leak prevention: preprocessor.fit, selector.fit, model.fit are all called
    only on the training fold inside each CV split. No test-fold data is ever
    seen during fitting.

    Args:
        configs: list of dicts with keys:
            'name' (str), 'preprocessor', 'selector', 'model'
        X: (n_samples, n_features) spectra array
        y: (n_samples,) target values
        groups: (n_samples,) group IDs for GroupKFold
        n_splits: number of outer CV folds
        save_dir: directory to save results CSV (None = skip)
        filename: CSV filename within save_dir
        verbose: print per-pipeline progress

    Returns:
        DataFrame sorted by CV_RMSE with columns:
          name, CV_RMSE, CV_RMSE_std, CV_R2, RPD, Train_RMSE, Overfit_ratio
    """
    y = np.asarray(y, dtype=float)
    gkf = GroupKFold(n_splits=n_splits)
    sd_y = float(np.std(y))

    rows = []
    for cfg in configs:
        name = cfg["name"]
        if verbose:
            print(f"  [{name}]", end=" ", flush=True)

        fold_val_rmse, fold_train_rmse, fold_r2 = [], [], []

        for tr, va in gkf.split(X, y, groups):
            pipe = _build_pipeline(cfg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X[tr], y[tr])

            pred_val = pipe.predict(X[va])
            pred_tr  = pipe.predict(X[tr])

            fold_val_rmse.append(_rmse(y[va], pred_val))
            fold_train_rmse.append(_rmse(y[tr], pred_tr))
            fold_r2.append(r2_score(y[va], pred_val))

        cv_rmse     = float(np.mean(fold_val_rmse))
        cv_rmse_std = float(np.std(fold_val_rmse))
        cv_r2       = float(np.mean(fold_r2))
        train_rmse  = float(np.mean(fold_train_rmse))
        rpd = sd_y / cv_rmse if cv_rmse > 0 else np.inf

        rows.append({
            "name":           name,
            "CV_RMSE":        cv_rmse,
            "CV_RMSE_std":    cv_rmse_std,
            "CV_R2":          cv_r2,
            "RPD":            rpd,
            "Train_RMSE":     train_rmse,
            "Overfit_ratio":  train_rmse / cv_rmse if cv_rmse > 0 else np.nan,
        })

        if verbose:
            rpd_label = (
                "excellent" if rpd > 3.0
                else "good"  if rpd > 2.5
                else "fair"  if rpd > 2.0
                else "poor"
            )
            print(
                f"RMSE={cv_rmse:.4f}±{cv_rmse_std:.4f}  "
                f"R²={cv_r2:.3f}  RPD={rpd:.2f}({rpd_label})  "
                f"TrainRMSE={train_rmse:.4f}"
            )

    df = pd.DataFrame(rows).sort_values("CV_RMSE").reset_index(drop=True)

    if save_dir:
        out = Path(save_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        if verbose:
            print(f"  → Saved: {out}")

    return df


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "",
    ax=None,
    save_path: str = None,
) -> None:
    """Scatter plot of predicted vs actual with R², RMSE, RPD annotations."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse_val = _rmse(y_true, y_pred)
    r2_val   = r2_score(y_true, y_pred)
    rpd_val  = float(np.std(y_true)) / rmse_val if rmse_val > 0 else np.inf

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, s=10, alpha=0.5)
    lo = min(y_true.min(), y_pred.min()) - 5
    hi = max(y_true.max(), y_pred.max()) + 5
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual MC (%)")
    ax.set_ylabel("Predicted MC (%)")
    ax.set_title(f"{title}\nRMSE={rmse_val:.3f}  R²={r2_val:.3f}  RPD={rpd_val:.2f}")
    ax.legend()

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


def print_summary_table(df: pd.DataFrame, title: str = "") -> None:
    """Pretty-print a comparison results DataFrame."""
    if title:
        width = 62
        print("\n" + "=" * width)
        print(f"  {title}")
        print("=" * width)

    display_cols = ["name", "CV_RMSE", "CV_RMSE_std", "CV_R2", "RPD", "Train_RMSE"]
    display_cols = [c for c in display_cols if c in df.columns]

    fmt = {
        "CV_RMSE":     "{:.4f}".format,
        "CV_RMSE_std": "{:.4f}".format,
        "CV_R2":       "{:.3f}".format,
        "RPD":         "{:.2f}".format,
        "Train_RMSE":  "{:.4f}".format,
    }
    styled = df[display_cols].copy()
    for col, f in fmt.items():
        if col in styled.columns:
            styled[col] = styled[col].apply(f)

    print(styled.to_string(index=True))
    print()
