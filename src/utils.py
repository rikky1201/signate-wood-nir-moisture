"""
Shared utilities: evaluation metrics, cross-validation helpers, and data loading.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# Japanese font setup for matplotlib
# ---------------------------------------------------------------------------
_JP_FONT_CANDIDATES = [
    "Noto Sans JP", "Yu Gothic", "Meiryo", "BIZ UDGothic",
    "MS Gothic", "IPAexGothic", "IPAGothic",
]

def setup_japanese_font() -> str:
    """Configure matplotlib to render Japanese text.

    Tries each candidate font in order and applies the first one found.
    Returns the name of the font that was set, or '' if none worked.

    Call this once at the top of each notebook after importing utils.
    """
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for font in _JP_FONT_CANDIDATES:
        if font in available:
            # Use "sans-serif" family so matplotlib walks the fallback list.
            # The JP font is first, DejaVu Sans provides symbol glyphs (e.g. ⁻¹).
            mpl.rcParams["font.family"] = "sans-serif"
            mpl.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
            return font
    return ""

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: Path = DATA_DIR, encoding: str = "cp932"):
    """Load train and test CSV files.

    Args:
        data_dir: Directory containing train.csv and test.csv.
        encoding: File encoding (default 'cp932' for Japanese Windows CSVs).

    Returns:
        train_df, test_df  (raw DataFrames with all columns intact)
    """
    train = pd.read_csv(data_dir / "train.csv", encoding=encoding)
    test = pd.read_csv(data_dir / "test.csv", encoding=encoding)
    return train, test


def parse_spectra(df: pd.DataFrame):
    """Split a DataFrame into metadata, target, and spectra array.

    Returns:
        meta_cols : DataFrame with non-spectral columns
        y         : Series of moisture content (None for test set)
        X         : numpy array (n_samples, n_wavelengths)
        wavenumbers: numpy array of wavenumber values (cm⁻¹), descending
    """
    # Identify spectral columns: they are numeric column names (wavenumbers)
    spectral_cols = [c for c in df.columns if _is_float(c)]
    meta_cols_names = [c for c in df.columns if c not in spectral_cols]

    X = df[spectral_cols].values.astype(np.float64)
    wavenumbers = np.array([float(c) for c in spectral_cols])

    # Target column is named '含水率' (U+542B U+6C34 U+7387) — moisture content
    MOISTURE_UNICODE = "\u542b\u6c34\u7387"
    moisture_col = [c for c in meta_cols_names if MOISTURE_UNICODE in c or "moisture" in c.lower()]
    if moisture_col:
        y = df[moisture_col[0]]
        meta = df[[c for c in meta_cols_names if c not in moisture_col]]
    else:
        y = None
        meta = df[meta_cols_names]

    return meta, y, X, wavenumbers


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def get_groups(meta: pd.DataFrame) -> np.ndarray:
    """Return group labels for GroupKFold.

    Uses 'species number' so that all measurements of the same tree species
    stay together in a single fold. This tests generalization across species,
    which is the scientifically relevant split for this dataset.
    Falls back to any 'number'-style column if 'species' is not found.
    """
    species_col = [c for c in meta.columns if "species" in c.lower()]
    if species_col:
        return meta[species_col[0]].values
    # fallback: use first integer-like column
    for c in meta.columns:
        if meta[c].dtype in (np.int64, np.int32):
            return meta[c].values
    raise ValueError("Cannot determine group column from meta DataFrame.")


def get_cv_splits(meta: pd.DataFrame, y, X, n_splits: int = 5):
    """Group K-Fold splits by species number.

    All measurements from the same tree species are kept together so that
    validation tests generalization to unseen species (or at least unseen
    specimens of a species).

    Args:
        meta: DataFrame from parse_spectra.
        y: Target array/Series.
        X: Feature array.
        n_splits: Number of folds (should be <= number of unique species).

    Yields:
        (train_idx, val_idx) tuples of integer indices.
    """
    groups = get_groups(meta)
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in gkf.split(X, y, groups):
        yield train_idx, val_idx


def cross_val_rmse(estimator, X, y, groups, n_splits: int = 5) -> tuple[float, float]:
    """Run GroupKFold CV and return (mean_rmse, std_rmse)."""
    from sklearn.base import clone
    scores = []
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in gkf.split(X, y, groups):
        est = clone(estimator)
        est.fit(X[train_idx], np.asarray(y)[train_idx])
        preds = est.predict(X[val_idx])
        scores.append(rmse(np.asarray(y)[val_idx], preds))
    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Submission helper
# ---------------------------------------------------------------------------

def make_submission(test_meta: pd.DataFrame, predictions: np.ndarray,
                    path: str = None) -> pd.DataFrame:
    """Create submission DataFrame matching the sample_submit.csv format.

    The format is header-less CSV with two columns:
      col0 = sample number, col1 = predicted moisture content.

    Args:
        test_meta: Meta DataFrame from parse_spectra on test set.
        predictions: 1-D prediction array aligned to test rows.
        path: If provided, saves CSV to this path.

    Returns:
        Submission DataFrame (without header).
    """
    sample_col = [c for c in test_meta.columns if "sample" in c.lower()]
    if not sample_col:
        raise ValueError("Cannot find 'sample number' column.")
    sub = pd.DataFrame({
        "sample_number": test_meta[sample_col[0]].values,
        "moisture": predictions,
    })
    if path:
        sub.to_csv(path, index=False, header=False)
        print(f"Saved submission: {path}")
    return sub
