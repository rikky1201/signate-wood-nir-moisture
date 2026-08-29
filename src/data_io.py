"""Data loading utilities for NIR spectral datasets."""
from dataclasses import dataclass
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from src.utils import load_data, parse_spectra, get_groups


@dataclass
class SpectralDataset:
    """Container for a spectral dataset."""
    X: np.ndarray            # (n_samples, n_wavelengths)
    y: np.ndarray            # (n_samples,)  NaN for test set
    groups: np.ndarray       # (n_samples,)  group IDs for CV
    wavelengths: np.ndarray  # (n_wavelengths,)
    meta: pd.DataFrame       # sample_number, species, etc.


def load_signate(split: str = "train", data_dir: Path = None) -> SpectralDataset:
    """Load SIGNATE wood NIR competition data.

    Args:
        split: 'train' or 'test'
        data_dir: directory containing train.csv / test.csv

    Returns:
        SpectralDataset with X, y (NaN for test), groups, wavelengths, meta
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"

    train_df, test_df = load_data(data_dir=data_dir)
    df = train_df if split == "train" else test_df

    meta, y_series, X, wn = parse_spectra(df)
    y = y_series.values.astype(float) if y_series is not None else np.full(len(df), np.nan)
    groups = get_groups(meta)

    return SpectralDataset(X=X, y=y, groups=groups, wavelengths=wn, meta=meta)


def load_csv(
    path: str,
    y_col: str,
    group_col: str = None,
    wavelength_start: float = None,
    wavelength_end: float = None,
    encoding: str = "utf-8",
) -> SpectralDataset:
    """Generic loader for a CSV where spectral columns have numeric headers.

    Args:
        path: path to CSV file
        y_col: column name of the target variable
        group_col: column name for group ID (None → no leak protection)
        wavelength_start / wavelength_end: optional range filter
        encoding: file encoding

    Returns:
        SpectralDataset
    """
    df = pd.read_csv(path, encoding=encoding)

    spectral_cols = []
    for c in df.columns:
        try:
            float(c)
            spectral_cols.append(c)
        except (ValueError, TypeError):
            pass

    wavelengths = np.array([float(c) for c in spectral_cols])

    if wavelength_start is not None or wavelength_end is not None:
        lo = wavelength_start if wavelength_start is not None else wavelengths.min()
        hi = wavelength_end if wavelength_end is not None else wavelengths.max()
        mask = (wavelengths >= lo) & (wavelengths <= hi)
        spectral_cols = [c for c, m in zip(spectral_cols, mask) if m]
        wavelengths = wavelengths[mask]

    X = df[spectral_cols].values.astype(np.float64)
    y = df[y_col].values.astype(np.float64) if y_col in df.columns else np.full(len(df), np.nan)

    meta_cols = [c for c in df.columns if c not in spectral_cols and c != y_col]
    meta = df[meta_cols]

    if group_col and group_col in df.columns:
        groups = df[group_col].values
    else:
        warnings.warn(
            "No group_col provided; using row indices as groups (no leak protection).",
            stacklevel=2,
        )
        groups = np.arange(len(df))

    return SpectralDataset(X=X, y=y, groups=groups, wavelengths=wavelengths, meta=meta)
