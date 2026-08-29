"""
Near-infrared spectra preprocessing functions for wood moisture content prediction.

Functional API (backward-compatible):
  snv, msc, savitzky_golay, apply_preprocessing

sklearn Transformer classes (use in Pipeline):
  IdentityTransformer, SNVTransformer, MSCTransformer,
  DetrendTransformer, SavitzkyGolayTransformer

Factory:
  build_preprocessor(spec)  →  transformer or Pipeline
"""
import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline as _Pipeline


def snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate (SNV) correction.

    Subtracts the mean and divides by the std for each sample row.
    Corrects for scatter and light path length variations.
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / std


def msc(X: np.ndarray, reference: np.ndarray = None) -> np.ndarray:
    """Multiplicative Scatter Correction (MSC).

    Fits each spectrum to a reference (mean spectrum by default) via
    linear regression and returns the corrected spectra.

    Args:
        X: (n_samples, n_features) spectra array.
        reference: 1-D reference spectrum. Defaults to mean of X.

    Returns:
        Corrected spectra with same shape as X.
    """
    if reference is None:
        reference = X.mean(axis=0)

    X_msc = np.zeros_like(X)
    for i, spectrum in enumerate(X):
        # fit: spectrum = a * reference + b
        coeffs = np.polyfit(reference, spectrum, deg=1)
        a, b = coeffs
        X_msc[i] = (spectrum - b) / a
    return X_msc


def savitzky_golay(
    X: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
    deriv: int = 1,
    delta: float = 1.0,
) -> np.ndarray:
    """Savitzky-Golay smoothing / derivative filter.

    Args:
        X: (n_samples, n_features) spectra array.
        window_length: Length of the filter window (must be odd).
        polyorder: Polynomial order for fitting within the window.
        deriv: Derivative order. 0 = smoothing only, 1 = 1st derivative.
        delta: Spacing of the data points (wavenumber step in cm⁻¹).

    Returns:
        Filtered spectra with same shape as X.
    """
    return savgol_filter(X, window_length=window_length, polyorder=polyorder,
                         deriv=deriv, delta=delta, axis=1)


def apply_preprocessing(X: np.ndarray, method: str = "snv") -> np.ndarray:
    """Convenience wrapper to apply a named preprocessing method.

    Args:
        X: Raw spectra array (n_samples, n_wavelengths).
        method: One of "raw", "snv", "msc", "sg1" (1st deriv), "sg2" (2nd deriv),
                "snv+sg1", "snv+sg2".

    Returns:
        Preprocessed spectra.
    """
    method = method.lower()

    if method == "raw":
        return X
    elif method == "snv":
        return snv(X)
    elif method == "msc":
        return msc(X)
    elif method == "sg1":
        return savitzky_golay(X, deriv=1)
    elif method == "sg2":
        return savitzky_golay(X, deriv=2)
    elif method == "snv+sg1":
        return savitzky_golay(snv(X), deriv=1)
    elif method == "snv+sg2":
        return savitzky_golay(snv(X), deriv=2)
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")


# ─── sklearn-compatible Transformer classes ──────────────────────────────────

class IdentityTransformer(BaseEstimator, TransformerMixin):
    """Pass-through: returns X unchanged. Use as no-op preprocessing step."""

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X


class SNVTransformer(BaseEstimator, TransformerMixin):
    """Standard Normal Variate (SNV) correction."""

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return snv(X)


class MSCTransformer(BaseEstimator, TransformerMixin):
    """Multiplicative Scatter Correction (MSC).

    Leak prevention: fit() stores the reference spectrum computed only from
    the training fold. transform() uses that stored reference.
    """

    def fit(self, X, y=None):
        self.reference_ = X.mean(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return msc(X, reference=self.reference_)


class DetrendTransformer(BaseEstimator, TransformerMixin):
    """Polynomial detrend along the wavelength axis."""

    def __init__(self, deg: int = 1):
        self.deg = deg

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        idx = np.arange(X.shape[1])
        out = np.empty_like(X)
        for i, row in enumerate(X):
            p = np.polyfit(idx, row, self.deg)
            out[i] = row - np.polyval(p, idx)
        return out


class SavitzkyGolayTransformer(BaseEstimator, TransformerMixin):
    """Savitzky-Golay smoothing / derivative filter."""

    def __init__(
        self,
        window_length: int = 11,
        polyorder: int = 2,
        deriv: int = 1,
        delta: float = 1.0,
    ):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return savitzky_golay(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            delta=self.delta,
        )


# ─── Factory ─────────────────────────────────────────────────────────────────

_STEP_BUILDERS = {
    "snv":     lambda: ("snv",     SNVTransformer()),
    "msc":     lambda: ("msc",     MSCTransformer()),
    "detrend": lambda: ("detrend", DetrendTransformer()),
    "sg1":     lambda: ("sg1",     SavitzkyGolayTransformer(deriv=1)),
    "sg2":     lambda: ("sg2",     SavitzkyGolayTransformer(deriv=2)),
}


def build_preprocessor(spec: str):
    """Return a sklearn transformer (or Pipeline) for a preprocessing spec string.

    Spec examples:
      'raw'      → IdentityTransformer
      'snv'      → SNVTransformer
      'sg2'      → SavitzkyGolayTransformer(deriv=2)
      'snv+sg1'  → Pipeline([SNVTransformer, SavitzkyGolayTransformer(deriv=1)])
      'msc+sg2'  → Pipeline([MSCTransformer, SavitzkyGolayTransformer(deriv=2)])
      'detrend+sg1' → Pipeline([DetrendTransformer, SavitzkyGolayTransformer(deriv=1)])

    Steps are applied left-to-right.
    """
    if spec.lower() == "raw":
        return IdentityTransformer()

    parts = [p.strip() for p in spec.lower().split("+")]
    steps = []
    seen: dict[str, int] = {}
    for part in parts:
        if part not in _STEP_BUILDERS:
            raise ValueError(
                f"Unknown preprocessing step '{part}'. "
                f"Valid steps: {list(_STEP_BUILDERS)} or 'raw'."
            )
        name, obj = _STEP_BUILDERS[part]()
        count = seen.get(name, 0)
        seen[name] = count + 1
        steps.append((f"{name}_{count}" if count else name, obj))

    if len(steps) == 1:
        return steps[0][1]
    return _Pipeline(steps)
