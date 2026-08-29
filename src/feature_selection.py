"""
Feature wavelength selection methods for NIR spectroscopy.
All selectors are sklearn-compatible (BaseEstimator + SelectorMixin).

Available:
  NoSelection    — full-spectrum baseline (always compare against this)
  CARSSelector   — Competitive Adaptive Reweighted Sampling (Li et al. 2009)

Stubs (interface only, raise NotImplementedError):
  SPASelector, RandomFrogSelector, MRFSelector
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import KFold


# ─── Full-spectrum baseline ──────────────────────────────────────────────────

class NoSelection(BaseEstimator, SelectorMixin):
    """Pass-through: keeps all features (full-spectrum baseline).

    Design note: always include this as a comparison.
    Li et al. (2009) found full-spectrum beat CARS/SPA for some models.
    """

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def _get_support_mask(self):
        return np.ones(self.n_features_in_, dtype=bool)


# ─── CARS ────────────────────────────────────────────────────────────────────

class CARSSelector(BaseEstimator, SelectorMixin):
    """Competitive Adaptive Reweighted Sampling (CARS).

    Li, H. et al. (2009), Analytica Chimica Acta, 648(1), 77–84.

    Algorithm:
      1. Exponential decay determines how many features to keep at run i:
             n_keep_i = round(p × exp(−k × i)),  k = ln(p/2) / (n_runs − 1)
         so the path goes from p features (i=0) → 2 features (i=n_runs−1).
      2. At each step, fit PLS on the current selection; use |coeff| as weights.
      3. Stochastic ARS: sample n_keep_i features with probability ∝ |coeff|.
      4. Compute inner-CV RMSE for the current selection.
      5. Return the selection that achieved minimum inner-CV RMSE.

    Leak prevention: all fitting (PLS + inner CV) uses only training-fold data,
    because CARSSelector.fit() is called inside the outer CV fold.
    """

    def __init__(
        self,
        n_components: int = 10,
        n_runs: int = 50,
        cv: int = 5,
        random_state=None,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.n_runs = n_runs
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        y = np.asarray(y, dtype=float)

        k = np.log(n_features / 2) / max(self.n_runs - 1, 1)

        selected = np.arange(n_features)
        best_cv_rmse = np.inf
        best_selected = selected.copy()

        for i in range(self.n_runs):
            n_keep = max(2, int(round(n_features * np.exp(-k * i))))

            if n_keep < len(selected) and len(selected) > 2:
                nc = min(self.n_components, len(selected) - 1, n_samples // 3)
                if nc < 1:
                    break

                pls = PLSRegression(n_components=nc, scale=False)
                pls.fit(X[:, selected], y)
                weights = np.abs(pls.coef_.ravel())

                w_sum = weights.sum()
                if w_sum == 0:
                    break
                probs = weights / w_sum

                n_sample = min(n_keep, len(selected))
                chosen = rng.choice(len(selected), size=n_sample, replace=False, p=probs)
                selected = np.sort(selected[chosen])

            if len(selected) < 2:
                break

            cv_rmse = self._inner_cv_rmse(X[:, selected], y)
            if cv_rmse < best_cv_rmse:
                best_cv_rmse = cv_rmse
                best_selected = selected.copy()

            if self.verbose:
                print(
                    f"  CARS {i+1:3d}/{self.n_runs}: "
                    f"n_feat={len(selected):4d}  inner_CV_RMSE={cv_rmse:.4f}"
                )

        self.n_features_in_ = n_features
        self.support_mask_ = np.zeros(n_features, dtype=bool)
        self.support_mask_[best_selected] = True
        self.best_n_features_ = int(len(best_selected))
        self.best_cv_rmse_ = float(best_cv_rmse)
        return self

    def _inner_cv_rmse(self, X, y) -> float:
        nc = min(self.n_components, X.shape[1] - 1, X.shape[0] // 3)
        if nc < 1:
            return np.inf
        n_splits = min(self.cv, X.shape[0])
        kf = KFold(n_splits=n_splits, shuffle=False)
        scores = []
        for tr, va in kf.split(X):
            nc_fold = min(nc, len(tr) - 1)
            if nc_fold < 1:
                continue
            pls = PLSRegression(n_components=nc_fold, scale=False)
            pls.fit(X[tr], y[tr])
            pred = pls.predict(X[va]).ravel()
            scores.append(float(np.sqrt(np.mean((y[va] - pred) ** 2))))
        return float(np.mean(scores)) if scores else np.inf

    def _get_support_mask(self):
        return self.support_mask_


# ─── Stubs (interface only) ──────────────────────────────────────────────────

class SPASelector(BaseEstimator, SelectorMixin):
    """Successive Projections Algorithm (SPA) — NOT IMPLEMENTED.

    Reference: Araújo et al. (2001), Chemom. Intell. Lab. Syst., 57(2), 65–73.
    """

    def __init__(self, max_features: int = 50):
        self.max_features = max_features

    def fit(self, X, y=None):
        raise NotImplementedError(
            "SPASelector is not yet implemented. "
            "See https://doi.org/10.1016/j.chemolab.2004.10.003"
        )

    def _get_support_mask(self):
        raise NotImplementedError


class RandomFrogSelector(BaseEstimator, SelectorMixin):
    """Random Frog (RF) wavelength selector — NOT IMPLEMENTED.

    Reference: Li, H. et al. (2012), Chemom. Intell. Lab. Syst., 111(1), 15–22.
    """

    def __init__(self, n_iterations: int = 10_000, threshold: float = 0.1):
        self.n_iterations = n_iterations
        self.threshold = threshold

    def fit(self, X, y=None):
        raise NotImplementedError(
            "RandomFrogSelector is not yet implemented. "
            "See https://doi.org/10.1016/j.chemolab.2012.03.009"
        )

    def _get_support_mask(self):
        raise NotImplementedError


class MRFSelector(BaseEstimator, SelectorMixin):
    """Modified Random Frog with CARS-style exponential decay — NOT IMPLEMENTED."""

    def __init__(
        self,
        n_components: int = 10,
        n_iterations: int = 10_000,
        threshold: float = 0.1,
    ):
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.threshold = threshold

    def fit(self, X, y=None):
        raise NotImplementedError("MRFSelector is not yet implemented.")

    def _get_support_mask(self):
        raise NotImplementedError
