"""
Regression model registry for NIR spectral moisture prediction.

PLSAutoCV  — PLS with n_components tuned by inner CV (recommended baseline)
get_model  — factory that returns a fresh estimator by name
MODEL_NAMES — list of available model keys
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


class PLSAutoCV(BaseEstimator, RegressorMixin):
    """PLS regression with n_components chosen by inner cross-validation.

    Leak prevention: all inner-CV fitting happens inside fit(), so as long as
    this model is used inside an outer CV fold, no data from the validation
    fold leaks into n_components selection.
    """

    def __init__(self, max_components: int = 30, cv: int = 5):
        self.max_components = max_components
        self.cv = cv

    def fit(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        max_nc = min(self.max_components, X.shape[1], X.shape[0] - 1)
        n_splits = min(self.cv, X.shape[0] // 2)

        best_n, best_rmse = 1, np.inf

        if n_splits >= 2:
            kf = KFold(n_splits=n_splits, shuffle=False)
            for n in range(1, max_nc + 1):
                scores = []
                for tr, va in kf.split(X):
                    nc_fold = min(n, len(tr) - 1)
                    if nc_fold < 1:
                        continue
                    pls = PLSRegression(n_components=nc_fold, scale=False)
                    pls.fit(X[tr], y[tr])
                    pred = pls.predict(X[va]).ravel()
                    scores.append(float(np.sqrt(np.mean((y[va] - pred) ** 2))))
                if scores and np.mean(scores) < best_rmse:
                    best_rmse = np.mean(scores)
                    best_n = n
        else:
            best_n = min(3, max_nc)

        self.best_n_components_ = best_n
        self.pls_ = PLSRegression(n_components=best_n, scale=False)
        self.pls_.fit(X, y)
        return self

    def predict(self, X):
        return self.pls_.predict(X).ravel()

    @property
    def coef_(self):
        return self.pls_.coef_


def get_model(name: str):
    """Return a fresh (unfitted) estimator by name.

    Available keys: 'pls', 'gpr', 'ridge', 'rf', 'gbm'

    GPR note: O(n³) in training — fine for n < 500, very slow beyond that.
    """
    name = name.lower()
    registry = {
        "pls": PLSAutoCV(max_components=30, cv=5),
        "gpr": GaussianProcessRegressor(
            kernel=1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=42,
        ),
        "ridge": Ridge(alpha=1.0),
        "rf": RandomForestRegressor(
            n_estimators=300, max_features=0.3, random_state=42, n_jobs=-1
        ),
        "gbm": GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ),
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {sorted(registry)}")
    return registry[name]


MODEL_NAMES = ["pls", "gpr", "ridge", "rf", "gbm"]
