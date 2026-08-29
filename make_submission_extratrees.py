"""
make_submission_extratrees.py
Best model: ExtraTrees + fold-internal wavenumber selection (sign>=0.8, std<=0.05)
CV-RMSE=17.03%  (GroupKFold, 元スケール, baseline 20.19% vs)

Final model training:
- Compute sign_consistency / r_std on ALL 13 training species
- Select wavenumbers: sign_cons>=0.8 AND r_std<=0.05
- Preprocess: SNV+SG1
- Train ExtraTrees on all training data
- Predict on test data
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from preprocessing import snv, savitzky_golay

# ---- Settings --------------------------------------------------------
SIGN_THRESH = 0.8
STD_THRESH  = 0.05
SEED        = 42
ET_KW       = dict(n_estimators=300, max_features=0.3, random_state=SEED, n_jobs=-1)

# ---- Load data -------------------------------------------------------
print("Loading data...")
df_tr = pd.read_csv('data/train.csv', encoding='cp932')
df_te = pd.read_csv('data/test.csv',  encoding='cp932')

META_TR = ['sample number', 'species number', '樹種', '含水率']
META_TE = ['sample number', 'species number', '樹種']

wn     = df_tr.columns.drop(META_TR).astype(float).values
X_tr   = df_tr.drop(columns=META_TR).values.astype(float)
y_tr   = df_tr['含水率'].values.astype(float)
g_tr   = df_tr['species number'].values

X_te   = df_te.drop(columns=META_TE).values.astype(float)
sn_te  = df_te['sample number'].values

print(f"  Train: {X_tr.shape}  y: {y_tr.min():.1f}-{y_tr.max():.1f}%")
print(f"  Test:  {X_te.shape}")
print(f"  Train species: {sorted(set(g_tr))}")
print(f"  Test  species: {sorted(set(df_te['species number']))}")

# ---- Preprocessing ---------------------------------------------------
def preproc(X):
    return savitzky_golay(snv(X), window_length=11, polyorder=2, deriv=1)

print("\nApplying SNV+SG1...")
X_tr_pp = preproc(X_tr)
X_te_pp = preproc(X_te)

# ---- Wavenumber selection (computed on ALL training data) -----------
print("Computing wavenumber selection on all training species...")

def corr_vec(X, yv):
    yc = yv - yv.mean(); Xc = X - X.mean(axis=0)
    num = (Xc * yc[:, None]).sum(0)
    den = np.sqrt((Xc**2).sum(0) * (yc**2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, 0.0)

train_species = sorted(set(g_tr))
r_list = []
for sp in train_species:
    mask = g_tr == sp
    if mask.sum() < 3:
        continue
    r_list.append(corr_vec(X_tr_pp[mask], y_tr[mask]))

r_mat      = np.array(r_list)            # (13, 1555)
sign_cons  = np.abs(np.sign(r_mat).sum(0)) / len(r_list)
r_std_arr  = r_mat.std(0)

sel = (sign_cons >= SIGN_THRESH) & (r_std_arr <= STD_THRESH)
n_sel = int(sel.sum())
print(f"  Selected wavenumbers: {n_sel} / {len(wn)}")
print(f"  4760 cm^-1 included: {sel[np.argmin(np.abs(wn - 4760))]}")
print(f"  5200 cm^-1 included: {sel[np.argmin(np.abs(wn - 5200))]}")
print(f"  6900 cm^-1 included: {sel[np.argmin(np.abs(wn - 6900))]}")

X_tr_sel = X_tr_pp[:, sel]
X_te_sel = X_te_pp[:, sel]

# ---- Train ExtraTrees on all training data ---------------------------
print(f"\nTraining ExtraTrees (n_estimators=300, max_features=0.3, n_feat={n_sel})...")
model = ExtraTreesRegressor(**ET_KW)
model.fit(X_tr_sel, y_tr)

# Check in-bag train R^2 (sanity)
pred_tr = model.predict(X_tr_sel)
ss_res  = ((y_tr - pred_tr)**2).sum()
ss_tot  = ((y_tr - y_tr.mean())**2).sum()
print(f"  Train R^2 (in-bag): {1 - ss_res/ss_tot:.4f}")
print(f"  Train RMSE (in-bag): {np.sqrt(np.mean((y_tr - pred_tr)**2)):.2f}%")

# ---- Predict on test -------------------------------------------------
print("\nPredicting on test data...")
pred_te = model.predict(X_te_sel)

print(f"  Predictions: min={pred_te.min():.2f}  max={pred_te.max():.2f}  mean={pred_te.mean():.2f}")
print(f"  Negative predictions: {(pred_te < 0).sum()} (clipped to 0)")

# Clip negative predictions (moisture content cannot be negative)
pred_te = np.clip(pred_te, 0, None)

# ---- Save submission -------------------------------------------------
sub = pd.DataFrame({'sample_number': sn_te, 'moisture': pred_te})
out_path = 'submissions/extratrees_sign08_std005.csv'
os.makedirs('submissions', exist_ok=True)
sub.to_csv(out_path, index=False, header=False)

print(f"\nSaved: {out_path}  ({len(sub)} rows)")
print("First 5 rows:")
print(sub.head().to_string(index=False))
