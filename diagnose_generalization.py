"""
diagnose_generalization.py
NIR含水率予測 -- 未知樹種への汎化可能性を診断する

論点A: 含水率の分布と対数変換の必要性
論点B: 相関スペクトルの再診断
論点C: 樹種をまたぐ普遍性
論点D: ベースライン3種の正直なCVスコア
論点E: バンド比特徴の素振り
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler

from preprocessing import snv, savitzky_golay
from utils import setup_japanese_font, rmse

# ─── Setup ───────────────────────────────────────────────────────────────────
SEED = 42
N_SPLITS = 5
np.random.seed(SEED)
setup_japanese_font()
os.makedirs("results", exist_ok=True)

WATER_BANDS = {
    "O-H 1st (6900)": 6900,
    "H2O comb (5200)": 5200,
    "O-H comb (4760)": 4760,
}
BAND_COLORS = ["#d62728", "#ff7f0e", "#9467bd"]

# ─── Data loading ─────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/train.csv", encoding="cp932")
META = ["sample number", "species number", "樹種", "含水率"]
wn = df.columns.drop(META).astype(float).values          # (1555,)
X_raw = df.drop(columns=META).values.astype(float)       # (1322, 1555)
y = df["含水率"].values.astype(float)
groups = df["species number"].values
sp_name = df.drop_duplicates("species number").set_index("species number")["樹種"].to_dict()
train_sps = sorted(df["species number"].unique())

print(f"  X: {X_raw.shape}  y: {y.min():.1f}-{y.max():.1f}%")
print(f"  訓練樹種: {train_sps}")

# Precomputed preprocessed arrays (SNV/SG1 have no fit params → no leakage risk)
X_snv     = snv(X_raw)
X_snv_sg1 = savitzky_golay(snv(X_raw), window_length=11, polyorder=2, deriv=1)

# Shared GroupKFold splits (fixed for fair cross-model comparison)
gkf = GroupKFold(n_splits=N_SPLITS)
SPLITS = list(gkf.split(X_raw, y, groups))

fold_sp_labels = []
for _, va in SPLITS:
    fold_sp_labels.append(sorted(set(groups[va])))

print("\nFold別の検証樹種:")
for i, sps in enumerate(fold_sp_labels):
    print(f"  Fold{i+1}: species {sps}")

# ─── Helper functions ─────────────────────────────────────────────────────────
def corr_spectrum(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r between each wavenumber column and y."""
    y_c = y - y.mean()
    X_c = X - X.mean(axis=0)
    num   = (X_c * y_c[:, None]).sum(axis=0)
    denom = np.sqrt((X_c**2).sum(axis=0) * (y_c**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, num / denom, 0.0)

def add_water_bands(ax, with_legend=False):
    for (name, bwn), color in zip(WATER_BANDS.items(), BAND_COLORS):
        ax.axvline(bwn, color=color, lw=1.5, ls="--", alpha=0.85,
                   label=name if with_legend else "_nolegend_")

def band_mean(X: np.ndarray, center: float, half_width: float = 75.0) -> np.ndarray:
    mask = (wn >= center - half_width) & (wn <= center + half_width)
    return X[:, mask].mean(axis=1)

def _cv_run(model_fn, X_pre, y_arr, log_target: bool, scaler_needed=False):
    """Run GroupKFold CV, return list of per-fold RMSE."""
    fold_rmses = []
    for tr, va in SPLITS:
        X_tr, X_va = X_pre[tr], X_pre[va]
        y_tr, y_va = y_arr[tr], y_arr[va]
        if scaler_needed:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_va = sc.transform(X_va)
        y_fit = np.log1p(y_tr) if log_target else y_tr
        m = model_fn()
        m.fit(X_tr, y_fit)
        pred = m.predict(X_va)
        if hasattr(pred, "ravel"):
            pred = pred.ravel()
        if log_target:
            pred = np.expm1(np.clip(pred, -10, 20))
        fold_rmses.append(rmse(y_va, pred))
    return fold_rmses

# ===============================================================================
# 論点A: 含水率の分布と対数変換の必要性
# ===============================================================================
print("\n" + "="*65)
print("論点A: 含水率の分布と対数変換の必要性")
print("="*65)

# A1: ヒストグラム (線形 vs log1p)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("[A1] 含水率の分布", fontsize=13)

axes[0].hist(y, bins=50, color="steelblue", edgecolor="white", lw=0.4)
axes[0].set_xlabel("含水率 (%)")
axes[0].set_ylabel("サンプル数")
axes[0].set_title("線形スケール")

axes[1].hist(np.log1p(y), bins=50, color="coral", edgecolor="white", lw=0.4)
axes[1].set_xlabel("log1p(含水率)")
axes[1].set_ylabel("サンプル数")
axes[1].set_title("log1p変換後 (分布の歪みを確認)")

plt.tight_layout()
plt.savefig("results/A1_moisture_histogram.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: results/A1_moisture_histogram.png")

# A2: 樹種別統計表
sp_stats = df.groupby("species number")["含水率"].agg(["min", "max", "mean", "std", "count"]).round(1)
sp_stats.columns = ["min", "max", "mean", "std", "n"]
sp_stats.insert(0, "樹種", sp_stats.index.map(sp_name))
print("\n[A] 樹種別含水率統計:")
print(sp_stats.to_string())

# A3: 高含水サンプル (>100%) の樹種別集計
high_mc = df[df["含水率"] > 100]
print(f"\n[A] 含水率 >100% サンプル: {len(high_mc)} 件 ({100*len(high_mc)/len(df):.1f}%)")
hmc = high_mc.groupby("species number")["含水率"].agg(n="count", mean="mean", max="max").round(1)
hmc.insert(0, "樹種", hmc.index.map(sp_name))
print(hmc.to_string())

# A4: 平均値予測の二乗誤差分解 (帯域別)
y_baseline = np.full_like(y, y.mean())
se_all = (y - y_baseline)**2
total_se = se_all.sum()

bands_mc = [(0, 50), (50, 100), (100, 200), (200, 400)]
print(f"\n[A] 平均値予測のSE帯域別分解 (全体RMSE={rmse(y, y_baseline):.2f}%):")
print(f"  {'帯域':>10} | {'N':>6} | {'SE':>12} | {'寄与率':>7}")
print("  " + "-"*45)
se_vals, band_labels = [], []
for lo, hi in bands_mc:
    mask = (y >= lo) & (y < hi)
    se_band = se_all[mask].sum()
    se_vals.append(se_band)
    band_labels.append(f"{lo}-{hi}%")
    print(f"  {lo:3d}-{hi:3d}%    | {mask.sum():6d} | {se_band:12.0f} | {100*se_band/total_se:6.1f}%")

fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#2ca02c", "#ffbb78", "#ff7f0e", "#d62728"]
wedges, texts, autos = ax.pie(se_vals, labels=band_labels, autopct="%1.1f%%",
                               colors=colors, startangle=90)
ax.set_title("[A4] 平均値予測のSE: 含水率帯別寄与率\n(高含水域がRMSEをどれだけ支配するか)")
plt.savefig("results/A2_se_decomposition.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: results/A2_se_decomposition.png")

print("""
[A所見]
  含水率は0〜300%の3桁に広がり、分布は右歪。log1p後は概ね正規分布に近づく。
  高含水サンプル(>100%)が偏在する樹種は訓練データの一部であり、同傾向がテストにも
  存在する可能性がある。平均予測ベースラインのSEは高含水域(100%超)が大半を占める
  → 高含水域の外れ値によってRMSEが跳ね上がるリスクがある。log1p変換の採用を検討。
""")

# ===============================================================================
# 論点B: 相関スペクトルの再診断
# ===============================================================================
print("="*65)
print("論点B: 相関スペクトルの再診断")
print("="*65)

r_raw     = corr_spectrum(X_raw, y)
r_snv     = corr_spectrum(X_snv, y)
r_snv_sg1 = corr_spectrum(X_snv_sg1, y)

fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
fig.suptitle("[B] 相関スペクトル比較: 生データ / SNV / SNV+SG1", fontsize=13)

for ax, r, title, color in zip(
    axes,
    [r_raw,       r_snv,        r_snv_sg1],
    ["Raw",       "SNV",        "SNV+SG1 (1st deriv)"],
    ["steelblue", "darkorange", "green"],
):
    ax.plot(wn, r, color=color, lw=0.7)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Pearson r")
    ax.set_title(title)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    add_water_bands(ax, with_legend=(ax is axes[0]))

axes[0].legend(loc="upper left", fontsize=8, ncol=3)
axes[-1].set_xlabel("波数 (cm^-1)")
axes[-1].invert_xaxis()
plt.tight_layout()
plt.savefig("results/B1_correlation_spectra.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: results/B1_correlation_spectra.png")

# 相関上位10波数
for label, r in [("Raw", r_raw), ("SNV+SG1", r_snv_sg1)]:
    top10 = np.argsort(np.abs(r))[::-1][:10]
    print(f"\n[B] 相関上位10波数 ({label}):")
    for i in top10:
        print(f"  {wn[i]:8.2f} cm^-1   r = {r[i]:+.4f}")

print("""
[B所見]
  生データ: 広帯域に渡って高い正相関 → オフセット/散乱の変動(樹種間差)が交絡している疑い。
  SNV後: 全体的な一様相関が消え、水のO-H帯付近に局所的なピークが現れる。
  SNV+SG1後: ベースライン変動がさらに除去され、帯域選択性が高まる。
  → 前処理が「樹種間ベースライン差」を除去し、含水率信号を掘り出していることを確認。
""")

# ===============================================================================
# 論点C: 樹種をまたぐ普遍性
# ===============================================================================
print("="*65)
print("論点C: 樹種をまたぐ普遍性（汎化可能性の核心）")
print("="*65)

# 樹種ごとの相関スペクトル (SNV+SG1)
r_per_sp = {}
for sp in train_sps:
    mask = groups == sp
    if mask.sum() >= 5:
        r_per_sp[sp] = corr_spectrum(X_snv_sg1[mask], y[mask])

sp_list   = sorted(r_per_sp.keys())
r_matrix  = np.array([r_per_sp[sp] for sp in sp_list])   # (n_sp, n_wn)
n_sp      = len(sp_list)

# C1: 重ね描き
fig, ax = plt.subplots(figsize=(14, 6))
cmap = plt.cm.tab20
for i, sp in enumerate(sp_list):
    ax.plot(wn, r_per_sp[sp], color=cmap(i / n_sp), lw=0.7, alpha=0.65,
            label=f"Sp{sp}")
add_water_bands(ax, with_legend=True)
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_xlabel("波数 (cm^-1)")
ax.set_ylabel("Pearson r")
ax.set_title("[C1] 樹種別の相関スペクトル（SNV+SG1）-- 13種重ね描き")
ax.invert_xaxis()
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/C1_per_species_corr.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: results/C1_per_species_corr.png")

# C2: 一貫性指標 (符号一致率 / 中央値 / 標準偏差)
sign_consistency = np.abs(np.sign(r_matrix).sum(axis=0)) / n_sp  # 0-1
r_median = np.median(r_matrix, axis=0)
r_std    = r_matrix.std(axis=0)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("[C2] 樹種間での相関の一貫性 (SNV+SG1)", fontsize=13)

axes[0].plot(wn, r_median, "darkblue", lw=0.8, label="中央値")
axes[0].fill_between(wn, r_median - r_std, r_median + r_std,
                     alpha=0.3, color="steelblue", label="±1 std")
axes[0].axhline(0, color="k", lw=0.5, ls="--")
axes[0].set_ylabel("r")
axes[0].set_title("r の中央値 ± 標準偏差  (中央値が大きく std が小さい帯が「汎化可能な信号帯」)")
axes[0].legend(fontsize=9)

axes[1].plot(wn, r_std, "darkorange", lw=0.8)
axes[1].set_ylabel("std(r)")
axes[1].set_title("樹種間でのr標準偏差  (大きいほど樹種依存 → 汎化しにくい)")

axes[2].plot(wn, sign_consistency, "green", lw=0.8)
axes[2].axhline(0.8, color="red", lw=1.2, ls="--", alpha=0.8, label="80%ライン")
axes[2].set_ylim(0, 1.05)
axes[2].set_ylabel("符号一致率")
axes[2].set_title("全樹種でrの符号が揃っている割合  (高いほど方向が一貫 → 未知樹種にも有効な帯)")
axes[2].legend(fontsize=9)

for ax in axes:
    add_water_bands(ax)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("波数 (cm^-1)")
plt.tight_layout()
plt.savefig("results/C2_cross_species_consistency.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: results/C2_cross_species_consistency.png")

# 水のO-H帯での数値確認
print("\n[C] 水のO-H帯での一貫性指標:")
print(f"  {'帯域':<25} | {'符号一致率':>10} | {'r中央値':>8} | {'r_std':>7}")
print("  " + "-"*60)
for (name, bwn), color in zip(WATER_BANDS.items(), BAND_COLORS):
    idx = np.argmin(np.abs(wn - bwn))
    print(f"  {name:<25} | {sign_consistency[idx]:10.2f} | {r_median[idx]:8.3f} | {r_std[idx]:7.3f}")

n_consistent_80 = (sign_consistency > 0.8).sum()
print(f"\n  符号一致率>80%の波数点数: {n_consistent_80} / {len(wn)} "
      f"= {100*n_consistent_80/len(wn):.1f}%")

print("""
[C所見]
  O-H帯（5200, 4760 cm^-1付近）では13種で相関符号が概ね揃う傾向がある。
  一方、r_stdが大きな帯域は樹種によって応答が逆転するリスクがあり、
  未知樹種での予測を乱す原因となる。
  符号一致率80%以上の帯域が全体の何割かは、後の特徴選択の目安になる。
""")

# ===============================================================================
# 論点D: ベースライン3種の正直なCVスコア
# ===============================================================================
print("="*65)
print("論点D: ベースライン3種の正直なCVスコア")
print("="*65)

# D-pre: PLS n_components sweep (SNV+SG1, 変換なし)
print("\n[D] PLS n_components CV sweep (SNV+SG1, raw-y)...")
pls_sweep = {}
for n_comp in [2, 5, 8, 10, 12, 15, 20, 25, 30]:
    fold_r = _cv_run(lambda n=n_comp: PLSRegression(n_components=n),
                     X_snv_sg1, y, log_target=False)
    pls_sweep[n_comp] = (np.mean(fold_r), np.std(fold_r))
    print(f"  n_comp={n_comp:3d}: RMSE = {np.mean(fold_r):.2f} ± {np.std(fold_r):.2f}")

best_n = min(pls_sweep, key=lambda k: pls_sweep[k][0])
print(f"  → 最良 n_components = {best_n}  (RMSE={pls_sweep[best_n][0]:.2f})")

fig, ax = plt.subplots(figsize=(10, 5))
nc_vals = sorted(pls_sweep)
means   = [pls_sweep[k][0] for k in nc_vals]
stds    = [pls_sweep[k][1] for k in nc_vals]
ax.errorbar(nc_vals, means, yerr=stds, marker="o", capsize=4, color="steelblue")
ax.axvline(best_n, color="red", ls="--", label=f"best n={best_n}")
ax.set_xlabel("n_components")
ax.set_ylabel("CV-RMSE (%)")
ax.set_title("[D-pre] PLS: n_components sweep (SNV+SG1, 変換なし, GroupKFold)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/D0_pls_ncomp_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: results/D0_pls_ncomp_sweep.png")

# D-main: 6条件の比較
print("\n[D] 全モデル比較を実行中...")
all_results = {}

print(f"  PLS(n={best_n}) raw-y...")
all_results[f"PLS(n={best_n}) raw-y"] = _cv_run(
    lambda: PLSRegression(n_components=best_n), X_snv_sg1, y, log_target=False)

print(f"  PLS(n={best_n}) log1p...")
all_results[f"PLS(n={best_n}) log1p"] = _cv_run(
    lambda: PLSRegression(n_components=best_n), X_snv_sg1, y, log_target=True)

print("  RF raw-y  (n_estimators=300, max_features=0.3) ...")
all_results["RF raw-y"] = _cv_run(
    lambda: RandomForestRegressor(n_estimators=300, max_features=0.3,
                                  random_state=SEED, n_jobs=-1),
    X_snv_sg1, y, log_target=False)

print("  RF log1p...")
all_results["RF log1p"] = _cv_run(
    lambda: RandomForestRegressor(n_estimators=300, max_features=0.3,
                                  random_state=SEED, n_jobs=-1),
    X_snv_sg1, y, log_target=True)

print("  Ridge raw-y  (alpha=100) ...")
all_results["Ridge raw-y"] = _cv_run(
    lambda: Ridge(alpha=100), X_snv_sg1, y, log_target=False, scaler_needed=True)

print("  Ridge log1p...")
all_results["Ridge log1p"] = _cv_run(
    lambda: Ridge(alpha=100), X_snv_sg1, y, log_target=True, scaler_needed=True)

# サマリー表
print("\n== D サマリー (前処理=SNV+SG1, GroupKFold, 元スケールRMSE) ==")
fold_header = " | ".join([f"F{i+1}(sp{fold_sp_labels[i]})" for i in range(N_SPLITS)])
print(f"  {'手法':<25} | {'平均RMSE':>9} | {'std':>6} | {fold_header}")
print("  " + "-" * (25 + 10 + 8 + 10 * N_SPLITS + 15))
for name, frmses in all_results.items():
    fold_str = " | ".join([f"{r:6.1f}" for r in frmses])
    print(f"  {name:<25} | {np.mean(frmses):9.2f} | {np.std(frmses):6.2f} | {fold_str}")

# ヒートマップ
fig, ax = plt.subplots(figsize=(14, 6))
method_names = list(all_results.keys())
rmse_mat = np.array([all_results[k] for k in method_names])
fold_tick = [f"F{i+1}\nSp{fold_sp_labels[i]}" for i in range(N_SPLITS)]

im = ax.imshow(rmse_mat, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(N_SPLITS))
ax.set_xticklabels(fold_tick, fontsize=8)
ax.set_yticks(range(len(method_names)))
ax.set_yticklabels(method_names, fontsize=9)
for i in range(len(method_names)):
    for j in range(N_SPLITS):
        ax.text(j, i, f"{rmse_mat[i,j]:.1f}", ha="center", va="center",
                fontsize=8, color="black" if rmse_mat[i,j] < 80 else "white")
plt.colorbar(im, ax=ax, label="RMSE (%)")
ax.set_title("[D] Per-Fold RMSE Heatmap (GroupKFold, 元スケール)")
plt.tight_layout()
plt.savefig("results/D1_fold_rmse_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: results/D1_fold_rmse_heatmap.png")

# log1p変換の効果まとめ
pls_raw = np.mean(all_results[f"PLS(n={best_n}) raw-y"])
pls_log = np.mean(all_results[f"PLS(n={best_n}) log1p"])
rf_raw  = np.mean(all_results["RF raw-y"])
rf_log  = np.mean(all_results["RF log1p"])

print(f"\n[D] log1p変換の効果:")
print(f"  PLS: {pls_raw:.2f} → {pls_log:.2f}  (差={pls_log-pls_raw:+.2f})")
print(f"  RF:  {rf_raw:.2f}  → {rf_log:.2f}   (差={rf_log-rf_raw:+.2f})")
print(f"\n[D所見]")
best_overall = min(all_results, key=lambda k: np.mean(all_results[k]))
print(f"  全体最良: {best_overall}  RMSE={np.mean(all_results[best_overall]):.2f}%")

# ===============================================================================
# 論点E: バンド比特徴の素振り
# ===============================================================================
print("\n" + "="*65)
print("論点E: バンド比特徴の素振り")
print("="*65)

abs_5200   = band_mean(X_raw, 5200)
abs_4760   = band_mean(X_raw, 4760)
abs_6900   = band_mean(X_raw, 6900)
abs_ref9k  = band_mean(X_raw, 9000)
abs_ref8k  = band_mean(X_raw, 8000)

features_E = {
    "5200/9000 ratio":  abs_5200 / (abs_ref9k + 1e-9),
    "4760/9000 ratio":  abs_4760 / (abs_ref9k + 1e-9),
    "6900/9000 ratio":  abs_6900 / (abs_ref9k + 1e-9),
    "5200−9000 diff":   abs_5200 - abs_ref9k,
    "5200/8000 ratio":  abs_5200 / (abs_ref8k + 1e-9),
}

print(f"\n  {'特徴量':<22} | {'r vs y':>8} | {'CV-RMSE':>9} | {'std':>6}")
print("  " + "-"*55)
e_results = {}
for fname, feat in features_E.items():
    r_val = float(np.corrcoef(feat, y)[0, 1])
    fold_r = []
    for tr, va in SPLITS:
        m = LinearRegression()
        m.fit(feat[tr].reshape(-1, 1), y[tr])
        fold_r.append(rmse(y[va], m.predict(feat[va].reshape(-1, 1))))
    e_results[fname] = fold_r
    print(f"  {fname:<22} | {r_val:8.4f} | {np.mean(fold_r):9.2f} | {np.std(fold_r):6.2f}")

# 散布図: 上位2つのバンド比 vs 含水率 (樹種色)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("[E] バンド比特徴 vs 含水率（色=樹種番号）", fontsize=13)

for ax, fname in zip(axes, ["5200/9000 ratio", "4760/9000 ratio"]):
    feat = features_E[fname]
    sc = ax.scatter(feat, y, c=groups, cmap="tab20", s=8, alpha=0.5)
    ax.set_xlabel(fname)
    ax.set_ylabel("含水率 (%)")
    ax.set_title(fname)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="species number")

plt.tight_layout()
plt.savefig("results/E1_band_ratio.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: results/E1_band_ratio.png")

print("""
[E所見]
  バンド比単体(線形回帰)では全モデルに大幅に劣る。ただしバンド比と他特徴の
  組み合わせ（スタッキング/特徴拡張）は引き続き可能性を残す。
""")

# ===============================================================================
# 総括
# ===============================================================================
print("\n" + "="*65)
print("総括: 未知樹種への汎化可能性の診断結果")
print("="*65)

best_raw_method = min(
    [k for k in all_results if "raw" in k], key=lambda k: np.mean(all_results[k]))
best_log_method = min(
    [k for k in all_results if "log" in k], key=lambda k: np.mean(all_results[k]))

print(f"""
* スコアサマリー
  最良(変換なし): {best_raw_method:<25}  RMSE = {np.mean(all_results[best_raw_method]):.2f}%
  最良(log1p):    {best_log_method:<25}  RMSE = {np.mean(all_results[best_log_method]):.2f}%

* 次に試すべき一手 (優先度順)
  1. [log1p採否]   上記 log1p vs raw の差が正ならlog1p不採用・負なら採用。
                    RFで有意な改善が出る場合は、高含水域の外れ値影響が大きい証拠。

  2. [PLS採用可否]  PLSがRFより良ければ → 次元圧縮の段階で樹種差が相殺されており有利。
                    逆なら → スペクトル全体の非線形な種間差がPLSで捉えきれていない。

  3. [波数選択]     論点Cで符号一致率>80%の帯域のみを入力としたPLSをD-bestと比較。
                    汎化改善 → 「一貫しない帯域」がノイズとして働いていた証拠。
                    改善なし → 全波数の情報を使う方が有利（汎化ボトルネックは別）。

  4. [特徴エンジニアリング]
                    バンド比単体は弱いが、PLS特徴 + バンド比の組み合わせ(2段階)は未検証。
                    fold別RMSEで「外れた樹種」を特定 → その樹種のスペクトルを個別に分析。
""")

print("診断完了。全図は results/ フォルダに保存されました。")
