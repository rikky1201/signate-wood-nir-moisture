# 実験ログ — 木材近赤外スペクトル含水率予測

| # | 日付 | モデル | 前処理 | 主なハイパーパラメータ | CV RMSE | Public RMSE | 順位 | 備考 |
|---|------|--------|--------|----------------------|---------|-------------|------|------|
| 1 | 2026-04-07 | Random Forest | snv+sg1 | n_estimators=300, max_features=0.3 | — | 18.85 | ~300/900 | **現行ベスト** |
| 2 | 2026-06-08 | Random Forest | msc+sg2+sqrt | n_estimators=300, max_features=0.3 | — | 24.85 | — | nb05: 前処理検索→悪化 |
| 3 | 2026-06-08 | Ridge | detrend+NoSel | α=1.0 | — | 26.18 | — | nb06フレームワーク: PLSで前処理選択→RidgeでFW最良→大幅悪化 |
| 4 | 2026-06-08 | RF+LGBM アンサンブル（加重） | snv+sg1 | 逆RMSE重み付け | — | 21.88 | — | nb07: LGBMが足を引っ張りRFより悪化 |

---

## 実験メモ

### #1 Random Forest + SNV+SG1（ベースライン）
- **前処理**: SNV（標準正規変量変換）→ Savitzky-Golay 1次微分（window=11, poly=2）
- **特徴量**: 前処理済みスペクトル全波数（1551次元, 9993〜4000 cm⁻¹）
- **モデル**: `RandomForestRegressor(n_estimators=300, max_features=0.3, random_state=42)`
- **CV戦略**: GroupKFold(n_splits=5), グループ=species number
- **Public RMSE**: 18.85（約300位/900人）
- **次の改善候補**:
  - [ ] PLSとのアンサンブル
  - [ ] 波数の選択（VIPスコアで絞り込み）
  - [ ] 樹種特徴量の追加（species numberをone-hot等）
  - [ ] ハイパーパラメータチューニング（max_depth, min_samples_leaf等）
  - [ ] 1D-CNNとの比較
