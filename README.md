# 木材近赤外スペクトル 含水率予測 — Signate コンペ解法

**SIGNATE** で開催された木材の近赤外（NIR）スペクトルデータを用いた含水率回帰タスクの解法リポジトリです。

## 問題概要

| 項目 | 内容 |
|------|------|
| タスク | 近赤外スペクトル（1555波数）から木材の含水率（%）を回帰予測 |
| 学習データ | 1,322 サンプル（複数樹種） |
| テストデータ | 550 サンプル |
| 評価指標 | RMSE（含水率 ≤ 170% のサンプルのみ） |
| CV 戦略 | GroupKFold (n_splits=5)、グループ = 樹種番号 |

樹種ごとにスペクトル特性が大きく異なるため、**種をまたいだ汎化**が最大の課題でした。

---

## アプローチと実験の流れ

| ノートブック | 内容 | 主な知見 |
|---|---|---|
| 01 | EDA | 樹種間でスペクトル形状・含水率分布が大きく異なることを確認 |
| 02 | PLS ベースライン | n_components を内部CVで自動選択する PLSAutoCV を実装 |
| 03 | 複数MLモデル比較 | RF が PLS を上回る；樹種グループを無視した CV はリーク |
| 04 | PLS 改善 | VIPスコアによる波数選択は効果薄 |
| 05–06 | 前処理系統探索 | SNV+SG1 が最も安定；MSC/Detrend は改善なし |
| 07 | LightGBM | RF との等重みアンサンブルで悪化 → 単体採用見送り |
| 08 | 樹種不変特徴量 | バンド比・PCA 投影特徴の追加は効果なし |
| 09 | ExtraTrees 単体 | RF より CV・LB ともに安定（最初の上位モデル） |
| 10–13 | 系統的最適化 | Fold3（特定樹種）が外れ値的に難しいことを特定 |
| 14–15 | PCA + PLS ブレンド | 3モデルブレンドで若干改善 |
| **16–21** | **1D-CNN 導入** | スペクトルの局所パターンを捉える CNN が ET より CV 改善 |
| **22** | **データ拡張** | ガウスノイズ拡張で CNN 安定性が向上 |
| **23** | **CNN+MLP+ET アンサンブル** | 3モデル等重み平均が単体より安定 |
| **24** | **頑健性チューニング** | seed分散・過学習ギャップ・予測分布の3軸で評価し最終設定を決定 |

---

## 最終モデル

前処理 → **SNV + Savitzky-Golay 1次微分**（window=41, polyorder=3）

| モデル | 設定 |
|---|---|
| ExtraTrees | n_estimators=500, max_features=0.2 |
| 1D-CNN | Conv×3 + Residual接続, Dropout=0.3, HuberLoss |
| ShallowMLP | Linear×3, BatchNorm, Dropout=0.4 |
| **アンサンブル** | **3モデル等重み平均、予測値を [0, 200] にクリップ** |

各ニューラルネットは複数乱数シード (42, 123, 456) で学習し平均することで seed 分散を抑制。

---

## 結果

| 指標 | 値 |
|------|-----|
| CV RMSE（F1,2,4,5 平均） | ~17.7% |
| CV Fold 標準偏差 | ~2.5% |
| 初期ベースライン Public RMSE | 18.85（RF のみ） |

---

## リポジトリ構成

```
.
├── notebooks/          # 実験ノートブック (01_eda 〜 24_robust_tuning)
├── src/
│   ├── preprocessing.py   # SNV, MSC, SG 等のスペクトル前処理
│   ├── models.py           # PLSAutoCV, RF, ET 等のモデル定義
│   ├── evaluation.py       # RMSE, GroupKFold ラッパー
│   ├── feature_selection.py
│   ├── data_io.py
│   └── utils.py            # データ読込・提出ファイル生成
├── results/            # 実験結果の可視化画像
├── data/
│   └── sample_submit.csv   # 提出フォーマットのサンプル
│   # train.csv / test.csv は Signate 利用規約により非公開
├── experiments.md      # 実験ログ（スコア・メモ）
└── run_experiments.py  # フレームワーク比較スクリプト
```

---

## 環境・実行方法

```bash
pip install numpy pandas scikit-learn scipy torch matplotlib
```

ノートブックは番号順に実行することで再現できます。データは [Signate](https://signate.jp) からダウンロードし `data/` に配置してください。

```
data/
├── train.csv
├── test.csv
└── sample_submit.csv
```

---

## 主な技術的知見

- **グループリーク防止**: 樹種を GroupKFold のグループとして使うことで、同一樹種が train/val に混在するリークを防止
- **Fold3 の難しさ**: 特定の樹種グループ（Fold3）が他 Fold より RMSE が約 2〜3 倍大きく、全体 CV スコアを押し上げていた。Fold3 除外スコアで意思決定することが重要
- **seed 安定性の重要性**: CV スコアだけでなく、複数乱数シードでの予測分散をモニタリングすることで過学習・不安定モデルを早期に排除できた
- **SNV+SG1 の頑健性**: 多くの前処理を試した結果、SNV → Savitzky-Golay 1次微分が樹種間の散乱差を最もうまく補正できた
