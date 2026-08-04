# M10 T1：early fusion vs MAGNET（回應「為何不做 early fusion」）

> 由 `scripts/m10_early_fusion_compare.py` 自動生成

> early fusion = 輸入層把配對 ADR 的 9 維特徵接在 TW 特徵後面，跑普通 LSTM；無圖、無閘門、無兩級耦合


## Run 層級（主檢定）

| arm | n_runs | test_IC mean | SD | min | max | Sharpe mean | Sharpe SD | dispersion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| early_fusion (seeds) | 5 | 0.1260 | 0.0141 | 0.1077 | 0.1431 | 4.280 | 0.775 | 1.256 |
| MAGNET@5e-4 (replicates) | 5 | 0.0334 | 0.0268 | 0.0131 | 0.0726 | 0.005 | 1.171 | 0.094 |
| MAGNET@5e-4 (seeds) | 3 | 0.0197 | 0.0071 | 0.0149 | 0.0279 | -0.253 | 0.364 | 0.072 |
| LSTM-only, no ADR (seeds) | 5 | 0.0264 | 0.0075 | 0.0177 | 0.0355 | 0.799 | 0.989 | 0.505 |

`dispersion` = std(ŷ) / std(y)。遠小於 1 代表 prediction collapse——模型退化成近乎常數預測。IC 是尺度不變量，看不出這件事。


## 逐名次平均實現報酬（跨 run 平均）

名次 0 = 當日預測最高。真實的排序訊號應呈單調遞減。

| arm | rank 0 | rank 1 | rank 2 | rank 3 | rank 4 | rank 5 | rank 6 |
|---|---|---|---|---|---|---|---|
| early_fusion (seeds) | +0.0040 | +0.0026 | +0.0015 | +0.0013 | -0.0000 | -0.0007 | -0.0023 |
| MAGNET@5e-4 (replicates) | +0.0016 | +0.0002 | +0.0009 | +0.0017 | +0.0001 | +0.0014 | +0.0005 |
| MAGNET@5e-4 (seeds) | +0.0014 | +0.0006 | +0.0004 | +0.0008 | +0.0007 | +0.0013 | +0.0013 |
| LSTM-only, no ADR (seeds) | +0.0012 | +0.0013 | +0.0013 | +0.0003 | +0.0017 | +0.0004 | +0.0002 |

Welch t 檢定（以 `early_fusion (seeds)` 為基準）：

| 對照 arm | ΔIC | t | p |
|---|---:|---:|---:|
| MAGNET@5e-4 (replicates) | +0.0927 | +6.851 | 0.0005 |
| MAGNET@5e-4 (seeds) | +0.1063 | +14.104 | 0.0000 |
| LSTM-only, no ADR (seeds) | +0.0996 | +13.939 | 0.0000 |

## 日層級（輔助，與 M8/M9 方法一致）

| 對照 arm | 平均每日 IC 差 | t | p | n_days |
|---|---:|---:|---:|---:|
| MAGNET@5e-4 (replicates) | +0.0927 | +3.655 | 0.0003 | 245 |
| MAGNET@5e-4 (seeds) | +0.1063 | +4.156 | 0.0000 | 245 |
| LSTM-only, no ADR (seeds) | +0.0996 | +4.065 | 0.0001 | 245 |

- **但書**：此檢定忽略 run 層級變異（M9 實測 sigma_run ≈ 0.027），屬 anti-conservative，不可單獨作為結論依據。


## 逐 run 明細

| arm | run tag | test_IC | Sharpe |
|---|---|---:|---:|
| early_fusion (seeds) | opt_p70_ef_r1 | 0.1179 | +5.278 |
| early_fusion (seeds) | opt_p70_ef_s7 | 0.1361 | +3.911 |
| early_fusion (seeds) | opt_p70_ef_s123 | 0.1253 | +4.143 |
| early_fusion (seeds) | opt_p70_ef_s2026 | 0.1077 | +3.282 |
| early_fusion (seeds) | opt_p70_ef_s314 | 0.1431 | +4.785 |
| MAGNET@5e-4 (replicates) | opt_p46_raw_lr5e4_s42 | 0.0149 | -0.026 |
| MAGNET@5e-4 (replicates) | opt_p66_fig1_lr5e4 | 0.0500 | +0.937 |
| MAGNET@5e-4 (replicates) | opt_p67_rep_s42_a | 0.0726 | +1.399 |
| MAGNET@5e-4 (replicates) | opt_p68_rep_s42_b | 0.0162 | -1.289 |
| MAGNET@5e-4 (replicates) | opt_p69_rep_s42_c | 0.0131 | -0.995 |
| MAGNET@5e-4 (seeds) | opt_p46_raw_lr5e4_s42 | 0.0149 | -0.026 |
| MAGNET@5e-4 (seeds) | opt_p47_raw_lr5e4_s7 | 0.0279 | -0.060 |
| MAGNET@5e-4 (seeds) | opt_p48_raw_lr5e4_s123 | 0.0164 | -0.672 |
| LSTM-only, no ADR (seeds) | opt_p71_lstm_s42 | 0.0283 | +0.498 |
| LSTM-only, no ADR (seeds) | opt_p71_lstm_s7 | 0.0177 | +1.513 |
| LSTM-only, no ADR (seeds) | opt_p71_lstm_s123 | 0.0355 | -0.758 |
| LSTM-only, no ADR (seeds) | opt_p71_lstm_s2026 | 0.0198 | +1.027 |
| LSTM-only, no ADR (seeds) | opt_p71_lstm_s314 | 0.0308 | +1.713 |

## 判讀

- vs `MAGNET@5e-4 (replicates)`：early fusion 優於 對照組 0.0927，run 層級 顯著（p = 0.0005）

- vs `MAGNET@5e-4 (seeds)`：early fusion 優於 對照組 0.1063，run 層級 顯著（p = 0.0000）

- vs `LSTM-only, no ADR (seeds)`：early fusion 優於 對照組 0.0996，run 層級 顯著（p = 0.0000）


- **determinism 註記**：early fusion 同 seed 位元確定，故其 SD 純為 seed 效應；MAGNET 的 replicate SD 另含 MPS run-to-run 變異。兩者的 SD 不同源，不可直接相互解讀。

- 若 early fusion 勝出，直接的意涵是：**在 ~1150 個訓練日的規模下，逐時步的 ADR-TW 互動比兩級潛在空間耦合更有價值，或 MAGNET 的 166K 參數相對資料量過大。** 兩者都指向同一個結論——現行 universe 撐不起現行架構。

- 無論方向為何，這個比較都**不能**單獨支持或否定兩級耦合的設計價值：k=7 的 benchmark 解析度（ΔIC≈0.02 需約 41 檔）遠不足以歸因。

