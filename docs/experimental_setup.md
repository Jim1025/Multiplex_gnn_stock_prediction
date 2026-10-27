# Experimental Setup

All settings are read from `configs/tw50.yaml` unless a CLI override is listed.
Overrides are applied before `config_snapshot.yaml` is written, so every run
directory records the values actually used.

**Main result (since 2026-10-27).** MAGNET with a same-industry relational
convolution as the TW intra-layer graph (run tags `tw50_rgcnSA_s*`, fold 2
`f2_rgcnSA_s*`; proposal §83). Its settings are in the "MAGNET (main result)"
column below and were read from those runs' `config_snapshot.yaml`. The
same-industry-only design was chosen after an ablation on the test windows
(proposal §79–§80). "Original MAGNET" and "MAGNET-v2" are kept for the record.

## 1. Data

| Item | Value |
|---|---|
| Markets | US (layer L1) and Taiwan (layer L2) |
| Universe | `tw50`: 30 US tickers, 50 TW tickers |
| ADR pairing | 7 of 50 TW stocks have a paired US listing (14.0%) |
| Sample period | 2019-01-01 to 2025-12-30 |
| Daily snapshots | 1,643 (after 60-day indicator warm-up) |
| Target `y` | TW next-day close-to-close log return, per stock |
| Feature source | Yahoo Finance daily OHLCV |

### Input features (`TECH_FEATURE_COLS`, F = 9)

`log_return`, `RSI_14`, `MACD`, `MACD_signal`, `MACD_hist`, `BB_pos`,
`MA5_dev`, `MA20_dev`, `log_volume_z`

The main result uses only `log_return`, and MAGNET-v2 a 3-feature subset
(`log_return`, `RSI_14`, `BB_pos`), both via `model.lstm.feature_subset`;
`input_dim` stays 9 and only `lstm.input_size` changes, so all shape
assertions remain valid.

### Split (walk-forward, no shuffling)

| Split | Indices | Dates | Days |
|---|---|---|---:|
| Train | 0–1149 | 2019-04-11 – 2023-12-19 | 1,150 (70%) |
| Validation | 1150–1396 | 2023-12-20 – 2024-12-25 | 247 (15%) |
| Test | 1397–1642 | 2024-12-26 – 2025-12-30 | 246 (15%) |

Sequences are cut with a strict `<` on the target date, so no snapshot can see
data from the day it predicts. Both markets use the same cut-off; a US close on
day *t*−1 lands before the TW open on day *t*, so the information is available
without look-ahead.

## 2. Graph construction

| Item | L1 (US) | L2 (TW) |
|---|---|---|
| Nodes | 30 | 50 |
| Edge rule | \|rho\| > 0.3 on `log_return` | same |
| Correlation window | 60 trading days, ending at *t*−1 | same |
| Mean edges | 619 | 1,107 |
| Mean density | 0.711 | 0.452 |
| Edge attribute | \|rho\|, fed into GAT attention | same |
| Cross-layer edges | 7 identity edges (paired ADR), weight fixed at 1 | — |

Graphs are rebuilt per snapshot; the window never touches the target date.

The main result does not use the TW correlation graph (`gat.graph_ablate:
empty_l2`). Its TW intra-layer graph is the same-industry relation: fixed for
the whole sample, from the industry category in the MOPS 2018Q4 holdings report
of the Yuanta Taiwan 50 ETF (known before the sample starts). 50 stocks fall in
18 industries, giving 260 directed edges; 9 stocks have no same-industry peer.

## 3. Model

| Component | Original MAGNET | MAGNET-v2 | MAGNET (main result) |
|---|---|---|---|
| Architecture flag | `magnet` | `magnet_weak_free` (`lambda_sparse = 0`) | `magnet_weak_free` + beta layer + `--tw-relational` (placement `replace_gat_l2`, 4 blocks, `same_industry`) |
| Features used | 9 | 3 (`log_return`, `RSI_14`, `BB_pos`) | 1 (`log_return`) |
| Look-back `T` | 20 | 1 | 1 |
| Shared LSTM | hidden 64, 1 layer, unidirectional, dropout 0.1 | same | same, with a shared BatchNorm on the input features |
| GAT (per layer, independent) | hidden 64, **2** layers, 4 heads, dropout 0.1, edge attr on | hidden 64, **1** layer, otherwise same | **US layer only**: hidden 64, 1 layer, otherwise same |
| TW intra-layer convolution | GAT on the TW correlation graph (row above) | same | **relational convolution in place of the TW GAT**: a learned linear map of the stock's own state, plus the mean of its same-industry peers times a learned block-diagonal weight (4 blocks of 16 x 16, initialised at 0) |
| Type projection | Linear to d' = 32, GELU, LayerNorm | Linear to d' = **29**, GELU, LayerNorm | Linear to d' = 32, GELU, LayerNorm |
| Raw skip (new) | — | `_FeatureNorm(3)` then `Linear(3, 3, bias=False)`, concatenated to give d' = 32 | — |
| Cross-layer coupling | identity edge + candidate edges `beta [30, 50]`, init 0 | same | beta layer, three terms added: the 7 paired stocks copy their US twin (one weight each); each TW stock's exposure to the cross-sectional mean of the US state (one weight each); coupling `B [30, 50]` applied to each US stock net of that mean |
| Fusion | per-node attention (hidden 64) and per-node per-dim sigmoid gate | same | same |
| Prediction head | Linear(32, 64), ReLU, Dropout 0.2, Linear(64, 1) | same | same |
| Trainable parameters | 166,242 | **48,772** | **45,284** |

MAGNET-v2 parameter breakdown: LSTM 17,664 / GAT_L1 8,512 / GAT_L2 8,512 /
proj_L1 1,943 / proj_L2 2,144 / skip_L1 15 / fusion 6,305 / head 2,177 /
`weak_beta` 1,500.

Main-result parameter breakdown: LSTM 17,154 / GAT_L1 8,512 / TW relational
convolution 5,248 (own-state weight 4,096, its bias 64, output bias 64, same-industry
weight 1,024) / proj_L1 2,144 / proj_L2 2,144 / fusion 6,305 / head 2,177 /
`weak_beta` 1,500 / pair and market-factor weights 50 + 50.

## 4. Loss

`L = 1.0 * MSE + 0.5 * RankNet_pairwise + 0.1 * variance_alignment` (MAGNET-v2).
The main result uses a rank weight of 1.0:
`L = 1.0 * MSE + 1.0 * RankNet_pairwise + 0.1 * variance_alignment`.

`align` (InfoNCE) is disabled: its positive pairs are defined on the diagonal,
which is not the paired set when n1 != n2. Measured gradient shares at a trained
MAGNET-v2 checkpoint are MSE 27%, rank 71%, variance 2% — the nominal 1 : 0.5
weighting is not the effective one, and this should be reported as such.

## 5. Training

| Item | Value |
|---|---|
| Optimizer | AdamW for the main result (Adam for MAGNET-v2), lr 1e-3, weight decay 1e-3 |
| Scheduler | CosineAnnealingWarmRestarts (T_0 = 10, T_mult = 2) |
| Batch size | 32 snapshots |
| Gradient clipping | 1.0 |
| Max epochs | 100 |
| Early stopping | validation cross-sectional IC, patience 15, min_delta 0 |
| Checkpoint | best validation IC; test is evaluated from that checkpoint only |
| Shuffling | disabled (time series) |
| Seeds | `torch.manual_seed` and `np.random.seed` both fixed |
| Seed set (10-seed runs) | 7, 11, 42, 99, 123, 314, 555, 777, 888, 2026 |

## 6. Evaluation

| Item | Value |
|---|---|
| Primary metric | IC: daily cross-sectional Pearson corr(y_hat, y), averaged over 246 test days |
| Secondary metric | RankIC: Spearman equivalent |
| Portfolio (diagnostic) | long/short 5 names per side, 252 periods/year, pre-cost |
| Noise floor | sd(daily IC) = 1/sqrt(k-1) = 0.143 at k = 50 |
| Seed variability | sigma_seed = 0.0062 (IC), from two independent 10-seed sets |
| Minimum detectable difference | 0.0188 at n = 3; 0.0082 at n = 10 (80% power, alpha 0.05) |
| Significance testing | `scripts/paired_daily.py`: paired-daily t (Newey-West corrected) and across-seed Welch, each Holm-corrected within a pre-specified family |

Three-seed comparisons are treated as directional only: at n = 3 a
non-parametric test cannot reach p < 0.05 (minimum two-sided p = 0.10), and the
measured Type-M exaggeration on this project's data is 1.41x.

## 7. Software and hardware

| Item | Value |
|---|---|
| Python | 3.14.2 |
| PyTorch | 2.11.0 |
| PyTorch Geometric | 2.7.0 (GATv2Conv) |
| NumPy / scikit-learn / SciPy | 2.4.3 / 1.9.0 / 1.17.1 |
| XGBoost / LightGBM | 3.4.1 / 4.7.0 — used only by the [24] reproduction (`scripts/bipartite_screen.py`); on macOS both need Homebrew `libomp` (22.1.7), and `xgboost` must be imported before `torch` |
| Device | Apple Silicon MPS |
| Experiment tracking | MLflow, local file store (`./mlruns`) |
| Reproducibility check | `scripts/freeze_k7.py --verify` — SHA-256 identical on the deterministic path, and \|d\| = 0.0000 on the stochastic path |
| Test suite | 267 tests collected (`pytest`) |
