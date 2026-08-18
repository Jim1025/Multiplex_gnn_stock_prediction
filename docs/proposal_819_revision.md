# proposal_819 修改建議

依教授 8/12 的兩點建議整理，並附一份「簡報 vs 現行實作」的逐項比對。
所有數值由本次重新量測產生，來源見第 5 節。

---

## 0. 先講三件最重要的

1. **簡報描述的模型不是被評估的模型。** p23 的 `MAGNET (current) +0.0439`
   是 F3 + 拼接式原始特徵跳接（raw skip）的成績，而跳接在 p14 流程圖、
   p15 Phase 1、Algorithm 1、p22 環境設定裡**一個字都沒出現**。這是全份
   簡報最嚴重的落差：本專案唯一在統計上站得住的改動沒有被寫進架構。

2. **p22 呈現為「調參成果」的兩項改動，在驗證集上都是 null。**

   | 改動 | val IC 差 | val 配對 t | test IC 差 | test 配對 t |
   |---|---:|---:|---:|---:|
   | T = 20 → 1 | −0.0001 | 0.935 | +0.0054 | 0.020 |
   | F = 9 → 3 | −0.0016 | 0.251 | −0.0004 | 0.887 |
   | **raw skip off → on** | **+0.0340** | **0.0002** | **+0.0191** | **0.025** |

   （各 10 顆種子，同種子配對）。T 與 F 兩列在驗證集上完全分不開，
   卻在簡報上以 test IC 的高低呈現為選擇依據——口試時會被問
   「你怎麼選 T=1 的」，而目前投影片給的答案是 test IC，這是協定問題。
   真正被驗證集選出來的只有跳接（val 10/10 顆種子全勝）。

3. **例子與圖都已有實測數值可用。** 第 1 節給出 2025-04-10 這一天
   逐階段的實際張量數值，第 2 節給出 p14 的新版規格。不需要畫示意圖，
   直接用真的數字。

---

## 1. 建議一：一個貫穿三階段的例子

### 1.1 選定日期：2025-04-10（預測 4/10 當天的台股報酬）

挑選理由與揭露：這是 test 期間 ADR 波動最大的一天（前一交易日 TSM
+11.60%），選它是為了讓數值在投影片上看得清楚，**不是因為它有代表性**。
該日 day IC +0.2078，而全 test 期平均 +0.0427（s42）。這句話要寫在頁面
下緣，口試委員一定會問。

備用例子（若想要一個下跌日）：2025-04-07，TSM −6.96% → 2330 預測第 50 名、
實際第 47 名、實際報酬 −10.51%，day IC +0.3879。

### 1.2 模型在這一天看到什麼

4/9 美股收盤（台北時間 4/10 清晨 4:00），30 檔美股平均 +11.82%，
7 檔配對 ADR 分別是：

| ADR | 前一日 log_return | 配對台股 |
|---|---:|---|
| ASX | +0.1181 | 3711 |
| TSM | +0.1160 | 2330 |
| UMC | +0.1009 | 2303 |
| HNHPF | +0.0635 | 2317 |
| IMOS | +0.0497 | 8150 |
| AUOTY | +0.0378 | 2409 |
| CHT | +0.0314 | 2412 |

當日圖：美股層 802 條邊、台股層 2,200 條邊，TSM 的入度 27。

### 1.3 Phase 1（以 TSM 節點為例）

| 步驟 | 形狀 | 實測 norm | 備註 |
|---|---|---:|---|
| 輸入 x（T=1, F=3） | 3 | 42.49 | [+0.1160, 42.49, +0.3153] |
| Shared LSTM | 64 | 1.27 | 兩層共用同一份權重 |
| GATv2（1 層 4 head, \|ρ\| 當邊特徵） | 64 | 0.22 | 全 test 平均，LSTM 平均 1.54 |
| Type projection + LayerNorm | 29 | 3.65 | **不是 32** |
| **raw skip**：`_FeatureNorm(3)` → `Linear(3,3)` | 3 | 1.16 | 值 [+0.955, −0.641, −0.104] |
| `h_L1 = concat(proj, skip)` | **32** | 3.83 | 跳接佔能量 9.1%（全期平均 1.3%） |

**這張表本身就是投影片上最有說服力的一頁**，因為它同時解釋了跳接為什麼
必要。把節點之間的平均餘弦相似度加上去（全 246 個 test 日）：

| 表示 | 節點間平均餘弦 |
|---|---:|
| 投影出來的 29 維 | **+0.999968**（min +0.9998） |
| 跳接的 3 維 | +0.3243 |
| 合起來的 h_L1（32 維） | +0.9921 |
| h_L2（無跳接） | +0.9993 |

也就是說：**編碼器輸出的 29 維，對 30 檔美股而言幾乎是同一個向量；
耦合點上全部的橫截面資訊都由那 3 維跳接攜帶，而它只佔 1.3% 的能量。**
這一句是整個 Phase 1 的重點，比目前 p15 的「Core spirit: make Phase 2
fusion meaningful」具體得多。

### 1.4 Phase 2

2330（有配對）：

```
identity   h_1(TSM)                     norm 3.83
candidate  Σ_i B_eff[i,2330] · h_1(i)   norm 1.24   （佔 24.5%）
ĥ_1(2330)  = identity + candidate       norm 5.07
gate       g(2330) ≈ 0.50（前 6 維 0.4869 0.4956 0.4967 0.5062 0.4968 0.5025）
h_f(2330)  = g ⊙ ĥ_1 + (1−g) ⊙ h_2      norm 3.59
```

2454（無配對，只有候選邊）：候選項 norm 0.348，對照自身 h_L2 的 3.93，
跨市場輸入佔 8.1%（43 檔未配對股全期平均 **7.0%**）。

候選邊學到什麼（λ=0，1,493 條全部非零，|B| 最大 0.0181、中位數 0.0028）：

| 台股 | 最強的 5 條候選邊 |
|---|---|
| 2330 | AMD .0181, NVDA .0178, MRVL .0170, LRCX .0169, AMAT .0164 |
| 2454 | MRVL .0061, AMD .0057, ASML .0052, TER .0051, MU .0051 |

**沒有給模型任何產業資訊，它自己把半導體聚在一起**——這是目前簡報完全
沒用到的一張好圖（30×50 的 B 熱圖，列依產業排序）。

### 1.5 Phase 3 與結果

| 台股 | ADR 前一日 | 預測名次 | 實際名次 | 實際報酬 |
|---|---:|---:|---:|---:|
| 2330 | TSM +11.60% | **1** | 12 | +9.47% |
| 3711 | ASX +11.81% | **2** | 3 | +9.53% |
| 2303 | UMC +10.09% | **3** | 17 | +9.44% |
| 2317 | HNHPF +6.35% | 7 | 30 | +9.33% |
| 2409 | AUOTY +3.78% | 11 | 16 | +9.45% |
| 8150 | IMOS +4.97% | 49 | 24 | +9.40% |
| 2412 | CHT +3.14% | **50** | **50** | +0.79% |

7 檔配對股的預測名次是 {1, 2, 3, 7, 11, 49, 50}，平均 17.6（隨機為 25.5）。
**模型把 ADR 漲最多的三檔放在第 1、2、3 名，把 ADR 漲最少的 CHT 放在
第 50 名——而 2412 當天確實是 50 檔裡的最後一名。** 這一句話就把恆等邊
的機制講完了，不需要任何示意圖。

要揭露：2330 預測第 1、實際第 12（前四分之一）；day IC +0.2078，
對照日 IC 的噪音底 1/√49 = 0.143。

### 1.6 把例子從軼事變成證據（建議獨立一頁）

單日例子一定會被問「你是不是挑過」。用全 246 天的統計回答：

| 統計量（246 個 test 日，s42） | 值 |
|---|---:|
| 7 檔配對股：ADR 報酬 vs **模型預測** 的日內 Spearman | **+0.687**（240/246 天為正） |
| 7 檔配對股：ADR 報酬 vs **實際報酬** 的日內 Spearman | **+0.095**（143/245 天為正） |

這兩個數字放在一起是本工作最誠實也最有力的一句話：

> The architecture transmits the ADR signal almost perfectly (+0.687).
> The signal itself is worth +0.095. The ceiling is in the data, not in the coupling.

它同時回答了「機制有沒有在動」與「為什麼 IC 只有 0.044」，並且自然
接到 Planned improvement。

### 1.7 p15 改寫（移除 Step 1/2/3）

教授要求拿掉 Step 的切分。建議把 p15 從「三個 Step 的清單」改成
「一條資料流 + 一個節點的實際數值」：

```
Phase 1: Dual-market encoding
One node, one day: TSM on 2025-04-09 (US close)

  x = [log_return +0.1160, RSI_14 42.49, BB_pos +0.3153]     R^{1x3}
        |                                               \
        | shared LSTM (H=64, one set of weights            \  raw skip
        |   for both markets)                               \  feature-wise norm
        v                                                     \  then Linear(3,3)
      GATv2, 1 layer, 4 heads, |rho| as edge attribute          \
        27 US neighbours on this day                             \
        |                                                         |
        v                                                         v
      Linear -> GELU -> LayerNorm  (29 dims)                  (3 dims)
        \______________________ concat ______________________/
                                |
                                v
                        h_1(TSM) in R^32   -> Phase 2

  Why the skip: without it the 29 projected dims have cosine +0.99997
  across all 30 US nodes. The 3 skip dims carry 1.3% of the energy and
  all of the cross-sectional information.
```

Algorithm 1 對應要加兩行：

```
7:  r_1 <- W_s · FeatureNorm(X_1[:, T])       # raw skip, no bias
8:  h_1 <- [ LayerNorm(GELU(P_1 · z_1)) ; r_1 ]
```

並在 Output 把維度寫成 `h_1 ∈ R^{n1 x d'}, d' = d_proj + d_raw = 29 + 3`。

---

## 2. 建議二：把文字換成圖

### 2.1 p14 流程圖（`docs/figures/magnet_flow_v2.svg`）新版規格

現行版本有三個問題：跳接沒畫、`d′ = 32` 標在投影層（實際是 29）、
loss 裡的 `λ‖B ⊙ M‖₁` 在現行最佳設定下 λ = 0。建議：

| 改動 | 內容 |
|---|---|
| 加跳接 | 從「L1: US ADR Market Layer」拉一條旁路直接進「Two-tier A12 Coupling」的輸入端，標 `FeatureNorm → Linear(3,3)`，用不同顏色 |
| 改維度標示 | 投影層改標 `29`，concat 節點標 `d′ = 29 + 3 = 32` |
| 標量測點 | 在三個位置加小標籤：`raw +0.0874` / `coupling point +0.0754` / `output +0.0439`，這是本專案的診斷主線，一眼就看得到資訊在哪裡流失 |
| 改 loss 方塊 | `L = L_MSE + 0.5·L_rank + 0.1·L_var`（λ = 0 in the reported model），並加註實測梯度佔比 27% / 71% / 2% |
| 保留 | `L_align removed after ablation` 這行很好，留著 |

### 2.2 其他建議改為圖的頁

| 頁 | 現況 | 建議 | 資料來源 |
|---|---|---|---|
| p4 Motivation | 四層 bullet | **耦合層級階梯圖**：feature / index / market-state / correlation / entity 五層由粗到細排一條軸，把 [19,20][15,21][23][1][24] 放上去，本工作放最右端 | 已有 `docs/related_work.md` |
| p9 Related work | 8 欄大表 | 表留著當附錄，正文改成 2D 定位圖（x = 耦合粒度，y = 對場次先後的處理），一眼看出空白區在哪 | 同上 |
| p10 System model | 已有圖 | 圖裡候選邊只畫了幾條，實際 1,500 條全非零。改成「7 條實線恆等邊 + 一片淡色候選邊」 | 本節量測 |
| **新增** | — | **B 矩陣熱圖**（30×50，列依產業排序），展示 2330 的最強邊是 AMD/NVDA/MRVL | 本節量測 |
| **新增** | — | **資訊流失階梯圖**（橫條）：raw 0.0874 / 耦合點 0.0754 / 輸出 0.0439 / 線性高原 0.10 | `docs/roadmap_stage_a.md` §2 |
| p22 環境設定 | 兩張小表 | 改成一張「四個設定 × val/test」的對照圖，並用顏色標出「驗證集分不開」的兩列 | 第 0 節的表 |
| p23 結果 | 11 列表格 | 表留著，另加一張 forest plot：各模型 test IC 與 95% CI，線性高原畫成灰帶 | `docs/results_table.md` |
| **新增** | — | **例子頁**（第 1.5 節那張 7 列表），配一張「ADR 報酬 vs 預測名次」散點 | 本節量測 |

已存在可直接用的圖：`docs/figures/graph_eda/`（8 張，圖結構 EDA）、
`docs/figures/m8_fig2_crosssection_noise.png`、`docs/figures/e8_noise_floor.png`。
目前一張都沒放進簡報，p10 與 p22 各補一張就能省掉大段文字。

---

## 3. 簡報 vs 現行實作的逐項比對

### A 級：描述的模型不是被評估的模型（必改）

| # | 位置 | 簡報寫的 | 實際的 |
|---|---|---|---|
| A1 | p14, p15, Alg 1, p22, Notations | 沒有跳接 | `raw_skip {l1: true, mode: concat, norm: batchnorm}`，是唯一通過四種檢定的改動（+0.0191） |
| A2 | p14「to shared latent d′ = 32」 | 投影輸出 32 | 投影輸出 **29**，concat 跳接的 3 維後才是 32（`proj_L1.out_features = 29`） |
| A3 | p8, p16 表, p18, p14 | 候選邊受 L1 稀疏約束、`E_act` 是「survivors after L1」 | **λ = 0**，1,493 條候選邊 **100% 非零**，沒有任何邊被篩掉，`E_act = E_cand`。λ=1e-3 曾讓候選邊只送出 0.08% 的訊號 |
| A4 | p18 Algorithm 3 第 6 行 | `L_rank ← Σ max(0, −(ŷu−ŷv)(yu−yv))` | RankNet：`Σ log(1+exp(−(ŷu−ŷv)))`，配對取 `yu > yv`。同一頁的浮動方塊寫的是對的，等於一頁上有兩個互相矛盾的公式 |
| A5 | p22「Trainable params: 48,772」 | — | 數字正確，但其中 **4,225（8.7%）是 `fusion.attn_mlp`，其輸出 `alpha` 從未進入 `h_fused` 或 loss，永遠停在初始值**。Algorithm 2 沒寫它是對的，但參數量把它算進去了 |
| A6 | p22 缺 GAT 層數 | — | 原始 MAGNET `gat.num_layers = 2`，MAGNET-v2 是 **1**。這是 v2 與原版的差異之一，簡報沒提 |

A5 的處置建議：把 `attn_mlp` 從模型移除並重跑（數值會變，成本高），
或在 p22 註明「48,772 total; 44,547 receive gradient」。建議後者，
並在論文的 limitation 提一句。

### B 級：數字與協定

| # | 位置 | 問題 |
|---|---|---|
| B1 | p22 T/F 兩張表 | 呈現為調參依據，但驗證集上 T 差 −0.0001（p=0.935）、F 差 −0.0016（p=0.251）。**必須改寫成「驗證集分不開，T=1 是先驗選擇（ADR 領先本來就是一個場次），F=3 是精簡選擇」**，否則等於自陳用 test 選超參 |
| B2 | p22 兩張表的 n | 未標種子數。T 表兩列都是 10 顆，F 表的 +0.0269/+0.0269 是單顆種子；10 顆的正確值是 F9 +0.0252 (sd 0.0062)、F3 +0.0248 (sd 0.0078) |
| B3 | p22 T 表 RankIC | 表上 +0.0354 / +0.0406，重算得 +0.0341 / +0.0404。差異來自「先平均 run 再算」與「先平均預測再算」兩種聚合，統一一種即可 |
| B4 | p23 結果表 | 缺 `Ridge, 30 US t-1 returns (+0.1020)`、`Constant (train-mean rank) (+0.0149)` 與 `MAGNET-v2 w/o skip (+0.0248)` 三列。常數對照尤其重要——它是排序任務真正的 null |
| B5 | p23 | 未揭露 baseline 只跑一組預設超參而 MAGNET 掃過約 50 組。這是審查者會第一個問的公平性缺口 |
| B6 | p18「Loss terms」 | 名目權重 1 : 0.5 : 0.1，實測梯度佔比 27% : 71% : 2%。另外 `L_var` 的目標是 `std(ŷ) → std(y)`，實測比值只有 **0.15**（全 test 平均），沒有達成 |
| B7 | 隱藏頁 slide 20 | 寫「R ≥ 5 replicates」，現行標準是探路 3 顆、下結論 10 顆。這頁的內容（為什麼用 IC 不用 MSE、IR ≈ IC·√breadth）口試會用到，建議取消隱藏並更新 |
| B8 | p19 Experiment plan | 全部寫成未來式，但實驗都跑完了。H1/H2 也沒有給答案 |

H1/H2 現在可以回答：

- **H1（ADR 領先改善配對股的排序）：支持。** 模型預測與 ADR 報酬的日內
  Spearman +0.687（240/246 天為正）；配對股實際報酬與 ADR 的 Spearman
  +0.095（143/245 天為正）——訊號真實但小。
- **H2（候選邊為未配對股帶來訊號）：未獲支持。** 43 檔未配對股的跨市場
  輸入僅佔 7.0%，且 early fusion（完全沒有候選邊）在 IC 上還略高。
  要正面回答需要跑 `--architecture magnet_no_a12` 的消融（成本 3 顆種子）。

### C 級：措辭與宣稱

| # | 位置 | 建議 |
|---|---|---|
| C1 | p6 Contribution | 「Two-tier edge design: each tier's contribution is separately measurable」目前只有恆等邊那一層真的量到了。改成「we measure what the identity tier carries (+0.687 transmitted, +0.095 available) and show the candidate tier adds 7.0% of input norm with no measurable IC gain」 |
| C2 | p17 Design points | 「Per node: firms differ in ADR informativeness」**實測未成立**：gate 全期均值 0.5018，跨節點 std **0.00053**，完全沒有分化。跨維 std 0.0108，略有分化。要嘛改成「designed per-node, measured to be flat — see Planned improvement」，要嘛就別宣稱 |
| C3 | p17 Readouts | 「g statistics → when ADR matters」同上，g 是常數，這個 readout 目前給不出資訊 |
| C4 | p24/p25 副標 | 「Two interventions, both increased the model's ability to fit, both made test IC worse」與「Reduce capacity first」已被跳接推翻（跳接提高了擬合能力**且**改善 test IC）。整段重寫，見第 4 節 |
| C5 | p16 表「Regularization: L1 λ‖B⊙M‖₁」 | 與 A3 同，改成 `λ = 0 in the reported model; λ = 1e-3 was tested and froze the edges` |

### 已確認一致（不用改）

- Algorithm 1 第 7–8 行 `LayerNorm(GELU(P·z))` 與 `TypeProjection.forward` 相同
- p18 預測頭 `R^d' → 64 → 1, ReLU, dropout 0.2` 與 `PredictionHead` 相同
  （先前記錄的「p18 寫 GELU 但碼是 ReLU」在 819 版已修正）
- Algorithm 2 只寫 gate、不寫 attention，與 `h_fused` 的實作一致
- p3 的 7 對 ADR、p22 的 n₁=30 / n₂=50 / n_p=7 / 1,150 / 247 / 246 / τ=0.3 全部正確
- p23 的 `MAGNET (current) +0.0439 (0.0174)` 與 `Early fusion +0.0518 (0.0049)` 正確

---

## 4. Planned improvement 兩頁的內容（2026-08-18 重寫）

**本節已依 §8/§9 的秩量測全面改寫。** 舊版把瓶頸歸給「編碼器丟失資訊」與
「融合破壞秩」，兩者都不正確：編碼器忠實傳遞了一個 rank-1 的輸入，
融合反而讓秩微升。正確的因果鏈見下。

### p24 — Planned improvement (1/2)

標題：`The bottleneck is cross-sectional rank, and it starts at the input`

**資訊階梯**（per-target ridge 探針在各位置抽得到的 test IC）

```
raw 3 features x 30 US nodes      +0.0874
coupling point h_1 (with skip)    +0.0754
model output                      +0.0439
linear plateau (6 methods)        +0.098 to +0.108
```

**秩的帳**（每個運算子之後，跨節點有效秩，246 個 test 日平均）

| 位置 | 節點數 | 有效秩 |
|---|---:|---:|
| 輸入 x（3 維，未正規化） | 30 | **1.002** |
| Shared LSTM (64) | 30 | 1.01 |
| GATv2 (64) | 30 | 1.01 |
| Type projection (29) | 30 | 1.00 |
| **raw skip 的 BN_F (3)** | 30 | **1.88** |
| h_L1 = concat (32) | 30 | 1.69 |
| **ĥ₁ 耦合後（台股側）** | 50 | **1.05** |
| h₂ 台股層 | 50 | 1.01 |
| h_f 融合後 | 50 | 1.10 |
| **ReLU(W₁h_f + b₁) 預測頭** | 50 | **1.57** |

三句話：

1. **塌縮的源頭是輸入尺度，不是編碼器。** RSI_14 約 50、BB_pos 約 0.5、
   log_return 約 0.01，差三個數量級，未正規化的 3 維向量幾乎就是
   「RSI × 單位向量」，30 檔美股全部指同一個方向（平均餘弦 +0.99999）。
   逐特徵正規化後升到 1.88，逐日橫截面標準化後升到 2.01。
2. **全模型只有兩個運算子在製造秩**：跳接裡的 BN_F、預測頭的 ReLU。
   **兩者都不屬於 multiplex 架構。**
3. **秩在耦合這一步掉最多**（1.69 → 1.05），因為 43 個台股槽位只拿到
   「一個純量 × 共同方向」。

**這解釋了三個先前無解的觀測**

| 觀測 | 機制 |
|---|---|
| 候選邊消融無效（−0.0012, p = 0.79） | B 本身有效秩 1.737、欄向量兩兩餘弦 −0.019（有結構），但 `cos(cand_j, (Σ_i B[i,j])·h̄) = +0.9986` —— **B 學到了結構卻表達不出來，因為它乘的東西是 rank-1 的** |
| gate 恆為 0.5016（跨節點 sd 0.0005） | `\|W_g\| = 0.247`，logit 只有 0.026 量級；sigmoid 在 0 附近近似線性，g ≈ 0.5 + logit/4。**gate 不是學到不該分化，是輸入沒有可分化的量** |
| GATv2 的注意力完全均勻 | 1,230 個 node-day 的正規化熵全部 = 1.000000。輸入共線 → 每個鄰居的 attention logit 相同 → softmax 必然均勻。**\|ρ\| 邊特徵對權重沒有任何影響** |

**邊的消融**（10 顆種子 × 246 天，同一批 checkpoint）

| 移除 | ΔIC | p |
|---|---:|---:|
| 恆等邊（7 條） | **−0.0320** | **6.9e-06** |
| 候選邊（1,493 條） | −0.0012 | 0.79 |
| 全部跨市場輸入 | −0.0504 | — |

### p25 — Planned improvement (2/2)

標題：`Four changes to MAGNET, ordered by where rank is created or lost`

| # | 改動 | 檢驗的假說 | 預期效果 | 成本 |
|---|---|---|---|---|
| **1** | **逐特徵正規化移到 LSTM 之前**（主幹與跳接共用同一個 BN_F） | 塌縮源自輸入尺度；主幹目前吃的是 rank-1.00，跳接吃的是 rank-1.88 | 輸入秩 1.00 → 1.88，整條主幹（LSTM / GAT / projection）第一次拿到可分辨的橫截面；GATv2 的注意力應開始非均勻 | 一個旗標，3+7 種子 |
| 2 | **加寬預測頭 / 改 per-target 讀出**（每檔一個 32→1 向量，1,600 參數） | ReLU 已是唯一在製造秩的主幹運算子（1.10 → 1.57）；而探針是 per-target 且達 +0.0754，共享 head 只有 +0.0439 | 縮小 58% 這個缺口 | 3+7 種子 |
| 3 | **改動 1 之後重測耦合層**（dense A、低秩 A、候選邊） | 目前候選邊無效是因為被乘的表示是 rank-1；若改動 1 讓 h_L1 的秩升高，同一組邊可能才開始有作用 | 若仍無效，則兩層邊設計確定被否證 | 3 種子 x 3 arm |
| 4 | **把候選邊寫成量測到的零結果** | — | IC 不變，改變論文主張 | 純分析 |

**降級的項目**：融合改拼接。理由不成立 —— 融合並未破壞秩（1.05 → 1.10），
拼接兩個 rank-1.05 與 rank-1.01 的近同向表示拿不到什麼。

**上限與否證條件**

- 架構層的改動上限是耦合點的 **+0.0754**，不是 +0.10 的線性高原。
  要碰到高原必須改編碼器或改輸入，不是改耦合。
- 若改動 1 讓秩升高但 IC 不動，瓶頸就在資料而非架構，
  下一步轉向外生變數與把 y 拆成隔夜跳空 + 盤中。

**口試用的一句話**

> Two operators in this model create cross-sectional rank: a BatchNorm inside
> the skip connection, and the ReLU in the prediction head. Neither belongs to
> the multiplex architecture. That is the finding, and it is what the next
> round of experiments is built on.

## 5. 數字來源

| 數值 | 來源 |
|---|---|
| 逐階段 norm、餘弦、gate、B、名次 | 本次以 `runs/20260816_1601_tw50_T1F3bnl1_s42/checkpoints/best.pt` 重跑 246 個 test 日；腳本在 scratchpad（`trace_day.py` 單日、`trace_agg.py` 全期），建議收進 `scripts/` |
| T / F / skip 的 val 與 test 配對 t | 各 arm 的 `runs/*/meta.json`，同種子配對，n = 10 |
| 參數拆解 48,772 = ... + attn_mlp 4,225 | `build_model` 後逐模組 `numel()` |
| 資訊流失階梯 | `docs/roadmap_stage_a.md` §2 |
| baseline 比較 | `docs/results_table.md` |
| 環境設定 | `docs/experimental_setup.md` |

### 順帶發現的三個實作缺口

1. `config_snapshot.yaml` 沒有記錄 `raw_skip.norm`（目前靠預設值
   `batchnorm` 才重現得出來）與 `weak_links.mode`（由 `build_model` 注入
   局部 cfg）。預設值一改，既有 run 就重現不了。
2. `fusion.attn_mlp`（4,225 參數）沒有梯度來源，永遠停在初始化值。
3. `roadmap_stage_a.md` 記的 LayerNorm 探針值 +0.0332 有誤，重測為
   **+0.0311**（`multiplex_gnn.py` 的註解才是對的）。已更正。

---

## 6. 「F3 + 拼接式跳接」要怎麼寫進簡報

以下是可直接照抄的內容。順序即建議的製作順序。

### 6.1 Notations（p12 Model 區塊補三列）

| Category | Notation | Description | Property |
|---|---|---|---|
| Model | `F′` | Number of features actually fed to the encoder (`feature_subset`) | Input |
| Model | `d_raw` | Width of the raw-skip path, `d_raw = F′` | Input |
| Model | `d_proj` | Projection output width, `d′ = d_proj + d_raw` | Input |
| Model | `S₁ ∈ R^{d_raw×F′}` | Raw-skip weight, no bias | Variable |
| Model | `r₁ᵢ(t) ∈ R^{d_raw}` | Raw-skip output of US node i | Variable |

並把既有的 `d′` 描述改成
`Shared latent dimension after projection and skip concatenation`。

`BN_F(·)` 需要一句定義（放在 p15 或註腳）：

> `BN_F` normalises **within a feature, across nodes** — the BatchNorm axis.
> LayerNorm normalises within a node across features; at `F′ = 3` that leaves
> one degree of freedom per node and deletes the cross-section.

### 6.2 p15 Phase 1：一條式子取代 Step 1/2/3

主式：

```
h₁ᵢ(t) = [ LayerNorm( GELU( P₁ · z₁ᵢ(t) ) ) ; S₁ · BN_F( x₁ᵢ(t)[T] ) ]  ∈ R^{d′}
           \___________ 29 dims ___________/   \______ 3 dims ______/
```

Algorithm 1 改動（原第 7 行拆成兩行）：

```
Input:  ... ; projections P₁, P₂; raw-skip S₁; feature-wise norm BN_F
Output: latent states h₁ ∈ R^{n₁×d′}, h₂ ∈ R^{n₂×d′}, d′ = d_proj + d_raw

6:  z₁ ← GAT₁(s₁, E₁, ω₁);  z₂ ← GAT₂(s₂, E₂, ω₂)
7:  r₁ ← S₁ · BN_F(X₁[:, T])                        ▷ raw skip, no bias
8:  h₁ ← [ LayerNorm(GELU(P₁ · z₁)) ; r₁ ]          ▷ concat, d′ = 29 + 3
9:  h₂ ← LayerNorm(GELU(P₂ · z₂))
10: return h₁, h₂
```

三個標註方塊（沿用現行版式）改成：`Temporal encoding` / `Graph` /
`Type-specific projection + raw skip`。

### 6.3 p14 流程圖

已產出 `docs/figures/magnet_flow_v3.svg`，可直接匯入取代現行圖。與 v2 的差異：

- 新增琥珀色 `Raw skip` 直通方塊，從 L1 輸入旁路到 concat
- ADR Projection 標 `29 dims`，新增 `concat → d′ = 29 + 3 = 32` 橫條
- 三個線性探針數字標在對應位置（`+0.0874` / `+0.0754` / `+0.0439`）
- 耦合方塊的 `L1 sparsity λ‖B⊙M‖₁` 改為 `λ = 0 in the reported model`
- loss 方塊移除 λ 項，加上實測梯度佔比 27% / 71% / 2%
- 圖例：琥珀色 = new in MAGNET-v2

### 6.4 新增一頁：`Why the encoder needs a bypass`

這是本工作唯一站得住的貢獻，值得一整頁。建議放在 Phase 1 之後。

**The projected representation is the same vector for every US node**

| Representation | cosine between the 30 US nodes | ridge probe test IC |
|---|---:|---:|
| raw 3 features (30 × 3) | — | +0.0874 |
| projected 29 dims | **+0.99997** (min +0.9998) | +0.0448 |
| projected 29 + skip 3 | +0.9921 | +0.0754 |
| model output | — | +0.0439 |

> The 3 skip dimensions carry 1.3% of the energy at the coupling point and
> essentially all of the cross-sectional information.

**The normaliser is the mechanism, not a detail**

| Normaliser | normalises | probe test IC |
|---|---|---:|
| none | — | +0.0874 |
| LayerNorm (feature axis) | within a node, across features | **+0.0311** |
| feature-wise (node axis) | within a feature, across nodes | **+0.0874** |

At `F′ = 3`, LayerNorm leaves one degree of freedom per node. Two earlier
versions of this experiment used it and were invalidated. `none` is
information-preserving but ill-conditioned in the linear layer (feature scales
differ by 5,000x), so the feature-wise normaliser is the one that works.

**Result (10 seeds each, matched seed set)**

| Metric | w/o skip | with skip | Δ | Welch | Paired t | Mann-Whitney | Wilcoxon | Wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IC | +0.0248 | +0.0439 | +0.0191 | 0.008 | 0.025 | 0.009 | 0.049 | 8/10 |
| RankIC | +0.0345 | +0.0515 | +0.0170 | 0.003 | 0.017 | 0.007 | 0.037 | 7/10 |

Selected on validation IC: +0.0313 → +0.0653, paired p = 0.0002, 10 of 10 seeds.

### 6.5 p22 環境設定補列

Data & Model 表加 `F′ = 3`、`d_proj = 29`、`d_raw = 3`、`L_GAT = 1`、`T_ρ = 60`、
`λ = 0`。「Setting for MAGNET」下的三行改成四行：

```
Trainable params: 48,772  (44,547 receive gradient)
Look-back T: 20 → 1        val IC −0.0001, p = 0.935  (not selected on validation)
Features F: 9 → 3          val IC −0.0016, p = 0.251  (not selected on validation)
Raw skip: off → on         val IC +0.0340, p = 0.0002, 10/10 seeds
```

**這四行放在一起才是誠實的呈現**：前兩項是先驗與精簡的選擇，第三項才是
被驗證集選出來的改動。目前的版面把三者都畫成調參成果。

### 6.6 p6 Contribution 與 p23 結果表

p6 第三塊 Contribution 加一條：

> A raw-feature bypass into the coupling point, with feature-wise
> normalisation: +0.0191 IC over the tuned model, significant under four
> tests and selected on validation.

p23 結果表加一列 `MAGNET-v2 w/o skip (ablation) | This work | 10 | +0.0248 |
0.0078 | +0.0345`，緊接在 `MAGNET (current)` 下方。沒有這一列，讀者看不出
+0.0439 裡有多少來自跳接。

### 6.7 一句口試用的說法

> The encoder compresses 30 US stocks into what is numerically one vector —
> cosine +0.99997 between any two nodes. A three-dimensional bypass carrying
> 1.3% of the energy restores the cross-section and is worth +0.0191 IC.

---

## 7. 一日預測的完整走查：2025-04-10 的 2330 與 2454

給 p15–p18 用的貫穿範例。所有數字取自 `runs/*tw50_T1F3bnl1_s42`，
且已與該 run 的 `predictions/test_predictions.csv` 逐位核對一致。

### 7.0 這一天的時序

| 時間（台北） | 事件 |
|---|---|
| 2025-04-09 台股收盤 | 2330 −3.87%、2454 −6.14%，RSI_14 掉到 22.7 / 27.1 |
| 2025-04-10 04:00 | 美股 4/9 收盤：TSM log_return **+0.1160** |
| 2025-04-10 09:00 | 台股開盤 —— 模型要在此之前給出 50 檔的排序 |
| 2025-04-10 收盤 | 2330 +9.47%、2454 +9.26%、50 檔平均 +7.90% |

特徵一律取 `date < target_date` 的最後一列，兩市場都停在 04-09，無前視。

### 7.1 Phase 1 — Dual-market encoding

輸入（T=1，3 特徵）：

| node | log_return | RSI_14 | BB_pos |
|---|---:|---:|---:|
| TSM (L1) | +0.1160 | 42.49 | +0.3153 |
| 2330 (L2) | −0.0387 | 22.72 | −0.1778 |
| 2454 (L2) | −0.0614 | 27.08 | −0.1749 |

當日圖：L1 802 條邊、L2 2,200 條邊。

| node | 入度 | 最強鄰居（\|rho\|） |
|---|---:|---|
| TSM | 27 | SMH .891, NVDA .865, ASX .858, AVGO .818, MRVL .801 |
| 2330 | 44 | 2317 .791, 2308 .753, 3711 .752, 2382 .738, 2354 .715 |
| 2454 | 45 | 3008 .736, 4938 .700, 2317 .682, 3711 .682, 2330 .663 |

TSM 的編碼路徑：

```
x (3)            ->  LSTM (64)  ->  GATv2 (64)  ->  proj (29)   範數 3.6500
x (3)            ->  BN_F -> Linear(3->3)  =  [0.955, -0.641, -0.104]  範數 1.1552
h_L1[TSM] (32)   =  concat(proj29, skip3)                       範數 3.8284
```

**當天 30 檔美股的 proj29 兩兩餘弦 = +1.00000。** 也就是說「TSM 昨夜漲 12%」
這件事在投影後完全消失，只剩下 3 維跳接（節點間餘弦 +0.8851）還帶著它。
跳接佔 TSM 的 h_L1 能量 9.1%（全體全期平均 1.3%）。

### 7.2 Phase 2 — Two-tier coupling and gated fusion

| | 2330（有配對 TSM） | 2454（無配對） |
|---|---:|---:|
| identity 項 | h₁[TSM]，範數 **3.8284** | 0 |
| candidate 項 | 範數 1.2420 | 範數 **0.3480** |
| ĥ₁ 範數 | 5.0691 | 0.3480 |
| 自身 h_L2 範數 | 3.9308 | 3.9308 |
| 最強候選邊 | AMD .0181, NVDA .0178, MRVL .0170 | MRVL .0061, AMD .0057, ASML .0052 |
| B 欄和 | +0.3197 | +0.0889 |

兩檔拿到的跨市場輸入差 **14.6 倍**，全部來自恆等邊的有無。

gate：

| | 均值 | min | max | h_f 範數 |
|---|---:|---:|---:|---:|
| 2330 | 0.5030 | 0.4826 | 0.5511 | 3.5910 |
| 2454 | 0.5015 | 0.4777 | 0.5292 | 2.0156 |

全體 50 檔 gate 均值 0.5016、跨節點 std 0.00046。**gate 幾乎沒有分化**：
兩檔用的是同一組混合權重，h_f 的差異完全來自 ĥ₁ 的大小，不是 gate 的決策。

### 7.3 Phase 3 — Rank-oriented prediction

`Linear(32,64) -> ReLU -> Dropout(0.2) -> Linear(64,1)`

| | ŷ | 預測名次 | 實際報酬 | 實際名次 |
|---|---:|---:|---:|---:|
| 2330 | +0.01451 | **1 / 50** | +9.47% | 12 |
| 2454 | −0.00191 | 28 / 50 | +9.26% | 32 |

當日 IC **+0.2078**、RankIC +0.1357。預測前 5：2330, 3711, 2303, 2892, 2890
（其中 2330 / 3711 / 2303 都是有 ADR 配對的）。
std(ŷ) = 0.00346 對 std(y) = 0.02401，比值 0.144 —— variance 項沒有把尺度拉上來，
但 IC 對尺度不敏感，不影響排序。

### 7.4 反事實：這一天的訊號從哪來

同一組權重，只把跨層輸入置零：

| 設定 | 當日 IC | 2330 ŷ / 名次 | 2454 ŷ / 名次 |
|---|---:|---:|---:|
| 完整模型 | **+0.2078** | +0.01451 / 1 | −0.00191 / 28 |
| 拿掉恆等邊 | −0.0607 | −0.00209 / 38 | −0.00191 / 28 |
| 拿掉候選邊 | +0.1365 | +0.00864 / 3 | −0.00191 / 43 |
| 完全沒有跨市場 | −0.4857 | −0.00191 / 40 | −0.00191 / 39 |

最後一列的兩個 ŷ 幾乎相同，不是巧合：沒有跨市場輸入時 50 檔的預測全部擠在
[−0.00195, −0.00120]，橫截面 std 只剩完整模型的 2.1%。

### 7.5 同樣的拆解，10 顆種子 × 246 天

| 設定 | mean IC | sd | 對完整模型 | 配對 t |
|---|---:|---:|---:|---:|
| 完整模型 | +0.0475 | 0.0180 | — | — |
| 拿掉恆等邊 | +0.0155 | 0.0182 | **−0.0320** | **6.9e-06** |
| 拿掉候選邊 | +0.0463 | 0.0067 | −0.0012 | 0.79 |
| 完全沒有跨市場 | −0.0029 | 0.0030 | −0.0504 | — |

三個結論：

1. **恆等邊是承重結構。** 拿掉後 IC 掉 67%，p = 6.9e-06。
2. **候選邊是惰性的。** 差異 −0.0012、p = 0.79，10 顆種子裡有 6 顆拿掉後反而更好。
   **簡報 p21 的 H2（候選邊為未配對公司帶來訊號）判定為未獲支持。**
3. **台股層自己沒有任何橫截面排序能力。** 切斷跨市場後 IC = −0.0029，
   預測的橫截面 std 只剩 2.1%。模型全部的排序能力都經由跨市場路徑產生。

> 揭露：本表的「完整模型」為 +0.0475，而 `results_table.md` 報的是 +0.0439。
> 兩者差異源於 3 顆種子的 checkpoint 重評結果與其 `meta.json` 紀錄不一致
> （見 §8）。四個設定都用同一批 checkpoint 計算，故欄間的**差值**不受影響。

### 7.6 口述順序建議

p15 講輸入與編碼，停在「proj29 餘弦 +1.00000」；p16 講兩層邊，停在
「2330 拿到 3.83，2454 拿到 0.35」；p17 講 gate，停在「0.5030 對 0.5015」；
p18 講預測與當日 IC，最後用 7.4 的反事實表收尾。

---

## 9. Phase 2 / Phase 3 的逐運算子拆解（2025-04-10）

接續 §7 的走查與 §8 的 Phase 1 細節。同一個 run、同一天。

### 9.1 秩的帳（全部在同一個節點集上量，246 天平均）

| 表示 | 節點數 | 有效秩 |
|---|---:|---:|
| h_L1 耦合點（美股側） | 30 | 1.691 |
| **ĥ₁ 耦合後（台股側）** | 50 | **1.054** |
| h₂ 台股層 | 50 | 1.005 |
| h_f 融合後 | 50 | 1.096 |
| ReLU(W₁h_f + b₁) 預測頭隱藏層 | 50 | **1.568** |

**更正：秩是在耦合這一步掉的，不是融合。** 先前把美股側的 1.691 與台股側的
1.096 相比，兩者節點集不同，比較無效。正確的讀法是 1.691 → 1.054：把 30 個
近共線的美股向量映到 50 個台股槽位時，43 個槽位只拿到「一個純量 × 共同方向」。
融合反而讓秩微升（1.054 → 1.096），預測頭的 ReLU 才是真正產生秩的地方
（1.096 → 1.568）。

### 9.2 Phase 2

**恆等邊**：純索引，零參數。`ident[2330]` 與 `h_L1[TSM]` 逐位相同
（max\|diff\| = 0.00e+00）。43 檔無配對者為零向量。

**候選邊**：

| | B 欄和 | 非零 | 最強 6 條 | cand 範數 |
|---|---:|---:|---|---:|
| 2330 | +0.3197 | 29/30 | AMD .0181, NVDA .0178, MRVL .0170, LRCX .0169, AMAT .0164, ASML .0157 | 1.2420 |
| 2454 | +0.0889 | 30/30 | MRVL .0061, AMD .0057, ASML .0052, TER .0051, MU .0051, SMH .0050 | 0.3480 |

B 本身**不是** rank-1：奇異值 [0.1415, 0.0446, 0.0133, ...]，有效秩 1.737，
最佳 rank-1 近似誤差 35.5%，50 個欄向量兩兩餘弦 −0.019（近乎正交）。

但它的輸出是 rank-1：`cos(cand_j, (Σ_i B[i,j])·h̄)` 全期平均 **+0.9986**。

> **B 學到了結構，卻表達不出來，因為它乘的東西是 rank-1 的。**
> 1,493 條邊實際傳遞的自由度是每檔一個純量，共 50 個。

**gate**：`|W_g| = 0.2472`、`|b_g| = 0.0475`、`sigmoid(b_g) = 0.4998`。
2330 的 logit 均值 +0.0119（範圍 −0.070 ~ +0.205），2454 是 +0.0061。
跨節點 logit std 0.00583。sigmoid 在 0 附近近似線性（g ≈ 0.5 + logit/4），
logit 只有 ~0.026 量級，所以 g 動不了。

### 9.3 Phase 3

`Linear(32,64) → ReLU → Dropout(0.2, eval 關閉) → Linear(64,1)`

ReLU 平均活化 **24.7 / 64** 個單元，且**只有 39% 的節點與節點 0 共用同一組
活化樣式** —— 預測頭是全模型唯一給節點差別待遇的地方，也是唯一把秩拉高的
運算子。

當日輸出：

| | pre-ReLU 均值 | ŷ | 預測名次 | 實際 | 實際名次 |
|---|---:|---:|---:|---:|---:|
| 2330 | −0.0158 | +0.014510 | 1 | +9.47% | 12 |
| 2454 | −0.0057 | −0.001908 | 28 | +9.26% | 32 |

ŷ 橫截面 mean −0.00088、std 0.00342、範圍 [−0.00531, +0.01451]；
y 橫截面 mean +0.0790、std 0.02401。std 比值 0.143。

當日損失分量（若此批為訓練批）：

| 分量 | 原值 | 權重 | 加權後 | 佔損失值 |
|---|---:|---:|---:|---:|
| MSE | 0.006937 | 1.0 | 0.006937 | 2.0% |
| rank | 0.692593 | 0.5 | 0.346296 | **98.0%** |
| variance | 0.000424 | 0.1 | 0.000042 | 0.0% |
| L1 on B | 4.725 | **0** | 0 | — |

損失**值**的佔比 2 / 98 / 0，梯度佔比 27 / 71 / 2。簡報寫的 1 : 0.5 兩者皆非。
