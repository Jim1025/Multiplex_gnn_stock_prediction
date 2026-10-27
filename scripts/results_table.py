"""
results_table.py — 產出跨方法的總結果表（docs/results_table.md）

收三類方法，全部在同一個 tw50 walk-forward test（246 天）上：
  1. 神經模型：本專案的 MAGNET 各版本 + 六個文獻 baseline，逐 seed 平均
  2. 線性/樹模型：ridge 階梯、factor_vs_graph、relational_linear（§67）、[24] 二部圖重現，單一預測檔
  3. 空模型：RC 常數對照

另有一節準度小表（2026-10-13 加，_acc_block）：每個模型先在 val 上定尺度，再比 test 的
RMSE / R²_oos / 命中率。線性 baseline 的 val 預測由 tw_lag_channel / relational_linear 重算，
所以這一節需要資料快照；缺的時候只跳過這一節。

統計一律以目前最佳的 MAGNET（tw50_T1F3bnl1）為基準：
  - 神經 arm：取共同 seed，逐日 IC 序列先跨 seed 平均再配對檢定，
    另報跨 seed 的 Welch 與 Mann-Whitney（後者在 n<5 時無法達到 p<0.05）
  - 非神經：只有一條預測序列，僅報逐日配對

用法：
    .venv/bin/python scripts/results_table.py            # 印出並寫檔
    .venv/bin/python scripts/results_table.py --no-write
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 主結果 arm。2026-08-30 由 tw50_beta 換成 tw50_betaF1nA2r1
# （= beta 層 + F1 只留 log_return + 關掉台股層圖 A₂ + rank 損失權重 1.0）：
#   - 第一折 IC +0.1011 -> +0.1090、RankIC +0.0967 -> +0.1098，10/10 種子
#   - **第二折（train 0-902 / 評估窗 2023-12-20 ~ 2024-12-25，與第一折不重疊）
#     ΔIC +0.0308、ΔRankIC +0.0262，10/10 種子，且逐日檢定兩個指標都過
#     （p 0.0004 / 0.0036）** —— 本專案第一次有架構改動在逐日層級對自家基準顯著
#   - 換基準的依據是 proposal §36 事先登記的判準（§38 判定通過最高一檔）
#   - **不可寫「超越線性 baseline」**：第二折對 KTW+ 的 RankIC 是 −0.0115，
#     對 R2 ridge 恰好打平（+0.0000），IC 雖名目領先但無一顯著（§40）
#   - 須揭露的退步：過度離散再惡化，std 比 1.69 -> 2.59、MSE 0.00106 -> 0.00204
#   - 三個改動是**兩兩協同，沒有三階交互作用**（§35.3，2026-09-15 補完 2^3）。
#     主效應 F1 +0.0050 p=0.026、rank1.0 +0.0037 p=0.016、無A₂ −0.0006 ns
# 2026-10-27：主結果改為 tw50_rgcnSA（MAGNET + 同產業關係，§80、§83）。換的理由是結構：台股層內圖有作用、
# 兩個市場層都有層內圖模組；SA 是看過 §79 的消融後選的（事後選擇），數字偏高，對原主結果的 IC 沒有可靠差別。
# 原主結果（不帶關係，上面那段紀錄的 arm）保留為一列「MAGNET 不帶關係（原主結果）」。
BEST = "tw50_rgcnSA"            # 主結果，所有比較的基準（見上）
BEST_F2 = "f2_rgcnSA"           # 第二折的主結果
PRIOR = "tw50_betaF1nA2r1"      # 原主結果（不帶關係），2026-10-27 以前的基準
PRIOR_F2 = "f2_best"
MAIN_LABEL = "MAGNET（+ 同產業關係，主結果）"
PRIOR_LABEL = "MAGNET 不帶關係（原主結果）"
# 主結果自己的單市場消融（2026-10-27 事前登記，proposal §83.3）。依登記不進主表，只在「讀表注意」說明，數字由 run 算
SMKT_SA = ("tw50_smktSA", "f2_smktSA")

# (顯示名稱, arm 或 glob, 類別, 備註)
# 子列（「└」）的名稱要自己看得懂：表依 IC 排序，子列常不在母列正下方，
# 「同上」會指到錯的列（2026-10-17 [24] 全部列出後更明顯），所以寫出母版本。
NEURAL = [
    # 以下六格 + tw50_beta + tw50_beta_g_empty_l2 構成 §35.3 的 2^3 完整設計
    # （A=F1 / B=無A₂ / C=rank1.0）。原本的備註「三者缺一不可」已被 2026-09-15
    # 的分解證偽：三階項是零，F1 與 rank1.0 各有顯著主效應。
    (PRIOR_LABEL, PRIOR, "本專案",
     "F1 + 無A₂ + rank 1.0，2^3 的 (1,1,1)，§35.3；2026-10-27 以前的主結果"),
    # §71（2026-10-17）：登記外的探索，依登記的主表規則不論結果都列入。登記判準未通過
    # （兩折 ICIR 同為正，第一折 p 0.056），主結果 arm 不變。
    ("MAGNET + 關係型台股層（投影後版本）", "tw50_relTW", "本專案",
     "同產業 / 其他產業的鄰居平均 x W_r；探索性，判準未通過，§71"),
    # §80（2026-10-24）：使用者決定把關係型台股層簡化為「自身 + 同產業」（§79 的 SA arm，不重訓），
    # 這一列由 §78 的完整 2b（tw50_rgcnTW）換成 SA。SA 是看過 §79 的消融之後選的，數字偏高，不能當證據；
    # §78 的完整 2b 數字留在 proposal §78.6。
    (MAIN_LABEL, BEST, "本專案",
     "台股層內圖：同產業關係卷積（取代 GAT_L2）；事後選擇，數字偏高，§80、§83"),
    ("　└ F1 + rank 1.0（保留 A₂）", "tw50_betaF1r1", "本專案",
     "(1,0,1)，2^3 補格"),
    ("　└ 只有 rank 1.0（保留 A₂ 與 F3）", "tw50_betaR1", "本專案",
     "(0,0,1)，只調 rank 權重"),
    ("　└ 無A₂ + rank 1.0（F3）", "tw50_betanA2r1", "本專案",
     "(0,1,1)，2^3 補格"),
    ("　└ F1 + 無A₂（rank 0.5）", "tw50_betaF1nA2", "本專案",
     "(1,1,0)，未加 rank 權重"),
    ("　└ F1（單獨）", "tw50_betaF1", "本專案", "(1,0,0)，只換特徵"),
    ("MAGNET + beta 層（前版）", "tw50_beta", "本專案",
     "(0,0,0) 基準，每檔一個 alpha_j / gamma_j"),
    ("　└ beta 層前版，B 降秩 r=3", "tw50_betaLR3", "本專案", "B 參數 1500 -> 240，null"),
    ("　└ beta 層前版，per-target 讀出", "tw50_betaPT", "本專案", "已證偽，−0.0093"),
    ("　└ beta 層前版，多因子 k=3", "tw50_betaK3", "本專案", "已證偽，null"),
    ("MAGNET + 輸入正規化 + AdamW（前版）", "tw50_inbnw_noskip", "本專案",
     "F3, T=1, L=1, 無跳接"),
    ("　└ 輸入正規化前版，改用 Adam", "tw50_inbn_noskip", "本專案", "分離 AdamW 的貢獻"),
    ("　└ 輸入正規化前版，Adam + 拼接跳接", "tw50_inbn_skip", "本專案", "跳接是否仍有增量"),
    ("　└ 輸入正規化前版，F9", "tw50_inbnw_f9_noskip", "本專案", "特徵數是否仍無差異"),
    ("MAGNET + BN 拼接跳接（前版）", "tw50_T1F3bnl1", "本專案", "F3, T=1, L=1"),
    ("MAGNET 原始（9 特徵 T=20 L=2）", "tw50chk_wl0",  "本專案", "F9, T=20, L=2"),
    ("MAGNET F9 T=1 L=1",            "tw50_T1F9",     "本專案", "F9, T=1, L=1"),
    ("MAGNET F3（無跳接）",            "tw50_T1F3rb",   "本專案", "F3, T=1, L=1"),
    ("MAGNET + 稠密耦合 A",            "tw50_T1F9dense","本專案", "A[30,50] 可學"),
    ("Early fusion 對照",             "tw50chk_ef",    "本專案", "拼接後單一編碼器"),
    ("單市場消融（不帶關係的版本）", "tw50_smktA", "本專案",
     "disable_a12，跨市場全關"),
    ("單市場消融（前一版 arm）", "tw50_smktB", "本專案",
     "disable_a12，台股側 F3 + 有圖"),
    ("LSTM only（無圖）",              "tw50chk_lstm",  "本專案",
     "另一架構，非對等對照"),
    ("HGT [14]",                      "tw50_bl_hgt",       "文獻", "未調參"),
    ("DeltaLag [13]",                 "tw50_bl_delta_lag", "文獻", "未調參，預測退化"),
    ("MEIG [1]",                      "tw50_bl_meig",      "文獻", "未調參"),
    ("Adv-ALSTM [10]",                "tw50_bl_adv_alstm", "文獻", "未調參"),
    ("HATS [12]",                     "tw50_bl_hats",      "文獻", "未調參"),
    ("MAN-SF [11]",                   "tw50_bl_man_sf",    "文獻", "未調參"),
]
# [24] 兩折一律用滾動版（原文做法：250 天視窗、每 10 天重訓；2026-10-16 起統一，之前第一折用同協定版）。
# runs/ 與 runs_f2/ 裡有多批同名的 [24] run（含 2026-08-14 修正 winsorize 前視之前的舊版），
# 一律只取最新一批：主表第一折的 nonneural_stats 本來就取最新，第二折與 §43 經 _latest 解析，
# 避免把多批 run 當成多顆種子平均（§43 (B) 第一折原本就因此混入了 4 批）。
BIP = "*_bipartite_rolling_t2_{}"
# [24] 的 10 個模型（2026-10-17 起主表全部列出，依教授建議看上下限，§70、§73）：
# 8 個單一模型與兩種集成，兩折的顯示名稱相同。原本只列 LASSO 與 ens-avg。
BIP_MODELS = (("ens-avg", "8 個單一模型的平均（原文的集成）"),
              ("ens-med", "8 個單一模型的中位數"),
              ("LASSO", "單一模型"), ("RIDGE", "單一模型"), ("OLS", "單一模型"),
              ("RF", "單一模型"), ("AdaBoost", "單一模型"), ("XGBoost", "單一模型"),
              ("LGBM", "單一模型"), ("SVM", "單一模型"))


def bip_label(m: str) -> str:
    return f"[24] 二部圖 {'LightGBM' if m == 'LGBM' else m}"


NONNEURAL = [(bip_label(m), BIP.format(m), "文獻", note) for m, note in BIP_MODELS] + [
    ("KTW+（rank 目標）",     "*_fvg_KTWp",                          "線性", "美股+台股 80 維"),
    # §67（2026-10-13 加，scripts/relational_linear.py）。登記的確認未通過，
    # 依登記的規則仍列入：關係型的 IC 兩折都高於 KTW+、ICIR 只有第二折較高，兩者並列最高標；
    # 兩區塊的 IC 兩折也都高於 KTW+，依同一個公平原則列入。
    ("關係型線性（z 目標）",   "*_lin_relational",                    "線性", "美股自由 + 台股 3 個關係權重，§67"),
    ("兩區塊 ridge（z 目標）", "*_lin_twoblock",                      "線性", "美股、台股各自一個 alpha，§67"),
    ("R2（30 檔美股報酬）",    "*_fvg_R2",                            "線性", "per-target ridge"),
    ("RC 常數對照",           "*_ridge_RC",                          "空模型", "0 特徵"),
]
# 種子集成列（2026-10-11 加）：主結果 arm 的全部種子先逐 (日, 檔) 平均 y_hat、再算所有欄位，
# 與 §43 (A) 的「先平均預測再算 IC」是同一個數字。n 記平均了幾顆種子，但只有一條序列，
# 所以沒有 ± sd、跨種子欄為「—」；逐日 p 是它對統計基準（同一個 arm 逐 seed 平均）的
# 配對檢定，也就是集成增益本身的顯著性。統計基準仍是逐 seed 平均那一列（BEST）。
ENSEMBLE = [
    ("MAGNET（+ 同產業關係，10 顆集成）", BEST, "本專案", "主結果的 10 顆先平均預測再算 IC，§43 (A)"),
]
# 主表 / 附表（2026-10-16，依教授建議）：主表只放 MAGNET 自身的各版本與消融，以及文獻方法；
# 我們另外建構、又不是出自文獻的對照（同輸入的線性模型與空模型）移到附表。
# 單市場消融、LSTM only、Early fusion 對照是拿掉或替換 MAGNET 元件的消融，留在主表。
# 統計基準與所有檢定不變（附表的 p 仍是對 MAGNET 本版）；附表另列 MAGNET 本版與集成當參照。
# 以顯示名稱比對，兩折的名稱都列在這裡（第二折的 R2 / TW+ 名稱不同）。
SUPPLEMENT = frozenset({"KTW+（rank 目標）", "關係型線性（z 目標）", "兩區塊 ridge（z 目標）",
                        "R2（30 檔美股報酬）", "R2 per-target ridge", "TW+（美股+台股 80 維）",
                        "RC 常數對照"})


def _latest(pattern: str) -> str | None:
    """glob 到多個目錄時只取最新一批（目錄名以時間戳開頭，排序即時間序），回傳相對 ROOT 的路徑。"""
    ds = sorted(glob.glob(str(ROOT / pattern), recursive=True))
    return os.path.relpath(ds[-1], ROOT) if ds else None


# 預測檔來源。reeval = CPU 重評版（scripts/reeval_checkpoints.py 產生），
# recorded = run 當下寫的 CSV。含 GAT 的 arm 在 MPS 上的 scatter-add 不可重現，
# 同一份 best.pt 連評 5 次 test IC 全距 0.0169，所以本表預設走 reeval；
# 無重評檔的 arm（無 GAT，或本來就在 CPU 上評的新 run）自動回退到 recorded。
PREDICTIONS = "reeval"


TOPK = 10       # 與 scripts/portfolio_readout.py 的 TOPK_DEFAULT 同值。
                # 這是「讀法」的選擇不是模型超參，故不從 base.yaml 讀。
ANN = float(np.sqrt(252.0))

# 砍掉的兩欄是純換算，留著只是同一個數字換單位：
#   年化(簡單) = 日均報酬 x 252      年化波動 = 日 sd x sqrt(252)
# 兩者都能從留下的欄位還原，Sharpe 則是新資訊（兩者之比）。
PF_HEAD = ("| 日均報酬 ↑ | 日 sd ↓ | 年化(幾何) ↑ "
           "| Sharpe ↑ | MDD(複利) ↓ | MDD(加總) ↓ ")
PF_SEP = "|---:|---:|---:|---:|---:|---:"
PF_BLANK = "| — | — | — | — | — | — "

CAL_HEAD = ("| MSE (1e-3) ↓ | RMSE (%) ↓ | 離散比 (->1.0) | 校正後 R² ↑ ")
CAL_SEP = "|---:|---:|---:|---:"
CAL_BLANK = "| — | — | — | — "
# 主指標（2026-10-17，依教授建議只留一到兩個指標）：主表只報 IC（主指標）與 RankIC（第二指標）
# 及兩者的逐日檢定；ICIR、Rank ICIR 與跨種子檢定移到附錄 A（OTHER_HEAD），數字與檢定都不變。
# 第二折的表頭（主表與附錄 B 共用）
F2_MAIN_HEAD = ["| 方法 | n | IC ↑ | RankIC ↑ | 逐日 p (IC) | 逐日 p (RankIC) |",
                "|---|---:|---:|---:|---:|---:|"]
F2_OTHER_HEAD = ["| 方法 | n | ICIR ↑ | Rank ICIR ↑ |",
                 "|---|---:|---:|---:|"]


def topk_ret(p_hat, y, k: int = TOPK) -> float:
    """Top-k 多空、等權、金額中性的當日報酬。

    權重是 ±1/(2k)，所以 sum|w| = 1（總曝險 1、淨曝險 0）；
    因此 (多腿均值 − 空腿均值) 要再除以 2 才是「單位本金」的報酬。
    不除等於偷偷假設 2 倍槓桿。
    """
    a = np.asarray(p_hat, float)
    b = np.asarray(y, float)
    o = np.argsort(-a)
    return float((b[o[:k]].mean() - b[o[-k:]].mean()) / 2.0)


def _mdd(curve) -> float:
    peak = np.maximum.accumulate(curve)
    return float(((peak - curve) / peak).max())


def portfolio_stats(series):
    """一批（逐 seed 的）Top-K 日報酬序列 -> 八個組合指標。

    逐 seed 算完再平均，與 ICIR 同一個慣例。

    **y 是 log return**（`src/dataset/pipeline.py:659` 的
    `log(Close/Close.shift(1))`），所以複利不能用 (1+r) 連乘：

        年化(幾何) = exp(mean(r) x 252) − 1
        MDD(複利)  的權益曲線 = exp(cumsum(r))

    MDD(加總) 用 1 + cumsum(r)，即獲利不滾入的固定名目本金——
    金額中性多空每天重設回同樣的名目曝險，這個讀法才對應實際操作。

    已知近似：對 log return 取橫截面平均**不等於**等權組合的報酬
    （後者是簡單報酬的算術平均）。實測差 +0.14 bp/日 = +0.36 pp/年，
    不影響 Sharpe，也不影響 §60.4 的打平成本（兩種算法都是 33 bp）。

    **這不是回測。** 無成本、無衝擊、無流動性限制、每日全額換手、
    可完全放空，而且時序上不可執行（訊號在隔夜跳空發生之後才存在，
    proposal §61.3(2)）。用途是把 IC 翻譯成組合單位，見 proposal §60.6。
    """
    rows = []
    for r in series:
        r = np.asarray(r, float)
        r = r[~np.isnan(r)]
        if r.size < 2:
            continue
        mu, sd = float(r.mean()), float(r.std(ddof=1))
        rows.append((mu, sd, mu * 252.0, float(np.exp(mu * 252.0) - 1.0),
                     sd * ANN, (mu / sd * ANN) if sd > 1e-15 else np.nan,
                     _mdd(np.exp(np.cumsum(r))), _mdd(1.0 + np.cumsum(r))))
    if not rows:
        return None
    m = np.array(rows, float).mean(axis=0)
    return dict(zip(("r_mu", "r_sd", "ann_s", "ann_g", "vol",
                     "sharpe", "mdd_c", "mdd_a"), m.tolist()))


def pf_cells(pf) -> str:
    if pf is None:
        return PF_BLANK
    return (f"| {pf['r_mu'] * 100:+.4f}% | {pf['r_sd'] * 100:.4f}% "
            f"| {pf['ann_g'] * 100:+.1f}% | {pf['sharpe']:.2f} "
            f"| {pf['mdd_c'] * 100:.2f}% | {pf['mdd_a'] * 100:.2f}% ")


def calib_stats(Hn, Yn) -> dict | None:
    """校準診斷：MSE / RMSE / 離散比 / 校正後 R²。

    **為什麼這四個要一起看。** §60.1 宣告 ŷ 是「基數分數、**尺度未校正**」，
    所以 MSE / RMSE 量到的主要是**尺度誤差**，不是排序品質。單看它們會得到
    與 IC 幾乎相反的排名——實測 IC 最好的 arm（+0.1090）MSE 是 IC 最差那一版
    （+0.0517）的 4.6 倍，而兩者的 R² 都是負的（−4.03 / −0.10）。
    後兩欄就是把那個負 R² 拆成可讀的部分：

      離散比   = mean_t[ std(ŷ_t) / std(y_t) ]，逐日算完再平均。
                 與 §60.7 / §52 引用的 **2.59**（主結果 arm）與 **1.69**
                 （前版 beta）同定義。**注意它不等於** src/train/metrics.py 的
                 `dispersion`——那個是攤平的，同一個 arm 給 2.13；差在攤平版
                 把大盤的跨日變動也算進 std。
      校正後 R² = **先把每天的橫截面均值從 ŷ 與 y 各自扣掉**，再攤平配
                 OLS（y ~ a·ŷ + b）之後的 level R²，代數上等於扣完均值的
                 `corr²`。列它是為了與 Gu/Kelly/Xiu 那類報 level R² 的文獻
                 對照，並且把一件事講明：ŷ 的數值可預測性大約就是 IC² 的量級。

                 **逐日扣均值是必要的，不是口味問題。** 不扣的話 `corr²` 會
                 混進「有沒有預測到當天的大盤方向」——實測 `R2 ridge` 的
                 逐日橫截面均值與 y 的均值相關 +0.5144，把它的攤平 R² 抬到
                 0.0940，比 IC 更高的本專案主結果（0.0173）還高三倍多。
                 扣完之後兩者分別是 0.0171 與 0.0140，回到與 IC 一致的次序。
                 本表所有其他指標都是橫截面的，這一欄也必須是。

                 它**不等於** `mean_t(IC_t)²`（本表主結果 0.0135 vs 0.0119）：
                 攤平的 corr 用橫截面離散度給每天加權，`mean IC` 每天等權。

    MSE / RMSE 攤平計算，與 `src/train/metrics.py:regression_metrics` 同。
    讀 MSE 需要錨，所以本函式一併回傳 `var_y` / `sd_y`，由表下的註腳印出。
    """
    H = np.asarray(Hn, float)
    Y = np.asarray(Yn, float)
    ok = np.isfinite(H) & np.isfinite(Y)
    if ok.sum() < 3:
        return None
    hf, yf = H[ok], Y[ok]
    d = hf - yf
    mse = float((d * d).mean())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sh, sy = np.nanstd(H, axis=1), np.nanstd(Y, axis=1)
    good = np.isfinite(sh) & np.isfinite(sy) & (sy > 1e-15)
    disp = float(np.mean(sh[good] / sy[good])) if good.any() else float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Hd = H - np.nanmean(H, axis=1, keepdims=True)
        Yd = Y - np.nanmean(Y, axis=1, keepdims=True)
    hd, yd = Hd[ok], Yd[ok]
    r = (float(np.corrcoef(hd, yd)[0, 1])
         if hd.std() > 1e-15 and yd.std() > 1e-15 else float("nan"))
    return dict(mse=mse, rmse=float(np.sqrt(mse)), disp=disp,
                r2cal=float(r * r) if np.isfinite(r) else float("nan"),
                var_y=float(yf.var()), sd_y=float(yf.std()))


def cal_cells(c, sd=None) -> str:
    if c is None:
        return CAL_BLANK
    g = (lambda k: (sd or {}).get(k, float("nan")))
    return (f"| {pm(c['mse'] * 1e3, g('mse') * 1e3, 3, False)} "
            f"| {pm(c['rmse'] * 100, g('rmse') * 100, 3, False)} "
            f"| {pm(c['disp'], g('disp'), 3, False)} "
            f"| {pm(c['r2cal'], g('r2cal'), 4)} ")


def cal_mean_sd(cals):
    """一批（逐 seed 的）calib_stats -> (均值 dict, 跨 seed sd dict)。"""
    cals = [c for c in cals if c]
    if not cals:
        return None, None
    ks = ("mse", "rmse", "disp", "r2cal", "var_y", "sd_y")
    A = {k: np.array([c[k] for c in cals], float) for k in ks}
    return ({k: float(np.nanmean(v)) for k, v in A.items()},
            {k: (float(np.nanstd(v, ddof=1)) if v.size > 1 else float("nan"))
             for k, v in A.items()})


def _pred_file(run_dir: str) -> str | None:
    """該 run 的 test 預測檔：預設走 CPU 重評版，沒有時回退到 run 當下寫的 CSV。"""
    f = os.path.join(run_dir, "predictions", "test_predictions_reeval.csv")
    if PREDICTIONS != "reeval" or not os.path.exists(f):
        f = os.path.join(run_dir, "predictions", "test_predictions.csv")
    return f if os.path.exists(f) else None


def daily_series(run_dir: str):
    f = _pred_file(run_dir)
    if f is None:
        return None
    return _frame_series(pd.read_csv(f))


def _frame_series(df: pd.DataFrame):
    """預測表（target_date, ticker, y_hat, y）-> (日期, 逐日 IC, 逐日 RankIC, 逐日 Top-K 報酬, 校準)。"""
    H = df.pivot(index="target_date", columns="ticker", values="y_hat").sort_index()
    Y = df.pivot(index="target_date", columns="ticker", values="y").sort_index()
    Hn, Yn = H.to_numpy(), Y.to_numpy()
    ic = np.full(len(Yn), np.nan)
    ric = np.full(len(Yn), np.nan)
    rk = np.full(len(Yn), np.nan)          # Top-K 多空的日報酬（組合欄用）
    for t in range(len(Yn)):
        a, b = Hn[t], Yn[t]
        # 50 檔完全相同（相關係數無定義）才略過。不用 std == 0：同值的 std 因捨入不一定恰為 0，
        # 結果隨加總順序與 pandas 是否裝了 bottleneck 而不同（2026-10-21，§77）。
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            continue
        ic[t] = np.corrcoef(a, b)[0, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = stats.spearmanr(a, b).statistic
        ric[t] = np.nan if np.isnan(v) else v
        rk[t] = topk_ret(a, b)
    return H.index.to_numpy(), ic, ric, rk, calib_stats(Hn, Yn)


def icir(arr) -> float:
    """ICIR = mean(逐日 IC) / sd(逐日 IC)。

    與 `src/train/metrics.py:176` 同一個定義（ddof=1），只是多了 nan-aware，
    因為 `daily_series` 會把 50 檔 y 或 y_hat 完全相同的日子標成 NaN
    （第一折有 1 天 50 檔報酬完全相同）。

    為什麼要報這個：§57.8 的恆等式是 r_t = IC_t x sigma_t，所以
    **IC 只決定組合報酬的分子**。若 sigma_t 為常數，
    Sharpe = ICIR x sqrt(252)——與 Sharpe 對應的是 ICIR，不是 IC。
    實測（proposal §60）：本專案最佳 arm 的 IC 高於 KTW+，ICIR 卻較低
    （0.4368 vs 0.4810），Top-10 多空的 Sharpe 也較低（5.86 vs 6.53）。
    """
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    if a.size < 2:
        return float("nan")
    sd = a.std(ddof=1)
    return float("nan") if sd < 1e-12 else float(a.mean() / sd)


def find_seeds(arm: str) -> dict[str, str]:
    out = {}
    for d in sorted(glob.glob(str(ROOT / "runs" / "**" / f"*{arm}_s*"), recursive=True)):
        name = re.sub(r"^\d{8}_\d{4}_", "", os.path.basename(d))
        mm = re.match(rf"^{re.escape(arm)}_s(\d+)$", name)
        if mm and os.path.exists(os.path.join(d, "meta.json")):
            out[mm.group(1)] = d
    return out


def colmean(stack: np.ndarray) -> np.ndarray:
    out = np.full(stack.shape[1], np.nan)
    ok = ~np.isnan(stack).all(axis=0)
    out[ok] = np.nanmean(stack[:, ok], axis=0)
    return out


def neural_stats(arm: str):
    runs = find_seeds(arm)
    if not runs:
        return None
    ics, rics, rks, cals = [], [], [], []
    per_ic, per_ric, per_icir, per_ricir, params = [], [], [], [], None
    for s, d in sorted(runs.items(), key=lambda kv: int(kv[0])):
        tm = (json.load(open(os.path.join(d, "meta.json"))).get("test_metrics") or {})
        rv = os.path.join(d, "meta_reeval.json")
        if PREDICTIONS == "reeval" and os.path.exists(rv):
            tm = {**tm, **json.load(open(rv))["reevaluated"]}
        if tm.get("IC") is None:
            continue
        ser = daily_series(d)
        # RankIC 改由預測檔重算，不讀 meta.json。
        #
        # 2026-09-01：meta.json 的 RankIC 是用舊的 `argsort(argsort())` 取名次算的，
        # 對平手值給相異名次。實測 y（次日報酬）有 208/246 天存在平手——多半是
        # 恰好 0% 的股票——所以那個欄位一直帶著系統性偏差（proposal §44.7、D1）。
        # `daily_series` 用的是 scipy 的 spearmanr，平手取平均名次，是正確的。
        #
        # 影響：IC 逐位元不變（實測差 0.000000），RankIC 上移約 +0.0013 ~ +0.0017，
        # 各 arm 同向，差值幾乎不動（第二折對 KTW+ −0.0114 -> −0.0115），
        # 沒有任何結論翻轉。改完之後本表與 proposal 內文的 RankIC 才一致。
        per_ic.append(tm["IC"])
        per_ric.append(float(np.nanmean(ser[2])) if ser is not None else tm["RankIC"])
        # ICIR 逐 seed 算完再平均，不是拿跨 seed 平均後的日序列去算——
        # 後者是「集成」的 ICIR，系統性偏高（§43 的 A 區塊在講同一件事）。
        per_icir.append(icir(ser[1]) if ser is not None
                        else (tm.get("ICIR") if tm.get("ICIR") is not None else np.nan))
        # RankICIR 同慣例：逐 seed 用**重算的**逐日 RankIC 序列算，再跨 seed 平均。
        # 不讀 meta.json——那裡的 RankIC 是 pre-D1 的名次算法（§44.7）。
        per_ricir.append(icir(ser[2]) if ser is not None else np.nan)
        if ser is not None:
            ics.append(ser[1]); rics.append(ser[2]); rks.append(ser[3])
            cals.append(ser[4])
            dates = ser[0]
        if params is None:
            params = tm.get("n_params")
    if not per_ic:
        return None
    cal, cal_sd = cal_mean_sd(cals)
    return dict(seeds=sorted(runs, key=int), n=len(per_ic), cal=cal, cal_sd=cal_sd,
                ic=float(np.mean(per_ic)), ic_sd=float(np.std(per_ic, ddof=1)) if len(per_ic) > 1 else np.nan,
                ric=float(np.mean(per_ric)), ric_sd=float(np.std(per_ric, ddof=1)) if len(per_ric) > 1 else np.nan,
                icir=float(np.nanmean(per_icir)) if per_icir else np.nan,
                icir_sd=(float(np.nanstd(per_icir, ddof=1))
                         if sum(~np.isnan(per_icir)) > 1 else np.nan),
                ricir=float(np.nanmean(per_ricir)) if per_ricir else np.nan,
                ricir_sd=(float(np.nanstd(per_ricir, ddof=1))
                          if sum(~np.isnan(per_ricir)) > 1 else np.nan),
                per_ic=dict(zip(sorted(runs, key=int), per_ic)),
                per_ric=dict(zip(sorted(runs, key=int), per_ric)),
                daily_ic=colmean(np.vstack(ics)) if ics else None,
                daily_ric=colmean(np.vstack(rics)) if rics else None,
                pf=portfolio_stats(rks) if rks else None,
                dates=dates if ics else None)


def nonneural_stats(pat: str):
    ds = sorted(glob.glob(str(ROOT / "runs" / "**" / pat), recursive=True))
    if not ds:
        return None
    ser = daily_series(ds[-1])
    if ser is None:
        return None
    dates, ic, ric, rk, cal = ser
    return dict(n=1, cal=cal, cal_sd=None, ic=float(np.nanmean(ic)), ic_sd=np.nan,
                ric=float(np.nanmean(ric)), ric_sd=np.nan,
                icir=icir(ic), icir_sd=np.nan,
                ricir=icir(ric), ricir_sd=np.nan,
                daily_ic=ic, daily_ric=ric, pf=portfolio_stats([rk]),
                dates=dates, seeds=None)


def ensemble_stats(arm: str):
    """種子集成（見 ENSEMBLE）：全部種子的 y_hat 逐 (日, 檔) 平均成一條序列，
    之後與 nonneural_stats 同算法。n 為種子數；seeds=None，所以跨種子欄印「—」。"""
    frames = [pd.read_csv(f) for f in (_pred_file(d) for _, d in
                                       sorted(find_seeds(arm).items(), key=lambda kv: int(kv[0])))
              if f is not None]
    if not frames:
        return None
    df = (pd.concat(frames).groupby(["target_date", "ticker"])
          .agg(y_hat=("y_hat", "mean"), y=("y", "first")).reset_index())
    dates, ic, ric, rk, cal = _frame_series(df)
    return dict(n=len(frames), cal=cal, cal_sd=None, ic=float(np.nanmean(ic)), ic_sd=np.nan,
                ric=float(np.nanmean(ric)), ric_sd=np.nan,
                icir=icir(ic), icir_sd=np.nan,
                ricir=icir(ric), ricir_sd=np.nan,
                daily_ic=ic, daily_ric=ric, pf=portfolio_stats([rk]),
                dates=dates, seeds=None)


def paired_daily(base: dict, other: dict, key: str):
    """回傳 (差, HAC p)。差為正代表 base 較優。"""
    a, b = base["daily_" + key], other["daily_" + key]
    if a is None or b is None or len(a) != len(b):
        return np.nan, np.nan
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    n = len(d)
    lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    x = d - d.mean()
    var = float((x @ x) / n)
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1.0)) * float((x[k:] @ x[:-k]) / n)
    se = np.sqrt(max(var, 1e-24) / n)
    t = d.mean() / se
    return float(d.mean()), float(2 * stats.t.sf(abs(t), n - 1))


def across_seed(base: dict, other: dict, key: str):
    """同種子集合的跨 seed 檢定。回傳 (Welch p, Mann-Whitney p, n_common)。"""
    if base.get("seeds") is None or other.get("seeds") is None:
        return np.nan, np.nan, 0
    common = sorted(set(base["per_" + key]) & set(other["per_" + key]), key=int)
    if len(common) < 2:
        return np.nan, np.nan, len(common)
    a = np.array([base["per_" + key][s] for s in common])
    b = np.array([other["per_" + key][s] for s in common])
    _, pw = stats.ttest_ind(a, b, equal_var=False)
    _, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(pw), float(pu), len(common)


def pm(v, sd, nd=4, signed=True) -> str:
    """把「值」與「跨 seed sd」併成一格 `+0.1090 ± 0.0051`。

    n=1 的確定性 baseline（Ridge/LASSO 等）沒有 seed 變異，sd 為 NaN，
    這時只印值——注意那**不代表它沒有不確定性**，只代表它的不確定性
    來自資料而不是種子，那一側由逐日 HAC 檢定負責（見「讀表注意」）。
    """
    a = fmt(v, nd, signed)
    if sd is None or (isinstance(sd, float) and np.isnan(sd)):
        return a
    return f"{a} ± {fmt(sd, nd, False)}"


def fmt(v, nd=4, signed=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def _smkt_sa_note(base: dict) -> str | None:
    """「讀表注意」裡主結果自己的單市場消融（SMKT_SA，proposal §83.3）。缺 run 時回傳 None。

    兩折都是同種子配對：主結果 − 單市場的差、為正的種子數、配對 p、跨市場佔主結果的比例。
    第一折的逐種子 IC / RankIC 與主表同口徑（neural_stats）；第二折的評估窗截到 2024-12-25，
    逐日略過規則同 _fold2_rows（§77）。兩折的數字都與登記分析 `scripts/sa_single_market_eval.py` 相同。
    """
    def f2_seeds(arm, cut="2024-12-25"):
        out = {}
        for s, d in find_seeds(arm).items():
            f = os.path.join(d, "predictions", "test_predictions.csv")
            if not os.path.exists(f):
                continue
            ic, ric = [], []
            for t, g in pd.read_csv(f).groupby("target_date"):
                if str(t)[:10] > cut or g.y.nunique() == 1 or g.y_hat.nunique() == 1:
                    continue
                ic.append(np.corrcoef(g.y_hat, g.y)[0, 1])
                ric.append(stats.spearmanr(g.y_hat, g.y).statistic)
            out[s] = (float(np.mean(ic)), float(np.nanmean(ric)))
        return {"IC": {s: v[0] for s, v in out.items()}, "RankIC": {s: v[1] for s, v in out.items()}}

    abl1 = neural_stats(SMKT_SA[0])
    full2, abl2 = f2_seeds(BEST_F2), f2_seeds(SMKT_SA[1])
    if abl1 is None or not full2["IC"] or not abl2["IC"]:
        return None
    folds = [({"IC": base["per_ic"], "RankIC": base["per_ric"]},
              {"IC": abl1["per_ic"], "RankIC": abl1["per_ric"]}),
             (full2, abl2)]
    res = {}
    for i, (full, abl) in enumerate(folds):
        common = sorted(set(full["IC"]) & set(abl["IC"]), key=int)
        for k in ("IC", "RankIC"):
            x = np.array([full[k][s] for s in common])
            y = np.array([abl[k][s] for s in common])
            d = x - y
            res[i, k] = dict(abl=y.mean(), d=d.mean(), pos=int((d > 0).sum()), n=len(d),
                             p=float(stats.ttest_rel(x, y).pvalue), share=d.mean() / x.mean())

    def diffs(k):
        return "/ ".join(f"{res[i, k]['d']:+.4f}（{res[i, k]['pos']}/{res[i, k]['n']} 為正，"
                          f"配對 p {res[i, k]['p']:.1e}）" for i in (0, 1))
    return ("- **主結果自己的單市場消融**（2026-10-27 事前登記，proposal §83.3；依登記不進主表）："
            f"`{SMKT_SA[0]}` / `{SMKT_SA[1]}` 的 IC {res[0, 'IC']['abl']:+.4f} / {res[1, 'IC']['abl']:+.4f}、"
            f"RankIC {res[0, 'RankIC']['abl']:+.4f} / {res[1, 'RankIC']['abl']:+.4f}（第一折 / 第二折）。"
            f"主結果 − 單市場：IC {diffs('IC')}；RankIC {diffs('RankIC')}。"
            f"**跨市場佔主結果 IC 的 {res[0, 'IC']['share']:.0%} / {res[1, 'IC']['share']:.0%}、"
            f"RankIC 的 {res[0, 'RankIC']['share']:.0%} / {res[1, 'RankIC']['share']:.0%}**"
            "（上一條的 92% 是原主結果第一折的數字，第二折見 proposal §64.8）。"
            "主表「單市場消融（不帶關係的版本）」那一列是原主結果的消融：它的逐日 p 是主結果對該列，"
            "跨了兩個 arm，不是主結果自己的消融。")


def _fold2_rows():
    """第二折的對照列。找不到必要的 run 時回傳 None。"""
    import glob as _g
    CUT = "2024-12-25"

    def _load(pat):
        return _load_frames([pd.read_csv(f) for f in sorted(_g.glob(pat, recursive=True))])

    def _load_frames(frames):
        """每個 frame 是一個 run 的預測表；集成列傳進來的是已平均好的單一表。"""
        ds = []
        cals = []
        for df in frames:
            o = {}
            hh, yy = [], []          # 校準診斷用：只收進入評估窗的那些天
            for d, g in df.groupby("target_date"):
                d = str(d)[:10]
                # 略過規則同 _frame_series：50 檔完全相同才略過（§77）
                if d > CUT or g.y.nunique() == 1 or g.y_hat.nunique() == 1:
                    continue
                o[d] = (np.corrcoef(g.y_hat, g.y)[0, 1],
                        stats.spearmanr(g.y_hat, g.y).statistic,
                        topk_ret(g.y_hat.to_numpy(), g.y.to_numpy()))
                hh.append(g.y_hat.to_numpy()); yy.append(g.y.to_numpy())
            if o:
                ds.append(o)
                # 每天的檔數必須一致才疊得起來；不一致就放棄這一個 run 的
                # 校準欄（其餘欄位不受影響），不要靜默地補 NaN 算出偏差值。
                if hh and len({len(a) for a in hh}) == 1:
                    cals.append(calib_stats(np.vstack(hh), np.vstack(yy)))
        if not ds:
            return None
        k = sorted(set.intersection(*[set(x) for x in ds]))
        # P[run, 指標(0=IC, 1=RankIC, 2=Top-K 日報酬), 日]——保留逐 run 的
        # 日序列才算得出跨 run 的 sd。V 是它對 run 取平均，與加 sd 欄之前
        # 逐位元相同（第 2 個指標只給組合欄用，不進 V 的既有欄位）。
        P = np.array([[[x[d][i] for d in k] for i in (0, 1, 2)] for x in ds])
        # nan-aware（2026-10-17，§74）：Spearman 無定義的日子略過，與第一折的 colmean / nanmean 相同；
        # 沒有 NaN 的列數值不變。§74 當時是 DeltaLag 第二折種子 7 的 22 天，§77 起這些天在上面就略過，這裡只是保險。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            V = np.nanmean(P, axis=0)
        # ICIR 逐 run 算完再平均。用 V[0]（跨 run 平均後的日序列）算會得到
        # 「集成」的 ICIR，系統性偏高，與主表的慣例不一致。
        per_ir = np.array([icir(P[r, 0]) for r in range(P.shape[0])])
        per_rir = np.array([icir(P[r, 1]) for r in range(P.shape[0])])
        ir = float(np.nanmean(per_ir))
        rir = float(np.nanmean(per_rir))
        cm, cs = cal_mean_sd(cals)
        return (k, V, ir, P, per_ir, portfolio_stats(list(P[:, 2, :])),
                rir, per_rir, cm, cs)

    def _sd(a):
        """跨 run 的樣本 sd（ddof=1）。n<2 時回 NaN，`fmt` 會印成「—」。"""
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        return float(a.std(ddof=1)) if a.size >= 2 else float("nan")

    def _hac(d):
        n = len(d); L = int(4 * (n / 100) ** (2 / 9)); s2 = np.var(d, ddof=1)
        for l in range(1, L + 1):
            s2 += 2 * (1 - l / (L + 1)) * np.cov(d[l:], d[:-l], ddof=1)[0, 1]
        tt = d.mean() / (s2 / n) ** 0.5
        return 2 * (1 - stats.norm.cdf(abs(tt)))

    b = _load(str(ROOT / f"runs/**/*{BEST_F2}_s*/predictions/test_predictions.csv"))
    if b is None:
        return None
    kb, BESTV, BESTIR, BESTP, BESTIRS, BESTPF, BESTRIR, BESTRIRS, BESTCAL, BESTCALSD = b
    # 種子集成列（見 ENSEMBLE）：f2_best 的全部種子先逐 (日, 檔) 平均 y_hat，當成一條序列。
    # n 印種子數；只有一條序列，sd 為 NaN（pm 只印值）。逐日 p 是集成增益本身的檢定。
    # IC 高於逐 seed 平均，所以放在第一列，維持依 IC 排序。
    ens_files = sorted(_g.glob(str(ROOT / f"runs/**/*{BEST_F2}_s*/predictions/test_predictions.csv"),
                              recursive=True))
    ens = None
    if ens_files:
        edf = (pd.concat([pd.read_csv(f) for f in ens_files]).groupby(["target_date", "ticker"])
               .agg(y_hat=("y_hat", "mean"), y=("y", "first")).reset_index())
        ens = _load_frames([edf])
    # 列序依第二折 IC 由高到低自動排（2026-10-17 起；原本手動插入，[24] 換成 8 個模型的集成後
    # 手排的順序已經不對）。主結果 arm 依自己的 IC 落位，集成列 IC 最高、仍在第一列。
    # [24] 列出全部 10 個模型（BIP_MODELS），一律取最新一批。
    rows = [(ENSEMBLE[0][0], ens, len(ens_files)),
            (f"**{MAIN_LABEL}**", None, None)]
    pats = [("關係型線性（z 目標）", "runs_f2/*_lin_relational/predictions/test_predictions.csv"),
            ("MAGNET + 關係型台股層（投影後版本）", "runs/**/*f2_relTW_s*/predictions/test_predictions.csv"),
            (PRIOR_LABEL, f"runs/**/*{PRIOR_F2}_s*/predictions/test_predictions.csv"),
            ("兩區塊 ridge（z 目標）", "runs_f2/*_lin_twoblock/predictions/test_predictions.csv"),
            ("KTW+（rank 目標）", "runs_f2/*fvg_KTWp/predictions/*.csv"),
            ("R2 per-target ridge", "runs_f2/*ridge_R2/predictions/*.csv"),
            ("TW+（美股+台股 80 維）", "runs_f2/*fvg_TWp/predictions/*.csv"),
            ("MAGNET 基準（自家前版）", "runs/**/*f2_base_s*/predictions/test_predictions.csv"),
            ("RC 常數對照", "runs_f2/*ridge_RC/predictions/*.csv")]
    pats += [(bip_label(m), (_latest(f"runs_f2/{BIP.format(m)}") or "runs_f2/__missing__")
              + "/predictions/test_predictions.csv") for m, _ in BIP_MODELS]
    # 6 個文獻神經模型的第二折（2026-10-17 補跑，§74）：各 3 顆種子，設定與第一折相同（未調參），
    # tag 由第一折的 tw50_bl_<name> 換成 f2_bl_<name>；顯示名稱兩折相同。找不到 run 時該列略過。
    pats += [(lab, f"runs/**/*f2_bl_{arm[len('tw50_bl_'):]}_s*/predictions/test_predictions.csv")
             for lab, arm, cat, _ in NEURAL if cat == "文獻"]
    for lab, pat in pats:
        rows.append((lab, _load(str(ROOT / pat)), None))

    def _ic_key(row):
        lab, r, _ = row
        if r is None:
            return BESTV[0].mean() if lab.startswith("**") else -np.inf
        kk = r[0]
        return r[1][0][[kk.index(d) for d in kk if d in kb]].mean()
    rows = sorted(rows, key=lambda row: -_ic_key(row))
    # 主表（IC、RankIC 與兩者的逐日 p）/ 附錄 A 的其他指標（ICIR、Rank ICIR）/ 組合讀法 / 校準診斷；
    # labs 與四者逐列對齊，回傳時配成 (名稱, 列) 供 main() 分主表與附錄 B
    sig, oth, pf_rows, cal_rows, labs = [], [], [], [], []
    for lab, r, n_seed in rows:
        if r is None and lab.startswith("**"):
            labs.append(lab)
            sig.append(
                f"| {lab} | {BESTP.shape[0]} "
                f"| {pm(BESTV[0].mean(), _sd(BESTP[:, 0].mean(axis=1)))} "
                f"| {pm(BESTV[1].mean(), _sd(BESTP[:, 1].mean(axis=1)))} | — | — |")
            oth.append(
                f"| {lab} | {BESTP.shape[0]} "
                f"| {pm(BESTIR, _sd(BESTIRS), signed=False)} "
                f"| {pm(BESTRIR, _sd(BESTRIRS), signed=False)} |")
            pf_rows.append(f"| {lab} | {BESTP.shape[0]} " + pf_cells(BESTPF) + "|")
            cal_rows.append(f"| {lab} | {BESTP.shape[0]} "
                            + cal_cells(BESTCAL, BESTCALSD) + "|")
            continue
        if r is None:
            continue
        labs.append(lab)
        kk, V, IR, P, IRS, PF, RIR, RIRS, CM, CS = r
        n_col = n_seed if n_seed else P.shape[0]
        ii = [kk.index(d) for d in kk if d in kb]
        jj = [kb.index(d) for d in kk if d in kb]
        cells = []
        for i in (0, 1):
            dd = BESTV[i][jj] - V[i][ii]
            dd = dd[np.isfinite(dd)]          # 同上：略過 RankIC 無定義的日子
            cells += [f"{dd.mean():+.4f}", f"{_hac(dd):.4f}"]
        # IC / RankIC 的 sd 與其點估計取同一組日子（ii）。ICIR 的點估計 IR 是在
        # 完整 kk 上算的，其 sd 因此也用完整 kk，兩者口徑一致。
        sig.append(
            f"| {lab} | {n_col} "
            f"| {pm(np.nanmean(V[0][ii]), _sd(np.nanmean(P[:, 0][:, ii], axis=1)))} "
            f"| {pm(np.nanmean(V[1][ii]), _sd(np.nanmean(P[:, 1][:, ii], axis=1)))} "
            f"| {cells[1]} | {cells[3]} |")
        oth.append(
            f"| {lab} | {n_col} "
            f"| {pm(IR, _sd(IRS), signed=False)} "
            f"| {pm(RIR, _sd(RIRS), signed=False)} |")
        pf_rows.append(f"| {lab} | {n_col} " + pf_cells(PF) + "|")
        cal_rows.append(f"| {lab} | {n_col} " + cal_cells(CM, CS) + "|")
    return tuple(list(zip(labs, x)) for x in (sig, oth, pf_rows, cal_rows))


def _ens_block():
    """§43 的種子集成與等權混合。缺 run 時回傳 None。"""
    import glob as _g

    def _pred(d, cut=None):
        f = os.path.join(d, "predictions", "test_predictions_reeval.csv")
        if PREDICTIONS != "reeval" or not os.path.exists(f):
            f = os.path.join(d, "predictions", "test_predictions.csv")
        if not os.path.exists(f):
            return None
        df = pd.read_csv(f)
        df["target_date"] = df.target_date.astype(str).str[:10]
        return df[df.target_date <= cut] if cut else df

    def _metrics(df):
        """回傳 {date: (IC, RankIC, 離散比, Top-K 日報酬)}。"""
        o = {}
        for d, g in df.groupby("target_date"):
            if g.y.nunique() == 1 or g.y_hat.nunique() == 1:     # 同 _frame_series（§77）
                continue
            o[d] = (np.corrcoef(g.y_hat, g.y)[0, 1],
                    stats.spearmanr(g.y_hat, g.y).statistic,
                    g.y_hat.std() / g.y.std(),
                    topk_ret(g.y_hat.to_numpy(), g.y.to_numpy()))
        return o

    def _per_seed(pat, cut=None):
        """(1) 先算每顆種子的逐日指標，再跨種子平均。"""
        ds = [x for x in (_pred(d, cut)
                          for d in sorted(_g.glob(str(ROOT / pat), recursive=True)))
              if x is not None]
        if not ds:
            return None
        ms = [_metrics(x) for x in ds]
        k = sorted(set.intersection(*[set(m) for m in ms]))
        # Q[seed, 指標(0=IC 1=RankIC 2=離散比 3=Top-K 日報酬), 日]。
        # ICIR / Rank ICIR / 組合指標都要**逐 seed 算完再平均**，
        # 不能拿跨 seed 平均後的序列去算——那是「集成」的值，系統性偏高，
        # 而這正是本區塊 (A) 在量的東西。
        Q = np.array([[[m[d][i] for d in k] for i in (0, 1, 2, 3)] for m in ms])
        return k, Q.mean(axis=0), len(ds), Q

    def _ens(pat, cut=None):
        """(2) 先平均全部種子的預測，再算逐日指標。"""
        ds = [x for x in (_pred(d, cut)
                          for d in sorted(_g.glob(str(ROOT / pat), recursive=True)))
              if x is not None]
        if not ds:
            return None
        df = (pd.concat(ds).groupby(["target_date", "ticker"])
              .agg(y_hat=("y_hat", "mean"), y=("y", "first")).reset_index())
        m = _metrics(df)
        k = sorted(m)
        return k, np.array([[m[d][i] for d in k] for i in (0, 1, 2, 3)]), df

    def _hac(d):
        n = len(d)
        lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
        x = d - d.mean()
        var = float((x @ x) / n)
        for j in range(1, lag + 1):
            var += 2.0 * (1.0 - j / (lag + 1.0)) * float((x[j:] @ x[:-j]) / n)
        return float(2 * stats.t.sf(abs(d.mean() / np.sqrt(max(var, 1e-24) / n)), n - 1))

    def _cmp(mine, other):
        km, M = mine[0], mine[1]
        ko, O = other[0], other[1]
        k = sorted(set(km) & set(ko))
        ii = [km.index(d) for d in k]
        jj = [ko.index(d) for d in k]
        return [(M[i][ii] - O[i][jj]).mean() for i in (0, 1)], \
               [_hac(M[i][ii] - O[i][jj]) for i in (0, 1)]

    FOLDS = (("第一折", f"runs/**/*{BEST}_s*", None,
              "runs/linear/*_fvg_KTWp/predictions/*.csv"),
             ("第二折", f"runs/**/*{BEST_F2}_s*", "2024-12-25",
              "runs_f2/*_fvg_KTWp/predictions/*.csv"))

    out, ok = [], False
    out.append("### (A) 種子集成：先平均預測，再算 IC")
    out.append("")
    out.append("| 折 | n 種子 | 聚合 | IC ↑ | RankIC ↑ | **ICIR ↑** | **Rank ICIR ↑** "
               "| 離散比 std(y_hat)/std(y) " + PF_HEAD + "|")
    out.append("|---|---:|---|---:|---:|---:|---:|---:" + PF_SEP + "|")
    ens_cache = {}
    for lab, pat, cut, _ in FOLDS:
        ps = _per_seed(pat, cut)
        en = _ens(pat, cut)
        if ps is None or en is None:
            continue
        ok = True
        ens_cache[lab] = (en, cut)
        Q = ps[3]
        s_ir = float(np.nanmean([icir(Q[j, 0]) for j in range(Q.shape[0])]))
        s_rir = float(np.nanmean([icir(Q[j, 1]) for j in range(Q.shape[0])]))
        e_ir, e_rir = icir(en[1][0]), icir(en[1][1])
        out.append(f"| {lab} | {ps[2]} | 先算 IC 再平均 | {ps[1][0].mean():+.4f} "
                   f"| {ps[1][1].mean():+.4f} | {s_ir:.4f} | {s_rir:.4f} "
                   f"| {ps[1][2].mean():.2f} "
                   + pf_cells(portfolio_stats(list(Q[:, 3, :]))) + "|")
        out.append(f"| {lab} | {ps[2]} | **先平均預測再算 IC** "
                   f"| **{en[1][0].mean():+.4f}** | **{en[1][1].mean():+.4f}** "
                   f"| **{e_ir:.4f}** | **{e_rir:.4f}** "
                   f"| **{en[1][2].mean():.2f}** "
                   + pf_cells(portfolio_stats([en[1][3]])) + "|")
        # 「增益」是兩列相減，沒有對應的日報酬序列，組合欄留空
        out.append(f"| {lab} | | 增益 | {en[1][0].mean() - ps[1][0].mean():+.4f} "
                   f"| {en[1][1].mean() - ps[1][1].mean():+.4f} "
                   f"| **{e_ir - s_ir:+.4f}** | **{e_rir - s_rir:+.4f}** "
                   f"| {en[1][2].mean() - ps[1][2].mean():+.2f} "
                   + PF_BLANK + "|")
    if not ok:
        return None

    NEW_BLS = (("關係型線性（z 目標）", "*_lin_relational"), ("兩區塊 ridge（z 目標）", "*_lin_twoblock"))
    BLS = (("KTW+（rank 目標）", "*_fvg_KTWp"), *NEW_BLS, ("[24] 二部圖 LASSO", BIP.format("LASSO")),
           ("R2 per-target ridge", "*_ridge_R2"), ("[24] 二部圖 ens-avg", BIP.format("ens-avg")),
           ("RC 常數對照", "*_ridge_RC"))
    out += ["", "### (B) 集成後對線性 baseline 的逐日檢定", "",
            "| 折 | 對照 | 其 IC ↑ | 其 RankIC ↑ | 其 **ICIR ↑** | 其 **Rank ICIR ↑** "
            "| 逐日 p (IC) | 逐日 p (RankIC) " + PF_HEAD + "|",
            "|---|---|---:|---:|---:|---:|---:|---:" + PF_SEP + "|"]
    for lab, _, cut, _ in FOLDS:
        if lab not in ens_cache:
            continue
        en, cut = ens_cache[lab]
        root = "runs/linear" if lab == "第一折" else "runs_f2"
        for bl, bp in BLS:
            # [24] 只取最新一批（見 BIP 的說明）；其餘照舊。R2 的三批 ridge_R2 逐位元相同，平均不改數字。
            pat = _latest(f"{root}/{bp}") if bl.startswith("[24]") else f"{root}/{bp}"
            b = _per_seed(pat, cut) if pat else None
            if b is None:
                continue
            d, pv = _cmp(en, b)
            Qb = b[3]
            b_ir = float(np.nanmean([icir(Qb[j, 0]) for j in range(Qb.shape[0])]))
            b_rir = float(np.nanmean([icir(Qb[j, 1]) for j in range(Qb.shape[0])]))
            out.append(f"| {lab} | {bl} | {b[1][0].mean():+.4f} | {b[1][1].mean():+.4f} "
                       f"| {b_ir:.4f} | {b_rir:.4f} "
                       f"| {pv[0]:.4f} | {pv[1]:.4f} "
                       + pf_cells(portfolio_stats(list(Qb[:, 3, :]))) + "|")

    mix = {}
    out += ["", "### (C) 與 KTW+ 等權混合（w=0.5，逐日橫截面 z 分數，未調參）", "",
            "| 折 | 方法 | IC ↑ | RankIC ↑ | **ICIR ↑** | **Rank ICIR ↑** "
            "| 逐日 p (IC) | 逐日 p (RankIC) " + PF_HEAD + "|",
            "|---|---|---:|---:|---:|---:|---:|---:" + PF_SEP + "|"]
    for lab, _, cut, kp in FOLDS:
        if lab not in ens_cache:
            continue
        en, cut = ens_cache[lab]
        kf = sorted(_g.glob(str(ROOT / kp)))
        if not kf:
            continue
        K = pd.read_csv(kf[-1])
        K["target_date"] = K.target_date.astype(str).str[:10]
        if cut:
            K = K[K.target_date <= cut]
        j = en[2].merge(K[["target_date", "ticker", "y_hat"]],
                        on=["target_date", "ticker"], suffixes=("_m", "_b"))

        def _z(g, c):
            v = g[c].to_numpy()
            sd = v.std()
            return (v - v.mean()) / sd if sd > 1e-12 else v * 0.0

        for c, src in (("zm", "y_hat_m"), ("zb", "y_hat_b")):
            j[c] = j.groupby("target_date", group_keys=False).apply(
                lambda g: pd.Series(_z(g, src), index=g.index), include_groups=False)
        rows = [("純 KTW+", 0.0), ("**等權混合 w=0.5**", 0.5), ("純 MAGNET（集成）", 1.0)]
        base = None
        for nm, w in rows:
            jj = j.assign(y_hat=w * j.zm + (1 - w) * j.zb)
            m = _metrics(jj)
            k = sorted(m)
            V = np.array([[m[d][i] for d in k] for i in (0, 1, 3)])
            pf = pf_cells(portfolio_stats([V[2]]))
            ir4 = f"| {icir(V[0]):.4f} | {icir(V[1]):.4f} "
            if w == 0.5:
                mix[lab] = (k, V, cut)
            if w == 0.0:
                base = V
                out.append(f"| {lab} | {nm} | {V[0].mean():+.4f} | {V[1].mean():+.4f} "
                           + ir4 + "| — | — " + pf + "|")
                continue
            d0, d1 = V[0] - base[0], V[1] - base[1]
            out.append(f"| {lab} | {nm} | {V[0].mean():+.4f} | {V[1].mean():+.4f} "
                       + ir4 + f"| {_hac(d0):.4f} | {_hac(d1):.4f} " + pf + "|")

    # (C2)（2026-10-13 加）：§67 之後最高標不只 KTW+，混合模型對另外兩個線性 baseline 也要報。
    # 混合仍只混 KTW+、w 仍是 0.5，沒有因為新的 baseline 改動。差 = 混合 − 對照。
    out += ["", "### (C2) 等權混合對 §67 兩個線性 baseline 的逐日檢定", "",
            "§67 之後最高標不只 KTW+。混合仍只混 KTW+、w 仍是 0.5，沒有因為新的 baseline 改動；"
            "差 = 混合 − 對照。", "",
            "| 折 | 對照 | 其 IC ↑ | 其 RankIC ↑ | 混合 − 對照 IC | 逐日 p (IC) "
            "| 混合 − 對照 RankIC | 逐日 p (RankIC) |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for lab, _, _, _ in FOLDS:
        if lab not in mix:
            continue
        k, V, cut = mix[lab]
        root = "runs/linear" if lab == "第一折" else "runs_f2"
        for bl, bp in NEW_BLS:
            b = _per_seed(f"{root}/{bp}", cut)
            if b is None:
                continue
            d, pv = _cmp((k, V), b)
            out.append(f"| {lab} | {bl} | {b[1][0].mean():+.4f} | {b[1][1].mean():+.4f} "
                       f"| {d[0]:+.4f} | {pv[0]:.4f} | {d[1]:+.4f} | {pv[1]:.4f} |")
    return out


# 準度小表（2026-10-13 加，描述性，未登記）。y_hat 是「基數分數、尺度未校正」（§60.1），
# 校準診斷的 MSE 量的是尺度，不能拿來比準度。這裡讓每個模型先在 val 上定一個尺度：
# 逐日扣掉橫截面平均後，以最小平方擬合 y ≈ b·y_hat，b 固定後套到 test，再算 RMSE、R²_oos 與命中率。
# 乘一個倍數不改相關係數，所以 IC / RankIC 不受影響。
# 線性 baseline 的 runs/ 只存 test 預測，val 預測由 tw_lag_channel / relational_linear 重算；
# 重算的 test 預測與存檔的最大差 ≤ ACC_TOL x 存檔的最大絕對值才採用，否則該列跳過。
# 存檔只有約 8 位有效數字：KTW+、R2、RC 實測相對差最大 1.2e-6，關係型與兩區塊約 1e-16。
# ACC_TOL 是重現閘門的數值容忍度，不是模型超參，故不放 config。
ACC_TOL = 1e-5
# (顯示名稱, val 預測的重算方式, 第一折存檔, 第二折存檔)。存檔與主表兩折各自讀的是同一份；
# R2 兩折的檔名不同（fvg / ridge），但都等於 tw_lag_channel.ridge(D, "US30", "raw")。
ACC_LINEAR = (
    ("KTW+（rank 目標）", "ktw", "runs/linear/*_fvg_KTWp/predictions/test_predictions.csv",
     "runs_f2/*_fvg_KTWp/predictions/test_predictions.csv"),
    ("關係型線性（z 目標）", "relational", "runs/linear/*_lin_relational/predictions/test_predictions.csv",
     "runs_f2/*_lin_relational/predictions/test_predictions.csv"),
    ("兩區塊 ridge（z 目標）", "twoblock", "runs/linear/*_lin_twoblock/predictions/test_predictions.csv",
     "runs_f2/*_lin_twoblock/predictions/test_predictions.csv"),
    ("R2（30 檔美股報酬）", "r2", "runs/linear/*_fvg_R2/predictions/test_predictions.csv",
     "runs_f2/*_ridge_R2/predictions/test_predictions.csv"),
    ("RC 常數對照", "rc", "runs/linear/*_ridge_RC/predictions/test_predictions.csv",
     "runs_f2/*_ridge_RC/predictions/test_predictions.csv"),
)
ANCHOR = "不選股（每檔猜當天平均）"


def _holdout_block():
    """2026 保留期（§75）：凍結的第一折模型在 2026-01-02 ~ 2026-10-16 的 IC 與 RankIC。

    只讀 scripts/holdout_2026_eval.py --run 寫下的 predictions/holdout2026_predictions.csv，
    不重新評估。缺檔時回傳 None。統計與判定的程式在 holdout_2026_eval.py（登記版），這裡只重算同一組數字。
    """
    try:
        import holdout_2026_eval as H
    except Exception as e:                      # noqa: BLE001
        print(f"[skip] 2026 保留期（{e}）")
        return None
    P = H.load_saved()
    seeds = sorted(set(P.get("主 arm", {})) & set(P.get("關係型", {})))
    if len(seeds) < 2:
        return None
    D = {a: {s: H.daily(P[a][s]) for s in seeds} for a in P}
    days = sorted(set.intersection(*[set(D[a][s].index) for a in D for s in seeds]))
    per = {a: {k: np.array([np.nanmean(D[a][s].loc[days, k]) for s in seeds]) for k in ("ic", "ric")} for a in D}
    ens = {}
    for a in P:
        e = (pd.concat([P[a][s] for s in seeds]).groupby(["target_date", "ticker"])
             .agg(y_hat=("y_hat", "mean"), y=("y", "first")).reset_index())
        ens[a] = H.daily(e).loc[days]
    out = ["| 方法 | n | IC ↑ | RankIC ↑ | 配對 p (IC) | 配對 p (RankIC) | 逐日 p (IC) | 逐日 p (RankIC) |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    lab = {"主 arm": PRIOR_LABEL, "關係型": "MAGNET + 關係型台股層（投影後版本）"}
    for a in ("關係型", "主 arm"):
        out.append(f"| {lab[a]}（10 顆集成） | {len(seeds)} | {np.nanmean(ens[a].ic):+.4f} | {np.nanmean(ens[a].ric):+.4f} "
                   "| — | — | — | — |")
    for a in ("關係型", "主 arm"):
        cells = []
        if a == "關係型":
            for k in ("ic", "ric"):
                cells.append(f"{stats.ttest_rel(per['關係型'][k], per['主 arm'][k]).pvalue:.4f}")
            for k in ("ic", "ric"):
                ma = np.nanmean(np.vstack([D["主 arm"][s].loc[days, k] for s in seeds]), 0)
                mb = np.nanmean(np.vstack([D["關係型"][s].loc[days, k] for s in seeds]), 0)
                cells.append(f"{H.hac_p(mb - ma):.4f}")
        else:
            cells = ["—"] * 4
        out.append(f"| {lab[a]} | {len(seeds)} | {pm(per[a]['ic'].mean(), per[a]['ic'].std(ddof=1))} "
                   f"| {pm(per[a]['ric'].mean(), per[a]['ric'].std(ddof=1))} | " + " | ".join(cells) + " |")
    # §81（2026-10-26）：凍結的 SA（自身 + 同產業，§80）的探索性第二次評估，只讀 holdout_2026_sa_eval.py --run
    # 存下的預測。日子沿用上面兩個 arm 的同一組；配對 p 括號內是 Bonferroni x2（同一假設的第二次檢驗）。
    sa = {}
    for s, d in H.runs("tw50_rgcnSA").items():
        f = d / "predictions" / H.OUT_NAME
        if f.exists():
            df = pd.read_csv(f, dtype={"ticker": str})
            df["target_date"] = df["target_date"].astype(str).str[:10]
            sa[s] = df
    has_sa = set(seeds) <= set(sa)
    if has_sa:
        Ds = {s: H.daily(sa[s]) for s in seeds}
        has_sa = all(set(days) <= set(Ds[s].index) for s in seeds)
    if has_sa:
        ps = {k: np.array([np.nanmean(Ds[s].loc[days, k]) for s in seeds]) for k in ("ic", "ric")}
        e = (pd.concat([sa[s] for s in seeds]).groupby(["target_date", "ticker"])
             .agg(y_hat=("y_hat", "mean"), y=("y", "first")).reset_index())
        e = H.daily(e).loc[days]
        sl = f"**{MAIN_LABEL}**（探索性的第二次檢驗，§81）"
        cells = []
        for k in ("ic", "ric"):
            p = stats.ttest_rel(ps[k], per["主 arm"][k]).pvalue
            cells.append(f"{p:.4f}（x2 {min(1.0, 2 * p):.4f}）")
        for k in ("ic", "ric"):
            ma = np.nanmean(np.vstack([D["主 arm"][s].loc[days, k] for s in seeds]), 0)
            mb = np.nanmean(np.vstack([Ds[s].loc[days, k] for s in seeds]), 0)
            cells.append(f"{H.hac_p(mb - ma):.4f}")
        out.append(f"| {sl}（10 顆集成） | {len(seeds)} | {np.nanmean(e.ic):+.4f} | {np.nanmean(e.ric):+.4f} | — | — | — | — |")
        out.append(f"| {sl} | {len(seeds)} | {pm(ps['ic'].mean(), ps['ic'].std(ddof=1))} "
                   f"| {pm(ps['ric'].mean(), ps['ric'].std(ddof=1))} | " + " | ".join(cells) + " |")
    return days, out, has_sa


def _acc_block():
    """準度小表（說明見 ACC_TOL 上方）。缺資料時回傳 None。

    主 arm 讀 run 當下寫的 val / test 預測檔。它沒有 CPU 重評檔（2026-10-13 檢查），
    所以與訊號層讀的是同一份。
    """
    try:
        import relational_linear as RL
        import tw_lag_channel as T
    except Exception as e:                      # 例如沒有資料快照的環境
        print(f"[skip] 準度小表（{e}）")
        return None

    def dm(P):
        return P - np.nanmean(P, 1, keepdims=True)

    def evaluate(Hva, Yva, Hte, Yte):
        """val 定尺度 -> test 的逐日橫截面 MSE、不選股的 MSE、命中率。50 檔 y 完全相同的日子不用（同訊號層，§77）。"""
        okv, okt = np.nanmax(Yva, 1) > np.nanmin(Yva, 1), np.nanmax(Yte, 1) > np.nanmin(Yte, 1)
        hv, yv = dm(Hva[okv]), dm(Yva[okv])
        den = np.nansum(hv * hv)
        b = float(np.nansum(hv * yv) / den) if den > 1e-24 else 0.0
        ht, yt = dm(Hte[okt]), dm(Yte[okt])
        sh = np.sign(Hte[okt] - np.nanmedian(Hte[okt], 1, keepdims=True))
        sy = np.sign(Yte[okt] - np.nanmedian(Yte[okt], 1, keepdims=True))
        m = (sh != 0) & (sy != 0)               # 剛好等於中位數的不計
        hit = np.array([np.mean(a[k] == c[k]) if k.any() else np.nan for a, c, k in zip(sh, sy, m)])
        return dict(loss=np.nanmean((yt - b * ht) ** 2, 1), base=np.nanmean(yt ** 2, 1), hit=hit)

    def summ(ev):
        return (1 - np.nansum(ev["loss"]) / np.nansum(ev["base"]),
                float(np.sqrt(np.nanmean(ev["loss"])) * 100), float(np.nanmean(ev["hit"]) * 100))

    folds = {}
    for fi, (f, c) in enumerate(T.FOLDS.items()):
        D = T.load_fold(c["cfg"], c["f2"])
        Yv, Yt = D["y_val"], D["y_test"]
        # 主結果（2026-10-27 起為 SA）；T.FOLDS 的 main 仍是原主結果，另列一列
        mpat = f"runs/**/*{BEST if fi == 0 else BEST_F2}_s*/predictions/{{}}_predictions.csv"
        Mv, sv = T.preds(mpat, "val", D["d_val"], D["codes"])
        Mt, st = T.preds(mpat, "test", D["d_test"], D["codes"])
        if not len(sv) or sv != st:
            print(f"[skip] 準度小表 {f}（主結果的 val / test 預測檔對不上）")
            continue
        rows = [(f"**{MAIN_LABEL}**", len(sv),
                 [evaluate(Mv[i], Yv, Mt[i], Yt) for i in range(len(sv))]),
                (ENSEMBLE[0][0], len(sv), [evaluate(Mv.mean(0), Yv, Mt.mean(0), Yt)])]
        Pv, spv = T.preds(c["main"], "val", D["d_val"], D["codes"])
        Pt, spt = T.preds(c["main"], "test", D["d_test"], D["codes"])
        if len(spv) and spv == spt:
            rows.append((PRIOR_LABEL, len(spv), [evaluate(Pv[i], Yv, Pt[i], Yt) for i in range(len(spv))]))
        g = RL.grids(c["cfg"])
        peers, others = RL.relations(D["codes"])
        nu = D["us_train"].shape[1]
        X = {sp: np.hstack([D[f"us_{sp}"], D[f"tw_{sp}"]]) for sp in ("train", "val", "test")}
        Z = dict(zip(("train", "val", "test"), RL.standardize(X["train"], X["val"], X["test"])))
        Tz = T.zrow(D["y_train"])
        k, r2 = T.ridge(D, "US30+TW50", "rank"), T.ridge(D, "US30", "raw")
        rel = RL.relational(Z, Tz, Yv, nu, g["relational_lambda"], peers, others)[2]
        tb = RL.twoblock(Z, Tz, Yv, nu, g["twoblock_alpha"])[2]
        mu = D["y_train"].mean(0)
        fit = {"ktw": (k["val"], k["test"]), "r2": (r2["val"], r2["test"]),
               "relational": (rel["val"], rel["test"]), "twoblock": (tb["val"], tb["test"]),
               "rc": (np.tile(mu, (len(Yv), 1)), np.tile(mu, (len(Yt), 1)))}
        for lab, key, *pats in ACC_LINEAR:
            S, _ = T.preds(pats[fi], "test", D["d_test"], D["codes"])
            if not len(S):
                print(f"[skip] 準度小表 {f} {lab}（找不到預測檔）")
                continue
            S, (hv, ht) = S[-1], fit[key]
            ok = np.isfinite(S) & np.isfinite(ht)
            err = float(np.abs(S[ok] - ht[ok]).max() / max(np.nanmax(np.abs(S)), 1e-12))
            print(f"[準度] {f} {lab}：重算與存檔 test 預測的相對差 {err:.1e}")
            if err > ACC_TOL:
                print(f"[skip] 準度小表 {f} {lab}（相對差超過 {ACC_TOL:g}，不是同一個模型）")
                continue
            rows.append((lab, 1, [evaluate(hv, Yv, S, Yt)]))
        evs = rows[0][2]
        folds[f] = dict(rows=rows, base=evs[0]["base"],
                        loss=np.mean([e["loss"] for e in evs], 0), hit=np.mean([e["hit"] for e in evs], 0))
    if not folds:
        return None

    def cell(vals, fmt_v, fmt_sd):
        v = np.asarray(vals, float)
        return fmt_v.format(v.mean()) + (f" ± {fmt_sd.format(v.std(ddof=1))}" if v.size > 1 else "")

    out = ["| 折 | 方法 | n | RMSE (%) ↓ | R²_oos ↑ | 命中率 ↑ | 逐日 p（R²_oos） | 逐日 p（命中率） |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for f, F in folds.items():
        out.append(f"| {f} | {ANCHOR} | — | {np.sqrt(np.nanmean(F['base'])) * 100:.4f} | 0（依定義） "
                   f"| 50%（依定義） | {T.hac_p(F['loss'] - F['base']):.4f} | {T.hac_p(F['hit'] - 0.5):.4f} |")
        for i, (lab, n, evs) in enumerate(F["rows"]):
            S = np.array([summ(e) for e in evs])
            if i == 0:
                p1 = p2 = "—"
            else:
                p1 = f"{T.hac_p(np.mean([e['loss'] for e in evs], 0) - F['loss']):.4f}"
                p2 = f"{T.hac_p(np.mean([e['hit'] for e in evs], 0) - F['hit']):.4f}"
            out.append(f"| {f} | {lab} | {n} | {cell(S[:, 1], '{:.4f}', '{:.4f}')} "
                       f"| {cell(S[:, 0], '{:+.4f}', '{:.4f}')} | {cell(S[:, 2], '{:.2f}%', '{:.2f}')} "
                       f"| {p1} | {p2} |")

    base = np.concatenate([F["base"] for F in folds.values()])
    mloss = np.concatenate([F["loss"] for F in folds.values()])
    mhit = np.concatenate([F["hit"] for F in folds.values()])
    out += ["", f"### 兩折合併（{len(base)} 天，對主結果的種子平均）", "",
            f"主結果兩折合併：`R²_oos` {1 - mloss.sum() / base.sum():+.4f}、"
            f"命中率 {np.nanmean(mhit) * 100:.2f}%。差 = 該列 − 主結果，**正值代表該列較好**。", "",
            "| 方法 | R²_oos 差 ↑ | 逐日 p | 命中率差（百分點）↑ | 逐日 p |",
            "|---|---:|---:|---:|---:|"]
    labs = [lab for lab, _, _ in next(iter(folds.values()))["rows"][1:]]
    pooled = [(ANCHOR, base, np.full(len(base), 0.5))]
    for lab in labs:
        got = [next((evs for l2, _, evs in F["rows"] if l2 == lab), None) for F in folds.values()]
        if any(x is None for x in got):
            continue
        pooled.append((lab, np.concatenate([np.mean([e["loss"] for e in evs], 0) for evs in got]),
                       np.concatenate([np.mean([e["hit"] for e in evs], 0) for evs in got])))
    for lab, loss, hit in pooled:
        dl, dh = loss - mloss, hit - mhit
        out.append(f"| {lab} | {-dl.sum() / base.sum():+.4f} | {T.hac_p(dl):.4f} "
                   f"| {np.nanmean(dh) * 100:+.2f} | {T.hac_p(dh):.4f} |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="產出跨方法總結果表")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--predictions", choices=["reeval", "recorded"],
                    default="reeval",
                    help="reeval=CPU 重評（預設，可重現）；"
                         "recorded=run 當下寫的 CSV（含 GAT 的 arm 走 MPS，不可重現）")
    args = ap.parse_args()
    global PREDICTIONS
    PREDICTIONS = args.predictions

    rows = []
    for label, arm, cat, note in NEURAL:
        st = neural_stats(arm)
        if st is None:
            print(f"[skip] {label}（找不到 {arm}）"); continue
        rows.append((label, cat, note, st, arm))
    for label, pat, cat, note in NONNEURAL:
        st = nonneural_stats(pat)
        if st is None:
            print(f"[skip] {label}（找不到 {pat}）"); continue
        rows.append((label, cat, note, st, pat))
    for label, arm, cat, note in ENSEMBLE:
        st = ensemble_stats(arm)
        if st is None:
            print(f"[skip] {label}（找不到 {arm}）"); continue
        rows.append((label, cat, note, st, "ens:" + arm))   # 鍵與 BEST 不同，不會被當成基準

    base = next(st for _, _, _, st, arm in rows if arm == BEST)
    out = []
    out.append("# 跨方法結果表")
    out.append("")
    out.append("由 `scripts/results_table.py` 產生。universe = tw50（US 30 / TW 50，"
               "配對 7 檔），**第一折** walk-forward test 246 天（2024-12-26 ~ 2025-12-30）。第二折的獨立驗證見下方專節——**單折排名會翻轉，兩節必須一起看**。")
    out.append("")
    out.append(f"統計基準 = **{BEST}**，即主結果「MAGNET（+ 同產業關係）」（2026-10-27 起；台股層內圖改為同產業關係卷積，看過 §79 的消融之後選的，數字偏高，proposal §80、§83）。原主結果（不帶關係，`{PRIOR}` / `{PRIOR_F2}`）保留為""「MAGNET 不帶關係（原主結果）」一列。"
               "**主指標是 IC，第二指標是 RankIC**（2026-10-17，依教授建議只留一到兩個指標）："
               "主表只列這兩個指標與它們的逐日檢定（`逐日 p (IC)`、`逐日 p (RankIC)`，"
               "都是主結果對該列）。ICIR、Rank ICIR、跨種子檢定、組合讀法、校準診斷、"
               "§43 的集成分析與準度小表都移到附錄，數字與檢定都沒有改。差值欄不列，自己減得出來。")
    out.append("")
    out.append("**主表與附錄 B（2026-10-16，依教授建議）**：兩折的主表只放 MAGNET 自身的各版本與消融，"
               "以及文獻方法（`[n]` 與 [24]）。我們另外建構、又不是出自文獻的對照"
               "（同輸入的線性模型 KTW+、R2、TW+、關係型、兩區塊，以及空模型 RC）放在附錄 B。")
    out.append("")
    out.append("**[24] 列出全部 10 個模型（2026-10-17）**：8 個單一模型（OLS、LASSO、RIDGE、RF、SVM、"
               "AdaBoost、XGBoost、LightGBM）與兩種集成（平均、中位數）。兩折都是滾動版"
               "（250 天視窗、每 10 天重訓、t 檢定篩邊 τ = 2，原文做法），只取最新一批（proposal §70）。"
               "2026-10-21 起改用原始報酬評估；之前誤用截尾後的報酬，IC 略為偏高（§76）。")
    out.append("")
    ordered = sorted(rows, key=lambda r: -r[3]["ic"])
    main_rows = [r for r in ordered if r[0] not in SUPPLEMENT]
    # 附表：移出主表的列，加上 MAGNET 本版與集成兩列當參照（數字與主表相同）
    app_rows = [r for r in ordered if r[0] in SUPPLEMENT or r[4] in (BEST, "ens:" + BEST)]

    def main_table(rs):
        """主表：IC（主指標）與 RankIC（第二指標），以及兩者對統計基準的逐日 HAC p。"""
        o = ["| 方法 | 設定 | n | IC ↑ | RankIC ↑ | 逐日 p (IC) | 逐日 p (RankIC) |",
             "|---|---|---:|---:|---:|---:|---:|"]
        for label, cat, note, st, arm in rs:
            _, p_ic = paired_daily(base, st, "ic")
            _, p_ric = paired_daily(base, st, "ric")
            is_base = (arm == BEST)
            star = " **" if is_base else " "
            o.append(
                f"|{star}{label}{star.strip()} | {note} | {st['n']} | "
                f"{pm(st['ic'], st['ic_sd'])} | "
                f"{pm(st['ric'], st['ric_sd'])} | "
                f"{'—' if is_base else fmt(p_ic, signed=False)} | "
                f"{'—' if is_base else fmt(p_ric, signed=False)} |")
        return o

    def other_table(rs):
        """附錄 A：ICIR、Rank ICIR，與 IC 的跨種子檢定（原本在訊號層的其餘欄）。"""
        o = ["| 方法 | n | ICIR ↑ | Rank ICIR ↑ | 跨種子 Welch p (IC) | MW p (IC) |",
             "|---|---:|---:|---:|---:|---:|"]
        for label, cat, note, st, arm in rs:
            pw, pu, _ = across_seed(base, st, "ic")
            star = " **" if arm == BEST else " "
            o.append(
                f"|{star}{label}{star.strip()} | {st['n']} | "
                f"{pm(st.get('icir'), st.get('icir_sd'), signed=False)} | "
                f"{pm(st.get('ricir'), st.get('ricir_sd'), signed=False)} | "
                f"{fmt(pw, signed=False)} | {fmt(pu, signed=False)} |")
        return o

    def pf_table(rs):
        o = ["| 方法 | n " + PF_HEAD + "|", "|---|---:" + PF_SEP + "|"]
        for label, cat, note, st, arm in rs:
            star = " **" if arm == BEST else " "
            o.append(f"|{star}{label}{star.strip()} | {st['n']} "
                     + pf_cells(st.get("pf")) + "|")
        return o

    def cal_table(rs):
        o = ["| 方法 | n " + CAL_HEAD + "|", "|---|---:" + CAL_SEP + "|"]
        for label, cat, note, st, arm in rs:
            star = " **" if arm == BEST else " "
            o.append(f"|{star}{label}{star.strip()} | {st['n']} "
                     + cal_cells(st.get("cal"), st.get("cal_sd")) + "|")
        return o

    out.append("## 第一折（主結果）")
    out.append("")
    out.append("### 主表：IC 與 RankIC")
    out.append("")
    out += main_table(main_rows)
    # ── 第二折驗證（proposal §36 事先登記、§38/§40 判定）──────────────
    out.append("")
    out.append("## 第二折驗證（獨立測試期）")
    out.append("")
    out.append("切法：train 0-902 / val 903-1149，評估窗 **2023-12-20 ~ 2024-12-25**"
               "（246 天），與第一折的測試期完全不重疊，且模型從未在其上做過任何選擇。"
               "所有線性 baseline 皆以第二折切法重新擬合，非沿用第一折的版本。"
               "[24] 兩折都用滾動版（原文做法，2026-10-16 統一）。"
               "6 個文獻神經模型 2026-10-17 以第二折切法重訓，各 3 顆種子、設定與第一折相同（未調參，§74）。")
    out.append("")
    f2 = _fold2_rows()
    if f2 is None:
        out.append("_（第二折的 run 或 baseline 尚未齊備）_")
    else:
        f2_sig, f2_oth, f2_pf, f2_cal = f2
        out.append("### 主表：IC 與 RankIC（第二折）")
        out.append("")
        out += F2_MAIN_HEAD
        out += [line for lab, line in f2_sig if lab not in SUPPLEMENT]
    out.append("")
    out.append("**判定（§38，原主結果）**：對自家基準 ΔIC +0.0308 / ΔRankIC +0.0262，"
               "10/10 種子，**逐日檢定兩個指標都過**（0.0004 / 0.0036）——"
               "本專案第一次有架構改動在逐日層級對自家基準顯著。")
    out.append("")
    out.append("**但（§40，原主結果）**：對線性 baseline（見附錄 B）**不成立**。§40 當時 IC 名目領先每一個"
               "（+0.0055 ~ +0.0203）卻無一顯著；RankIC 與 R2 ridge 恰好打平"
               "（+0.0000），並輸給 KTW+（−0.0115）。"
               "§67 加入兩個線性 baseline 之後（2026-10-13），IC 輸關係型線性（−0.0088，p 0.32）、"
               "贏兩區塊 ridge（+0.0024，p 0.82），RankIC 兩者都輸（−0.0153 / −0.0026，p 0.11 / 0.82），"
               "仍無一顯著。"
               "**兩折合看，相對最佳線性 baseline 是打平，不是超越。**")
    out.append("")
    # ── 2026 保留期（§75）：登記的一次性確認，凍結的第一折模型 ─────────────
    hb = _holdout_block()
    if hb is not None:
        hdays, hlines, has_sa = hb
        out.append("## 2026 保留期（凍結的第一折模型，§75、§81）")
        out.append("")
        out.append(f"評估窗 **{hdays[0]} ~ {hdays[-1]}**（{len(hdays)} 天），資料在 2026-10-18 下載，§75 的登記檢驗只用這一次"
                   "（事前登記，proposal §75）。§75 登記的是前四列的兩個 arm（第一折已訓練好的模型，各 10 顆種子），不重訓、不調參。"
                   "`配對 p` 是投影後版本對原主結果（不帶關係）的同種子配對 t 檢定（登記的判準），`逐日 p` 是種子平均的逐日差的 HAC 檢定。"
                   "**§75 登記的判定：主判準（IC）與第二指標（RankIC）都未通過（當時主結果不改）。**")
        if has_sa:
            out.append("")
            out.append("最後兩列是 2026-10-26 加的**探索性第二次評估**（規則先寫好，proposal §81）：凍結的 SA（自身 + 同產業，§80）"
                       "對原主結果。這是看過 §75 的結果之後才做的同一假設的第二次檢驗，不是確認；配對 p 括號內是 Bonferroni x2。"
                       "SA 自 2026-10-27 起是主結果（§83）：**主結果在 2026 年只有這一次探索性的數字**，"
                       "要寫成「探索性的第二次評估，與原主結果分不出差別」，不能寫成保留期確認。")
        out.append("")
        out += hlines
        out.append("")
    # ── 附錄 A：主表方法的其他指標 ────────────────────────────────
    out.append("## 附錄 A：其他指標（主表的方法）")
    out.append("")
    out.append("主指標（IC、RankIC）以外的指標，2026-10-17 由主表移來，數字與檢定都沒有改。"
               "列序與各折的主表相同（依 IC 排序）。")
    out.append("")
    out.append("### A1. 第一折：ICIR 與跨種子檢定")
    out.append("")
    out += other_table(main_rows)
    out.append("")
    out.append("### A2. 第一折：組合讀法")
    out.append("")
    out.append(f"**這不是回測。** Top-{TOPK} 多空、等權、金額中性、逐日再平衡，"
               "**無交易成本、無市場衝擊、無流動性限制、可完全放空**。"
               "用途是把 IC 翻譯成組合單位，不是策略績效（proposal §60.6 / §60.8）。"
               "**而且它在時序上不可執行**：`y` 從前一日 13:30 收盤起算，訊號要到美股收盤"
               "（台北 04:00／05:00）才存在；§58.3 的分數加權組合有 77.9% 的報酬落在這段隔夜跳空，"
               "拿到訊號後能執行的盤中腿日均只有 +0.0415%、不顯著（§61.3(2)）。"
               "本專案最佳 arm 的**打平來回成本是 33 bp，而台灣證交稅單項就 30 bp**"
               "（§60.4、`scripts/portfolio_readout.py`）。列序與主表相同。")
    out.append("")
    out += pf_table(main_rows)
    out.append("")
    out.append("### A3. 第一折：校準診斷")
    out.append("")
    _bc = base.get("cal")
    _anchor = (f"`Var(y) = {_bc['var_y']:.3e}`（在本欄的單位下是 "
               f"**{_bc['var_y'] * 1e3:.3f}**），`std(y) = {_bc['sd_y'] * 100:.3f}%`"
               if _bc else "（基準列缺預測檔，錨值無法計算）")
    # 下面三段文字裡的列數、Spearman、「幾列在錨之上」、KTW+ 的 IC 名次原本是寫死的
    # （34 列 / +0.90 / 30 列 / 第二高）；2026-10-11 加了集成列之後改成計算值。
    # 計算值在加列之前逐字重現原本的四個數字。「在錨之上」以表上顯示的三位小數比較——
    # RC 的 MSE 與 Var(y) 在這個精度相同，算「恰好落在」，不計入。
    # 2026-10-16 分出附表之後：列數、Spearman、錨之上的列數只算主表；KTW+ 的名次算主表加附表。
    _cals = [st["cal"] for _, _, _, st, _ in main_rows if st.get("cal")]
    _rho = (stats.spearmanr([c["mse"] for c in _cals], [c["disp"] for c in _cals]).statistic
            if len(_cals) > 2 else float("nan"))
    _above = (sum(round(c["mse"] * 1e3, 3) > round(_bc["var_y"] * 1e3, 3) for c in _cals)
              if _bc else 0)
    # 「低於錨」的列（主表加附表）原本寫死為 [24] 0.297 與 R2 0.373；
    # 2026-10-16 [24] 改用滾動版後 MSE 會變，改成計算值。比較精度同 _above（三位小數）。
    _below = ([(r[0], r[3]["cal"]["mse"] * 1e3) for r in ordered if r[3].get("cal")
               and round(r[3]["cal"]["mse"] * 1e3, 3) < round(_bc["var_y"] * 1e3, 3)] if _bc else [])
    _below_txt = ("少數低於它的列（"
                  + "、".join(f"{'附表的 ' if lab in SUPPLEMENT else ''}`{lab}` {m:.3f}" for lab, m in _below)
                  + "）靠的是**比常數還往均值縮**，不是靠預測得準——") if _below else ""
    _k = next((i for i, r in enumerate(ordered, 1) if r[0].startswith("KTW+")), 0)
    _krank = "零一二三四五六七八九十"[_k] if 0 < _k <= 10 else f" {_k} "
    # 2026-10-17 [24] 全部列出後：[24] 的預測是收縮的（離散比 < 1），與過度離散的 MAGNET 各版本混排，
    # Spearman 由 +0.84 降到 +0.65（不含 [24] 仍是 +0.91），措辭改成依計算值決定，並並列兩個數字。
    _cals_nb = [st["cal"] for lab, _, _, st, _ in main_rows if st.get("cal") and not lab.startswith("[24]")]
    _rho_nb = (stats.spearmanr([c["mse"] for c in _cals_nb], [c["disp"] for c in _cals_nb]).statistic
               if len(_cals_nb) > 2 else float("nan"))
    _verb = "大致就是" if _rho >= 0.8 else ("多半跟著" if _rho >= 0.5 else "只部分跟著")
    out.append("**這一組不是準度，讀法和主表相反。** §60.1 宣告 `y_hat` 是"
               "**基數分數、尺度未校正**，所以 MSE / RMSE 量到的主要是**尺度誤差**，"
               "不是排序品質。列序仍照 IC 排，於是本表最重要的一件事直接看得到："
               f"**MSE 由小到大{_verb}離散比由小到大**"
               f"（{len(_cals)} 列的 Spearman = **{_rho:+.2f}**；不含 [24] 的 {len(_cals_nb)} 列是 "
               f"{_rho_nb:+.2f}，[24] 的預測是收縮的，離散比都小於 1），**它量的不是排序品質**。")
    out.append("")
    out.append(f"錨：{_anchor}。`R² = 1 − MSE / Var(y)`，所以 **`MSE (1e-3)` 超過"
               "那個值就代表 level R² 是負的**。關鍵是誰沒超過：附表的 `RC 常數對照` "
               "（離散比 0.026）**恰好落在 0.405**——一個永遠猜均值的模型正是 "
               "`MSE = Var(y)` 的定義。" + _below_txt
               + f"本表 {len(_cals)} 列裡有 {_above} 列在 0.405 之上。"
               "`src/train/metrics.py:regression_metrics` 的 docstring 早就記了"
               "這件事：**MSE 獎勵塌縮**。")
    out.append("")
    out.append("另一頭同樣說明問題：附表的 `KTW+` MSE 是 **37.181**（Var(y) 的 92 倍）、"
               f"離散比 **11.49**，它在主表與附表合計 {len(ordered)} 個方法中 IC 第{_krank}高。"
               "**沒有任何一個方法的 `y_hat` 是校準過的**，所以 MSE 不是一個可以排名次的軸。")
    out.append("")
    out.append("後兩欄把負 R² 拆開：`離散比` 說 `y_hat` 的橫截面比 `y` 寬幾倍"
               "（1.0 是校準），`校正後 R²` 說**扣掉每日橫截面均值、再把尺度仿射"
               "校正之後**還剩多少 level 可解釋變異。它與 `mean IC²` 同量級但不相等"
               "（主結果 **0.0135** vs `mean_t(IC_t)² = 0.1090² = 0.0119`），"
               "差在攤平的 corr 用橫截面離散度給每天"
               "加權。**這一欄才是與 Gu/Kelly/Xiu 那類 level R² 可比的數字。**")
    out.append("")
    out += cal_table(main_rows)
    if f2 is not None:
        out.append("")
        out.append("### A4. 第二折：ICIR")
        out.append("")
        out += F2_OTHER_HEAD + [line for lab, line in f2_oth if lab not in SUPPLEMENT]
        out.append("")
        out.append("### A5. 第二折：組合讀法")
        out.append("")
        out.append("建構與警語同 A2。列序與第二折主表相同。")
        out.append("")
        out += ["| 方法 | n " + PF_HEAD + "|", "|---|---:" + PF_SEP + "|"]
        out += [line for lab, line in f2_pf if lab not in SUPPLEMENT]
        out.append("")
        out.append("### A6. 第二折：校準診斷")
        out.append("")
        out.append("定義與警語同 A3。列序與第二折主表相同。")
        out.append("")
        out += ["| 方法 | n " + CAL_HEAD + "|", "|---|---:" + CAL_SEP + "|"]
        out += [line for lab, line in f2_cal if lab not in SUPPLEMENT]
    out.append("")
    out.append("## 附錄 B：同輸入的對照（非文獻）")
    out.append("")
    out.append("我們另外建構、又不是出自文獻的對照：吃同一組昨日報酬的線性模型"
               "（KTW+、R2、TW+、關係型、兩區塊）與空模型 RC，2026-10-16 依教授建議由主表移來。"
               "欄位定義、統計基準與檢定都和主表與附錄 A 相同，`逐日 p` 仍是主結果對該列；"
               "主結果與其 10 顆集成兩列只是參照，數字與主表相同。")
    out.append("")
    for title, lines in (("B1. 第一折：IC 與 RankIC", main_table(app_rows)),
                         ("B2. 第一折：ICIR 與跨種子檢定", other_table(app_rows)),
                         ("B3. 第一折：組合讀法", pf_table(app_rows)),
                         ("B4. 第一折：校準診斷", cal_table(app_rows))):
        out += [f"### {title}", ""] + lines + [""]
    if f2 is not None:
        refs = {f"**{MAIN_LABEL}**", ENSEMBLE[0][0]}

        def app2(xs):
            return [line for lab, line in xs if lab in SUPPLEMENT or lab in refs]
        for title, head, xs in (
                ("B5. 第二折：IC 與 RankIC", F2_MAIN_HEAD, f2_sig),
                ("B6. 第二折：ICIR", F2_OTHER_HEAD, f2_oth),
                ("B7. 第二折：組合讀法", ["| 方法 | n " + PF_HEAD + "|", "|---|---:" + PF_SEP + "|"], f2_pf),
                ("B8. 第二折：校準診斷", ["| 方法 | n " + CAL_HEAD + "|", "|---|---:" + CAL_SEP + "|"], f2_cal)):
            out += [f"### {title}", ""] + head + app2(xs) + [""]
    out.append("## 附錄 C：種子集成與混合（proposal §43）")
    out.append("")
    out.append("以下三塊**不需要重訓任何模型**，全部由既有預測檔重算。"
               "(A) 改變聚合方式，(B) 是它對 baseline 的後果，"
               "(C) 是與 KTW+（§43 當時的最強線性 baseline）的等權混合，"
               "(C2) 是混合模型對 §67 兩個線性 baseline 的檢定。")
    out.append("")
    eb = _ens_block()
    if eb is None:
        out.append("_（集成所需的預測檔尚未齊備）_")
    else:
        out += eb
    out.append("")
    out.append("## 附錄 D：準度指標，先在 val 定尺度（proposal §68）")
    out.append("")
    out.append("**為什麼要這一節。** `y_hat` 是「基數分數、尺度未校正」（proposal §60.1），"
               "上面校準診斷的 MSE 量到的是尺度，不能拿來比準度。這一節讓每個模型先站上同一個報酬尺度，再比準度：")
    out.append("")
    out.append("1. 每天先扣掉 50 檔的橫截面平均。任務只比相對強弱，沒有模型在預測大盤當天的漲跌。")
    out.append("2. 每個模型在 **val** 上以最小平方擬合一個倍數 `b`（`y ≈ b·y_hat`），固定後套到 test。"
               "test 不參與定尺度。")
    out.append("3. test 上算 `RMSE` 與 `R²_oos`，基準是「不選股」：每檔都猜當天平均，`R²_oos = 0`。"
               "`命中率` 只用名次，不需要定尺度。")
    out.append("")
    out.append("乘一個倍數不改相關係數，所以本節不影響任何 IC / RankIC。"
               "**本節是看過示範結果之後才加的，屬描述性，p 值未校正多重比較。**"
               "只列主結果、它的集成與附表的線性 baseline。線性 baseline 的 `runs/` 只存 test 預測，"
               "val 預測由 `tw_lag_channel.py` / `relational_linear.py` 重算，"
               f"重算的 test 預測與存檔一致才採用（相對差 ≤ {ACC_TOL:g}）。")
    out.append("")
    out.append("**箭頭：↑ 越高越好，↓ 越低越好。** p 值是雙尾、沒有方向，誰比較好看合併表的差值欄。"
               f"`{ANCHOR}` 列的兩個 p 是主結果對它的檢定（`R²_oos` 是否大於 0、命中率是否大於 50%），"
               "其餘列的 p 都是該列對主結果。")
    out.append("")
    ab = _acc_block()
    if ab is None:
        out.append("_（準度小表所需的資料或預測檔尚未齊備）_")
    else:
        out += ab
    out.append("")
    out.append("## 每一欄怎麼算")
    out.append("")
    out.append("記號：某一天的橫截面有 `n` 檔台股，`y_hat` 是預測、`y` 是**實際次日 "
               "log return**（`log(Close_t / Close_{t-1})`，"
               "`src/dataset/pipeline.py:659`，由 `graph_builder.py` 的 "
               "`_extract_target_returns` 取出）。`T` 是**有效交易日數**——"
               "50 檔 `y` 或 `y_hat` 完全相同（`nunique() == 1`，相關係數無定義）的日子一律跳過，"
               "第一折 246 天扣掉 1 天（該日 50 檔報酬完全相同）得 **T=245**。"
               "2026-10-21 以前用 `std() == 0` 判斷：同值的 std 因捨入不一定恰為 0，"
               "結果隨 pandas 是否裝了 bottleneck 而不同（proposal §77）。")
    out.append("")
    out.append("### 主表與附錄 A 的欄")
    out.append("")
    out.append("| 欄 | 算式 | 出處 |")
    out.append("|---|---|---|")
    out.append("| `n` | 種子數。線性／樹模型是確定性擬合（固定設計矩陣上的 "
               "`Ridge.fit` 或封閉解），**沒有 seed 可補**，一律 n=1 | `find_seeds` |")
    out.append("| `IC`（主指標） | 逐日 `corr(y_hat_t, y_t)`（Pearson）對 T 天取平均，"
               "再跨 seed 取平均。**逐 seed 的值讀自 `meta.json`**"
               "（`--predictions reeval` 時改讀 `meta_reeval.json`） | `neural_stats` |")
    out.append("| `ICIR`（附錄 A） | `mean_t(IC_t) / sd_t(IC_t)`，`ddof=1`。"
               "**逐 seed 算完再跨 seed 平均**——拿跨 seed 平均後的日序列去算"
               "會得到「集成」的 ICIR，系統性偏高 | `icir()` |")
    out.append("| `RankIC`（第二指標） | 同 `IC` 但用 Spearman（平手取**平均名次**，`scipy.spearmanr`）。"
               "**一律由預測檔重算，不讀 `meta.json`** | `daily_series` |")
    out.append("| `Rank ICIR`（附錄 A） | `mean_t(RankIC_t) / sd_t(RankIC_t)`，`ddof=1`，"
               "逐 seed 算完再平均。**它沒有組合讀法**——`ICIR` 對應 Sharpe 是因為"
               "§57.8 的恆等式建立在 Pearson IC 上，`RankIC` 沒有對應的恆等式（§57.3）。"
               "它只回答「名次相關本身穩不穩」 | `icir(daily RankIC)` |")
    out.append("| `± sd` | 跨 seed 的樣本標準差，`ddof=1`。n=1 時不印——"
               "**那不代表沒有不確定性**，只代表它來自資料而非種子 | `pm()` |")
    out.append("| `逐日 p (IC)` / `逐日 p (RankIC)` | 取「統計基準與該列」的**逐日 IC（或 RankIC）差值序列** `d_t`，"
               "做 Newey-West HAC 修正的雙尾 t 檢定；"
               "`lag = floor(4 x (T/100)^(2/9))`，T=245 時 **lag=4**，`df = T−1` "
               "| `paired_daily` |")
    out.append("| `跨種子 Welch p (IC)`（附錄 A） | 在**共同種子集合**上，對兩組逐 seed 的 IC 做 "
               "Welch t 檢定（不假設等變異） | `across_seed` |")
    out.append("| `MW p (IC)`（附錄 A） | 同 Welch，改用 Mann-Whitney U，雙尾。"
               "**n=3 時下限是 `2/C(6,3) = 0.1000`**，達不到 0.05 | `across_seed` |")
    out.append("")
    out.append("### 校準診斷的欄")
    out.append("")
    out.append("**這四欄不是準度指標，是宣告的一致性檢查。** 它們回答「`y_hat` "
               "離 `y` 的尺度有多遠」，不回答「排序對不對」——後者是 IC / RankIC 的事。")
    out.append("")
    out.append("| 欄 | 算式 | 出處 |")
    out.append("|---|---|---|")
    out.append("| `MSE (1e-3)` | 攤平所有 (日, 檔) 算 `mean((y_hat − y)²)`，"
               "再 x1000。與 `src/train/metrics.py:regression_metrics` 同定義。"
               "**它是損失裡 `mse: 1.0` 那一項在測試集上的值** | `calib_stats` |")
    out.append("| `RMSE (%)` | `sqrt(MSE)` 再 x100，與 `std(y)` 同單位可直接比 "
               "| `calib_stats` |")
    out.append("| `離散比` | `mean_t[ std(y_hat_t) / std(y_t) ]`，**逐日算完再平均**。"
               "與 §60.7 / §52 引用的 2.59（主結果）、1.69（前版 beta）同定義。"
               "**不等於** `regression_metrics` 的 `dispersion`（那個攤平算，"
               "同一個 arm 給 2.13）——差在攤平版把大盤的跨日變動也算進 `std` "
               "| `calib_stats` |")
    out.append("| `校正後 R²` | **先逐日把橫截面均值從 `y_hat` 與 `y` 各自扣掉**，"
               "再攤平配 OLS（`y ~ a·y_hat + b`）後的 level R²；代數上等於扣完均值的 "
               "`corr²`，故直接用 `corr²` 算。**逐日扣均值是必要的**——不扣會混進"
               "「有沒有猜中當天大盤方向」，實測 `R2 ridge` 的日均值相關 +0.5144 "
               "會把它的攤平 R² 抬到 0.0940，超過 IC 更高的主結果（0.0173）三倍多；"
               "扣完之後是 0.0171 vs 0.0140，回到與 IC 一致的次序。"
               "與 `mean_t(IC_t)²` 不相等（0.0135 vs 0.0119）：攤平的 corr 用橫截面"
               "離散度給每天加權，`mean IC` 每天等權 | `calib_stats` |")
    out.append("")
    out.append("### 組合讀法的欄")
    out.append("")
    out.append(f"**建構**：每天把當日 `y_hat` 由高到低排序，"
               f"**前 {TOPK} 檔做多、後 {TOPK} 檔做空、等權**。"
               f"權重是 `±1/(2K)`，所以 `sum|w| = 1`（總曝險 1、淨曝險 0），"
               "因此當日報酬是")
    out.append("")
    out.append("```")
    out.append(f"r_t = ( mean(y[多腿 {TOPK} 檔]) − mean(y[空腿 {TOPK} 檔]) ) / 2")
    out.append("```")
    out.append("")
    out.append("那個 `/2` 不可省——不除等於偷偷假設 2 倍槓桿。"
               "六個指標**一律逐 seed 算完再平均**，與 `ICIR` 同慣例。")
    out.append("")
    out.append("| 欄 | 算式 | 為什麼是這個形式 |")
    out.append("|---|---|---|")
    out.append("| `日均報酬` | `mean(r)` | — |")
    out.append("| `日 sd` | `sd(r, ddof=1)` | — |")
    out.append("| `年化(幾何)` | `exp(mean(r) x 252) − 1` | **`y` 是 log return，"
               "所以複利用 `exp` 不是 `(1+r)` 連乘。** 用錯公式在本專案的量級上"
               "差約 1.1 個百分點（82.0% vs 83.1%） |")
    out.append("| `Sharpe` | `mean(r) / sd(r) x sqrt(252)`，`rf = 0` | "
               "金額中性多空是**自融資**的，多空相抵不佔用本金，"
               "所以無風險利率為 0。淨多頭策略就必須扣 |")
    out.append("| `MDD(複利)` | 權益曲線 `exp(cumsum(r))`，"
               "`max_t (peak_t − v_t)/peak_t` | 獲利滾入的讀法 |")
    out.append("| `MDD(加總)` | 權益曲線 `1 + cumsum(r)`，同上取最大回撤 | "
               "**獲利不滾入的固定名目本金**。金額中性多空每天重設回同樣曝險，"
               "這個讀法才對應實際操作 |")
    out.append("")
    out.append("**已移除的兩欄可以自己還原**，它們是純換算、沒有新資訊：")
    out.append("")
    out.append("```")
    out.append("年化(簡單) = 日均報酬 x 252          年化波動 = 日 sd x sqrt(252)")
    out.append("Sharpe     = 年化(簡單) / 年化波動    （所以 Sharpe 才是新資訊）")
    out.append("```")
    out.append("")
    out.append("**已知近似**：對 log return 取橫截面平均**不等於**等權組合的報酬"
               "（後者是簡單報酬的算術平均）。實測差 **+0.14 bp/日 = +0.36 pp/年**，"
               "不影響 `Sharpe`，也不影響 §60.4 的打平成本（兩種算法都是 33 bp）。")
    out.append("")
    out.append("### §43 專屬的欄")
    out.append("")
    out.append("| 欄 | 算式 |")
    out.append("|---|---|")
    out.append("| `離散比 std(y_hat)/std(y)` | 逐日的橫截面標準差之比，對 T 天取平均。"
               "**在 ŷ 宣告為「基數分數、尺度未校正」下這是規範選擇不是缺陷**"
               "（IC 與 RankIC 都看不到它，proposal §57.11 / §60.7） |")
    out.append("| `其 IC` / `其 RankIC` / `其 ICIR` / `其 Rank ICIR`（B 表） | "
               "該 baseline 自己的值，算法同訊號層（逐 seed 算完再平均） |")
    out.append("| `ICIR` / `Rank ICIR`（A 表） | **兩列的算法刻意不同，這正是 A 表在量的東西**："
               "「先算 IC 再平均」列是**逐 seed** 算完再平均；"
               "「先平均預測再算 IC」列是拿**集成後的單一日序列**去算。"
               "後者系統性偏高，`增益` 列就是這個差 |")
    out.append("| `增益`（A 表） | 「先平均預測再算 IC」減「先算 IC 再平均」。"
               "它是兩列相減，**沒有對應的日報酬序列**，故組合欄留空 |")
    out.append("")
    out.append("### 準度小表的欄")
    out.append("")
    out.append("下標 `d` 表示扣掉當天 50 檔的橫截面平均；`b` 只用 val 擬合（`b = Σ y_d·y_hat_d / Σ y_hat_d²`）。"
               "MAGNET 各版本逐 seed 算完再平均（`± sd` 為跨 seed），檢定用跨 seed 平均後的逐日序列；"
               "集成列先平均 10 顆的預測再定尺度。日子的取捨同訊號層（50 檔 `y` 完全相同的日子跳過）。")
    out.append("")
    out.append("| 欄 | 算式 | 方向 |")
    out.append("|---|---|---|")
    out.append("| `RMSE (%)` | test 上 `sqrt(mean_t mean_j (y_d − b·y_hat_d)²)`，再 x100。"
               "`不選股` 列就是當天報酬的橫截面標準差（平方平均） | **↓ 越低越好** |")
    out.append("| `R²_oos` | `1 − Σ_t Σ_j (y_d − b·y_hat_d)² / Σ_t Σ_j y_d²`。"
               "分母是「不選股」的誤差，所以 0 = 與不選股一樣，負值 = 比不選股差。"
               "與校準診斷的 `校正後 R²` 不同：那一欄的倍數是在 test 上擬合的，只能當上限 | **↑ 越高越好** |")
    out.append("| `命中率` | 每天把 `y_hat` 與 `y` 各自和當天中位數比，落在同一側的比例；"
               "剛好等於中位數的不計。50% = 亂猜；常態假設下約為 `50% + arcsin(IC)/π` | **↑ 越高越好** |")
    out.append("| `逐日 p（R²_oos）` | 逐日橫截面 MSE 差的 HAC 雙尾檢定（Diebold–Mariano 檢定），"
               "lag 同訊號層 | 無方向 |")
    out.append("| `逐日 p（命中率）` | 逐日命中率差的 HAC 雙尾檢定 | 無方向 |")
    out.append("| `R²_oos 差` / `命中率差`（合併表） | 該列 − 主結果，兩折的日子一起算；"
               "正值代表該列較好 | **↑ 越高越好**（站在該列的立場） |")
    out.append("")
    out.append("## 讀表注意")
    out.append("")
    out.append("**2026-10-27 起主結果改為「MAGNET（+ 同產業關係）」（SA，proposal §83）。**下列各條多數寫於此前，"
               "其中的「本版」「主結果 arm」指不帶關係的原主結果（表中「MAGNET 不帶關係（原主結果）」那一列）；"
               "換成 SA 之後的檢定數字以主表與附錄為準，兩者的比較見 proposal §78 – §83。")
    out.append("")
    out.append("- **欄名的 ↑ / ↓ 是「其他條件相同下哪個方向較好」**，"
               "**不是說該欄可以單獨拿來排名**。有四類欄位**刻意沒有箭頭**：<br>"
               "(1) **`sd` 各欄**（跨 seed 離散度）——離散度小只代表結果比較不挑種子，"
               "一個爛模型的 sd 小並不好，它不是績效；<br>"
               "(2) **`逐日 p (IC)` / `逐日 p (RankIC)` / `Welch p` / `MW p`**——這些是"
               "**「統計基準對該列」**的檢定與差值，不是該列自己的績效。"
               "p 小代表基準顯著贏過該列，站在該列的立場方向是相反的；<br>"
               "(3) **`離散比 std(y_hat)/std(y)`**（§43 A 表）——在 ŷ 宣告為"
               "「基數分數、尺度未校正」之下，IC 與 RankIC 對仿射變換不變、"
               "**兩者都看不到這個比值**，所以它是規範選擇不是缺陷"
               "（proposal §57.11、§60.7）；<br>"
               "(4) **準度小表的 `逐日 p（R²_oos）` / `逐日 p（命中率）`**——雙尾檢定，"
               "**方向看合併表的差值欄**。那裡有線性 baseline 贏過主結果的格子，"
               "不能套用 (2) 的「基準贏過該列」。")
    out.append("- **`MDD ↓` 的箭頭尤其不等於可以拿來排名**（見下方 MDD 那條）："
               "它兩折的名次會翻轉。箭頭只說明方向，不背書該欄的可靠度。")
    out.append("- **n 是種子數。** sigma_seed = 0.0062，n=3 的最小可偵測差異為 0.0188、"
               "n=10 為 0.0082。n=3 的比較若差異小於 0.019，「不顯著」不構成證據。")
    out.append("- **Mann-Whitney 在 n=3 的雙尾 p 下限是 0.10**（= 2/C(6,3)，"
               "`across_seed` 取共同種子集合，故 n=3 的列是 3 對 3），永遠無法達到 0.05。"
               "**附錄 A1 九列 n=3 的 MW p 全部恰為 0.1000——那是下界，不是量測值**，"
               "不可讀成「接近顯著」；同樣九列的 Welch p 是 0.0000 ~ 0.0016，"
               "IC 差距約 0.10 而 sd 約 0.01。"
               "n=1（非神經方法）則無法計算，跨種子欄為「—」。")
    out.append("- **第二折表也有 n 與 sd 欄了**（2026-09-14 補）。"
               "點估計逐位元未變，只是原本沒把離散度印出來。"
               "**但 ICIR 沒有逐日檢定可做**——它是整段期間兩個動差的比值，"
               "不存在逐日序列，所以它只有跨種子這一個檢定，"
               "而跨種子檢定**不包含日層級的不確定性**（那才是主要來源）。"
               "因此 ICIR 的差只能說「兩折同向」，不可升級成「顯著落後」。")
    out.append(f"- **附錄 A 的組合讀法（六欄）不是回測。** 建構：每天按 `y_hat` 取"
               f"**Top-{TOPK} 做多 / Bottom-{TOPK} 做空、等權、金額中性**"
               f"（權重 ±1/(2K)，`sum|w|=1`，所以多空腿的報酬差要除以 2），"
               "逐日再平衡。**無交易成本、無市場衝擊、無流動性限制、可完全放空**——"
               "所以 `Sharpe` 5.86 / `年化` 60% 這種數字**不可以寫成策略績效**"
               "（proposal §60.8）。打平成本與成本敏感度見 §60.4 與 "
               "`scripts/portfolio_readout.py`：本專案最佳 arm 的打平來回成本是 "
               "**33 bp**，而台灣證交稅單項就 30 bp。")
    out.append("- **組合讀法在時序上不可執行**（2026-09-30 補，proposal §61.3(2)）："
               "模型要在 09:00 開盤前給出排序，輸入最晚到美股收盤（台北 04:00／05:00），"
               "但 `y` 是前一日 13:30 收盤到當日收盤——隔夜跳空那一段在訊號出現前就開始了。"
               "§58.3 的分數加權組合有 **77.9%** 的報酬落在這一段；拿到訊號後能執行的只有盤中腿"
               "（09:00 -> 13:30），日均 +0.0415%、逐日 HAC p 0.1395，不顯著。"
               "所以組合讀法的六欄**即使成本為零也不是可執行的報酬**，只能當「IC 換算成組合單位」。"
               "對照 Qlib：Alpha158 的標籤是 T+1 收盤到 T+2 收盤，標籤本身就是可執行的報酬，"
               "它的回測指標才成立；本表的 `y` 沒有這個性質。")
    out.append("- **組合欄的複利用 `exp`，因為 `y` 是 log return**"
               "（`src/dataset/pipeline.py:659`）。`年化(幾何)` = "
               "`exp(mean(r) x 252) − 1`，`MDD(複利)` 的權益曲線 = `exp(cumsum(r))`；"
               "`MDD(加總)` 用 `1 + cumsum(r)`，即獲利不滾入的固定名目本金"
               "（金額中性多空每天重設回同樣曝險，這個讀法才對應實際操作）。"
               "`年化(簡單)` = `mean(r) x 252`。六欄一律**逐 seed 算完再平均**，與 ICIR 同慣例。"
               "**已知近似**：對 log return 取橫截面平均不等於等權組合的報酬"
               "（後者是簡單報酬的算術平均），實測差 **+0.14 bp/日 = +0.36 pp/年**，"
               "不影響 Sharpe，也不影響打平成本（兩種算法都是 33 bp）。")
    out.append("- **`MDD` 是單路徑極值，一折只有一個觀測值，是六欄裡最不可靠的，"
               "而且它兩折的名次是反的**：第一折 MAGNET 4.51% 勝過 KTW+ 5.05%，"
               "第二折 1.70% 卻輸給 1.32%。**不要拿 MDD 做任何排名主張。**"
               "會翻轉正是它不可靠的直接證據——與 `ICIR` 兩折同向落後恰成對比。")
    out.append("- **逐日 p 用 Newey-West HAC 修正自相關**，以「日」為重複單位、種子視為固定，"
               "敏感但不外推到新種子。兩欄應一起看。")
    out.append("- **六個文獻 baseline 各只跑一組預設超參，MAGNET 跑了約 50 組設定。**"
               "兩折都只有 3 顆種子（第二折 2026-10-17 補跑，§74）。"
               "這是目前最大的公平性缺口，比較結果須據此保留。")
    out.append("- **`log_volume_z` 的資料缺陷不是文獻 baseline 表現差的原因**（2026-09-27 驗證，"
               "proposal §55.9(h)）：資料層有 398 列成交量為 0（7 個全市場日期與個股停牌），"
               "`log1p(0)` 的離群值讓這個特徵在 26.1% 的股票-日被壓縮（放大倍數中位數 4.63）。"
               "六個文獻 baseline 吃的 9 個特徵包含它；MAGNET 主結果 arm 只用 log_return，不受影響。"
               "以修正版特徵、同 config 同種子重訓：**IC 平均變動 +0.0006**（18 對，配對 p 0.57，"
               "8/18 為正），最大為 HATS +0.0066（3/3 種子，p 0.18）；修正後六個 baseline 仍在 "
               "−0.0037 ~ +0.0121。**比較結論不變**，上一條的超參缺口仍是主要保留。"
               "重現：`scripts/logvol_fix_check.py`。")
    out.append("- **ICIR = mean(逐日 IC) / sd(逐日 IC)**，逐 seed 算完再跨 seed 平均"
               "（不是拿跨 seed 平均後的日序列去算——那是**集成**的 ICIR，系統性偏高；"
               "§43 (A) 現在把這個差量出來了：**第一折 +0.0087、第二折 +0.0198**，"
               "與 IC 的集成增益 +0.0074 / +0.0130 同量級甚至更大）。"
               "**為什麼要看它**：§57.8 的恆等式是 `r_t = IC_t x sigma_t`，"
               "所以 IC 只決定組合報酬的**分子**；若 sigma_t 為常數，"
               "`Sharpe = ICIR x sqrt(252)`。**與回測 Sharpe 對應的是 ICIR，不是 IC。**"
               "實測（§60）本專案最佳 arm 的 IC 高於 KTW+（+0.1090 vs +0.1077）"
               "但 ICIR 較低（0.4368 vs 0.4810），Top-10 多空的 Sharpe 也較低"
               "（5.86 vs 6.53）——**IC 的排名不保證回測的排名**。")
    out.append("- **`Rank ICIR` 有報，但它沒有組合讀法**（2026-09-15 補；"
               "本表原本刻意不報它）。`ICIR` 之所以重要是因為 §57.8 的恆等式 "
               "`r_t = IC_t x sigma_t` 建立在 **Pearson** IC 上，所以 "
               "`Sharpe = ICIR x sqrt(252)`。**`RankIC` 沒有對應的恆等式**"
               "（§57.3：名次加權組合對應的是 `corr(rank(y_hat), r)`，不是 RankIC），"
               "因此 **`Rank ICIR` 不可以拿來推論任何組合績效**，"
               "它只回答「名次相關本身穩不穩」。"
               "補報的理由有二：Qlib / Stockformer 那一系的標準區塊是四個一起報（§60.1），"
               "以及 §35.3 發現 `無A₂` 對 RankIC（+0.0077）與 ICIR（−0.0258）方向相反，"
               "**需要它才答得出「那個 RankIC 增益本身穩不穩」**。")
    out.append("- **主表只報 IC 與 RankIC（附錄 A 另有 ICIR 與 Rank ICIR），四者都是相關係數的函數、"
               "對預測的尺度不敏感。**"
               "主結果 arm 的預測是**過度離散**的——逐日橫截面 "
               "std(y_hat)/std(y) = **2.59**（beta 層前版 1.69、更早的版本 0.47 是收縮）。"
               "MSE 隨之由 0.00106 升到 **0.00204**，是最早版本的 4.5 倍。"
               "排序能力的提升是真的，但任何報 MSE / R² 或做組合回測的地方"
               "都必須先處理這一點。")
    out.append("- **不可寫「超越線性 baseline」。** 第一折對 KTW+ 名目領先"
               "（IC +0.0013 / RankIC +0.0021）但逐日檢定全部不顯著；"
               "**第二折 RankIC 反而輸給 KTW+（−0.0115）、與 R2 ridge 恰好打平"
               "（+0.0000）**。兩折合看是打平。可以宣稱的是"
               "**對自家前一版的架構改善**（第二折逐日 p 0.0004 / 0.0036）。"
               "**此條指的是純 MAGNET；種子集成後仍成立**（對 KTW+ 兩折兩指標皆不顯著）。"
               "唯一在兩折上都顯著超越 KTW+ 的是 §43 (C) 的**等權混合**，"
               "但那是混合模型的主張，不是本架構單獨的主張。"
               "§67 加入的兩個線性 baseline 讓這條更明確：關係型線性的 IC 兩折都高於主 arm 的"
               "種子平均（+0.0011 / +0.0088，p 0.92 / 0.32），兩區塊 ridge 第一折也較高"
               "（+0.0073，p 0.63），全部不顯著。")
    out.append("- **`關係型線性` 與 `兩區塊 ridge`（2026-10-13 加，proposal §67）。** "
               "兩者都是線性-z 協定（逐日橫截面 z 目標、無截距、超參以 val 平均 IC 選），"
               "差在台股昨日報酬那一塊：兩區塊讓美股、台股各自收縮，台股 50x50 係數自由；"
               "關係型只用 3 個共用權重（自身、同產業平均、其他產業平均）。"
               "關係型是在 val 上事後找到的，登記後在 test 只算一次，對兩區塊的主判準**未通過**"
               "（−0.0063 / +0.0113，串接 p 0.65）。依登記的規則，關係型的 IC 兩折都高於 KTW+、"
               "ICIR 只有第二折較高，**與 KTW+ 並列最高標**；兩區塊的 IC 兩折也都高於 KTW+，"
               "依同一個公平原則列入。兩者的 `y_hat` 是 z 尺度，校準欄與 KTW+ 一樣不能拿來排名。"
               "重現：`scripts/relational_linear.py --save`。")
    out.append("- **單折排名會翻轉。** 第一折的優勢集中在測試期前半"
               "（IC 差前半 +0.0300、後半 −0.0141），第二折則相反（後半更好）。"
               "任何只根據單一測試期的排名都不可靠——這是本表最重要的保留條款。")
    out.append("- **`F1 + rank 1.0（保留 A₂）` 的 ICIR（0.4439）看起來比本版（0.4368）高，"
               "但那是噪音**：配對 p=0.2016，本版只在 4/10 顆種子落後。"
               "同一組配對下**本版的 RankIC 高 +0.0103，p<0.0001，10/10 種子**，"
               "IC 高 +0.0033（p=0.055）。組合層面也一致——Sharpe 5.92 vs 5.86 打平。"
               "**差別是 `無A₂` 買到的排序品質**（§35.3：`無A₂` 對 RankIC 的主效應 "
               "+0.0077，10/10）。這一格與 §60.2 那個已被 P6 證偽的「F1 單獨 ICIR 較高」"
               "是同一種讀法陷阱，**不要拿單一指標的名次下結論**。")
    out.append("- **本版的三個改動是兩兩協同，不是三階交互作用**"
               "（§35.3，2026-09-15 補完 2^3 的八格）。"
               "**三階項在 IC / RankIC / ICIR 上都是零**（−0.0002 / −0.0001 / +0.0008，"
               "p 全部 > 0.73）；三個兩兩交互項在 IC 上都是正且顯著。"
               "主效應是 **F1 +0.0050（p=0.026）**、**rank1.0 +0.0037（p=0.016）**、"
               "無A₂ −0.0006（ns）——"
               "舊版寫的「三個改動單獨都無效」描述的是**基準角落的條件效應**，"
               "不是主效應，那是只看一個角落造成的假象。"
               "`無A₂` 的真實作用是**用 ICIR 換 RankIC**"
               "（+0.0077 / −0.0258，兩者皆 10/10 種子），在 IC 上恰好抵消。")
    out.append("- **單市場消融是本表效果量最大的一格，也是「多層圖值不值得」的直接答案。**"
               "`disable_a12` 把 h_L1 在進 fusion 前零化，**參數量 48,548 與完整版完全相同**"
               "（美股側與耦合的 12,256 個參數梯度實測恰為 0），"
               "所以差異可以完全歸因到跨市場資訊。"
               "淨值：主結果 arm **ΔIC +0.1004**（10/10 種子、逐日 HAC p 3.3e-06）、"
               "前一版 arm +0.0961（10/10、3.2e-07）——"
               "**佔模型表現的 92%，切掉後低於常數對照。** 詳見 proposal §41。")
    smkt_note = _smkt_sa_note(base)
    if smkt_note:
        out.append(smkt_note)
    out.append("- **`LSTM only` 不能當單市場對照。** 它是另一個架構（無 GAT、無融合閘門、"
               "無 beta 層），差異裡混了「沒有跨市場」與「少三個模組」。"
               "要回答教授建議 1，用的是上面兩個 `disable_a12` 的消融 arm。")
    out.append("- **兩個單市場 arm 的方向是反的**：台股側資訊較「完整」的 B"
               "（3 特徵 + 2,200 條邊的台股圖）RankIC 反而比 A（1 特徵、圖只剩 self-loop）"
               "低 0.0292，10 顆種子無一例外。理由見 §37——"
               "RSI/BB 的逐日自相關 0.85–0.91（只能產生近乎不變的排序）、"
               "台股圖對相關股票做平滑（抹掉排序唯一需要的橫截面差異）。"
               "有跨市場訊號時這兩項的傷害被蓋過去，切掉後才顯現。")
    out.append("- 單市場的數字**不可解讀成「台股資料沒有預測力」**：T_history=1 是在"
               "「有跨市場資訊」的前提下選的，沒有它時只看前一天本來就極難預測。"
               "佐證：六個文獻的單市場 baseline 也全部落在 −0.0053 ~ +0.0113。")
    out.append("- **§43 (A) 的兩個數字是不同的估計對象，不是同一個東西的兩種算法。**"
               "「先算 IC 再平均」= 隨機抽一個訓練好的模型的期望表現；"
               "「先平均預測再算 IC」= 實際部署那套系統（跑 10 個模型取平均）的表現。"
               "後者較高是因為種子雜訊互相抵消——實測種子兩兩預測相關 0.819，"
               "即每顆種子的橫截面預測有 **18.1%** 是種子特異雜訊"
               "（變異數分解獨立給出 17.7%）。理論 IC(S) = IC(1)·sqrt(S·SNR/(1+S·SNR))，"
               "SNR=4.54，預測 S=1->10 增益 +0.0101，實測 +0.0088。"
               "**論文兩個都要報。**")
    out.append("- **主表的「MAGNET（+ 同產業關係，10 顆集成）」列**（2026-10-11 加，2026-10-27 起改為主結果的集成）就是 §43 (A) 的"
               "「先平均預測再算 IC」：10 顆種子的 `y_hat` 逐 (日, 檔) 平均成一條序列，"
               "再照其他列的算法算每一欄。`n` 記平均了幾顆種子，但只有一條序列，"
               "所以沒有 `± sd`、跨種子欄為「—」；`逐日 p` 是它對統計基準"
               "（同一個 arm 的逐 seed 平均）的配對檢定，也就是**集成增益本身**的顯著性。"
               "統計基準不變，仍是逐 seed 平均那一列。它與其他列不是對等的算力比較（見下兩條）。")
    out.append("- **集成不是挑最好的種子。** 事後每天挑最佳種子可得 RankIC +0.2161，"
               "那是 cherry-picking、不可實現；集成用的是全部 10 顆、事先固定的規則，"
               "得到 +0.1202。另：邊際報酬在 **k=5 就飽和**"
               "（k=1 +0.1081、k=5 +0.1154、k=10 +0.1163、k=inf 的理論上限 +0.1204），"
               "種子加到 30 顆不值得。")
    out.append("- **集成必須聲明它用了 10 倍訓練算力，且不是對等的算力比較**——"
               "線性/樹 baseline 是凸問題的確定性解，沒有種子雜訊可平均。"
               "正當性有二：集成是隨機方法部署時的標準作法；"
               "我們自己的 baseline 集裡 [24] 就有 ens-avg 變體。")
    out.append("- **(B) 集成後仍然不能寫「超越所有 baseline」。** 對 KTW+ 兩折兩指標"
               "全部不顯著；表中 p 0.032 / 0.044 / 0.046 **通不過 Holm**"
               "（5 個對照時門檻 0.010）。集成改變的是：第二折 RankIC 對 KTW+ "
               "由 **−0.0115（輸）變成 +0.0006（平）**，且 20 個比較"
               "（2 折 x 5 對照 x 2 指標）**第一次全部同號為正**。"
               "§67 加入兩個線性 baseline 之後是 7 個對照、28 個比較，Holm 門檻變成 0.0071："
               "集成對它們的 8 個檢定全部不顯著（p 0.17 ~ 1.00），而且不再全部同號為正——"
               "第二折 RankIC 輸關係型線性（−0.0033），第一折 IC 與兩區塊 ridge 打平（+0.0000）。")
    out.append("- **(C) 等權混合在兩折上都顯著超越 KTW+**"
               "（四個檢定 p 0.0233 / 0.0041 / 0.0002 / 0.0177 全部 < 0.05，"
               "且兩折 ΔIC 都在逐日 MDE +0.0156 之上）。"
               "**w=0.5 是事先可指定的等權，沒有在測試集上調參**"
               "（附帶事實：兩折最佳 w 落在 0.55 與 0.6，等權接近最優）。"
               "機制：MAGNET 與 KTW+ 的逐日預測相關只有 **+0.472 / +0.490**，"
               "一半以上的橫截面資訊不重疊——這與 §37.2「優勢集中在最安靜的 20% 日子」一致。"
               "**但 §67 之後最高標不只 KTW+，原本寫的「唯一在兩折上都顯著超越最高標的設定」"
               "不再成立**（(C2)）：對關係型線性，IC 兩折都不顯著（+0.0180 / +0.0126，"
               "p 0.090 / 0.122），RankIC 只有第二折過 0.05（p 0.079 / 0.049）；對兩區塊 ridge，"
               "第二折兩個指標都顯著，第一折只有 RankIC 顯著（IC p 0.31）。"
               "能寫的是「混合模型兩折都顯著勝過它所混的 KTW+」。")
    out.append("- **(C) 的代價：主張形式改變。** 由「MAGNET 比線性準」變成"
               "**「MAGNET 提供線性模型抓不到的增量資訊」**。"
               "後者是可量測的科學陳述（相關 0.48），但**不是**「我們的架構單獨最好」。"
               "純 MAGNET（集成）對 KTW+ 仍然不顯著。")
    out.append("- **`rank_normalize`（訓練目標尺度不變化）已證偽，未列為表列 arm。**"
               "它的 IC 在多數種子上**無定義**——第一折 5/10、第二折 3/10 顆種子的預測"
               "塌縮成常數橫截面（83~100% 的測試日），對照 arm 是 0/10、0/10。"
               "只計未塌縮的種子，ΔIC 仍是 **−0.0724 / −0.1099**（同號 0/5、0/7）。"
               "機制是正規化後的 pairwise 損失在完全塌縮時恰為 ln(2)=0.6931，"
               "而排序不夠好時攤開來的損失是 0.9344——**塌縮是更低的損失狀態**。"
               "完整分析見 proposal §44，論文化素材見 §45。")
    out.append("- **RankIC 的平手處理已於 2026-09-01 修正（D1）。** 舊的 "
               "`_spearman_corr` 用 `argsort(argsort(·))` 取名次，對平手值給相異名次；"
               "實測 y 有 **208/246 天存在平手**（多半是恰好 0% 的股票），"
               "所以 `meta.json` 的 RankIC 一直帶系統性偏差，常數輸入更會回傳"
               "「ticker 順序 vs 真實報酬」的相關而非 NaN。"
               "現已改用平均名次，且本表的 RankIC 一律由預測檔重算。"
               "**影響：IC 逐位元不變，RankIC 上移約 +0.0013 ~ +0.0017、各 arm 同向，"
               "差值與所有 p 值幾乎不動（第二折對 KTW+ −0.0114 -> −0.0115），"
               "沒有結論翻轉。** 詳見 proposal §44.7 與 §54。"
               "掃過全部 631 個預測檔：**主結果 arm、beta 各版本、兩折的 f2_best / f2_base、"
               "全部線性 baseline 的塌縮天數皆為 0**，本表主要數字不受影響。"
               "兩個例外：`tw50_smktA` 每顆種子 1/246 天（對平均影響 < 0.0005）、"
               "`tw50_bl_delta_lag` 27/246 天（**其 RankIC 有 11% 的天數是 ticker 順序捏造的**）。")
    out.append("- DeltaLag 的預測退化（多日全 50 檔近乎同值），其數字不可信，待修。"
               "第二折同樣：種子 7 有 40 天 50 檔的預測完全相同（§74），這些天一律略過（§77）。"
               "第一折種子 7 的 CPU 重評檔有 25 天完全相同，另有 2 天只差 2–8 個 float32 ulp；"
               "這 2 天不算完全相同，仍計入（`meta_reeval.json` 的 IC 也計入）。")
    out.append("- 線性/樹模型（[24]、KTW+、關係型、兩區塊、R2、RC）無隨機種子，sd 欄為「—」。")
    txt = "\n".join(out)
    print(txt)
    if not args.no_write:
        p = ROOT / "docs" / "results_table.md"
        p.write_text(txt + "\n")
        print(f"\n-> 已寫入 {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
