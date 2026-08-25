"""
results_table.py — 產出跨方法的總結果表（docs/results_table.md）

收三類方法，全部在同一個 tw50 walk-forward test（246 天）上：
  1. 神經模型：本專案的 MAGNET 各版本 + 六個文獻 baseline，逐 seed 平均
  2. 線性/樹模型：ridge 階梯、factor_vs_graph、[24] 二部圖重現，單一預測檔
  3. 空模型：RC 常數對照

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

# 主結果 arm。2026-08-23 由 tw50_T1F3bnl1 換成 tw50_inbnw_noskip：
#   - IC +0.0475 -> +0.0517，種子 sd 0.0180 -> 0.0021（小 8.6 倍）
#   - 結構更簡單：拿掉拼接跳接，只把逐特徵正規化移到 LSTM 之前並改用 AdamW
#   - 換基準的理由不是 IC（兩者統計上分不出來，配對 t p=0.31），
#     而是跳接的增益已被證實由前處理缺陷造成：修好輸入正規化之後，
#     跳接的增量由 +0.0234（逐日與跨種子雙雙通過 Holm）掉到 +0.0060（皆不顯著）
#   - 代價：RankIC 由 +0.0513 掉到 +0.0351，須在表下揭露
BEST = "tw50_inbnw_noskip"      # 主結果，所有比較的基準

# (顯示名稱, arm 或 glob, 類別, 備註)
NEURAL = [
    ("MAGNET + beta 層（階段 P1）", "tw50_beta", "本專案",
     "每檔一個 alpha_j / gamma_j"),
    ("MAGNET + 輸入正規化 + AdamW（本版）", "tw50_inbnw_noskip", "本專案",
     "F3, T=1, L=1, 無跳接"),
    ("　└ 同上，Adam", "tw50_inbn_noskip", "本專案", "分離 AdamW 的貢獻"),
    ("　└ 同上，再加拼接跳接", "tw50_inbn_skip", "本專案", "跳接是否仍有增量"),
    ("　└ 同上，F9", "tw50_inbnw_f9_noskip", "本專案", "特徵數是否仍無差異"),
    ("MAGNET + BN 拼接跳接（前版）", "tw50_T1F3bnl1", "本專案", "F3, T=1, L=1"),
    ("MAGNET 原始（9 特徵 T=20 L=2）", "tw50chk_wl0",  "本專案", "F9, T=20, L=2"),
    ("MAGNET F9 T=1 L=1",            "tw50_T1F9",     "本專案", "F9, T=1, L=1"),
    ("MAGNET F3（無跳接）",            "tw50_T1F3rb",   "本專案", "F3, T=1, L=1"),
    ("MAGNET + 稠密耦合 A",            "tw50_T1F9dense","本專案", "A[30,50] 可學"),
    ("Early fusion 對照",             "tw50chk_ef",    "本專案", "拼接後單一編碼器"),
    ("LSTM only（無圖）",              "tw50chk_lstm",  "本專案", "無跨市場"),
    ("HGT [14]",                      "tw50_bl_hgt",       "文獻", "未調參"),
    ("DeltaLag [13]",                 "tw50_bl_delta_lag", "文獻", "未調參，預測退化"),
    ("MEIG [1]",                      "tw50_bl_meig",      "文獻", "未調參"),
    ("Adv-ALSTM [10]",                "tw50_bl_adv_alstm", "文獻", "未調參"),
    ("HATS [12]",                     "tw50_bl_hats",      "文獻", "未調參"),
    ("MAN-SF [11]",                   "tw50_bl_man_sf",    "文獻", "未調參"),
]
NONNEURAL = [
    ("[24] 二部圖 ens-avg",  "*_bipartite_walkforward_t2_ens-avg", "文獻", "t 檢定篩邊 + 集成"),
    ("[24] 二部圖 LASSO",    "*_bipartite_walkforward_t2_LASSO",   "文獻", ""),
    ("KTW+（rank 目標）",     "*_fvg_KTWp",                          "線性", "美股+台股 80 維"),
    ("R2（30 檔美股報酬）",    "*_fvg_R2",                            "線性", "per-target ridge"),
    ("RC 常數對照",           "*_ridge_RC",                          "空模型", "0 特徵"),
]


# 預測檔來源。reeval = CPU 重評版（scripts/reeval_checkpoints.py 產生），
# recorded = run 當下寫的 CSV。含 GAT 的 arm 在 MPS 上的 scatter-add 不可重現，
# 同一份 best.pt 連評 5 次 test IC 全距 0.0169，所以本表預設走 reeval；
# 無重評檔的 arm（無 GAT，或本來就在 CPU 上評的新 run）自動回退到 recorded。
PREDICTIONS = "reeval"


def daily_series(run_dir: str):
    f = os.path.join(run_dir, "predictions", "test_predictions_reeval.csv")
    if PREDICTIONS != "reeval" or not os.path.exists(f):
        f = os.path.join(run_dir, "predictions", "test_predictions.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    H = df.pivot(index="target_date", columns="ticker", values="y_hat").sort_index()
    Y = df.pivot(index="target_date", columns="ticker", values="y").sort_index()
    Hn, Yn = H.to_numpy(), Y.to_numpy()
    ic = np.full(len(Yn), np.nan)
    ric = np.full(len(Yn), np.nan)
    for t in range(len(Yn)):
        a, b = Hn[t], Yn[t]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        ic[t] = np.corrcoef(a, b)[0, 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = stats.spearmanr(a, b).statistic
        ric[t] = np.nan if np.isnan(v) else v
    return H.index.to_numpy(), ic, ric


def find_seeds(arm: str) -> dict[str, str]:
    out = {}
    for d in sorted(glob.glob(str(ROOT / "runs" / f"*{arm}_s*"))):
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
    ics, rics, per_ic, per_ric, params = [], [], [], [], None
    for s, d in sorted(runs.items(), key=lambda kv: int(kv[0])):
        tm = (json.load(open(os.path.join(d, "meta.json"))).get("test_metrics") or {})
        rv = os.path.join(d, "meta_reeval.json")
        if PREDICTIONS == "reeval" and os.path.exists(rv):
            tm = {**tm, **json.load(open(rv))["reevaluated"]}
        if tm.get("IC") is None:
            continue
        per_ic.append(tm["IC"]); per_ric.append(tm["RankIC"])
        ser = daily_series(d)
        if ser is not None:
            ics.append(ser[1]); rics.append(ser[2]); dates = ser[0]
        if params is None:
            params = tm.get("n_params")
    if not per_ic:
        return None
    return dict(seeds=sorted(runs, key=int), n=len(per_ic),
                ic=float(np.mean(per_ic)), ic_sd=float(np.std(per_ic, ddof=1)) if len(per_ic) > 1 else np.nan,
                ric=float(np.mean(per_ric)), ric_sd=float(np.std(per_ric, ddof=1)) if len(per_ric) > 1 else np.nan,
                per_ic=dict(zip(sorted(runs, key=int), per_ic)),
                per_ric=dict(zip(sorted(runs, key=int), per_ric)),
                daily_ic=colmean(np.vstack(ics)) if ics else None,
                daily_ric=colmean(np.vstack(rics)) if rics else None,
                dates=dates if ics else None)


def nonneural_stats(pat: str):
    ds = sorted(glob.glob(str(ROOT / "runs" / pat)))
    if not ds:
        return None
    ser = daily_series(ds[-1])
    if ser is None:
        return None
    dates, ic, ric = ser
    return dict(n=1, ic=float(np.nanmean(ic)), ic_sd=np.nan,
                ric=float(np.nanmean(ric)), ric_sd=np.nan,
                daily_ic=ic, daily_ric=ric, dates=dates, seeds=None)


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


def fmt(v, nd=4, signed=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


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

    base = next(st for _, _, _, st, arm in rows if arm == BEST)
    out = []
    out.append("# 跨方法結果表")
    out.append("")
    out.append("由 `scripts/results_table.py` 產生。universe = tw50（US 30 / TW 50，"
               "配對 7 檔），walk-forward test 246 天，不重切。")
    out.append("")
    out.append(f"統計基準 = **{BEST}**（本專案目前最佳）。差為正代表基準較優。")
    out.append("")
    out.append("| 方法 | 類別 | 設定 | n | test IC | sd | RankIC | sd | dIC | 逐日 p | 跨種子 Welch p | MW p |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, cat, note, st, arm in sorted(rows, key=lambda r: -r[3]["ic"]):
        d_ic, p_pd = paired_daily(base, st, "ic")
        pw, pu, _ = across_seed(base, st, "ic")
        is_base = (arm == BEST)
        star = " **" if is_base else " "
        out.append(
            f"|{star}{label}{star.strip()} | {cat} | {note} | {st['n']} | "
            f"{fmt(st['ic'])} | {fmt(st['ic_sd'], signed=False)} | "
            f"{fmt(st['ric'])} | {fmt(st['ric_sd'], signed=False)} | "
            f"{'—' if is_base else fmt(d_ic)} | "
            f"{'—' if is_base else fmt(p_pd, signed=False)} | "
            f"{fmt(pw, signed=False)} | {fmt(pu, signed=False)} |")
    out.append("")
    out.append("## 讀表注意")
    out.append("")
    out.append("- **n 是種子數。** sigma_seed = 0.0062，n=3 的最小可偵測差異為 0.0188、"
               "n=10 為 0.0082。n=3 的比較若差異小於 0.019，「不顯著」不構成證據。")
    out.append("- **Mann-Whitney 在 n=3 的雙尾 p 下限是 0.10**，永遠無法達到 0.05；"
               "n=1（非神經方法）則無法計算。跨種子欄為「—」代表該方法只有單一預測序列。")
    out.append("- **逐日 p 用 Newey-West HAC 修正自相關**，以「日」為重複單位、種子視為固定，"
               "敏感但不外推到新種子。兩欄應一起看。")
    out.append("- **六個文獻 baseline 各只跑一組預設超參，MAGNET 跑了約 50 組設定。**"
               "這是目前最大的公平性缺口，比較結果須據此保留。")
    out.append("- DeltaLag 的預測退化（多日全 50 檔近乎同值），其數字不可信，待修。")
    out.append("- 線性/樹模型（[24]、KTW+、R2、RC）無隨機種子，sd 欄為「—」。")
    txt = "\n".join(out)
    print(txt)
    if not args.no_write:
        p = ROOT / "docs" / "results_table.md"
        p.write_text(txt + "\n")
        print(f"\n-> 已寫入 {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
