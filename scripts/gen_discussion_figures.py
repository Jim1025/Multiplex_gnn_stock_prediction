"""
gen_discussion_figures.py — 評估章節的圖（英文，2:1）

2026-11-23 依教授意見重新規劃：每個議題對應 MAGNET 與 [1]、[24] 的一項設計差異，橫軸是一個變因，
四個方法（實驗組 MAGNET、MAGNET 單市場；對照組 [24] Bipartite、[1] MEIG）在每張圖都出現，
顏色與標記全章固定。

    eval_plan_overall.svg          議題 1 整體比較：變因 = 測試期（第一折 2025、第二折 2024），
                                   [24] 列出全部 8 個學習器；取代原本的結果表 + 累積 IC 圖
    eval_plan_info_source.svg      議題 2 資訊來源：變因 = 只用台股 / 只用美股 / 台股 + 美股
    eval_plan_info_lag.svg         議題 3 資訊時效：變因 = 美股輸入落後 1–4 天
    eval_plan_market_state.svg     議題 4 市場狀態：變因 = 美股前一晚等權報酬絕對值的五分位
    eval_plan_market_state_signed.svg （備用頁）議題 4 的延伸：變因 = 帶正負號的美股前一晚漲跌五分位
                                   （docs/figures/data/eval_market_state_signed.json，proposal_1209 §5）
    eval_plan_stock_linkage.svg    議題 5 個股與美股的對應：變因 = 台股的分組（ADR 配對 / 未配對電子 / 金融 / 傳產，
                                   另加未配對 43 檔合計），縱軸是逐檔 IC 貢獻的組內平均（proposal_1209 §6）
    eval_plan_ablation.svg         （備用頁）其他元件的消融，只拿掉誤差棒
    eval_plan_cum_ic.svg           （已不用）累積 IC，未改

全部是真實數據，沒有假設值：
    整體比較             由預測檔重算，算法與 scripts/results_table.py 相同（第一折 neural_stats /
                         nonneural_stats / paired_daily；第二折照 _fold2_rows 的規則），執行時印出數字供對照主表
    資訊來源、資訊時效   docs/figures/data/eval_bipartite_variants.json（scripts/bipartite_variants_eval.py --write）
    資訊時效的 MAGNET、MEIG  docs/figures/data/eval_input_lag.json（scripts/input_lag_eval.py --write，proposal_1209 §2）
    市場狀態             docs/figures/data/eval_market_state.json（scripts/market_state_eval.py --write）
    個股與美股的對應     docs/figures/data/eval_stock_linkage.json（scripts/stock_linkage_eval.py --write）
    消融                 docs/results_table.md 與 proposal_819 §83

    .venv/bin/python scripts/gen_discussion_figures.py [--fmt svg|png] [--out DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "svg.fonttype": "none"})

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "figures" / "data"
LASSO = "Bipartite-LASSO [24]"
# 每個方法固定一種顏色與一種標記，全章四張圖一致；圖例也固定照這個順序
STYLE = {
    "MAGNET": dict(color="#b5541c", marker="o", ls="-"),
    "MAGNET-single market": dict(color="#6b7280", marker="s", ls="-"),
    LASSO: dict(color="#1f5f99", marker="^", ls="-"),
    "MEIG [1]": dict(color="#2f7d4f", marker="D", ls="-"),
}
FMT = "svg"


def _load(name: str) -> dict:
    return json.loads((DATA / name).read_text())


def _finish(fig, ax, path, title, note=None):
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    bottom = 0.0
    if note:
        fig.text(0.01, 0.01, note, fontsize=10, color="#6b7280", va="bottom")
        bottom = 0.035 * (note.count("\n") + 1) + 0.01
    fig.tight_layout(rect=(0, bottom, 1, 1))
    path = path.with_suffix("." + FMT)
    fig.savefig(path, dpi=220 if FMT == "png" else None, facecolor="white")
    plt.close(fig)


def _method_legend(ax, names=tuple(STYLE), **kw):
    ax.legend(handles=[Patch(fc=STYLE[m]["color"], ec=STYLE[m]["color"]) for m in names],
              labels=list(names), frameon=False, **kw)


# ---------------------------------------------------------------- 議題 1：整體比較

F2_CUT = "2024-12-25"      # 第二折評估窗的最後一天（與 results_table._fold2_rows 相同）
BIP_LEARNERS = ("LASSO", "RIDGE", "OLS", "RF", "AdaBoost", "LGBM", "XGBoost", "SVM")   # 依第一折 IC 排序


def _f2_daily_ic(pattern: str):
    """第二折逐 run 的逐日 IC。規則同 results_table._fold2_rows：只取 ≤ F2_CUT 的日子，
    50 檔 y 或 y_hat 完全相同的日子略過，各 run 取共同日子。回傳 (日期, 陣列[run, 日])。"""
    per = []
    for f in sorted(glob.glob(str(ROOT / pattern), recursive=True)):
        df = pd.read_csv(f)
        o = {}
        for d, g in df.groupby("target_date"):
            d = str(d)[:10]
            if d > F2_CUT or g.y.nunique() == 1 or g.y_hat.nunique() == 1:
                continue
            o[d] = np.corrcoef(g.y_hat, g.y)[0, 1]
        if o:
            per.append(o)
    if not per:
        raise FileNotFoundError(pattern)
    k = sorted(set.intersection(*[set(x) for x in per]))
    return k, np.array([[x[d] for d in k] for x in per])


def _f2_hac(d) -> float:
    """與 results_table._fold2_rows 內的 _hac 相同（第二折主表的逐日 p）。"""
    n = len(d)
    L = int(4 * (n / 100) ** (2 / 9))
    s2 = np.var(d, ddof=1)
    for l in range(1, L + 1):
        s2 += 2 * (1 - l / (L + 1)) * np.cov(d[l:], d[:-l], ddof=1)[0, 1]
    return float(2 * (1 - stats.norm.cdf(abs(d.mean() / (s2 / n) ** 0.5))))


def _overall_rows() -> list[dict]:
    """兩折的 IC、種子 SD 與「MAGNET − 該方法」逐日 IC 差的 HAC p（MAGNET 本身 p = None）。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    import results_table as R  # noqa: E402

    specs = [("MAGNET", "MAGNET", "Exp", ("neural", R.BEST), f"runs/**/*{R.BEST_F2}_s*"),
             ("MAGNET-\nsingle market", "MAGNET-single market", "Exp",
              ("neural", R.SMKT_SA[0]), f"runs/**/*{R.SMKT_SA[1]}_s*")]
    specs += [("LightGBM" if m == "LGBM" else m, LASSO, "Con", ("bip", R.BIP.format(m)),
               (R._latest(f"runs_f2/{R.BIP.format(m)}") or "runs_f2/__missing__"))
              for m in BIP_LEARNERS]
    specs += [("MEIG [1]", "MEIG [1]", "Con", ("neural", "tw50_bl_meig"), "runs/**/*f2_bl_meig_s*")]

    base1 = R.neural_stats(R.BEST)
    kb, PB = _f2_daily_ic(f"runs/**/*{R.BEST_F2}_s*/predictions/test_predictions.csv")
    VB = np.nanmean(PB, axis=0)
    rows = []
    for label, style, group, (kind, key), f2_dir in specs:
        st = R.neural_stats(key) if kind == "neural" else R.nonneural_stats(key)
        p1 = None if key == R.BEST else R.paired_daily(base1, st, "ic")[1]
        k, P = _f2_daily_ic(f"{f2_dir}/predictions/test_predictions.csv")
        V = np.nanmean(P, axis=0)
        ii = [k.index(d) for d in k if d in set(kb)]
        jj = [kb.index(d) for d in k if d in set(kb)]
        sd2 = float(np.std(np.nanmean(P[:, ii], axis=1), ddof=1)) if P.shape[0] > 1 else np.nan
        dd = VB[jj] - V[ii]
        p2 = None if key == R.BEST else _f2_hac(dd[np.isfinite(dd)])
        rows.append(dict(label=label, style=style, group=group,
                         f1=(st["ic"], st["ic_sd"], p1, st["n"]),
                         f2=(float(np.nanmean(V[ii])), sd2, p2, P.shape[0])))
    return rows


def overall(out):
    rows = _overall_rows()
    print("議題 1（對照 docs/results_table.md 兩折主表）：")
    for r in rows:
        (a, sa, pa, na), (b, sb, pb, nb) = r["f1"], r["f2"]
        fp = (lambda p: "—" if p is None else f"{p:.4f}")
        print(f"  {r['label'].replace(chr(10), ' '):22s} 第一折 {a:+.4f} ± {sa:.4f} p {fp(pa):>6s} (n {na})"
              f" | 第二折 {b:+.4f} ± {sb:.4f} p {fp(pb):>6s} (n {nb})")

    # 實驗組 2 個、對照組 9 個（[24] 8 個學習器 + MEIG），等距排列
    xs = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(12, 6))
    w = 0.38
    for x, r in zip(xs, rows):
        c = STYLE[r["style"]]["color"]
        for k, (fold, hatch) in enumerate((("f1", None), ("f2", "///"))):
            ic, sd, p, _ = r[fold]
            xx = x + (k - 0.5) * w
            ax.bar(xx, ic, w * 0.92, color=c if hatch is None else "white", edgecolor=c, hatch=hatch, lw=1.3)
            lab = f"{ic:.3f}".replace("-", "−")
            top = max(ic, 0) + 0.003
            ax.text(xx, top, lab, ha="center", va="bottom", fontsize=8,
                    color="#111827" if r["style"] == "MAGNET" else "#374151",
                    fontweight="bold" if r["style"] == "MAGNET" else "normal")

    ax.set_xticks(xs, [r["label"] for r in rows], fontsize=11)
    # [24] 8 個學習器的組名，放在方法名稱下方
    bx0, bx1 = xs[2] - 0.35, xs[-2] + 0.35
    ax.text((bx0 + bx1) / 2, -0.050, "Bipartite [24] (8 learners)", ha="center", va="top",
            fontsize=11, color="#1f5f99")
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylim(-0.015, 0.18)
    ax.set_yticks([0, 0.05, 0.10, 0.15])
    ax.set_ylabel("Test IC (mean of daily IC)")
    ax.legend(handles=[Patch(fc="#6b7280", ec="#6b7280"), Patch(fc="white", ec="#6b7280", hatch="///")],
              labels=["Fold 1 (test 2025)", "Fold 2 (test 2024)"], frameon=False, ncol=2,
              loc="upper right", bbox_to_anchor=(1.0, 0.93))
    _finish(fig, ax, out / "eval_plan_overall", "Effect of test period: MAGNET vs. cross-market methods",
            note="Seed means (MAGNET and MAGNET-single market 10 seeds, MEIG 3 seeds; each [24] learner one "
                 "deterministic run).")


# ---------------------------------------------------------------- 議題 2：資訊來源

def info_source(out):
    # 變因 = 預測用的資訊來源。MAGNET 的「只用台股」就是單市場版本；MAGNET 沒有只用美股的版本（沒跑）；
    # MEIG 原生是跨市場架構，拿掉任一市場等於改它的架構，所以只有台股 + 美股。
    # [24] 三種來源都是原文做過的設定：只用台股 = §5.2.2 同市場對照、只用美股 = 主設定（主表）、
    # 台股 + 美股 = Appendix E（proposal_1209 §1.3）。
    v = _load("eval_bipartite_variants.json")
    lasso, mag = v["models"]["LASSO"]["ic"], v["magnet"]
    meig = _load("eval_input_lag.json")["methods"]["MEIG [1]"]["1"]
    srcs = ["TW only", "US only", "TW + US"]
    nan = float("nan")
    # (來源, 槽位) -> (方法, IC, 種子 SD)；None = 不存在，附上說明
    cell = {
        ("TW only", 0): ("MAGNET-single market", mag["tw50_smktSA"]["ic"], mag["tw50_smktSA"]["ic_sd"]),
        ("US only", 0): None,
        ("TW + US", 0): ("MAGNET", mag["tw50_rgcnSA"]["ic"], mag["tw50_rgcnSA"]["ic_sd"]),
        ("TW only", 1): (LASSO, lasso["tw"], nan),
        ("US only", 1): (LASSO, lasso["us"], nan),
        ("TW + US", 1): (LASSO, lasso["both"], nan),
        ("TW only", 2): None,
        ("US only", 2): None,
        ("TW + US", 2): ("MEIG [1]", meig["ic"], meig["ic_sd"]),
    }
    missing = {0: "not\nrun", 2: "n/a"}
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.26
    for i, s in enumerate(srcs):
        for slot in range(3):
            x = i + (slot - 1) * w
            c = cell[(s, slot)]
            if c is None:
                ax.text(x, 0.003, missing[slot], ha="center", va="bottom", fontsize=9, color="#9ca3af")
                continue
            m, val, sd = c
            ax.bar(x, val, w * 0.9, color=STYLE[m]["color"], edgecolor=STYLE[m]["color"])
            ax.text(x, max(val, 0) + 0.003, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(srcs)), srcs)
    ax.set_xlim(-0.5, len(srcs) - 0.5)
    ax.set_xlabel("Predictors: which market's information is used")
    ax.set_ylabel("Test IC (fold 1)")
    _method_legend(ax, loc="upper left")
    _finish(fig, ax, out / "eval_plan_info_source", "Effect of information source",
            note="[24] uses the source settings of its own paper: TW only = same-market baseline (Sec. 5.2.2), "
                 "US only = main setting,\nTW + US = Appendix E. MEIG is cross-market by design. "
                 "MAGNET with US only was not run.")


# ---------------------------------------------------------------- 議題 3：資訊時效

def info_lag(out):
    lags = [1, 2, 3, 4]
    # 全部是真實值（proposal_1209 §1.4、§2）。MAGNET 與 MEIG 的落後 2–4 天是重新訓練的延遲版
    # （只換美股層輸入，圖、台股輸入與模型都不變；MAGNET 10 顆、MEIG 3 顆種子平均）。
    # 單市場版本不用美股、不受落後天數影響，畫成水平線，代表「跨市場資訊完全沒有作用」的水準。
    v = _load("eval_bipartite_variants.json")
    lasso, mag = v["models"]["LASSO"]["ic"], v["magnet"]
    nn = _load("eval_input_lag.json")["methods"]
    ref = mag["tw50_smktSA"]["ic"]
    data = {
        "MAGNET": ([nn["MAGNET"][str(l)]["ic"] for l in lags], [nn["MAGNET"][str(l)]["ic_sd"] for l in lags]),
        "MAGNET-single market": ([ref] * 4, None),
        LASSO: ([lasso["us"], lasso["lag2"], lasso["lag3"], lasso["lag4"]], None),
        "MEIG [1]": ([nn["MEIG [1]"][str(l)]["ic"] for l in lags],
                     [nn["MEIG [1]"][str(l)]["ic_sd"] for l in lags]),
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    # MAGNET 落後 2 天以上與單市場重疊，畫在最上層，免得被單市場的方塊蓋住
    zorder = {"MAGNET": 5, LASSO: 4, "MEIG [1]": 3, "MAGNET-single market": 2}
    for m, (ys, sds) in data.items():
        s = STYLE[m]
        ax.plot(lags, ys, color=s["color"], ls=s["ls"], lw=2, marker=s["marker"], ms=6,
                    mfc=s["color"], mew=2, zorder=zorder[m],
                    label=m + (" (no US input)" if m == "MAGNET-single market" else ""))
    ax.set_xticks(lags, ["t−1\n(default)", "t−2", "t−3", "t−4"])
    ax.set_xlim(0.7, 4.3)
    ax.set_ylim(-0.035, 0.13)
    ax.set_xlabel("Lag of US input (trading days)")
    ax.set_ylabel("Test IC (fold 1)")
    ax.legend(frameon=False, loc="upper right")
    _finish(fig, ax, out / "eval_plan_info_lag", "Effect of information timeliness",
            note="Only the US input is lagged; graphs and TW input still use data up to t−1 (as in Sec. 5.4 of [24]).\n"
                 "Seed means (MAGNET 10 seeds, MEIG 3 seeds).")


# ---------------------------------------------------------------- 議題 4：市場狀態

def market_state(out):
    qs = ["Q1\n(calm)", "Q2", "Q3", "Q4", "Q5\n(large move)"]
    # 全部是真實值：各組的平均逐日 IC。不畫誤差線（四條線疊在一起太亂），標準誤的大小寫在圖下的註記
    methods = _load("eval_market_state.json")["methods"]
    key = {LASSO: "LASSO [24]"}                     # JSON 裡的名稱
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(qs))
    se_max = 0.0
    for m, s in STYLE.items():
        cells = methods[key.get(m, m)]["ic"]
        se_max = max(se_max, *(c["se"] for c in cells))
        ax.plot(x, [c["mean"] for c in cells], color=s["color"], ls=s["ls"], lw=2, marker=s["marker"], ms=6,
                mfc=s["color"], mew=2, label=m)
    ax.set_xticks(x, qs)
    ax.set_xlabel("Size of last night's US move: quintile of |equal-weighted US return|")
    ax.set_ylabel("Mean daily IC (fold 1)")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    _finish(fig, ax, out / "eval_plan_market_state", "Effect of market state",
            note=f"Each point averages about 49 days; its standard error is 0.01–{se_max:.2f}, "
                 f"so gaps smaller than about 0.05 are within noise.")


def market_state_signed(out):
    # 議題 4 的延伸（proposal_1209 §5）：Q1 跌最多、Q5 漲最多；兩端都是大波動，看傳導是否不對稱
    d = _load("eval_market_state_signed.json")
    methods = d["methods"]
    key = {LASSO: "LASSO [24]"}
    qs = [f"Q{c['q']}\n({c['mean']:+.1f}%)".replace("-", "−") for c in d["quintiles"]]
    qs[0] = qs[0].replace("Q1", "Q1 drop")
    qs[-1] = qs[-1].replace("Q5", "Q5 rally")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(qs))
    for m, s in STYLE.items():
        cells = methods[key.get(m, m)]["ic"]
        ax.plot(x, [c["mean"] for c in cells], color=s["color"], ls=s["ls"], lw=2, marker=s["marker"], ms=6,
                mfc=s["color"], mew=2, label=m)
    ax.set_xticks(x, qs)
    ax.set_xlabel("Last night's US move: quintile of signed equal-weighted US return (group mean)")
    ax.set_ylabel("Mean daily IC (fold 1)")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    _finish(fig, ax, out / "eval_plan_market_state_signed", "Effect of market state: drop vs. rally",
            note="Each point averages about 49 days (standard error 0.01–0.04). Q5 − Q1 is not significant "
                 "for any method (Welch t, p ≥ 0.35).")


# ---------------------------------------------------------------- 議題 5：個股與美股的對應

def stock_linkage(out):
    # 變因 = 台股與美股的對應關係（proposal_1209 §6）。逐日 IC 拆成每檔的 z(ŷ)·z(y) 相加，
    # 長條是組內每檔的平均貢獻；方法在每檔都一樣好時，各組都等於它的全期 IC。
    # 最右一組是三個未配對組合併的 43 檔，對應結論「其餘 43 檔分不出差別」，用虛線隔開。
    d = _load("eval_stock_linkage.json")
    lv = d["levels"]
    key = {LASSO: "LASSO [24]"}
    groups = [("ADR-paired", "ADR-paired"), ("Unpaired electronics", "Unpaired\nelectronics"),
              ("Financials", "Financials"), ("Traditional", "Traditional"), ("Unpaired (all 43)", "All unpaired")]
    labels = [f"{lab}\n({d['groups'][g]['n']})" for g, lab in groups]
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.2
    se_max = 0.0
    for gi, (g, _) in enumerate(groups):
        x0 = gi + (0.25 if gi == len(groups) - 1 else 0.0)
        for mi, m in enumerate(STYLE):
            c = lv[key.get(m, m)]["ic"][g]
            se_max = max(se_max, c["se"])
            x = x0 + (mi - 1.5) * w
            ax.bar(x, c["mean"], w * 0.9, color=STYLE[m]["color"], edgecolor=STYLE[m]["color"])
            ax.text(x, max(c["mean"], 0) + 0.003, f"{c['mean']:.3f}".replace("-", "−"), ha="center",
                    va="bottom", fontsize=7)
    xs = [gi + (0.25 if gi == len(groups) - 1 else 0.0) for gi in range(len(groups))]
    ax.axvline((xs[-2] + xs[-1]) / 2, color="#9ca3af", ls=":", lw=1.2)
    ax.set_xticks(xs, labels)
    ax.set_xlim(xs[0] - 0.55, xs[-1] + 0.55)
    ax.set_ylim(-0.02, 0.33)
    ax.set_xlabel("TW stocks grouped by their link to the US layer (fixed before the sample)")
    ax.set_ylabel("Per-stock contribution to IC (fold 1)")
    _method_legend(ax, loc="upper right", ncol=2)
    b = d["contrasts"]["ic"]["B"]                                   # 配對股上 MAGNET − LASSO（登記的對比 B）
    r = d["descriptive"]["ic"]["MAGNET - LASSO [24] | Unpaired (all 43)"]
    _finish(fig, ax, out / "eval_plan_stock_linkage", "Effect of stock-level linkage",
            note="Daily IC = mean over the 50 stocks of z(ŷ)·z(y); each bar averages that term over the group, "
                 "so a method equally good on every stock\nshows its overall IC in every group. "
                 f"MAGNET − [24]: {b['mean']:+.3f} per ADR-paired stock (HAC p {b['p_holm']:.3f} after Holm), "
                 f"{r['mean']:+.3f} on the other 43 (p {r['p']:.2f}).\n"
                 f"Seed means (MAGNET 10 seeds, MEIG 3 seeds); standard errors 0.01–{se_max:.2f}.".replace("-0", "−0"))


# ---------------------------------------------------------------- 備用頁

def ablation(out):
    # 真實數字：proposal §83.3、§83.4、scripts/sa_us_graph_eval.py（第一折 IC，種子平均 ± 種子 SD）
    arms = [("MAGNET", 0.1127, 0.0039, None),
            ("w/o relational\nTW layer", 0.1090, 0.0051, "//"),
            ("w/o US\nintra-layer graph", 0.1027, 0.0034, ".."),
            ("MAGNET-\nsingle market", 0.0250, 0.0021, "--")]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (lab, v, sd, h) in enumerate(arms):
        ax.bar(i, v, 0.6, color="#b5541c" if h is None else "white",
               edgecolor="#b5541c", hatch=h, lw=1.5)
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=11)
    ax.set_xticks(range(len(arms)), [a[0] for a in arms])
    ax.set_ylabel("Test IC (fold 1, mean of 10 seeds)")
    _finish(fig, ax, out / "eval_plan_ablation", "Ablation: where the gain comes from",
            note="Real results. Fold 2 and RankIC are reported in the text.")


def cum_ic(out):
    """第一折累積逐日 IC（真實）：主結果、[24] LASSO、MEIG [1]。資料讀取方式與主表（results_table.py）相同。
    2026-11-23 起議題 1 改用 eval_plan_overall，這張不再放進評估章節（累積曲線會誇大逐日差距）。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    import results_table as R  # noqa: E402
    specs = [("MAGNET", R.neural_stats("tw50_rgcnSA")),
             (LASSO, R.nonneural_stats(R.BIP.format("LASSO"))),
             ("MEIG [1]", R.neural_stats("tw50_bl_meig"))]
    fig, ax = plt.subplots(figsize=(10, 5))
    for m, st in specs:
        v = np.nan_to_num(np.asarray(st["daily_ic"], float))
        dates = np.asarray([np.datetime64(str(d)[:10]) for d in st["dates"]])
        c = np.cumsum(v)
        s = STYLE[m]
        ax.plot(dates, c, color=s["color"], ls="-", lw=2, marker=s["marker"], ms=6, mfc=s["color"],
                mew=2, markevery=40, label=f"{m} (mean IC {np.nanmean(st['daily_ic']):.3f})")
        ax.text(dates[-1], c[-1], f"  {c[-1]:.1f}", va="center", color=s["color"], fontsize=11)
    ax.set_xlabel("Test-period trading date (fold 1, 2025)")
    ax.set_ylabel("Cumulative daily IC")
    ax.legend(frameon=False, loc="upper left")
    _finish(fig, ax, out / "eval_plan_cum_ic", "Temporal stability: cumulative daily IC",
            note="Real results. MAGNET: 10-seed mean; MEIG: 3-seed mean.")


def main():
    global FMT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/discussion")
    ap.add_argument("--fmt", choices=["svg", "png"], default="svg", help="png 給簡報用")
    a = ap.parse_args()
    FMT = a.fmt
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (overall, info_source, info_lag, market_state, market_state_signed, stock_linkage, ablation, cum_ic):
        f(out)
    print(f"寫入 {out}（{FMT}）")


if __name__ == "__main__":
    main()
