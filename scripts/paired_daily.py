"""
paired_daily.py — 用逐日配對檢定重新判讀既有的 tw50 對照組

為什麼需要這支腳本
──────────────────────────────────────────────────────────────
本專案先前的架構結論，幾乎都是「跑 3 顆種子、比 test IC 的平均」。
實測 sigma_seed = 0.0062（兩組各 10 顆種子獨立量到同一個值），代入
雙樣本 t 檢定的檢定力公式：

    n= 3/組  最小可偵測差異 MDE = 0.0188
    n=10/組                      0.0082

而整個 MAGNET 家族的 test IC 落在 +0.001 ~ +0.034，全距只有 0.033。
換句話說：家族內幾乎每一組兩兩比較，效果量都遠小於 n=3 能偵測的門檻。
過去說「某個機制沒用」的那些結論，多數不是「證實無效」，是「測不出」。

具體例子：把 7 檔配對台股的預測換成目前已知最好的方法（KTW+），
全體 IC 增量只有 +0.0078 —— 低於 MDE 0.0188，跨種子平均永遠看不到；
但同一組預測用逐日配對檢定是 t=+4.21, p<0.0001。

本腳本同時報兩個檢定，因為它們回答的是不同問題
──────────────────────────────────────────────────────────────
  paired-daily  以「日」為重複單位，種子視為固定。
                先把同一個 arm 各種子的逐日 IC 平均成一條序列（消掉種子
                噪音），再對兩條序列做 246 天的配對 t。
                回答：「這組已訓練的模型，在這段測試期上是否較佳」。
                敏感度高約一個數量級，但不外推到新的種子。

  across-seed   以「種子」為重複單位，日已被平均掉。
                對每個 arm 的 per-seed 平均 IC 做 Welch t。
                回答：「換一顆新種子，這個設定是否較佳」。
                這才是架構宣稱該用的檢定，但 n=3 時 MDE = 0.0188。

兩個都報，並且都不要單獨解讀：paired-daily 顯著而 across-seed 不顯著，
正確的說法是「效果在這段期間確實存在，但小於種子變異，尚無法宣稱
對任意種子成立」——這正是本專案多數架構改動的真實狀態。

其他處理
  - Newey-West HAC：逐日 IC 序列有自相關，配對序列的 t 用 Bartlett 核
    修正（lag 自動取 floor(4*(n/100)^(2/9))），另報未修正的 t 供對照。
  - 只用兩個 arm 的種子交集，避免 10 種子 arm 與 3 種子 arm 不對等。
  - 家族內用 Holm 校正；跨家族不校正（家族是事先指定的，不是掃出來的）。

用法
    .venv/bin/python scripts/paired_daily.py
    .venv/bin/python scripts/paired_daily.py --family 跨市場機制
    .venv/bin/python scripts/paired_daily.py --base tw50chk_wl0 --vs tw50_L1_wl0 tw50_T1F9
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 事先指定的比較家族。key = 家族名，value = (基準 arm, [對照 arm...])
# 事先指定是為了讓 Holm 校正有意義——若改成掃所有兩兩組合，
# 家族大小會膨脹到 465，任何真實效果都會被校正吃光。
FAMILIES: dict[str, tuple[str, list[str]]] = {
    "跨市場機制": ("tw50chk_wl0", [
        "tw50chk_wfree",     # 候選邊 lambda=1e-3（已量到只送出 0.08% 訊號）
        "tw50chk_imed",      # intermediate fusion，但 weak_mode=None
        "tw50_imedwl0",      # intermediate + 候選邊 lambda=0
        "fx_dense",          # 稠密 EF 對照組
    ]),
    "initial residual": ("tw50_imedwl0", [
        "tw50_imedres25", "tw50_imedres5", "tw50_imedres75", "tw50_imedres100",
    ]),
    "GAT 深度": ("tw50chk_wl0", ["tw50_L1_wl0"]),
    "L=1 下的融合位置": ("tw50_L1_wl0", ["tw50_L1_imed0", "tw50_L1_imed5"]),
    "輸入維度 T/F": ("tw50_L1_wl0", [
        "tw50_T1F9", "tw50_T20F1", "tw50_T5F9", "tw50_T5F1", "tw50_T1F1",
    ]),
    "損失函數": ("tw50chk_wl0", ["fx_rank"]),
    "文獻 baseline": ("tw50chk_wl0", [
        "tw50_bl_hgt", "tw50_bl_delta_lag", "tw50_bl_meig",
        "tw50_bl_adv_alstm", "tw50_bl_hats", "tw50_bl_man_sf",
        "tw50chk_ef", "tw50chk_lstm",
    ]),
}

SIGMA_SEED = 0.0062      # 兩組 10 顆種子各自量到的 test IC 標準差


def discover(universe: str = "tw50") -> dict[str, dict[str, str]]:
    """掃 runs/，回傳 {arm: {seed: run_dir}}，只收有預測檔的 run。"""
    arms: dict[str, dict[str, str]] = defaultdict(dict)
    for d in sorted(glob.glob(str(ROOT / "runs" / "*"))):
        cp = os.path.join(d, "config_snapshot.yaml")
        if not os.path.exists(cp):
            continue
        try:
            c = yaml.safe_load(open(cp))
        except Exception:
            continue
        if (c.get("data") or {}).get("universe") != universe:
            continue
        # reeval 模式下，沒有 checkpoint 的 run（ridge / bipartite 等非神經
        # 基準）本來就沒有重評對象——它們是 sklearn 在 CPU 上算的，
        # 記錄的 CSV 就是權威值。這種 run 回退到 test_predictions.csv，
        # 並在下方明列，不做靜默回退。
        if (PRED_FILE != "test_predictions.csv"
                and not os.path.exists(os.path.join(d, "predictions", PRED_FILE))
                and not os.path.exists(os.path.join(d, "checkpoints", "best.pt"))
                and os.path.exists(os.path.join(d, "predictions", "test_predictions.csv"))):
            _FALLBACK.append(os.path.basename(d))
            _PRED_OVERRIDE[d] = "test_predictions.csv"
        if not os.path.exists(os.path.join(d, "predictions",
                                           _PRED_OVERRIDE.get(d, PRED_FILE))):
            continue
        name = re.sub(r"^\d{8}_\d{4}_", "", os.path.basename(d))
        m = re.match(r"^(.*)_s(\d+)$", name)
        arm, seed = (m.group(1), m.group(2)) if m else (name, "NA")
        arms[arm][seed] = d
    return arms


# 讀哪一份 predictions。預設是 run 當下寫的 test_predictions.csv；
# 2026-08-16 之前的 run 那份是在 MPS 上算的、不可重現（見
# scripts/reeval_checkpoints.py），要用 --predictions reeval 換成
# CPU 重評版本 test_predictions_reeval.csv。
PRED_FILE = "test_predictions.csv"
_PRED_OVERRIDE: dict[str, str] = {}
_FALLBACK: list[str] = []


def daily_series(run_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """回傳 (dates, IC 序列, RankIC 序列)。全數為 NaN 的日子保留，配對時再遮。"""
    df = pd.read_csv(os.path.join(run_dir, "predictions",
                                  _PRED_OVERRIDE.get(run_dir, PRED_FILE)))
    H = df.pivot(index="target_date", columns="ticker", values="y_hat").sort_index()
    Y = df.pivot(index="target_date", columns="ticker", values="y").sort_index()
    Hn, Yn = H.to_numpy(), Y.to_numpy()
    ic = np.full(len(Yn), np.nan)
    ric = np.full(len(Yn), np.nan)
    for t in range(len(Yn)):
        a, b = Hn[t], Yn[t]
        if np.std(a) == 0 or np.std(b) == 0 or len(a) < 3:
            continue          # 例如全 50 檔 y 皆為 0 的日子（已知 test 有 1 天）
        ic[t] = np.corrcoef(a, b)[0, 1]
        # 已知 test 期有 1 天全部 50 檔 y 為 0；np.std 那關擋掉的是 y，
        # 但預測若剛好常數，spearmanr 會發 ConstantInputWarning 並回 NaN。
        # 這是資料狀況不是錯誤，明確吞掉警告以免蓋住真正的輸出。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", stats.ConstantInputWarning)
            v = stats.spearmanr(a, b).statistic
        ric[t] = np.nan if np.isnan(v) else v
    return H.index.to_numpy(), ic, ric


def arm_series(arm: str, runs: dict[str, str], seeds: list[str]) -> dict:
    """把指定種子的逐日序列平均成一條，並回傳 per-seed 的平均 IC。"""
    dates_ref = None
    ics, rics, per_seed = [], [], {}
    for s in seeds:
        d, ic, ric = daily_series(runs[s])
        if dates_ref is None:
            dates_ref = d
        elif len(d) != len(dates_ref) or not (d == dates_ref).all():
            raise ValueError(f"{arm} seed={s} 的測試日期與其他種子不一致")
        ics.append(ic)
        rics.append(ric)
        per_seed[s] = (np.nanmean(ic), np.nanmean(ric))
    # 全種子皆為 NaN 的日子照樣留成 NaN（test 有 1 天 50 檔 y 全為 0，
    # 該日任何模型的 IC 都無定義）。np.nanmean 對整欄 NaN 會發 RuntimeWarning，
    # 這裡先遮起來自己填 NaN，避免把已知的資料狀況印成一堆警告。
    def _colmean(stack: np.ndarray) -> np.ndarray:
        out = np.full(stack.shape[1], np.nan)
        ok = ~np.isnan(stack).all(axis=0)
        out[ok] = np.nanmean(stack[:, ok], axis=0)
        return out

    return {
        "dates": dates_ref,
        "ic": _colmean(np.vstack(ics)),                # 種子平均的逐日序列
        "ric": _colmean(np.vstack(rics)),
        "per_seed": per_seed,
        "seeds": seeds,
    }


def nw_tstat(d: np.ndarray) -> tuple[float, float, int]:
    """配對差序列的 Newey-West HAC t 值。回傳 (t, p, lag)。"""
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 3:
        return np.nan, np.nan, 0
    lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    x = d - d.mean()
    g0 = float((x @ x) / n)
    var = g0
    for k in range(1, lag + 1):
        gk = float((x[k:] @ x[:-k]) / n)
        var += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    var = max(var, 1e-24)
    se = np.sqrt(var / n)
    t = d.mean() / se
    return float(t), float(2 * stats.t.sf(abs(t), n - 1)), lag


def compare(A: dict, B: dict) -> dict:
    """A 為基準；正的 d 代表 B 較差（A - B）。"""
    out = {}
    for key, label in (("ic", "IC"), ("ric", "RankIC")):
        a, b = A[key], B[key]
        ok = ~(np.isnan(a) | np.isnan(b))
        d = a[ok] - b[ok]
        t_pd, p_pd = stats.ttest_rel(a[ok], b[ok])
        t_nw, p_nw, lag = nw_tstat(d)
        # across-seed：per-seed 平均值的 Welch
        i = 0 if key == "ic" else 1
        va = np.array([v[i] for v in A["per_seed"].values()])
        vb = np.array([v[i] for v in B["per_seed"].values()])
        t_as, p_as = stats.ttest_ind(va, vb, equal_var=False)
        # 逐日序列的相關係數決定配對檢定能消掉多少噪音：
        # sd(A-B)^2 = sd(A)^2 + sd(B)^2 - 2*r*sd(A)*sd(B)。
        # r 接近 1 時配對極敏感（例如只換了 50 檔中的 7 檔預測）；
        # r 接近 0 時配對幾乎沒有好處，sd(d) 反而是單一序列的 sqrt(2) 倍。
        r_day = float(np.corrcoef(a[ok], b[ok])[0, 1])
        n_d = int(ok.sum())
        tc, tb = stats.t.ppf(0.975, n_d - 1), stats.t.ppf(0.80, n_d - 1)
        mde_pd = float((tc + tb) * d.std(ddof=1) / np.sqrt(n_d))
        out[label] = dict(
            mean_A=float(np.nanmean(a)), mean_B=float(np.nanmean(b)),
            d=float(d.mean()), n_days=n_d, r_day=r_day, mde_pd=mde_pd,
            t_pd=float(t_pd), p_pd=float(p_pd),
            t_nw=t_nw, p_nw=p_nw, lag=lag,
            t_as=float(t_as), p_as=float(p_as),
        )
    return out


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni 校正後的 p（保序）。"""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def mde(n: int, sd: float = SIGMA_SEED) -> float:
    """雙樣本、每組 n、80% 檢定力、alpha=0.05 的最小可偵測差異。"""
    if n < 2:
        return float("inf")
    tc = stats.t.ppf(0.975, 2 * n - 2)
    tb = stats.t.ppf(0.80, 2 * n - 2)
    return float((tc + tb) * sd * np.sqrt(2.0 / n))


def run_family(name: str, base: str, members: list[str],
               arms: dict[str, dict[str, str]], metric: str) -> None:
    if base not in arms:
        print(f"\n[{name}] 基準 {base} 沒有預測檔，跳過")
        return
    rows, ps = [], []
    for mem in members:
        if mem not in arms:
            print(f"[{name}] 找不到 {mem}，跳過")
            continue
        seeds = sorted(set(arms[base]) & set(arms[mem]))
        if not seeds:
            print(f"[{name}] {base} 與 {mem} 沒有共同種子，跳過")
            continue
        A = arm_series(base, arms[base], seeds)
        B = arm_series(mem, arms[mem], seeds)
        r = compare(A, B)[metric]
        r.update(arm=mem, n_seeds=len(seeds), seeds=seeds)
        rows.append(r)
        ps.append(r["p_pd"])
    if not rows:
        return
    # 兩欄都要校正：家族內同時看了逐日與跨種子兩個檢定，
    # 只校正其中一欄等於留了一條沒補的多重比較漏洞。
    adj = holm(ps)
    adj_as = holm([r["p_as"] for r in rows])

    print(f"\n{'='*104}")
    print(f"家族「{name}」  基準 = {base}  指標 = {metric}")
    print(f"{'='*104}")
    print(f"{'對照 arm':20s} {'sd':>3s} {'基準':>8s} {'對照':>8s} {'差':>8s} {'r_day':>6s} "
          f"| {'配對t':>6s} {'HACp':>7s} {'Holm':>7s} {'MDE':>7s} "
          f"| {'種子t':>6s} {'p':>7s} {'Holm':>7s} {'MDE':>7s} | 判定")
    n_unres = 0
    for r, pa, pb in zip(rows, adj, adj_as):
        m_as = mde(r["n_seeds"])
        # 判定：任一檢定顯著才算有結論；兩者都不顯著且效果量低於兩個 MDE = 無結論
        # 判定一律用 Holm 校正後的 p：家族內做了多次比較，未校正的 p 會高估證據。
        sig_pd = pa < 0.05
        sig_as = pb < 0.05
        if sig_pd or sig_as:
            verdict = ("兩者皆顯著" if (sig_pd and sig_as)
                       else ("僅逐日顯著" if sig_pd else "僅跨種子顯著"))
        elif abs(r["d"]) < min(r["mde_pd"], m_as):
            verdict = "無結論"; n_unres += 1
        else:
            verdict = "不顯著"
        print(f"{r['arm']:20s} {r['n_seeds']:3d} {r['mean_A']:+8.4f} {r['mean_B']:+8.4f} "
              f"{r['d']:+8.4f} {r['r_day']:+6.2f} "
              f"| {r['t_pd']:+6.2f} {r['p_nw']:7.4f} {pa:7.4f} {r['mde_pd']:7.4f} "
              f"| {r['t_as']:+6.2f} {r['p_as']:7.4f} {pb:7.4f} {m_as:7.4f} | {verdict}")
    if n_unres:
        print(f"\n  無結論（效果量同時低於兩個 MDE）：{n_unres}/{len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="逐日配對檢定重新判讀既有對照組")
    ap.add_argument("--universe", default="tw50")
    ap.add_argument("--metric", default="both", choices=["IC", "RankIC", "both"])
    ap.add_argument("--family", nargs="*", default=None,
                    help="只跑指定家族，預設全部")
    ap.add_argument("--base", default=None, help="臨時指定基準 arm（搭配 --vs）")
    ap.add_argument("--vs", nargs="*", default=None, help="臨時指定對照 arm")
    ap.add_argument("--list", action="store_true", help="列出可用的 arm 後結束")
    ap.add_argument("--predictions", choices=["recorded", "reeval"], default="recorded",
                    help="recorded=run 當下寫的 CSV（MPS，不可重現）；"
                         "reeval=CPU 重評版本，由 scripts/reeval_checkpoints.py 產生")
    args = ap.parse_args()

    global PRED_FILE
    if args.predictions == "reeval":
        PRED_FILE = "test_predictions_reeval.csv"
    print(f"[paired_daily] predictions = {PRED_FILE}")

    arms = discover(args.universe)
    if _FALLBACK:
        print(f"[paired_daily] 下列 {len(_FALLBACK)} 個 run 沒有 checkpoint"
              f"（非神經基準，sklearn/CPU 產生，記錄值即權威值），"
              f"回退讀 test_predictions.csv：")
        for name in sorted(_FALLBACK):
            print(f"    {name}")
    if args.list:
        print(f"universe={args.universe} 有預測檔的 arm：")
        for a in sorted(arms):
            print(f"  {a:24s} seeds={sorted(arms[a])}")
        return

    metrics = ["IC", "RankIC"] if args.metric == "both" else [args.metric]
    if args.base:
        fams = {"臨時比較": (args.base, args.vs or sorted(set(arms) - {args.base}))}
    else:
        fams = FAMILIES if args.family is None else {
            k: v for k, v in FAMILIES.items() if k in args.family}

    print(f"universe={args.universe}  找到 {len(arms)} 個 arm")
    print(f"檢定力參考：n=3/組 MDE={mde(3):.4f}   n=5 {mde(5):.4f}   n=10 {mde(10):.4f}")
    for metric in metrics:
        for name, (base, members) in fams.items():
            run_family(name, base, members, arms, metric)

    print(f"\n{'='*104}")
    print("讀法提醒")
    print("  r_day  ：兩個 arm 逐日 IC 序列的相關係數，決定配對能消掉多少噪音。")
    print("           sd(A-B)^2 = sd(A)^2 + sd(B)^2 - 2*r*sd(A)*sd(B)。")
    print("           r 高（同一模型只改少數節點）-> 配對極敏感；")
    print("           r 低（兩個獨立訓練的架構）  -> 配對幾乎沒好處，sd(d) 反而更大。")
    print("           本專案的架構對照多屬後者，所以逐日配對並不是萬靈丹。")
    print("  MDE    ：該檢定在 80% 檢定力、alpha=0.05 下的最小可偵測差異。")
    print("           跨種子用 sigma_seed=%.4f；逐日用該對照實際的 sd(A-B)。" % SIGMA_SEED)
    print("  Holm   ：家族內多重比較校正，逐日與跨種子各自校正一次。")
    print("           判定一律以 Holm 後的 p 為準；未校正的 p 只作參考。")
    print("  判定   ：兩個檢定都不顯著、且效果量同時低於兩個 MDE = 「無結論」，")
    print("           這與「證實無效」是不同的宣稱，論文中不可互換。")
    print()
    print("  重要限制：n=3 時跨種子的 p 值完全依賴常態假設（Welch 只有約 2 個自由度）。")
    print("           無母數的 Mann-Whitney 在 3 vs 3 的最小雙尾 p 是 0.10，")
    print("           也就是說 3 顆種子在無母數下永遠無法達到 p<0.05。")
    print("           因此本表的跨種子 p<0.05 應視為提示，不足以單獨支撐架構宣稱；")
    print("           要下結論請把種子數提到 10（MDE 從 0.0188 降到 0.0082）。")


if __name__ == "__main__":
    main()
