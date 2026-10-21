"""
holdout_2026_eval.py — 2026 保留期確認：關係型台股層（proposal §75）

登記：reg_holdout_2026.md（計算前寫下並記 SHA-256，全文見 §75）。
凍結模型：第一折已訓練好的主 arm（tw50_betaF1nA2r1）與關係型（tw50_relTW）各 10 顆種子，
用各自的 best.pt 在 CPU 上評估，不重訓、不調任何東西。

  --check   只用既有資料（data/graphs/snapshots_tw50、data/features）重評第一折的 test，
            必須重現每個 run 存檔的 test 預測（最大絕對差 <= 1e-6）。不碰 2026。
  --run     用 data/holdout_2026 的資料評估 2024-12-26 ~ 2026-10-16（一次跑完）：
              1. 健全性：2025-12-30 以前的預測必須與存檔相同（同上門檻），否則停止，不印任何 2026 的數字；
              2. 2026 年的預測存成各 run 的 predictions/holdout2026_predictions.csv；
              3. 印出登記的統計與判定。
  --report  只讀已存的 holdout2026 預測，重印統計與判定（不重新評估）。

登記的統計（2026-01-02 ~ 2026-10-16 的全部快照日）：
  主判準   IC：同種子配對（n = 10），關係型 − 主 arm 的 IC 差 > 0 且配對 t 檢定 p < 0.05（雙尾）
  第二指標 RankIC：同上；與 IC 一起做 Holm（2 個）
  一併報   種子平均的逐日 IC / RankIC 差（逐日 HAC）、10 顆集成、逐種子的值、ICIR / RankICIR

只讀資料與 checkpoint；--run 另寫 predictions/holdout2026_predictions.csv。

用法：
    .venv/bin/python scripts/holdout_2026_eval.py --check
    .venv/bin/python scripts/holdout_2026_eval.py --run
    .venv/bin/python scripts/holdout_2026_eval.py --report
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.dataset.multiplex_dataset import MultiplexDataset, multiplex_collate  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train.evaluator import evaluate  # noqa: E402
from src.train.utils import load_checkpoint  # noqa: E402

ARMS = {"主 arm": "tw50_betaF1nA2r1", "關係型": "tw50_relTW"}
HOLD = ROOT / "data" / "holdout_2026"
FOLD1_TEST = ("2024-12-26", "2025-12-30")
H_START, H_END = "2026-01-01", "2026-10-16"
TOL = 1e-6
OUT_NAME = "holdout2026_predictions.csv"
CPU = torch.device("cpu")


def runs(arm: str) -> dict[int, Path]:
    out = {}
    for d in glob.glob(str(ROOT / "runs" / "**" / f"*{arm}_s*"), recursive=True):
        m = re.search(rf"_{re.escape(arm)}_s(\d+)$", d)
        if m and (Path(d) / "checkpoints" / "best.pt").exists():
            out[int(m.group(1))] = Path(d)
    return dict(sorted(out.items()))


def predict(run_dir: Path, snapshot_dir: Path, features_dir: Path, val_end: int | None) -> pd.DataFrame:
    """用 run 自己的 config 快照建模型、載 best.pt，評估 snapshot_dir 的 test 段（val_end 起）。"""
    cfg = yaml.safe_load(open(run_dir / "config_snapshot.yaml"))
    cfg["data"]["snapshot_dir"] = str(snapshot_dir)
    cfg["data"]["features_dir"] = str(features_dir)
    if val_end is not None:
        cfg["data"]["split"] = {"train_end": val_end, "val_end": val_end}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
        cfg_path = f.name
    try:
        ds = MultiplexDataset(snapshot_dir=str(snapshot_dir), features_dir=str(features_dir),
                              T=cfg["model"]["lstm"]["T_history"], split="test", config_path=cfg_path)
        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, collate_fn=multiplex_collate)
        model = build_model(cfg).to(CPU)
        load_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer=None, map_location=CPU)
        df = evaluate(model, loader, CPU)["predictions"]
    finally:
        Path(cfg_path).unlink(missing_ok=True)
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str)
    df["target_date"] = df["target_date"].astype(str).str[:10]
    return df


def max_diff_vs_stored(run_dir: Path, df: pd.DataFrame, lo: str, hi: str) -> float:
    st = pd.read_csv(run_dir / "predictions" / "test_predictions.csv", dtype={"ticker": str})
    st["target_date"] = st["target_date"].astype(str).str[:10]
    a = df[(df.target_date >= lo) & (df.target_date <= hi)].set_index(["target_date", "ticker"]).y_hat
    b = st[(st.target_date >= lo) & (st.target_date <= hi)].set_index(["target_date", "ticker"]).y_hat
    if len(a) != len(b) or not a.index.sort_values().equals(b.index.sort_values()):
        return float("inf")
    return float((a - b.reindex(a.index)).abs().max())


# ---------------------------------------------------------------- 統計
def daily(df: pd.DataFrame) -> pd.DataFrame:
    """逐日 (IC, RankIC)；std 為 0 或名次相關無定義的日子記 NaN（與主表的處理相同）。"""
    rows = {}
    for d, g in df.groupby("target_date"):
        if g.y.std() < 1e-12 or g.y_hat.std() < 1e-12:
            rows[d] = (np.nan, np.nan)
            continue
        rows[d] = (np.corrcoef(g.y_hat, g.y)[0, 1], stats.spearmanr(g.y_hat, g.y).statistic)
    return pd.DataFrame.from_dict(rows, orient="index", columns=["ic", "ric"]).sort_index()


def hac_p(d) -> float:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    x = d - d.mean()
    var = float(x @ x) / n
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1.0)) * float(x[k:] @ x[:-k]) / n
    return float(2 * stats.t.sf(abs(d.mean() / np.sqrt(max(var, 1e-24) / n)), n - 1))


def icir(s) -> float:
    s = np.asarray(s, float)
    s = s[np.isfinite(s)]
    return float(s.mean() / s.std(ddof=1))


def holm(ps, alpha=0.05):
    order = np.argsort(ps)
    keep = np.zeros(len(ps), bool)
    for k, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - k):
            keep[i] = True
        else:
            break
    return keep


def report(preds: dict[str, dict[int, pd.DataFrame]]) -> None:
    seeds = sorted(set(preds["主 arm"]) & set(preds["關係型"]))
    D = {a: {s: daily(preds[a][s]) for s in seeds} for a in preds}
    days = sorted(set.intersection(*[set(D[a][s].index) for a in D for s in seeds]))
    print(f"\n######## 2026 保留期：{days[0]} ~ {days[-1]}，{len(days)} 個快照日，同種子配對 {len(seeds)} 顆：{seeds}")
    per = {a: {k: np.array([np.nanmean(D[a][s].loc[days, k]) for s in seeds]) for k in ("ic", "ric")} for a in D}
    pv = {}
    for k, lab in (("ic", "IC（主判準）"), ("ric", "RankIC（第二指標）")):
        a, b = per["主 arm"][k], per["關係型"][k]
        d = b - a
        p = stats.ttest_rel(b, a).pvalue
        pv[k] = (d.mean(), p)
        print(f"  {lab:16s} 主 arm {a.mean():+.4f} -> 關係型 {b.mean():+.4f}  差 {d.mean():+.4f}"
              f"（配對 t p {p:.4f}，{int((d > 0).sum())}/{len(d)} 為正）")
    keep = holm(np.array([pv["ic"][1], pv["ric"][1]]))
    print(f"  Holm（IC、RankIC 兩個）：IC {'拒絕' if keep[0] else '不拒絕'}、RankIC {'拒絕' if keep[1] else '不拒絕'}")
    print("  一併報：")
    for k, lab in (("ic", "IC"), ("ric", "RankIC")):
        ma = np.nanmean(np.vstack([D["主 arm"][s].loc[days, k] for s in seeds]), 0)
        mb = np.nanmean(np.vstack([D["關係型"][s].loc[days, k] for s in seeds]), 0)
        dd = mb - ma
        print(f"      種子平均的逐日 {lab} 差 {np.nanmean(dd):+.4f}（逐日 HAC p {hac_p(dd):.4f}）")
    for a in ("主 arm", "關係型"):
        ens = (pd.concat([preds[a][s] for s in seeds]).groupby(["target_date", "ticker"])
               .agg(y_hat=("y_hat", "mean"), y=("y", "first")).reset_index())
        e = daily(ens).loc[days]
        print(f"      {a} 10 顆集成：IC {np.nanmean(e.ic):+.4f}、RankIC {np.nanmean(e.ric):+.4f}")
    for a in ("主 arm", "關係型"):
        ir = [icir(D[a][s].loc[days, "ic"]) for s in seeds]
        rir = [icir(D[a][s].loc[days, "ric"]) for s in seeds]
        print(f"      {a} ICIR {np.mean(ir):.4f}、RankICIR {np.mean(rir):.4f}（逐種子算完再平均）")
    print("      逐種子 IC（主 arm / 關係型）：" + "；".join(
        f"s{s} {per['主 arm']['ic'][i]:+.4f} / {per['關係型']['ic'][i]:+.4f}" for i, s in enumerate(seeds)))
    ic_ok = pv["ic"][0] > 0 and pv["ic"][1] < 0.05
    ric_ok = pv["ric"][0] > 0 and pv["ric"][1] < 0.05
    print(f"\n== 判定：主判準（IC）{'通過' if ic_ok else '未通過'}；第二指標（RankIC）{'通過' if ric_ok else '未通過'}"
          f"；Holm 後兩個都過：{'是' if (keep.all() and ic_ok and ric_ok) else '否'}")


def load_saved() -> dict[str, dict[int, pd.DataFrame]]:
    out = {}
    for lab, arm in ARMS.items():
        out[lab] = {}
        for s, d in runs(arm).items():
            f = d / "predictions" / OUT_NAME
            if f.exists():
                df = pd.read_csv(f, dtype={"ticker": str})
                df["target_date"] = df["target_date"].astype(str).str[:10]
                out[lab][s] = df
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="2026 保留期確認（proposal §75）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(load_saved())
        return
    if a.check:
        worst = 0.0
        for lab, arm in ARMS.items():
            for s, d in runs(arm).items():
                df = predict(d, ROOT / "data/graphs/snapshots_tw50", ROOT / "data/features", None)
                m = max_diff_vs_stored(d, df, *FOLD1_TEST)
                worst = max(worst, m)
                print(f"  {lab} s{s}: 重評第一折 test 與存檔的最大差 {m:.2e}")
        print(f"\n--check：最大差 {worst:.2e}（門檻 {TOL}） -> {'通過' if worst <= TOL else '未通過'}")
        return
    # --run
    snap = HOLD / "graphs" / "snapshots_tw50"
    files = sorted(snap.glob("graph_*.pt"))
    val_end = sum(1 for f in files if f.stem.split("_")[1] < FOLD1_TEST[0])   # 從 2024-12-26 開始評估
    preds, worst = {}, 0.0
    for lab, arm in ARMS.items():
        preds[lab] = {}
        for s, d in runs(arm).items():
            df = predict(d, snap, HOLD / "features", val_end)
            m = max_diff_vs_stored(d, df, *FOLD1_TEST)
            worst = max(worst, m)
            preds[lab][s] = df[(df.target_date >= H_START) & (df.target_date <= H_END)].reset_index(drop=True)
    print(f"健全性：holdout 資料重評第一折 test 與存檔的最大差 {worst:.2e}（門檻 {TOL}）")
    if worst > TOL:
        raise SystemExit("健全性未通過，停止；不印、不存任何 2026 的結果")
    for lab, arm in ARMS.items():
        for s, d in runs(arm).items():
            preds[lab][s].to_csv(d / "predictions" / OUT_NAME, index=False)
    report(preds)


if __name__ == "__main__":
    main()
