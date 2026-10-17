"""
tw_relational_eval.py — 關係型台股層傳遞（proposal §71）的事前登記分析

登記見 proposal §71（reg_tw_relational.md，計算前寫下並記 SHA-256）。本檔在登記之前寫好、
一併記 SHA-256，run 全部完成後才跑。方法與 P10 相同（scripts/tw_lag_channel.py 的區塊 H 與
verdict()），共用其載入與統計函式：

  區塊 A（判準）  同種子配對，關係型 arm 對存檔的主 arm：ICIR（主指標）、IC（護欄），
                  一併報 RankIC、RankICIR、sd(IC)、離散比
  區塊 B（描述）  種子平均的逐日 IC 差（HAC）、10 顆集成、對 P10b、
                  程式漂移檢查（今天的程式重跑主 arm）與關係型 arm 對重跑
  區塊 C（描述）  學到的 W_r 大小（meta.json），以及把訓練好的 W 歸零後的 val / test 指標
                  （關係訊息在訓練好的模型裡的貢獻；CPU 評估）

只讀資料、runs/ 的預測檔與 checkpoint，不寫入任何檔案。

用法：
    .venv/bin/python scripts/tw_relational_eval.py              # 區塊 A B C
    .venv/bin/python scripts/tw_relational_eval.py --blocks A
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
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
import tw_lag_channel as T  # noqa: E402  載入、summ、hac_p、disp、MDE 與 P10 同一份
from src.dataset.multiplex_dataset import MultiplexDataset, multiplex_collate  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train.evaluator import evaluate  # noqa: E402
from src.train.utils import load_checkpoint  # noqa: E402

ARMS = {
    "第一折": dict(rel="runs/**/*tw50_relTW_s*/predictions/{}_predictions.csv",
                  rerun="runs/**/*tw50_rerun_s*/predictions/{}_predictions.csv"),
    "第二折": dict(rel="runs/**/*f2_relTW_s*/predictions/{}_predictions.csv",
                  rerun="runs/**/*f2_rerun_s*/predictions/{}_predictions.csv"),
}
N_SEEDS = 10


def _pair(lab, a, b) -> dict:
    d = b - a
    p = stats.ttest_rel(b, a).pvalue if len(d) > 1 else np.nan
    print(f"      {lab:24s} {a.mean():+.4f} -> {b.mean():+.4f}  差 {d.mean():+.4f}"
          f"（配對 p {p:.4f}，{int((d > 0).sum())}/{len(d)} 為正）")
    return dict(d=float(d.mean()), p=float(p), npos=int((d > 0).sum()), n=len(d))


def _align(A, sa, B, sb):
    common = sorted(set(sa) & set(sb))
    return A[[sa.index(s) for s in common]], B[[sb.index(s) for s in common]], common


def _compare(lab_a, A, sa, lab_b, B, sb, Y) -> dict:
    """同種子配對：A -> B。回傳 ICIR 與 IC 的配對差。"""
    A, B, common = _align(A, sa, B, sb)
    print(f"    [{lab_a} -> {lab_b}] 共同種子 {len(common)} 顆：{common}")
    if len(common) < 2:
        print("      種子不足，跳過")
        return {}
    ra, rb = [T.summ(a, Y) for a in A], [T.summ(b, Y) for b in B]
    out = {}
    for key, lab in (("ICIR", "ICIR"), ("IC", "IC"), ("RIC", "RankIC"),
                     ("RICIR", "RankICIR"), ("sd", "sd(IC)")):
        out[key] = _pair(lab, np.array([r[key] for r in ra]), np.array([r[key] for r in rb]))
    out["disp"] = _pair("離散比 std(ŷ)/std(y)", np.array([T.disp(a, Y) for a in A]),
                        np.array([T.disp(b, Y) for b in B]))
    out["ser"] = (np.nanmean([r["ser"] for r in ra], 0), np.nanmean([r["ser"] for r in rb], 0))
    return out


# ---------------------------------------------------------------- 區塊 A：判準
def block_a(D, M, sm, R, sr) -> dict:
    print(f"[A] 判準：關係型 arm 對主 arm（存檔），找到 {len(sr)} 顆種子")
    if len(sr) != N_SEEDS:
        print(f"    注意：登記的是 {N_SEEDS} 顆種子，實際 {len(sr)} 顆")
    r = _compare("主 arm", M, sm, "關係型", R, sr, D["y_test"])
    if not r:
        return {}
    return dict(dICIR=r["ICIR"]["d"], p=r["ICIR"]["p"], dIC=r["IC"]["d"], n=r["ICIR"]["n"])


def verdict(res: dict) -> None:
    """與 P10 相同：兩折 ICIR 差都 > 0 且各自配對 p < 0.05；兩折 IC 平均差都 >= −MDE。"""
    if len(res) < 2 or any(not v for v in res.values()):
        print("\n== 判定：資料不足，不判定")
        return
    ok_icir = all(v["dICIR"] > 0 and v["p"] < 0.05 for v in res.values())
    ok_ic = all(v["dIC"] >= -T.MDE for v in res.values())
    same_sign = all(v["dICIR"] > 0 for v in res.values())
    print(f"\n== 判定：ICIR 條件 {'通過' if ok_icir else '未通過'}；IC 護欄（>= −{T.MDE}）"
          f"{'通過' if ok_ic else '未通過'} -> {'通過' if ok_icir and ok_ic else '未通過'}"
          f"（兩折 ICIR 差{'同為正' if same_sign else '不同為正'}）")


# ---------------------------------------------------------------- 區塊 B：描述
def block_b(D, f, M, sm, R, sr, arms) -> None:
    Y = D["y_test"]
    print("[B] 描述（不計入判準）")
    A, B, common = _align(M, sm, R, sr)
    if len(common) >= 2:
        sa = np.nanmean([T.summ(a, Y)["ser"] for a in A], 0)
        sb = np.nanmean([T.summ(b, Y)["ser"] for b in B], 0)
        d = sb - sa
        print(f"    種子平均的逐日 IC：關係型 − 主 arm {np.nanmean(d):+.4f}，逐日 HAC p {T.hac_p(d):.4f}")
        ea, eb = T.summ(np.nanmean(A, 0), Y), T.summ(np.nanmean(B, 0), Y)
        de = eb["ser"] - ea["ser"]
        print(f"    {len(common)} 顆集成（先平均預測）：主 arm IC {ea['IC']:+.4f} / ICIR {ea['ICIR']:.4f}"
              f" / RankIC {ea['RIC']:+.4f}；關係型 IC {eb['IC']:+.4f} / ICIR {eb['ICIR']:.4f}"
              f" / RankIC {eb['RIC']:+.4f}；逐日 IC 差 {np.nanmean(de):+.4f}（HAC p {T.hac_p(de):.4f}）")
    P, sp = T.preds(f["p10b"], "test", D["d_test"], D["codes"])
    print("    對 P10b（外掛的凍結通道 ④）：原生 − 外掛，> 0 表示原生較好")
    _compare("P10b", P, sp, "關係型", R, sr, Y)
    Q, sq = T.preds(arms["rerun"], "test", D["d_test"], D["codes"])
    print(f"    程式漂移檢查：今天的程式重跑主 arm（同 config、同種子），找到 {len(sq)} 顆")
    if len(sq):
        _compare("主 arm（存檔）", M, sm, "主 arm（重跑）", Q, sq, Y)
        _compare("主 arm（重跑）", Q, sq, "關係型", R, sr, Y)


# ---------------------------------------------------------------- 區塊 C：學到的 W 與歸零
def _run_dirs(pattern: str) -> list[Path]:
    return sorted({Path(f).parents[1] for f in glob.glob(str(ROOT / pattern.format("test")),
                                                         recursive=True)})


def _pivot(df: pd.DataFrame, dates, codes) -> np.ndarray:
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str)
    df["target_date"] = df["target_date"].astype(str).str[:10]
    H = df.pivot(index="target_date", columns="ticker", values="y_hat")
    return H.reindex(index=dates, columns=codes).to_numpy(float)


def block_c(D, arms, splits=("val", "test")) -> None:
    dirs = _run_dirs(arms["rel"])
    print(f"[C] 學到的 W_r 與歸零（描述），{len(dirs)} 個 run")
    if not dirs:
        return
    metas = [json.load(open(d / "meta.json")).get("tw_relational") or {} for d in dirs]
    names = metas[0].get("relations", [])
    for nm in names:
        v = np.array([m["W_fro"][nm] for m in metas])
        print(f"    ||W_{nm}||_F：平均 {v.mean():.4f}，sd {v.std(ddof=1):.4f}，範圍 {v.min():.4f} ~ {v.max():.4f}")
    cfg0 = yaml.safe_load(open(dirs[0] / "config_snapshot.yaml"))
    snap = str(dirs[0] / "config_snapshot.yaml")
    kw = dict(snapshot_dir=str(ROOT / cfg0["data"]["snapshot_dir"]),
              features_dir=str(ROOT / cfg0["data"]["features_dir"]),
              T=cfg0["model"]["lstm"]["T_history"], config_path=snap)
    L = dict(batch_size=32, shuffle=False, collate_fn=multiplex_collate)
    loaders = {sp: torch.utils.data.DataLoader(MultiplexDataset(split=sp, **kw), **L)
               for sp in splits}
    dev = torch.device("cpu")
    res = {sp: {"on": [], "off": []} for sp in splits}
    max_dev = 0.0
    for d in dirs:
        cfg = yaml.safe_load(open(d / "config_snapshot.yaml"))
        m = build_model(cfg).to(dev)
        load_checkpoint(d / "checkpoints" / "best.pt", m, optimizer=None, map_location=dev)
        for sp in splits:
            Y = D[f"y_{sp}"]
            for key in ("on", "off"):
                if key == "off":
                    with torch.no_grad():
                        m.tw_rel_W.zero_()
                P = _pivot(evaluate(m, loaders[sp], dev)["predictions"], D[f"d_{sp}"], D["codes"])
                res[sp][key].append(T.summ(P, Y))
                if key == "on" and sp == "test":
                    rec = _pivot(pd.read_csv(d / "predictions" / "test_predictions.csv",
                                             dtype={"ticker": str}), D["d_test"], D["codes"])
                    max_dev = max(max_dev, float(np.nanmax(np.abs(P - rec))))
            load_checkpoint(d / "checkpoints" / "best.pt", m, optimizer=None, map_location=dev)
    if "test" in splits:
        print(f"    健全性：重評的 test 預測與存檔的最大差 {max_dev:.2e}")
    for sp in splits:
        print(f"    {sp}：訓練好的模型 -> 只把 W 歸零（其餘權重不動）")
        for k, lab in (("IC", "IC"), ("ICIR", "ICIR")):
            _pair(lab, np.array([r[k] for r in res[sp]["on"]]), np.array([r[k] for r in res[sp]["off"]]))


def main() -> None:
    ap = argparse.ArgumentParser(description="關係型台股層傳遞（proposal §71）的登記分析")
    ap.add_argument("--blocks", nargs="*", default=list("ABC"), choices=list("ABC"))
    args = ap.parse_args()
    res = {}
    for fold, f in T.FOLDS.items():
        arms = ARMS[fold]
        D = T.load_fold(f["cfg"], f["f2"])
        M, sm = T.preds(f["main"], "test", D["d_test"], D["codes"])
        R, sr = T.preds(arms["rel"], "test", D["d_test"], D["codes"])
        print(f"\n######## {fold}：測試窗 {D['d_test'][0]} ~ {D['d_test'][-1]}（{len(D['d_test'])} 天），"
              f"主 arm {len(sm)} 顆、關係型 {len(sr)} 顆")
        if "A" in args.blocks:
            res[fold] = block_a(D, M, sm, R, sr) if len(sr) else {}
        if "B" in args.blocks and len(sr):
            block_b(D, f, M, sm, R, sr, arms)
        if "C" in args.blocks:
            block_c(D, arms)
    if "A" in args.blocks:
        verdict(res)


if __name__ == "__main__":
    main()
