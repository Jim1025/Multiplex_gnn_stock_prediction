"""
check_us_scale.py — 教授 11-25 意見的檢驗 2：美股輸入的幅度（第一折）

登記：docs/proposal_1209_revision.md §9 附錄（計算前寫下）。描述性分析，--write 只跑一次。

三部分：
    4.1 MAGNET 的探針   凍結的 tw50_rgcnSA 10 顆，只把 x_seq_L1 的 log_return 欄乘 k（k = 0, 0.5, 1, 2, 4），
                        不改權重、圖與台股輸入。u_k = ŷ_k − ŷ_0；D_k = 50 檔 u_k 的標準差（ddof=0）；
                        G_k = D_k/D_1 對日子取中位數再對種子平均（線性模型 G_k = k）。
    4.2 [24] 的截尾     照 bipartite_screen.py：訓練期（前 1150 天）每檔美股報酬的 0.5 / 99.5 百分位，
                        數每個 test 日被截的檔數。
    4.3 資料清理改寫     data/features/adr 的 log_return 與 data/raw/adr 收盤算出的 log 報酬差 > 1e-6 的美股日，
                        對應到它餵給的台股目標日；Q5 的 MAGNET − LASSO 差，包含與排除這些日子。

用法：
    .venv/bin/python -I scripts/check_us_scale.py --smoke    # 隨機初始化的模型、3 天，不讀 checkpoint 與預測
    .venv/bin/python -I scripts/check_us_scale.py --check    # 只做 k = 1 的重現檢查
    .venv/bin/python -I scripts/check_us_scale.py --write    # 只跑一次；結果檔已存在就拒絕
    .venv/bin/python -I scripts/check_us_scale.py --report   # 之後只讀結果檔重印
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import market_state_eval as MS  # noqa: E402
import results_table as R  # noqa: E402
from src.dataset.config import load_universe  # noqa: E402
from src.dataset.multiplex_dataset import TECH_COLS, MultiplexDataset, multiplex_collate  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train.evaluator import batch_to_device  # noqa: E402
from src.train.utils import load_checkpoint  # noqa: E402

OUT_JSON = ROOT / "docs" / "figures" / "data" / "eval_check_us_scale.json"
ARM = "tw50_rgcnSA"
KS = (0.0, 0.5, 1.0, 2.0, 4.0)
RET = TECH_COLS.index("log_return")
N_Q = 5
BIP_CONFIG = ROOT / "configs" / "tw50.yaml"      # bipartite_screen.py 的預設 --config
EXPECT_Q5_GAP = -0.0589


# ── 4.1 探針 ────────────────────────────────────────────────────────────────
def forward_all(model, loader, k: float) -> tuple[list[str], np.ndarray]:
    dates, out = [], []
    with torch.no_grad():
        for b in loader:
            b = batch_to_device(b, torch.device("cpu"))
            x = b["x_seq_L1"].clone()
            x[..., RET] = x[..., RET] * k
            b["x_seq_L1"] = x
            yh, _ = model(b)
            out.append(yh.numpy().astype(np.float64))
            dates += [str(d)[:10] for d in b["target_date"]]
    return dates, np.concatenate(out)


def probe_seed(run_dir: str | None, cfg: dict, loader, init_only: bool = False):
    """回傳 (日期, {k: ŷ_k [天, 50]})。init_only 只用於冒煙測試（不讀 checkpoint）。"""
    torch.manual_seed(0)
    model = build_model(cfg)
    if not init_only:
        load_checkpoint(Path(run_dir) / "checkpoints" / "best.pt", model, optimizer=None, map_location="cpu")
    model.eval()
    Y, dates = {}, None
    for k in KS:
        d, Y[k] = forward_all(model, loader, k)
        if dates is None:
            dates = d
        elif d != dates:
            raise ValueError("各 k 的日期順序不同")
    return dates, Y


def scale_stats(Y: dict, q: np.ndarray) -> dict:
    u = {k: Y[k] - Y[0.0] for k in KS}
    D = {k: u[k].std(axis=1, ddof=0) for k in KS}
    ok = D[1.0] > 0
    g = {k: np.where(ok, D[k] / np.where(ok, D[1.0], 1.0), np.nan) for k in KS}
    A = np.array([np.corrcoef(Y[1.0][t], u[1.0][t])[0, 1] if ok[t] else np.nan for t in range(len(q))])
    return dict(G={k: float(np.nanmedian(g[k])) for k in KS},
                G2_by_q=[float(np.nanmedian(g[2.0][q == i])) for i in range(N_Q)],
                A_by_q=[float(np.nanmedian(A[q == i])) for i in range(N_Q)],
                D1_by_q=[float(np.median(D[1.0][q == i])) for i in range(N_Q)])


# ── 4.2 [24] 的截尾 ─────────────────────────────────────────────────────────
def bip_clip(test_dates: list[str]) -> np.ndarray:
    """每個 test 日 30 檔美股輸入裡超出 [24] 截尾界線的檔數。"""
    cfg = yaml.safe_load(open(BIP_CONFIG))
    q = float(cfg["baselines"]["bipartite"]["winsor_pct"])
    kw = dict(snapshot_dir=cfg["data"]["snapshot_dir"], features_dir=cfg["data"]["features_dir"],
              T=int(cfg["model"]["lstm"]["T_history"]), config_path=str(BIP_CONFIG))
    ds_all = MultiplexDataset(split="all", **kw)
    n_tr = len(MultiplexDataset(split="train", **kw))
    n_va = len(MultiplexDataset(split="val", **kw))
    if q != 0.5 or n_tr != 1150:
        raise SystemExit(f"[24] 的設定不是登記的值（winsor_pct {q}、n_tr {n_tr}），停止")
    X = np.stack([ds_all[i]["x_seq_L1"][-1, :, RET].numpy().astype(np.float64) for i in range(len(ds_all))])
    d_all = [str(ds_all[i]["target_date"])[:10] for i in range(len(ds_all))]
    lo, hi = np.percentile(X[:n_tr], [q, 100 - q], axis=0)
    te = X[n_tr + n_va:]
    if d_all[n_tr + n_va:] != list(test_dates):
        raise ValueError("[24] 重建的 test 日期與探針不同")
    return ((te < lo) | (te > hi)).sum(axis=1)


# ── 4.3 資料清理改寫 ────────────────────────────────────────────────────────
def rewritten(test_dates: list[str]) -> tuple[np.ndarray, list[list[str]], int]:
    """(每個 test 日被改寫的檔數, 被改寫的代號, 落在 test 用到的美股日裡的補值筆數)。"""
    us = load_universe("tw50").us_nodes
    tgt = pd.to_datetime(test_dates)
    cnt = np.zeros(len(tgt), int)
    names = [[] for _ in tgt]
    n_imp = 0
    for tk in us:
        f = pd.read_csv(ROOT / f"data/features/adr/{tk}.csv", index_col="Date", parse_dates=True).sort_index()
        raw = pd.read_csv(ROOT / f"data/raw/adr/{tk}.csv", index_col="Date", parse_dates=True).sort_index()["Close"]
        prev = pd.Series(f.index[:-1], index=f.index[1:])
        pos = f.index.searchsorted(tgt, side="left") - 1          # 嚴格早於目標日的最後一列
        for i, p in enumerate(pos):
            d = f.index[p]
            if d not in prev.index:
                continue
            p0 = prev[d]
            if d not in raw.index or p0 not in raw.index:
                n_imp += 1
                continue
            lr = float(np.log(raw[d] / raw[p0]))
            if abs(float(f.loc[d, "log_return"]) - lr) > 1e-6:
                cnt[i] += 1
                names[i].append(tk)
    return cnt, names, n_imp


def q5_gap(dates: list[str], affected: np.ndarray, q: np.ndarray) -> dict:
    mg = R.neural_stats(ARM)
    ls = R.nonneural_stats(R.BIP.format("LASSO"))
    dm = [str(x)[:10] for x in mg["dates"]]
    dl = [str(x)[:10] for x in ls["dates"]]
    if dm != list(dates) or dl != list(dates):
        raise ValueError("主表的日期與探針不同")
    diff = np.asarray(mg["daily_ic"], float) - np.asarray(ls["daily_ic"], float)
    m5 = q == N_Q - 1
    inc = float(np.nanmean(diff[m5]))
    exc = float(np.nanmean(diff[m5 & (affected == 0)]))
    if abs(inc - EXPECT_Q5_GAP) >= 5e-5:
        raise SystemExit(f"Q5 的 MAGNET − LASSO {inc:+.4f} 與 §1.2 的 {EXPECT_Q5_GAP:+.4f} 不符，停止")
    n_exc = int((m5 & (affected == 0) & ~np.isnan(diff)).sum())
    flip = np.sign(exc) != np.sign(inc)
    half = abs(exc) < 0.5 * abs(inc)
    return dict(include=inc, exclude=exc, n_include=int((m5 & ~np.isnan(diff)).sum()), n_exclude=n_exc,
                judgement=("Q5 的差距有相當部分落在美股輸入被資料清理改寫的日子" if (flip or half)
                           else "排除這些日子後 Q5 的差距仍在"))


def judge_probe(G: dict, G2q: list) -> dict:
    if G[2.0] < 1.5 and G[4.0] < 2.0:
        s = "MAGNET 的美股路徑會壓縮大幅度的輸入"
    elif G[2.0] >= 1.8 and G[4.0] >= 3.2:
        s = "MAGNET 對美股輸入的反應近似線性（壓縮假設不成立，不能用它解釋 Q5）"
    else:
        s = "部分壓縮"
    return dict(probe=s, by_state=("大行情日壓縮得更多" if G2q[N_Q - 1] < G2q[0] else "大行情日沒有壓縮得更多"))


def report(res: dict) -> None:
    p = res["probe"]
    print(f"4.1 探針（{p['n_seeds']} 顆，{res['n_days']} 天；k=1 重現最大差 {p['repro_maxdiff']:.1e}）")
    print("  G_k（線性 = k）：" + "  ".join(f"k={k}: {v:.3f}" for k, v in p["G"].items())
          + "   逐種子 G_4 範圍 " + f"{min(p['G4_per_seed']):.2f}–{max(p['G4_per_seed']):.2f}")
    print("  G_2 依 |m| 五分位：" + " / ".join(f"{v:.3f}" for v in p["G2_by_q"]))
    print("  A = corr(ŷ_1, u_1) 依五分位：" + " / ".join(f"{v:.3f}" for v in p["A_by_q"]))
    print("  D_1（美股部分的橫截面標準差）依五分位：" + " / ".join(f"{v:.2e}" for v in p["D1_by_q"]))
    print(f"  判讀：{p['judgement']['probe']}；{p['judgement']['by_state']}")
    c = res["bip_clip"]
    print(f"\n4.2 [24] 的截尾：每日被截檔數的平均依五分位 " + " / ".join(f"{v:.2f}" for v in c["mean_by_q"])
          + f"；Q5 被截 ≥ 1 檔 {len(c['q5_days'])} 天")
    for x in c["q5_days"]:
        print(f"    {x['date']}  |m| {x['abs_m']:.2f}%  被截 {x['n']} 檔")
    print(f"  判讀：{c['judgement']}")
    w = res["rewritten"]
    print(f"\n4.3 資料清理改寫：受影響天數依五分位 " + " / ".join(str(v) for v in w["days_by_q"])
          + f"；補值筆數 {w['n_imputed']}")
    for x in w["q5_days"]:
        print(f"    {x['date']}  |m| {x['abs_m']:.2f}%  {x['n']} 檔：{', '.join(x['tickers'])}")
    g = w["q5_gap"]
    print(f"  Q5 的 MAGNET − LASSO：包含 {g['include']:+.4f}（{g['n_include']} 天），排除 {g['exclude']:+.4f}（{g['n_exclude']} 天）")
    print(f"  判讀：{g['judgement']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    gx = ap.add_mutually_exclusive_group(required=True)
    gx.add_argument("--smoke", action="store_true")
    gx.add_argument("--check", action="store_true")
    gx.add_argument("--write", action="store_true")
    gx.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(json.loads(OUT_JSON.read_text()))
        return
    if args.write and OUT_JSON.exists():
        raise SystemExit(f"{OUT_JSON.relative_to(ROOT)} 已存在，--write 只能跑一次；請用 --report")
    torch.set_num_threads(4)

    runs = R.find_seeds(ARM)
    if len(runs) != 10:
        raise SystemExit(f"{ARM} 應有 10 顆，找到 {len(runs)}")
    order = sorted(runs.items(), key=lambda kv: int(kv[0]))
    cfgs = {s: yaml.safe_load(open(Path(d) / "config_snapshot.yaml")) for s, d in order}
    c0 = cfgs[order[0][0]]
    if any(c["data"] != c0["data"] or c["model"] != c0["model"] for c in cfgs.values()):
        raise SystemExit("各種子的資料或模型設定不同，停止")
    cfg_path = str(Path(order[0][1]) / "config_snapshot.yaml")
    ds = MultiplexDataset(snapshot_dir=str(ROOT / c0["data"]["snapshot_dir"]),
                          features_dir=str(ROOT / c0["data"]["features_dir"]),
                          T=c0["model"]["lstm"]["T_history"], split="test", config_path=cfg_path)

    if args.smoke:
        sub = torch.utils.data.Subset(ds, [0, 1, 2])
        ld = DataLoader(sub, batch_size=64, shuffle=False, collate_fn=multiplex_collate, num_workers=0)
        dates, Y = probe_seed(None, c0, ld, init_only=True)
        st = scale_stats(Y, np.array([0, 2, 4]))
        print("冒煙測試（隨機初始化、3 天）：G =", {k: round(v, 3) for k, v in st["G"].items()})
        print("沒有讀 checkpoint 與預測，沒有寫檔")
        return

    ld = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=multiplex_collate, num_workers=0)
    tick = list(ds.tw_codes)
    print("檢查（登記第 4.1 節）：k = 1 的預測與主表的預測檔")
    per_seed, dates, maxdiff = {}, None, 0.0
    for s, d in order:
        dts, Y = probe_seed(d, cfgs[s], ld)
        pred = pd.read_csv(R._pred_file(d), dtype={"ticker": str})
        H = pred.pivot(index="target_date", columns="ticker", values="y_hat").sort_index()
        same = [str(x)[:10] for x in H.index] == dts and list(H.columns) == tick
        md = float(np.abs(H.to_numpy(float) - Y[1.0]).max()) if same else float("inf")
        print(f"  s{s:>4s}  最大差 {md:.1e}{'' if md <= 1e-5 else '  <-- 不符'}")
        if md > 1e-5:
            raise SystemExit("重現檢查未通過，停止（不印探針結果）")
        maxdiff = max(maxdiff, md)
        per_seed[s] = Y
        dates = dts
    print("  通過\n")
    if args.check:
        return

    absm = np.abs(MS.us_move(np.asarray(dates)))
    q = MS.quintile(absm)
    stats_s = {s: scale_stats(Y, q) for s, Y in per_seed.items()}
    G = {k: float(np.mean([st["G"][k] for st in stats_s.values()])) for k in KS}
    G2q = [float(np.mean([st["G2_by_q"][i] for st in stats_s.values()])) for i in range(N_Q)]
    probe = dict(n_seeds=len(per_seed), repro_maxdiff=maxdiff, ks=list(KS),
                 G={str(k): v for k, v in G.items()},
                 G4_per_seed=[st["G"][4.0] for st in stats_s.values()],
                 G2_by_q=G2q,
                 A_by_q=[float(np.mean([st["A_by_q"][i] for st in stats_s.values()])) for i in range(N_Q)],
                 D1_by_q=[float(np.mean([st["D1_by_q"][i] for st in stats_s.values()])) for i in range(N_Q)],
                 judgement=judge_probe(G, G2q))

    clip = bip_clip(dates)
    mean_q = [float(clip[q == i].mean()) for i in range(N_Q)]
    q5c = [dict(date=dates[t], abs_m=float(absm[t] * 100), n=int(clip[t]))
           for t in np.flatnonzero((q == N_Q - 1) & (clip > 0))]
    bip = dict(mean_by_q=mean_q, q5_days=q5c,
               judgement=("[24] 在大行情日基本上仍是線性的" if mean_q[N_Q - 1] < 1
                          else f"[24] 在大行情日也會截掉部分極端輸入（平均 {mean_q[N_Q - 1]:.1f} 檔）"))

    cnt, names, n_imp = rewritten(dates)
    q5w = [dict(date=dates[t], abs_m=float(absm[t] * 100), n=int(cnt[t]), tickers=names[t])
           for t in np.flatnonzero((q == N_Q - 1) & (cnt > 0))]
    rw = dict(days_by_q=[int(((q == i) & (cnt > 0)).sum()) for i in range(N_Q)], n_imputed=n_imp,
              q5_days=q5w, all_days=[dict(date=dates[t], n=int(cnt[t]), tickers=names[t])
                                     for t in np.flatnonzero(cnt > 0)],
              q5_gap=q5_gap(dates, cnt, q))

    res = dict(fold=1, n_days=len(dates), start=dates[0], end=dates[-1], arm=ARM,
               definition=("u_k = yhat(US log_return x k) - yhat(US log_return x 0); D_k = cross-sectional std of u_k; "
                           "G_k = median over days of D_k/D_1, averaged over seeds (linear model: G_k = k)"),
               probe=probe, bip_clip=bip, rewritten=rw)
    OUT_JSON.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    report(res)
    print(f"\n寫入 {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
