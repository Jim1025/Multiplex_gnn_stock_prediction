"""
tw_relational_posthoc.py — 關係型台股層（proposal §71）的事後分析

**看過登記判定之後才算，不能改變判定**（判定見 scripts/tw_relational_eval.py 與 §71.5）。

  (a) 同種子兩個 arm 的逐種子 ICIR 相關與配對差 sd：回看檢定力落在登記時的哪個情境
  (b) 只把一種關係的 W 歸零（val / test）：哪一種關係在出力
  (c) 訊息的方向：Δŷ = ŷ(完整) − ŷ(該關係的 W = 0) 與該關係鄰居昨日平均報酬的逐日橫截面相關
      （同產業 > 0 表示延續，對照 §62.4、§67.3）
  (d) §68 的準度指標（val 定尺度）：離散比變大之後，校準後的準度有沒有變差。
      evaluate / summ 與 results_table.py 的 _acc_block 逐字相同

只讀資料、runs/ 的預測檔與 checkpoint，不寫入任何檔案。

用法：
    .venv/bin/python scripts/tw_relational_posthoc.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import tw_lag_channel as T  # noqa: E402
import tw_relational_eval as E  # noqa: E402
from src.dataset.features import TECH_FEATURE_COLS  # noqa: E402
from src.dataset.multiplex_dataset import MultiplexDataset, multiplex_collate  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train.evaluator import evaluate as run_eval  # noqa: E402
from src.train.utils import load_checkpoint  # noqa: E402

RET = TECH_FEATURE_COLS.index("log_return")
CPU = torch.device("cpu")


# ---------------------------------------------------------------- (d) 與 _acc_block 逐字相同
def dm(P):
    return P - np.nanmean(P, 1, keepdims=True)


def acc_eval(Hva, Yva, Hte, Yte):
    okv, okt = np.nanstd(Yva, 1) > 1e-12, np.nanstd(Yte, 1) > 1e-12
    hv, yv = dm(Hva[okv]), dm(Yva[okv])
    den = np.nansum(hv * hv)
    b = float(np.nansum(hv * yv) / den) if den > 1e-24 else 0.0
    ht, yt = dm(Hte[okt]), dm(Yte[okt])
    sh = np.sign(Hte[okt] - np.nanmedian(Hte[okt], 1, keepdims=True))
    sy = np.sign(Yte[okt] - np.nanmedian(Yte[okt], 1, keepdims=True))
    m = (sh != 0) & (sy != 0)
    hit = np.array([np.mean(a[k] == c[k]) if k.any() else np.nan for a, c, k in zip(sh, sy, m)])
    return dict(loss=np.nanmean((yt - b * ht) ** 2, 1), base=np.nanmean(yt ** 2, 1), hit=hit, b=b)


def acc_summ(ev):
    return (1 - np.nansum(ev["loss"]) / np.nansum(ev["base"]),
            float(np.sqrt(np.nanmean(ev["loss"])) * 100), float(np.nanmean(ev["hit"]) * 100))


def block_d(D, f, arms) -> None:
    out = {}
    for lab, pat in (("主 arm", f["main"]), ("關係型", arms["rel"])):
        Hv, sv = T.preds(pat, "val", D["d_val"], D["codes"])
        Ht, st = T.preds(pat, "test", D["d_test"], D["codes"])
        assert sv == st, f"{lab} 的 val / test 預測檔對不上"
        out[lab] = (sv, [acc_eval(Hv[i], D["y_val"], Ht[i], D["y_test"]) for i in range(len(sv))])
    print("(d) §68 的準度指標（val 定尺度）")
    for lab, (_, evs) in out.items():
        S = np.array([acc_summ(e) for e in evs])
        print(f"      {lab}：R²_oos {S[:, 0].mean():+.4f} ± {S[:, 0].std(ddof=1):.4f}，RMSE {S[:, 1].mean():.4f}%，"
              f"命中率 {S[:, 2].mean():.2f}%，val 定的倍數 b 平均 {np.mean([e['b'] for e in evs]):.3f}")
    (sa, ea), (sb, eb) = out["主 arm"], out["關係型"]
    common = sorted(set(sa) & set(sb))
    A, B = [ea[sa.index(s)] for s in common], [eb[sb.index(s)] for s in common]
    ra, rb = np.array([acc_summ(e)[0] for e in A]), np.array([acc_summ(e)[0] for e in B])
    ha, hb = np.array([acc_summ(e)[2] for e in A]), np.array([acc_summ(e)[2] for e in B])
    print(f"      關係型 − 主 arm：R²_oos {np.mean(rb - ra):+.4f}（配對 p {stats.ttest_rel(rb, ra).pvalue:.4f}，"
          f"{int((rb > ra).sum())}/{len(common)} 為正）；命中率 {np.mean(hb - ha):+.2f} 百分點"
          f"（配對 p {stats.ttest_rel(hb, ha).pvalue:.4f}，{int((hb > ha).sum())}/{len(common)} 為正）")
    dl = np.mean([e["loss"] for e in B], 0) - np.mean([e["loss"] for e in A], 0)
    dh = np.mean([e["hit"] for e in B], 0) - np.mean([e["hit"] for e in A], 0)
    print(f"      逐日 HAC（種子平均）：MSE 差 {dl.mean():+.3e}（p {T.hac_p(dl):.4f}，負 = 關係型較準）；"
          f"命中率差 p {T.hac_p(dh):.4f}")


# ---------------------------------------------------------------- (a)–(c)
def block_abc(D, f, arms) -> None:
    M, sm = T.preds(f["main"], "test", D["d_test"], D["codes"])
    R, sr = T.preds(arms["rel"], "test", D["d_test"], D["codes"])
    A, B, _ = E._align(M, sm, R, sr)
    a = np.array([T.summ(x, D["y_test"])["ICIR"] for x in A])
    b = np.array([T.summ(x, D["y_test"])["ICIR"] for x in B])
    print(f"(a) corr(主 arm, 關係型 的逐種子 ICIR) {np.corrcoef(a, b)[0, 1]:+.2f}；配對差 sd {np.std(b - a, ddof=1):.4f}")

    dirs = E._run_dirs(arms["rel"])
    snap = dirs[0] / "config_snapshot.yaml"
    cfg0 = yaml.safe_load(open(snap))
    kw = dict(snapshot_dir=str(ROOT / cfg0["data"]["snapshot_dir"]),
              features_dir=str(ROOT / cfg0["data"]["features_dir"]),
              T=cfg0["model"]["lstm"]["T_history"], config_path=str(snap))
    L = dict(batch_size=32, shuffle=False, collate_fn=multiplex_collate)
    ds = {sp: MultiplexDataset(split=sp, **kw) for sp in ("val", "test")}
    lo = {sp: torch.utils.data.DataLoader(ds[sp], **L) for sp in ds}
    rets = {}                    # 昨日報酬：與模型吃的是同一個張量，對齊 D 的日期與代號順序
    for sp in ds:
        rows = {str(ds[sp][i]["target_date"])[:10]: ds[sp][i]["x_seq_L2"][-1, :, RET].numpy()
                for i in range(len(ds[sp]))}
        rets[sp] = np.array([rows[d] for d in D[f"d_{sp}"]])
    res = {sp: {k: [] for k in ("full", "no_same", "no_other")} for sp in ds}
    cont = {sp: {"same": [], "other": []} for sp in ds}
    for d in dirs:
        m = build_model(yaml.safe_load(open(d / "config_snapshot.yaml")))
        names = list(m.tw_rel_names)
        Ms = m.tw_rel_M.numpy()
        for sp in ds:
            P = {}
            for key, rel in (("full", None), ("no_same", "same_industry"), ("no_other", "other_industry")):
                load_checkpoint(d / "checkpoints" / "best.pt", m, optimizer=None, map_location=CPU)
                if rel is not None:
                    with torch.no_grad():
                        m.tw_rel_W[names.index(rel)].zero_()
                P[key] = E._pivot(run_eval(m, lo[sp], CPU)["predictions"], D[f"d_{sp}"], D["codes"])
                res[sp][key].append(T.summ(P[key], D[f"y_{sp}"]))
            for lab, key, rel in (("same", "no_same", "same_industry"), ("other", "no_other", "other_industry")):
                r = names.index(rel)
                dlt = P["full"] - P[key]
                nb = rets[sp] @ Ms[r].T                       # 每檔的鄰居平均昨日報酬
                has = Ms[r].sum(1) > 0                        # 沒有同業的 9 檔不計
                cs = [np.corrcoef(dlt[t, has], nb[t, has])[0, 1] for t in range(len(dlt))
                      if np.std(dlt[t, has]) > 0 and np.std(nb[t, has]) > 0]
                cont[sp][lab].append(np.mean(cs))
    for sp in ds:
        full = np.array([x["IC"] for x in res[sp]["full"]])
        print(f"(b) {sp}：只把一種關係的 W 歸零（對完整模型配對）")
        for key, lab in (("no_same", "W_同產業 = 0"), ("no_other", "W_其他產業 = 0")):
            v = np.array([x["IC"] for x in res[sp][key]])
            print(f"      {lab:14s} IC {full.mean():+.4f} -> {v.mean():+.4f}  差 {np.mean(v - full):+.4f}"
                  f"（配對 p {stats.ttest_rel(v, full).pvalue:.4f}，{int((v < full).sum())}/{len(v)} 下降）")
        for lab, nm in (("same", "同產業"), ("other", "其他產業")):
            c = np.array(cont[sp][lab])
            print(f"(c) {sp}：corr(Δŷ, {nm}鄰居昨日平均報酬) 平均 {c.mean():+.3f}（範圍 {c.min():+.3f} ~ {c.max():+.3f}）")


def main() -> None:
    for fold, f in T.FOLDS.items():
        arms = E.ARMS[fold]
        D = T.load_fold(f["cfg"], f["f2"])
        print(f"\n######## {fold}")
        block_abc(D, f, arms)
        block_d(D, f, arms)


if __name__ == "__main__":
    main()
