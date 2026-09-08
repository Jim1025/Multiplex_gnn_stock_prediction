"""
beta_industry_clustering.py — ③ 的可解釋性證據（proposal §52 的 P1）

B[i, j] 是「美股 i -> 台股 j」的跨層權重。每一**欄**是一檔台股在 30 檔美股上的
載重向量。問題：在完全沒有給模型任何產業標籤的情況下，同產業的台股是否
自己拿到相似的載重？

量法（全部逐 seed 做，再跨 seed 彙總）：
  1. 取訓練後的 B_eff = weak_beta * weak_mask，形狀 [30, 50]
  2. 每一欄先去均值再單位化——去均值是必要的，因為欄和落在 ③ 的零空間裡
     （加常數不改變輸出，§33.2a），不去掉它會把一個無作用的自由度算進相似度
  3. 算 50x50 的欄間餘弦
  4. **分離度 = mean(同產業配對) − mean(異產業配對)**

用法：
    .venv/bin/python scripts/beta_industry_clustering.py
    .venv/bin/python scripts/beta_industry_clustering.py --arm tw50_beta
    .venv/bin/python scripts/beta_industry_clustering.py --no-fig
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.config import load_universe   # noqa: E402

ARM_DEFAULT = "tw50_betaF1nA2r1"


def find_seeds(arm: str) -> dict[str, str]:
    out = {}
    for d in sorted(glob.glob(str(ROOT / "runs" / "**" / f"*{arm}_s*"), recursive=True)):
        m = re.fullmatch(rf"\d{{8}}_\d{{4}}_{re.escape(arm)}_s(\d+)", os.path.basename(d))
        if m and os.path.exists(os.path.join(d, "checkpoints", "best.pt")):
            out[m.group(1)] = d
    return out


def load_B(run_dir: str) -> np.ndarray | None:
    """回傳 B_eff = weak_beta * weak_mask，[n1, n2]。找不到權重時回 None。"""
    sd = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"),
                    map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    if "weak_beta" in sd:
        B = sd["weak_beta"].numpy()
    elif "weak_U" in sd and "weak_V" in sd:          # 低秩 B = U Vᵀ
        B = (sd["weak_U"] @ sd["weak_V"].t()).numpy()
    else:
        return None
    mask = sd.get("weak_mask")
    return B * mask.numpy() if mask is not None else B


def separation(B: np.ndarray, ind: list[str]) -> tuple[float, float, float, np.ndarray]:
    """回傳 (分離度, 同產業平均, 異產業平均, 50x50 餘弦矩陣)。"""
    C = B - B.mean(axis=0, keepdims=True)            # 去掉零空間的欄和成分
    n = np.linalg.norm(C, axis=0, keepdims=True)
    C = C / np.maximum(n, 1e-12)
    cos = C.T @ C
    same, diff = [], []
    for a, b in itertools.combinations(range(len(ind)), 2):
        (same if ind[a] == ind[b] else diff).append(cos[a, b])
    s, d = float(np.mean(same)), float(np.mean(diff))
    return s - d, s, d, cos


def main() -> None:
    ap = argparse.ArgumentParser(description="B 的欄位是否自己分出產業")
    ap.add_argument("--arm", default=ARM_DEFAULT)
    ap.add_argument("--universe", default="tw50")
    ap.add_argument("--no-fig", action="store_true")
    ap.add_argument("--n-perm", type=int, default=1000,
                    help="排列檢定的洗牌次數")
    ap.add_argument("--out", default="docs/figures/beta_industry_clustering.svg")
    a = ap.parse_args()

    u = load_universe(a.universe)
    tw = list(u.tw_nodes)
    ind = [u.industry.get(c, "其他") for c in tw]

    runs = find_seeds(a.arm)
    if not runs:
        print(f"找不到 {a.arm} 的 run")
        return

    seps, sames, diffs, mats, Bs = [], [], [], [], []
    for s, d in sorted(runs.items(), key=lambda kv: int(kv[0])):
        B = load_B(d)
        if B is None:
            continue
        sep, sm, df, cos = separation(B, ind)
        seps.append(sep); sames.append(sm); diffs.append(df); mats.append(cos)
        Bs.append(B)
        print(f"  seed {s:>4}   分離度 {sep:+.4f}   同產業 {sm:+.4f}   異產業 {df:+.4f}")

    if not seps:
        print("沒有可用的 weak_beta")
        return

    seps = np.array(seps)
    print(f"\narm = {a.arm}   n = {len(seps)} 顆種子")
    print(f"  分離度  平均 {seps.mean():+.4f}   sd {seps.std(ddof=1):.4f}   "
          f"範圍 [{seps.min():+.4f}, {seps.max():+.4f}]")
    print(f"  為正的種子數  {int((seps > 0).sum())}/{len(seps)}")
    print(f"  同產業平均 {np.mean(sames):+.4f}   異產業平均 {np.mean(diffs):+.4f}")

    # 跨 seed 平均的餘弦矩陣：最近鄰
    M = np.mean(mats, axis=0)
    np.fill_diagonal(M, -np.inf)
    print(f"\n最近鄰（跨 {len(mats)} 顆種子平均的欄間餘弦）")
    print(f"  {'台股':>18}{'產業':>8}   最近的三檔")
    for j in np.argsort(-M.max(axis=1))[:12]:
        top = np.argsort(-M[j])[:3]
        nb = "  ".join(f"{tw[k]}({ind[k]}) {M[j, k]:+.2f}" for k in top)
        print(f"  {tw[j]:>18}{ind[j]:>8}   {nb}")

    # 指名個股的最近鄰（供文件/投影片引用時可查證）
    named = ["2330", "2303", "2412", "2881", "2882", "1301", "2308"]
    have = [c for c in named if c in tw]
    if have:
        print(f"\n指名個股的最近鄰")
        for c in have:
            j = tw.index(c)
            top = np.argsort(-M[j])[:3]
            nb = "  ".join(f"{tw[k]}({ind[k]}) {M[j, k]:+.2f}" for k in top)
            print(f"  {c}({ind[j]})".ljust(24) + nb)

    # 排列檢定：把產業標籤洗牌，看分離度的虛無分布。
    # 沒有這一格，+0.30 只是一個描述性數字——50 檔股票、18 個產業，
    # 光靠隨機分組也會有非零的分離度。
    rng = np.random.default_rng(42)
    obs = seps.mean()
    null = []
    for _ in range(a.n_perm):
        sh = list(ind)
        rng.shuffle(sh)
        null.append(np.mean([separation(B, sh)[0] for B in Bs]))
    null = np.array(null)
    pval = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    print(f"\n排列檢定（洗牌產業標籤 {a.n_perm} 次）")
    print(f"  觀測分離度 {obs:+.4f}")
    print(f"  虛無分布   平均 {null.mean():+.4f}   sd {null.std(ddof=1):.4f}   "
          f"95 百分位 {np.percentile(null, 95):+.4f}")
    print(f"  p = {pval:.4f}"
          + ("   （已達檢定的下限，實際更小）" if pval <= 1.0 / (len(null) + 1) else ""))

    if not a.no_fig:
        _figure(M, tw, ind, ROOT / a.out, a.arm, len(mats))


def _figure(M: np.ndarray, tw: list[str], ind: list[str], out: Path,
            arm: str, n_seed: int) -> None:
    """欄間餘弦熱圖，列/欄依產業排序。分塊結構即為證據。"""
    # 先粗分（電子 / 金融 / 其餘）再依細產業排。純中文字序會把電子類打散
    # （光電業、其他電子業、半導體業… 不相鄰），分塊結構就看不出來。
    ELEC = {"半導體業", "電腦及週邊設備業", "電子零組件業", "光電業",
            "通信網路業", "其他電子業", "電子通路業", "資訊服務業"}
    grp = {c: (0 if c in ELEC else (1 if c == "金融保險業" else 2)) for c in set(ind)}
    order = sorted(range(len(tw)), key=lambda j: (grp[ind[j]], ind[j], tw[j]))
    Z = M[np.ix_(order, order)].copy()
    np.fill_diagonal(Z, 1.0)
    v = float(np.nanmax(np.abs(Z[~np.eye(len(Z), dtype=bool)])))
    cell, pad = 13, 132
    W = H = pad + cell * len(tw) + 26
    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="Helvetica Neue, Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    for r in range(len(order)):
        for c in range(len(order)):
            x = Z[r, c] / max(v, 1e-9)
            x = max(-1.0, min(1.0, x))
            col = (f"rgb(255,{int(255-100*x)},{int(255-140*x)})" if x > 0
                   else f"rgb({int(255+140*x)},{int(255+120*x)},255)")
            p.append(f'<rect x="{pad+c*cell}" y="{pad+r*cell}" width="{cell}" '
                     f'height="{cell}" fill="{col}"/>')
    # 產業分界線
    bounds = [i for i in range(1, len(order)) if ind[order[i]] != ind[order[i-1]]]
    for b in bounds:
        q = pad + b * cell
        p.append(f'<line x1="{pad}" y1="{q}" x2="{pad+cell*len(order)}" y2="{q}" '
                 f'stroke="#1f2933" stroke-width="1.4"/>')
        p.append(f'<line x1="{q}" y1="{pad}" x2="{q}" y2="{pad+cell*len(order)}" '
                 f'stroke="#1f2933" stroke-width="1.4"/>')
    # 產業標籤
    segs, st = [], 0
    for b in bounds + [len(order)]:
        segs.append((ind[order[st]], st, b)); st = b
    for name, s0, s1 in segs:
        mid = pad + (s0 + s1) / 2 * cell
        p.append(f'<text x="{pad-8}" y="{mid+4}" font-size="11" fill="#1f2933" '
                 f'text-anchor="end">{name}</text>')
        p.append(f'<text transform="translate({mid},{pad-8}) rotate(-90)" '
                 f'font-size="11" fill="#1f2933" text-anchor="start">{name}</text>')
    p.append(f'<text x="{pad}" y="{H-8}" font-size="12" fill="#5b6672">'
             f'cosine between columns of B, averaged over {n_seed} seeds  ({arm})'
             f'   —  no industry label was given to the model</text>')
    p.append("</svg>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(p))
    print(f"\n-> 圖已寫入 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
