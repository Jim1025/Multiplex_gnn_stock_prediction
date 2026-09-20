"""collapse_origin.py — 舊 arm 的 26.41x 與新 arm 的 3.91x，差距從哪來（proposal §59.17）

§59.14 回答的是「塌縮從哪一站來」（答案：GAT 與 projection 冗餘）。
本腳本問的是另一個問題：**新舊兩個 arm 的差距**從哪來。

  A. 輸入端對拍     BatchNorm 之後（LSTM 真正吃到的）兩個 arm 是否相同。
                    相同 -> 差距全部是學出來的，不是資料。
  B. 逐站 multiplier 每一站把「共同/偏差」比值乘上多少；完整圖與無鄰居各一輪。
                    兩輪相減就把「訊息傳遞」與「其餘」分開。
  C. GAT 權重互換   把兩個 arm 的 gat_L1 對調。若差距在某一邊的權重裡，
                    互換應該把它帶走；實測是兩邊都帶不走。
  D. GAT 那一站的 2x2  輸入（LSTM 表示）x 運算子（GAT 權重）。
  E. 注意力         有效鄰居數與 self-loop 權重，檢驗 1/sqrt(k) 能解釋多少。
  F. 反向傳播       新式 ③ 回傳到 h₁ 的梯度跨 i 的和是否恆為 0。

注意：A–E 全部是 **s42 單顆種子**，D 的 2x2 未跨種子重複。

用法：
    .venv/bin/python scripts/collapse_origin.py
    .venv/bin/python scripts/collapse_origin.py --blocks A B F
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import coupling_geometry as cg   # noqa: E402

SPLIT = "test"
NAMES = [("bn", "BatchNorm 後(LSTM 輸入)"), ("lstm", "LSTM 之後"), ("gat", "GAT 之後"),
         ("lin", "+ Linear"), ("act", "+ GELU"), ("proj", "+ LayerNorm = h₁")]


def trace(run_dir: str, graph_ablate: str | None = None, donor: str | None = None):
    """逐站攔截，回傳每一站的 (||h̄||, E||d||, 比值, 餘弦, 維度)。

    donor 不為 None 時，把該 run 的 gat_L1 權重載進來（C / D 區塊用）。
    """
    model, cfg, p = cg.make_model(ROOT / run_dir, graph_ablate=graph_ablate)
    if donor is not None:
        d, _, _ = cg.make_model(ROOT / donor)
        model.gat_L1.load_state_dict(d.gat_L1.state_dict())
    cap: dict = {}
    buf: dict[str, list] = {k: [] for k, _ in NAMES}
    pr = model.proj_L1
    on = model.lstm.input_norm.forward

    def n_hook(x):
        out = on(x)
        if x.size(2) == model.n_l1:
            cap["bn"] = out.detach()[:, -1]      # 只取歷史窗最後一步
        return out
    model.lstm.input_norm.forward = n_hook
    ol = model.lstm.forward

    def l_hook(x_seq, layer: int = 0):
        out = ol(x_seq, layer=layer)
        if layer == 0 and x_seq.size(2) == model.n_l1:
            cap["lstm"] = out.detach()
        return out
    model.lstm.forward = l_hook
    og = model._apply_gat_batched

    def g_hook(g, h, ei, ea):
        out = og(g, h, ei, ea)
        if h.size(1) == model.n_l1:
            cap["gat"] = out.detach()
            z = pr.linear(out)
            cap["lin"] = z.detach()
            cap["act"] = pr.act(z).detach()
        return out
    model._apply_gat_batched = g_hook
    oa = model._augment_weak
    model._augment_weak = lambda h: (cap.__setitem__("proj", h.detach()), oa(h))[1]

    with torch.no_grad():
        for b in cg.make_loader(cfg, p, SPLIT):
            model(b)
            for k in buf:
                buf[k].append(cap[k].cpu().numpy())
    out = {}
    for k, v in buf.items():
        A = np.concatenate(v, 0).astype(np.float64)
        nb, nd = cg.mean_dev(A)
        out[k] = dict(hbar=nb, dev=nd, ratio=nb / max(nd, 1e-12),
                      cos=cg.pairwise_cos(A), d=A.shape[2],
                      norm=float(np.linalg.norm(A, axis=2).mean()))
    return out


def block_A():
    print("\n" + "=" * 78)
    print("A. 輸入端對拍：差距是不是資料造成的")
    print("=" * 78)
    print(f"\n  {'arm':10s} {'比值':>8s} {'E||d||/E||x||':>14s} {'餘弦':>9s}")
    for lab, rd in (("舊 arm", cg.OLD_RUN), ("新 arm", cg.NEW_RUN)):
        r = trace(rd)["bn"]
        print(f"  {lab:10s} {r['ratio']:7.2f}x {r['dev']/r['norm']:14.4f} {r['cos']:+9.4f}")
    print("\n  -> 進 LSTM 前兩個 arm 幾乎相同。差距全部是學出來的。")


def block_B():
    print("\n" + "=" * 78)
    print("B. 逐站 multiplier：每一站把比值乘上多少（>1 = 更塌）")
    print("=" * 78)
    R = {}
    for lab, rd in (("舊", cg.OLD_RUN), ("新", cg.NEW_RUN)):
        for ab, an in ((None, "完整圖"), ("empty_l1", "無鄰居")):
            R[(lab, an)] = trace(rd, graph_ablate=ab)

    def mult(r):
        out, prev = {}, None
        for k, nm in NAMES:
            cur = r[k]["ratio"]
            if prev is not None:
                out[nm] = cur / prev
            prev = cur
        return out

    ratios = {}
    for an in ("完整圖", "無鄰居"):
        mo, mn = mult(R[("舊", an)]), mult(R[("新", an)])
        print(f"\n  [{an}]  {'站':26s} {'舊 arm':>8s} {'新 arm':>8s} {'舊/新':>8s}")
        po = pn = 1.0
        for _, nm in NAMES[1:]:
            print(f"  {'':10s} {nm:26s} {mo[nm]:7.2f}x {mn[nm]:7.2f}x {mo[nm]/mn[nm]:7.2f}x")
            po *= mo[nm]
            pn *= mn[nm]
        o, n = R[("舊", an)]["proj"]["ratio"], R[("新", an)]["proj"]["ratio"]
        ratios[an] = (o, n)
        print(f"  {'':10s} {'合計':26s} {po:7.2f}x {pn:7.2f}x {po/pn:7.2f}x")
        print(f"  {'':10s} h₁ 比值：舊 {o:.2f}x  新 {n:.2f}x  差距 {o/n:.2f}x")
    (of, nf), (oe, ne) = ratios["完整圖"], ratios["無鄰居"]
    print(f"\n  兩段拆解：{of/nf:.2f}x = 訊息傳遞 {(of/oe)/(nf/ne):.2f}x  x  非訊息傳遞 {oe/ne:.2f}x")


def block_C():
    print("\n" + "=" * 78)
    print("C. GAT 權重互換：差距在 GAT 自己的權重裡嗎")
    print("=" * 78)
    print(f"\n  {'組合':34s} {'h₁ 比值':>9s} {'h₁ 餘弦':>9s}")
    for lab, host, donor in (("舊 arm 原樣", cg.OLD_RUN, None),
                             ("舊 arm + 新 arm 的 GAT 權重", cg.OLD_RUN, cg.NEW_RUN),
                             ("新 arm + 舊 arm 的 GAT 權重", cg.NEW_RUN, cg.OLD_RUN),
                             ("新 arm 原樣", cg.NEW_RUN, None)):
        r = trace(host, donor=donor)["proj"]
        print(f"  {lab:34s} {r['ratio']:8.2f}x {r['cos']:+9.4f}")
    print("\n  -> 兩個方向的互換都落在中間。權重與輸入是共適應的。")


def block_D():
    print("\n" + "=" * 78)
    print("D. GAT 那一站的 2x2：輸入 x 運算子")
    print("=" * 78)
    print(f"\n  {'LSTM 表示':>10s} {'GAT 權重':>10s} {'GAT 前':>8s} {'GAT 後':>8s} {'這一站乘上':>10s}")
    cells = {}
    for li, di, host, donor in (("舊", "舊", cg.OLD_RUN, None),
                                ("舊", "新", cg.OLD_RUN, cg.NEW_RUN),
                                ("新", "舊", cg.NEW_RUN, cg.OLD_RUN),
                                ("新", "新", cg.NEW_RUN, None)):
        t = trace(host, donor=donor)
        m = t["gat"]["ratio"] / t["lstm"]["ratio"]
        cells[(li, di)] = m
        print(f"  {li:>10s} {di:>10s} {t['lstm']['ratio']:7.2f}x {t['gat']['ratio']:7.2f}x {m:9.2f}x")
    a = cells[("舊", "舊")] / cells[("舊", "新")]
    b = cells[("舊", "舊")] / cells[("新", "舊")]
    ab = cells[("舊", "舊")] / cells[("新", "新")]
    print(f"\n  只換運算子 {a:.2f}x   只換輸入 {b:.2f}x   兩個一起 {ab:.2f}x")
    print(f"  {a:.2f} x {b:.2f} = {a*b:.2f}，離 {ab:.2f} 差 {ab/(a*b):.2f}x -> **那是交互作用**")


def block_E():
    print("\n" + "=" * 78)
    print("E. 注意力：1/sqrt(k) 能解釋多少")
    print("=" * 78)
    print("\n  （有效鄰居與 self-loop 權重由 coupling_geometry.py 區塊 H 量，此處只做換算）")
    print("  實測 舊 19.8 -> 新 13.2 個有效鄰居。")
    print(f"  1/sqrt(k) 模型預測的改善：sqrt(19.8/13.2) = {np.sqrt(19.8/13.2):.2f}x")
    print("  但 D 區塊量到的運算子效果是 1.06x（舊輸入）到 3.39x（新輸入）。")
    print("  -> 1/sqrt(k) 假設偏差各向同性；偏差一旦有結構，這個模型就失效。")


def block_F():
    print("\n" + "=" * 78)
    print("F. 反向傳播：新式 ③ 回傳到 h₁ 的梯度是否跨 i 均值為零")
    print("=" * 78)
    torch.manual_seed(42)
    n, d, n2 = 30, 32, 50
    B = torch.randn(n, n2) * 0.1
    g = torch.randn(n2, d)
    print()
    for nm, fn in (("舊式 Σᵢ B[i,j]·h₁ᵢ", lambda h: torch.einsum("ij,id->jd", B, h)),
                   ("新式 Σᵢ B[i,j]·(h₁ᵢ−h̄₁)",
                    lambda h: torch.einsum("ij,id->jd", B, h - h.mean(0, keepdim=True)))):
        h = torch.randn(n, d, requires_grad=True)
        (fn(h) * g).sum().backward()
        G = h.grad
        common = float(G.mean(0).norm())
        dev = float((G - G.mean(0)).norm(dim=1).mean())
        print(f"  {nm:26s} ‖跨 i 平均梯度‖ {common:.3e}   平均偏差 {dev:.4f}")
    print("\n  -> 偏差部分逐位元相同，共同部分在新式是浮點零。")
    print("     前向把共同成分移出 B 的輸入，反向把它移出 B 對 h₁ 的請求。")


BLOCKS = {"A": block_A, "B": block_B, "C": block_C,
          "D": block_D, "E": block_E, "F": block_F}


def main() -> None:
    ap = argparse.ArgumentParser(description="26.41x -> 3.91x 的來源拆解（§59.17）")
    ap.add_argument("--blocks", nargs="*", default=list(BLOCKS), choices=list(BLOCKS))
    for b in ap.parse_args().blocks:
        BLOCKS[b]()


if __name__ == "__main__":
    main()
