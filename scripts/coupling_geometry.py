"""
coupling_geometry.py — 耦合層的最佳化幾何（proposal §59）

回答教授的兩個問題：
  Q1 「為何共同成分是個股偏差的 26 倍？」
  Q2 「為何既有機制在舊寫法下學不起來？」

六個區塊，全部可獨立重跑：

  A. h₁ 的共同／離散分解
     把 h̄₁ 再拆成「跨日常數 c」與「隨日變動 m(t)」。前者不是市場，
     是編碼器的固定偏移；LayerNorm 又把每個向量長度釘在 sqrt(d')。
     所以那個比值量的是「30 檔美股有多像」，不是「大盤有多大」。

  B. 前向傳播逐階段的塌縮軌跡
     原始特徵 -> LSTM -> GAT -> projection。找出塌縮發生在哪一站。

  C. L1 圖密度與原始特徵的離散度
     檢驗塌縮是資料造成的還是模型造成的。原始特徵要逐特徵標準化後再看，
     否則會被單一大尺度特徵（RSI_14，量級 53.8）主導。

  D. 耦合層對 B 的曲率
     耦合層對 B 是線性的，所以 Gauss-Newton Hessian 就是 Gram(H)，
     可解析算，不需要 autograd Hessian（這是 §49.4 說「沒量過條件數」的正解）。

  E. B 的初始梯度分解
     把 dL/dB 逐欄拆成「整欄同動（只改欄和＝市場曝險）」與
     「欄內的差（唯一能造出個股結構的方向）」。這是 Q2 的直接證據。

  F. 訓練後 |B| 的跨種子分布
     舊式 vs 新式的實際結果。

  H. GAT 的注意力分布（稀疏化還有沒有邊際價值）
     取訓練後的 GAT，逐目的節點看注意力怎麼攤。三個量：
       - self-loop 權重：節點保留自己多少（過度平滑的直接指標）
       - 有效鄰居數 exp(entropy)：實際上在平均幾個鄰居
       - 前 k 強鄰居的質量佔比：若前 5 強已佔九成，top-k 過濾等於沒做事
     本區塊複製 GATEncoder.forward 以取出 attention，因此內建對拍。

  H. GAT 注意力的集中度
     塌縮發生在 GAT（區塊 B）。這一塊回答「是圖太密，還是注意力壓不出
     差距」——量有效鄰居數 exp(H(α))、正規化熵、自環權重，以及注意力
     與邊權 |rho| 的秩相關。決定稀疏化會不會咬到。

  G. ② 到底在傳遞什麼
     h̄₁(t) = c + m(t)。把 h̄₁ 凍結成 c（拿掉所有隨日變動的市場資訊）再看
     test IC 掉多少，就知道 ② 是真的在傳市場狀態，還是只是一個學出來的
     每檔固定截距。本區塊會重新實作 beta 分支，因此先跑一次「不凍結」
     版本與 baseline 對拍，確認重實作沒有偏離模型本體才往下做。

用法：
    .venv/bin/python scripts/coupling_geometry.py
    .venv/bin/python scripts/coupling_geometry.py --blocks A B E
    .venv/bin/python scripts/coupling_geometry.py --split train
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.multiplex_dataset import MultiplexDataset, multiplex_collate   # noqa: E402
from src.models import build_model                                              # noqa: E402
from src.train.evaluator import evaluate                                        # noqa: E402
from src.train.utils import batch_to_device, load_checkpoint                    # noqa: E402

# 舊式＝候選邊直接乘 h₁ᵢ；新式＝beta 層（②+③）。兩者的 config 只差
# weak_links 區塊，其餘完全相同（已用 diff 確認），所以可以直接對比。
OLD_RUN = "runs/tw50/inputnorm/20260822_1746_tw50_inbnw_noskip_s42"
NEW_RUN = "runs/tw50/beta/20260824_1740_tw50_beta_s42"
OLD_ARM = "tw50_inbnw_noskip"
NEW_ARM = "tw50_beta"
SEED = 42
GRAD_BATCHES = 8          # E 區塊平均幾個 batch 的梯度


# ── 共用工具 ──────────────────────────────────────────────────────────

def load_cfg(run_dir: Path) -> tuple[dict, str]:
    p = str(run_dir / "config_snapshot.yaml")
    return yaml.safe_load(open(p)), p


def make_model(run_dir: Path, trained: bool = True):
    cfg, p = load_cfg(run_dir)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = build_model(cfg)
    if trained:
        load_checkpoint(run_dir / "checkpoints" / "best.pt", model,
                        optimizer=None, map_location=torch.device("cpu"))
    model.eval()
    return model, cfg, p


def make_loader(cfg: dict, cfg_path: str, split: str) -> DataLoader:
    ds = MultiplexDataset(snapshot_dir=str(ROOT / cfg["data"]["snapshot_dir"]),
                          features_dir=str(ROOT / cfg["data"]["features_dir"]),
                          T=cfg["model"]["lstm"]["T_history"],
                          split=split, config_path=cfg_path)
    return DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]),
                      shuffle=False, collate_fn=multiplex_collate, num_workers=0)


def grab_h1(model, loader) -> np.ndarray:
    """攔截進入 _augment_weak 的 h_L1，回傳 [T, n1, d']。"""
    buf: list[np.ndarray] = []
    orig = model._augment_weak
    model._augment_weak = lambda h: (buf.append(h.detach().cpu().numpy()), orig(h))[1]
    with torch.no_grad():
        for b in loader:
            model(b)
    model._augment_weak = orig
    return np.concatenate(buf, axis=0)


def mean_dev(A: np.ndarray) -> tuple[float, float]:
    """A: [T, n, d] -> (逐日 ||橫截面平均|| 的平均, 逐日逐檔 ||偏差|| 的平均)。"""
    A = np.asarray(A, dtype=np.float64)
    mu = A.mean(axis=1)
    dv = A - mu[:, None, :]
    return (float(np.linalg.norm(mu, axis=1).mean()),
            float(np.linalg.norm(dv, axis=2).mean()))


def pairwise_cos(A: np.ndarray) -> float:
    """A: [T, n, d] -> 逐日算 n 檔兩兩餘弦的平均，再跨日平均。"""
    out = []
    for At in np.asarray(A, dtype=np.float64):
        Z = At / np.maximum(np.linalg.norm(At, axis=1, keepdims=True), 1e-12)
        C = Z @ Z.T
        out.append(C[np.triu_indices(At.shape[0], 1)].mean())
    return float(np.mean(out))


def load_B(run_dir: str) -> np.ndarray | None:
    """訓練後的 B_eff = weak_beta * weak_mask。"""
    sd = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"),
                    map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    if "weak_beta" in sd:
        B = sd["weak_beta"].numpy()
    elif "weak_U" in sd and "weak_V" in sd:
        B = (sd["weak_U"] @ sd["weak_V"].t()).numpy()
    else:
        return None
    mask = sd.get("weak_mask")
    return B * mask.numpy() if mask is not None else B


def find_seeds(arm: str) -> dict[str, str]:
    out = {}
    for d in sorted(glob.glob(str(ROOT / "runs" / "**" / f"*{arm}_s*"), recursive=True)):
        m = re.fullmatch(rf"\d{{8}}_\d{{4}}_{re.escape(arm)}_s(\d+)", os.path.basename(d))
        if m and os.path.exists(os.path.join(d, "checkpoints", "best.pt")):
            out[m.group(1)] = d
    return out


# ── A. h₁ 的共同／離散分解 ────────────────────────────────────────────

def block_A(split: str) -> None:
    print("\n" + "=" * 78)
    print("A. h₁ 的共同／離散分解 —— 那個比值量的是什麼")
    print("=" * 78)
    for lab, rd in (("舊 arm", OLD_RUN), ("新 arm", NEW_RUN)):
        model, cfg, p = make_model(ROOT / rd)
        H = grab_h1(model, make_loader(cfg, p, split))
        T, n1, d = H.shape
        hbar = H.mean(axis=1)
        c = hbar.mean(axis=0)                    # 跨日常數
        m = hbar - c[None, :]                    # 隨日變動
        nb, nd = mean_dev(H)
        n_c, n_m = float(np.linalg.norm(c)), float(np.linalg.norm(m, axis=1).mean())
        print(f"\n  [{lab}]  T={T}  n1={n1}  d'={d}   sqrt(d')={np.sqrt(d):.4f}")
        print(f"    每檔 ||h₁ᵢ||                    {np.linalg.norm(H, axis=2).mean():8.4f}"
              f"   <- LayerNorm 把它釘在 sqrt(d')")
        # c 與 m 正交（m 是對時間平均的偏差），所以能量可加、佔比才會加總到 100%。
        # 直接用範數比會加總超過 100%，那是錯的。
        e_c, e_m = n_c ** 2, float((np.linalg.norm(m, axis=1) ** 2).mean())
        print(f"    ||h̄₁|| 共同成分                 {nb:8.4f}")
        print(f"      其中 跨日常數 ||c||           {n_c:8.4f}   能量佔 {e_c / (e_c + e_m) * 100:5.1f}%"
              f"   <- 這一塊不是市場")
        print(f"      其中 隨日變動 ||m(t)||        {n_m:8.4f}   能量佔 {e_m / (e_c + e_m) * 100:5.1f}%"
              f"   <- 這才是市場狀態")
        print(f"    ||h₁ᵢ − h̄₁|| 個股偏差           {nd:8.4f}")
        print(f"    共同／偏差                      {nb / nd:8.2f}x")
        print(f"    隨日變動／偏差                  {n_m / nd:8.2f}x   <- 誠實的「市場 vs 個股」比")
        print(f"    30 檔兩兩餘弦                   {pairwise_cos(H):+8.4f}"
              f"   <- 1.0 = 完全同一個方向")


# ── B. 前向傳播逐階段的塌縮軌跡 ──────────────────────────────────────

def block_B(split: str) -> None:
    print("\n" + "=" * 78)
    print("B. 塌縮發生在前向傳播的哪一站")
    print("=" * 78)

    def trace(rd: str, lab: str, trained: bool) -> None:
        model, cfg, p = make_model(ROOT / rd, trained=trained)
        cap: dict[str, torch.Tensor] = {}
        orig_lstm = model.lstm.forward
        def lstm_hook(x_seq, layer: int = 0):
            out = orig_lstm(x_seq, layer=layer)
            if layer == 0 and x_seq.size(2) == model.n_l1:
                cap["lstm"] = out.detach()
            return out
        model.lstm.forward = lstm_hook
        orig_gat = model._apply_gat_batched
        def gat_hook(g, h, ei, ea):
            out = orig_gat(g, h, ei, ea)
            if h.size(1) == model.n_l1:
                cap["gat"] = out.detach()
            return out
        model._apply_gat_batched = gat_hook
        orig_aug = model._augment_weak
        def aug_hook(h):
            cap["proj"] = h.detach()
            return orig_aug(h)
        model._augment_weak = aug_hook

        buf = {k: [] for k in ("raw", "lstm", "gat", "proj")}
        with torch.no_grad():
            for b in make_loader(cfg, p, split):
                buf["raw"].append(b["x_seq_L1"][:, -1].numpy())    # 最後一步的原始特徵
                model(b)
                for k in ("lstm", "gat", "proj"):
                    buf[k].append(cap[k].numpy())
        print(f"\n  [{lab}]")
        print(f"    {'階段':26s} {'d':>4s} {'||mean||':>10s} {'||dev||':>9s}"
              f" {'比值':>9s} {'兩兩餘弦':>10s}")
        for key, name in (("raw", "1. 原始特徵（最後一步）"), ("lstm", "2. LSTM 之後"),
                          ("gat", "3. GAT 之後"), ("proj", "4. projection 之後 = h₁")):
            A = np.concatenate(buf[key], 0)
            nb, nd = mean_dev(A)
            print(f"    {name:26s} {A.shape[2]:4d} {nb:10.4f} {nd:9.4f}"
                  f" {nb / max(nd, 1e-12):8.2f}x {pairwise_cos(A):+10.4f}")

    trace(OLD_RUN, "舊 arm @ 初始化（完全未訓練）", trained=False)
    trace(OLD_RUN, "舊 arm @ 訓練後", trained=True)
    trace(NEW_RUN, "新 arm @ 訓練後", trained=True)


# ── C. L1 圖密度與原始特徵的離散度 ───────────────────────────────────

def block_C(split: str) -> None:
    print("\n" + "=" * 78)
    print("C. 塌縮是資料造成的還是模型造成的")
    print("=" * 78)
    cfg, p = load_cfg(ROOT / OLD_RUN)
    n1 = None
    deg, n_edges, X = [], [], []
    for b in make_loader(cfg, p, split):
        n1 = b["x_seq_L1"].size(2)
        eis = b["edge_index_L1"]
        for ei in (eis if isinstance(eis, (list, tuple)) else [eis]):
            deg.append(np.bincount(ei[1].numpy(), minlength=n1))
            n_edges.append(ei.shape[1])
        X.append(b["x_seq_L1"][:, -1].numpy())
    deg = np.array(deg)
    X = np.concatenate(X, 0).astype(np.float64)
    full = n1 * (n1 - 1)
    print(f"\n  L1（美股）圖：{n1} 個節點，平均 {np.mean(n_edges):.1f} 條有向邊"
          f"（滿圖 {full}，密度 {np.mean(n_edges) / full * 100:.1f}%）")
    print(f"    入度：平均 {deg.mean():.1f}  中位數 {np.median(deg):.1f}"
          f"  最小 {deg.min()}  最大 {deg.max()}")
    print(f"    -> 一層 GAT 平均把每檔股票對 {deg.mean():.0f} / {n1 - 1} 檔取加權平均")

    nb, nd = mean_dev(X)
    print(f"\n  原始特徵，未標準化      ：兩兩餘弦 {pairwise_cos(X):+.4f}"
          f"   ||mean||/||dev|| {nb / nd:.2f}x")
    Xs = (X - X.mean(axis=(0, 1), keepdims=True)) / (X.std(axis=(0, 1), keepdims=True) + 1e-9)
    nbs, nds = mean_dev(Xs)
    print(f"  原始特徵，逐特徵 z-score：兩兩餘弦 {pairwise_cos(Xs):+.4f}"
          f"   ||mean||/||dev|| {nbs / nds:.2f}x   <- 資料本身分得很開")
    print(f"\n  未標準化的高餘弦是單一大尺度特徵造成的假象：")
    print(f"    {'特徵':>6s} {'|值| 平均':>12s} {'跨股票 sd':>12s} {'sd/|值|':>9s}")
    for k in range(X.shape[2]):
        lvl = np.abs(X[:, :, k]).mean()
        across = X[:, :, k].std(axis=1).mean()
        flag = "  <- 主導範數" if across / max(lvl, 1e-9) < 0.3 else ""
        print(f"    {k:6d} {lvl:12.4f} {across:12.4f} {across / max(lvl, 1e-9):9.3f}{flag}")


# ── D. 耦合層對 B 的曲率 ─────────────────────────────────────────────

def block_D(split: str) -> None:
    print("\n" + "=" * 78)
    print("D. 耦合層對 B 的曲率（Gauss-Newton Hessian = Gram(H)，可解析算）")
    print("=" * 78)
    print("\n  舊式 out_j = Σᵢ B[i,j]·h₁ᵢ  -> 對 B[:,j] 的 Jacobian 是 H = [h₁₁ … h₁ₙ]")
    print("  新式 ③ 只乘 h₁ᵢ − h̄₁       -> Jacobian 是去均值後的 H，共同模態已移交給 γ")
    for lab, rd, trained in (("舊 arm @ 初始化", OLD_RUN, False),
                             ("舊 arm @ 訓練後", OLD_RUN, True),
                             ("新 arm @ 訓練後", NEW_RUN, True)):
        model, cfg, p = make_model(ROOT / rd, trained=trained)
        H = grab_h1(model, make_loader(cfg, p, split)).astype(np.float64)
        ev = np.array([np.linalg.eigvalsh(Ht @ Ht.T)[::-1] for Ht in H]).mean(axis=0)
        print(f"\n  [{lab}]")
        print(f"    λ1 {ev[0]:11.4f}   λ2 {ev[1]:11.5f}   λ1/λ2 {ev[0] / ev[1]:10.1f}"
              f"   λ1 佔 trace {ev[0] / ev.sum() * 100:6.2f}%")
    print("\n  註：λ_min 在數值上是 0（30 個向量近似線性相依），所以不報 λ1/λ_min。")
    print("      有意義的是 λ1/λ2：單一學習率下步長上限由 λ1 決定，")
    print("      次大方向的等效步長就被壓這個倍數。")


# ── E. B 的初始梯度分解 ──────────────────────────────────────────────

def block_E() -> None:
    print("\n" + "=" * 78)
    print("E. B 的初始梯度往哪個方向指（Q2 的直接證據）")
    print("=" * 78)
    print("\n  舊式 dL/dB[i,j] = <g_j, h₁ᵢ>。30 個 h₁ᵢ 幾乎是同一個向量，")
    print("  所以這 30 個偏導數也幾乎相同 -> 梯度只推得動「整欄一起動」，")
    print("  而整欄一起動只改欄和（＝市場曝險），造不出個股結構。")
    for lab, rd in (("舊參數化（B 直接乘 h₁ᵢ）", OLD_RUN),
                    ("新參數化（B 只乘 h₁ᵢ − h̄₁）", NEW_RUN)):
        model, cfg, p = make_model(ROOT / rd, trained=False)
        # 必須用 train 模式：這裡量的是「訓練第一步時 B 收到什麼」，
        # eval 模式會關掉 dropout 並讓 BatchNorm 改用未適應的 running stats，
        # 量到的梯度量級差一個數量級，不是訓練時的實況。
        model.train()
        acc_B, acc_all = None, None
        for k, b in enumerate(make_loader(cfg, p, "train")):
            if k >= GRAD_BATCHES:
                break
            b = batch_to_device(b, torch.device("cpu"))
            y_hat, extras = model(b)
            loss, _ = model.compute_loss(y_hat, b["y"], extras)
            model.zero_grad()
            loss.backward()
            g = model.weak_beta.grad.detach().clone()
            acc_B = g if acc_B is None else acc_B + g
            others = np.array([q.grad.detach().abs().mean().item()
                               for q in model.parameters()
                               if q.grad is not None and q.numel() > 1])
            acc_all = others if acc_all is None else acc_all + others
        G = (acc_B / GRAD_BATCHES).numpy().astype(np.float64)
        median = float(np.median(acc_all / GRAD_BATCHES))
        col_mean = G.mean(axis=0, keepdims=True)          # 整欄同動的分量
        resid = G - col_mean                              # 欄內的差
        e_common = float((col_mean ** 2).sum() * G.shape[0])
        e_resid = float((resid ** 2).sum())
        tot = e_common + e_resid
        rms = lambda a: float(np.sqrt((a ** 2).mean()))
        print(f"\n  [{lab}]")
        print(f"    dL/dB 逐格 RMS                        {rms(G):.3e}")
        print(f"      共同分量（只改欄和＝市場曝險）      {rms(np.broadcast_to(col_mean, G.shape)):.3e}"
              f"   能量佔 {e_common / tot * 100:6.2f}%")
        print(f"      殘差分量（唯一能造出個股結構的）    {rms(resid):.3e}"
              f"   能量佔 {e_resid / tot * 100:6.2f}%")
        print(f"    同一次前向，模型參數梯度中位數        {median:.3e}")
        print(f"    -> B 的梯度{'高於' if rms(G) > median else '低於'}中位數，"
              f"但有用的殘差分量是中位數的 {rms(resid) / median:.4f} 倍")


# ── F. 訓練後 |B| 的跨種子分布 ───────────────────────────────────────

def block_F() -> None:
    print("\n" + "=" * 78)
    print("F. 訓練後 |B| 的跨種子分布（B 從 0 初始化）")
    print("=" * 78)
    for lab, arm in (("舊式", OLD_ARM), ("新式", NEW_ARM)):
        rows = []
        for seed, d in sorted(find_seeds(arm).items(), key=lambda kv: int(kv[0])):
            B = load_B(d)
            if B is not None:
                rows.append((seed, np.abs(B).max(), np.abs(B).mean()))
        if not rows:
            print(f"\n  [{lab} {arm}] 找不到 checkpoint")
            continue
        mx = np.array([r[1] for r in rows])
        mn = np.array([r[2] for r in rows])
        print(f"\n  [{lab} {arm}]  n={len(rows)} 顆種子")
        print(f"    |B|max   範圍 {mx.min():.6f} – {mx.max():.6f}   中位數 {np.median(mx):.6f}")
        print(f"    |B|mean  範圍 {mn.min():.6f} – {mn.max():.6f}   中位數 {np.median(mn):.6f}")


# ── G. ② 到底在傳遞什麼 ──────────────────────────────────────────────

def block_G(split: str) -> None:
    print("\n" + "=" * 78)
    print("G. ② 是在傳市場狀態，還是只是一個每檔固定的截距")
    print("=" * 78)
    print("\n  h̄₁(t) = c + m(t)。凍結成 c 等於拿掉所有隨日變動的市場資訊，")
    print("  但保留 γ_j·c 這個每檔固定的偏移。IC 掉多少 = m(t) 的價值。")

    dev = torch.device("cpu")
    model, cfg, p = make_model(ROOT / NEW_RUN)
    if not getattr(model, "beta_layer", False):
        print("\n  NEW_RUN 不是 beta 層，跳過。")
        return

    # c 取自 train split：test 期的均值屬於未來資訊，不能拿來當凍結值。
    H_tr = grab_h1(model, make_loader(cfg, p, "train"))
    c = torch.tensor(H_tr.mean(axis=1).mean(axis=0), dtype=torch.float32)
    print(f"\n  c 由 train split 取得，||c|| = {c.norm():.4f}")

    def patched(m, freeze: bool):
        """重新實作 _augment_weak 的 beta 分支；freeze=True 時把 h̄₁ 換成 c。

        這裡刻意複製模型邏輯（而非改模型），所以下面一定要跑對拍：
        freeze=False 的結果必須與 baseline 逐位元相同，否則這個區塊的
        結論不可信。
        """
        def fn(h_L1):
            ident = h_L1.index_select(dim=1, index=m.pair_src)
            ident = ident * m.has_pair.view(1, -1, 1).to(ident.dtype)
            beta_eff = m._weak_beta_full() * m.weak_mask
            hbar_real = h_L1.mean(dim=1, keepdim=True)
            hbar_use = (c.view(1, 1, -1).to(h_L1.dtype).expand_as(hbar_real)
                        if freeze else hbar_real)
            out = torch.zeros_like(ident)
            if m.beta_use_identity:
                out = out + ident * m.beta_alpha.view(1, -1, 1)
            if m.beta_use_factor:
                out = out + m.beta_gamma.view(1, -1, 1) * hbar_use
            if m.beta_use_residual:
                out = out + torch.einsum("ij,bid->bjd", beta_eff, h_L1 - hbar_real)
            return out
        return fn

    def ic_of(mutate=None) -> tuple[float, float]:
        m, cfg_, p_ = make_model(ROOT / NEW_RUN)
        if mutate is not None:
            mutate(m)
        st = evaluate(m, make_loader(cfg_, p_, split), dev,
                      eval_cfg=cfg_.get("evaluation"))
        return float(st["IC"]), float(st["RankIC"])

    base_ic, base_ric = ic_of()
    echo_ic, echo_ric = ic_of(lambda m: setattr(m, "_augment_weak", patched(m, False)))
    if abs(echo_ic - base_ic) > 1e-9 or abs(echo_ric - base_ric) > 1e-9:
        raise SystemExit(
            f"[G] 對拍失敗：重實作的 beta 分支與模型本體不一致"
            f"（IC {echo_ic:.10f} vs {base_ic:.10f}）。模型可能已改動，"
            f"請先更新 patched() 再取用本區塊的結論。")
    print(f"  對拍通過：重實作的 beta 分支與模型本體逐位元相同")

    frz_ic, frz_ric = ic_of(lambda m: setattr(m, "_augment_weak", patched(m, True)))
    off_ic, off_ric = ic_of(lambda m: m.beta_gamma.data.zero_())

    print(f"\n    {'設定':28s} {'IC':>9s} {'RankIC':>9s} {'ΔIC':>9s}")
    print(f"    {'baseline':28s} {base_ic:9.4f} {base_ric:9.4f} {0.0:+9.4f}")
    print(f"    {'② 的 h̄₁ 凍結成常數 c':28s} {frz_ic:9.4f} {frz_ric:9.4f}"
          f" {frz_ic - base_ic:+9.4f}   <- 少掉的是 m(t)")
    print(f"    {'② 整個關掉（γ = 0）':28s} {off_ic:9.4f} {off_ric:9.4f}"
          f" {off_ic - base_ic:+9.4f}   <- 少掉的是 m(t) + 靜態截距")
    d_dyn = base_ic - frz_ic
    d_tot = base_ic - off_ic
    if d_tot > 0:
        print(f"\n    ② 的價值裡：隨日變動 {d_dyn / d_tot * 100:.0f}%，"
              f"靜態截距 {(d_tot - d_dyn) / d_tot * 100:.0f}%")
    print(f"    註：這是單一種子的事後消融，與 §33.5 的重訓消融（10 顆種子）不同量，")
    print(f"        不可互相比較。")


# ── H. GAT 的注意力分布 ──────────────────────────────────────────────

def _gat_attention(gat, x, edge_index, edge_attr):
    """複製 GATEncoder.forward，但把每層的 attention 收出來。

    複製模型邏輯，所以呼叫端一定要對拍：回傳的 h 必須與
    gat(x, edge_index, edge_attr) 逐位元相同。
    """
    alphas = []
    h = x
    for i, conv in enumerate(gat.convs):
        h, (ei_out, a) = conv(h, edge_index, edge_attr=edge_attr,
                              return_attention_weights=True)
        alphas.append((ei_out.detach(), a.detach()))
        if i < len(gat.convs) - 1:
            h = torch.relu(h)
            h = gat.dropout(h)
    return h, alphas


def _attn_stats(ei_out, a, n: int, ei_raw=None, ea_raw=None) -> dict:
    """逐目的節點統計注意力。a: [E', heads] -> 先對 heads 取平均。

    ei_raw / ea_raw 是**加自環之前**的原圖。給了就順便算
    Spearman(alpha, |rho|)——注意力有沒有在用邊權。這一項是判斷
    「top-k by |rho| 是把 GAT 已有的偏好寫死，還是外加一個它沒在用的結構」。
    """
    a = a.mean(dim=1).numpy().astype(np.float64)
    dst = ei_out[1].numpy()
    src = ei_out[0].numpy()
    rho = None
    if ei_raw is not None and ea_raw is not None:
        rho = {(int(u), int(v)): float(r) for u, v, r in
               zip(ei_raw[0].numpy(), ei_raw[1].numpy(), ea_raw.flatten().numpy())}
    self_w, perp, top3, top5, top8, deg, sp = [], [], [], [], [], [], []
    for j in range(n):
        m = dst == j
        if m.sum() == 0:
            continue
        w = a[m]
        w = w / max(w.sum(), 1e-12)          # 同一目的節點的 alpha 本應和為 1
        is_self = src[m] == j
        self_w.append(float(w[is_self].sum()))
        p = w[w > 0]
        perp.append(float(np.exp(-(p * np.log(p)).sum())))
        nb = np.sort(w[~is_self])[::-1]      # 非自環，由大到小
        tot = nb.sum()
        deg.append(int(nb.size))
        for lst, k in ((top3, 3), (top5, 5), (top8, 8)):
            lst.append(float(nb[:k].sum() / tot) if tot > 1e-12 else np.nan)
        if rho is not None and (~is_self).sum() >= 3:
            wn = w[~is_self]
            rn = np.array([rho.get((int(u), j), np.nan) for u in src[m][~is_self]])
            ok = ~np.isnan(rn)
            if ok.sum() >= 3:
                r = stats.spearmanr(wn[ok], rn[ok]).statistic
                if not np.isnan(r):
                    sp.append(float(r))
    return {"self": np.mean(self_w), "perp": np.mean(perp), "deg": np.mean(deg),
            "top3": np.nanmean(top3), "top5": np.nanmean(top5), "top8": np.nanmean(top8),
            "sp": float(np.mean(sp)) if sp else float("nan")}


def block_H(split: str) -> None:
    print("\n" + "=" * 78)
    print("H. GAT 的注意力分布 —— 稀疏化還有沒有邊際價值")
    print("=" * 78)
    print("\n  self-loop 權重 = 節點保留自己多少。均勻攤在 d 個鄰居 + 自己時 = 1/(d+1)。")
    print("  有效鄰居數 = exp(entropy)，均勻時等於 d+1。")
    print("  前 k 強佔比 = 非自環的注意力質量有多少集中在前 k 個鄰居。")

    for lab, rd in (("舊 arm", OLD_RUN), ("新 arm", NEW_RUN)):
        model, cfg, p = make_model(ROOT / rd)
        rows: dict[tuple[str, int], list[dict]] = {}
        checked = False
        with torch.no_grad():
            for b in make_loader(cfg, p, split):
                for side, gat, key, n in (("L1", model.gat_L1, "edge_index_L1", model.n_l1),
                                          ("L2", model.gat_L2, "edge_index_L2", model.n_l2)):
                    x_seq = b["x_seq_L1"] if side == "L1" else b["x_seq_L2"]
                    h_lstm = model.lstm(x_seq, layer=0 if side == "L1" else 1)
                    eis = b[key]
                    eas = b["edge_attr_L1" if side == "L1" else "edge_attr_L2"]
                    eis = eis if isinstance(eis, (list, tuple)) else [eis]
                    eas = eas if isinstance(eas, (list, tuple)) else [eas]
                    for t, (ei, ea) in enumerate(zip(eis, eas)):
                        if ei.shape[1] == 0:
                            continue
                        h_manual, alphas = _gat_attention(gat, h_lstm[t], ei, ea)
                        if not checked:
                            ref = gat(h_lstm[t], ei, edge_attr=ea)
                            d = float((h_manual - ref).abs().max())
                            if d > 0:
                                raise SystemExit(
                                    f"[H] 對拍失敗：手動 forward 與 GATEncoder 不一致"
                                    f"（max|diff| = {d:.3e}）。GATEncoder.forward 可能已改動，"
                                    f"請先更新 _gat_attention()。")
                            print(f"\n  對拍通過：手動 forward 與 GATEncoder 逐位元相同")
                            checked = True
                        for li, (ei_out, a) in enumerate(alphas):
                            # 只有第一層的輸入是原圖節點特徵，邊權對應才有意義
                            rows.setdefault((side, li), []).append(
                                _attn_stats(ei_out, a, n,
                                            ei if li == 0 else None,
                                            ea if li == 0 else None))
        print(f"\n  [{lab}]")
        print(f"    {'層':>10s} {'平均入度':>9s} {'self 權重':>10s} {'均勻時':>8s}"
              f" {'有效鄰居':>9s} {'前3強':>8s} {'前5強':>8s} {'前8強':>8s}"
              f" {'ρ 秩相關':>10s}")
        for (side, li), lst in sorted(rows.items()):
            m = {k: float(np.mean([r[k] for r in lst])) for k in lst[0]}
            print(f"    {side + ' 第' + str(li + 1) + '層':>10s} {m['deg']:9.1f}"
                  f" {m['self']:10.4f} {1.0 / (m['deg'] + 1):8.4f}"
                  f" {m['perp']:9.1f} {m['top3']:8.3f} {m['top5']:8.3f} {m['top8']:8.3f}"
                  f" {m['sp']:+10.4f}")
    print("\n  怎麼讀：self 權重接近『均勻時』-> 節點幾乎被鄰居平均取代（過度平滑）。")
    print("  前 5 強佔比接近 1.0 -> GAT 自己已經在忽略其餘鄰居，top-k 過濾是 no-op。")
    print("  ρ 秩相關高但前 k 強佔比接近均勻 -> 注意力『排序對、但壓不出差距』，")
    print("  瓶頸是動態範圍而不是哪些邊存在。")


BLOCKS = {"A": block_A, "B": block_B, "C": block_C,
          "D": block_D, "E": block_E, "F": block_F, "G": block_G,
          "H": block_H}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", nargs="*", default=list(BLOCKS),
                    choices=list(BLOCKS), help="要跑哪幾個區塊（預設全部）")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="A/B/C/D/G/H 用哪一個 split（E 固定用 train，因為那是訓練時的梯度）")
    args = ap.parse_args()
    for key in args.blocks:
        fn = BLOCKS[key]
        fn(args.split) if key in ("A", "B", "C", "D", "G", "H") else fn()
    print()


if __name__ == "__main__":
    main()
