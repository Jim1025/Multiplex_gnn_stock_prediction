"""
e8_noise_floor.py — E8：擴充後 universe 的 daily-IC 噪音地板

M8 實驗 5 在 k=3..7 量到 sigma(daily IC) 與零訊號理論值 1/sqrt(k-1) 逐點
吻合（k=7 實測 0.404 vs 理論 0.408），並據此外推出「分辨 Delta IC = 0.02
需約 41 檔」。擴充後 TW 層有 50 檔，這裡把那條曲線量到 k=50，確認外推
成立、並給出 E9 需要的樣本數。

為什麼不能直接沿用 m8_crosssection_noise.py：
    該檔用 list(combinations(range(n), k)) 枚舉所有子集再抽樣。n=7 時最多
    35 個子集，n=50 時 C(50,25) 約 1.26e14，會直接耗盡記憶體。本檔改為
    直接抽樣子集（RNG.choice(n, k, replace=False)），對統計量等價。

三種量法：
    A. 隨機預測器蒙地卡羅——只需要真實 y。對每個測試日抽 k 檔、生成與 y
       獨立的隨機 y_hat，量 daily IC 的散布。這是「零訊號」的定義本身，
       不依賴任何模型，是噪音地板最乾淨的量法。
       真正要檢驗的是：報酬有厚尾，Pearson 相關的 Var(r) ~ 1/(k-1) 這條
       常態近似在真實分布下是否仍成立。
    B. tw50 實際預測的子抽樣——確認真實 run 的 IC 散布確實落在地板上。
    C. k7 的 21 個 run（M8 原始 TAGS）——用同一支程式碼重跑 k=3..7，
       驗證本檔的實作能重現 M8 記錄的 0.404。

y 直接由 tw50 快照的 test split 取得，不依賴任何訓練 run 是否存在。

輸出：docs/e8_noise_floor.md + docs/figures/e8_noise_floor.png
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.multiplex_dataset import MultiplexDataset  # noqa: E402

SEED = 42
N_SUBSETS_PER_DAY = 20      # 與 M8 相同
N_MC_PER_SUBSET = 50        # 隨機預測器的重複次數
K_GRID = [3, 4, 5, 6, 7, 10, 15, 20, 25, 30, 40, 50]

# M8 實驗 5 的 21 個 run（Part D 全部 family x 3 seeds）
M8_TAGS = [
    "opt_p20_adv_alstm", "opt_p55_advalstm_s7", "opt_p56_advalstm_s123",
    "opt_p22_man_sf", "opt_p57_mansf_s7", "opt_p58_mansf_s123",
    "opt_p29_dl_lr5e4_pat15", "opt_p59_dl5e4_s7", "opt_p60_dl5e4_s123",
    "opt_p23_hgt", "opt_p61_hgt_s7", "opt_p62_hgt_s123",
    "opt_p33_meig_lr5e4_pat15", "opt_p63_meig5e4_s7", "opt_p64_meig5e4_s123",
    "opt_p2_variance_penalty", "opt_p37_magnet_seed7", "opt_p38_magnet_seed123",
    "opt_p46_raw_lr5e4_s42", "opt_p47_raw_lr5e4_s7", "opt_p48_raw_lr5e4_s123",
]


# ---------------------------------------------------------------------------
# 資料
# ---------------------------------------------------------------------------

def load_tw50_y() -> np.ndarray:
    """tw50 test split 的真實標籤，[n_days, 50]。"""
    ds = MultiplexDataset(snapshot_dir="data/graphs/snapshots_tw50",
                          features_dir="data/features", split="test",
                          config_path="configs/tw50.yaml")
    return np.stack([ds[i]["y"].numpy() for i in range(len(ds))], axis=0)


def load_preds_wide(tag: str) -> np.ndarray | None:
    """把一個 run 的 test predictions 轉成 [n_days, n_tickers] 的 (y_hat, y) 對。"""
    hits = sorted(glob.glob(str(ROOT / "runs" / "**" / f"*{tag}" /
                                "predictions" / "test_predictions.csv")))
    if not hits:
        return None
    df = pd.read_csv(hits[-1])
    yh = df.pivot(index="target_date", columns="ticker", values="y_hat").to_numpy()
    y = df.pivot(index="target_date", columns="ticker", values="y").to_numpy()
    return np.stack([yh, y], axis=0)          # [2, n_days, n_tickers]


# ---------------------------------------------------------------------------
# 量測
# ---------------------------------------------------------------------------

def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """對最後一軸逐列算 Pearson r；常數列回 nan。"""
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    na = np.sqrt((a * a).sum(axis=-1))
    nb = np.sqrt((b * b).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(axis=-1) / (na * nb)


def sigma_random_predictor(y: np.ndarray, k: int, rng: np.random.Generator) -> float:
    """
    A：隨機預測器。每個測試日抽 k 檔，配 N_MC_PER_SUBSET 組獨立隨機 y_hat，
    回傳所有 daily IC 的標準差。
    """
    n_days, n = y.shape
    ics = []
    for d in range(n_days):
        # 每日抽 N_SUBSETS_PER_DAY 個不重複子集
        if k == n:
            subsets = np.arange(n)[None, :]
        else:
            subsets = np.stack([rng.choice(n, size=k, replace=False)
                                for _ in range(N_SUBSETS_PER_DAY)], axis=0)
        ys = y[d][subsets]                                  # [S, k]
        ys = np.repeat(ys, N_MC_PER_SUBSET, axis=0)         # [S*R, k]
        yh = rng.standard_normal(ys.shape)                  # 與 y 獨立
        r = _pearson_rows(yh, ys)
        ics.append(r[np.isfinite(r)])
    return float(np.std(np.concatenate(ics)))


def sigma_from_predictions(pair: np.ndarray, k: int, rng: np.random.Generator) -> float:
    """B/C：實際預測的子抽樣。pair = [2, n_days, n_tickers]。"""
    yh_all, y_all = pair
    n_days, n = y_all.shape
    if k > n:
        return float("nan")
    ics = []
    for d in range(n_days):
        if k == n:
            subsets = np.arange(n)[None, :]
        else:
            subsets = np.stack([rng.choice(n, size=k, replace=False)
                                for _ in range(N_SUBSETS_PER_DAY)], axis=0)
        r = _pearson_rows(yh_all[d][subsets], y_all[d][subsets])
        ics.append(r[np.isfinite(r)])
    return float(np.std(np.concatenate(ics)))


# ---------------------------------------------------------------------------
# 推導量
# ---------------------------------------------------------------------------

def k_needed(delta: float, n_days: int) -> int:
    """單 run 的 246 日 mean IC 之 95% CI 不含 0 所需的 k（沿用 M8 定義）。"""
    sigma_needed = delta / 1.96 * np.sqrt(n_days)
    return int(np.ceil(1.0 / sigma_needed ** 2 + 1))


def seeds_needed(delta: float, sigma_run: float) -> int:
    """雙樣本 t 檢定（alpha=0.05 雙尾, power=0.8）每組所需 run 數。"""
    return int(np.ceil(2 * (1.96 + 0.8416) ** 2 * sigma_run ** 2 / delta ** 2))


def main() -> None:
    rng = np.random.default_rng(SEED)

    y50 = load_tw50_y()
    n_days, n_stocks = y50.shape
    print(f"[E8] tw50 test split: {n_days} 天 x {n_stocks} 檔")

    # A：隨機預測器
    sig_mc = {k: sigma_random_predictor(y50, k, rng) for k in K_GRID}

    # B：tw50 實際預測（若有 run）
    tw50_pair = load_preds_wide("tw50_smoke")
    sig_tw50 = ({k: sigma_from_predictions(tw50_pair, k, rng) for k in K_GRID}
                if tw50_pair is not None else {})

    # C：k7 的 21 個 run，重現 M8
    k7_vals: dict[int, list[float]] = {k: [] for k in K_GRID if k <= 7}
    n_k7 = 0
    for tag in M8_TAGS:
        pair = load_preds_wide(tag)
        if pair is None:
            continue
        n_k7 += 1
        for k in k7_vals:
            k7_vals[k].append(sigma_from_predictions(pair, k, rng))
    sig_k7 = {k: float(np.mean(v)) for k, v in k7_vals.items() if v}

    theory = {k: 1.0 / np.sqrt(k - 1) for k in K_GRID}

    # ── 報告 ────────────────────────────────────────────────
    L = [
        "# E8：擴充後 universe 的 daily-IC 噪音地板",
        "",
        f"測試窗 {n_days} 天 x {n_stocks} 檔（tw50 test split，與 k=7 凍結基準同一段日期）。",
        f"每個測試日抽 {N_SUBSETS_PER_DAY} 個 k 檔子集；隨機預測器每個子集再抽 "
        f"{N_MC_PER_SUBSET} 組獨立 y_hat。seed={SEED}。",
        "",
        "## sigma(daily IC) vs cross-section 大小",
        "",
        "| k | 理論 1/sqrt(k-1) | A 隨機預測器 | B tw50 實際預測 | C k7 實際預測 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in K_GRID:
        b = f"{sig_tw50[k]:.3f}" if k in sig_tw50 and np.isfinite(sig_tw50[k]) else "—"
        c = f"{sig_k7[k]:.3f}" if k in sig_k7 else "—"
        L.append(f"| {k} | {theory[k]:.3f} | {sig_mc[k]:.3f} | {b} | {c} |")

    sig50 = sig_mc[50]
    se50 = sig50 / np.sqrt(n_days)
    se7 = sig_mc[7] / np.sqrt(n_days)
    L += [
        "",
        f"C 欄為 M8 實驗 5 的 {n_k7} 個 run 以本檔程式碼重算，作為實作驗證"
        f"（M8 記錄的 k=7 值為 0.404）。",
        "",
        "## 擴充買到了什麼",
        "",
        "| 量 | k=7 | k=50 | 改善 |",
        "|---|---:|---:|---:|",
        f"| sigma(daily IC) | {sig_mc[7]:.3f} | {sig50:.3f} | {sig_mc[7]/sig50:.2f}x |",
        f"| {n_days} 日 mean IC 的 SE | {se7:.4f} | {se50:.4f} | {se7/se50:.2f}x |",
        "",
        f"- 分辨 Delta IC = 0.02（單 run 95% CI 不含 0）需 k ≈ {k_needed(0.02, n_days)} 檔；"
        f"Delta IC = 0.01 需 k ≈ {k_needed(0.01, n_days)} 檔。k=50 跨過了前者，未跨過後者。",
        "",
        "## E9 的樣本數",
        "",
        "k=7 實測 sigma_run = 0.027，與當時 246 日 mean IC 的 SE（0.026）同量級，",
        "亦即 run 間變異主要就是 IC 估計噪音。若此關係在 k=50 維持，",
        f"sigma_run 應落在 {se50:.3f} 附近（保守估到 0.015）。",
        "",
        "雙樣本 t 檢定（alpha=0.05 雙尾、power=0.8）每組所需 run 數：",
        "",
        "| 要偵測的 Delta IC | sigma_run = 0.010 | 0.015 | 0.027（k=7 水準） |",
        "|---:|---:|---:|---:|",
    ]
    for d in (0.10, 0.05, 0.02, 0.01):
        L.append(f"| {d:.2f} | {seeds_needed(d, 0.010)} | {seeds_needed(d, 0.015)} "
                 f"| {seeds_needed(d, 0.027)} |")
    L += [
        "",
        "Phase 0 在 k=7 量到 early fusion 領先 MAGNET 約 0.10。若該差距維持，",
        "每組 2-3 個 seed 即足夠；若如事先登記般收縮到 0.02 量級，每組需 4-9 個。",
        "建議先跑 5 個 seed，落在中間地帶再補。",
    ]

    out_md = ROOT / "docs" / "e8_noise_floor.md"
    out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L))

    # ── 圖 ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ks = list(K_GRID)
    ax.plot(ks, [theory[k] for k in ks], "s--", color="#c0392b", lw=1.5,
            label=r"theory $1/\sqrt{k-1}$ (zero signal)")
    ax.plot(ks, [sig_mc[k] for k in ks], "o-", color="0.15", lw=1.8,
            label="A. random predictor (Monte Carlo)")
    if sig_tw50:
        ax.plot(ks, [sig_tw50[k] for k in ks], "^-", color="#2980b9", lw=1.4,
                label="B. tw50 actual predictions")
    if sig_k7:
        kk = sorted(sig_k7)
        ax.plot(kk, [sig_k7[k] for k in kk], "v-", color="#e67e22", lw=1.4,
                label=f"C. k7 actual predictions ({n_k7} runs)")
    ax.axvline(7, color="0.75", lw=1, ls=":")
    ax.axvline(50, color="0.75", lw=1, ls=":")
    ax.set_xlabel("cross-section size k (stocks per day)")
    ax.set_ylabel("std of daily IC")
    ax.set_title("Daily-IC noise floor vs universe size", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out_png = ROOT / "docs" / "figures" / "e8_noise_floor.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    print(f"\n[E8] 報告 → {out_md}\n[E8] 圖   → {out_png}")


if __name__ == "__main__":
    main()
