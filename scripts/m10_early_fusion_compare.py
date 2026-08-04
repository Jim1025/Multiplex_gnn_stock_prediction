"""
m10_early_fusion_compare.py — 第 0 階段 T1：early fusion vs MAGNET

動機（口試提問 Q7）：
  「為什麼不做 early fusion？」
  既有三個 Stage 0 ablation 全部是「拿掉 ADR」，沒有一個是「用最笨的方式使用 ADR」。
  本腳本比較：
    - baseline_early_fusion — 輸入層把配對 ADR 的 9 維特徵接在 TW 特徵後面，
      跑一個普通 LSTM（無圖、無閘門、無兩級耦合）
    - MAGNET @ lr 5e-4 — M8 Part E 的 5 次受控 replicate

  若 early fusion 贏，兩級耦合架構即失去正當性。

重要觀察（本次實測）：
  baseline_early_fusion 在同 seed 下**位元確定**——重複執行的預測檔完全相同。
  M8 記錄的 MPS run-to-run 非確定性只發生在 GNN（scatter）路徑，
  純 LSTM 模型不受影響。因此 early fusion 的變異軸只有 seed，
  MAGNET 則同時有 run（sigma≈0.027）與 seed（sigma≈0.007）兩個變異來源。
  兩個 MAGNET arm 皆列出，避免只挑對自己有利的那組比。

兩層檢定（兩者都報，因為它們回答不同問題）：
  1) run 層級：Welch t 檢定。M9 已證明 run 才是變異單位，這是誠實的主檢定。
  2) 日層級：同測試日配對的每日 IC 差。與 M8/M9 方法一致，n 大很多，
     但忽略 run 層級變異，屬 anti-conservative，只作輔助。

用法：
    .venv/bin/python scripts/m10_early_fusion_compare.py
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ttest_1samp, ttest_ind

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.metrics import (  # noqa: E402
    long_short_metrics,
    rank_bucket_returns,
    regression_metrics,
)

ARMS = {
    # early fusion 同 seed 位元確定 → 變異軸只有 seed
    "early_fusion (seeds)": [
        "opt_p70_ef_r1",      # seed 42（config 預設）
        "opt_p70_ef_s7", "opt_p70_ef_s123",
        "opt_p70_ef_s2026", "opt_p70_ef_s314",
    ],
    # M8 Part E：同 seed 同 config 的 5 次受控 replicate（MPS 非位元確定）
    "MAGNET@5e-4 (replicates)": [
        "opt_p46_raw_lr5e4_s42", "opt_p66_fig1_lr5e4",
        "opt_p67_rep_s42_a", "opt_p68_rep_s42_b", "opt_p69_rep_s42_c",
    ],
    # M8 Part B：同 config 不同 seed
    "MAGNET@5e-4 (seeds)": [
        "opt_p46_raw_lr5e4_s42", "opt_p47_raw_lr5e4_s7",
        "opt_p48_raw_lr5e4_s123",
    ],
    # 關鍵對照：與 early fusion 完全同架構但不看 ADR（同協議 lr 5e-4、同 5 個 seed）
    # 這一組隔離出「ADR 資訊本身」的貢獻，排除「較小模型較適合小資料」的解釋
    "LSTM-only, no ADR (seeds)": [
        "opt_p71_lstm_s42", "opt_p71_lstm_s7", "opt_p71_lstm_s123",
        "opt_p71_lstm_s2026", "opt_p71_lstm_s314",
    ],
}
REF_ARM = "early_fusion (seeds)"


def pred_path(tag: str) -> Path:
    hits = sorted(glob.glob(
        str(ROOT / "runs" / f"*{tag}" / "predictions" / "test_predictions.csv")
    ))
    if not hits:
        raise FileNotFoundError(f"找不到 run：{tag}")
    return Path(hits[-1])


def daily_ic_series(tag: str) -> pd.Series:
    df = pd.read_csv(pred_path(tag))
    out = {}
    for d, g in df.groupby("target_date"):
        if g["y_hat"].std() > 0 and g["y"].std() > 0:
            out[d] = float(np.corrcoef(g["y_hat"], g["y"])[0, 1])
    return pd.Series(out)


def daily_cross_sections(tag: str):
    df = pd.read_csv(pred_path(tag))
    yh, yy = [], []
    for _, g in df.groupby("target_date", sort=True):
        g = g.sort_values("ticker")
        yh.append(g["y_hat"].to_numpy(dtype=np.float64))
        yy.append(g["y"].to_numpy(dtype=np.float64))
    return yh, yy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--out-md", default=str(ROOT / "docs" / "m10_early_fusion.md"))
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    pf = cfg["evaluation"]["portfolio"]
    n_side, ppy = int(pf["n_side"]), int(pf["periods_per_year"])

    series = {arm: [daily_ic_series(t) for t in tags] for arm, tags in ARMS.items()}
    mean_ic = {arm: np.array([s.mean() for s in ss]) for arm, ss in series.items()}

    sharpe, dispersion, buckets = {}, {}, {}
    for arm, tags in ARMS.items():
        sh, dp, bk = [], [], []
        for t in tags:
            yh, yy = daily_cross_sections(t)
            sh.append(long_short_metrics(yh, yy, n_side, ppy)["Sharpe"])
            dp.append(regression_metrics(
                np.concatenate(yh), np.concatenate(yy))["dispersion"])
            bk.append(rank_bucket_returns(yh, yy)["by_rank"])
        sharpe[arm] = np.array(sh)
        dispersion[arm] = np.array(dp)
        buckets[arm] = np.asarray(bk).mean(axis=0)

    a = REF_ARM
    others = [k for k in ARMS if k != a]

    day_mean = {arm: pd.concat(ss, axis=1).mean(axis=1) for arm, ss in series.items()}

    comparisons = []
    for b in others:
        t_run, p_run = ttest_ind(mean_ic[a], mean_ic[b], equal_var=False)
        idx = day_mean[a].index.intersection(day_mean[b].index)
        diff = (day_mean[a][idx] - day_mean[b][idx]).dropna()
        t_day, p_day = ttest_1samp(diff, 0.0)
        comparisons.append({
            "b": b, "t_run": t_run, "p_run": p_run,
            "d": mean_ic[a].mean() - mean_ic[b].mean(),
            "t_day": t_day, "p_day": p_day,
            "mean_dIC": diff.mean(), "n_day": len(diff),
        })

    L: list[str] = []
    L.append("# M10 T1：early fusion vs MAGNET（回應「為何不做 early fusion」）\n")
    L.append("> 由 `scripts/m10_early_fusion_compare.py` 自動生成\n")
    L.append(
        "> early fusion = 輸入層把配對 ADR 的 9 維特徵接在 TW 特徵後面，"
        "跑普通 LSTM；無圖、無閘門、無兩級耦合\n"
    )

    L.append("\n## Run 層級（主檢定）\n")
    L.append(
        "| arm | n_runs | test_IC mean | SD | min | max | Sharpe mean | "
        "Sharpe SD | dispersion |"
    )
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        m, s, d = mean_ic[arm], sharpe[arm], dispersion[arm]
        L.append(
            f"| {arm} | {m.size} | {m.mean():.4f} | {m.std(ddof=1):.4f} | "
            f"{m.min():.4f} | {m.max():.4f} | {s.mean():.3f} | "
            f"{s.std(ddof=1):.3f} | {d.mean():.3f} |"
        )
    L.append(
        "\n`dispersion` = std(ŷ) / std(y)。遠小於 1 代表 prediction collapse——"
        "模型退化成近乎常數預測。IC 是尺度不變量，看不出這件事。\n"
    )

    L.append("\n## 逐名次平均實現報酬（跨 run 平均）\n")
    L.append("名次 0 = 當日預測最高。真實的排序訊號應呈單調遞減。\n")
    L.append("| arm | " + " | ".join(f"rank {i}" for i in range(7)) + " |")
    L.append("|---" * 8 + "|")
    for arm in ARMS:
        cells = " | ".join(f"{v:+.4f}" for v in buckets[arm])
        L.append(f"| {arm} | {cells} |")
    L.append(f"\nWelch t 檢定（以 `{a}` 為基準）：\n")
    L.append("| 對照 arm | ΔIC | t | p |")
    L.append("|---|---:|---:|---:|")
    for c in comparisons:
        L.append(
            f"| {c['b']} | {c['d']:+.4f} | {c['t_run']:+.3f} | {c['p_run']:.4f} |"
        )

    L.append("\n## 日層級（輔助，與 M8/M9 方法一致）\n")
    L.append("| 對照 arm | 平均每日 IC 差 | t | p | n_days |")
    L.append("|---|---:|---:|---:|---:|")
    for c in comparisons:
        L.append(
            f"| {c['b']} | {c['mean_dIC']:+.4f} | {c['t_day']:+.3f} | "
            f"{c['p_day']:.4f} | {c['n_day']} |"
        )
    L.append(
        "\n- **但書**：此檢定忽略 run 層級變異（M9 實測 sigma_run ≈ 0.027），"
        "屬 anti-conservative，不可單獨作為結論依據。\n"
    )

    L.append("\n## 逐 run 明細\n")
    L.append("| arm | run tag | test_IC | Sharpe |")
    L.append("|---|---|---:|---:|")
    for arm, tags in ARMS.items():
        for t, m, s in zip(tags, mean_ic[arm], sharpe[arm]):
            L.append(f"| {arm} | {t} | {m:.4f} | {s:+.3f} |")

    L.append("\n## 判讀\n")
    for c in comparisons:
        sig = "顯著" if c["p_run"] < 0.05 else "不顯著"
        direction = "優於" if c["d"] > 0 else "劣於"
        L.append(
            f"- vs `{c['b']}`：early fusion {direction} 對照組 "
            f"{abs(c['d']):.4f}，run 層級 {sig}（p = {c['p_run']:.4f}）\n"
        )
    L.append(
        "\n- **determinism 註記**：early fusion 同 seed 位元確定，"
        "故其 SD 純為 seed 效應；MAGNET 的 replicate SD 另含 MPS run-to-run 變異。"
        "兩者的 SD 不同源，不可直接相互解讀。\n"
    )
    L.append(
        "- 若 early fusion 勝出，直接的意涵是：**在 ~1150 個訓練日的規模下，"
        "逐時步的 ADR-TW 互動比兩級潛在空間耦合更有價值，"
        "或 MAGNET 的 166K 參數相對資料量過大。** 兩者都指向同一個結論——"
        "現行 universe 撐不起現行架構。\n"
    )
    L.append(
        "- 無論方向為何，這個比較都**不能**單獨支持或否定兩級耦合的設計價值："
        "k=7 的 benchmark 解析度（ΔIC≈0.02 需約 41 檔）遠不足以歸因。\n"
    )

    Path(args.out_md).write_text("\n".join(L) + "\n")
    print(f"[m10-ef] 寫出 {args.out_md}")
    for arm in ARMS:
        m, s = mean_ic[arm], sharpe[arm]
        print(f"{arm:<14} IC {m.mean():+.4f} ± {m.std(ddof=1):.4f}   "
              f"Sharpe {s.mean():+.3f} ± {s.std(ddof=1):.3f}")
    print(f"Welch t={t_run:+.3f} p={p_run:.4f}")


if __name__ == "__main__":
    main()
