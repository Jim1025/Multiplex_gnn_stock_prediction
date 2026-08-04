"""
m10_portfolio_metrics.py — 第 0 階段：對既有 run 回算 level R² 與組合指標

動機（口試提問）：
  Q6「模型已算出隔日報酬，為何還要算 IC，不直接看損失函數？」
     → 需要證據顯示「數值」不可預測而「次序」可預測：
        level R² ≈ 0 而 IC > 0，且零預測器在 MSE 上很有競爭力卻毫無排序能力。
  跨市場文獻（Gao et al. 2022、Liu et al. 2026）皆報 Sharpe，
  只報統計指標的是我們 → 補上組合讀法。

做法：
  不重訓。直接讀 runs/<slug>/predictions/test_predictions.csv，
  重建每日橫截面後回算 IC / RankIC / R² / R²_zero / Sharpe / hit rate / 逐名次報酬。

  另加一列「零預測器」對照：ŷ ≡ 0。它的 IC 未定義（無排序），
  但 MSE 與真實模型同級——這正是「MSE 不忠實於任務目標」的直接展示。

  以及「隨機預測器」的虛無分布：k=7 時多空各 2 檔的 Sharpe 本身噪音極大，
  不給虛無帶就報 Sharpe 會被誤讀。此處以蒙地卡羅估其分位數。

用法：
    python scripts/m10_portfolio_metrics.py
    python scripts/m10_portfolio_metrics.py --filter opt_p
    python scripts/m10_portfolio_metrics.py --null-draws 5000
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.metrics import (  # noqa: E402
    aggregate_ic,
    long_short_metrics,
    rank_bucket_returns,
    regression_metrics,
)


def load_daily(pred_csv: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """把 predictions CSV 還原成每日橫截面。ticker 順序在各日內固定。"""
    df = pd.read_csv(pred_csv)
    daily_yh: list[np.ndarray] = []
    daily_y:  list[np.ndarray] = []
    dates:    list[str] = []
    for date, grp in df.groupby("target_date", sort=True):
        grp = grp.sort_values("ticker")
        daily_yh.append(grp["y_hat"].to_numpy(dtype=np.float64))
        daily_y.append(grp["y"].to_numpy(dtype=np.float64))
        dates.append(str(date))
    return daily_yh, daily_y, dates


def read_architecture(run_dir: Path) -> str:
    snap = run_dir / "config_snapshot.yaml"
    if not snap.exists():
        return "?"
    try:
        with open(snap) as f:
            cfg = yaml.safe_load(f) or {}
        return str(cfg.get("model", {}).get("architecture", "?"))
    except Exception:
        return "?"


def summarize(
    daily_yh: list[np.ndarray],
    daily_y:  list[np.ndarray],
    n_side: int,
    periods_per_year: int,
) -> dict:
    ic  = aggregate_ic(daily_yh, daily_y)
    reg = regression_metrics(np.concatenate(daily_yh), np.concatenate(daily_y))
    pf  = long_short_metrics(daily_yh, daily_y, n_side, periods_per_year)
    rb  = rank_bucket_returns(daily_yh, daily_y)
    return {
        "n_days":   len(daily_yh),
        "IC":       ic["IC"],
        "RankIC":   ic["RankIC"],
        "ICIR":     ic["ICIR"],
        "RMSE":     reg["RMSE"],
        "R2":       reg["R2"],
        "R2_zero":  reg["R2_zero"],
        "Sharpe":   pf["Sharpe"],
        "hit_rate": pf["hit_rate"],
        "ann_return": pf["ann_return"],
        "by_rank":  rb["by_rank"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--config",   default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--filter",   default="", help="只處理 slug 含此子字串的 run")
    ap.add_argument("--out-md",   default=str(ROOT / "docs" / "m10_portfolio_metrics.md"))
    ap.add_argument("--out-csv",  default=str(ROOT / "docs" / "m10_portfolio_metrics.csv"))
    ap.add_argument("--null-draws", type=int, default=2000,
                    help="隨機預測器蒙地卡羅次數（0 = 跳過）")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    pf_cfg = cfg["evaluation"]["portfolio"]
    n_side = int(pf_cfg["n_side"])
    ppy    = int(pf_cfg["periods_per_year"])

    runs_dir = Path(args.runs_dir)
    rows: list[dict] = []
    ref_daily_y: list[np.ndarray] | None = None

    for pred_csv in sorted(runs_dir.glob("*/predictions/test_predictions.csv")):
        slug = pred_csv.parents[1].name
        if args.filter and args.filter not in slug:
            continue
        daily_yh, daily_y, _ = load_daily(pred_csv)
        if not daily_yh:
            continue
        if ref_daily_y is None:
            ref_daily_y = daily_y
        row = {"run": slug, "arch": read_architecture(pred_csv.parents[1])}
        row.update(summarize(daily_yh, daily_y, n_side, ppy))
        rows.append(row)

    if not rows:
        print("[m10] 找不到任何 test_predictions.csv")
        return

    # 零預測器對照：ŷ ≡ 0（IC 未定義，但 MSE 具競爭力）
    if ref_daily_y is not None:
        zero_yh = [np.zeros_like(a) for a in ref_daily_y]
        row = {"run": "ZERO_PREDICTOR", "arch": "-"}
        row.update(summarize(zero_yh, ref_daily_y, n_side, ppy))
        rows.append(row)

    df = pd.DataFrame(rows)
    df_out = df.drop(columns=["by_rank"])
    df_out = df_out.sort_values("IC", ascending=False, na_position="last")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)

    # ── Markdown 報表 ────────────────────────────────────────────
    finite_ic = df_out["IC"].dropna()
    finite_r2 = df_out["R2"].dropna()
    lines: list[str] = []
    lines.append("# M10 第 0 階段：level R² 與組合指標（對既有 run 回算）\n")
    lines.append(f"> 由 `scripts/m10_portfolio_metrics.py` 自動生成\n")
    lines.append(
        f"> 組合設定：多空各 {n_side} 檔、等權、金額中性、"
        f"年化因子 {ppy}；**無交易成本、無流動性限制，屬 pre-cost 診斷**\n"
    )
    lines.append(f"> 共 {len(df_out) - 1} 個 run + 1 個零預測器對照\n")

    lines.append("\n## 全表\n")
    lines.append(
        "| run | arch | n_days | IC | RankIC | RMSE | R2 | R2_zero | "
        "Sharpe | hit_rate |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df_out.iterrows():
        def f(v, p=4):
            return "n/a" if (isinstance(v, float) and math.isnan(v)) else f"{v:.{p}f}"
        lines.append(
            f"| {r['run']} | {r['arch']} | {int(r['n_days'])} | {f(r['IC'])} | "
            f"{f(r['RankIC'])} | {f(r['RMSE'], 5)} | {f(r['R2'], 5)} | "
            f"{f(r['R2_zero'], 5)} | {f(r['Sharpe'], 3)} | {f(r['hit_rate'], 3)} |"
        )

    # ── 隨機預測器虛無分布 ────────────────────────────────────────
    null_sharpe: np.ndarray | None = None
    if args.null_draws > 0 and ref_daily_y is not None:
        rng = np.random.default_rng(int(cfg["training"]["seed"]))
        draws = []
        for _ in range(args.null_draws):
            rnd = [rng.standard_normal(a.shape) for a in ref_daily_y]
            draws.append(
                long_short_metrics(rnd, ref_daily_y, n_side, ppy)["Sharpe"]
            )
        null_sharpe = np.asarray([d for d in draws if not math.isnan(d)])

    if null_sharpe is not None and null_sharpe.size > 0:
        lines.append("\n## Sharpe 的虛無分布（隨機預測器）\n")
        lines.append(
            f"以 {null_sharpe.size} 次隨機預測（同一組實現報酬、預測值為標準常態）"
            f"估得的 Sharpe 分布：\n"
        )
        lines.append(
            f"- mean = {null_sharpe.mean():+.3f}, "
            f"**std = {null_sharpe.std():.3f}**\n"
        )
        qs = [5, 25, 50, 75, 90, 95, 99]
        cells = ", ".join(
            f"p{q} = {np.percentile(null_sharpe, q):+.3f}" for q in qs
        )
        lines.append(f"- 分位數：{cells}\n")
        lines.append(
            "- **判讀警告**：k=7 時多空各 2 檔的 Sharpe 標準差約 1.0。"
            "任何單一 run 的 Sharpe 若未附此虛無帶即無法判讀；"
            "Sharpe ≈ 1.4 僅落在隨機分布的 90 百分位，不構成證據。\n"
        )
        # 各 run 的虛無分位
        lines.append("\n各 run 的 Sharpe 在虛無分布中的百分位：\n")
        for _, r in df_out.iterrows():
            s = r["Sharpe"]
            if isinstance(s, float) and math.isnan(s):
                continue
            pct = 100.0 * float((null_sharpe < s).mean())
            lines.append(f"- `{r['run']}`: Sharpe {s:+.3f} → 第 {pct:.1f} 百分位")

    lines.append("\n## 判讀\n")
    if len(finite_r2) > 0:
        lines.append(
            f"- level R² 全距 [{finite_r2.min():.5f}, {finite_r2.max():.5f}]，"
            f"中位數 {finite_r2.median():.5f}\n"
        )
    if len(finite_ic) > 0:
        lines.append(
            f"- IC 全距 [{finite_ic.min():.4f}, {finite_ic.max():.4f}]，"
            f"中位數 {finite_ic.median():.4f}\n"
        )
    lines.append(
        "- 零預測器（ŷ ≡ 0）的 IC 未定義——它完全沒有排序能力；"
        "但其 RMSE 與多數模型同級。單看損失函數無法區分「會排序」與「不會排序」的模型。\n"
    )
    lines.append(
        "- 兩個量測的是不同的東西：level R² 衡量報酬「數值」的可解釋變異，"
        "IC 衡量「次序」。前者 ≈ 0（實測全為負）而後者 > 0，"
        "正是選用 IC 而非 MSE 的實證理由。\n"
    )
    lines.append(
        "- Sharpe 提供同一組預測的經濟讀法，但其虛無帶（std ≈ 1.0）"
        "與 IC 的噪音下界同源：k=7 的橫截面太窄。"
        "組合指標並未繞過解析度問題，只是換一個尺度呈現同一個限制。\n"
    )

    lines.append("\n## 逐名次平均實現報酬\n")
    lines.append("名次 0 = 當日預測最高。訊號若真實，應呈單調遞減。\n")
    for _, r in df.iterrows():
        br = r["by_rank"]
        cells = " | ".join("n/a" if math.isnan(v) else f"{v:+.5f}" for v in br)
        lines.append(f"- `{r['run']}`: {cells}")

    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"[m10] 寫出 {args.out_md}")
    print(f"[m10] 寫出 {args.out_csv}")
    print(df_out.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
