"""
freeze_k7.py — E0：把 k=7 的結果凍結成不可變 artifact，並提供事後驗證

為什麼需要：
  第 1 階段（universe 擴充）會改變節點順序與張量形狀，
  現有 graph snapshot 與所有既有 run 的可比性都會失效。
  重構之後必須能證明「模型語意沒有被偷偷改變」，
  唯一的方法是先把現況固定成可機器驗證的基準。

驗收機制（E7 用）：
  - baseline_lstm 與 baseline_early_fusion 走純 LSTM 路徑，同 seed **位元確定**
    → 重構後把 universe 設回 7 對重跑，predictions CSV 的 SHA-256 必須完全相同
  - MAGNET 走 GNN scatter 路徑，MPS 下非位元確定
    → 只能比對 test_IC 是否落在 M9 實測的 run 層級噪音帶（sigma ≈ 0.027）內

用法：
    .venv/bin/python scripts/freeze_k7.py --emit      # 產生 manifest（只做一次）
    .venv/bin/python scripts/freeze_k7.py --verify    # 重構後驗證
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "docs" / "frozen_k7_manifest.json"

# 位元確定（純 LSTM 路徑）——可做 SHA-256 逐位元比對
DETERMINISTIC_RUNS = [
    "opt_p70_ef_r1", "opt_p70_ef_s7", "opt_p70_ef_s123",
    "opt_p70_ef_s2026", "opt_p70_ef_s314",
    "opt_p71_lstm_s42", "opt_p71_lstm_s7", "opt_p71_lstm_s123",
    "opt_p71_lstm_s2026", "opt_p71_lstm_s314",
]

# 非位元確定（GNN scatter 路徑）——只能比對統計量
STOCHASTIC_RUNS = [
    "opt_p46_raw_lr5e4_s42", "opt_p47_raw_lr5e4_s7", "opt_p48_raw_lr5e4_s123",
    "opt_p66_fig1_lr5e4", "opt_p67_rep_s42_a",
    "opt_p68_rep_s42_b", "opt_p69_rep_s42_c",
]

# M9 實測的 run 層級標準差，作為隨機路徑的比對容許帶
SIGMA_RUN = 0.027


def pred_path(tag: str) -> Path:
    hits = sorted((ROOT / "runs").glob(f"**/*{tag}/predictions/test_predictions.csv"))
    if not hits:
        raise FileNotFoundError(f"找不到 run：{tag}")
    return hits[-1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def mean_ic(path: Path) -> float:
    df = pd.read_csv(path)
    ics = []
    for _, g in df.groupby("target_date"):
        if g["y_hat"].std() > 0 and g["y"].std() > 0:
            ics.append(float(np.corrcoef(g["y_hat"], g["y"])[0, 1]))
    return float(np.mean(ics)) if ics else float("nan")


def git_ref() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def build_manifest() -> dict:
    from src.dataset.config import PAIR_MAP
    from src.dataset.multiplex_dataset import ADR_TICKERS, TW_CODES

    with open(ROOT / "configs" / "base.yaml") as f:
        cfg = yaml.safe_load(f)

    ref = pd.read_csv(pred_path(DETERMINISTIC_RUNS[0]))
    dates = sorted(ref["target_date"].unique())

    m: dict = {
        "created":  str(date.today()),
        "git_ref":  git_ref(),
        "purpose":  "E0 freeze of the k=7 universe prior to TW50 expansion",
        "universe": {
            "n_l1": len(ADR_TICKERS),
            "n_l2": len(TW_CODES),
            "adr_tickers": list(ADR_TICKERS),
            "tw_codes":    [str(c) for c in TW_CODES],
            "pairing":     {k: v["tw"] for k, v in PAIR_MAP.items()},
            "pairing_rate": 1.0,
        },
        "protocol": {
            "split":            cfg["data"]["split"],
            "T_history":        cfg["model"]["lstm"]["T_history"],
            "F":                cfg["model"]["lstm"]["input_dim"],
            "seed":             cfg["training"]["seed"],
            "early_stop_metric": cfg["training"]["early_stop_metric"],
            "test_start":       str(dates[0]),
            "test_end":         str(dates[-1]),
            "n_test_days":      len(dates),
        },
        "counts": {
            "runs":      len([d for d in (ROOT / "runs").glob("**/") if (d / "meta.json").exists()]),
            "snapshots": len(list((ROOT / "data" / "graphs" / "snapshots").glob("*.pt"))),
        },
        "noise_floor": {
            "k": 7,
            "measured_sigma_daily_IC": 0.404,
            "theory_1_over_sqrt_k_minus_1": 0.408,
            "sigma_run": SIGMA_RUN,
            "k_needed_for_dIC_0.02": 41,
            "k_needed_for_dIC_0.01": 158,
        },
        "deterministic_runs": {},
        "stochastic_runs": {},
    }

    for tag in DETERMINISTIC_RUNS:
        p = pred_path(tag)
        m["deterministic_runs"][tag] = {
            "sha256":  sha256(p),
            "test_IC": round(mean_ic(p), 6),
        }
    for tag in STOCHASTIC_RUNS:
        p = pred_path(tag)
        m["stochastic_runs"][tag] = {"test_IC": round(mean_ic(p), 6)}

    return m


def verify(m: dict) -> int:
    fails = 0
    print(f"[freeze] manifest 建立於 {m['created']}  git_ref={m['git_ref'][:8]}")
    print(f"[freeze] universe n_l1={m['universe']['n_l1']} "
          f"n_l2={m['universe']['n_l2']} pairing_rate={m['universe']['pairing_rate']}")

    print("\n-- 位元確定路徑（SHA-256 必須完全相同）--")
    for tag, rec in m["deterministic_runs"].items():
        try:
            got = sha256(pred_path(tag))
        except FileNotFoundError as e:
            print(f"  MISSING  {tag}  ({e})")
            fails += 1
            continue
        ok = got == rec["sha256"]
        print(f"  {'OK  ' if ok else 'FAIL'}  {tag}  "
              f"{rec['sha256'][:16]} vs {got[:16]}")
        fails += 0 if ok else 1

    print(f"\n-- 隨機路徑（test_IC 需落在 ±3 sigma_run = ±{3*SIGMA_RUN:.3f}）--")
    for tag, rec in m["stochastic_runs"].items():
        try:
            got = mean_ic(pred_path(tag))
        except FileNotFoundError as e:
            print(f"  MISSING  {tag}  ({e})")
            fails += 1
            continue
        d = abs(got - rec["test_IC"])
        ok = d <= 3 * SIGMA_RUN
        print(f"  {'OK  ' if ok else 'FAIL'}  {tag}  "
              f"{rec['test_IC']:+.4f} vs {got:+.4f}  (|d|={d:.4f})")
        fails += 0 if ok else 1

    print(f"\n[freeze] {'全部通過' if fails == 0 else f'{fails} 項未通過'}")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit",   action="store_true", help="產生 manifest")
    ap.add_argument("--verify", action="store_true", help="對照 manifest 驗證")
    args = ap.parse_args()

    if args.emit:
        m = build_manifest()
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
        print(f"[freeze] 寫出 {MANIFEST}")
        print(f"[freeze] 位元確定 run {len(m['deterministic_runs'])} 個、"
              f"隨機 run {len(m['stochastic_runs'])} 個")
        return

    if args.verify:
        if not MANIFEST.exists():
            print(f"[freeze] 找不到 {MANIFEST}，請先 --emit")
            sys.exit(1)
        sys.exit(1 if verify(json.loads(MANIFEST.read_text())) else 0)

    ap.print_help()


if __name__ == "__main__":
    main()
