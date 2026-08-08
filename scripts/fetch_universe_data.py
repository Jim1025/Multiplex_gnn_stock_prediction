"""
fetch_universe_data.py — E2：依 universe 定義下載尚未取得的原始 OHLCV

設計要點：
  - **絕不覆寫既有檔案**。k=7 的 14 檔原始資料是 E0 凍結基準的一部分，
    重新下載會改變股利調整時點、破壞 E7 回歸測試。既有檔一律跳過。
  - 下載參數已用既有資料反推驗證：auto_adjust=True 可重現 TSM / 2330
    （價格層級最大相對誤差 0.23%，來自下載後才發放的股利；
    log return 最大絕對差 8.9e-07、相關 1.0000000000，故混用下載時點安全）
  - 欄位順序與既有檔一致：Date, Close, High, Low, Open, Volume

用法：
    .venv/bin/python scripts/fetch_universe_data.py --dry-run
    .venv/bin/python scripts/fetch_universe_data.py
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "configs" / "universe" / "universe_tw_2019.json"
COLS = ["Close", "High", "Low", "Open", "Volume"]

START, END = "2019-01-01", "2026-01-01"


def plan(universe: dict) -> list[tuple[str, str, Path]]:
    """回傳 (yfinance symbol, 檔名, 目標路徑)，已排除既有檔。"""
    jobs: list[tuple[str, str, Path]] = []

    tw_dir = ROOT / "data" / "raw" / "tw"
    for node in universe["tw_nodes"]:
        code = node["code"]
        dest = tw_dir / f"{code}.csv"
        if not dest.exists():
            jobs.append((f"{code}.TW", code, dest))

    us_dir = ROOT / "data" / "raw" / "adr"
    us = ([a["ticker"] for a in universe["us_layer"]["adr_pairs"]]
          + [i["ticker"] for i in universe["us_layer"]["info_sources"]])
    for t in us:
        dest = us_dir / f"{t}.csv"
        if not dest.exists():
            jobs.append((t, t, dest))

    return jobs


def fetch(symbol: str, retries: int = 3, pause: float = 2.0) -> pd.DataFrame | None:
    import yfinance as yf
    for attempt in range(retries):
        try:
            df = yf.download(symbol, start=START, end=END, auto_adjust=True,
                             progress=False, threads=False)
        except Exception as e:
            print(f"      attempt {attempt + 1} 例外：{e}")
            df = None
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            missing = [c for c in COLS if c not in df.columns]
            if missing:
                print(f"      缺欄位 {missing}")
                return None
            out = df[COLS].copy()
            out.index.name = "Date"
            return out.sort_index()
        time.sleep(pause * (attempt + 1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=str(UNIVERSE))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pause", type=float, default=0.6,
                    help="每檔之間的間隔秒數（避免被限流）")
    args = ap.parse_args()

    universe = json.loads(Path(args.universe).read_text())
    jobs = plan(universe)

    print(f"[E2] universe：TW {universe['n_tw_nodes']} 檔、"
          f"US {universe['us_layer']['n_us_nodes']} 檔")
    print(f"[E2] 待下載 {len(jobs)} 檔（既有檔一律跳過，不覆寫）\n")

    if args.dry_run:
        for sym, name, dest in jobs:
            print(f"  {sym:<10} → {dest.relative_to(ROOT)}")
        return

    ok, failed = [], []
    for i, (sym, name, dest) in enumerate(jobs, 1):
        df = fetch(sym)
        if df is None:
            print(f"  [{i:>2}/{len(jobs)}] {sym:<10} 失敗")
            failed.append(sym)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest)
        span = f"{df.index.min().date()} → {df.index.max().date()}"
        print(f"  [{i:>2}/{len(jobs)}] {sym:<10} rows={len(df):<5} {span}")
        ok.append((sym, len(df), str(df.index.min().date())))
        time.sleep(args.pause)

    print(f"\n[E2] 成功 {len(ok)} 檔、失敗 {len(failed)} 檔")
    if failed:
        print(f"[E2] 失敗清單（需從 universe 移除或改用替代標的）：{failed}")

    # 起始日一致性檢查：晚於 2019-01-31 者代表 2019 年資料不完整
    late = [(s, d) for s, _, d in ok if d > "2019-01-31"]
    if late:
        print(f"\n[E2] 警告：以下標的起始日晚於 2019-01-31，"
              f"2019 年交易紀錄不完整，須依 provisional 條款重新檢視：")
        for s, d in late:
            print(f"      {s:<10} 起始 {d}")


if __name__ == "__main__":
    main()
