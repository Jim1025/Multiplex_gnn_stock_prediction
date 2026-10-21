"""
build_holdout_2026.py — 2026 保留期的資料（proposal §75）

使用者 2026-10-18 授權動用 2026 年資料，並同意下載：80 檔（美股 30、台股 50）的日線 OHLCV，
2025-10-01 ~ 2026-10-16，來源 Yahoo Finance（yfinance，auto_adjust=True），約 2 MB。

**不改任何資料層程式（pipeline / features / graph_builder），不覆寫既有資料**：
所有輸出都在 data/holdout_2026/，既有的 data/raw、data/processed、data/features、data/graphs 只讀。

步驟
  --download  下載到 data/holdout_2026/download/{adr,tw}/。參數與 fetch_universe_data.py 相同；
              已存在的檔不重下。每個檔的 SHA-256 寫進報告。
  --build     1. 接合：既有 data/raw 的檔案內容逐位元組保留（到該檔的最後一天：2025-12-31，k=7 的 14 檔是
                 2025-12-30），後面接下載檔裡那天之後的列。下載時點不同，股利調整的基準不同，
                 所以下載檔的 OHLC 先乘上一個常數，讓接合日的收盤與既有檔相同；常數倍不改報酬，成交量不調整。
                 重疊期（2025-10-01 起到接合日）兩邊的 log return 最大絕對差須 <= 1e-4，否則停止。
              2. 用既有的資料層類別產生 processed、features 與圖快照（2019-01-01 ~ 2026-10-16），
                 設定與原本的建置相同（DataPipeline()、FeatureBuilder() 的預設；圖：tw50、corr_window 60、門檻 0.3）。
  --verify    2025-12-30 以前的快照：兩層節點的 log_return 特徵與補值旗標、所有邊與邊權、y 必須與
              data/graphs/snapshots_tw50 完全相同，否則停止。成交量類特徵會不同
              （pipeline 的成交量 IQR clip 用全期分位數），只記錄差異；
              要比較的兩個 arm 都只吃 log_return（feature_subset），不受影響。
              另列出 2026 年被補值（is_imputed）的日期，供檢查交易日曆。

報告：data/holdout_2026/build_report.json（不進 git，data/ 已在 .gitignore）。

用法：
    .venv/bin/python scripts/build_holdout_2026.py --download
    .venv/bin/python scripts/build_holdout_2026.py --build
    .venv/bin/python scripts/build_holdout_2026.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UNIVERSE = ROOT / "configs" / "universe" / "universe_tw_2019.json"
HOLD = ROOT / "data" / "holdout_2026"
COLS = ["Close", "High", "Low", "Open", "Volume"]
DL_START, DL_END = "2025-10-01", "2026-10-17"        # yfinance 的 end 不含當天：資料到 2026-10-16
# 既有 data/raw 的最後一天：66 檔是 2025-12-31，k=7 凍結基準的 14 檔（7 台股 + 7 美股）是 2025-12-30。
# 每檔以自己的最後一天當接合點（2026-10-18 修正，§75：原本假設全部是 12-31，接合時就停了）。
LAST_OLD_MIN = "2025-12-30"
LAST_OLD_SNAPSHOT = "2025-12-30"                     # 既有 snapshots_tw50 的最後一個目標日
BUILD_END = "2026-10-16"
OVERLAP_TOL = 1e-4
REPORT = HOLD / "build_report.json"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tickers() -> list[tuple[str, str, str]]:
    """(market, yfinance symbol, 檔名)，與 fetch_universe_data.plan() 同一份清單。"""
    u = json.loads(UNIVERSE.read_text())
    out = [("tw", f"{n['code']}.TW", n["code"]) for n in u["tw_nodes"]]
    us = [a["ticker"] for a in u["us_layer"]["adr_pairs"]] + [i["ticker"] for i in u["us_layer"]["info_sources"]]
    out += [("adr", t, t) for t in us]
    return out


def load_report() -> dict:
    return json.loads(REPORT.read_text()) if REPORT.exists() else {}


def save_report(rep: dict) -> None:
    HOLD.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------- 下載
def fetch(symbol: str, retries: int = 3, pause: float = 2.0) -> pd.DataFrame | None:
    import yfinance as yf
    for attempt in range(retries):
        try:
            df = yf.download(symbol, start=DL_START, end=DL_END, auto_adjust=True,
                             progress=False, threads=False)
        except Exception as e:                                   # noqa: BLE001
            print(f"      attempt {attempt + 1} 例外：{e}")
            df = None
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if any(c not in df.columns for c in COLS):
                return None
            out = df[COLS].copy()
            out.index.name = "Date"
            return out.sort_index()
        time.sleep(pause * (attempt + 1))
    return None


def download() -> None:
    rep = load_report()
    files = rep.setdefault("download", {})
    for i, (mkt, sym, name) in enumerate(tickers(), 1):
        dest = HOLD / "download" / mkt / f"{name}.csv"
        if dest.exists():
            print(f"  [{i:>2}/80] {sym:<9} 已存在，跳過")
            continue
        df = fetch(sym)
        if df is None:
            raise SystemExit(f"下載失敗：{sym}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest)
        files[f"{mkt}/{name}.csv"] = dict(rows=len(df), first=str(df.index.min().date()),
                                          last=str(df.index.max().date()), sha256=sha256(dest))
        print(f"  [{i:>2}/80] {sym:<9} rows={len(df):<4} {df.index.min().date()} -> {df.index.max().date()}")
        time.sleep(0.6)
    rep["download_finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_report(rep)
    total = sum((HOLD / "download" / k).stat().st_size for k in files)
    print(f"\n下載完成：{len(files)} 檔，合計 {total / 1e6:.2f} MB")


# ---------------------------------------------------------------- 接合與建置
def splice() -> dict:
    """既有 raw 的位元組 + 調整基準後的 2026 列 -> data/holdout_2026/raw/。回傳每檔的檢查值。"""
    out = {}
    for mkt, sym, name in tickers():
        old_p = ROOT / "data" / "raw" / mkt / f"{name}.csv"
        new_p = HOLD / "download" / mkt / f"{name}.csv"
        old = pd.read_csv(old_p, index_col="Date", parse_dates=True)
        new = pd.read_csv(new_p, index_col="Date", parse_dates=True)
        last_old = old.index.max()
        if last_old < pd.Timestamp(LAST_OLD_MIN):
            raise SystemExit(f"{old_p} 的最後一天 {last_old.date()} 早於 {LAST_OLD_MIN}")
        common = old.index.intersection(new.index)
        common = common[common >= pd.Timestamp(DL_START)]
        if last_old not in common:
            raise SystemExit(f"{name}：下載檔沒有 {last_old.date()}，無法對齊基準")
        f = float(old.loc[last_old, "Close"] / new.loc[last_old, "Close"])
        lr_old = np.log(old.loc[common, "Close"]).diff().dropna()
        lr_new = np.log(new.loc[common, "Close"]).diff().dropna()
        dmax = float((lr_old - lr_new).abs().max())
        vol_eq = float((old.loc[common, "Volume"] == new.loc[common, "Volume"]).mean())
        if dmax > OVERLAP_TOL:
            raise SystemExit(f"{name}：重疊期 log return 最大差 {dmax:.2e} > {OVERLAP_TOL}，停止")
        add = new[new.index > last_old].copy()
        for c in ("Close", "High", "Low", "Open"):
            add[c] = add[c] * f
        dest = HOLD / "raw" / mkt / f"{name}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        text = old_p.read_text()
        if not text.endswith("\n"):
            text += "\n"
        dest.write_text(text + add.to_csv(header=False))
        out[f"{mkt}/{name}"] = dict(last_old=str(last_old.date()), scale=f,
                                    overlap_days=int(len(common)), overlap_lr_maxdiff=dmax,
                                    overlap_volume_equal=vol_eq, rows_2026=int(len(add)),
                                    first_2026=str(add.index.min().date()) if len(add) else None,
                                    last_2026=str(add.index.max().date()) if len(add) else None)
    return out


def build() -> None:
    from src.dataset.config import load_universe
    from src.dataset.features import FeatureBuilder
    from src.dataset.graph_builder import MultiplexGraphBuilder
    from src.dataset.pipeline import DataPipeline

    rep = load_report()
    rep["splice"] = splice()
    sc = rep["splice"]
    print(f"接合：{len(sc)} 檔；重疊期 log return 最大差 {max(v['overlap_lr_maxdiff'] for v in sc.values()):.2e}；"
          f"2026 年列數 {min(v['rows_2026'] for v in sc.values())} ~ {max(v['rows_2026'] for v in sc.values())}")
    pipe = DataPipeline()
    for mkt in ("adr", "tw"):
        pipe.run_batch(str(HOLD / "raw" / mkt), str(HOLD / "processed" / mkt))
    fb = FeatureBuilder()
    for mkt in ("adr", "tw"):
        fb.run_batch(str(HOLD / "processed" / mkt), str(HOLD / "features" / mkt))
    u = load_universe("tw50")
    gb = MultiplexGraphBuilder(pair_map=u.pairing, adr_dir=str(HOLD / "features" / "adr"),
                               tw_dir=str(HOLD / "features" / "tw"), corr_window=60,
                               corr_threshold=0.3, universe=u)
    gb.build_sequence(pd.Timestamp("2019-01-01"), pd.Timestamp(BUILD_END),
                      out_dir=str(HOLD / "graphs" / "snapshots_tw50"))
    rep["build_finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_report(rep)


# ---------------------------------------------------------------- 驗證
def _edges(snap) -> dict:
    out = {}
    for et in snap.edge_types:
        st = snap[et]
        out[str(et)] = (st.edge_index, getattr(st, "edge_attr", None))
    return out


def verify() -> None:
    from src.dataset.features import TECH_FEATURE_COLS
    ret = TECH_FEATURE_COLS.index("log_return")
    old_dir = ROOT / "data" / "graphs" / "snapshots_tw50"
    new_dir = HOLD / "graphs" / "snapshots_tw50"
    old_files = sorted(old_dir.glob("graph_*.pt"))
    bad, other_maxdiff = [], {}
    for f in old_files:
        g = new_dir / f.name
        if not g.exists():
            bad.append((f.name, "缺檔")); continue
        a, b = torch.load(f, weights_only=False), torch.load(g, weights_only=False)
        if a.target_date != b.target_date:
            bad.append((f.name, "target_date")); continue
        for nt in ("adr", "tw"):
            xa, xb = a[nt].x, b[nt].x
            if xa.shape != xb.shape or not torch.equal(xa[:, ret], xb[:, ret]):
                bad.append((f.name, f"{nt}.x[log_return]"))
            for flag in ("is_imputed", "is_long_gap"):
                if not torch.equal(a[nt][flag], b[nt][flag]):
                    bad.append((f.name, f"{nt}.{flag}"))
            for j in range(xa.shape[1]):
                if j != ret and xa.shape == xb.shape:
                    d = float((xa[:, j] - xb[:, j]).abs().max())
                    k = TECH_FEATURE_COLS[j]
                    other_maxdiff[k] = max(other_maxdiff.get(k, 0.0), d)
        if not torch.equal(a["tw"].y, b["tw"].y):
            bad.append((f.name, "tw.y"))
        ea, eb = _edges(a), _edges(b)
        if ea.keys() != eb.keys():
            bad.append((f.name, "edge types")); continue
        for k in ea:
            (ia, wa), (ib, wb) = ea[k], eb[k]
            if not torch.equal(ia, ib) or ((wa is None) != (wb is None)) or (wa is not None and not torch.equal(wa, wb)):
                bad.append((f.name, f"edges {k}"))
    new_files = sorted(new_dir.glob("graph_*.pt"))
    n_2026 = sum(1 for f in new_files if f.stem.split("_")[1] >= "2026-01-01")
    imputed = {}
    for mkt in ("adr", "tw"):
        days = set()
        for p in sorted((HOLD / "processed" / mkt).glob("*.csv")):
            if p.name == "pipeline_report.csv":
                continue
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            if "is_imputed" in df.columns:
                d = df[(df.index >= "2026-01-01") & (df["is_imputed"].astype(bool))].index
                days.update(str(x.date()) for x in d)
        imputed[mkt] = sorted(days)
    rep = load_report()
    rep["verify"] = dict(old_snapshots=len(old_files), mismatches=len(bad), mismatch_examples=bad[:10],
                         other_feature_maxdiff=other_maxdiff, new_snapshots=len(new_files),
                         snapshots_2026=n_2026,
                         first_2026=next((f.stem.split("_")[1] for f in new_files if f.stem.split("_")[1] >= "2026-01-01"), None),
                         last_2026=new_files[-1].stem.split("_")[1] if new_files else None,
                         imputed_days_2026=imputed,
                         finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    save_report(rep)
    print(json.dumps(rep["verify"], indent=2, ensure_ascii=False))
    if bad:
        raise SystemExit(f"驗證失敗：{len(bad)} 處不同（log_return / 邊 / y 必須完全相同）")
    print("\n驗證通過：2025-12-30 以前的快照，log_return、邊與 y 完全相同")


def main() -> None:
    ap = argparse.ArgumentParser(description="2026 保留期的資料（proposal §75）")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.download:
        download()
    if a.build:
        build()
    if a.verify:
        verify()


if __name__ == "__main__":
    main()
