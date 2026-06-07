"""
migrate_runs.py — 一次性把現有 hash-based artifacts 整理進 runs/<slug>/ 佈局
Corresponds to plan §"Run Naming Refactor"

對每個 MLflow run：
  1. 從 start_time + tag 算 slug = YYYYMMDD_HHMM_<tag>（衝突時加 _2、_3）
  2. 建立 runs/<slug>/{checkpoints,predictions,figures}/
  3. 搬移 checkpoints/<run_id>/{best,final}.pt
  4. 搬移 predictions/<run_id>_{test,val}_predictions.csv → 改名為 {test,val}_predictions.csv
  5. 搬移 figures/<run_id>/* （若存在）
  6. 從 mlruns params 重建 config_snapshot.yaml
  7. 寫 meta.json
  8. 重建 / 更新 runs/INDEX.csv（按 start_time 排序）

安全：
  - 不刪除 mlruns/，不動 MLflow 內部 artifacts
  - 用 shutil.move（同分割區 rename，極快）
  - 對已遷移的 slug → skip（idempotent）
  - 對搬不到的檔案只警告
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import yaml
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT    = PROJECT_ROOT / "runs"
OLD_CKPT     = PROJECT_ROOT / "checkpoints"
OLD_PRED     = PROJECT_ROOT / "predictions"
OLD_FIG      = PROJECT_ROOT / "figures"
TRACKING_URI = "file:./mlruns"
EXPERIMENT   = "MAGNET_M4_baseline"

INDEX_COLS = [
    "slug", "run_id", "tag", "start_time", "end_time", "status",
    "n_epochs", "best_epoch", "best_val_IC",
    "test_IC", "test_RankIC", "test_ICIR", "test_MSE",
]


# ---------------------------------------------------------------------------
# Slug 計算
# ---------------------------------------------------------------------------

def make_slug(start_time_ms: int, tag: str, taken: set[str]) -> str:
    """{YYYYMMDD}_{HHMM}_{tag}[_${n}]，避開 taken set 中已存在的。"""
    dt = datetime.fromtimestamp(start_time_ms / 1000.0)
    base = f"{dt.strftime('%Y%m%d_%H%M')}_{tag}"
    slug = base
    i = 2
    while slug in taken:
        slug = f"{base}_{i}"
        i += 1
    return slug


# ---------------------------------------------------------------------------
# Config snapshot 重建（從 MLflow params 還原 base.yaml 三大區塊）
# ---------------------------------------------------------------------------

def _set_nested(d: dict, dotted_key: str, val) -> None:
    """把 'a.b.c' 寫進巢狀 dict。"""
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = val


def _coerce(s: str):
    """MLflow 全部以字串存 params；嘗試還原型別。"""
    if s in ("True", "true"):  return True
    if s in ("False", "false"): return False
    if s == "None":            return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def build_config_snapshot(params: dict) -> dict:
    """從 flatten 的 MLflow params 還原成巢狀 dict。"""
    cfg: dict = {}
    for k, v in params.items():
        if k in ("tag",):
            continue
        _set_nested(cfg, k, _coerce(v))
    return cfg


# ---------------------------------------------------------------------------
# 檔案搬移
# ---------------------------------------------------------------------------

def _move_if_exists(src: Path, dst: Path) -> bool:
    """搬不到回 False（只警告）。"""
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return True


def migrate_one_run(run, taken_slugs: set[str]) -> Optional[dict]:
    """搬移一個 run 的所有 artifacts，回傳 INDEX.csv 用的列 dict。"""
    run_id = run.info.run_id
    tag    = run.data.tags.get("mlflow.runName", "untagged")
    start_ms = run.info.start_time or 0
    end_ms   = run.info.end_time   or 0

    slug = make_slug(start_ms, tag, taken_slugs)
    taken_slugs.add(slug)

    run_dir = RUNS_ROOT / slug
    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"  ⏭  {slug} 已存在，skip（idempotent）")
        return _row_from_run(run, slug)

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    n_moved = 0

    # ── checkpoints
    for name in ["best.pt", "final.pt"]:
        if _move_if_exists(OLD_CKPT / run_id / name, run_dir / "checkpoints" / name):
            n_moved += 1
    # 空了的舊資料夾清掉
    old_ckpt_dir = OLD_CKPT / run_id
    if old_ckpt_dir.exists() and not any(old_ckpt_dir.iterdir()):
        old_ckpt_dir.rmdir()

    # ── predictions（檔名改成 test/val_predictions.csv）
    for kind in ["test", "val"]:
        src = OLD_PRED / f"{run_id}_{kind}_predictions.csv"
        dst = run_dir / "predictions" / f"{kind}_predictions.csv"
        if _move_if_exists(src, dst):
            n_moved += 1

    # ── figures
    old_fig_dir = OLD_FIG / run_id
    if old_fig_dir.exists():
        for p in old_fig_dir.iterdir():
            if p.is_file():
                shutil.move(str(p), str(run_dir / "figures" / p.name))
                n_moved += 1
        if not any(old_fig_dir.iterdir()):
            old_fig_dir.rmdir()

    # ── config_snapshot.yaml（從 params 還原）
    cfg_snap = build_config_snapshot(run.data.params)
    with open(run_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg_snap, f, sort_keys=False, allow_unicode=True)

    # ── meta.json
    meta = {
        "slug":       slug,
        "run_id":     run_id,
        "tag":        tag,
        "start_time": _iso(start_ms),
        "end_time":   _iso(end_ms),
        "status":     run.info.status,
        "metrics":    {k: float(v) for k, v in run.data.metrics.items()},
        "migrated_from_hash_layout": True,
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  ✓ Migrated {slug}  (run_id={run_id[:8]}..., {n_moved} files moved)")
    return _row_from_run(run, slug)


def _iso(ms: int) -> str:
    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000.0).isoformat(timespec="seconds")


def _row_from_run(run, slug: str) -> dict:
    m = run.data.metrics
    return {
        "slug":          slug,
        "run_id":        run.info.run_id,
        "tag":           run.data.tags.get("mlflow.runName", "untagged"),
        "start_time":    _iso(run.info.start_time or 0),
        "end_time":      _iso(run.info.end_time   or 0),
        "status":        run.info.status,
        "n_epochs":      int(m.get("best_epoch", -1)) + 1 if "best_epoch" in m else "",
        "best_epoch":    int(m["best_epoch"])     if "best_epoch"  in m else "",
        "best_val_IC":   round(m["best_val_IC"], 6) if "best_val_IC" in m else "",
        "test_IC":       round(m["test/IC"],     6) if "test/IC"     in m else "",
        "test_RankIC":   round(m["test/RankIC"], 6) if "test/RankIC" in m else "",
        "test_ICIR":     round(m["test/ICIR"],   6) if "test/ICIR"   in m else "",
        "test_MSE":      round(m["test/MSE"],    8) if "test/MSE"    in m else "",
    }


# ---------------------------------------------------------------------------
# INDEX.csv
# ---------------------------------------------------------------------------

def write_index(rows: list[dict]) -> None:
    """按 start_time ASC 排序後重寫 INDEX.csv。"""
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["start_time"] or "")
    with open(RUNS_ROOT / "INDEX.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    os.chdir(PROJECT_ROOT)
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        print(f"❌ 找不到 experiment '{EXPERIMENT}'", file=sys.stderr)
        return 1

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time ASC"],
        max_results=1000,
    )

    print("─" * 60)
    print(f"📦 MAGNET 訓練紀錄遷移（hash → slug）")
    print(f"  experiment : {EXPERIMENT}")
    print(f"  runs       : {len(runs)}")
    print(f"  runs root  : {RUNS_ROOT}")
    print("─" * 60)

    if not runs:
        print("⚠️  沒有任何 run 可遷移")
        return 0

    # 預掃描已存在的 slug 防衝突
    taken: set[str] = set()
    if RUNS_ROOT.exists():
        for p in RUNS_ROOT.iterdir():
            if p.is_dir() and p.name != "comparison":
                taken.add(p.name)

    rows = []
    for run in runs:
        row = migrate_one_run(run, taken)
        if row:
            rows.append(row)

    write_index(rows)
    print("─" * 60)
    print(f"✓ INDEX.csv built with {len(rows)} rows → {RUNS_ROOT / 'INDEX.csv'}")
    print("─" * 60)

    # 摘要表
    print("\n📋 摘要")
    fmt = "{:<28} {:<8} {:<10} {:>10} {:>10}"
    print(fmt.format("slug", "tag", "status", "best_val_IC", "test_IC"))
    for r in rows:
        print(fmt.format(
            r["slug"], r["tag"][:8], r["status"][:10],
            str(r["best_val_IC"]) if r["best_val_IC"] != "" else "—",
            str(r["test_IC"])     if r["test_IC"]     != "" else "—",
        ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
