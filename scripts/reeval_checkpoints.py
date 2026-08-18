"""
reeval_checkpoints.py — 在確定性 device 上重評既有 run 的 best.pt

為什麼需要：
    2026-08-16 之前的 run 把 metric 算在 MPS 上。GATv2Conv 的鄰居聚合走
    scatter-add，MPS 後端的浮點加總順序 run 之間不固定，因此 meta.json 與
    predictions/test_predictions.csv 記的是「一次抽樣」，連用同一份 best.pt
    在同一台機器上都重現不了。實測同一份 best.pt 在 MPS 上連評 5 次
    test IC = +0.0466 / +0.0362 / +0.0318 / +0.0359 / +0.0297（全距 0.0169），
    CPU 則是 5 次位元相同的 +0.058480。

    本腳本用 CPU 重評，產生可重現的權威數字。權重本身沒問題——
    save/load 是位元正確的，壞的只有記錄 metric 時所在的 device。

輸出：
    runs/<slug>/meta_reeval.json     每個 run 的重評結果（不覆寫 meta.json）
    docs/reeval_summary.csv          逐 arm 匯總

執行：
    .venv/bin/python scripts/reeval_checkpoints.py --tags tw50_T1F3bnl1
    .venv/bin/python scripts/reeval_checkpoints.py --all-tw50
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.multiplex_dataset import MultiplexDataset, multiplex_collate
from src.models import build_model
from src.train.evaluator import evaluate
from src.train.utils import load_checkpoint

_DS_CACHE: dict = {}


def _loader(cfg: dict, cfg_path: str, split: str, batch_size: int) -> DataLoader:
    key = (cfg["data"]["snapshot_dir"], cfg["data"]["features_dir"],
           cfg["model"]["lstm"]["T_history"], split, cfg_path)
    if key not in _DS_CACHE:
        _DS_CACHE[key] = MultiplexDataset(
            snapshot_dir=str(ROOT / cfg["data"]["snapshot_dir"]),
            features_dir=str(ROOT / cfg["data"]["features_dir"]),
            T=cfg["model"]["lstm"]["T_history"],
            split=split, config_path=cfg_path,
        )
    return DataLoader(_DS_CACHE[key], batch_size=batch_size, shuffle=False,
                      collate_fn=multiplex_collate, num_workers=0)


def reeval_run(run_dir: Path, device: torch.device) -> dict | None:
    ckpt = run_dir / "checkpoints" / "best.pt"
    meta_path = run_dir / "meta.json"
    if not ckpt.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    cfg_path = str(run_dir / "config_snapshot.yaml")
    cfg = yaml.safe_load(open(cfg_path))

    model = build_model(cfg).to(device)
    load_checkpoint(ckpt, model, optimizer=None, map_location=device)
    ld = _loader(cfg, cfg_path, "test", int(cfg["training"]["batch_size"]))
    st = evaluate(model, ld, device, eval_cfg=cfg.get("evaluation"))

    out = {
        "slug": run_dir.name,
        "tag": meta.get("tag"),
        "best_epoch": meta.get("best_epoch"),
        "recorded": {k: meta["test_metrics"].get(k) for k in ("IC", "RankIC", "ICIR", "MSE")},
        "reevaluated": {"IC": float(st["IC"]), "RankIC": float(st["RankIC"]),
                        "ICIR": None if np.isnan(st["ICIR"]) else float(st["ICIR"]),
                        "MSE": float(st["MSE"])},
        "eval_device": str(device),
    }
    out["delta_IC"] = out["reevaluated"]["IC"] - (out["recorded"]["IC"] or 0.0)
    (run_dir / "meta_reeval.json").write_text(json.dumps(out, indent=2))
    # 可重現的 predictions，與 meta_reeval.json 同一次 forward
    st["predictions"].to_csv(run_dir / "predictions" / "test_predictions_reeval.csv",
                             index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None,
                    help="arm tag（不含 _s<seed> 後綴）")
    ap.add_argument("--all-tw50", action="store_true",
                    help="重評所有 tag 以 tw50 開頭的 run")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    run_dirs = sorted(Path(p) for p in glob.glob(str(ROOT / "runs" / "*"))
                      if Path(p).is_dir())

    def wanted(d: Path) -> bool:
        mp = d / "meta.json"
        if not mp.exists():
            return False
        tag = json.loads(mp.read_text()).get("tag", "")
        arm = re.sub(r"_s\d+$", "", tag)
        if args.all_tw50:
            return tag.startswith("tw50")
        return args.tags is not None and arm in args.tags

    targets = [d for d in run_dirs if wanted(d)]
    print(f"[reeval] {len(targets)} runs on {device}\n")

    rows, failed = [], []
    for d in targets:
        try:
            r = reeval_run(d, device)
        except Exception as e:                       # noqa: BLE001
            failed.append((d.name, f"{type(e).__name__}: {e}"))
            continue
        if r is None:
            continue
        rows.append(r)
        print(f"  {r['tag']:28s} recorded {r['recorded']['IC']:+.4f}  "
              f"reeval {r['reevaluated']['IC']:+.4f}  Δ {r['delta_IC']:+.4f}")

    if failed:
        print(f"\n[reeval] {len(failed)} run 無法重評（架構已改變等）：")
        for name, err in failed:
            print(f"    {name}: {err[:110]}")

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_arm[re.sub(r"_s\d+$", "", r["tag"] or "")].append(r)

    summary = []
    for arm, rs in sorted(by_arm.items()):
        rec = np.array([x["recorded"]["IC"] for x in rs], dtype=float)
        new = np.array([x["reevaluated"]["IC"] for x in rs], dtype=float)
        recr = np.array([x["recorded"]["RankIC"] for x in rs], dtype=float)
        newr = np.array([x["reevaluated"]["RankIC"] for x in rs], dtype=float)
        summary.append({
            "arm": arm, "n": len(rs),
            "IC_recorded": rec.mean(), "IC_recorded_sd": rec.std(ddof=1) if len(rs) > 1 else np.nan,
            "IC_reeval": new.mean(), "IC_reeval_sd": new.std(ddof=1) if len(rs) > 1 else np.nan,
            "RankIC_recorded": recr.mean(), "RankIC_recorded_sd": recr.std(ddof=1) if len(rs) > 1 else np.nan,
            "RankIC_reeval": newr.mean(), "RankIC_reeval_sd": newr.std(ddof=1) if len(rs) > 1 else np.nan,
            "mean_delta_IC": (new - rec).mean(),
            "max_abs_delta_IC": np.abs(new - rec).max(),
        })
    df = pd.DataFrame(summary).sort_values("IC_reeval", ascending=False)
    out_csv = ROOT / "docs" / "reeval_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[reeval] arm 匯總 → {out_csv}\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
