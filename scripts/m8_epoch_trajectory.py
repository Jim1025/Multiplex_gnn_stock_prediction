"""
m8_epoch_trajectory.py — 路線 A 實驗 3（Figure 1）：逐 epoch val/test IC 軌跡

對 --save-every-epoch 產生的 runs：
    1. 逐 epoch checkpoint 在 val 與 test 上各評一次 IC
    2. 存 runs/<slug>/epoch_trajectory.csv
    3. 畫 Figure 1（每 run 兩欄：軌跡圖 + val-vs-test 散點與 Pearson r）

執行：
    python scripts/m8_epoch_trajectory.py
    python scripts/m8_epoch_trajectory.py --runs opt_p65_fig1_lr1e3 opt_p66_fig1_lr5e4
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from src.train.utils import get_device


def find_run_dir(tag: str) -> Path:
    hits = sorted(glob.glob(str(ROOT / "runs" / f"*{tag}")))
    if not hits:
        raise SystemExit(f"找不到 run dir for tag={tag}")
    return Path(hits[-1])


def make_loader(cfg_path: str, split: str, cfg: dict) -> DataLoader:
    ds = MultiplexDataset(
        snapshot_dir=str(ROOT / cfg["data"]["snapshot_dir"]),
        features_dir=str(ROOT / cfg["data"]["features_dir"]),
        T=cfg["model"]["lstm"]["T_history"],
        split=split,
        config_path=cfg_path,
    )
    return DataLoader(ds, batch_size=32, shuffle=False,
                      collate_fn=multiplex_collate, num_workers=0)


def trajectory_for_run(tag: str, device: torch.device) -> tuple[pd.DataFrame, Path]:
    run_dir = find_run_dir(tag)
    cfg_path = str(run_dir / "config_snapshot.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg).to(device)
    val_loader  = make_loader(cfg_path, "val", cfg)
    test_loader = make_loader(cfg_path, "test", cfg)

    ckpts = sorted(glob.glob(str(run_dir / "checkpoints" / "epoch_*.pt")))
    if not ckpts:
        raise SystemExit(f"{run_dir} 無 epoch_*.pt（需以 --save-every-epoch 訓練）")

    rows = []
    for p in ckpts:
        epoch = int(re.search(r"epoch_(\d+)\.pt", p).group(1))
        state = torch.load(p, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        v = evaluate(model, val_loader,  device, criterion=None)["IC"]
        t = evaluate(model, test_loader, device, criterion=None)["IC"]
        rows.append({"epoch": epoch, "val_IC": v, "test_IC": t})
        print(f"  [{run_dir.name}] epoch {epoch:03d}  val_IC={v:+.4f}  test_IC={t:+.4f}")

    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    csv_path = run_dir / "epoch_trajectory.csv"
    df.to_csv(csv_path, index=False)
    return df, run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["opt_p65_fig1_lr1e3", "opt_p66_fig1_lr5e4"])
    ap.add_argument("--out", type=str,
                    default="docs/figures/m8_fig1_epoch_trajectory.png")
    args = ap.parse_args()

    device = get_device("auto")
    print(f"[device] {device}")

    panels = []
    for tag in args.runs:
        df, run_dir = trajectory_for_run(tag, device)
        panels.append((tag, df))

    n = len(panels)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.6 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, (tag, df) in enumerate(panels):
        label = tag.split("fig1_")[-1].replace("lr1e3", "lr 1e-3").replace("lr5e4", "lr 5e-4")
        best_ep = int(df.loc[df["val_IC"].idxmax(), "epoch"])

        ax = axes[i, 0]
        ax.axhline(0, color="0.8", lw=0.8)
        ax.plot(df["epoch"], df["val_IC"],  color="0.15", lw=1.6, marker="o", ms=3, label="val IC")
        ax.plot(df["epoch"], df["test_IC"], color="#c0392b", lw=1.6, marker="s", ms=3, label="test IC")
        ax.axvline(best_ep, color="0.5", ls="--", lw=1.0,
                   label=f"best-val epoch ({best_ep})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("daily-mean IC")
        ax.set_title(f"MAGNET ({label}, seed 42): per-epoch IC trajectory")
        ax.legend(frameon=False, fontsize=8)

        ax = axes[i, 1]
        r = np.corrcoef(df["val_IC"], df["test_IC"])[0, 1]
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.scatter(df["val_IC"], df["test_IC"], s=22, color="0.25")
        bv = df.loc[df["val_IC"].idxmax()]
        ax.scatter([bv["val_IC"]], [bv["test_IC"]], s=70, facecolors="none",
                   edgecolors="#c0392b", lw=1.8, label="selected checkpoint")
        ax.set_xlabel("val IC")
        ax.set_ylabel("test IC")
        ax.set_title(f"val vs test across epochs (Pearson r = {r:+.2f})")
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
