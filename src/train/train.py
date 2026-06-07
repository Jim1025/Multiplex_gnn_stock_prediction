"""
train.py — M4 主訓練迴圈 + CLI 入口
Corresponds to IMPLEMENTATION_SPEC §7 (training loop) / §8.4

使用：
    python -m src.train.train --epochs 3 --tag sanity
    python -m src.train.train --tag full

流程：
    1. 讀 config → set_seed(42) → get_device("auto")
    2. 載入 train / val / test 三個 DataLoader（multiplex_collate）
    3. 建立 MAGNET、Optimizer (Adam)
    4. MLflow start_run（log_params + per-epoch metrics）
    5. 訓練迴圈（含 grad_clip / early stopping on val IC）
    6. 訓練結束 → 載入 best ckpt → test set 一次性評估
    7. 上傳 best/final ckpt + val/test predictions.csv 進 MLflow artifacts

Early Stopping：monitor = val IC（越大越好）；patience = config.training.patience
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

# MLflow 3.x 預設拒絕 file:// 後端；此專案明確選用本地 mlruns/，
# 故在 import 前設定 opt-out。可由環境變數覆寫。
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset.multiplex_dataset import MultiplexDataset, multiplex_collate
from src.models.multiplex_gnn import MAGNET
from src.train.evaluator import evaluate
from src.train.losses import build_criterion
from src.train.utils import (
    batch_to_device,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


# ---------------------------------------------------------------------------
# 工具：把巢狀 config 攤平成 mlflow.log_params 可吃的 flat dict
# ---------------------------------------------------------------------------

def _flatten(d: dict, parent: str = "", sep: str = ".") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten(v, key, sep))
        else:
            out[key] = v
    return out


# ---------------------------------------------------------------------------
# 單一 epoch：訓練
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     MAGNET,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device:    torch.device,
    grad_clip: float,
    log_every: int,
    epoch:     int,
) -> dict:
    model.train()

    total_sum = 0.0
    mse_sum   = 0.0
    rank_sum  = 0.0
    align_sum = 0.0
    n_samples = 0

    t0 = time.time()
    for step, batch in enumerate(loader):
        batch = batch_to_device(batch, device)
        y_hat, extras = model(batch)
        y = batch["y"]

        loss, comps = criterion(
            y_hat=y_hat,
            y=y,
            h_L1=extras.get("h_L1"),
            h_L2=extras.get("h_L2"),
        )

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        B = y.size(0)
        total_sum += float(loss.item()) * B
        mse_sum   += comps["mse"]   * B
        rank_sum  += comps["rank"]  * B
        align_sum += comps["align"] * B
        n_samples += B

        if (step + 1) % max(1, log_every) == 0:
            avg = total_sum / max(1, n_samples)
            print(
                f"  [epoch {epoch:03d} step {step+1:04d}/{len(loader):04d}] "
                f"loss_total={avg:.5f}  "
                f"(mse={comps['mse']:.5f} rank={comps['rank']:.5f} align={comps['align']:.5f})"
            )

    return {
        "loss_total": total_sum / max(1, n_samples),
        "loss_mse":   mse_sum   / max(1, n_samples),
        "loss_rank":  rank_sum  / max(1, n_samples),
        "loss_align": align_sum / max(1, n_samples),
        "epoch_time_sec": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 主訓練函數
# ---------------------------------------------------------------------------

def train(
    config_path: str = "configs/base.yaml",
    epochs:      Optional[int] = None,
    tag:         str = "full",
) -> str:
    """
    主訓練流程。

    Args:
        config_path: configs/base.yaml
        epochs:      若提供，覆寫 config.training.max_epochs（sanity 用）
        tag:         MLflow run name（"sanity" | "full" | ...）

    Returns:
        mlflow_run_id (str)
    """
    # ── 讀 config ──────────────────────────────────────────────────
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    t_cfg  = cfg["training"]
    mf_cfg = cfg["mlflow"]
    d_cfg  = cfg["data"]

    seed       = int(t_cfg.get("seed", 42))
    max_epochs = int(epochs) if epochs is not None else int(t_cfg.get("max_epochs", 100))
    patience   = int(t_cfg.get("patience", 10))
    batch_size = int(t_cfg.get("batch_size", 32))
    lr         = float(t_cfg.get("lr", 1e-3))
    weight_dec = float(t_cfg.get("weight_decay", 1e-4))
    num_workers = int(t_cfg.get("num_workers", 0))
    pin_memory  = bool(t_cfg.get("pin_memory", False))
    grad_clip   = float(t_cfg.get("grad_clip", 1.0))
    log_every   = int(t_cfg.get("log_every_n_steps", 10))
    min_delta   = float(t_cfg.get("min_delta", 0.0))

    set_seed(seed)
    device = get_device(t_cfg.get("device", "auto"))
    print(f"[train] device={device} | tag={tag} | max_epochs={max_epochs} | patience={patience}")

    # ── DataLoaders ────────────────────────────────────────────────
    snap_dir = d_cfg["snapshot_dir"]
    feat_dir = d_cfg["features_dir"]

    train_ds = MultiplexDataset(snapshot_dir=snap_dir, features_dir=feat_dir,
                                 T=cfg["model"]["lstm"]["T_history"],
                                 split="train", config_path=config_path)
    val_ds   = MultiplexDataset(snapshot_dir=snap_dir, features_dir=feat_dir,
                                 T=cfg["model"]["lstm"]["T_history"],
                                 split="val", config_path=config_path)
    test_ds  = MultiplexDataset(snapshot_dir=snap_dir, features_dir=feat_dir,
                                 T=cfg["model"]["lstm"]["T_history"],
                                 split="test", config_path=config_path)

    common_loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,            # 時序資料禁洗牌
        collate_fn=multiplex_collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    train_loader = DataLoader(train_ds, **common_loader_kwargs)
    val_loader   = DataLoader(val_ds,   **common_loader_kwargs)
    test_loader  = DataLoader(test_ds,  **common_loader_kwargs)

    # ── Model / Optimizer / Criterion ─────────────────────────────
    model = MAGNET(cfg).to(device)
    criterion = build_criterion(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_dec)

    # ── MLflow setup ──────────────────────────────────────────────
    mlflow.set_tracking_uri(mf_cfg.get("tracking_uri", "file:./mlruns"))
    mlflow.set_experiment(mf_cfg.get("experiment_name", "MAGNET_M4_baseline"))

    # checkpoint 路徑
    ckpt_root = Path("checkpoints")
    pred_root = Path("predictions")
    ckpt_root.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=tag) as run:
        run_id = run.info.run_id
        print(f"[mlflow] run_id={run_id}")

        # log all hyperparams
        params_flat = _flatten({
            "model":        cfg["model"],
            "loss_weights": cfg["loss_weights"],
            "align_loss":   cfg["align_loss"],
            "training":     {k: v for k, v in t_cfg.items()},
        })
        # 覆寫 epochs（CLI 可能改）
        params_flat["training.max_epochs"] = max_epochs
        params_flat["tag"] = tag
        # mlflow 限制 value 長度，整批 log
        mlflow.log_params({k: str(v) for k, v in params_flat.items()})

        # ── 訓練迴圈 ───────────────────────────────────────────────
        best_val_ic   = -float("inf")
        best_epoch    = -1
        patience_cnt  = 0
        ckpt_dir      = ckpt_root / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path  = ckpt_dir / "best.pt"
        final_ckpt_path = ckpt_dir / "final.pt"

        for epoch in range(max_epochs):
            train_stats = train_one_epoch(
                model=model, loader=train_loader, optimizer=optimizer,
                criterion=criterion, device=device,
                grad_clip=grad_clip, log_every=log_every, epoch=epoch,
            )

            val_stats = evaluate(model, val_loader, device, criterion=criterion)
            val_ic = val_stats["IC"]

            # MLflow log
            mlflow.log_metrics({
                "train/loss_total": train_stats["loss_total"],
                "train/loss_mse":   train_stats["loss_mse"],
                "train/loss_rank":  train_stats["loss_rank"],
                "train/loss_align": train_stats["loss_align"],
                "train/epoch_time_sec": train_stats["epoch_time_sec"],
                "val/loss_total":   val_stats.get("loss_total", float("nan")),
                "val/loss_mse":     val_stats.get("loss_mse", float("nan")),
                "val/MSE":          val_stats["MSE"],
                "val/MAE":          val_stats["MAE"],
                "val/IC":           val_stats["IC"],
                "val/RankIC":       val_stats["RankIC"],
                "val/ICIR":         val_stats["ICIR"] if not np.isnan(val_stats["ICIR"]) else 0.0,
                "val/RankICIR":     val_stats["RankICIR"] if not np.isnan(val_stats["RankICIR"]) else 0.0,
                "learning_rate":    optimizer.param_groups[0]["lr"],
            }, step=epoch)

            print(
                f"[epoch {epoch:03d}] "
                f"train_loss={train_stats['loss_total']:.5f} | "
                f"val_loss={val_stats.get('loss_total', float('nan')):.5f} "
                f"val_IC={val_ic:.4f} val_RankIC={val_stats['RankIC']:.4f} "
                f"({train_stats['epoch_time_sec']:.1f}s)"
            )

            # Early stopping on val IC（越大越好）
            improved = (not np.isnan(val_ic)) and (val_ic > best_val_ic + min_delta)
            if improved:
                best_val_ic = val_ic
                best_epoch  = epoch
                patience_cnt = 0
                save_checkpoint(best_ckpt_path, model, optimizer,
                                epoch=epoch, best_val_ic=best_val_ic,
                                extras={"val_stats": {k: v for k, v in val_stats.items()
                                                        if k not in ("predictions", "daily_IC", "daily_RankIC")}})
                print(f"    ↑ new best val_IC={best_val_ic:.4f} (epoch {epoch}) → saved")
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"[early stopping] patience {patience} reached at epoch {epoch}")
                    break

        # 存 final
        save_checkpoint(final_ckpt_path, model, optimizer,
                        epoch=epoch, best_val_ic=best_val_ic, extras={"final": True})

        # ── 載入 best → test 一次性評估 ───────────────────────────
        if best_ckpt_path.exists():
            load_checkpoint(best_ckpt_path, model, optimizer=None, map_location=device)
            print(f"[test] loaded best ckpt (epoch={best_epoch}, val_IC={best_val_ic:.4f})")
        else:
            print("[test] WARNING: best ckpt 不存在（IC 從未為正），用 final 評估")

        test_stats = evaluate(model, test_loader, device, criterion=criterion)
        print(
            f"[test] IC={test_stats['IC']:.4f} RankIC={test_stats['RankIC']:.4f} "
            f"ICIR={test_stats['ICIR']:.4f} MSE={test_stats['MSE']:.5f}"
        )

        # log test metrics
        mlflow.log_metrics({
            "test/IC":       test_stats["IC"],
            "test/RankIC":   test_stats["RankIC"],
            "test/ICIR":     test_stats["ICIR"] if not np.isnan(test_stats["ICIR"]) else 0.0,
            "test/RankICIR": test_stats["RankICIR"] if not np.isnan(test_stats["RankICIR"]) else 0.0,
            "test/MSE":      test_stats["MSE"],
            "test/MAE":      test_stats["MAE"],
            "test/RMSE":     test_stats["RMSE"],
            "best_epoch":    float(best_epoch),
            "best_val_IC":   best_val_ic,
        })

        # ── 上傳 artifacts ─────────────────────────────────────────
        if bool(mf_cfg.get("log_artifacts", True)):
            test_pred_path = pred_root / f"{run_id}_test_predictions.csv"
            val_pred_path  = pred_root / f"{run_id}_val_predictions.csv"
            test_stats["predictions"].to_csv(test_pred_path, index=False)
            val_pred_dump = evaluate(model, val_loader, device, criterion=criterion)["predictions"]
            val_pred_dump.to_csv(val_pred_path, index=False)
            mlflow.log_artifact(str(test_pred_path), artifact_path="predictions")
            mlflow.log_artifact(str(val_pred_path),  artifact_path="predictions")
            if best_ckpt_path.exists():
                mlflow.log_artifact(str(best_ckpt_path), artifact_path="checkpoints")
            mlflow.log_artifact(str(final_ckpt_path), artifact_path="checkpoints")

        return run_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAGNET M4 訓練主程式")
    p.add_argument("--config", type=str, default="configs/base.yaml",
                   help="YAML 路徑（預設 configs/base.yaml）")
    p.add_argument("--epochs", type=int, default=None,
                   help="覆寫 config.training.max_epochs（sanity 用 3）")
    p.add_argument("--tag", type=str, default="full",
                   help="MLflow run name（建議 sanity / full / ablation_xxx）")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_id = train(config_path=args.config, epochs=args.epochs, tag=args.tag)
    print(f"\n✅ Done. MLflow run_id = {run_id}")
