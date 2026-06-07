"""
utils.py — M4 訓練工具函數
Corresponds to IMPLEMENTATION_SPEC §8.3 (training utilities)

提供：
  - get_device(spec): 解析 "auto"/"cpu"/"mps"/"cuda" → torch.device
  - set_seed(seed):   同時固定 torch / numpy / random / cudnn 三者
  - save_checkpoint / load_checkpoint
"""

from __future__ import annotations

import os
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device(spec: str = "auto") -> torch.device:
    """
    根據設定字串解析 device。

    Args:
        spec: "auto" | "cpu" | "mps" | "cuda"

    Returns:
        torch.device

    Note:
        - "auto" 優先 MPS（Apple Silicon），其次 CUDA，最後 CPU。
        - 若指定 mps/cuda 但不可用，回退到 CPU 並警告。
    """
    spec = spec.lower().strip()

    if spec == "cpu":
        return torch.device("cpu")

    if spec == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        warnings.warn("MPS 不可用，回退到 CPU", RuntimeWarning)
        return torch.device("cpu")

    if spec == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        warnings.warn("CUDA 不可用，回退到 CPU", RuntimeWarning)
        return torch.device("cpu")

    if spec == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    raise ValueError(f"未知 device spec: {spec!r}（應為 auto/cpu/mps/cuda）")


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """同時固定 torch / numpy / random，並設 cudnn deterministic。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn deterministic（若有 CUDA）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_val_ic: float,
    extras: dict[str, Any] | None = None,
) -> None:
    """儲存 checkpoint 至 path（自動建立父目錄）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch":                epoch,
        "best_val_ic":          float(best_val_ic),
        "extras":               extras or {},
    }
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> dict:
    """從 path 載入 checkpoint，並把參數 load 進 model（optimizer 選擇性）。"""
    path = Path(path)
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# Batch device transfer
# ---------------------------------------------------------------------------

def batch_to_device(batch: dict, device: torch.device) -> dict:
    """
    把 multiplex_collate 產出的 batch 全部搬到指定 device。

    處理三種型態：
      - Tensor：直接 .to(device)
      - list[Tensor]（edge_index / edge_attr）：逐個 .to(device)
      - list[str]（target_date）：保持不動
    """
    out: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            out[k] = [t.to(device, non_blocking=True) for t in v]
        else:
            out[k] = v
    return out
