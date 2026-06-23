"""
test_model_sanity.py — M3 模型 sanity check
Corresponds to IMPLEMENTATION_SPEC §10 Step 5

驗證 MAGNET + MultiplexDataset 端到端可跑：
  - forward shape 正確（[B, n]）
  - 輸出無 NaN
  - backward 可微分（所有 trainable parameter grad 非 None）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

# 確保 project root 在 sys.path（pytest 從 tests/ 目錄跑時）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.multiplex_dataset import (
    MultiplexDataset,
    multiplex_collate,
    N_NODES,
    F,
)
from src.models import build_model
from src.models.baseline_lstm import BaselineLSTM
from src.models.baseline_tw_gnn import BaselineTWGNN
from src.models.multiplex_gnn import MAGNET


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="module")
def config(project_root: Path) -> dict:
    with open(project_root / "configs" / "base.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def train_dataset(project_root: Path) -> MultiplexDataset:
    # 用較小 T 避免 DataLoader 啟動慢；T=20 是 base.yaml 預設
    return MultiplexDataset(
        snapshot_dir=str(project_root / "data" / "graphs" / "snapshots"),
        features_dir=str(project_root / "data" / "features"),
        T=20,
        split="train",
        config_path=str(project_root / "configs" / "base.yaml"),
    )


@pytest.fixture(scope="module")
def model(config: dict) -> MAGNET:
    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])
    return MAGNET(config)


@pytest.fixture(scope="module")
def small_batch(train_dataset: MultiplexDataset) -> dict:
    """組一個小 batch（B=4）作為通用測試輸入。"""
    samples = [train_dataset[i] for i in range(4)]
    return multiplex_collate(samples)


# ---------------------------------------------------------------------------
# 測試
# ---------------------------------------------------------------------------

def test_dataset_smoke(train_dataset: MultiplexDataset) -> None:
    """Dataset 基本健檢：長度合理、單筆 shape 正確、ticker 順序正確。"""
    assert len(train_dataset) > 0, "train split 不應為空"
    sample = train_dataset[0]

    # Shape 檢查
    assert sample["x_seq_L1"].shape == (20, N_NODES, F)
    assert sample["x_seq_L2"].shape == (20, N_NODES, F)
    assert sample["y"].shape == (N_NODES,)
    assert sample["edge_index_L1"].dim() == 2 and sample["edge_index_L1"].shape[0] == 2
    assert sample["edge_attr_L1"].shape[1] == 1
    assert sample["edge_index_L2"].dim() == 2 and sample["edge_index_L2"].shape[0] == 2
    assert sample["edge_attr_L2"].shape[1] == 1
    assert isinstance(sample["target_date"], str)

    # Ticker 順序固定
    adr_order, tw_order = MultiplexDataset.get_ticker_order()
    assert len(adr_order) == N_NODES
    assert len(tw_order) == N_NODES
    assert adr_order == ["TSM", "UMC", "ASX", "CHT", "IMOS", "AUOTY", "HNHPF"]


def test_magnet_forward_shape(model: MAGNET, small_batch: dict) -> None:
    """forward 輸出 shape 必須 == [B, n]，且 extras 五個張量 shape 正確。"""
    model.eval()
    B = small_batch["x_seq_L1"].size(0)
    with torch.no_grad():
        y_hat, extras = model(small_batch)

    d_prime = model.head.mlp[0].in_features
    assert y_hat.shape == (B, N_NODES), f"y_hat shape 錯誤：{y_hat.shape}"
    assert extras["h_L1"].shape   == (B, N_NODES, d_prime)
    assert extras["h_L2"].shape   == (B, N_NODES, d_prime)
    assert extras["h_fused"].shape == (B, N_NODES, d_prime)
    assert extras["alpha"].shape   == (B, N_NODES, 1)
    assert extras["gate"].shape    == (B, N_NODES, d_prime)


def test_no_nan_in_output(model: MAGNET, small_batch: dict) -> None:
    """forward 輸出絕不可含 NaN（驗證資料前處理與模型初始化健康）。"""
    model.eval()
    with torch.no_grad():
        y_hat, extras = model(small_batch)
    assert torch.isfinite(y_hat).all(), "y_hat 含 NaN 或 inf"
    for key, t in extras.items():
        assert torch.isfinite(t).all(), f"extras[{key}] 含 NaN 或 inf"


def test_backward_pass(model: MAGNET, small_batch: dict) -> None:
    """loss.backward() 後所有「主路徑」trainable parameter grad 非 None 且為 finite。

    例外（已知未使用，故意設計）：
      - fusion.attn_mlp.*：per-node attention `alpha` 為 SPEC §4.1 規定的
        分析用張量，h_fused 公式（§4.2）只用 gate。alpha 預留給 §4.3
        Volatility Adaptive Weighting，MVP 階段 deferred，故此處不參與
        loss 計算、無 grad。
    """
    model.train()

    # forward
    y_hat, extras = model(small_batch)
    y = small_batch["y"]

    # 計算 combined loss
    loss, components = model.compute_loss(y_hat, y, extras)
    assert torch.isfinite(loss), f"loss 含 NaN：components={components}"

    # backward
    model.zero_grad()
    loss.backward()

    # SPEC §4.3 deferred：fusion.attn_mlp 在 MVP 中不參與主損失
    EXPECTED_UNUSED_PREFIXES = ("fusion.attn_mlp",)

    params_no_grad: list[str] = []
    params_nan_grad: list[str] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(prefix) for prefix in EXPECTED_UNUSED_PREFIXES):
            continue   # 已知 deferred 模組，跳過
        if p.grad is None:
            params_no_grad.append(name)
        elif not torch.isfinite(p.grad).all():
            params_nan_grad.append(name)

    assert not params_no_grad, (
        f"以下主路徑 trainable parameter 沒有 grad：{params_no_grad[:5]} ..."
    )
    assert not params_nan_grad, (
        f"以下 parameter grad 含 NaN/inf：{params_nan_grad[:5]} ..."
    )


def test_dataloader_integration(train_dataset: MultiplexDataset, model: MAGNET) -> None:
    """DataLoader + collate_fn + MAGNET 端到端 1 個 batch 跑通。"""
    loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,            # walk-forward 訓練不要 shuffle（M4 也保持）
        collate_fn=multiplex_collate,
        num_workers=0,
    )
    batch = next(iter(loader))
    model.eval()
    with torch.no_grad():
        y_hat, _ = model(batch)
    assert y_hat.shape == (8, N_NODES)


# ---------------------------------------------------------------------------
# M6 Stage 0 — Ablation baseline smoke tests
# ---------------------------------------------------------------------------

def _smoke_forward_backward(model, batch: dict, extra_skip_prefixes: tuple = ()) -> None:
    """共用 helper：forward shape 對 + loss 可微分 + 主路徑 grad 非 None。

    Args:
        extra_skip_prefixes: 額外跳過的參數前綴。對 magnet_no_a12 而言，
                             gat_L1 / proj_L1 的 grad 預期為 None（ADR 路徑刻意切斷）。
    """
    B = batch["x_seq_L2"].size(0)
    model.train()
    y_hat, extras = model(batch)
    assert y_hat.shape == (B, N_NODES), f"y_hat shape 錯誤：{y_hat.shape}"
    assert torch.isfinite(y_hat).all(), "y_hat 含 NaN/inf"
    for key in ("h_L1", "h_L2", "h_fused"):
        assert key in extras, f"extras 缺少 {key}"

    loss, _ = model.compute_loss(y_hat, batch["y"], extras)
    assert torch.isfinite(loss), "loss 含 NaN"
    model.zero_grad()
    loss.backward()

    # MAGNET 已知 fusion.attn_mlp 在 MVP 未參與主損失（SPEC §4.3 deferred）
    skip = ("fusion.attn_mlp",) + tuple(extra_skip_prefixes)
    no_grad: list[str] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(prefix) for prefix in skip):
            continue
        if p.grad is None:
            no_grad.append(name)
    assert not no_grad, f"以下參數無 grad：{no_grad[:5]} ..."


def test_baseline_lstm_smoke(config: dict, small_batch: dict) -> None:
    """BaselineLSTM（無圖、無 ADR）forward + backward 可跑。"""
    torch.manual_seed(config["training"]["seed"])
    model = BaselineLSTM(config)
    _smoke_forward_backward(model, small_batch)


def test_baseline_tw_gnn_smoke(config: dict, small_batch: dict) -> None:
    """BaselineTWGNN（單層 TW GNN，無 ADR）forward + backward 可跑。"""
    torch.manual_seed(config["training"]["seed"])
    model = BaselineTWGNN(config)
    _smoke_forward_backward(model, small_batch)


def test_magnet_no_a12_smoke(config: dict, small_batch: dict) -> None:
    """MAGNET + disable_a12（h_L1 強制零化）forward + backward 可跑。
    ADR 編碼路徑（gat_L1 / proj_L1）刻意切斷，預期無 grad。"""
    torch.manual_seed(config["training"]["seed"])
    cfg = {**config, "model": {**config["model"], "architecture": "magnet_no_a12"}}
    model = build_model(cfg)
    assert isinstance(model, MAGNET)
    assert model.disable_a12 is True
    # disable_a12=True 時 L1 端的 GAT / Projection 整段被切斷（h_L1 進 fusion 前歸零）
    _smoke_forward_backward(model, small_batch, extra_skip_prefixes=("gat_L1", "proj_L1"))


def test_build_model_dispatch(config: dict) -> None:
    """build_model 依 architecture 派對應 class，未知值要報錯。"""
    cfg_lstm = {**config, "model": {**config["model"], "architecture": "baseline_lstm"}}
    cfg_twgn = {**config, "model": {**config["model"], "architecture": "baseline_tw_gnn"}}
    cfg_mag  = {**config, "model": {**config["model"], "architecture": "magnet"}}
    cfg_bad  = {**config, "model": {**config["model"], "architecture": "nope"}}

    assert isinstance(build_model(cfg_lstm), BaselineLSTM)
    assert isinstance(build_model(cfg_twgn), BaselineTWGNN)
    assert isinstance(build_model(cfg_mag),  MAGNET)
    with pytest.raises(ValueError):
        build_model(cfg_bad)
