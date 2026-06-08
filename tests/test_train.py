"""
test_train.py — M4 訓練 / 評估配套測試
Corresponds to IMPLEMENTATION_SPEC §6 / §7

涵蓋四個測試：
  - test_metrics_ic_basic         手算 IC vs cross_sectional_ic
  - test_evaluator_no_nan         evaluator 跑 val loader 不產生 NaN
  - test_train_3epoch_smoke       3-epoch 訓練 train_loss 必須下降
  - test_checkpoint_roundtrip     save → load 後 forward 結果一致
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.multiplex_dataset import MultiplexDataset, multiplex_collate, N_NODES
from src.models.multiplex_gnn import MAGNET
from src.train.evaluator import evaluate
from src.train.losses import build_criterion
from src.train.metrics import (
    aggregate_ic,
    cross_sectional_ic,
    regression_metrics,
)
from src.train.utils import (
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="module")
def config(project_root: Path) -> dict:
    with open(project_root / "configs" / "base.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def device() -> torch.device:
    # 測試一律用 CPU，避免 MPS 跨平台噪音
    return torch.device("cpu")


@pytest.fixture(scope="module")
def model(config: dict, device: torch.device) -> MAGNET:
    set_seed(config["training"]["seed"])
    m = MAGNET(config).to(device)
    return m


@pytest.fixture(scope="module")
def val_dataset(project_root: Path) -> MultiplexDataset:
    return MultiplexDataset(
        snapshot_dir=str(project_root / "data" / "graphs" / "snapshots"),
        features_dir=str(project_root / "data" / "features"),
        T=20,
        split="val",
        config_path=str(project_root / "configs" / "base.yaml"),
    )


@pytest.fixture(scope="module")
def train_dataset(project_root: Path) -> MultiplexDataset:
    return MultiplexDataset(
        snapshot_dir=str(project_root / "data" / "graphs" / "snapshots"),
        features_dir=str(project_root / "data" / "features"),
        T=20,
        split="train",
        config_path=str(project_root / "configs" / "base.yaml"),
    )


# ---------------------------------------------------------------------------
# Test 1: metrics
# ---------------------------------------------------------------------------

def test_metrics_ic_basic() -> None:
    """已知 y_hat, y 的 IC 結果應與手算值一致。"""
    # 完美正相關：IC = 1
    y     = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    y_hat = 2.0 * y + 0.5
    ic_pearson  = cross_sectional_ic(y_hat, y, method="pearson")
    ic_spearman = cross_sectional_ic(y_hat, y, method="spearman")
    assert abs(ic_pearson  - 1.0) < 1e-6, f"完美線性 IC 應為 1，得到 {ic_pearson}"
    assert abs(ic_spearman - 1.0) < 1e-6, f"完美單調 RankIC 應為 1，得到 {ic_spearman}"

    # 完美反相關：IC = -1
    y_hat_neg = -y
    assert abs(cross_sectional_ic(y_hat_neg, y, method="pearson")  + 1.0) < 1e-6
    assert abs(cross_sectional_ic(y_hat_neg, y, method="spearman") + 1.0) < 1e-6

    # 聚合測試：兩日 IC 都是 1 → 平均=1，std=0 → ICIR=NaN
    daily_yh = [y_hat.numpy(), y_hat.numpy()]
    daily_y  = [y.numpy(),     y.numpy()]
    agg = aggregate_ic(daily_yh, daily_y)
    assert abs(agg["IC"] - 1.0) < 1e-6
    assert np.isnan(agg["ICIR"]), "兩日 IC 完全相同（std=0）時 ICIR 應為 NaN"

    # regression metrics
    y_arr  = np.array([1.0, 2.0, 3.0])
    yh_arr = np.array([1.5, 2.0, 2.5])
    reg = regression_metrics(yh_arr, y_arr)
    expected_mse = ((0.5)**2 + 0.0 + (0.5)**2) / 3.0
    assert abs(reg["MSE"] - expected_mse) < 1e-9


def test_combinedloss_variance_penalty() -> None:
    """variance penalty 應在 ŷ 為常數時最大、ŷ 振幅=y 時為 0、且可微。"""
    from src.models.prediction_head import CombinedLoss

    loss_cfg  = {"mse": 1.0, "rank": 0.0, "align": 0.0, "variance": 1.0}
    align_cfg = {"enabled": False, "temperature": 0.1}
    crit = CombinedLoss(loss_cfg, align_cfg)

    y = torch.tensor([[-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, -0.03]])    # std ≈ 0.0196
    std_y = float(y.std(dim=-1, unbiased=False).item())

    # Case A: ŷ 為常數（pred collapse）→ variance penalty 應 ≈ std(y)²
    y_hat_const = torch.zeros_like(y, requires_grad=True)
    _, comps_a = crit(y_hat=y_hat_const, y=y)
    assert abs(comps_a["variance"] - std_y**2) < 1e-6, (
        f"常數預測時 var loss 應 = std(y)² = {std_y**2:.6f}，得到 {comps_a['variance']:.6f}"
    )

    # Case B: ŷ == y → variance penalty = 0
    y_hat_perfect = y.clone().detach().requires_grad_(True)
    _, comps_b = crit(y_hat=y_hat_perfect, y=y)
    assert abs(comps_b["variance"]) < 1e-6, (
        f"完美預測時 var loss 應 = 0，得到 {comps_b['variance']:.6f}"
    )

    # Case C: 可微分（梯度非 None 且 finite）
    y_hat_grad = torch.zeros_like(y, requires_grad=True)
    total, _ = crit(y_hat=y_hat_grad, y=y)
    total.backward()
    assert y_hat_grad.grad is not None
    assert torch.isfinite(y_hat_grad.grad).all()
    # 常數預測的梯度應為 0（mse 對常數的梯度），由 variance penalty 提供推力
    # 因為 std(const) 對 const 微分為 0，variance penalty 的梯度也為 0
    # 所以這裡只檢查不爆 NaN，不要求 grad 非零


# ---------------------------------------------------------------------------
# Test 2: evaluator no NaN
# ---------------------------------------------------------------------------

def test_evaluator_no_nan(
    model: MAGNET,
    val_dataset: MultiplexDataset,
    config: dict,
    device: torch.device,
) -> None:
    """evaluator 跑 val loader 一次，所有 scalar 指標應為 finite。"""
    loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=multiplex_collate,
        num_workers=0,
    )
    criterion = build_criterion(config).to(device)
    result = evaluate(model, loader, device, criterion=criterion)

    # scalar 指標應為 finite（或刻意允許的 NaN 場景外）
    scalar_keys = ["MSE", "MAE", "RMSE", "IC", "RankIC",
                   "loss_total", "loss_mse", "loss_rank", "loss_align"]
    for k in scalar_keys:
        assert k in result, f"evaluator 缺少欄位 {k}"
        v = result[k]
        assert isinstance(v, float)
        assert np.isfinite(v), f"指標 {k} 為 NaN/inf: {v}"

    # ICIR 可為 NaN（n_days 太少或 std=0）；只檢查它存在
    assert "ICIR" in result and "RankICIR" in result

    # predictions DF schema
    df = result["predictions"]
    assert set(df.columns) == {"target_date", "ticker", "y_hat", "y"}
    assert len(df) == len(val_dataset) * N_NODES


# ---------------------------------------------------------------------------
# Test 3: train 3-epoch smoke（loss 應下降）
# ---------------------------------------------------------------------------

def test_train_3epoch_smoke(
    train_dataset: MultiplexDataset,
    config: dict,
    device: torch.device,
) -> None:
    """跑 3 epoch（用 train 的前 16 筆切片加速），train loss 必須下降。"""
    set_seed(config["training"]["seed"])

    # 為加速測試：只取前 16 筆 train sample
    from torch.utils.data import Subset
    subset = Subset(train_dataset, list(range(16)))
    loader = DataLoader(subset, batch_size=4, shuffle=False,
                        collate_fn=multiplex_collate, num_workers=0)

    model = MAGNET(config).to(device)
    criterion = build_criterion(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=float(config["training"]["lr"]),
                                  weight_decay=float(config["training"]["weight_decay"]))

    from src.train.utils import batch_to_device

    epoch_losses: list[float] = []
    for epoch in range(3):
        model.train()
        total = 0.0
        nb = 0
        for batch in loader:
            batch = batch_to_device(batch, device)
            y_hat, extras = model(batch)
            loss, _ = criterion(y_hat=y_hat, y=batch["y"],
                                  h_L1=extras.get("h_L1"), h_L2=extras.get("h_L2"))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
            nb += 1
        epoch_losses.append(total / max(1, nb))

    print(f"smoke epoch_losses = {epoch_losses}")
    # 寬鬆條件：最後 epoch 比第一個 epoch 低（避免短訓練的噪音卡住測試）
    assert epoch_losses[-1] < epoch_losses[0], (
        f"3 epoch 後 loss 沒下降：{epoch_losses}"
    )


# ---------------------------------------------------------------------------
# Test 4: checkpoint roundtrip
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip(
    val_dataset: MultiplexDataset,
    config: dict,
    device: torch.device,
) -> None:
    """save → load 後，相同輸入下 forward 結果一致（位元級一致需 eval 模式）。"""
    set_seed(config["training"]["seed"])
    model_a = MAGNET(config).to(device)
    optimizer = torch.optim.Adam(model_a.parameters(), lr=1e-3)

    # 跑一個小 batch 取 reference 輸出
    sample_idxs = [0, 1, 2, 3]
    samples = [val_dataset[i] for i in sample_idxs]
    batch = multiplex_collate(samples)
    from src.train.utils import batch_to_device
    batch = batch_to_device(batch, device)

    model_a.eval()
    with torch.no_grad():
        y_hat_a, _ = model_a(batch)

    # 存檔 → 新 model load
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "ckpt.pt"
        save_checkpoint(ckpt_path, model_a, optimizer,
                        epoch=0, best_val_ic=0.123, extras={"hello": "world"})

        set_seed(0)   # 故意換 seed，確認 load 真的覆寫了 weight
        model_b = MAGNET(config).to(device)
        ckpt = load_checkpoint(ckpt_path, model_b, optimizer=None, map_location=device)

        assert ckpt["epoch"] == 0
        assert abs(ckpt["best_val_ic"] - 0.123) < 1e-9
        assert ckpt["extras"]["hello"] == "world"

        model_b.eval()
        with torch.no_grad():
            y_hat_b, _ = model_b(batch)

        assert torch.allclose(y_hat_a, y_hat_b, atol=1e-6), (
            "save→load 後 forward 結果不一致"
        )
