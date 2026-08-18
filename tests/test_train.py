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
    long_short_metrics,
    rank_bucket_returns,
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


def test_level_r2_definitions() -> None:
    """R2（均值基準）與 R2_zero（零基準）需與手算一致，且零預測器的
    R2_zero 恰為 0——這正是「level R2 ≈ 0 代表數值不可預測」的判讀基準。"""
    y      = np.array([1.0, 2.0, 3.0])
    y_hat  = np.array([1.5, 2.0, 2.5])

    reg = regression_metrics(y_hat, y)
    # SS_res = 0.25 + 0 + 0.25 = 0.5；SS_tot = 1 + 0 + 1 = 2；SS_zero = 1 + 4 + 9 = 14
    assert abs(reg["R2"]      - (1.0 - 0.5 / 2.0))  < 1e-12
    assert abs(reg["R2_zero"] - (1.0 - 0.5 / 14.0)) < 1e-12

    # 零預測器：R2_zero 恰為 0；R2 為負（比猜均值還差）
    reg_zero = regression_metrics(np.zeros_like(y), y)
    assert abs(reg_zero["R2_zero"]) < 1e-12
    assert reg_zero["R2"] < 0.0

    # 完美預測：兩者皆為 1
    reg_perfect = regression_metrics(y, y)
    assert abs(reg_perfect["R2"] - 1.0)      < 1e-12
    assert abs(reg_perfect["R2_zero"] - 1.0) < 1e-12


def test_long_short_metrics_basic() -> None:
    """多空組合 PnL / Sharpe 需與手算一致，且方向相反時 Sharpe 變號。"""
    yh = [np.array([4.0, 3.0, 2.0, 1.0]), np.array([4.0, 3.0, 2.0, 1.0])]
    ys = [np.array([0.02, 0.01, -0.01, -0.02]), np.array([0.01, 0.0, 0.0, -0.01])]

    pf = long_short_metrics(yh, ys, n_side=1, periods_per_year=252)
    # day1: 0.02 - (-0.02) = 0.04；day2: 0.01 - (-0.01) = 0.02
    assert pf["n_days"] == 2
    assert abs(pf["mean_daily_pnl"] - 0.03) < 1e-12
    assert abs(pf["cum_log_return"] - 0.06) < 1e-12
    assert pf["hit_rate"] == 1.0

    arr = np.array([0.04, 0.02])
    expected = arr.mean() / arr.std(ddof=1) * np.sqrt(252)
    assert abs(pf["Sharpe"] - expected) < 1e-9

    # 預測完全反向 → PnL 變號，Sharpe 變負
    pf_inv = long_short_metrics([-a for a in yh], ys, n_side=1, periods_per_year=252)
    assert abs(pf_inv["mean_daily_pnl"] + 0.03) < 1e-12
    assert pf_inv["Sharpe"] < 0

    # n_side 過大導致每日皆不足 2*n_side → 無有效交易日
    pf_empty = long_short_metrics(yh, ys, n_side=3, periods_per_year=252)
    assert pf_empty["n_days"] == 0
    assert np.isnan(pf_empty["Sharpe"])


def test_rank_bucket_returns_ordering() -> None:
    """逐名次平均報酬：名次 0 = 當日預測最高，應取回對應的實現報酬。"""
    yh = [np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0])]
    ys = [np.array([0.03, 0.02, 0.01]), np.array([0.01, 0.02, 0.03])]

    out = rank_bucket_returns(yh, ys)
    assert out["n_days"] == 2
    assert out["n_days_per_rank"] == [2, 2, 2]
    # 兩天預測方向相反但實現報酬也相反 → 逐名次平均仍為 0.03 / 0.02 / 0.01
    assert np.allclose(out["by_rank"], [0.03, 0.02, 0.01])


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

    # 未提供 eval_cfg 時不得自行產生組合指標（超參只能來自 config）
    assert "Sharpe" not in result


def test_evaluator_portfolio_metrics(
    model: MAGNET,
    val_dataset: MultiplexDataset,
    config: dict,
    device: torch.device,
) -> None:
    """提供 evaluation.portfolio 設定時，evaluator 應產出組合指標。"""
    loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=multiplex_collate,
        num_workers=0,
    )
    eval_cfg = config["evaluation"]
    assert 2 * int(eval_cfg["portfolio"]["n_side"]) <= N_NODES, (
        "n_side 設定過大：2*n_side 必須 <= 橫截面寬度 k"
    )

    result = evaluate(model, loader, device, criterion=None, eval_cfg=eval_cfg)

    for k in ["Sharpe", "mean_daily_pnl", "std_daily_pnl", "hit_rate",
              "ann_return", "cum_log_return", "pf_n_days"]:
        assert k in result, f"缺少組合指標 {k}"
    assert result["pf_n_days"] == len(val_dataset)
    assert 0.0 <= result["hit_rate"] <= 1.0
    assert len(result["daily_pnl"]) == result["pf_n_days"]
    assert len(result["rank_bucket_returns"]) == N_NODES


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


# ---------------------------------------------------------------------------
# Test 5: checkpoint 必須重現自己記錄的 test IC
# ---------------------------------------------------------------------------
# 迴歸來源：2026-08-16 那批 run 的 meta.json / predictions CSV 都無法由自己的
# best.pt 重現。原因不在 save/load（那條路位元正確），而在 metric 算在 MPS 上：
# GATv2Conv 的鄰居聚合走 scatter-add，MPS 後端的浮點加總順序 run 之間不固定。
# 同一份 best.pt 在 MPS 上連評 5 次 test IC = +0.0466 / +0.0362 / +0.0318 /
# +0.0359 / +0.0297（全距 0.0169，與跨 seed 的 sd 0.0174 同量級），
# 在 CPU 上則是 5 次位元相同的 +0.058480。
#
# 下面兩個測試鎖住的是「記錄下來的數字可以被重現」這個性質本身，
# 不是某個特定 device 的行為。

def _train_a_few_steps(model, dataset, config, device, n_steps: int = 3) -> None:
    """跑幾個 optimizer step，讓 BatchNorm running stats 離開初始值。"""
    loader = DataLoader(dataset, batch_size=4, shuffle=False,
                        collate_fn=multiplex_collate, num_workers=0)
    criterion = build_criterion(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step, batch in enumerate(loader):
        if step >= n_steps:
            break
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor)
                     else [t.to(device) for t in v]
                     if isinstance(v, list) and v and isinstance(v[0], torch.Tensor)
                     else v)
                 for k, v in batch.items()}
        y_hat, extras = model(batch)
        loss, _ = model.compute_loss(y_hat, batch["y"], extras)
        opt.zero_grad()
        loss.backward()
        opt.step()


def test_checkpoint_reproduces_recorded_test_ic(
    val_dataset: MultiplexDataset,
    train_dataset: MultiplexDataset,
    config: dict,
    device: torch.device,
) -> None:
    """存檔當下記錄的 IC，換一個 model 實例重載後必須一致到 1e-4 內。

    這正是 train.py 寫 meta.json 的流程：evaluate 記數字 → save_checkpoint
    → （之後）load_checkpoint → evaluate。兩次 IC 對不上就代表記錄不可信。
    """
    set_seed(config["training"]["seed"])
    model = MAGNET(config).to(device)
    _train_a_few_steps(model, train_dataset, config, device)

    loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                        collate_fn=multiplex_collate, num_workers=0)

    # 1) 評估並「記錄」——等同 meta.json 的 test_metrics
    recorded = evaluate(model, loader, device)
    recorded_ic = recorded["IC"]

    # 2) 存檔（權重此刻的狀態）
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "best.pt"
        save_checkpoint(ckpt, model, optimizer=None, epoch=0,
                        best_val_ic=recorded_ic)

        # 3) 全新實例重載後重評
        reloaded = MAGNET(config).to(device)
        load_checkpoint(ckpt, reloaded, optimizer=None, map_location=device)
        replay = evaluate(reloaded, loader, device)

    drift = abs(replay["IC"] - recorded_ic)
    assert drift < 1e-4, (
        f"checkpoint 無法重現自己記錄的 test IC："
        f"記錄 {recorded_ic:+.6f} vs 重載 {replay['IC']:+.6f}（差 {drift:.2e}）。"
        f"metric 若算在非確定性 device 上就會這樣。"
    )


def test_evaluate_is_deterministic(
    val_dataset: MultiplexDataset,
    train_dataset: MultiplexDataset,
    config: dict,
    device: torch.device,
) -> None:
    """同一份權重連評兩次，預測必須位元相同。

    MPS 上這條會掛（scatter-add 加總順序不固定），CPU 上恆成立。
    記錄用的 device 必須通過這個測試。
    """
    set_seed(config["training"]["seed"])
    model = MAGNET(config).to(device)
    _train_a_few_steps(model, train_dataset, config, device)

    loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                        collate_fn=multiplex_collate, num_workers=0)
    first  = evaluate(model, loader, device)
    second = evaluate(model, loader, device)

    max_dev = float(np.abs(
        first["predictions"].y_hat.to_numpy()
        - second["predictions"].y_hat.to_numpy()
    ).max())
    assert max_dev == 0.0, (
        f"同一份權重兩次評估的預測不同（最大差 {max_dev:.3e}）——"
        f"此 device 不可用於產生記錄值"
    )
    assert first["IC"] == second["IC"]


def test_eval_device_is_reproducible(config: dict) -> None:
    """base.yaml 的 eval_device 必須是已知可重現的 device。

    這是設定層的護欄：把它改成 mps/auto 會讓所有 meta.json 重新變成
    無法重現的單次抽樣，而且不會有任何錯誤訊息。
    """
    eval_device = config["training"].get("eval_device", "cpu")
    assert eval_device == "cpu", (
        f"training.eval_device = {eval_device!r}；記錄用的評估只允許 cpu，"
        f"因為 MPS/CUDA 的 scatter-add 聚合順序不保證 run 間一致。"
        f"訓練仍可用 training.device 走 MPS。"
    )
