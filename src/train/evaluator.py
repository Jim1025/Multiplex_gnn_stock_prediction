"""
evaluator.py — M4 評估迴圈
Corresponds to IMPLEMENTATION_SPEC §6 (evaluation)

職責：
  - 對給定 DataLoader（val 或 test）跑一次 forward
  - 聚合所有 batch 的預測與真實值 → 計算 IC / RankIC / ICIR / MSE / MAE / RMSE
  - 同時收集 loss 分量（若提供 criterion）
  - 產出 predictions DataFrame（含 target_date / ticker / y_hat / y）供 artifact 上傳
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.dataset.multiplex_dataset import ADR_TICKERS, TW_CODES, N_NODES
from src.models.multiplex_gnn import MAGNET
from src.models.prediction_head import CombinedLoss
from src.train.metrics import aggregate_ic, regression_metrics
from src.train.utils import batch_to_device


@torch.no_grad()
def evaluate(
    model:      MAGNET,
    loader:     DataLoader,
    device:     torch.device,
    criterion:  Optional[CombinedLoss] = None,
) -> dict:
    """
    跑一個 DataLoader 並回傳完整評估結果。

    Args:
        model     : MAGNET 實例（會切到 eval 模式）
        loader    : DataLoader（由 multiplex_collate 整理 batch）
        device    : torch.device
        criterion : 可選的 CombinedLoss；提供時會計算 loss 與分量

    Returns:
        dict 包含：
            loss_total / loss_mse / loss_rank / loss_align : float（criterion 提供時）
            MSE / MAE / RMSE                                : float
            IC / ICIR / RankIC / RankICIR                   : float
            predictions : pd.DataFrame [target_date, ticker, y_hat, y]
    """
    model.eval()

    # 逐日累積（n=7 cross-section per day）
    daily_y_hats: list[np.ndarray] = []
    daily_ys:     list[np.ndarray] = []
    pred_rows:    list[dict] = []

    loss_total_sum = 0.0
    loss_mse_sum   = 0.0
    loss_rank_sum  = 0.0
    loss_align_sum = 0.0
    n_loss_samples = 0   # 以 batch B 為單位的加權因子

    # TW ticker 順序作為輸出標籤（預測目標為 TW(t+1) log_return）
    tw_labels = TW_CODES

    for batch in loader:
        batch = batch_to_device(batch, device)
        y_hat, extras = model(batch)        # y_hat: [B, n]
        y = batch["y"]                       # [B, n]

        if criterion is not None:
            loss, comps = criterion(
                y_hat=y_hat,
                y=y,
                h_L1=extras.get("h_L1"),
                h_L2=extras.get("h_L2"),
            )
            B = y.size(0)
            loss_total_sum += float(loss.item()) * B
            loss_mse_sum   += comps["mse"]   * B
            loss_rank_sum  += comps["rank"]  * B
            loss_align_sum += comps["align"] * B
            n_loss_samples += B

        # 攤平成「每日 cross-section」
        yh_np = y_hat.detach().cpu().numpy()  # [B, n]
        y_np  = y.detach().cpu().numpy()      # [B, n]
        dates = batch["target_date"]          # list[str], len=B

        for b in range(yh_np.shape[0]):
            daily_y_hats.append(yh_np[b])
            daily_ys.append(y_np[b])
            for j, ticker in enumerate(tw_labels):
                pred_rows.append({
                    "target_date": dates[b],
                    "ticker":      ticker,
                    "y_hat":       float(yh_np[b, j]),
                    "y":           float(y_np[b, j]),
                })

    # 聚合
    ic_dict = aggregate_ic(daily_y_hats, daily_ys)
    reg_dict = regression_metrics(
        np.stack(daily_y_hats, axis=0) if daily_y_hats else np.zeros((0, N_NODES)),
        np.stack(daily_ys,     axis=0) if daily_ys     else np.zeros((0, N_NODES)),
    )

    result: dict = {
        **reg_dict,
        "IC":       ic_dict["IC"],
        "ICIR":     ic_dict["ICIR"],
        "RankIC":   ic_dict["RankIC"],
        "RankICIR": ic_dict["RankICIR"],
        "daily_IC":     ic_dict["daily_IC"],
        "daily_RankIC": ic_dict["daily_RankIC"],
        "predictions":  pd.DataFrame(pred_rows),
    }

    if criterion is not None and n_loss_samples > 0:
        result["loss_total"] = loss_total_sum / n_loss_samples
        result["loss_mse"]   = loss_mse_sum   / n_loss_samples
        result["loss_rank"]  = loss_rank_sum  / n_loss_samples
        result["loss_align"] = loss_align_sum / n_loss_samples

    return result
