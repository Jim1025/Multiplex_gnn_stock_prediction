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

from src.dataset.multiplex_dataset import TW_CODES
from src.models.multiplex_gnn import MAGNET
from src.models.prediction_head import CombinedLoss
from src.train.metrics import (
    aggregate_ic,
    long_short_metrics,
    rank_bucket_returns,
    regression_metrics,
)
from src.train.utils import batch_to_device


@torch.no_grad()
def evaluate(
    model:      MAGNET,
    loader:     DataLoader,
    device:     torch.device,
    criterion:  Optional[CombinedLoss] = None,
    eval_cfg:   Optional[dict] = None,
) -> dict:
    """
    跑一個 DataLoader 並回傳完整評估結果。

    Args:
        model     : MAGNET 實例（會切到 eval 模式）
        loader    : DataLoader（由 multiplex_collate 整理 batch）
        device    : torch.device
        criterion : 可選的 CombinedLoss；提供時會計算 loss 與分量
        eval_cfg  : 可選的 base.yaml `evaluation` 區塊；提供 `portfolio`
                    時才計算組合指標（超參一律由 config 提供，此處不設預設值）

    Returns:
        dict 包含：
            loss_total / loss_mse / loss_rank / loss_align : float（criterion 提供時）
            MSE / MAE / RMSE / R2 / R2_zero                 : float
            IC / ICIR / RankIC / RankICIR                   : float
            Sharpe / mean_daily_pnl / hit_rate / ...        : float（eval_cfg 提供時）
            rank_bucket_returns : list[float]（eval_cfg 提供時）
            predictions : pd.DataFrame [target_date, ticker, y_hat, y]
    """
    model.eval()

    # 逐日累積（每日一個 n2 維 cross-section）
    daily_y_hats: list[np.ndarray] = []
    daily_ys:     list[np.ndarray] = []
    pred_rows:    list[dict] = []

    loss_total_sum    = 0.0
    loss_mse_sum      = 0.0
    loss_rank_sum     = 0.0
    loss_align_sum    = 0.0
    loss_variance_sum = 0.0
    n_loss_samples    = 0   # 以 batch B 為單位的加權因子

    # TW ticker 順序作為輸出標籤（預測目標為 TW(t+1) log_return）。
    # E5：改為向 loader 持有的 Dataset 詢問，而非用模組層級的 k7 常數。
    # 標籤與 y_hat 的欄位順序若對不上，predictions CSV 會把每一欄掛到錯的
    # 公司；下游指標照算不誤，錯誤完全靜默。故下方另加欄數檢查。
    tw_labels = list(getattr(getattr(loader, "dataset", None), "tw_codes", TW_CODES))

    for batch in loader:
        batch = batch_to_device(batch, device)
        y_hat, extras = model(batch)        # y_hat: [B, n2]
        y = batch["y"]                       # [B, n2]

        if y_hat.size(1) != len(tw_labels):
            raise ValueError(
                f"預測欄數 {y_hat.size(1)} 與 TW 標籤數 {len(tw_labels)} 不符——"
                f"模型與 Dataset 可能屬於不同 universe。"
            )

        if criterion is not None:
            loss, comps = criterion(
                y_hat=y_hat,
                y=y,
                h_L1=extras.get("h_L1"),
                h_L2=extras.get("h_L2"),
            )
            B = y.size(0)
            loss_total_sum    += float(loss.item()) * B
            loss_mse_sum      += comps["mse"]   * B
            loss_rank_sum     += comps["rank"]  * B
            loss_align_sum    += comps["align"] * B
            loss_variance_sum += comps.get("variance", 0.0) * B
            n_loss_samples    += B

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
        np.stack(daily_y_hats, axis=0) if daily_y_hats else np.zeros((0, len(tw_labels))),
        np.stack(daily_ys,     axis=0) if daily_ys     else np.zeros((0, len(tw_labels))),
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

    # 組合指標（pre-cost）：僅在 config 明確提供時計算，避免在此硬編超參
    pf_cfg = (eval_cfg or {}).get("portfolio")
    if pf_cfg:
        pf = long_short_metrics(
            daily_y_hats, daily_ys,
            n_side=int(pf_cfg["n_side"]),
            periods_per_year=int(pf_cfg["periods_per_year"]),
        )
        result["Sharpe"]         = pf["Sharpe"]
        result["mean_daily_pnl"] = pf["mean_daily_pnl"]
        result["std_daily_pnl"]  = pf["std_daily_pnl"]
        result["hit_rate"]       = pf["hit_rate"]
        result["cum_log_return"] = pf["cum_log_return"]
        result["ann_return"]     = pf["ann_return"]
        result["pf_n_days"]      = pf["n_days"]
        result["daily_pnl"]      = pf["daily_pnl"]
        result["rank_bucket_returns"] = rank_bucket_returns(
            daily_y_hats, daily_ys
        )["by_rank"]

    if criterion is not None and n_loss_samples > 0:
        result["loss_total"]    = loss_total_sum    / n_loss_samples
        result["loss_mse"]      = loss_mse_sum      / n_loss_samples
        result["loss_rank"]     = loss_rank_sum     / n_loss_samples
        result["loss_align"]    = loss_align_sum    / n_loss_samples
        result["loss_variance"] = loss_variance_sum / n_loss_samples

    return result
