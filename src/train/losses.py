"""
losses.py — M4 損失函數薄包裝
Corresponds to IMPLEMENTATION_SPEC §5.2

主要 CombinedLoss 已在 src/models/prediction_head.py 實作，
此模組只提供 build_criterion 工廠函數（讀 config）方便 train.py 呼叫。
"""

from __future__ import annotations

from src.models.prediction_head import CombinedLoss


def build_criterion(cfg: dict) -> CombinedLoss:
    """
    從整份 config 建立 CombinedLoss。

    Args:
        cfg: configs/base.yaml 全部解析結果

    Returns:
        CombinedLoss
    """
    return CombinedLoss(
        loss_cfg=cfg.get("loss_weights", {}),
        align_cfg=cfg.get("align_loss", {}),
    )
