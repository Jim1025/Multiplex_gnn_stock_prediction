from src.models.multiplex_gnn import MAGNET
from src.models.baseline_lstm import BaselineLSTM
from src.models.baseline_tw_gnn import BaselineTWGNN
from src.models.encoders import SharedLSTM, GATEncoder, TypeProjection
from src.models.fusion import CrossLayerFusion
from src.models.prediction_head import PredictionHead, CombinedLoss


VALID_ARCHITECTURES = ("magnet", "baseline_lstm", "baseline_tw_gnn", "magnet_no_a12")


def build_model(cfg: dict):
    """
    M6 Stage 0 model factory。

    依 cfg["model"]["architecture"] 分派到對應 baseline / MAGNET 變體。
    "magnet_no_a12" 是 MAGNET 的 ablation：保留架構，但跨層 ADR → TW 訊號零化。

    Args:
        cfg: 完整 base.yaml 解析結果（含 model / loss_weights / align_loss）

    Returns:
        nn.Module — forward 簽名統一為 (batch dict) → (y_hat, extras dict)
    """
    arch = str(cfg.get("model", {}).get("architecture", "magnet")).lower()
    if arch not in VALID_ARCHITECTURES:
        raise ValueError(
            f"未知 architecture={arch!r}；可選：{VALID_ARCHITECTURES}"
        )

    if arch == "baseline_lstm":
        return BaselineLSTM(cfg)
    if arch == "baseline_tw_gnn":
        return BaselineTWGNN(cfg)
    if arch == "magnet_no_a12":
        # 注入 disable_a12 flag（MAGNET 內部會讀）
        cfg_local = {**cfg, "model": {**cfg["model"], "disable_a12": True}}
        return MAGNET(cfg_local)
    return MAGNET(cfg)


__all__ = [
    "MAGNET",
    "BaselineLSTM",
    "BaselineTWGNN",
    "SharedLSTM",
    "GATEncoder",
    "TypeProjection",
    "CrossLayerFusion",
    "PredictionHead",
    "CombinedLoss",
    "build_model",
    "VALID_ARCHITECTURES",
]
