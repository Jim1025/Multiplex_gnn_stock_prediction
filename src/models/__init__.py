from src.models.multiplex_gnn import MAGNET
from src.models.baseline_lstm import BaselineLSTM
from src.models.baseline_tw_gnn import BaselineTWGNN
from src.models.baseline_advalstm import BaselineAdvALSTM
from src.models.baseline_hats import BaselineHATS
from src.models.encoders import SharedLSTM, GATEncoder, TypeProjection
from src.models.fusion import CrossLayerFusion
from src.models.prediction_head import PredictionHead, CombinedLoss


VALID_ARCHITECTURES = (
    # M6 Stage 0 (內部 ablation)
    "magnet", "baseline_lstm", "baseline_tw_gnn", "magnet_no_a12",
    # M7 external baselines
    "adv_alstm", "hats",
)


def build_model(cfg: dict):
    """
    Model factory：依 cfg["model"]["architecture"] 分派。

    支援：
        magnet             — 完整 MAGNET (reference)
        baseline_lstm      — LSTM-only（Stage 0 ablation）
        baseline_tw_gnn    — TW-only 單層 GNN（Stage 0 ablation）
        magnet_no_a12      — MAGNET 但切斷 A12 跨層訊號（Stage 0 ablation）
        adv_alstm          — Adv-ALSTM 外部 baseline (Feng 2019, IJCAI)

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
    if arch == "adv_alstm":
        return BaselineAdvALSTM(cfg)
    if arch == "hats":
        return BaselineHATS(cfg)
    return MAGNET(cfg)


__all__ = [
    "MAGNET",
    "BaselineLSTM",
    "BaselineTWGNN",
    "BaselineAdvALSTM",
    "BaselineHATS",
    "SharedLSTM",
    "GATEncoder",
    "TypeProjection",
    "CrossLayerFusion",
    "PredictionHead",
    "CombinedLoss",
    "build_model",
    "VALID_ARCHITECTURES",
]
