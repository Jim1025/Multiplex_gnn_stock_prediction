from src.models.multiplex_gnn import MAGNET
from src.models.baseline_lstm import BaselineLSTM
from src.models.baseline_tw_gnn import BaselineTWGNN
from src.models.baseline_advalstm import BaselineAdvALSTM
from src.models.baseline_hats import BaselineHATS
from src.models.baseline_mansf import BaselineMANSF
from src.models.baseline_hgt import BaselineHGT
from src.models.baseline_deltalag import BaselineDeltaLag
from src.models.baseline_meig import BaselineMEIG
from src.models.encoders import SharedLSTM, GATEncoder, TypeProjection
from src.models.fusion import CrossLayerFusion
from src.models.prediction_head import PredictionHead, CombinedLoss


VALID_ARCHITECTURES = (
    # M6 Stage 0 (內部 ablation)
    "magnet", "baseline_lstm", "baseline_tw_gnn", "magnet_no_a12",
    # M7 external baselines
    "adv_alstm", "hats", "man_sf", "hgt", "delta_lag", "meig",
    # M8 hierarchical A12 (strong identity pairs + learned weak links)
    "magnet_weak_free", "magnet_weak_industry",
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
    if arch == "man_sf":
        return BaselineMANSF(cfg)
    if arch == "hgt":
        return BaselineHGT(cfg)
    if arch == "delta_lag":
        return BaselineDeltaLag(cfg)
    if arch == "meig":
        return BaselineMEIG(cfg)
    if arch in ("magnet_weak_free", "magnet_weak_industry"):
        # 注入 weak_links.mode（MAGNET 內部會讀）；保留 cfg 原有的 lambda 等設定
        mode = "free" if arch == "magnet_weak_free" else "industry"
        weak_cfg = {**(cfg["model"].get("weak_links", {}) or {}), "mode": mode}
        cfg_local = {**cfg, "model": {**cfg["model"], "weak_links": weak_cfg}}
        return MAGNET(cfg_local)
    return MAGNET(cfg)


__all__ = [
    "MAGNET",
    "BaselineLSTM",
    "BaselineTWGNN",
    "BaselineAdvALSTM",
    "BaselineHATS",
    "BaselineMANSF",
    "BaselineHGT",
    "BaselineDeltaLag",
    "BaselineMEIG",
    "SharedLSTM",
    "GATEncoder",
    "TypeProjection",
    "CrossLayerFusion",
    "PredictionHead",
    "CombinedLoss",
    "build_model",
    "VALID_ARCHITECTURES",
]
