"""
baseline_hats.py — M7 External Baseline #2: HATS (Hierarchical Graph Attention)

Reference:
    Kim, R., So, C. H., Jeong, M., Lee, S., Kim, J., & Kang, J. (2019).
    HATS: A hierarchical graph attention network for stock movement prediction.
    NeurIPS 2019 Workshop on Robust AI in Financial Services.
    Paper:  https://arxiv.org/abs/1908.07999
    Author: https://github.com/dmis-lab/hats

論文原始任務為 S&P 500 next-day 漲跌 classification，使用 Wikidata 多種
relation type（product-market、holding 等）。本檔改造為 log_return
regression，並簡化為單一 relation type = industry（我們的 PAIR_MAP 已含
industry 欄位）。核心技術（stock → sector → market 三層階層 + 跨層
attention）保留。

架構：
    x_seq_L2 [B, T, n, F]  ← 只用 TW 端序列（單市場 baseline）
        → SharedLSTM                → [B, n, H_lstm]   (stock-level)
        → sector mean-pool          → [B, S, H_lstm]   (S=4 sectors)
        → Sector-level GAT (S×S)    → [B, S, H_gat]    (跨 sector attention)
        → broadcast 回 stocks       → [B, n, H_gat]
        → Fusion MLP(stock, sector) → [B, n, d']
        → PredictionHead            → [B, n]

Sector 對應（由 src/dataset/config.py::PAIR_MAP[k]["industry"] 推得）：
    idx 0 = 光電    → AUOTY
    idx 1 = 半導體  → TSM, UMC, ASX, IMOS
    idx 2 = 電信    → CHT
    idx 3 = 電子    → HNHPF

只使用 TW 端資料（batch["x_seq_L2"]），完全忽略 ADR 序列與所有 ADR 圖。
用途是驗證「單市場 + 產業階層圖」能達到什麼 IC，作為 Tier 2 對照。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.dataset.config import PAIR_MAP
from src.models.encoders import SharedLSTM, GATEncoder, TypeProjection
from src.models.prediction_head import PredictionHead, CombinedLoss


# ---------------------------------------------------------------------------
# Sector mapping helper（從 PAIR_MAP 讀，保證與 dataloader ticker 順序對齊）
# ---------------------------------------------------------------------------

def _build_sector_mapping() -> tuple[list[str], list[int]]:
    """
    Returns:
        sectors      : list[str]  依字典序排序的獨立 industry 名稱
        stock2sector : list[int]  長度 = len(PAIR_MAP)，每檔 stock 對應的 sector index
    """
    tickers = list(PAIR_MAP.keys())
    industries = [PAIR_MAP[t]["industry"] for t in tickers]
    sectors = sorted(set(industries))
    stock2sector = [sectors.index(PAIR_MAP[t]["industry"]) for t in tickers]
    return sectors, stock2sector


class BaselineHATS(nn.Module):
    """
    HATS baseline: hierarchical stock → sector aggregation + sector-level attention.

    單市場（只用 TW L2）。industry 分層來自 PAIR_MAP。
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m_cfg    = cfg["model"]
        lstm_cfg = m_cfg["lstm"]
        gat_cfg  = m_cfg["gat"]
        proj_cfg = m_cfg["projection"]
        head_cfg = m_cfg["prediction_head"]

        H_lstm  = lstm_cfg["hidden_dim"]
        H_gat   = gat_cfg["hidden_dim"]
        d_prime = proj_cfg["d_prime"]

        # Stock-level 時序編碼（LSTM shared 就好，這個 baseline 只用 TW）
        self.lstm = SharedLSTM(lstm_cfg)

        # Sector 分層資訊
        sectors, stock2sector = _build_sector_mapping()
        self.n_sectors = len(sectors)
        # buffer：不參與梯度、但會 follow model 到 device
        self.register_buffer(
            "stock2sector",
            torch.tensor(stock2sector, dtype=torch.long),
            persistent=False,
        )
        # Sector 完全連通圖（固定 edge_index，S×S 全連通 + self-loop）
        s = self.n_sectors
        src = torch.arange(s).repeat_interleave(s)
        dst = torch.arange(s).repeat(s)
        self.register_buffer(
            "sector_edge_index",
            torch.stack([src, dst], dim=0),
            persistent=False,
        )
        # Sector-level GAT（吃 sector embeddings + 全連通邊）
        # 覆寫 use_edge_attr=False：sector 全連通圖無邊權，避免 lin_edge 成為死支
        sector_gat_cfg = {**gat_cfg, "use_edge_attr": False}
        self.sector_gat = GATEncoder(sector_gat_cfg)

        # Fusion MLP：把 stock-level LSTM 表示與 sector-level GAT 表示 concat 後投到 d'
        self.fusion = nn.Sequential(
            nn.Linear(H_lstm + H_gat, d_prime),
            nn.GELU(),
            nn.LayerNorm(d_prime),
        )

        # Predictor
        self.predictor = PredictionHead(head_cfg, d_prime=d_prime)

        # 統一 loss
        self.criterion = CombinedLoss(
            loss_cfg=cfg.get("loss_weights", {}),
            align_cfg=cfg.get("align_loss", {}),
        )

        assert lstm_cfg["input_dim"] == 9, (
            f"SharedLSTM input_dim 應與 TECH_FEATURE_COLS 維度一致（9），"
            f"當前為 {lstm_cfg['input_dim']}"
        )

    def _aggregate_by_sector(self, h_stock: Tensor) -> Tensor:
        """
        Args:
            h_stock: [B, n, H_lstm]

        Returns:
            h_sector: [B, S, H_lstm]  每個 sector 為其成員 stocks 的 mean pool
        """
        B, n, H = h_stock.shape
        S = self.n_sectors
        # 計數矩陣（避免 empty sector 分母為 0，加 1 保護）
        counts = torch.zeros(S, device=h_stock.device, dtype=h_stock.dtype)
        counts.index_add_(0, self.stock2sector,
                          torch.ones(n, device=h_stock.device, dtype=h_stock.dtype))
        counts = counts.clamp(min=1.0)  # 每個 sector 至少視為 1 檔（避免 NaN）

        # 對 batch 中每張快照做 scatter mean（用 index_add 逐檔股票加總）
        h_sector = torch.zeros(B, S, H, device=h_stock.device, dtype=h_stock.dtype)
        for i in range(n):
            s_idx = int(self.stock2sector[i].item())
            h_sector[:, s_idx] += h_stock[:, i]
        h_sector = h_sector / counts.view(1, S, 1)
        return h_sector

    def _apply_sector_gat(self, h_sector: Tensor) -> Tensor:
        """對 batch 內每張快照跑一次 sector-level GAT（S×S 全連通固定圖）。"""
        B, S, _ = h_sector.shape
        outs = []
        # 沒有 edge_attr（sector 全連通沒有邊權），傳 None
        for b in range(B):
            outs.append(self.sector_gat(h_sector[b], self.sector_edge_index, None))
        return torch.stack(outs, dim=0)   # [B, S, H_gat]

    def forward(self, batch: dict) -> tuple[Tensor, dict]:
        x_L2 = batch["x_seq_L2"]                            # [B, T, n, F]

        # Stock-level
        h_stock = self.lstm(x_L2)                           # [B, n, H_lstm]

        # Sector-level
        h_sector_pool = self._aggregate_by_sector(h_stock)  # [B, S, H_lstm]
        h_sector_gat  = self._apply_sector_gat(h_sector_pool)  # [B, S, H_gat]

        # Broadcast sector embedding 回每檔 stock
        # index_select 沿 dim=1 用 stock2sector 挑對應 sector
        h_sector_per_stock = h_sector_gat.index_select(1, self.stock2sector)  # [B, n, H_gat]

        # Fusion
        h_concat = torch.cat([h_stock, h_sector_per_stock], dim=-1)           # [B, n, H_lstm + H_gat]
        h_final  = self.fusion(h_concat)                                      # [B, n, d']

        y_hat = self.predictor(h_final)                                       # [B, n]

        # 對齊 MAGNET 簽名（extras 不用於 loss）
        extras = {
            "h_L1":    h_final,
            "h_L2":    h_final,
            "h_fused": h_final,
            "alpha":   torch.zeros(*h_final.shape[:-1], 1, device=h_final.device),
            "gate":    torch.zeros_like(h_final),
            "h_sector": h_sector_gat,   # 供分析：sector-level 表示
        }
        return y_hat, extras

    def compute_loss(
        self,
        y_hat:  Tensor,
        y:      Tensor,
        extras: dict,
    ) -> tuple[Tensor, dict]:
        return self.criterion(
            y_hat=y_hat,
            y=y,
            h_L1=extras.get("h_L1"),
            h_L2=extras.get("h_L2"),
        )
