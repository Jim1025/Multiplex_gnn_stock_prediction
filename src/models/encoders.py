"""
encoders.py — Phase 1 編碼器模組
Corresponds to IMPLEMENTATION_SPEC §3.1 / §3.2 / §3.3

三個類別：
  - SharedLSTM     : 時序編碼，L1 與 L2 共用同一組權重
  - GATEncoder     : 圖編碼，L1/L2 各自獨立一份
  - TypeProjection : 投影至共同 d' 維潛在空間，L1/L2 各自獨立
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATv2Conv


# ---------------------------------------------------------------------------
# SharedLSTM
# ---------------------------------------------------------------------------

class SharedLSTM(nn.Module):
    """
    時序編碼器（Shared LSTM）。
    L1 (ADR) 與 L2 (TW) 共用同一份 LSTM 權重，
    以保證「時序動態」概念跨市場一致，讓後續 alignment loss 有意義。

    Corresponds to IMPLEMENTATION_SPEC §3.1 Step 1

    Args:
        cfg (dict): base.yaml 中的 model.lstm 區塊，包含：
            input_dim  (int)  : 節點特徵維度 F（預設 9）
            hidden_dim (int)  : LSTM hidden size H_lstm（預設 64）
            num_layers (int)  : LSTM 層數（預設 1）
            T_history  (int)  : 回看步數 T（預設 20，僅供外部參照）
            bidirectional (bool): 預測任務固定 False，不可看未來
            dropout    (float): LSTM dropout（num_layers>1 時才生效）

    Shapes:
        forward input  x_seq : [B, T, n, F]
        forward output       : [B, n, H_lstm]
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.hidden_dim = cfg["hidden_dim"]
        self.num_layers = cfg["num_layers"]
        self.bidirectional = cfg.get("bidirectional", False)
        dropout = cfg.get("dropout", 0.0) if cfg["num_layers"] > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=cfg["input_dim"],
            hidden_size=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout,
        )

    def forward(self, x_seq: Tensor) -> Tensor:
        """
        Args:
            x_seq: [B, T, n, F]

        Returns:
            h: [B, n, H_lstm]  — 每個節點最後一步的 hidden state
        """
        B, T, n, F = x_seq.shape
        # reshape: [B, T, n, F] → [B*n, T, F]
        x = x_seq.permute(0, 2, 1, 3).reshape(B * n, T, F)
        _, (h_n, _) = self.lstm(x)
        # h_n: [num_layers * num_directions, B*n, H]
        # 取最後一層的正向 hidden
        h = h_n[-1]          # [B*n, H_lstm]
        h = h.reshape(B, n, self.hidden_dim)
        return h              # [B, n, H_lstm]


# ---------------------------------------------------------------------------
# GATEncoder
# ---------------------------------------------------------------------------

class GATEncoder(nn.Module):
    """
    圖注意力網路編碼器（GATv2，L1 與 L2 各自獨立一份）。
    使用 torch_geometric.nn.GATv2Conv，支援 edge_attr（邊權 |ρ|）。

    Corresponds to IMPLEMENTATION_SPEC §3.2 Step 2

    Args:
        cfg (dict): base.yaml 中的 model.gat 區塊，包含：
            hidden_dim  (int)  : GAT hidden size H_gat（預設 64）
            num_layers  (int)  : GAT 層數（預設 2）
            num_heads   (int)  : attention heads（預設 4）
            dropout     (float): attention dropout（預設 0.1）
            use_edge_attr (bool): 是否使用 edge_attr（預設 True）
            concat      (bool) : 中間層是否 concat heads（預設 True）

    Shapes:
        forward input  x          : [n, H_lstm]
                       edge_index : [2, E]
                       edge_attr  : [E, 1]
        forward output             : [n, H_gat]
            H_gat = hidden_dim * num_heads（中間層 concat=True 時）
                  = hidden_dim（最後一層 concat=False）
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        hidden_dim = cfg["hidden_dim"]
        num_layers = cfg["num_layers"]
        num_heads = cfg["num_heads"]
        dropout = cfg.get("dropout", 0.0)
        edge_dim = 1 if cfg.get("use_edge_attr", True) else None

        self.convs = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)

        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            # 最後一層：concat=False、heads=1 → 輸出維度 = hidden_dim
            # 其他層：concat=True → 輸出維度 = hidden_dim * num_heads
            in_dim = hidden_dim if i == 0 else hidden_dim * num_heads
            heads = 1 if is_last else num_heads
            concat = False if is_last else cfg.get("concat", True)
            self.convs.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                )
            )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        """
        Args:
            x          : [n, H_lstm]
            edge_index : [2, E]
            edge_attr  : [E, 1] or None

        Returns:
            x : [n, H_gat]  (H_gat = hidden_dim，最後一層 concat=False)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
                x = self.dropout(x)
        return x   # [n, H_gat]


# ---------------------------------------------------------------------------
# TypeProjection
# ---------------------------------------------------------------------------

class TypeProjection(nn.Module):
    """
    型別特定投影層（Type-Specific Projection）。
    將 L1/L2 的 GAT 輸出投影至同一個 d' 維潛在空間，
    讓後續 CrossLayerFusion 能做有意義的跨市場融合。

    Corresponds to IMPLEMENTATION_SPEC §3.3 Step 3

    Args:
        cfg (dict): base.yaml 中的 model.projection 區塊，包含：
            d_prime    (int)  : 共同潛在維度（預設 32）
            activation (str)  : 激活函數（預設 "gelu"）
            layer_norm (bool) : 是否加 LayerNorm（預設 True）
        in_dim (int): 輸入維度（= H_gat = gat.hidden_dim）

    Shapes:
        forward input  x : [n, in_dim] 或 [B, n, in_dim]
        forward output   : [n, d_prime] 或 [B, n, d_prime]
    """

    def __init__(self, cfg: dict, in_dim: int) -> None:
        super().__init__()
        d_prime = cfg["d_prime"]
        act_name = cfg.get("activation", "gelu").lower()
        use_ln = cfg.get("layer_norm", True)

        self.linear = nn.Linear(in_dim, d_prime)
        self.act = nn.GELU() if act_name == "gelu" else nn.ReLU()
        self.norm = nn.LayerNorm(d_prime) if use_ln else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [..., in_dim]

        Returns:
            x: [..., d_prime]
        """
        return self.norm(self.act(self.linear(x)))
