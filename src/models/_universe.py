"""
_universe.py — 模型層取得 universe 的單一入口（E6）

為什麼需要：
    E6 之前，需要知道節點結構的模型各自從 PAIR_MAP / ADR_TICKERS / TW_CODES
    推導，等於把「universe 是什麼」複製了四份，而且全部寫死成 k7。
    擴充後這些推導會產出長度 7 的索引去切 30/50 的張量。

    節點結構屬於資料定義，不是模型超參，因此與 Dataset 走同一條路：
    從 base.yaml 的 data.universe 讀。build_model(cfg) 的簽名不必改，
    換 universe 只要改 yaml 一處。

不放進 src/models/__init__.py 的原因：
    __init__.py 會 import 各模型類別，而各模型要 import 這個 helper，
    放在那裡會造成循環匯入。
"""

from __future__ import annotations

from src.dataset.config import Universe, load_universe


def universe_from_cfg(cfg: dict) -> Universe:
    """
    從 base.yaml 解析結果取出 universe。

    未指定 data.universe 時回傳 DEFAULT_UNIVERSE（k7）——舊的 config
    快照沒有這個欄位，必須維持可重跑。
    """
    return load_universe((cfg.get("data") or {}).get("universe"))
