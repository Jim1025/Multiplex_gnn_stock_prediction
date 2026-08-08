# config.py
#
# ─────────────────────────────────────────────────────────────────────────
# E3 變更說明（經授權變更資料層，2026-08-08）
#
# 為什麼要改：
#   原本整個 universe 只由下面的 PAIR_MAP 表達，而它是「一對一 dict」。
#   由此推導出的下游結構寫死了三個假設：
#     (1) n1 == n2（方陣）
#     (2) 每個節點都有配對（恆等邊 = 對角線）
#     (3) 配對靠共用索引，亦即硬編碼 p(j) = j
#   擴充後的 universe（US 30 / TW 50、配對率 14%）三個假設全部不成立：
#   節點數不對稱、43 檔台股沒有對應美股、恆等邊是稀疏索引映射而非對角線。
#   論文裡的 h1_j = h1_{p(j)} + sum_i B_eff[i,j] h1_i（無配對節點無第一項）
#   是這裡目前無法實例化的一般化形式。
#
# 為什麼用「可加式」而非直接改寫：
#   E4（graph_builder）/ E5（multiplex_dataset）/ E6（models）要分批遷移。
#   若此處直接改掉介面，中間每一步 repo 都是壞的，而且 freeze_k7.py --verify
#   會在遷移完成前一直失敗，等於失去回歸保護。
#   因此 PAIR_MAP 與其三個 helper 完全不動，新的 Universe API 並存；
#   下游逐一遷移完畢後，PAIR_MAP 才降級為 k7 的相容層。
#
# 影響範圍：
#   本次不改變任何既有符號的值或語意，故 k=7 的模型輸入不受影響。
#   驗證：freeze_k7.py --verify 應維持 17/17 通過。
# ─────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PAIR_MAP = {
    "TSM":   {"tw": "2330", "tier": 1, "market": "NYSE",   "industry": "半導體", "company": "台積電"},
    "UMC":   {"tw": "2303", "tier": 1, "market": "NYSE",   "industry": "半導體", "company": "聯電"},
    "ASX":   {"tw": "3711", "tier": 1, "market": "NYSE",   "industry": "半導體", "company": "日月光"},
    "CHT":   {"tw": "2412", "tier": 1, "market": "NYSE",   "industry": "電信",   "company": "中華電信"},
    "IMOS":  {"tw": "8150", "tier": 1, "market": "NASDAQ", "industry": "半導體", "company": "南茂"},
    "AUOTY": {"tw": "2409", "tier": 2, "market": "OTC",    "industry": "光電",   "company": "友達"},
    # "ASUUY": {"tw": "2357", "tier": 2, "market": "OTC",    "industry": "電子",   "company": "華碩"},
    "HNHPF": {"tw": "2317", "tier": 2, "market": "OTC",    "industry": "電子",   "company": "鴻海"},
}

# 衍生輔助函式
def get_pair_dict() -> dict:
    """回傳簡單的 ADR→TW 對應，相容於既有程式。"""
    return {k: v["tw"] for k, v in PAIR_MAP.items()}

def get_tier_1_pairs() -> dict:
    """只回傳主板配對（用於嚴謹路徑訓練）。"""
    return {k: v["tw"] for k, v in PAIR_MAP.items() if v["tier"] == 1}

def get_industry(adr_ticker: str) -> str:
    return PAIR_MAP[adr_ticker]["industry"]


# ═════════════════════════════════════════════════════════════════════════
# Universe API（E3 新增）
# ═════════════════════════════════════════════════════════════════════════

_ROOT = Path(__file__).resolve().parents[2]
_TW50_JSON = _ROOT / "configs" / "universe" / "universe_tw_2019.json"

DEFAULT_UNIVERSE = "k7"
VALID_UNIVERSES = ("k7", "tw50")


@dataclass(frozen=True)
class Universe:
    """
    一個 universe 的完整定義。節點順序即張量索引順序，**不可任意變動**——
    快照、權重矩陣與凍結基準都依賴它。

    Attributes
    ----------
    name        : "k7" | "tw50"
    us_nodes    : L1 節點代號，順序固定
    tw_nodes    : L2 節點代號（預測目標），順序固定
    pairing     : 部分映射 us_ticker -> tw_code。
                  len(pairing) 可遠小於 len(tw_nodes)——這正是擴充的重點。
    industry    : 代號 -> 產業別，US / TW 兩側合併於同一 dict
    """

    name:     str
    us_nodes: tuple[str, ...]
    tw_nodes: tuple[str, ...]
    pairing:  dict[str, str]
    industry: dict[str, str]

    # ── 尺寸 ────────────────────────────────────────────────
    @property
    def n_l1(self) -> int:
        return len(self.us_nodes)

    @property
    def n_l2(self) -> int:
        return len(self.tw_nodes)

    @property
    def n_pairs(self) -> int:
        return len(self.pairing)

    @property
    def pairing_rate(self) -> float:
        return self.n_pairs / self.n_l2 if self.n_l2 else 0.0

    # ── 索引 ────────────────────────────────────────────────
    @property
    def pair_index(self) -> tuple[int, ...]:
        """
        每個 TW 節點 j 對應的 US 節點索引 p(j)；無配對者為 -1。

        這是取代「恆等邊 = 對角線」假設的核心結構。k7 下它恰為
        (0, 1, ..., 6)，故舊行為是新結構的特例。
        """
        us_idx = {t: i for i, t in enumerate(self.us_nodes)}
        tw_to_us = {tw: us_idx[us] for us, tw in self.pairing.items()}
        return tuple(tw_to_us.get(code, -1) for code in self.tw_nodes)

    def validate(self) -> None:
        if len(set(self.us_nodes)) != len(self.us_nodes):
            raise ValueError(f"[{self.name}] us_nodes 有重複代號")
        if len(set(self.tw_nodes)) != len(self.tw_nodes):
            raise ValueError(f"[{self.name}] tw_nodes 有重複代號")
        bad_us = set(self.pairing) - set(self.us_nodes)
        if bad_us:
            raise ValueError(f"[{self.name}] pairing 的 US 端不在 us_nodes：{sorted(bad_us)}")
        bad_tw = set(self.pairing.values()) - set(self.tw_nodes)
        if bad_tw:
            raise ValueError(f"[{self.name}] pairing 的 TW 端不在 tw_nodes：{sorted(bad_tw)}")
        if len(set(self.pairing.values())) != len(self.pairing):
            raise ValueError(f"[{self.name}] 同一檔 TW 被多個 US 配對（恆等邊須為 1-to-1）")
        missing = [c for c in self.tw_nodes if c not in self.industry]
        if missing:
            raise ValueError(f"[{self.name}] 以下 TW 節點缺產業別：{missing}")


def _build_k7() -> Universe:
    """由 PAIR_MAP 推導，順序與既有 ADR_TICKERS / TW_CODES 完全一致。"""
    us = tuple(PAIR_MAP.keys())
    tw = tuple(v["tw"] for v in PAIR_MAP.values())
    industry = {k: v["industry"] for k, v in PAIR_MAP.items()}
    industry.update({v["tw"]: v["industry"] for v in PAIR_MAP.values()})
    return Universe(
        name="k7",
        us_nodes=us,
        tw_nodes=tw,
        pairing={k: v["tw"] for k, v in PAIR_MAP.items()},
        industry=industry,
    )


def _build_tw50() -> Universe:
    """
    由 configs/universe/universe_tw_2019.json 載入（E1 定義、E2 驗證）。

    節點順序：US = 7 檔 ADR（PAIR_MAP 順序）+ 資訊源（JSON 順序）；
              TW = JSON 的 tw_nodes 順序（已依代號排序）。
    """
    if not _TW50_JSON.exists():
        raise FileNotFoundError(f"找不到 universe 定義：{_TW50_JSON}")
    u = json.loads(_TW50_JSON.read_text())

    adr = [a["ticker"] for a in u["us_layer"]["adr_pairs"]]
    info = [i["ticker"] for i in u["us_layer"]["info_sources"]]
    us = tuple(adr + info)
    tw = tuple(n["code"] for n in u["tw_nodes"])

    industry = {n["code"]: n["industry"] for n in u["tw_nodes"]}
    industry.update({a["ticker"]: PAIR_MAP[a["ticker"]]["industry"]
                     for a in u["us_layer"]["adr_pairs"] if a["ticker"] in PAIR_MAP})
    industry.update({i["ticker"]: i["category"] for i in u["us_layer"]["info_sources"]})

    pairing = {a["ticker"]: a["tw"] for a in u["us_layer"]["adr_pairs"]
               if a["tw"] in set(tw)}
    return Universe(name="tw50", us_nodes=us, tw_nodes=tw,
                    pairing=pairing, industry=industry)


_CACHE: dict[str, Universe] = {}


def load_universe(name: str | None = None) -> Universe:
    """
    取得指定 universe。未指定時回傳 DEFAULT_UNIVERSE（k7）。

    保持 k7 為預設，是為了讓 E4-E6 遷移期間既有路徑行為不變，
    且 freeze_k7.py --verify 全程可執行。
    """
    key = (name or DEFAULT_UNIVERSE).lower()
    if key not in VALID_UNIVERSES:
        raise ValueError(f"未知 universe={name!r}；可選：{VALID_UNIVERSES}")
    if key not in _CACHE:
        u = _build_k7() if key == "k7" else _build_tw50()
        u.validate()
        _CACHE[key] = u
    return _CACHE[key]
