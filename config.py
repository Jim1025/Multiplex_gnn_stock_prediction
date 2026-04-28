# config.py
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