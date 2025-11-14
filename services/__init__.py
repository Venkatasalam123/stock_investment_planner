"""Service layer for external market data integrations."""

# Re-export from core for backward compatibility
from services.core import (
    LLMServiceError,
    llm_pick_and_allocate,
    StockSnapshot,
    fetch_snapshots,
    MarketMood,
    MarketMoodError,
    calculate_market_mood,
    NewsItem,
    fetch_headlines,
    AVAILABLE_INDICES,
    fetch_fii_trend,
    fetch_index_constituents,
    fetch_nifty_500_constituents,
)

__all__ = [
    # LLM
    "LLMServiceError",
    "llm_pick_and_allocate",
    # Market Data
    "StockSnapshot",
    "fetch_snapshots",
    # Market Mood
    "MarketMood",
    "MarketMoodError",
    "calculate_market_mood",
    # News
    "NewsItem",
    "fetch_headlines",
    # NSE
    "AVAILABLE_INDICES",
    "fetch_fii_trend",
    "fetch_index_constituents",
    "fetch_nifty_500_constituents",
]

