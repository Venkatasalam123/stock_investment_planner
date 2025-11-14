"""Core services layer for external market data integrations."""

from services.core.llm import LLMServiceError, llm_pick_and_allocate
from services.core.market_data import StockSnapshot, fetch_snapshots
from services.core.market_mood import MarketMood, MarketMoodError, calculate_market_mood
from services.core.news import NewsItem, fetch_headlines
from services.core.nse import (
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

