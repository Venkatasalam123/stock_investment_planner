"""Market mood (Fear/Greed) index calculator for Indian markets."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


@dataclass
class MarketMood:
    """Market mood index with interpretation."""
    index: float  # 0-100 scale
    sentiment: str  # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    description: str
    recommendation: str


class MarketMoodError(RuntimeError):
    """Raised when market mood cannot be calculated."""


def _check_rate_limit_error(exc: Exception) -> bool:
    """Check if error is due to rate limiting."""
    error_str = str(exc).lower()
    return "rate limit" in error_str or "too many requests" in error_str


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=lambda e: isinstance(e, Exception) and (_check_rate_limit_error(e) or isinstance(e, (ConnectionError, TimeoutError))),
)
def _fetch_nifty_data() -> pd.DataFrame:
    """Fetch Nifty 50 index data."""
    time.sleep(0.2)  # Small delay to reduce rate limiting
    ticker = yf.Ticker("^NSEI")
    history = ticker.history(period="3mo", interval="1d")
    if history.empty:
        raise MarketMoodError("Unable to fetch Nifty 50 data")
    return history


def _calculate_volatility_score(returns: pd.Series) -> float:
    """Calculate volatility component (0-25 points).
    
    Higher volatility = more fear.
    """
    if returns.empty:
        return 12.5  # Neutral
    
    volatility = returns.std()
    # Normalize: typical daily volatility is 1-2%, extreme is 3%+
    if volatility < 0.01:
        return 20  # Low volatility (greed)
    elif volatility < 0.015:
        return 12.5  # Normal volatility (neutral)
    elif volatility < 0.025:
        return 5  # High volatility (fear)
    else:
        return 0  # Extreme volatility (extreme fear)


def _calculate_momentum_score(returns: pd.Series, rsi: Optional[float]) -> float:
    """Calculate momentum component (0-25 points).
    
    Positive momentum = greed, negative = fear.
    """
    if returns.empty:
        return 12.5  # Neutral
    
    # Recent returns (last 5 days)
    recent_returns = returns.tail(5).mean() if len(returns) >= 5 else returns.mean()
    
    # Convert to score (0-25)
    # Positive returns = greed, negative = fear
    momentum_score = 12.5 + (recent_returns * 1000)  # Scale returns
    momentum_score = max(0, min(25, momentum_score))  # Clamp to 0-25
    
    # Adjust based on RSI if available
    if rsi is not None:
        if rsi < 30:  # Oversold (fear)
            momentum_score = max(0, momentum_score - 5)
        elif rsi > 70:  # Overbought (greed)
            momentum_score = min(25, momentum_score + 5)
    
    return momentum_score


def _calculate_market_position_score(history: pd.DataFrame) -> float:
    """Calculate market position component (0-25 points).
    
    Based on distance from recent highs/lows.
    """
    if history.empty or "Close" not in history.columns:
        return 12.5  # Neutral
    
    close = history["Close"]
    current_price = close.iloc[-1]
    
    # Recent high/low (last 30 days)
    recent_window = close.tail(30) if len(close) >= 30 else close
    recent_high = recent_window.max()
    recent_low = recent_window.min()
    
    if recent_high == recent_low:
        return 12.5  # Neutral
    
    # Position in recent range (0 = at low, 1 = at high)
    position = (current_price - recent_low) / (recent_high - recent_low)
    
    # Convert to score: near low = fear, near high = greed
    return position * 25


def _calculate_volume_score(history: pd.DataFrame) -> float:
    """Calculate volume component (0-25 points).
    
    High volume on down days = fear, high volume on up days = greed.
    """
    if history.empty or "Volume" not in history.columns or "Close" not in history.columns:
        return 12.5  # Neutral
    
    volume = history["Volume"]
    close = history["Close"]
    
    # Calculate price changes
    price_changes = close.pct_change().dropna()
    
    # Recent volume vs average
    recent_volume = volume.tail(5).mean() if len(volume) >= 5 else volume.mean()
    avg_volume = volume.mean()
    
    if avg_volume == 0:
        return 12.5  # Neutral
    
    volume_ratio = recent_volume / avg_volume
    
    # Recent price trend
    recent_trend = price_changes.tail(5).mean() if len(price_changes) >= 5 else price_changes.mean()
    
    # High volume on down days = fear, high volume on up days = greed
    if recent_trend < 0 and volume_ratio > 1.2:
        return 5  # Fear (panic selling)
    elif recent_trend > 0 and volume_ratio > 1.2:
        return 20  # Greed (buying frenzy)
    elif recent_trend < 0:
        return 10  # Mild fear
    elif recent_trend > 0:
        return 15  # Mild greed
    else:
        return 12.5  # Neutral


def calculate_market_mood() -> MarketMood:
    """Calculate market mood (Fear/Greed) index for Indian markets.
    
    Returns a MarketMood object with index (0-100) and interpretation.
    Index ranges:
    - 0-25: Extreme Fear (buying opportunity)
    - 26-45: Fear
    - 46-55: Neutral
    - 56-75: Greed
    - 76-100: Extreme Greed (be cautious)
    """
    try:
        history = _fetch_nifty_data()
    except Exception as exc:
        raise MarketMoodError(f"Failed to fetch market data: {exc}") from exc
    
    if history.empty:
        raise MarketMoodError("No market data available")
    
    close = history["Close"]
    returns = close.pct_change().dropna()
    
    # Calculate RSI for Nifty
    try:
        import pandas_ta as ta
        rsi_series = ta.rsi(close, length=14)
        rsi = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else None
    except Exception:
        rsi = None
    
    # Calculate components (each 0-25 points, total 0-100)
    volatility_score = _calculate_volatility_score(returns)
    momentum_score = _calculate_momentum_score(returns, rsi)
    position_score = _calculate_market_position_score(history)
    volume_score = _calculate_volume_score(history)
    
    # Total index
    mood_index = volatility_score + momentum_score + position_score + volume_score
    
    # Determine sentiment
    if mood_index <= 25:
        sentiment = "Extreme Fear"
        description = "Market is in extreme fear. High volatility, negative momentum, and selling pressure indicate panic."
        recommendation = "Consider increasing allocation - fear often creates buying opportunities. However, ensure fundamentals are strong."
    elif mood_index <= 45:
        sentiment = "Fear"
        description = "Market shows fear with negative sentiment and increased volatility."
        recommendation = "Good time to accumulate quality stocks at lower prices. Be selective and focus on fundamentals."
    elif mood_index <= 55:
        sentiment = "Neutral"
        description = "Market sentiment is balanced with no extreme emotions."
        recommendation = "Normal market conditions. Proceed with standard investment strategy based on individual stock analysis."
    elif mood_index <= 75:
        sentiment = "Greed"
        description = "Market shows greed with positive momentum and buying interest."
        recommendation = "Be cautious. Market may be overvalued. Consider taking profits and reducing exposure to risky positions."
    else:
        sentiment = "Extreme Greed"
        description = "Market is in extreme greed. High prices, low volatility, and strong buying indicate euphoria."
        recommendation = "High risk of correction. Consider reducing positions, taking profits, and waiting for better entry points."
    
    return MarketMood(
        index=round(mood_index, 2),
        sentiment=sentiment,
        description=description,
        recommendation=recommendation,
    )

