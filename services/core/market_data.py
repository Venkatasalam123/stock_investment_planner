"""Market data helpers for price history and fundamental metrics."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class MarketDataError(RuntimeError):
    """Raised when price or fundamental data cannot be fetched."""


@dataclass
class StockSnapshot:
    symbol: str
    ticker: str
    short_name: str | None
    price_history: pd.DataFrame = field(repr=False)
    change_1m: float | None
    change_6m: float | None
    fundamentals: Dict[str, Any]
    revenue_growth_yoy: float | None
    revenue_cagr_3y: float | None
    dividend_yield: float | None
    free_cash_flow: float | None
    operating_margin: float | None
    promoter_holding_pct: float | None = None
    promoter_holding_change: float | None = None
    forecast: pd.DataFrame | None = field(default=None, repr=False)
    forecast_slope: float | None = None
    rsi_14: float | None = None
    moving_average_50: float | None = None
    moving_average_200: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    bollinger_upper: float | None = None
    bollinger_lower: float | None = None
    bollinger_middle: float | None = None
    avg_volume_20: float | None = None
    volume_ratio: float | None = None
    dist_52w_high: float | None = None
    dist_52w_low: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ticker": self.ticker,
            "short_name": self.short_name,
            "change_1m": self.change_1m,
            "change_6m": self.change_6m,
            "fundamentals": self.fundamentals,
            "revenue_growth_yoy": self.revenue_growth_yoy,
            "revenue_cagr_3y": self.revenue_cagr_3y,
            "dividend_yield": self.dividend_yield,
            "free_cash_flow": self.free_cash_flow,
            "operating_margin": self.operating_margin,
            "promoter_holding_pct": self.promoter_holding_pct,
            "promoter_holding_change": self.promoter_holding_change,
            "forecast_slope": self.forecast_slope,
            "rsi_14": self.rsi_14,
            "moving_average_50": self.moving_average_50,
            "moving_average_200": self.moving_average_200,
            "macd": self.macd,
            "macd_signal": self.macd_signal,
            "bollinger_upper": self.bollinger_upper,
            "bollinger_lower": self.bollinger_lower,
            "bollinger_middle": self.bollinger_middle,
            "avg_volume_20": self.avg_volume_20,
            "volume_ratio": self.volume_ratio,
            "dist_52w_high": self.dist_52w_high,
            "dist_52w_low": self.dist_52w_low,
        }


def _normalise_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol.endswith(".NS"):
        return symbol
    return f"{symbol}.NS"


def _calculate_returns(history: pd.DataFrame) -> Tuple[float | None, float | None]:
    close = history["Close"]
    if close.empty:
        return None, None

    change_6m = None
    try:
        change_6m = float(close.iloc[-1] / close.iloc[0] - 1)
    except (ZeroDivisionError, IndexError):
        change_6m = None

    change_1m = None
    if len(close) >= 21:  # approx. 1 trading month
        try:
            change_1m = float(close.iloc[-1] / close.iloc[-21] - 1)
        except (ZeroDivisionError, IndexError):
            change_1m = None

    return change_1m, change_6m


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
def _download_history(ticker: yf.Ticker) -> pd.DataFrame:
    time.sleep(0.1)  # Small delay to reduce rate limiting
    history = ticker.history(period="1y", interval="1d", auto_adjust=True)
    if history.empty:
        raise MarketDataError("Empty history response from yfinance")
    return history


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=lambda e: isinstance(e, Exception) and (_check_rate_limit_error(e) or isinstance(e, (ConnectionError, TimeoutError))),
)
def _fetch_company_info(ticker: yf.Ticker) -> Dict[str, Any]:
    time.sleep(0.1)  # Small delay to reduce rate limiting
    info = ticker.get_info()
    if not isinstance(info, dict) or not info:
        raise MarketDataError("Company fundamentals not available")
    return info


def _extract_fundamentals(info: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["totalRevenue", "trailingEps", "trailingPE", "returnOnEquity", "debtToEquity"]
    fundamentals = {key: info.get(key) for key in keys}

    # Attempt to coerce to float where sensible.
    for key in ["trailingEps", "trailingPE", "returnOnEquity", "debtToEquity"]:
        value = fundamentals.get(key)
        if value is not None:
            try:
                fundamentals[key] = float(value)
            except (TypeError, ValueError):
                fundamentals[key] = None

    revenue = fundamentals.get("totalRevenue")
    if revenue is not None:
        try:
            fundamentals["totalRevenue"] = float(revenue)
        except (TypeError, ValueError):
            fundamentals["totalRevenue"] = None

    return fundamentals


def _extract_financial_metric(
    df: pd.DataFrame | None,
    metric: str,
) -> pd.Series | None:
    if df is None or df.empty:
        return None

    index_map = {idx.lower(): idx for idx in df.index}
    metric_key = metric.lower()
    if metric_key not in index_map:
        return None

    series = df.loc[index_map[metric_key]]
    if isinstance(series, pd.Series):
        series = series.dropna()
    return series


def _compute_revenue_growth(financials: pd.DataFrame | None) -> Tuple[float | None, float | None]:
    revenue_series = _extract_financial_metric(financials, "Total Revenue")
    if revenue_series is None or revenue_series.empty:
        return None, None

    revenue_series = revenue_series.sort_index()
    if len(revenue_series) < 2:
        return None, None

    latest = revenue_series.iloc[-1]
    prev = revenue_series.iloc[-2]
    yoy = None
    if prev and prev != 0:
        yoy = float(latest / prev - 1)

    cagr = None
    if len(revenue_series) >= 4:
        base = revenue_series.iloc[-4]
        if base and base > 0:
            periods = len(revenue_series.iloc[-4:])
            cagr = float((latest / base) ** (1 / (periods - 1)) - 1)

    return yoy, cagr


def _compute_operating_margin(financials: pd.DataFrame | None) -> float | None:
    revenue_series = _extract_financial_metric(financials, "Total Revenue")
    operating_series = _extract_financial_metric(financials, "Operating Income")
    if (
        revenue_series is None
        or operating_series is None
        or revenue_series.empty
        or operating_series.empty
    ):
        return None

    latest_revenue = revenue_series.sort_index().iloc[-1]
    latest_operating = operating_series.sort_index().iloc[-1]
    if not latest_revenue or latest_revenue == 0:
        return None
    try:
        return float(latest_operating / latest_revenue)
    except (TypeError, ZeroDivisionError):
        return None


def _extract_free_cash_flow(cashflow: pd.DataFrame | None) -> float | None:
    if cashflow is None or cashflow.empty:
        return None
    index_map = {idx.lower(): idx for idx in cashflow.index}
    for key in ["free cash flow", "freecashflow"]:
        if key in index_map:
            series = cashflow.loc[index_map[key]]
            if isinstance(series, pd.Series) and not series.empty:
                return float(series.sort_index().iloc[-1])
            break
    return None


def _compute_forecast(
    history: pd.DataFrame, periods: int = 126
) -> Tuple[pd.DataFrame | None, float | None]:
    if history.empty or "Close" not in history.columns:
        return None, None
    prices = history["Close"].dropna()
    if len(prices) < 10:
        return None, None

    x = np.arange(len(prices))
    log_prices = np.log(prices)
    try:
        slope, intercept = np.polyfit(x, log_prices, deg=1)
    except np.linalg.LinAlgError:
        return None, None

    future_x = np.arange(len(prices), len(prices) + periods)
    forecast_log = intercept + slope * future_x
    forecast_prices = np.exp(forecast_log)

    last_date = history.index[-1]
    future_dates = pd.bdate_range(last_date, periods=periods + 1)[1:]
    forecast_df = pd.DataFrame({"date": future_dates, "forecast": forecast_prices})
    # forecast_slope: expected 6-month return (positive = bullish, negative = bearish)
    forecast_slope = None
    if len(forecast_prices) > 0 and prices.iloc[-1] > 0:
        forecast_slope = float(forecast_prices[-1] / prices.iloc[-1] - 1)
    return forecast_df, forecast_slope


def _calculate_technical_indicators(history: pd.DataFrame) -> Dict[str, float | None]:
    if history.empty:
        return {}

    df = history.copy()
    indicators: Dict[str, float | None] = {}

    close = df["Close"]
    volume = df.get("Volume")

    try:
        rsi_series = ta.rsi(close, length=14)
        indicators["rsi_14"] = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else None
    except Exception:
        indicators["rsi_14"] = None

    indicators["moving_average_50"] = (
        float(close.rolling(50).mean().dropna().iloc[-1]) if len(close) >= 50 else None
    )
    indicators["moving_average_200"] = (
        float(close.rolling(200).mean().dropna().iloc[-1]) if len(close) >= 200 else None
    )

    try:
        macd = ta.macd(close)
        if macd is not None and not macd.empty:
            # pandas_ta returns columns like MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
            macd_cols = [col for col in macd.columns if col.startswith("MACD_") and not col.startswith("MACDs_") and not col.startswith("MACDh_")]
            signal_cols = [col for col in macd.columns if col.startswith("MACDs_")]
            if macd_cols and signal_cols:
                indicators["macd"] = float(macd.iloc[-1][macd_cols[0]])
                indicators["macd_signal"] = float(macd.iloc[-1][signal_cols[0]])
            else:
                # Fallback to common column names
                if "MACD_12_26_9" in macd.columns:
                    indicators["macd"] = float(macd.iloc[-1]["MACD_12_26_9"])
                if "MACDs_12_26_9" in macd.columns:
                    indicators["macd_signal"] = float(macd.iloc[-1]["MACDs_12_26_9"])
        else:
            indicators["macd"] = indicators["macd_signal"] = None
    except Exception:
        indicators["macd"] = indicators["macd_signal"] = None

    try:
        bbands = ta.bbands(close, length=20)
        if bbands is not None and not bbands.empty:
            indicators["bollinger_lower"] = float(bbands.iloc[-1]["BBL_20_2.0"])
            indicators["bollinger_middle"] = float(bbands.iloc[-1]["BBM_20_2.0"])
            indicators["bollinger_upper"] = float(bbands.iloc[-1]["BBU_20_2.0"])
        else:
            indicators["bollinger_lower"] = indicators["bollinger_middle"] = indicators["bollinger_upper"] = None
    except Exception:
        indicators["bollinger_lower"] = indicators["bollinger_middle"] = indicators["bollinger_upper"] = None

    if volume is not None and not volume.empty:
        avg_vol_series = volume.rolling(20).mean().dropna()
        avg_volume_20 = float(avg_vol_series.iloc[-1]) if not avg_vol_series.empty else None
        indicators["avg_volume_20"] = avg_volume_20
        if avg_volume_20 and avg_volume_20 != 0:
            indicators["volume_ratio"] = float(volume.iloc[-1] / avg_volume_20)
        else:
            indicators["volume_ratio"] = None
    else:
        indicators["avg_volume_20"] = indicators["volume_ratio"] = None

    latest_close = close.iloc[-1]
    # Use last 252 trading days (approx 52 weeks) for 52-week high/low
    close_52w = close.tail(252) if len(close) >= 252 else close
    high_52w = close_52w.max()
    low_52w = close_52w.min()
    if high_52w and high_52w > 0:
        indicators["dist_52w_high"] = float(latest_close / high_52w - 1)
    else:
        indicators["dist_52w_high"] = None
    if low_52w and low_52w > 0:
        indicators["dist_52w_low"] = float(latest_close / low_52w - 1)
    else:
        indicators["dist_52w_low"] = None

    return indicators


def _fetch_snapshot(symbol: str) -> StockSnapshot:
    ticker_symbol = _normalise_symbol(symbol)
    ticker = yf.Ticker(ticker_symbol)

    history = _download_history(ticker)
    change_1m, change_6m = _calculate_returns(history)

    info: Dict[str, Any] = {}
    try:
        info = _fetch_company_info(ticker)
    except Exception:
        info = {}

    fundamentals = {}
    if info:
        fundamentals = _extract_fundamentals(info)
    else:
        fundamentals = {}

    financials = None
    cashflow = None
    try:
        time.sleep(0.1)  # Small delay to reduce rate limiting
        financials = ticker.financials
    except Exception:
        financials = None
    try:
        time.sleep(0.1)  # Small delay to reduce rate limiting
        cashflow = ticker.cashflow
    except Exception:
        cashflow = None

    revenue_growth_yoy, revenue_cagr_3y = _compute_revenue_growth(financials)
    operating_margin = _compute_operating_margin(financials)
    free_cash_flow = _extract_free_cash_flow(cashflow)
    dividend_yield = None
    if info:
        dividend_yield = info.get("dividendYield")
        if dividend_yield is not None:
            try:
                dividend_yield = float(dividend_yield)
            except (TypeError, ValueError):
                dividend_yield = None

    short_name = info.get("shortName") if info else None

    forecast_df, forecast_slope = _compute_forecast(history)
    technicals = _calculate_technical_indicators(history)

    return StockSnapshot(
        symbol=symbol,
        ticker=ticker_symbol,
        short_name=short_name,
        price_history=history.reset_index(),
        change_1m=change_1m,
        change_6m=change_6m,
        fundamentals=fundamentals,
        revenue_growth_yoy=revenue_growth_yoy,
        revenue_cagr_3y=revenue_cagr_3y,
        dividend_yield=dividend_yield,
        free_cash_flow=free_cash_flow,
        operating_margin=operating_margin,
        forecast=forecast_df,
        forecast_slope=forecast_slope,
        rsi_14=technicals.get("rsi_14"),
        moving_average_50=technicals.get("moving_average_50"),
        moving_average_200=technicals.get("moving_average_200"),
        macd=technicals.get("macd"),
        macd_signal=technicals.get("macd_signal"),
        bollinger_upper=technicals.get("bollinger_upper"),
        bollinger_lower=technicals.get("bollinger_lower"),
        bollinger_middle=technicals.get("bollinger_middle"),
        avg_volume_20=technicals.get("avg_volume_20"),
        volume_ratio=technicals.get("volume_ratio"),
        dist_52w_high=technicals.get("dist_52w_high"),
        dist_52w_low=technicals.get("dist_52w_low"),
    )


def fetch_snapshots(
    symbols: Iterable[str],
    max_workers: int = 8,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[List[StockSnapshot], Dict[str, str]]:
    """Fetch price/fundamental snapshots in parallel.

    Returns a tuple (snapshots, failures) where failures maps symbol->reason.
    
    Args:
        symbols: List of stock symbols to fetch
        max_workers: Maximum number of parallel workers
        progress_callback: Optional callback function(symbol, completed, total) called after each fetch
    """
    snapshots: List[StockSnapshot] = []
    failures: Dict[str, str] = {}
    
    symbols_list = list(dict.fromkeys(symbols))
    total = len(symbols_list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_fetch_snapshot, symbol): symbol for symbol in symbols_list
        }
        
        completed = 0
        for future in as_completed(future_map):
            symbol = future_map[future]
            completed += 1
            if progress_callback:
                progress_callback(symbol, completed, total)
            try:
                snapshot = future.result()
            except MarketDataError as exc:
                error_msg = str(exc)
                # Check for rate limiting
                if _check_rate_limit_error(exc):
                    failures[symbol] = "Rate limited - retrying with delays"
                else:
                    failures[symbol] = error_msg
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc)
                # Check for rate limiting
                if _check_rate_limit_error(exc):
                    failures[symbol] = "Rate limited - retrying with delays"
                else:
                    failures[symbol] = f"Unexpected error: {error_msg}"
            else:
                snapshots.append(snapshot)

    return snapshots, failures

