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
    # Advanced Financial Metrics
    roic: float | None = None  # Return on Invested Capital
    roa: float | None = None  # Return on Assets
    gross_margin: float | None = None
    net_margin: float | None = None
    ebitda_margin: float | None = None
    profit_growth_yoy: float | None = None
    peg_ratio: float | None = None  # P/E to Growth
    price_to_book: float | None = None  # P/B Ratio
    price_to_sales: float | None = None  # P/S Ratio
    ev_to_ebitda: float | None = None
    market_cap: float | None = None
    enterprise_value: float | None = None
    asset_turnover: float | None = None
    inventory_turnover: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    beta: float | None = None  # Stock volatility vs market
    # Debt & Financial Health
    interest_coverage: float | None = None  # Interest Coverage Ratio
    debt_to_assets: float | None = None  # Debt-to-Assets Ratio
    total_debt: float | None = None  # Current Debt Levels
    working_capital: float | None = None  # Working Capital
    # Cash Flow Analysis
    operating_cash_flow: float | None = None
    investing_cash_flow: float | None = None
    financing_cash_flow: float | None = None
    capex: float | None = None  # Capital Expenditure
    cash_flow_per_share: float | None = None
    # Advanced Technical Indicators
    stochastic_k: float | None = None  # Stochastic Oscillator
    stochastic_d: float | None = None  # Stochastic Signal
    williams_r: float | None = None  # Williams %R
    obv: float | None = None  # On-Balance Volume
    support_level: float | None = None  # Support price level
    resistance_level: float | None = None  # Resistance price level
    # Risk Metrics
    sharpe_ratio: float | None = None  # Risk-adjusted returns (approx)
    sortino_ratio: float | None = None  # Downside risk measure
    max_drawdown: float | None = None  # Maximum drawdown
    volatility: float | None = None  # Price volatility (std dev)
    # Timing Factors
    earnings_date: str | None = None  # Next earnings date
    ex_dividend_date: str | None = None  # Next ex-dividend date
    # Ownership (if available)
    institutional_ownership: float | None = None  # Institutional ownership %
    float_shares: float | None = None  # Shares outstanding / float

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
            # Advanced Financial Metrics
            "roic": self.roic,
            "roa": self.roa,
            "gross_margin": self.gross_margin,
            "net_margin": self.net_margin,
            "ebitda_margin": self.ebitda_margin,
            "profit_growth_yoy": self.profit_growth_yoy,
            "peg_ratio": self.peg_ratio,
            "price_to_book": self.price_to_book,
            "price_to_sales": self.price_to_sales,
            "ev_to_ebitda": self.ev_to_ebitda,
            "market_cap": self.market_cap,
            "enterprise_value": self.enterprise_value,
            "asset_turnover": self.asset_turnover,
            "inventory_turnover": self.inventory_turnover,
            "current_ratio": self.current_ratio,
            "quick_ratio": self.quick_ratio,
            "beta": self.beta,
            # Debt & Financial Health
            "interest_coverage": self.interest_coverage,
            "debt_to_assets": self.debt_to_assets,
            "total_debt": self.total_debt,
            "working_capital": self.working_capital,
            # Cash Flow Analysis
            "operating_cash_flow": self.operating_cash_flow,
            "investing_cash_flow": self.investing_cash_flow,
            "financing_cash_flow": self.financing_cash_flow,
            "capex": self.capex,
            "cash_flow_per_share": self.cash_flow_per_share,
            # Advanced Technical Indicators
            "stochastic_k": self.stochastic_k,
            "stochastic_d": self.stochastic_d,
            "williams_r": self.williams_r,
            "obv": self.obv,
            "support_level": self.support_level,
            "resistance_level": self.resistance_level,
            # Risk Metrics
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            # Timing Factors
            "earnings_date": self.earnings_date,
            "ex_dividend_date": self.ex_dividend_date,
            # Ownership
            "institutional_ownership": self.institutional_ownership,
            "float_shares": self.float_shares,
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


def _extract_advanced_metrics_from_info(info: Dict[str, Any]) -> Dict[str, float | None]:
    """Extract advanced financial metrics available from yfinance ticker.info."""
    metrics = {}
    
    # Valuation metrics
    for key, target in [
        ("priceToBook", "price_to_book"),
        ("priceToSalesTrailing12Months", "price_to_sales"),
        ("pegRatio", "peg_ratio"),
        ("enterpriseToEbitda", "ev_to_ebitda"),
        ("marketCap", "market_cap"),
        ("enterpriseValue", "enterprise_value"),
        ("beta", "beta"),
    ]:
        value = info.get(key)
        if value is not None:
            try:
                metrics[target] = float(value)
            except (TypeError, ValueError):
                metrics[target] = None
        else:
            metrics[target] = None
    
    # Profitability metrics
    for key, target in [
        ("returnOnAssets", "roa"),
        ("grossMargins", "gross_margin"),
        ("profitMargins", "net_margin"),
        ("ebitdaMargins", "ebitda_margin"),
    ]:
        value = info.get(key)
        if value is not None:
            try:
                # These are typically percentages (0-1 range)
                val = float(value)
                # If value > 1, it's likely in percentage form, convert
                if val > 1:
                    val = val / 100
                metrics[target] = val
            except (TypeError, ValueError):
                metrics[target] = None
        else:
            metrics[target] = None
    
    return metrics


def _compute_roic(financials: pd.DataFrame | None, balance_sheet: pd.DataFrame | None) -> float | None:
    """Calculate Return on Invested Capital (ROIC).
    ROIC = NOPAT / Invested Capital
    Invested Capital = Total Assets - Cash - Current Liabilities (or Debt + Equity)
    """
    if financials is None or balance_sheet is None:
        return None
    
    # Try to get operating income (NOPAT proxy)
    operating_income = _extract_financial_metric(financials, "Operating Income")
    if operating_income is None or operating_income.empty:
        return None
    
    # Try to get invested capital components
    total_assets = _extract_financial_metric(balance_sheet, "Total Assets")
    cash = _extract_financial_metric(balance_sheet, "Cash And Cash Equivalents")
    current_liabilities = _extract_financial_metric(balance_sheet, "Current Liabilities")
    
    if total_assets is None or total_assets.empty:
        return None
    
    try:
        latest_operating = operating_income.sort_index().iloc[-1]
        latest_assets = total_assets.sort_index().iloc[-1]
        
        invested_capital = latest_assets
        if cash is not None and not cash.empty:
            latest_cash = cash.sort_index().iloc[-1]
            invested_capital -= latest_cash
        if current_liabilities is not None and not current_liabilities.empty:
            latest_liab = current_liabilities.sort_index().iloc[-1]
            invested_capital -= latest_liab
        
        if invested_capital and invested_capital > 0:
            return float(latest_operating / invested_capital)
    except (IndexError, TypeError, ZeroDivisionError):
        pass
    
    return None


def _compute_profit_growth(financials: pd.DataFrame | None) -> float | None:
    """Calculate year-over-year profit growth."""
    net_income = _extract_financial_metric(financials, "Net Income")
    if net_income is None or net_income.empty or len(net_income) < 2:
        return None
    
    net_income = net_income.sort_index()
    latest = net_income.iloc[-1]
    prev = net_income.iloc[-2]
    
    if prev and prev != 0:
        try:
            return float(latest / prev - 1)
        except (TypeError, ZeroDivisionError):
            return None
    return None


def _compute_asset_turnover(financials: pd.DataFrame | None, balance_sheet: pd.DataFrame | None) -> float | None:
    """Calculate Asset Turnover Ratio = Revenue / Average Total Assets."""
    revenue = _extract_financial_metric(financials, "Total Revenue")
    total_assets = _extract_financial_metric(balance_sheet, "Total Assets")
    
    if revenue is None or revenue.empty or total_assets is None or total_assets.empty:
        return None
    
    try:
        latest_revenue = revenue.sort_index().iloc[-1]
        latest_assets = total_assets.sort_index().iloc[-1]
        
        if latest_assets and latest_assets > 0:
            return float(latest_revenue / latest_assets)
    except (IndexError, TypeError, ZeroDivisionError):
        pass
    
    return None


def _compute_inventory_turnover(financials: pd.DataFrame | None, balance_sheet: pd.DataFrame | None) -> float | None:
    """Calculate Inventory Turnover = COGS / Average Inventory."""
    cogs = _extract_financial_metric(financials, "Cost Of Revenue")
    inventory = _extract_financial_metric(balance_sheet, "Inventory")
    
    if cogs is None or cogs.empty or inventory is None or inventory.empty:
        return None
    
    try:
        latest_cogs = cogs.sort_index().iloc[-1]
        latest_inventory = inventory.sort_index().iloc[-1]
        
        if latest_inventory and latest_inventory > 0:
            return float(latest_cogs / latest_inventory)
    except (IndexError, TypeError, ZeroDivisionError):
        pass
    
    return None


def _compute_liquidity_ratios(balance_sheet: pd.DataFrame | None) -> Tuple[float | None, float | None]:
    """Calculate Current Ratio and Quick Ratio."""
    if balance_sheet is None or balance_sheet.empty:
        return None, None
    
    current_assets = _extract_financial_metric(balance_sheet, "Current Assets")
    current_liabilities = _extract_financial_metric(balance_sheet, "Current Liabilities")
    inventory = _extract_financial_metric(balance_sheet, "Inventory")
    
    if current_assets is None or current_assets.empty or current_liabilities is None or current_liabilities.empty:
        return None, None
    
    try:
        latest_ca = current_assets.sort_index().iloc[-1]
        latest_cl = current_liabilities.sort_index().iloc[-1]
        
        if latest_cl and latest_cl > 0:
            current_ratio = float(latest_ca / latest_cl)
            
            # Quick Ratio = (Current Assets - Inventory) / Current Liabilities
            quick_ratio = current_ratio
            if inventory is not None and not inventory.empty:
                latest_inv = inventory.sort_index().iloc[-1]
                quick_ratio = float((latest_ca - latest_inv) / latest_cl)
            
            return current_ratio, quick_ratio
    except (IndexError, TypeError, ZeroDivisionError):
        pass
    
    return None, None


def _compute_debt_metrics(financials: pd.DataFrame | None, balance_sheet: pd.DataFrame | None) -> Dict[str, float | None]:
    """Calculate debt-related metrics."""
    metrics = {}
    
    if balance_sheet is None or financials is None:
        return {"interest_coverage": None, "debt_to_assets": None, "total_debt": None, "working_capital": None}
    
    # Total Debt
    total_debt = _extract_financial_metric(balance_sheet, "Total Debt")
    long_term_debt = _extract_financial_metric(balance_sheet, "Long Term Debt")
    current_debt = _extract_financial_metric(balance_sheet, "Current Debt")
    
    total_debt_value = None
    if total_debt is not None and not total_debt.empty:
        total_debt_value = float(total_debt.sort_index().iloc[-1])
    elif long_term_debt is not None and current_debt is not None:
        if not long_term_debt.empty and not current_debt.empty:
            total_debt_value = float(long_term_debt.sort_index().iloc[-1]) + float(current_debt.sort_index().iloc[-1])
    
    metrics["total_debt"] = total_debt_value
    
    # Debt-to-Assets Ratio
    total_assets = _extract_financial_metric(balance_sheet, "Total Assets")
    if total_debt_value is not None and total_assets is not None and not total_assets.empty:
        latest_assets = total_assets.sort_index().iloc[-1]
        if latest_assets and latest_assets > 0:
            metrics["debt_to_assets"] = float(total_debt_value / latest_assets)
        else:
            metrics["debt_to_assets"] = None
    else:
        metrics["debt_to_assets"] = None
    
    # Interest Coverage Ratio = EBIT / Interest Expense
    operating_income = _extract_financial_metric(financials, "Operating Income")
    interest_expense = _extract_financial_metric(financials, "Interest Expense")
    
    if operating_income is not None and not operating_income.empty and interest_expense is not None and not interest_expense.empty:
        latest_ebit = operating_income.sort_index().iloc[-1]
        latest_interest = abs(interest_expense.sort_index().iloc[-1])
        if latest_interest and latest_interest > 0:
            metrics["interest_coverage"] = float(latest_ebit / latest_interest)
        else:
            metrics["interest_coverage"] = None
    else:
        metrics["interest_coverage"] = None
    
    # Working Capital = Current Assets - Current Liabilities
    current_assets = _extract_financial_metric(balance_sheet, "Current Assets")
    current_liabilities = _extract_financial_metric(balance_sheet, "Current Liabilities")
    
    if current_assets is not None and not current_assets.empty and current_liabilities is not None and not current_liabilities.empty:
        latest_ca = current_assets.sort_index().iloc[-1]
        latest_cl = current_liabilities.sort_index().iloc[-1]
        metrics["working_capital"] = float(latest_ca - latest_cl)
    else:
        metrics["working_capital"] = None
    
    return metrics


def _extract_cash_flow_metrics(cashflow: pd.DataFrame | None, shares_outstanding: float | None = None) -> Dict[str, float | None]:
    """Extract cash flow metrics from cashflow statement."""
    metrics = {}
    
    if cashflow is None or cashflow.empty:
        return {
            "operating_cash_flow": None,
            "investing_cash_flow": None,
            "financing_cash_flow": None,
            "capex": None,
            "cash_flow_per_share": None,
        }
    
    # Operating Cash Flow
    operating_cf = _extract_financial_metric(cashflow, "Operating Cash Flow")
    if operating_cf is None:
        operating_cf = _extract_financial_metric(cashflow, "Total Cash From Operating Activities")
    if operating_cf is not None and not operating_cf.empty:
        metrics["operating_cash_flow"] = float(operating_cf.sort_index().iloc[-1])
    else:
        metrics["operating_cash_flow"] = None
    
    # Investing Cash Flow
    investing_cf = _extract_financial_metric(cashflow, "Total Cashflows From Investing Activities")
    if investing_cf is not None and not investing_cf.empty:
        metrics["investing_cash_flow"] = float(investing_cf.sort_index().iloc[-1])
    else:
        metrics["investing_cash_flow"] = None
    
    # Financing Cash Flow
    financing_cf = _extract_financial_metric(cashflow, "Total Cash From Financing Activities")
    if financing_cf is not None and not financing_cf.empty:
        metrics["financing_cash_flow"] = float(financing_cf.sort_index().iloc[-1])
    else:
        metrics["financing_cash_flow"] = None
    
    # Capital Expenditure (usually negative in investing cash flow)
    capex = _extract_financial_metric(cashflow, "Capital Expenditure")
    if capex is None:
        # Sometimes it's listed differently
        for col_name in cashflow.index:
            if "cap" in str(col_name).lower() and "exp" in str(col_name).lower():
                try:
                    capex_series = cashflow.loc[col_name]
                    if isinstance(capex_series, pd.Series) and not capex_series.empty:
                        capex = capex_series
                        break
                except Exception:
                    pass
    
    if capex is not None and not capex.empty:
        metrics["capex"] = abs(float(capex.sort_index().iloc[-1]))  # Absolute value
    else:
        metrics["capex"] = None
    
    # Cash Flow per Share
    if metrics["operating_cash_flow"] is not None and shares_outstanding is not None and shares_outstanding > 0:
        metrics["cash_flow_per_share"] = float(metrics["operating_cash_flow"] / shares_outstanding)
    else:
        metrics["cash_flow_per_share"] = None
    
    return metrics


def _calculate_advanced_technical_indicators(history: pd.DataFrame) -> Dict[str, float | None]:
    """Calculate advanced technical indicators."""
    indicators = {}
    
    if history.empty or "Close" not in history.columns:
        return {
            "stochastic_k": None,
            "stochastic_d": None,
            "williams_r": None,
            "obv": None,
            "support_level": None,
            "resistance_level": None,
        }
    
    close = history["Close"]
    high = history.get("High", close)
    low = history.get("Low", close)
    volume = history.get("Volume")
    
    # Stochastic Oscillator (14-period)
    try:
        if len(close) >= 14:
            stoch_period = 14
            lowest_low = low.rolling(window=stoch_period).min()
            highest_high = high.rolling(window=stoch_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            k_percent = k_percent.dropna()
            if not k_percent.empty:
                indicators["stochastic_k"] = float(k_percent.iloc[-1])
                # D is 3-period SMA of K
                d_period = 3
                if len(k_percent) >= d_period:
                    d_percent = k_percent.rolling(window=d_period).mean().dropna()
                    if not d_percent.empty:
                        indicators["stochastic_d"] = float(d_percent.iloc[-1])
                else:
                    indicators["stochastic_d"] = None
            else:
                indicators["stochastic_k"] = indicators["stochastic_d"] = None
        else:
            indicators["stochastic_k"] = indicators["stochastic_d"] = None
    except Exception:
        indicators["stochastic_k"] = indicators["stochastic_d"] = None
    
    # Williams %R (14-period)
    try:
        if len(close) >= 14:
            wr_period = 14
            highest_high = high.rolling(window=wr_period).max()
            lowest_low = low.rolling(window=wr_period).min()
            wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
            wr = wr.dropna()
            if not wr.empty:
                indicators["williams_r"] = float(wr.iloc[-1])
            else:
                indicators["williams_r"] = None
        else:
            indicators["williams_r"] = None
    except Exception:
        indicators["williams_r"] = None
    
    # On-Balance Volume (OBV)
    try:
        if volume is not None and not volume.empty and len(close) > 1:
            obv_values = pd.Series(index=close.index, dtype=float)
            obv_values.iloc[0] = volume.iloc[0]
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv_values.iloc[i] = obv_values.iloc[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv_values.iloc[i] = obv_values.iloc[i-1] - volume.iloc[i]
                else:
                    obv_values.iloc[i] = obv_values.iloc[i-1]
            if not obv_values.empty:
                indicators["obv"] = float(obv_values.iloc[-1])
            else:
                indicators["obv"] = None
        else:
            indicators["obv"] = None
    except Exception:
        indicators["obv"] = None
    
    # Support and Resistance levels (simplified - using recent highs/lows)
    try:
        if len(close) >= 20:
            # Resistance: recent 20-day high
            recent_high = high.tail(20).max()
            # Support: recent 20-day low
            recent_low = low.tail(20).min()
            indicators["resistance_level"] = float(recent_high)
            indicators["support_level"] = float(recent_low)
        else:
            indicators["resistance_level"] = indicators["support_level"] = None
    except Exception:
        indicators["resistance_level"] = indicators["support_level"] = None
    
    return indicators


def _calculate_risk_metrics(history: pd.DataFrame, risk_free_rate: float = 0.06) -> Dict[str, float | None]:
    """Calculate risk-adjusted return metrics."""
    metrics = {}
    
    if history.empty or "Close" not in history.columns or len(history) < 30:
        return {
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown": None,
            "volatility": None,
        }
    
    close = history["Close"]
    
    # Calculate daily returns
    returns = close.pct_change().dropna()
    
    if len(returns) < 30:
        return {
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown": None,
            "volatility": None,
        }
    
    # Volatility (annualized standard deviation)
    try:
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)  # Annualized
        metrics["volatility"] = float(annual_vol)
    except Exception:
        metrics["volatility"] = None
    
    # Annualized return
    try:
        daily_return = returns.mean()
        annual_return = daily_return * 252  # Annualized (approx)
        excess_return = annual_return - (risk_free_rate / 100)  # Convert to decimal
    except Exception:
        excess_return = None
        annual_return = None
    
    # Sharpe Ratio = (Return - Risk Free Rate) / Volatility
    try:
        if excess_return is not None and metrics["volatility"] is not None and metrics["volatility"] > 0:
            metrics["sharpe_ratio"] = float(excess_return / metrics["volatility"])
        else:
            metrics["sharpe_ratio"] = None
    except Exception:
        metrics["sharpe_ratio"] = None
    
    # Sortino Ratio = (Return - Risk Free Rate) / Downside Deviation
    try:
        if excess_return is not None:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                downside_dev_annual = downside_std * np.sqrt(252)
                if downside_dev_annual > 0:
                    metrics["sortino_ratio"] = float(excess_return / downside_dev_annual)
                else:
                    metrics["sortino_ratio"] = None
            else:
                metrics["sortino_ratio"] = None
        else:
            metrics["sortino_ratio"] = None
    except Exception:
        metrics["sortino_ratio"] = None
    
    # Maximum Drawdown
    try:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics["max_drawdown"] = float(drawdown.min())  # Negative value
    except Exception:
        metrics["max_drawdown"] = None
    
    return metrics


def _extract_timing_factors(info: Dict[str, Any]) -> Dict[str, str | None]:
    """Extract timing-related information."""
    metrics = {}
    
    # Earnings date
    earnings_date = info.get("earningsDate") or info.get("mostRecentQuarter") or info.get("nextFiscalYearEnd")
    if earnings_date:
        if isinstance(earnings_date, (list, tuple)) and len(earnings_date) > 0:
            metrics["earnings_date"] = str(earnings_date[0]) if earnings_date[0] else None
        else:
            metrics["earnings_date"] = str(earnings_date)
    else:
        metrics["earnings_date"] = None
    
    # Ex-dividend date
    ex_dividend_date = info.get("exDividendDate") or info.get("dividendDate")
    if ex_dividend_date:
        metrics["ex_dividend_date"] = str(ex_dividend_date)
    else:
        metrics["ex_dividend_date"] = None
    
    return metrics


def _extract_ownership_info(info: Dict[str, Any]) -> Dict[str, float | None]:
    """Extract ownership-related information."""
    metrics = {}
    
    # Institutional ownership %
    inst_own = info.get("heldPercentInstitutions") or info.get("institutionOwnershipPercentage")
    if inst_own is not None:
        try:
            val = float(inst_own)
            if val > 1:  # If in percentage form
                val = val / 100
            metrics["institutional_ownership"] = val
        except (TypeError, ValueError):
            metrics["institutional_ownership"] = None
    else:
        metrics["institutional_ownership"] = None
    
    # Float shares / Shares outstanding
    float_shares = info.get("floatShares") or info.get("sharesOutstanding")
    if float_shares is not None:
        try:
            metrics["float_shares"] = float(float_shares)
        except (TypeError, ValueError):
            metrics["float_shares"] = None
    else:
        metrics["float_shares"] = None
    
    return metrics


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
    balance_sheet = None
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
    try:
        time.sleep(0.1)  # Small delay to reduce rate limiting
        balance_sheet = ticker.balance_sheet
    except Exception:
        balance_sheet = None

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
    
    # Extract advanced metrics from info
    advanced_metrics = _extract_advanced_metrics_from_info(info) if info else {}
    
    # Calculate additional metrics from financials/balance sheet
    roic = _compute_roic(financials, balance_sheet)
    profit_growth_yoy = _compute_profit_growth(financials)
    asset_turnover = _compute_asset_turnover(financials, balance_sheet)
    inventory_turnover = _compute_inventory_turnover(financials, balance_sheet)
    current_ratio, quick_ratio = _compute_liquidity_ratios(balance_sheet)
    
    # Calculate debt metrics
    debt_metrics = _compute_debt_metrics(financials, balance_sheet)
    
    # Extract cash flow metrics
    shares_outstanding = info.get("sharesOutstanding") if info else None
    try:
        shares_outstanding = float(shares_outstanding) if shares_outstanding else None
    except (TypeError, ValueError):
        shares_outstanding = None
    
    cashflow_metrics = _extract_cash_flow_metrics(cashflow, shares_outstanding)
    
    # Calculate advanced technical indicators
    advanced_technicals = _calculate_advanced_technical_indicators(history)
    
    # Calculate risk metrics
    risk_metrics = _calculate_risk_metrics(history)
    
    # Extract timing factors
    timing_factors = _extract_timing_factors(info) if info else {}
    
    # Extract ownership info
    ownership_info = _extract_ownership_info(info) if info else {}

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
        # Advanced Financial Metrics
        roic=roic,
        roa=advanced_metrics.get("roa"),
        gross_margin=advanced_metrics.get("gross_margin"),
        net_margin=advanced_metrics.get("net_margin"),
        ebitda_margin=advanced_metrics.get("ebitda_margin"),
        profit_growth_yoy=profit_growth_yoy,
        peg_ratio=advanced_metrics.get("peg_ratio"),
        price_to_book=advanced_metrics.get("price_to_book"),
        price_to_sales=advanced_metrics.get("price_to_sales"),
        ev_to_ebitda=advanced_metrics.get("ev_to_ebitda"),
        market_cap=advanced_metrics.get("market_cap"),
        enterprise_value=advanced_metrics.get("enterprise_value"),
        asset_turnover=asset_turnover,
        inventory_turnover=inventory_turnover,
        current_ratio=current_ratio,
        quick_ratio=quick_ratio,
        beta=advanced_metrics.get("beta"),
        # Debt & Financial Health
        interest_coverage=debt_metrics.get("interest_coverage"),
        debt_to_assets=debt_metrics.get("debt_to_assets"),
        total_debt=debt_metrics.get("total_debt"),
        working_capital=debt_metrics.get("working_capital"),
        # Cash Flow Analysis
        operating_cash_flow=cashflow_metrics.get("operating_cash_flow"),
        investing_cash_flow=cashflow_metrics.get("investing_cash_flow"),
        financing_cash_flow=cashflow_metrics.get("financing_cash_flow"),
        capex=cashflow_metrics.get("capex"),
        cash_flow_per_share=cashflow_metrics.get("cash_flow_per_share"),
        # Advanced Technical Indicators
        stochastic_k=advanced_technicals.get("stochastic_k"),
        stochastic_d=advanced_technicals.get("stochastic_d"),
        williams_r=advanced_technicals.get("williams_r"),
        obv=advanced_technicals.get("obv"),
        support_level=advanced_technicals.get("support_level"),
        resistance_level=advanced_technicals.get("resistance_level"),
        # Risk Metrics
        sharpe_ratio=risk_metrics.get("sharpe_ratio"),
        sortino_ratio=risk_metrics.get("sortino_ratio"),
        max_drawdown=risk_metrics.get("max_drawdown"),
        volatility=risk_metrics.get("volatility"),
        # Timing Factors
        earnings_date=timing_factors.get("earnings_date"),
        ex_dividend_date=timing_factors.get("ex_dividend_date"),
        # Ownership
        institutional_ownership=ownership_info.get("institutional_ownership"),
        float_shares=ownership_info.get("float_shares"),
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

