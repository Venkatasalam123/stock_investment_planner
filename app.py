import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
import yfinance as yf

MIN_SELL_PROFIT_ABS = 5000.0  # Minimum absolute profit (₹) to justify selling
MIN_SELL_PROFIT_PCT = 0.05    # Minimum profit percentage to justify selling

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
from storage.database import (
    fetch_recent_runs,
    fetch_run_by_id,
    load_run_data,
    log_run,
    fetch_all_predictions,
    fetch_predictions_by_run_id,
    update_prediction_price,
    delete_unknown_runs,
    update_prediction_action,
    PredictionTracking,
)


load_dotenv()

st.set_page_config(
    page_title="Stock Investment AI Assistant",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=3600)
def load_nifty500() -> pd.DataFrame:
    return fetch_nifty_500_constituents()


@st.cache_data(ttl=1800)
def load_fii_trend() -> pd.DataFrame:
    return fetch_fii_trend()


@st.cache_data(ttl=3600)
def load_index_constituents(index_name: str) -> pd.DataFrame:
    normalized = index_name.strip().upper()
    if normalized == "NIFTY 500":
        return load_nifty500()
    return fetch_index_constituents(index_name)


@st.cache_data(ttl=3600)
def load_symbol_catalog() -> pd.DataFrame:
    base = load_nifty500().copy()
    if "symbol" not in base.columns:
        raise RuntimeError("Nifty 500 constituents do not include a symbol column")
    name_cols = [col for col in base.columns if "company" in col or "name" in col]
    company_series = base[name_cols[0]] if name_cols else base["symbol"]
    catalog = pd.DataFrame(
        {
            "symbol": base["symbol"].astype(str).str.upper(),
            "name": company_series.astype(str),
        }
    ).drop_duplicates(subset="symbol")
    catalog.sort_values("symbol", inplace=True)
    catalog.reset_index(drop=True, inplace=True)
    return catalog


def _prepare_history_frame(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df.set_index("Date", inplace=True)
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df.set_index("date", inplace=True)
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)
    df.index = df.index.normalize()
    return df


def _resolve_price_for_date(
    snapshot: StockSnapshot,
    target_date: datetime,
    history: pd.DataFrame,
) -> Tuple[Optional[float], Optional[datetime]]:
    target = pd.to_datetime(target_date).tz_localize(None).normalize()
    if target in history.index:
        return float(history.loc[target]["Close"]), target

    try:
        nearest_pos = history.index.get_indexer([target], method="nearest")
    except Exception:
        nearest_pos = [-1]
    if nearest_pos[0] != -1:
        nearest_idx = history.index[nearest_pos[0]]
        if abs(nearest_idx - target) <= pd.Timedelta(days=10):
            return float(history.iloc[nearest_pos[0]]["Close"]), nearest_idx

    ticker = snapshot.ticker or snapshot.symbol
    if ticker and not ticker.endswith(".NS"):
        yf_ticker = ticker
    else:
        yf_ticker = f"{snapshot.symbol}.NS"

    try:
        yf_data = yf.download(yf_ticker, start=target - pd.Timedelta(days=15), end=target + pd.Timedelta(days=15))
    except Exception:
        return None, None
    if yf_data.empty:
        return None, None
    yf_data.index = pd.to_datetime(yf_data.index).tz_localize(None).normalize()
    nearest_pos = yf_data.index.get_indexer([target], method="nearest")
    if nearest_pos[0] == -1:
        return None, None
    nearest_idx = yf_data.index[nearest_pos[0]]
    return float(yf_data.iloc[nearest_pos[0]]["Close"]), nearest_idx


def enrich_purchase_lots(snapshot: StockSnapshot, lots: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    summary = {
        "latest_close": None,
        "latest_date": None,
        "total_shares": 0,
        "total_cost": 0.0,
        "total_value": 0.0,
        "avg_cost": None,
    }
    if not lots:
        return [], summary
    history = _prepare_history_frame(snapshot.price_history)
    if history.empty or "Close" not in history.columns:
        return [], summary
    latest_date = history.index[-1]
    latest_close = float(history.iloc[-1]["Close"])
    enriched: List[Dict[str, Any]] = []
    total_shares = 0
    total_cost = 0.0
    total_value = 0.0
    for lot in lots:
        try:
            target_date = pd.to_datetime(lot.get("date"))
        except Exception:
            continue
        shares = int(lot.get("shares") or 0)
        if shares <= 0:
            continue
        price, matched_date = _resolve_price_for_date(snapshot, target_date, history)
        cost = None
        current_value = latest_close * shares
        unrealized = None
        unrealized_pct = None
        if price is not None:
            cost = price * shares
            unrealized = current_value - cost
            unrealized_pct = (unrealized / cost) if cost else None
            total_cost += cost
        total_shares += shares
        total_value += current_value
        enriched.append(
            {
                "symbol": snapshot.symbol,
                "requested_date": pd.to_datetime(target_date).date(),
                "matched_date": matched_date.date() if matched_date is not None else None,
                "shares": shares,
                "purchase_price": price,
                "current_price": latest_close,
                "cost_basis": cost,
                "current_value": current_value,
                "unrealized_pl": unrealized,
                "unrealized_pl_pct": unrealized_pct,
            }
        )
    avg_cost = (total_cost / total_shares) if total_shares and total_cost else None
    summary.update(
        {
            "latest_close": latest_close,
            "latest_date": latest_date,
            "total_shares": total_shares,
            "total_cost": total_cost,
            "total_value": total_value,
            "avg_cost": avg_cost,
        }
    )
    return enriched, summary


def show_purchase_history(enriched_lots: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    if not enriched_lots:
        return
    st.subheader("Your Holding Lots")
    df = pd.DataFrame(enriched_lots)
    df.rename(
        columns={
            "requested_date": "Input Date",
            "matched_date": "Matched Date",
            "shares": "Shares",
            "purchase_price": "Purchase Price",
            "current_price": "Current Price",
            "cost_basis": "Cost Basis",
            "current_value": "Current Value",
            "unrealized_pl": "Unrealised P/L",
            "unrealized_pl_pct": "Unrealised %",
        },
        inplace=True,
    )
    st.dataframe(
        df.style.format(
            {
                "Purchase Price": "₹{:,.2f}",
                "Current Price": "₹{:,.2f}",
                "Cost Basis": "₹{:,.2f}",
                "Current Value": "₹{:,.2f}",
                "Unrealised P/L": "₹{:,.2f}",
                "Unrealised %": "{:+.2%}",
            },
            na_rep="NA",
        ),
        use_container_width=True,
        hide_index=True,
    )
    total_shares = summary.get("total_shares", 0)
    total_cost = summary.get("total_cost")
    total_value = summary.get("total_value")
    latest_close = summary.get("latest_close")
    latest_date = summary.get("latest_date")
    avg_cost = summary.get("avg_cost")
    if total_cost is not None and total_value is not None:
        total_pl = total_value - total_cost
        total_pl_pct = total_pl / total_cost if total_cost else None
        date_str = pd.to_datetime(latest_date).strftime("%d %b %Y") if latest_date else "recent"
        st.info(
            f"You hold **{int(total_shares)} share(s)** worth ₹{total_value:,.2f}. "
            f"Unrealised P/L: {('₹{:+,.2f}'.format(total_pl)) if pd.notna(total_pl) else 'NA'}"
            + (
                f" ({total_pl_pct:+.2%})" if total_pl_pct is not None else ""
            )
            + f". Latest close ₹{latest_close:,.2f} on {date_str}."
        )
    st.session_state["latest_holding_summary"] = {
        "avg_cost": avg_cost,
        "latest_price": latest_close,
        "total_shares": total_shares,
    }


def sidebar_inputs() -> Dict[str, Any]:
    st.sidebar.markdown("### 🎯 Analysis Setup")

    analysis_mode = st.sidebar.radio(
        "Choose Analysis Flow",
        ("Market Basket (Index)", "Single Stock Focus"),
        index=0,
        help=(
            "Market Basket analyses the top constituents of a chosen index. "
            "Single Stock Focus lets you deep-dive into one ticker at a time."
        ),
    )
    st.sidebar.info("Single Stock Focus is ideal when you want a yes/no or sell verdict on a specific ticker.")
    st.sidebar.caption("Analyse an individual stock or screen a basket from your chosen market cap index, then proceed.")
    is_single_stock = analysis_mode == "Single Stock Focus"

    horizon_years = st.sidebar.slider(
        "Investment Horizon (years)",
        min_value=1,
        max_value=10,
        value=5,
        help="How long you plan to stay invested.",
    )

    index_choice: Optional[str] = None
    selected_symbol: Optional[str] = None

    if not is_single_stock:
        index_choice = st.sidebar.selectbox(
            "Market Index Universe",
            AVAILABLE_INDICES,
            index=AVAILABLE_INDICES.index("NIFTY 50") if "NIFTY 50" in AVAILABLE_INDICES else 0,
            help="Pick the index whose constituents you would like to screen.",
        )
    else:
        catalog = load_symbol_catalog()
        symbol_options = catalog["symbol"].tolist()
        names_map = dict(zip(catalog["symbol"], catalog["name"]))
        selected_symbol = st.sidebar.selectbox(
            "Stock to Analyse",
            symbol_options,
            index=symbol_options.index("TCS") if "TCS" in symbol_options else 0,
            format_func=lambda sym: f"{sym} – {names_map.get(sym, 'Unknown')}",
            help="Start typing to quickly locate a ticker from the Nifty 500 universe.",
        )

    invest_amount = st.sidebar.number_input(
        "Planned Investment Amount (₹)",
        min_value=0.0,
        value=100000.0,
        step=1000.0,
        help="Total amount you intend to allocate across recommendations.",
    )

    strategy_notes = st.sidebar.text_area(
        "Additional Strategy Notes",
        value="",
        help="Any preferences or constraints you'd like the assistant to consider.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Stock-Specific Guidance")
    purchase_lots: List[Dict[str, Any]] = []

    if is_single_stock:
        evaluation_symbol = selected_symbol or ""
        evaluation_position = st.sidebar.selectbox(
            "Current Position",
            options=["Do not own", "Already own"],
            help="Let the assistant know if you already hold this stock.",
        )
    else:
        evaluation_symbol = st.sidebar.text_input(
            "Optional: Spotlight a stock",
            value="",
            help="Enter a ticker (e.g. TCS) if you'd like individual guidance alongside the basket.",
        ).strip().upper()
        evaluation_position = st.sidebar.selectbox(
            "Current Position",
            options=["Do not own", "Already own"],
            help="Helps tailor the recommendation if you already hold the spotlight stock.",
            disabled=not evaluation_symbol,
        )

    if evaluation_position == "Already own" and (is_single_stock or evaluation_symbol):
        lot_count = int(
            st.sidebar.number_input(
                "How many purchase lots?",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                key=f"lot-count-{analysis_mode}-{evaluation_symbol or 'basket'}",
            )
        )
        default_date = datetime.today().date()
        for idx in range(lot_count):
            lot_date = st.sidebar.date_input(
                f"Purchase date for lot {idx + 1}",
                value=default_date,
                key=f"lot-date-{analysis_mode}-{evaluation_symbol or 'basket'}-{idx}",
                help="Pick the trade date when this lot was acquired.",
            )
            lot_shares = int(
                st.sidebar.number_input(
                    f"Shares bought in lot {idx + 1}",
                    min_value=1,
                    value=10,
                    step=1,
                    key=f"lot-shares-{analysis_mode}-{evaluation_symbol or 'basket'}-{idx}",
                )
            )
            purchase_lots.append({
                "date": lot_date.isoformat(),
                "shares": lot_shares,
            })

    shares_owned = sum(lot["shares"] for lot in purchase_lots)

    return {
        "analysis_mode": analysis_mode,
        "index_choice": index_choice,
        "selected_symbol": selected_symbol,
        "horizon_years": horizon_years,
        "invest_amount": invest_amount,
        "strategy_notes": strategy_notes,
        "evaluation_symbol": evaluation_symbol,
        "evaluation_position": evaluation_position,
        "purchase_lots": purchase_lots,
        "shares_owned": shares_owned,
    }


def render_intro(is_agentic: bool = False) -> None:
    # Help button with tooltip (shortened for mobile)
    if is_agentic:
        help_text = """
        **Agentic AI Investment Assistant**
        
        📊 **Market Basket** - Analyze stocks from indices
        🎯 **Single Stock Focus** - Deep-dive with profit calculations
        📰 **News & Sentiment** - Real-time headlines & market mood
        💡 **AI Recommendations** - Buy/sell guidance with forecasts
        
        **5 Specialized Agents** work together:
        Data Collection → News → Market Mood → Analysis → Recommendations
        
        **Usage:** Configure sidebar → Choose mode → Run Analysis
        """
    else:
        help_text = """
        **Stock Investment AI Companion**
        
        📊 **Market Basket** - Screen stocks from major indices
        🎯 **Single Stock Focus** - Analyze individual stocks
        📰 **News & Sentiment** - Market headlines & mood indicators
        💡 **AI Recommendations** - Buy/sell guidance with 6M forecasts
        
        **Usage:** Configure sidebar → Choose mode → Run Analysis
        """
    
    # Mobile-friendly sidebar prompt - visible banner
    st.info(
        "📱 **Mobile users:** Tap the ☰ menu icon (top left) or swipe from the left edge "
        "to open the sidebar and configure your settings.",
        icon="⚙️"
    )
    
    # Create columns for title and help button
    col1, col2 = st.columns([0.95, 0.05])
    with col1:
        st.title("📊 Stock Investment AI Companion")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        # Using tooltip via button help parameter
        if st.button("❓", help=help_text, key="help_button_intro"):
            pass  # Button click does nothing, tooltip shows on hover


def show_snapshot_table(snapshots: List[StockSnapshot]) -> None:
    """Display comprehensive snapshot table with all 40+ metrics organized by category."""
    
    # Create tabs for different metric categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview & Price", 
        "💰 Valuation & Profitability", 
        "💳 Debt & Cash Flow",
        "📈 Technical & Risk",
        "🏢 Ownership & Timing"
    ])
    
    # Tab 1: Overview & Price Action
    with tab1:
        df_overview = pd.DataFrame([
            {
                "Symbol": snapshot.symbol,
                "Company": snapshot.short_name,
                "1M Change": snapshot.change_1m,
                "6M Change": snapshot.change_6m,
                "Forecast 6M": snapshot.forecast_slope,
                "Beta": snapshot.beta,
                "Volatility": snapshot.volatility,
                "Max Drawdown": snapshot.max_drawdown,
                "Sharpe Ratio": snapshot.sharpe_ratio,
                "Sortino Ratio": snapshot.sortino_ratio,
                "Dist 52W High": snapshot.dist_52w_high,
                "Dist 52W Low": snapshot.dist_52w_low,
                "Support Level": snapshot.support_level,
                "Resistance Level": snapshot.resistance_level,
            }
            for snapshot in snapshots
        ])
        
        st.dataframe(
            df_overview.style.format(
                {
                    "1M Change": "{:.2%}",
                    "6M Change": "{:.2%}",
                    "Forecast 6M": "{:.2%}",
                    "Beta": "{:.2f}",
                    "Volatility": "{:.2%}",
                    "Max Drawdown": "{:.2%}",
                    "Sharpe Ratio": "{:.2f}",
                    "Sortino Ratio": "{:.2f}",
                    "Dist 52W High": "{:.2%}",
                    "Dist 52W Low": "{:.2%}",
                    "Support Level": "{:,.2f}",
                    "Resistance Level": "{:,.2f}",
                },
                na_rep="NA",
            ),
            use_container_width=True,
            hide_index=True,
        )
    
    # Tab 2: Valuation & Profitability
    with tab2:
        df_valuation = pd.DataFrame([
            {
                "Symbol": snapshot.symbol,
                "Company": snapshot.short_name,
                "Market Cap (₹B)": snapshot.market_cap / 1e9 if snapshot.market_cap else None,
                "Enterprise Value (₹B)": snapshot.enterprise_value / 1e9 if snapshot.enterprise_value else None,
                "P/E": snapshot.fundamentals.get("trailingPE"),
                "PEG": snapshot.peg_ratio,
                "P/B": snapshot.price_to_book,
                "P/S": snapshot.price_to_sales,
                "EV/EBITDA": snapshot.ev_to_ebitda,
                "EPS": snapshot.fundamentals.get("trailingEps"),
                "Revenue (₹M)": snapshot.fundamentals.get("totalRevenue") / 1e6 if snapshot.fundamentals.get("totalRevenue") else None,
                "ROE": snapshot.fundamentals.get("returnOnEquity"),
                "ROIC": snapshot.roic,
                "ROA": snapshot.roa,
                "Gross Margin": snapshot.gross_margin,
                "Net Margin": snapshot.net_margin,
                "EBITDA Margin": snapshot.ebitda_margin,
                "Operating Margin": snapshot.operating_margin,
                "Revenue Growth YoY": snapshot.revenue_growth_yoy,
                "Revenue CAGR 3Y": snapshot.revenue_cagr_3y,
                "Profit Growth YoY": snapshot.profit_growth_yoy,
            }
            for snapshot in snapshots
        ])
        
        st.dataframe(
            df_valuation.style.format(
                {
                    "Market Cap (₹B)": "{:.2f}",
                    "Enterprise Value (₹B)": "{:.2f}",
                    "P/E": "{:.2f}",
                    "PEG": "{:.2f}",
                    "P/B": "{:.2f}",
                    "P/S": "{:.2f}",
                    "EV/EBITDA": "{:.2f}",
                    "EPS": "{:.2f}",
                    "Revenue (₹M)": "{:,.0f}",
                    "ROE": "{:.2%}",
                    "ROIC": "{:.2%}",
                    "ROA": "{:.2%}",
                    "Gross Margin": "{:.2%}",
                    "Net Margin": "{:.2%}",
                    "EBITDA Margin": "{:.2%}",
                    "Operating Margin": "{:.2%}",
                    "Revenue Growth YoY": "{:.2%}",
                    "Revenue CAGR 3Y": "{:.2%}",
                    "Profit Growth YoY": "{:.2%}",
                },
                na_rep="NA",
            ),
            use_container_width=True,
            hide_index=True,
        )
    
    # Tab 3: Debt & Cash Flow
    with tab3:
        df_debt_cf = pd.DataFrame([
            {
                "Symbol": snapshot.symbol,
                "Company": snapshot.short_name,
                "Total Debt (₹M)": snapshot.total_debt / 1e6 if snapshot.total_debt else None,
                "Debt/Equity": snapshot.fundamentals.get("debtToEquity"),
                "Debt/Assets": snapshot.debt_to_assets,
                "Interest Coverage": snapshot.interest_coverage,
                "Working Capital (₹M)": snapshot.working_capital / 1e6 if snapshot.working_capital else None,
                "Current Ratio": snapshot.current_ratio,
                "Quick Ratio": snapshot.quick_ratio,
                "Op Cash Flow (₹M)": snapshot.operating_cash_flow / 1e6 if snapshot.operating_cash_flow else None,
                "Inv Cash Flow (₹M)": snapshot.investing_cash_flow / 1e6 if snapshot.investing_cash_flow else None,
                "Fin Cash Flow (₹M)": snapshot.financing_cash_flow / 1e6 if snapshot.financing_cash_flow else None,
                "CapEx (₹M)": snapshot.capex / 1e6 if snapshot.capex else None,
                "Free Cash Flow (₹M)": snapshot.free_cash_flow / 1e6 if snapshot.free_cash_flow else None,
                "Cash Flow/Share": snapshot.cash_flow_per_share,
            }
            for snapshot in snapshots
        ])
        
        st.dataframe(
            df_debt_cf.style.format(
                {
                    "Total Debt (₹M)": "{:,.0f}",
                    "Debt/Equity": "{:.2f}",
                    "Debt/Assets": "{:.2%}",
                    "Interest Coverage": "{:.2f}x",
                    "Working Capital (₹M)": "{:,.0f}",
                    "Current Ratio": "{:.2f}",
                    "Quick Ratio": "{:.2f}",
                    "Op Cash Flow (₹M)": "{:,.0f}",
                    "Inv Cash Flow (₹M)": "{:,.0f}",
                    "Fin Cash Flow (₹M)": "{:,.0f}",
                    "CapEx (₹M)": "{:,.0f}",
                    "Free Cash Flow (₹M)": "{:,.0f}",
                    "Cash Flow/Share": "{:.2f}",
                },
                na_rep="NA",
            ),
            use_container_width=True,
            hide_index=True,
        )
    
    # Tab 4: Technical & Risk
    with tab4:
        df_technical = pd.DataFrame([
            {
                "Symbol": snapshot.symbol,
                "Company": snapshot.short_name,
                "RSI (14)": snapshot.rsi_14,
                "Stochastic K": snapshot.stochastic_k,
                "Stochastic D": snapshot.stochastic_d,
                "Williams %R": snapshot.williams_r,
                "MACD": snapshot.macd,
                "MACD Signal": snapshot.macd_signal,
                "50 DMA": snapshot.moving_average_50,
                "200 DMA": snapshot.moving_average_200,
                "BB Upper": snapshot.bollinger_upper,
                "BB Middle": snapshot.bollinger_middle,
                "BB Lower": snapshot.bollinger_lower,
                "OBV": snapshot.obv,
                "Avg Vol (20d)": snapshot.avg_volume_20,
                "Volume Ratio": snapshot.volume_ratio,
                "Asset Turnover": snapshot.asset_turnover,
                "Inv Turnover": snapshot.inventory_turnover,
            }
            for snapshot in snapshots
        ])
        
        st.dataframe(
            df_technical.style.format(
                {
                    "RSI (14)": "{:.1f}",
                    "Stochastic K": "{:.1f}",
                    "Stochastic D": "{:.1f}",
                    "Williams %R": "{:.1f}",
                    "MACD": "{:.2f}",
                    "MACD Signal": "{:.2f}",
                    "50 DMA": "{:,.2f}",
                    "200 DMA": "{:,.2f}",
                    "BB Upper": "{:,.2f}",
                    "BB Middle": "{:,.2f}",
                    "BB Lower": "{:,.2f}",
                    "OBV": "{:,.0f}",
                    "Avg Vol (20d)": "{:,.0f}",
                    "Volume Ratio": "{:.2f}",
                    "Asset Turnover": "{:.2f}",
                    "Inv Turnover": "{:.2f}",
                },
                na_rep="NA",
            ),
            use_container_width=True,
            hide_index=True,
        )
    
    # Tab 5: Ownership & Timing
    with tab5:
        df_ownership = pd.DataFrame([
            {
                "Symbol": snapshot.symbol,
                "Company": snapshot.short_name,
                "Promoter Holding %": snapshot.promoter_holding_pct,
                "Promoter Holding Δ": snapshot.promoter_holding_change,
                "Inst Ownership %": snapshot.institutional_ownership,
                "Float Shares (M)": snapshot.float_shares / 1e6 if snapshot.float_shares else None,
                "Dividend Yield": snapshot.dividend_yield,
                "Next Earnings": snapshot.earnings_date,
                "Ex-Dividend Date": snapshot.ex_dividend_date,
            }
            for snapshot in snapshots
        ])
        
        st.dataframe(
            df_ownership.style.format(
                {
                    "Promoter Holding %": "{:.2%}",
                    "Promoter Holding Δ": "{:.2%}",
                    "Inst Ownership %": "{:.2%}",
                    "Float Shares (M)": "{:.2f}",
                    "Dividend Yield": "{:.2%}",
                },
                na_rep="NA",
            ),
            use_container_width=True,
            hide_index=True,
        )


def show_news(news_map: Dict[str, List[NewsItem]]) -> None:
    if not news_map:
        st.info("No news headlines fetched. Set NEWS_API_KEY to enrich the analysis.")
        return

    st.subheader("Recent Headlines")
    records: List[Dict[str, str]] = []
    for symbol, articles in news_map.items():
        for article in articles:
            records.append(
                {
                    "Symbol": symbol,
                    "Headline": article.title,
                    "Source": article.source or "Unknown",
                    "Published": article.published_at.strftime("%Y-%m-%d %H:%M")
                    if article.published_at
                    else "",
                    "Link": article.url,
                }
            )

    if not records:
        st.info("No recent headlines found for the selected companies.")
        return

    news_df = pd.DataFrame(records)
    st.dataframe(
        news_df,
        use_container_width=True,
        column_config={
            "Link": st.column_config.LinkColumn("Link"),
            "Headline": st.column_config.TextColumn(width="large"),
        },
        hide_index=True,
    )


def show_market_mood(market_mood: MarketMood | None) -> None:
    if market_mood is None:
        st.info("Market mood data not available.")
        return

    st.subheader("Market Mood Index (Fear/Greed)")
    
    # Color coding based on sentiment
    if market_mood.index <= 25:
        color = "🔴"  # Extreme Fear - Red
    elif market_mood.index <= 45:
        color = "🟠"  # Fear - Orange
    elif market_mood.index <= 55:
        color = "🟡"  # Neutral - Yellow
    elif market_mood.index <= 75:
        color = "🟢"  # Greed - Green
    else:
        color = "🟢"  # Extreme Greed - Bright Green

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            "Mood Index",
            f"{market_mood.index:.1f}/100",
            delta=f"{market_mood.sentiment}",
        )
        st.markdown(f"**{color} {market_mood.sentiment}**")
    
    with col2:
        st.info(f"**{market_mood.description}**")
        st.toast(f"Recommendation: {market_mood.recommendation}", icon="💡")
    
    # Visual gauge
    st.progress(market_mood.index / 100)


def show_fii_trend_chart(fii_trend: pd.DataFrame | None) -> None:
    if fii_trend is None or fii_trend.empty:
        st.warning("FII trend data could not be loaded.")
        return

    chart_df = fii_trend.dropna(subset=["date", "netBuySell"]).tail(60)
    if chart_df.empty:
        st.warning("No recent FII trend data available after filtering.")
        return

    fig = px.bar(
        chart_df,
        x="date",
        y="netBuySell",
        title="Foreign Institutional Investor Net Buy/Sell (₹ Crores)",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_llm_recommendations(result, snapshots: List[StockSnapshot], compact: bool = False, is_single_stock: bool = False) -> None:
    """Display LLM recommendations with optional compact mode using expanders."""
    st.subheader("💡 LLM Allocation Suggestions")
    st.write(result.summary)
    
    snapshot_map = {snapshot.symbol: snapshot for snapshot in snapshots}
    
    if not result.allocations:
        # When LLM recommends waiting, show detailed analysis
        st.warning("⚠️ LLM recommended waiting before investing.")
        
        if result.guidance:
            # Extract and highlight key metrics from guidance
            guidance_lower = result.guidance.lower()
            
            # Show guidance with enhanced formatting
            with st.expander("📊 Detailed Reasoning & Metric Analysis", expanded=True):
                st.markdown("**Why wait?**")
                st.write(result.guidance)
                
                # Try to extract and display key metrics if mentioned
                if any(keyword in guidance_lower for keyword in ["p/e", "pe ratio", "price-to-earnings"]):
                    # Calculate and show average P/E if available
                    pe_values = [
                        s.fundamentals.get("trailingPE") 
                        for s in snapshots 
                        if s.fundamentals and s.fundamentals.get("trailingPE") is not None
                    ]
                    if pe_values:
                        avg_pe = sum(pe_values) / len(pe_values)
                        st.metric("Average P/E Ratio", f"{avg_pe:.2f}", 
                                help="P/E > 25 typically indicates overvaluation")
                
                if any(keyword in guidance_lower for keyword in ["rsi", "overbought"]):
                    # Calculate and show average RSI if available
                    rsi_values = [s.rsi_14 for s in snapshots if s.rsi_14 is not None]
                    if rsi_values:
                        avg_rsi = sum(rsi_values) / len(rsi_values)
                        overbought_count = sum(1 for rsi in rsi_values if rsi > 70)
                        st.metric("Average RSI (14)", f"{avg_rsi:.1f}", 
                                help=f"{overbought_count} stocks are overbought (RSI > 70)")
                
                if any(keyword in guidance_lower for keyword in ["52w", "52-week", "52 week"]):
                    # Calculate and show 52W high distance stats
                    dist_52w_values = [s.dist_52w_high for s in snapshots if s.dist_52w_high is not None]
                    if dist_52w_values:
                        avg_dist = sum(dist_52w_values) / len(dist_52w_values)
                        near_high_count = sum(1 for dist in dist_52w_values if dist > -0.05)  # Within 5% of 52W high
                        st.metric("Avg Distance from 52W High", f"{avg_dist:.2%}",
                                help=f"{near_high_count} stocks are within 5% of 52W high")
                
                if any(keyword in guidance_lower for keyword in ["forecast", "bearish", "bullish"]):
                    # Calculate and show forecast trends
                    forecast_values = [s.forecast_slope for s in snapshots if s.forecast_slope is not None]
                    if forecast_values:
                        bearish_count = sum(1 for f in forecast_values if f < 0)
                        avg_forecast = sum(forecast_values) / len(forecast_values)
                        st.metric("Forecast Trend", f"{avg_forecast:.2%} average",
                                help=f"{bearish_count}/{len(forecast_values)} stocks show bearish 6M forecasts")
                
                if any(keyword in guidance_lower for keyword in ["beta", "volatility"]):
                    # Calculate and show beta/volatility
                    beta_values = [s.beta for s in snapshots if s.beta is not None]
                    if beta_values:
                        avg_beta = sum(beta_values) / len(beta_values)
                        high_vol_count = sum(1 for b in beta_values if b > 1.2)
                        st.metric("Average Beta", f"{avg_beta:.2f}",
                                help=f"{high_vol_count} stocks have high volatility (Beta > 1.2)")
                
                if any(keyword in guidance_lower for keyword in ["peg", "price/earnings to growth"]):
                    # Calculate and show PEG ratios
                    peg_values = [s.peg_ratio for s in snapshots if s.peg_ratio is not None]
                    if peg_values:
                        avg_peg = sum(peg_values) / len(peg_values)
                        overvalued_peg = sum(1 for p in peg_values if p > 1.5)
                        st.metric("Average PEG Ratio", f"{avg_peg:.2f}",
                                help=f"{overvalued_peg} stocks have PEG > 1.5 (overvalued)")
                
                # Show market mood context
                if any(keyword in guidance_lower for keyword in ["greed", "fear", "market mood"]):
                    st.markdown("---")
                    st.markdown("**Market Sentiment Context:**")
                    st.markdown("Market mood index reflects overall market sentiment (Fear to Greed scale). "
                              "A high greed index suggests overvaluation, while fear indicates potential buying opportunities.")
        else:
            st.info("No detailed reasoning provided. The recommendation is based on overall market conditions.")
    else:
        # Normal display for when allocations are provided
        if result.guidance:
            # Show guidance as both toast and visible info for important messages
            if any(keyword in result.guidance.lower() for keyword in ["cautious", "wait", "avoid", "warning", "risk"]):
                st.toast("⚠️ " + result.guidance[:100], icon="⚠️")
            st.info(result.guidance)
        
        # Summary table
        allocations_df = pd.DataFrame(
            [
                {
                    "Symbol": allocation.symbol,
                    "Allocation %": f"{allocation.allocation_pct:.1f}%",
                }
                for allocation in result.allocations
            ]
        )
        st.dataframe(allocations_df, use_container_width=True, hide_index=True)
        
        # Technical indicators summary in expander
        if not compact:
            tech_summary_data = []
            for allocation in result.allocations:
                snapshot = snapshot_map.get(allocation.symbol)
                if snapshot is None:
                    continue
                tech_summary_data.append({
                    "Symbol": allocation.symbol,
                    "Forecast 6M": snapshot.forecast_slope,
                    "RSI (14)": snapshot.rsi_14,
                    "50DMA vs 200DMA": (
                        "Golden Cross" if snapshot.moving_average_50 and snapshot.moving_average_200 
                        and snapshot.moving_average_50 > snapshot.moving_average_200 
                        else "Death Cross" if snapshot.moving_average_50 and snapshot.moving_average_200 
                        else "N/A"
                    ),
                    "MACD Signal": (
                        "Bullish" if snapshot.macd and snapshot.macd_signal 
                        and snapshot.macd > snapshot.macd_signal 
                        else "Bearish" if snapshot.macd and snapshot.macd_signal 
                        else "N/A"
                    ),
                    "Volume Ratio": snapshot.volume_ratio,
                    "Dist from 52W High": snapshot.dist_52w_high,
                })
                
            if tech_summary_data:
                with st.expander("📊 Technical Indicators Summary", expanded=False):
                    tech_df = pd.DataFrame(tech_summary_data)
                    st.dataframe(
                        tech_df.style.format(
                            {
                                "Forecast 6M": "{:.2%}",
                                "RSI (14)": "{:.1f}",
                                "Volume Ratio": "{:.2f}",
                                "Dist from 52W High": "{:.2%}",
                            },
                            na_rep="N/A",
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
        
        # Detailed rationale and risks in expanders
        with st.expander("📝 Detailed Rationale & Risks", expanded=False):
            for allocation in result.allocations:
                st.markdown(f"### {allocation.symbol} ({allocation.allocation_pct:.1f}%)")
                st.markdown(f"**Rationale:** {allocation.rationale}")
                st.markdown(f"**Risks:** {allocation.risks}")
                st.markdown("---")

    # Single stock evaluation - only show if single stock focus is selected
    if result.evaluation and is_single_stock:
        with st.expander("🎯 Single Stock Evaluation", expanded=True):
            evaluation = result.evaluation
            symbol = evaluation.get("symbol", "Unknown")
            recommendation = (evaluation.get("recommendation") or "N/A").upper()
            st.write(f"**{symbol}** — Recommendation: **{recommendation}**")
            
            col1, col2 = st.columns(2)
            with col1:
                investor_pos = evaluation.get("investor_position")
                if investor_pos:
                    st.markdown(f"**Position:** {investor_pos}")
                if evaluation.get("shares_owned"):
                    st.markdown(f"**Holding:** {evaluation['shares_owned']} share(s)")
            with col2:
                snapshot = snapshot_map.get(symbol)
                slope = getattr(snapshot, "forecast_slope", None) if snapshot else None
                if slope is not None:
                    trend_text = "UPWARD" if slope > 0 else "DOWNWARD"
                    st.markdown(f"**6M Forecast:** {trend_text} {slope:+.2%}")
            
            lot_summary = evaluation.get("lot_summary") or {}
            latest_price = lot_summary.get("latest_close")
            avg_cost = lot_summary.get("avg_cost")
            total_shares = lot_summary.get("total_shares")
            
            shares_to_sell = evaluation.get("shares_to_sell")
            if shares_to_sell is not None and latest_price and avg_cost is not None:
                shares_to_sell = min(shares_to_sell, total_shares or shares_to_sell)
                profit_per_share = latest_price - avg_cost
                profit_pct = (profit_per_share / avg_cost) if avg_cost else None
                expected_profit = profit_per_share * shares_to_sell
                proceeds = latest_price * shares_to_sell
                
                st.toast(f"💡 Suggested: Sell {shares_to_sell} share(s)", icon="💡")
                st.markdown(
                    f"- Proceeds: ₹{proceeds:,.2f} | Profit: ₹{expected_profit:,.2f} "
                    f"({profit_per_share:,.2f} per share)"
                )
                
                meaningful_profit = expected_profit >= MIN_SELL_PROFIT_ABS and (profit_pct is None or profit_pct >= MIN_SELL_PROFIT_PCT)
                if not meaningful_profit:
                    st.warning(
                        f"⚠️ Profit below threshold (₹{MIN_SELL_PROFIT_ABS:,.0f} or {MIN_SELL_PROFIT_PCT:.0%}). "
                        "Consider holding for long-term targets."
                    )
            
            if evaluation.get("reasoning"):
                st.markdown(f"**Reasoning:** {evaluation['reasoning']}")
            if evaluation.get("confidence"):
                st.markdown(f"**Confidence:** {evaluation['confidence']}")
            
            if evaluation.get("purchase_lots"):
                with st.expander("📋 Purchase History", expanded=False):
                    lot_table = pd.DataFrame(evaluation["purchase_lots"]).rename(
                        columns={
                            "requested_date": "Input Date",
                            "matched_date": "Matched Date",
                            "shares": "Shares",
                            "purchase_price": "Purchase Price",
                            "current_price": "Current Price",
                            "unrealized_pl": "Unrealised P/L",
                            "unrealized_pl_pct": "Unrealised %",
                        }
                    )
                    st.dataframe(
                        lot_table.style.format(
                            {
                                "Purchase Price": "₹{:,.2f}",
                                "Current Price": "₹{:,.2f}",
                                "Unrealised P/L": "₹{:,.2f}",
                                "Unrealised %": "{:+.2%}",
                            },
                            na_rep="NA",
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )


def show_price_charts(result, snapshots: List[StockSnapshot], is_single_stock: bool = False) -> None:
    """Display price charts with forecasts for recommended stocks and evaluated stock."""
    snapshot_map = {snapshot.symbol: snapshot for snapshot in snapshots}
    
    # Charts for recommended stocks
    if result.allocations:
        st.subheader("📈 Price Charts & Forecasts (Recommended Stocks)")
        for allocation in result.allocations:
            snapshot = snapshot_map.get(allocation.symbol)
            if snapshot is None or snapshot.price_history is None or snapshot.price_history.empty:
                continue
            
            with st.expander(f"{allocation.symbol} - Chart & Analysis", expanded=False):
                history_df = snapshot.price_history.copy()
                if "Date" in history_df.columns:
                    history_df.rename(columns={"Date": "date"}, inplace=True)
                elif "date" not in history_df.columns:
                    history_df.reset_index(inplace=True)
                if "date" not in history_df.columns:
                    continue
                history_tail = history_df.tail(130)
                
                forecast_slope = getattr(snapshot, "forecast_slope", None)
                forecast_color = "green" if forecast_slope and forecast_slope > 0 else "red"
                forecast_title = f"{allocation.symbol} – Last 6 Months Price & Forecast"
                if forecast_slope is not None:
                    trend_text = "BULLISH" if forecast_slope > 0 else "BEARISH"
                    forecast_title += f" ({trend_text}: {forecast_slope:.2%})"
                
                fig = px.line(history_tail, x="date", y="Close", title=forecast_title)
                if snapshot.forecast is not None and not snapshot.forecast.empty:
                    fig.add_scatter(
                        x=snapshot.forecast["date"],
                        y=snapshot.forecast["forecast"],
                        mode="lines",
                        name="Forecast (6M)",
                        line=dict(dash="dash", color=forecast_color),
                    )
                
                # Add technical indicator overlays
                if snapshot.moving_average_50 is not None and "date" in history_tail.columns:
                    ma50_series = history_tail.set_index("date")["Close"].rolling(50).mean()
                    if not ma50_series.empty:
                        fig.add_scatter(
                            x=ma50_series.index,
                            y=ma50_series.values,
                            mode="lines",
                            name="50 DMA",
                            line=dict(color="orange", width=1),
                        )
                if snapshot.moving_average_200 is not None and "date" in history_tail.columns:
                    ma200_series = history_tail.set_index("date")["Close"].rolling(200).mean()
                    if not ma200_series.empty:
                        fig.add_scatter(
                            x=ma200_series.index,
                            y=ma200_series.values,
                            mode="lines",
                            name="200 DMA",
                            line=dict(color="purple", width=1),
                        )
                
                st.plotly_chart(fig, use_container_width=True)
                
                if forecast_slope is not None and forecast_slope < 0:
                    st.warning(
                        f"⚠️ **Warning**: {allocation.symbol} has a BEARISH 6-month forecast ({forecast_slope:.2%}). "
                        "Consider this risk carefully before investing."
                    )
    
    # Chart for evaluated stock - only show if single stock focus is selected
    if result.evaluation and is_single_stock:
        evaluation = result.evaluation
        symbol = evaluation.get("symbol", "Unknown")
        snapshot = snapshot_map.get(symbol)
        
        if snapshot and snapshot.forecast is not None and not snapshot.forecast.empty:
            with st.expander(f"{symbol} - Historical & Forecast Chart", expanded=True):
                history_df = snapshot.price_history.copy()
                if "Date" in history_df.columns:
                    history_df.rename(columns={"Date": "date"}, inplace=True)
                elif "date" not in history_df.columns:
                    history_df.reset_index(inplace=True)
                if "date" in history_df.columns:
                    history_df = history_df.tail(180)
                    fig_eval = px.line(history_df, x="date", y="Close", title=f"{symbol} – Historical & Forecasted Prices")
                    fig_eval.add_scatter(
                        x=snapshot.forecast["date"],
                        y=snapshot.forecast["forecast"],
                        mode="lines",
                        name="Forecast (6M)",
                        line=dict(dash="dash", color="blue"),
                    )
                    st.plotly_chart(fig_eval, use_container_width=True)


def show_run_details(run) -> None:
    """Display full details of a specific run using tabs and expanders."""
    st.markdown("---")
    st.markdown(f"### Run Details: {run.created_at.strftime('%Y-%m-%d %H:%M')}")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Horizon", f"{run.horizon_years} years")
    with col2:
        st.metric("Universe", run.stock_universe)
    with col3:
        st.metric("Amount", f"₹{run.invest_amount:,.0f}")
    with col4:
        if run.market_mood_index is not None:
            st.metric("Mood", f"{run.market_mood_index:.1f}", f"{run.market_mood_sentiment or 'N/A'}")
    
    # Strategy notes in expander
    if run.strategy_notes:
        with st.expander("📝 Strategy Notes", expanded=False):
            st.write(run.strategy_notes)
    
    # LLM summary
    with st.expander("💡 LLM Summary & Guidance", expanded=True):
        st.write(run.llm_summary)
        if run.llm_guidance:
            st.info(run.llm_guidance)
    
    # Load and display data in tabs
    try:
        run_data = load_run_data(run)
        snapshots = run_data["snapshots"]
        news_map = run_data["news_map"]
        fii_trend = run_data["fii_trend"]
        evaluation = run_data["evaluation"]
        
        tabs = st.tabs(["📊 Allocations", "📈 Market Data", "📰 News & FII"])
        
        with tabs[0]:
            if run.suggestions:
                suggestions_df = pd.DataFrame([
                    {
                        "Symbol": s.symbol,
                        "Allocation %": f"{s.allocation_pct:.1f}%",
                    }
                    for s in run.suggestions
                ])
                st.dataframe(suggestions_df, use_container_width=True, hide_index=True)
                
                # Detailed rationale in expander
                with st.expander("📝 Detailed Rationale & Risks", expanded=False):
                    for s in run.suggestions:
                        st.markdown(f"### {s.symbol} ({s.allocation_pct:.1f}%)")
                        st.markdown(f"**Rationale:** {s.rationale}")
                        st.markdown(f"**Risks:** {s.risks}")
                        st.markdown("---")
            
            # Only show single stock evaluation if it was a single stock run
            if evaluation and (run.universe_name or "").startswith("Single Stock:"):
                with st.expander("🎯 Single Stock Evaluation", expanded=False):
                    st.json(evaluation)
        
        with tabs[1]:
            if snapshots:
                with st.expander("📊 Market Snapshot", expanded=True):
                    show_snapshot_table(snapshots)
        
        with tabs[2]:
            if news_map:
                show_news(news_map)
            if fii_trend is not None and not fii_trend.empty:
                show_fii_trend_chart(fii_trend)
            
    except Exception as exc:
        st.error(f"Error loading run data: {exc}")


def show_recent_runs() -> None:
    # Count unknown runs
    runs = fetch_recent_runs(limit=1000)  # Get more to count unknown
    unknown_count = sum(1 for run in runs if run.universe_name is None)
    runs = runs[:10]  # Limit for display
    
    # Add button to delete unknown runs if any exist
    if unknown_count > 0:
        st.warning(f"⚠️ Found {unknown_count} run(s) with 'Unknown' universe_name.")
        if st.button("🗑️ Delete All Unknown Runs", type="secondary", key="delete_unknown_runs"):
            deleted_count = delete_unknown_runs()
            if deleted_count > 0:
                st.success(f"✅ Successfully deleted {deleted_count} unknown run(s).")
                st.rerun()
        st.markdown("---")
    
    if not runs:
        st.info("No previous runs found. Run an analysis to get started.")
        return
    
    # Show summary table
    data = [
        {
            "Timestamp": run.created_at.strftime("%Y-%m-%d %H:%M"),
            "Horizon": run.horizon_years,
            "Stocks": run.stock_universe,
            "Amount": f"₹{run.invest_amount:,.0f}",
            "Suggestions": len(run.suggestions),
            "Summary": run.llm_summary[:80] + ("…" if len(run.llm_summary) > 80 else ""),
        }
        for run in runs
    ]
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    
    # Allow selecting a run to view details
    st.markdown("---")
    st.subheader("View Run Details")
    run_options = {
        f"{run.created_at.strftime('%Y-%m-%d %H:%M')} - {run.universe_name or 'Unknown'}": run.id
        for run in runs
    }
    
    if run_options:
        selected_run_label = st.selectbox(
            "Select a run to view full details:",
            options=list(run_options.keys()),
            key="run_selector"
        )
        
        if selected_run_label:
            selected_run_id = run_options[selected_run_label]
            selected_run = fetch_run_by_id(selected_run_id)
            if selected_run:
                show_run_details(selected_run)


def run_pipeline(inputs: Dict[str, Any]) -> None:
    news_api_key = os.getenv("NEWS_API_KEY")
    total_steps = 7
    current_step = 0
    progress = st.progress(0.0, text="Starting analysis…")
    step_placeholder = st.empty()

    def advance(step_label: str) -> None:
        nonlocal current_step
        current_step += 1
        progress.progress(current_step / total_steps, text=step_label)
        step_placeholder.success(step_label)

    analysis_mode: str = inputs["analysis_mode"]
    is_single_stock = analysis_mode == "Single Stock Focus"
    index_choice: Optional[str] = inputs.get("index_choice")
    selected_symbol: Optional[str] = (
        (inputs.get("selected_symbol") or "").strip().upper() if is_single_stock else None
    )
    evaluation_symbol: Optional[str] = (inputs.get("evaluation_symbol") or "").strip().upper()
    evaluation_position: Optional[str] = inputs.get("evaluation_position") or "Do not own"
    purchase_lots: List[Dict[str, Any]] = inputs.get("purchase_lots") or []
    shares_owned = sum(int(lot.get("shares", 0)) for lot in purchase_lots) if evaluation_position == "Already own" else 0

    if is_single_stock:
        evaluation_symbol = selected_symbol or evaluation_symbol

    if is_single_stock:
        if not evaluation_symbol:
            progress.empty()
            step_placeholder.error("Please choose a stock in the sidebar before running the analysis.")
            return
        symbol_pool: List[str] = [evaluation_symbol]
        universe_label = f"Single Stock: {evaluation_symbol}"
    else:
        if not index_choice:
            progress.empty()
            step_placeholder.error("Please select an index universe before running the analysis.")
            return
        step_placeholder.info(f"Loading constituents for {index_choice}…")
        try:
            constituents = load_index_constituents(index_choice)
        except Exception as exc:  # noqa: BLE001
            progress.empty()
            step_placeholder.error(f"Unable to fetch {index_choice} constituents: {exc}")
            return
        symbols = (
            constituents.get("symbol", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .str.upper()
            .tolist()
        )
        if not symbols:
            progress.empty()
            step_placeholder.error(f"{index_choice} constituents list is empty.")
            return
        if evaluation_symbol and evaluation_symbol not in symbols:
            symbols.append(evaluation_symbol)
        symbol_pool = list(dict.fromkeys(symbols))
        universe_label = f"Index Basket: {index_choice}"

    advance("Symbol universe prepared.")

    step_placeholder.info("Fetching price history and fundamentals…")
    fundamentals_progress = st.empty()
    
    def update_fundamentals_progress(symbol: str, done: int, total: int) -> None:
        fundamentals_progress.info(f"Fetching fundamentals… ({done}/{total}) – latest: {symbol}")
    
    try:
        snapshots, failures = fetch_snapshots(
            symbol_pool,
            max_workers=6,  # Reduced to help with rate limiting
            progress_callback=update_fundamentals_progress,
        )
    except Exception as exc:  # noqa: BLE001
        progress.empty()
        step_placeholder.error(f"Unable to fetch market data: {exc}")
        fundamentals_progress.empty()
        return
    finally:
        fundamentals_progress.empty()
    if failures:
        st.warning(
            "Some symbols could not be processed: "
            + ", ".join(f"{symbol} ({reason})" for symbol, reason in list(failures.items())[:10])
        )
    if not snapshots:
        progress.empty()
        step_placeholder.error("No market data could be fetched. Try reducing the number of stocks.")
        return

    evaluation_snapshot = None
    enriched_purchase_lots: List[Dict[str, Any]] = []
    lot_summary: Dict[str, Any] = {}
    if evaluation_symbol:
        evaluation_snapshot = next((snap for snap in snapshots if snap.symbol == evaluation_symbol), None)
        if evaluation_snapshot and purchase_lots:
            enriched_purchase_lots, lot_summary = enrich_purchase_lots(
                evaluation_snapshot,
                purchase_lots,
            )
            if lot_summary.get("total_shares"):
                shares_owned = lot_summary["total_shares"]
        elif evaluation_position == "Already own" and not evaluation_snapshot:
            st.warning(
                f"Could not fetch historical data for {evaluation_symbol}. Purchase lot analysis may be incomplete."
            )

    advance("Fetched price & fundamentals.")

    forecast_threshold = -0.02
    if is_single_stock:
        llm_snapshots = snapshots
        bearish = [snap for snap in snapshots if getattr(snap, "forecast_slope", 0) < forecast_threshold]
        if bearish:
            bearish_snap = bearish[0]
            if evaluation_position == "Already own" and shares_owned > 0:
                recommendation_hint = (
                    f"You currently hold {shares_owned} share(s); strongly consider booking profits or tightening stops."
                )
            else:
                recommendation_hint = "Avoid fresh entries until price momentum turns positive."
            st.error(
                f"📉 Guardrail alert: {bearish_snap.symbol} shows a negative 6-month trend"
                f" ({bearish_snap.forecast_slope:.2%}). {recommendation_hint}"
            )
    else:
        excluded_snapshots = [
            snap
            for snap in snapshots
            if getattr(snap, "forecast_slope", None) is not None
            and getattr(snap, "forecast_slope", None) < forecast_threshold
            and snap.symbol != evaluation_symbol
        ]
        if excluded_snapshots:
            st.warning(
                "Forecast guardrail removed declining candidates: "
                + ", ".join(
                    f"{snap.symbol} ({getattr(snap, 'forecast_slope', 0):.2%})"
                    for snap in excluded_snapshots[:10]
                )
            )
        llm_snapshots = [snap for snap in snapshots if snap not in excluded_snapshots]
        if not llm_snapshots:
            progress.empty()
            step_placeholder.error(
                "All screened candidates currently show adverse forecasts. Adjust filters or try later."
            )
            return

    step_placeholder.info("Fetching FII trend data…")
    fii_trend = None
    try:
        fii_trend = load_fii_trend()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"FII trend data unavailable: {exc}")
    advance("FII trend data ready.")

    step_placeholder.info("Calculating market mood index…")
    market_mood = None
    try:
        market_mood = calculate_market_mood()
    except MarketMoodError as exc:
        st.warning(f"Market mood calculation failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Market mood data unavailable: {exc}")
    advance("Market mood calculated.")

    step_placeholder.info("Fetching news headlines…")
    news_progress = st.empty()

    def update_news_progress(symbol: str, done: int, total: int) -> None:
        news_progress.info(f"Fetching news headlines… ({done}/{total}) – latest: {symbol}")

    news_map: Dict[str, List[NewsItem]] = {}
    news_errors: Dict[str, str] = {}
    try:
        news_map, news_errors = fetch_headlines(
            [snap.symbol for snap in snapshots],
            api_key=news_api_key,
            max_per_symbol=3,
            max_workers=min(len(snapshots), 8) or 1,
            symbol_to_name={snap.symbol: snap.short_name or snap.symbol for snap in snapshots},
            progress_callback=update_news_progress,
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"News fetching failed: {exc}")
    finally:
        news_progress.empty()

    if news_errors:
        st.warning(
            "Some news queries failed: "
            + ", ".join(f"{symbol} ({reason})" for symbol, reason in news_errors.items())
        )
    advance("News headlines fetched.")

    step_placeholder.info("Requesting LLM allocation guidance…")
    try:
        llm_news_map = {snap.symbol: news_map.get(snap.symbol, []) for snap in llm_snapshots}
        llm_result, llm_raw = llm_pick_and_allocate(
            investment_horizon=int(inputs["horizon_years"]),
            invest_amount=float(inputs["invest_amount"]),
            strategy_notes=str(inputs["strategy_notes"]),
            snapshots=llm_snapshots,
            news_map=llm_news_map,
            fii_trend=fii_trend,
            market_mood=market_mood,
            evaluation_symbol=evaluation_symbol or None,
            evaluation_position=evaluation_position,
            evaluation_shares=shares_owned,
            evaluation_lots=enriched_purchase_lots,
        )
    except LLMServiceError as exc:
        progress.empty()
        step_placeholder.error(f"LLM service error: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        progress.empty()
        step_placeholder.error(f"Unexpected error when calling LLM: {exc}")
        return
    advance("LLM guidance received.")

    if enriched_purchase_lots:
        formatted_summary = {
            "latest_close": lot_summary.get("latest_close"),
            "latest_date": lot_summary.get("latest_date").isoformat() if lot_summary.get("latest_date") else None,
            "total_shares": lot_summary.get("total_shares"),
            "total_cost": lot_summary.get("total_cost"),
            "total_value": lot_summary.get("total_value"),
            "avg_cost": lot_summary.get("avg_cost"),
        }
        if llm_result.evaluation is None:
            llm_result.evaluation = {}
        llm_result.evaluation.setdefault("purchase_lots", enriched_purchase_lots)
        llm_result.evaluation.setdefault("shares_owned", shares_owned)
        llm_result.evaluation.setdefault("investor_position", evaluation_position)
        llm_result.evaluation.setdefault("lot_summary", formatted_summary)

    step_placeholder.info("Saving run to database…")
    try:
        run_id = log_run(
            horizon_years=int(inputs["horizon_years"]),
            stock_universe=len(symbol_pool),
            invest_amount=float(inputs["invest_amount"]),
            strategy_notes=str(inputs["strategy_notes"]),
            snapshots=llm_snapshots,
            news_map=llm_news_map,
            llm_result=llm_result,
            llm_raw=llm_raw,
            fii_trend=fii_trend,
            universe_name=universe_label,
            custom_symbols=None,
            market_mood=market_mood,
            stock_evaluation=llm_result.evaluation,
        )
    except Exception as exc:  # noqa: BLE001
        st.toast(f"⚠️ Failed to persist run: {str(exc)[:50]}...", icon="⚠️")
        run_id = None
    else:
        st.toast("💾 Run saved to database", icon="💾")
    advance("Run saved.")

    progress.empty()
    step_placeholder.empty()
    st.toast("✅ Analysis complete!", icon="✅")

    if run_id:
        st.toast(f"✅ Run logged successfully with ID {run_id[:8]}...", icon="✅")

    # Organize results into tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary & Recommendations", "📈 Market Data", "📰 News & Analysis", "📉 Charts & Forecasts"])
    
    with tab1:
        show_market_mood(market_mood)
        if enriched_purchase_lots:
            with st.expander("💼 Your Purchase History", expanded=True):
                show_purchase_history(enriched_purchase_lots, lot_summary)
        show_llm_recommendations(result=llm_result, snapshots=snapshots, compact=True, is_single_stock=is_single_stock)
    
    with tab2:
        with st.expander("📊 Market Snapshot (Full Data)", expanded=True):
            show_snapshot_table(snapshots)
    
    with tab3:
        show_news(news_map)
        show_fii_trend_chart(fii_trend)
    
    with tab4:
        show_price_charts(llm_result, snapshots, is_single_stock=is_single_stock)


def show_performance_tracking() -> None:
    """Display performance tracking - select a run to see suggested vs current prices with graphs."""
    
    runs = fetch_recent_runs(limit=50)
    if not runs:
        st.info("No runs tracked yet. Run an analysis to start tracking.")
        return
    
    # Run selector
    run_options = {
        f"{run.created_at.strftime('%Y-%m-%d %H:%M')} - {run.universe_name or 'Unknown'} - {run.invest_amount:,.0f}": run.id
        for run in runs
    }
    
    if not run_options:
        st.info("No runs available.")
        return
    
    selected_run_label = st.selectbox(
        "Select a run to track performance:",
        options=list(run_options.keys()),
        key="performance_run_selector"
    )
    
    if not selected_run_label:
        return
    
    selected_run_id = run_options[selected_run_label]
    selected_run = fetch_run_by_id(selected_run_id)
    
    if not selected_run:
        st.error("Run not found.")
        return
    
    # Show run summary
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Date", selected_run.created_at.strftime("%Y-%m-%d") if selected_run.created_at else "N/A")
    with col2:
        st.metric("Horizon", f"{selected_run.horizon_years} years")
    with col3:
        st.metric("Investment Amount", f"₹{selected_run.invest_amount:,.0f}")
    with col4:
        days_since = (datetime.utcnow() - selected_run.created_at).days if selected_run.created_at else None
        st.metric("Days Since", days_since or "N/A")
    
    # Get predictions for this run
    predictions = fetch_predictions_by_run_id(selected_run_id)
    if not predictions:
        st.info("No predictions found for this run.")
        return
    
    # Update prices button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Update Prices", type="primary", use_container_width=True):
            st.toast("🔄 Updating prices... This may take a moment.", icon="🔄")
            updated_count = 0
            error_count = 0
            
            progress_bar = st.progress(0)
            total = len(predictions)
            
            for idx, pred in enumerate(predictions):
                try:
                    ticker_symbol = f"{pred.symbol}.NS" if not pred.symbol.endswith('.NS') else pred.symbol
                    ticker = yf.Ticker(ticker_symbol)
                    
                    # Fetch current price
                    hist = ticker.history(period="1d")
                    current_price = None
                    if not hist.empty and "Close" in hist.columns:
                        current_price = float(hist["Close"].iloc[-1])
                    
                    # Fetch current fundamental metrics
                    current_metrics = None
                    if current_price:
                        try:
                            info = ticker.get_info()
                            # Extract current fundamental metrics
                            current_metrics = {
                                "pe": info.get("trailingPE"),
                                "peg": info.get("pegRatio"),
                                "roe": info.get("returnOnEquity"),
                                "roic": None,  # Will need to calculate from financials if needed
                                "debt_to_equity": info.get("debtToEquity"),
                                "interest_coverage": None,  # Will need to calculate if needed
                                "revenue_growth": None,  # Will need to calculate if needed
                                "profit_growth": None,  # Will need to calculate if needed
                                "beta": info.get("beta"),
                            }
                            # Get ROIC if available (would need to calculate from financials/balance sheet)
                            # For now, we'll use what's available from info
                        except Exception:
                            current_metrics = None
                    
                    if current_price:
                        update_prediction_price(pred.id, current_price, current_metrics=current_metrics)
                        updated_count += 1
                    else:
                        error_count += 1
                except Exception:
                    error_count += 1
                
                progress_bar.progress((idx + 1) / total)
            
            progress_bar.empty()
            if updated_count > 0:
                st.toast(f"✅ Updated {updated_count} price(s).", icon="✅")
            if error_count > 0:
                st.toast(f"⚠️ Failed to update {error_count} price(s).", icon="⚠️")
            st.rerun()
    
    # Refresh predictions after update
    predictions = fetch_predictions_by_run_id(selected_run_id)
    
    # Create tabs for price tracking and fundamental metrics tracking
    perf_tab1, perf_tab2 = st.tabs(["💰 Price Performance", "📊 Fundamental Metrics Change"])
    
    with perf_tab1:
        # Price Performance Tab
        pred_data = []
        symbols = []
        suggested_prices = []
        current_prices = []
        returns = []
        allocation_pcts = []
        total_invested = 0.0
        total_current_value = 0.0
        
        for pred in predictions:
            return_pct = None
            if pred.current_price and pred.suggested_price:
                return_pct = (pred.current_price - pred.suggested_price) / pred.suggested_price
            
            # Calculate profit for this stock if investment amount is known
            stock_profit = None
            stock_value = None
            if pred.current_price and pred.suggested_price and pred.allocation_pct and selected_run.invest_amount:
                invested_amount = selected_run.invest_amount * (pred.allocation_pct / 100)
                shares = invested_amount / pred.suggested_price
                current_value = shares * pred.current_price
                stock_profit = current_value - invested_amount
                stock_value = current_value
                total_invested += invested_amount
                total_current_value += current_value
            
            pred_data.append({
                "Symbol": pred.symbol,
                "Allocation %": f"{pred.allocation_pct:.1f}%",
                "Suggested Price (Then)": f"₹{pred.suggested_price:,.2f}",
                "Current Price (Now)": f"₹{pred.current_price:,.2f}" if pred.current_price else "Not updated",
                "Change": f"₹{pred.current_price - pred.suggested_price:+,.2f}" if pred.current_price and pred.suggested_price else "N/A",
                "Return %": f"{return_pct:+.2%}" if return_pct is not None else "N/A",
                "Profit": f"₹{stock_profit:+,.2f}" if stock_profit is not None else "N/A",
                "Days": pred.days_since_suggestion or "N/A",
            })
            
            if pred.current_price and pred.suggested_price:
                symbols.append(pred.symbol)
                suggested_prices.append(pred.suggested_price)
                current_prices.append(pred.current_price)
                returns.append(return_pct * 100)  # Convert to percentage
                allocation_pcts.append(pred.allocation_pct)
        
        # Display summary metrics
        if total_invested > 0:
            total_profit = total_current_value - total_invested
            total_profit_pct = (total_profit / total_invested) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Invested", f"₹{total_invested:,.2f}")
            with col2:
                st.metric("Current Value", f"₹{total_current_value:,.2f}")
            with col3:
                st.metric("Total Profit/Loss", f"₹{total_profit:+,.2f}", f"{total_profit_pct:+.2f}%")
            with col4:
                updated_count = sum(1 for p in predictions if p.current_price is not None)
                st.metric("Prices Updated", f"{updated_count}/{len(predictions)}")
        
        # Display table
        st.markdown("### 📊 Stock Performance Table")
        df = pd.DataFrame(pred_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Charts
        if symbols and suggested_prices and current_prices:
            tab1, tab2, tab3 = st.tabs(["📈 Price Comparison", "📊 Returns by Stock", "💰 Profit by Stock"])
            
            with tab1:
                # Price comparison chart (Suggested vs Current)
                comparison_df = pd.DataFrame({
                    "Symbol": symbols,
                    "Suggested Price": suggested_prices,
                    "Current Price": current_prices,
                })
                
                fig = px.bar(
                    comparison_df,
                    x="Symbol",
                    y=["Suggested Price", "Current Price"],
                    title="Suggested Price (Then) vs Current Price (Now)",
                    labels={"value": "Price (₹)", "variable": "Price Type"},
                    barmode="group",
                    color_discrete_map={"Suggested Price": "blue", "Current Price": "green"},
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Returns percentage chart
                returns_df = pd.DataFrame({
                    "Symbol": symbols,
                    "Return %": returns,
                })
                
                fig = px.bar(
                    returns_df,
                    x="Symbol",
                    y="Return %",
                    title="Return Percentage by Stock",
                    labels={"Return %": "Return (%)", "Symbol": "Stock Symbol"},
                    color="Return %",
                    color_continuous_scale="RdYlGn",
                )
                fig.update_xaxes(tickangle=45)
                fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Profit by stock chart (if investment amount is known)
                if total_invested > 0:
                    profit_data = []
                    for pred in predictions:
                        if pred.current_price and pred.suggested_price and pred.allocation_pct:
                            invested_amount = selected_run.invest_amount * (pred.allocation_pct / 100)
                            shares = invested_amount / pred.suggested_price
                            current_value = shares * pred.current_price
                            stock_profit = current_value - invested_amount
                            profit_data.append({
                                "Symbol": pred.symbol,
                                "Profit": stock_profit,
                            })
                    
                    if profit_data:
                        profit_df = pd.DataFrame(profit_data)
                        fig = px.bar(
                            profit_df,
                            x="Symbol",
                            y="Profit",
                            title="Profit/Loss by Stock (₹)",
                            labels={"Profit": "Profit/Loss (₹)", "Symbol": "Stock Symbol"},
                            color="Profit",
                            color_continuous_scale="RdYlGn",
                        )
                        fig.update_xaxes(tickangle=45)
                        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Investment amount not available for profit calculation.")
    
    with perf_tab2:
        # Fundamental Metrics Change Tab
        st.markdown("### 📊 Fundamental Metrics Comparison")
        st.info("Compare fundamental metrics at suggestion time vs. current values. Click 'Update Prices' to refresh current metrics.")
        
        metrics_data = []
        for pred in predictions:
            metrics_data.append({
                "Symbol": pred.symbol,
                "Allocation %": f"{pred.allocation_pct:.1f}%",
                # P/E Ratio
                "P/E (Then)": f"{pred.suggested_pe:.2f}" if pred.suggested_pe else "N/A",
                "P/E (Now)": f"{pred.current_pe:.2f}" if pred.current_pe else "N/A",
                "P/E Δ": f"{(pred.current_pe - pred.suggested_pe):+.2f}" if pred.current_pe and pred.suggested_pe else "N/A",
                # PEG Ratio
                "PEG (Then)": f"{pred.suggested_peg:.2f}" if pred.suggested_peg else "N/A",
                "PEG (Now)": f"{pred.current_peg:.2f}" if pred.current_peg else "N/A",
                "PEG Δ": f"{(pred.current_peg - pred.suggested_peg):+.2f}" if pred.current_peg and pred.suggested_peg else "N/A",
                # ROE
                "ROE (Then)": f"{pred.suggested_roe:.2%}" if pred.suggested_roe else "N/A",
                "ROE (Now)": f"{pred.current_roe:.2%}" if pred.current_roe else "N/A",
                "ROE Δ": f"{(pred.current_roe - pred.suggested_roe):+.2%}" if pred.current_roe and pred.suggested_roe else "N/A",
                # ROIC
                "ROIC (Then)": f"{pred.suggested_roic:.2%}" if pred.suggested_roic else "N/A",
                "ROIC (Now)": f"{pred.current_roic:.2%}" if pred.current_roic else "N/A",
                "ROIC Δ": f"{(pred.current_roic - pred.suggested_roic):+.2%}" if pred.current_roic and pred.suggested_roic else "N/A",
                # Debt/Equity
                "Debt/Eq (Then)": f"{pred.suggested_debt_to_equity:.2f}" if pred.suggested_debt_to_equity else "N/A",
                "Debt/Eq (Now)": f"{pred.current_debt_to_equity:.2f}" if pred.current_debt_to_equity else "N/A",
                "Debt/Eq Δ": f"{(pred.current_debt_to_equity - pred.suggested_debt_to_equity):+.2f}" if pred.current_debt_to_equity and pred.suggested_debt_to_equity else "N/A",
                # Interest Coverage
                "Int Cov (Then)": f"{pred.suggested_interest_coverage:.2f}x" if pred.suggested_interest_coverage else "N/A",
                "Int Cov (Now)": f"{pred.current_interest_coverage:.2f}x" if pred.current_interest_coverage else "N/A",
                "Int Cov Δ": f"{(pred.current_interest_coverage - pred.suggested_interest_coverage):+.2f}x" if pred.current_interest_coverage and pred.suggested_interest_coverage else "N/A",
                # Revenue Growth
                "Rev Growth (Then)": f"{pred.suggested_revenue_growth:.2%}" if pred.suggested_revenue_growth else "N/A",
                "Rev Growth (Now)": f"{pred.current_revenue_growth:.2%}" if pred.current_revenue_growth else "N/A",
                "Rev Growth Δ": f"{(pred.current_revenue_growth - pred.suggested_revenue_growth):+.2%}" if pred.current_revenue_growth and pred.suggested_revenue_growth else "N/A",
                # Profit Growth
                "Profit Growth (Then)": f"{pred.suggested_profit_growth:.2%}" if pred.suggested_profit_growth else "N/A",
                "Profit Growth (Now)": f"{pred.current_profit_growth:.2%}" if pred.current_profit_growth else "N/A",
                "Profit Growth Δ": f"{(pred.current_profit_growth - pred.suggested_profit_growth):+.2%}" if pred.current_profit_growth and pred.suggested_profit_growth else "N/A",
                # Beta
                "Beta (Then)": f"{pred.suggested_beta:.2f}" if pred.suggested_beta else "N/A",
                "Beta (Now)": f"{pred.current_beta:.2f}" if pred.current_beta else "N/A",
                "Beta Δ": f"{(pred.current_beta - pred.suggested_beta):+.2f}" if pred.current_beta and pred.suggested_beta else "N/A",
            })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        else:
            st.info("No fundamental metrics data available. Run an analysis and then update prices to see metric changes.")


def main() -> None:
    inputs = sidebar_inputs()
    render_intro(is_agentic=False)

    st.caption(
        f"OpenAI key detected: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'} | "
        f"News API key detected: {'Yes' if os.getenv('NEWS_API_KEY') else 'No'}"
    )

    # Main sections with clear separation
    tab1, tab2, tab3 = st.tabs(["🔍 Investment Analysis", "📚 Recent Runs", "📊 Performance Tracking"])
    
    with tab1:
        st.markdown("### 🔍 Investment Analysis")
        st.markdown("Run a new analysis to get investment suggestions based on market data, news, and AI recommendations.")
        if st.button("Run Investment Analysis", type="primary", use_container_width=True):
            run_pipeline(inputs)
    
    with tab2:
        st.markdown("### 📚 Recent Runs")
        st.markdown("View details of your previous analysis runs.")
        show_recent_runs()
    
    with tab3:
        st.markdown("### 📊 Performance Tracking")
        st.markdown("Track how your investment suggestions have performed over time.")
        show_performance_tracking()


if __name__ == "__main__":
    main()

