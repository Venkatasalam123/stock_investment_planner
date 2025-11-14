"""
Agentic AI version of the Stock Investment AI App.

This version uses an agentic architecture where different agents handle
different aspects of the investment analysis workflow.
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
import yfinance as yf

# Import helper functions from original app
from app import (
    load_symbol_catalog,
    enrich_purchase_lots,
    show_snapshot_table,
    show_news,
    show_fii_trend_chart,
    show_market_mood,
    show_purchase_history,
    show_llm_recommendations,
    show_price_charts,
    show_run_details,
    show_recent_runs,
    show_performance_tracking,
    sidebar_inputs,
    render_intro,
)

# Import agents
from services.agents import CoordinatorAgent
from services.core.market_data import StockSnapshot
from services.core.market_mood import MarketMood
from services.core.news import NewsItem
from services.core.nse import AVAILABLE_INDICES
from storage.database import (
    fetch_recent_runs,
    fetch_run_by_id,
    log_run,
)

load_dotenv()

st.set_page_config(
    page_title="Stock Investment AI Assistant (Agents)",
    page_icon="🤖",
    layout="wide",
)


def run_agentic_pipeline(inputs: Dict[str, Any]) -> None:
    """
    Run the investment analysis pipeline using agentic AI.
    
    This function orchestrates agents to:
    1. Collect market data
    2. Fetch news headlines
    3. Calculate market mood
    4. Analyze and filter stocks
    5. Generate recommendations
    """
    # Initialize progress tracking
    progress = st.progress(0.0, text="🤖 Initializing agents...")
    step_placeholder = st.empty()
    
    # Progress tracking callbacks
    current_progress = [0.0]
    total_agents = 5  # Data Collection, Market Mood, News, Analysis, Recommendation
    
    def update_progress(value: float) -> None:
        current_progress[0] = value
        progress.progress(value, text=f"🤖 Agents working... {int(value * 100)}%")
    
    def update_step(step_text: str) -> None:
        step_placeholder.info(f"🤖 {step_text}")
    
    # Fundamentals progress placeholder
    fundamentals_progress = st.empty()
    news_progress = st.empty()
    
    def fundamentals_progress_callback(symbol: str, done: int, total: int) -> None:
        fundamentals_progress.info(f"📊 Fetching fundamentals… ({done}/{total}) – latest: {symbol}")
    
    def news_progress_callback(symbol: str, done: int, total: int) -> None:
        news_progress.info(f"📰 Fetching news headlines… ({done}/{total}) – latest: {symbol}")
    
    try:
        # Prepare context for agents
        analysis_mode: str = inputs["analysis_mode"]
        is_single_stock = analysis_mode == "Single Stock Focus"
        index_choice: Optional[str] = inputs.get("index_choice")
        selected_symbol: Optional[str] = (
            (inputs.get("selected_symbol") or "").strip().upper() if is_single_stock else None
        )
        evaluation_symbol: Optional[str] = (inputs.get("evaluation_symbol") or "").strip().upper()
        evaluation_position: Optional[str] = inputs.get("evaluation_position") or "Do not own"
        purchase_lots: List[Dict[str, Any]] = inputs.get("purchase_lots", [])
        shares_owned = sum(int(lot.get("shares", 0)) for lot in purchase_lots) if evaluation_position == "Already own" else 0
        
        if is_single_stock:
            evaluation_symbol = selected_symbol or evaluation_symbol
        
        # Validate inputs
        if is_single_stock:
            if not evaluation_symbol:
                progress.empty()
                step_placeholder.error("❌ Please choose a stock in the sidebar before running the analysis.")
                return
        else:
            if not index_choice:
                progress.empty()
                step_placeholder.error("❌ Please select an index universe before running the analysis.")
                return
        
        # Initialize coordinator agent
        coordinator = CoordinatorAgent()
        
        # Prepare context for agents
        context: Dict[str, Any] = {
            "analysis_mode": analysis_mode,
            "is_single_stock": is_single_stock,
            "index_choice": index_choice,
            "selected_symbol": selected_symbol,
            "evaluation_symbol": evaluation_symbol,
            "evaluation_position": evaluation_position,
            "purchase_lots": purchase_lots,
            "shares_owned": shares_owned,
            "horizon_years": int(inputs["horizon_years"]),
            "invest_amount": float(inputs["invest_amount"]),
            "strategy_notes": str(inputs.get("strategy_notes", "")),
            "progress_callback": update_progress,
            "step_callback": update_step,
            "fundamentals_progress_callback": fundamentals_progress_callback,
            "news_progress_callback": news_progress_callback,
        }
        
        # Execute coordinator agent (which executes all sub-agents)
        update_step("🤖 Starting agentic analysis pipeline...")
        result = coordinator.execute(context)
        
        # Check if execution was successful
        if not result.success:
            progress.empty()
            error_msg = result.message or "Agent execution failed"
            error_details = ", ".join([f"{k}: {v}" for k, v in result.state.errors.items()])
            step_placeholder.error(f"❌ {error_msg}\n\nDetails: {error_details}")
            fundamentals_progress.empty()
            news_progress.empty()
            return
        
        # Extract results from agent state
        symbol_pool = result.get_data("symbol_pool")
        universe_label = result.get_data("universe_label")
        snapshots: List[StockSnapshot] = result.get_data("snapshots", [])
        fii_trend = result.get_data("fii_trend")
        market_mood: Optional[MarketMood] = result.get_data("market_mood")
        news_map: Dict[str, List[NewsItem]] = result.get_data("news_map", {})
        llm_snapshots: List[StockSnapshot] = result.get_data("llm_snapshots", snapshots)
        llm_result = result.get_data("llm_result")
        llm_raw = result.get_data("llm_raw")
        
        # Handle purchase lots enrichment if needed (after agent execution to avoid circular imports)
        enriched_purchase_lots: List[Dict[str, Any]] = []
        lot_summary: Dict[str, Any] = {}
        if result.get_data("needs_purchase_lots_enrichment") and evaluation_symbol:
            evaluation_snapshot = result.get_data("evaluation_snapshot")
            purchase_lots_from_agent = result.get_data("purchase_lots", purchase_lots)
            if evaluation_snapshot and purchase_lots_from_agent:
                try:
                    enriched_purchase_lots, lot_summary = enrich_purchase_lots(
                        evaluation_snapshot,
                        purchase_lots_from_agent,
                    )
                except Exception as e:
                    st.warning(f"⚠️ Purchase lots enrichment failed: {str(e)}")
        
        if not snapshots:
            progress.empty()
            step_placeholder.error("❌ No market data could be fetched. Try again later.")
            fundamentals_progress.empty()
            news_progress.empty()
            return
        
        # Display warnings for metadata
        metadata = result.state.metadata
        if "fetch_failures" in metadata:
            failures = metadata["fetch_failures"]
            if failures:
                st.warning(
                    "⚠️ Some symbols could not be processed: "
                    + ", ".join(f"{symbol} ({reason})" for symbol, reason in list(failures.items())[:10])
                )
        
        if "news_errors" in metadata:
            news_errors = metadata["news_errors"]
            if news_errors:
                st.warning(
                    "⚠️ Some news queries failed: "
                    + ", ".join(f"{symbol} ({reason})" for symbol, reason in list(news_errors.items())[:5])
                )
        
        if "bearish_warning" in metadata and is_single_stock:
            bearish = metadata["bearish_warning"]
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
        
        if "excluded_snapshots" in metadata and not is_single_stock:
            excluded_snapshots = metadata["excluded_snapshots"]
            if excluded_snapshots:
                st.warning(
                    "⚠️ Forecast guardrail removed declining candidates: "
                    + ", ".join(
                        f"{snap.symbol} ({getattr(snap, 'forecast_slope', 0):.2%})"
                        for snap in excluded_snapshots[:10]
                    )
                )
        
        if not llm_result:
            progress.empty()
            step_placeholder.error("❌ Failed to generate recommendations. Please try again.")
            fundamentals_progress.empty()
            news_progress.empty()
            return
        
        # Update shares_owned if enriched lots were processed
        if enriched_purchase_lots and lot_summary.get("total_shares"):
            shares_owned = lot_summary["total_shares"]
        
        # Enrich evaluation with purchase lots if available
        if enriched_purchase_lots and llm_result.evaluation:
            formatted_summary = {
                "latest_close": lot_summary.get("latest_close"),
                "latest_date": lot_summary.get("latest_date").isoformat() if lot_summary.get("latest_date") else None,
                "total_shares": lot_summary.get("total_shares"),
                "total_cost": lot_summary.get("total_cost"),
                "total_value": lot_summary.get("total_value"),
                "avg_cost": lot_summary.get("avg_cost"),
            }
            llm_result.evaluation.setdefault("purchase_lots", enriched_purchase_lots)
            llm_result.evaluation.setdefault("shares_owned", shares_owned)
            llm_result.evaluation.setdefault("investor_position", evaluation_position)
            llm_result.evaluation.setdefault("lot_summary", formatted_summary)
        
        # Save run to database
        update_step("💾 Saving run to database...")
        try:
            run_id = log_run(
                horizon_years=int(inputs["horizon_years"]),
                stock_universe=len(symbol_pool) if symbol_pool else 0,
                invest_amount=float(inputs["invest_amount"]),
                strategy_notes=str(inputs.get("strategy_notes", "")),
                snapshots=llm_snapshots,
                news_map={snap.symbol: news_map.get(snap.symbol, []) for snap in llm_snapshots},
                llm_result=llm_result,
                llm_raw=llm_raw,
                fii_trend=fii_trend,
                universe_name=universe_label,
                custom_symbols=None,
                market_mood=market_mood,
                stock_evaluation=llm_result.evaluation if llm_result.evaluation else None,
            )
            st.toast("💾 Run saved to database", icon="💾")
        except Exception as exc:
            st.toast(f"⚠️ Failed to persist run: {str(exc)[:50]}...", icon="⚠️")
            run_id = None
        
        update_progress(1.0)
        progress.empty()
        step_placeholder.empty()
        fundamentals_progress.empty()
        news_progress.empty()
        st.toast("✅ Agentic analysis complete!", icon="✅")
        
        if run_id:
            st.toast(f"✅ Run logged successfully with ID {run_id[:8]}...", icon="✅")
        
        # Display results
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Summary & Recommendations", 
            "📈 Market Data", 
            "📰 News & Analysis", 
            "📉 Charts & Forecasts"
        ])
        
        with tab1:
            show_market_mood(market_mood)
            if enriched_purchase_lots:
                with st.expander("💼 Your Purchase History", expanded=True):
                    show_purchase_history(enriched_purchase_lots, lot_summary)
            show_llm_recommendations(
                result=llm_result, 
                snapshots=snapshots, 
                compact=True, 
                is_single_stock=is_single_stock
            )
        
        with tab2:
            with st.expander("📊 Market Snapshot (Full Data)", expanded=True):
                show_snapshot_table(snapshots)
        
        with tab3:
            show_news(news_map)
            show_fii_trend_chart(fii_trend)
        
        with tab4:
            show_price_charts(llm_result, snapshots, is_single_stock=is_single_stock)
        
    except Exception as exc:
        progress.empty()
        step_placeholder.error(f"❌ Unexpected error in agentic pipeline: {str(exc)}")
        fundamentals_progress.empty()
        news_progress.empty()
        st.exception(exc)


def main() -> None:
    """Main entry point for the agentic app."""
    inputs = sidebar_inputs()
    render_intro()
    
    st.caption(
        f"🤖 Agentic Mode | "
        f"OpenAI key: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'} | "
        f"News API key: {'✅' if os.getenv('NEWS_API_KEY') else '❌'}"
    )
    
    # Add info about agentic mode
    with st.expander("ℹ️ About Agentic Mode", expanded=False):
        st.markdown("""
        This version uses an **agentic AI architecture** where specialized agents handle different aspects:
        
        - 🤖 **Data Collection Agent**: Fetches market data, stock prices, fundamentals, and FII trends
        - 📰 **News Collection Agent**: Gathers recent news headlines for stocks
        - 📊 **Market Mood Agent**: Calculates market sentiment (fear/greed index)
        - 🔍 **Analysis Agent**: Performs technical/fundamental analysis and filters candidates
        - 💡 **Recommendation Agent**: Uses LLM to generate investment recommendations
        
        All agents are orchestrated by a **Coordinator Agent** that manages the workflow.
        """)
    
    # Main sections
    tab1, tab2, tab3 = st.tabs([
        "🔍 Investment Analysis", 
        "📚 Recent Runs", 
        "📊 Performance Tracking"
    ])
    
    with tab1:
        st.markdown("### 🔍 Investment Analysis (Agentic)")
        st.markdown("Run a new analysis using agentic AI to get investment suggestions based on market data, news, and AI recommendations.")
        if st.button("🤖 Run Agentic Analysis", type="primary", use_container_width=True):
            run_agentic_pipeline(inputs)
    
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

