"""
Lightweight health-check / testing tool for the Stock Investment AI App.

Usage:
    python -m stock_investment_planner.app_health_check

This script focuses on:
1. Testing all available market index choices (data connectivity).
2. Verifying that LLM allocations are deterministic for fixed inputs.
3. Sanity-checking the core data used in each main UI tab.
4. Running a few common checks (env vars, DB connectivity).
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

from services.core.nse import AVAILABLE_INDICES, fetch_index_constituents
from services.core.llm import llm_pick_and_allocate, LLMServiceError
from services.agents.data_collection_agent import DataCollectionAgent
from services.agents.news_agent import NewsAgent
from services.agents.market_mood_agent import MarketMoodAgent
from services.agents.base_agent import AgentResult
from storage.database import init_db, fetch_recent_runs


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_all_indices() -> bool:
    """Test that all AVAILABLE_INDICES can fetch constituents."""
    _print_header("1) Testing all market indices (INDEX_CHOICE)")
    ok = True
    for index_name in AVAILABLE_INDICES:
        try:
            df = fetch_index_constituents(index_name)
            if df is None or df.empty:
                ok = False
                print(f"[FAIL] {index_name}: empty constituents dataframe")
            elif "symbol" not in df.columns and "Symbol" not in df.columns:
                ok = False
                print(f"[FAIL] {index_name}: no 'symbol' column in constituents")
            else:
                print(f"[PASS] {index_name}: {len(df)} constituents")
        except Exception as exc:
            ok = False
            print(f"[ERROR] {index_name}: {type(exc).__name__} - {exc}")
    return ok


def _run_data_collection(index_choice: str) -> Tuple[AgentResult, Dict[str, Any]]:
    """Run DataCollectionAgent for a given index and return agent result and context."""
    data_agent = DataCollectionAgent()

    context: Dict[str, Any] = {
        "analysis_mode": "Index Basket",
        "is_single_stock": False,
        "index_choice": index_choice,
        "selected_symbol": None,
        "evaluation_symbol": None,
        "evaluation_position": "Do not own",
        "purchase_lots": [],
        "shares_owned": 0,
        "horizon_years": 3,
        "horizon_months": 36,  # 3 years = 36 months
        "invest_amount": 100000.0,
        "strategy_notes": "Health check run",
        # Accept arbitrary args because fetch_snapshots may call this with
        # (symbol, done, total) or similar. We don't need the values here.
        "progress_callback": lambda *args, **kwargs: None,
        "step_callback": lambda text: None,
        "fundamentals_progress_callback": lambda symbol, done, total: None,
        "news_progress_callback": lambda symbol, done, total: None,
    }

    result = data_agent.execute(context)
    return result, context


def test_deterministic_llm() -> bool:
    """
    Test that two LLM calls with the SAME inputs return the same allocations.

    Assumes:
    - OPENAI_API_KEY is set
    - Temperature is 0 in services.core.llm._invoke_llm
    """
    _print_header("2) Testing LLM determinism (same inputs -> same allocations)")

    # Use a representative index if available
    index_choice = "NIFTY 50" if "NIFTY 50" in AVAILABLE_INDICES else AVAILABLE_INDICES[0]

    data_result, context = _run_data_collection(index_choice)
    if not data_result.success:
        print("[FAIL] DataCollectionAgent failed; cannot test LLM determinism.")
        print(f"Errors: {data_result.state.errors}")
        return False

    snapshots = data_result.get_data("snapshots", [])
    fii_trend = data_result.get_data("fii_trend")
    symbol_pool = data_result.get_data("symbol_pool", [])

    if not snapshots:
        print("[FAIL] No snapshots returned from DataCollectionAgent.")
        return False

    # Run MarketMoodAgent
    mood_agent = MarketMoodAgent()
    mood_result = mood_agent.execute(context)
    market_mood = mood_result.get_data("market_mood")

    # Run NewsAgent
    news_agent = NewsAgent()
    news_result = news_agent.execute({"snapshots": snapshots})
    news_map = news_result.get_data("news_map", {})

    try:
        parsed1, raw1, usage1 = llm_pick_and_allocate(
            investment_horizon=context["horizon_months"],
            invest_amount=context["invest_amount"],
            strategy_notes=context["strategy_notes"],
            snapshots=snapshots,
            news_map=news_map,
            fii_trend=fii_trend,
            market_mood=market_mood,
            evaluation_symbol=None,
            evaluation_position=None,
            evaluation_shares=None,
            evaluation_lots=None,
        )
        parsed2, raw2, usage2 = llm_pick_and_allocate(
            investment_horizon=context["horizon_months"],
            invest_amount=context["invest_amount"],
            strategy_notes=context["strategy_notes"],
            snapshots=snapshots,
            news_map=news_map,
            fii_trend=fii_trend,
            market_mood=market_mood,
            evaluation_symbol=None,
            evaluation_position=None,
            evaluation_shares=None,
            evaluation_lots=None,
        )
    except LLMServiceError as exc:
        print(f"[ERROR] LLM call failed: {exc}")
        return False

    allocs1 = [(a.symbol, round(a.allocation_pct, 1)) for a in parsed1.allocations]
    allocs2 = [(a.symbol, round(a.allocation_pct, 1)) for a in parsed2.allocations]

    if allocs1 == allocs2 and parsed1.summary == parsed2.summary:
        print(f"[PASS] LLM allocations and summary are deterministic for {index_choice}.")
        return True

    print("[FAIL] LLM outputs differ between two runs with identical inputs.")
    print(f"Allocations #1: {allocs1}")
    print(f"Allocations #2: {allocs2}")
    return False


def test_core_tab_data() -> bool:
    """
    Sanity-check that the core data backing each main UI tab is present:
    - Market mood (tab: Summary & Recommendations)
    - Snapshots table (tab: Market Data)
    - News & FII trend (tab: News & Analysis)
    - Price history for recommended stocks (tab: Charts & Forecasts)
    """
    _print_header("3) Testing core tab data availability")

    index_choice = "NIFTY 50" if "NIFTY 50" in AVAILABLE_INDICES else AVAILABLE_INDICES[0]
    data_result, context = _run_data_collection(index_choice)
    if not data_result.success:
        print("[FAIL] DataCollectionAgent failed; cannot test tab data.")
        print(f"Errors: {data_result.state.errors}")
        return False

    snapshots = data_result.get_data("snapshots", [])
    fii_trend = data_result.get_data("fii_trend")

    if not snapshots:
        print("[FAIL] No snapshots; Market Data tab would be empty.")
        return False

    # Market mood
    mood_agent = MarketMoodAgent()
    mood_result = mood_agent.execute(context)
    market_mood = mood_result.get_data("market_mood")
    if market_mood is None:
        print("[WARN] Market mood not available (Summary tab will show info message).")
    else:
        print("[PASS] Market mood available.")

    # News
    news_agent = NewsAgent()
    news_result = news_agent.execute({"snapshots": snapshots})
    news_map = news_result.get_data("news_map", {})
    if not news_map:
        print("[WARN] News map is empty (News tab will show info message).")
    else:
        print(f"[PASS] News available for {len(news_map)} symbol(s).")

    # LLM results (for Summary & Charts tabs)
    try:
        llm_result, raw, usage = llm_pick_and_allocate(
            investment_horizon=context["horizon_months"],
            invest_amount=context["invest_amount"],
            strategy_notes=context["strategy_notes"],
            snapshots=snapshots,
            news_map=news_map,
            fii_trend=fii_trend,
            market_mood=market_mood,
            evaluation_symbol=None,
            evaluation_position=None,
            evaluation_shares=None,
            evaluation_lots=None,
        )
    except LLMServiceError as exc:
        print(f"[FAIL] LLM call failed while testing tab data: {exc}")
        return False

    ok = True

    if not llm_result.summary:
        ok = False
        print("[FAIL] LLM summary is empty (Summary tab).")
    else:
        print("[PASS] LLM summary present.")

    if llm_result.allocations:
        # Ensure each recommended symbol has a snapshot and some price history
        snapshot_map = {s.symbol: s for s in snapshots}
        missing = []
        no_history = []
        for alloc in llm_result.allocations:
            snap = snapshot_map.get(alloc.symbol)
            if not snap:
                missing.append(alloc.symbol)
            elif snap.price_history is None or snap.price_history.empty:
                no_history.append(alloc.symbol)

        if missing:
            ok = False
            print(f"[FAIL] Missing snapshots for recommended symbols: {missing}")
        else:
            print("[PASS] Snapshots exist for all recommended symbols.")

        if no_history:
            print(f"[WARN] No price history for some recommended symbols: {no_history}")
        else:
            print("[PASS] Price history present for all recommended symbols.")
    else:
        print("[INFO] LLM returned no allocations (WAIT scenario). Charts tab will be limited.")

    if fii_trend is None or fii_trend.empty:
        print("[WARN] FII trend not available (News & Analysis tab will show warning).")
    else:
        print(f"[PASS] FII trend data available with {len(fii_trend)} rows.")

    return ok


def test_common_cases() -> bool:
    """
    Other common, lightweight checks:
    - Environment variables for API keys.
    - Database connectivity and ability to read recent runs.
    """
    _print_header("4) Testing common cases (env + DB)")
    ok = True

    # Env vars
    from os import getenv

    openai_key = getenv("OPENAI_API_KEY")
    if not openai_key:
        ok = False
        print("[FAIL] OPENAI_API_KEY is not set.")
    else:
        print("[PASS] OPENAI_API_KEY is set.")

    news_key = getenv("NEWS_API_KEY")
    if not news_key:
        print("[WARN] NEWS_API_KEY is not set (news may still work via RSS fallback).")
    else:
        print("[PASS] NEWS_API_KEY is set.")

    # DB connectivity
    try:
        init_db()
        runs = fetch_recent_runs(limit=5)
        print(f"[PASS] DB reachable. Recent runs fetched: {len(runs)}")
    except Exception as exc:
        ok = False
        print(f"[FAIL] Database connectivity/init failed: {type(exc).__name__} - {exc}")

    return ok


def main() -> None:
    load_dotenv()

    overall_ok = True
    overall_ok &= test_all_indices()
    overall_ok &= test_deterministic_llm()
    overall_ok &= test_core_tab_data()
    overall_ok &= test_common_cases()

    _print_header("Health check summary")
    if overall_ok:
        print("✅ All checks passed (with possible warnings).")
        sys.exit(0)
    else:
        print("❌ Some checks failed. See logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()


