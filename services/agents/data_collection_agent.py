"""Agent responsible for collecting market data."""

from typing import Any, Dict, List, Optional
from services.agents.base_agent import BaseAgent, AgentResult, AgentState
from services.core.market_data import StockSnapshot, fetch_snapshots
from services.core.nse import (
    AVAILABLE_INDICES,
    fetch_index_constituents,
    fetch_nifty_500_constituents,
    fetch_fii_trend,
)
import pandas as pd
import streamlit as st


class DataCollectionAgent(BaseAgent):
    """Agent that collects market data including stock prices, fundamentals, and FII trends."""
    
    def __init__(self):
        super().__init__(
            name="Data Collection Agent",
            description="Collects market data: stock prices, fundamentals, index constituents, and FII trends"
        )
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Collect market data based on context."""
        result = AgentResult(success=True, state=AgentState())
        
        try:
            # Determine symbol pool
            analysis_mode = context.get("analysis_mode")
            is_single_stock = analysis_mode == "Single Stock Focus"
            
            if is_single_stock:
                evaluation_symbol = context.get("evaluation_symbol") or context.get("selected_symbol")
                if not evaluation_symbol:
                    result.add_error("symbol_selection", "No symbol selected for single stock analysis")
                    return result
                symbol_pool = [evaluation_symbol.upper()]
                universe_label = f"Single Stock: {evaluation_symbol.upper()}"
            else:
                index_choice = context.get("index_choice")
                if not index_choice:
                    result.add_error("index_selection", "No index selected for basket analysis")
                    return result
                
                try:
                    if index_choice == "NIFTY 500":
                        constituents = fetch_nifty_500_constituents()
                    else:
                        constituents = fetch_index_constituents(index_choice)
                    
                    symbols = (
                        constituents.get("symbol", pd.Series(dtype=str))
                        .dropna()
                        .astype(str)
                        .str.upper()
                        .tolist()
                    )
                    
                    if not symbols:
                        result.add_error("constituents", f"{index_choice} constituents list is empty")
                        return result
                    
                    # Add evaluation symbol if provided
                    evaluation_symbol = context.get("evaluation_symbol")
                    if evaluation_symbol and evaluation_symbol.upper() not in symbols:
                        symbols.append(evaluation_symbol.upper())
                    
                    symbol_pool = list(dict.fromkeys(symbols))
                    universe_label = f"Index Basket: {index_choice}"
                    
                except Exception as e:
                    result.add_error("constituents_fetch", str(e))
                    return result
            
            result.add_data("symbol_pool", symbol_pool)
            result.add_data("universe_label", universe_label)
            result.add_data("is_single_stock", is_single_stock)
            
            # Fetch stock snapshots
            progress_callback = context.get("progress_callback")
            
            try:
                snapshots, failures = fetch_snapshots(
                    symbol_pool,
                    max_workers=6,
                    progress_callback=progress_callback,
                )
                
                if failures:
                    result.state.metadata["fetch_failures"] = failures
                
                if not snapshots:
                    result.add_error("snapshots", "No market data could be fetched")
                    return result
                
                result.add_data("snapshots", snapshots)
                
            except Exception as e:
                result.add_error("snapshots_fetch", str(e))
                return result
            
            # Fetch FII trend data
            try:
                fii_trend = fetch_fii_trend()
                result.add_data("fii_trend", fii_trend)
            except Exception as e:
                result.state.metadata["fii_error"] = str(e)
                # FII is optional, don't fail the agent
            
            return result
            
        except Exception as e:
            result.add_error("execution", str(e))
            return result

