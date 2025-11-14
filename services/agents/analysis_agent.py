"""Agent responsible for analyzing stock data and filtering candidates."""

from typing import Any, Dict, List
from services.agents.base_agent import BaseAgent, AgentResult, AgentState
from services.core.market_data import StockSnapshot


class AnalysisAgent(BaseAgent):
    """Agent that performs technical and fundamental analysis on stock data."""
    
    def __init__(self):
        super().__init__(
            name="Analysis Agent",
            description="Analyzes stock data, filters candidates, and enriches purchase lots"
        )
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Analyze stock data and filter candidates."""
        result = AgentResult(success=True, state=AgentState())
        
        try:
            snapshots = context.get("snapshots")
            if not snapshots:
                result.add_error("snapshots", "No snapshots available in context")
                return result
            
            is_single_stock = context.get("is_single_stock", False)
            evaluation_symbol = context.get("evaluation_symbol")
            evaluation_position = context.get("evaluation_position", "Do not own")
            purchase_lots = context.get("purchase_lots", [])
            
            # Apply forecast guardrail
            forecast_threshold = -0.02
            
            if is_single_stock:
                # For single stock, keep all snapshots but warn if bearish
                llm_snapshots = snapshots
                bearish = [
                    snap for snap in snapshots 
                    if getattr(snap, "forecast_slope", 0) < forecast_threshold
                ]
                
                if bearish:
                    result.state.metadata["bearish_warning"] = bearish
                    
                # Handle purchase lots enrichment if needed
                # Note: Purchase lots enrichment is handled in app_agents.py after agent execution
                # to avoid circular dependencies. We just mark that it needs to be done.
                if evaluation_symbol and purchase_lots:
                    evaluation_snapshot = next(
                        (snap for snap in snapshots if snap.symbol == evaluation_symbol),
                        None
                    )
                    if evaluation_snapshot:
                        result.add_data("needs_purchase_lots_enrichment", True)
                        result.add_data("evaluation_snapshot", evaluation_snapshot)
                        result.add_data("purchase_lots", purchase_lots)
                            
            else:
                # For basket, exclude bearish candidates (except evaluation symbol)
                excluded_snapshots = [
                    snap
                    for snap in snapshots
                    if getattr(snap, "forecast_slope", None) is not None
                    and getattr(snap, "forecast_slope", None) < forecast_threshold
                    and snap.symbol != evaluation_symbol
                ]
                
                if excluded_snapshots:
                    result.state.metadata["excluded_snapshots"] = excluded_snapshots
                
                llm_snapshots = [
                    snap for snap in snapshots 
                    if snap not in excluded_snapshots
                ]
                
                if not llm_snapshots:
                    result.add_error("filtering", "All candidates show adverse forecasts")
                    return result
            
            result.add_data("llm_snapshots", llm_snapshots)
            return result
            
        except Exception as e:
            result.add_error("execution", str(e))
            return result

