"""Agent responsible for generating investment recommendations using LLM."""

from typing import Any, Dict, Optional
from services.agents.base_agent import BaseAgent, AgentResult, AgentState
from services.core.llm import LLMServiceError, llm_pick_and_allocate
from services.core.market_data import StockSnapshot
from services.core.market_mood import MarketMood


class RecommendationAgent(BaseAgent):
    """Agent that uses LLM to generate investment recommendations."""
    
    def __init__(self):
        super().__init__(
            name="Recommendation Agent",
            description="Uses LLM to analyze data and generate investment recommendations"
        )
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Generate investment recommendations using LLM."""
        result = AgentResult(success=True, state=AgentState())
        
        try:
            llm_snapshots = context.get("llm_snapshots")
            if not llm_snapshots:
                result.add_error("snapshots", "No filtered snapshots available in context")
                return result
            
            news_map = context.get("news_map", {})
            fii_trend = context.get("fii_trend")
            market_mood = context.get("market_mood")
            
            investment_horizon = context.get("horizon_years")
            invest_amount = context.get("invest_amount")
            strategy_notes = context.get("strategy_notes", "")
            
            evaluation_symbol = context.get("evaluation_symbol")
            evaluation_position = context.get("evaluation_position", "Do not own")
            evaluation_shares = context.get("shares_owned", 0)
            evaluation_lots = context.get("enriched_purchase_lots", [])
            
            if investment_horizon is None or invest_amount is None:
                result.add_error("inputs", "Missing required investment parameters")
                return result
            
            # Filter news map to only include snapshots we're analyzing
            llm_news_map = {
                snap.symbol: news_map.get(snap.symbol, [])
                for snap in llm_snapshots
            }
            
            try:
                llm_result, llm_raw = llm_pick_and_allocate(
                    investment_horizon=int(investment_horizon),
                    invest_amount=float(invest_amount),
                    strategy_notes=str(strategy_notes),
                    snapshots=llm_snapshots,
                    news_map=llm_news_map,
                    fii_trend=fii_trend,
                    market_mood=market_mood,
                    evaluation_symbol=evaluation_symbol,
                    evaluation_position=evaluation_position,
                    evaluation_shares=evaluation_shares,
                    evaluation_lots=evaluation_lots,
                )
                
                result.add_data("llm_result", llm_result)
                result.add_data("llm_raw", llm_raw)
                return result
                
            except LLMServiceError as e:
                result.add_error("llm_service", str(e))
                return result
            except Exception as e:
                result.add_error("llm_execution", str(e))
                return result
            
        except Exception as e:
            result.add_error("execution", str(e))
            return result

