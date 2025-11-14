"""Agent responsible for calculating market mood index."""

from typing import Any, Dict
from services.agents.base_agent import BaseAgent, AgentResult, AgentState
from services.core.market_mood import MarketMood, MarketMoodError, calculate_market_mood


class MarketMoodAgent(BaseAgent):
    """Agent that calculates market mood (fear/greed) index."""
    
    def __init__(self):
        super().__init__(
            name="Market Mood Agent",
            description="Calculates market mood index based on volatility, momentum, and volume"
        )
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Calculate market mood index."""
        result = AgentResult(success=True, state=AgentState())
        
        try:
            market_mood = None
            try:
                market_mood = calculate_market_mood()
                result.add_data("market_mood", market_mood)
            except MarketMoodError as e:
                result.state.metadata["mood_error"] = str(e)
                # Market mood is optional, don't fail the agent
            except Exception as e:
                result.state.metadata["mood_error"] = str(e)
                # Market mood is optional, don't fail the agent
            
            return result
            
        except Exception as e:
            result.add_error("execution", str(e))
            return result

