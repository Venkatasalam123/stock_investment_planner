"""Agent responsible for collecting news headlines."""

from typing import Any, Dict, List
from services.agents.base_agent import BaseAgent, AgentResult, AgentState
from services.core.news import NewsItem, fetch_headlines
import os


class NewsAgent(BaseAgent):
    """Agent that collects news headlines for stocks."""
    
    def __init__(self):
        super().__init__(
            name="News Collection Agent",
            description="Collects recent news headlines for stock symbols"
        )
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Collect news headlines for the provided stock symbols."""
        result = AgentResult(success=True, state=AgentState())
        
        try:
            snapshots = context.get("snapshots")
            if not snapshots:
                result.add_error("snapshots", "No snapshots available in context")
                return result
            
            news_api_key = os.getenv("NEWS_API_KEY")
            progress_callback = context.get("news_progress_callback")
            
            symbols = [snap.symbol for snap in snapshots]
            symbol_to_name = {
                snap.symbol: snap.short_name or snap.symbol 
                for snap in snapshots
            }
            
            try:
                news_map, news_errors = fetch_headlines(
                    symbols,
                    api_key=news_api_key,
                    max_per_symbol=3,
                    max_workers=min(len(snapshots), 8) or 1,
                    symbol_to_name=symbol_to_name,
                    progress_callback=progress_callback,
                )
                
                if news_errors:
                    result.state.metadata["news_errors"] = news_errors
                
                result.add_data("news_map", news_map)
                return result
                
            except Exception as e:
                result.add_error("news_fetch", str(e))
                return result
            
        except Exception as e:
            result.add_error("execution", str(e))
            return result

