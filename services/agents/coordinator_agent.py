"""Coordinator agent that orchestrates all other agents."""

from typing import Any, Dict, List, Optional
from services.agents.base_agent import BaseAgent, AgentResult, AgentState
from services.agents.data_collection_agent import DataCollectionAgent
from services.agents.news_agent import NewsAgent
from services.agents.market_mood_agent import MarketMoodAgent
from services.agents.analysis_agent import AnalysisAgent
from services.agents.recommendation_agent import RecommendationAgent
import streamlit as st


class CoordinatorAgent(BaseAgent):
    """Orchestrates the execution of all agents in the correct order."""
    
    def __init__(self):
        super().__init__(
            name="Coordinator Agent",
            description="Orchestrates all agents to complete the investment analysis workflow"
        )
        self.agents: List[BaseAgent] = []
        self.initialize_agents()
    
    def initialize_agents(self) -> None:
        """Initialize all sub-agents."""
        self.agents = [
            DataCollectionAgent(),
            MarketMoodAgent(),
            NewsAgent(),
            AnalysisAgent(),
            RecommendationAgent(),
        ]
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute all agents in sequence."""
        result = AgentResult(success=True, state=AgentState())
        
        # Merge initial context into result state
        result.state.data.update(context)
        
        agent_names = [agent.name for agent in self.agents]
        total_agents = len(agent_names)
        
        # Progress tracking
        progress_callback = context.get("progress_callback")
        step_callback = context.get("step_callback")
        
        # Execute each agent in sequence
        for idx, agent in enumerate(self.agents):
            try:
                if step_callback:
                    step_callback(f"Executing {agent.name}... ({idx + 1}/{total_agents})")
                
                # Prepare agent context from accumulated state
                agent_context = result.state.data.copy()
                agent_context.update(result.state.metadata)
                
                # Add specific callbacks for agents that need them
                if agent.name == "Data Collection Agent":
                    agent_context["progress_callback"] = context.get("fundamentals_progress_callback")
                elif agent.name == "News Collection Agent":
                    agent_context["news_progress_callback"] = context.get("news_progress_callback")
                
                # Execute agent
                agent_result = agent.execute(agent_context)
                
                # Merge agent results into coordinator result
                result.state.data.update(agent_result.state.data)
                result.state.metadata.update(agent_result.state.metadata)
                result.state.errors.update(agent_result.state.errors)
                
                # If agent failed critically, stop execution
                if not agent_result.success and agent_result.state.errors:
                    critical_errors = [
                        key for key in agent_result.state.errors.keys()
                        if key not in ["snapshots", "fii_error", "mood_error", "news_errors"]
                    ]
                    if critical_errors:
                        result.success = False
                        result.message = f"Agent {agent.name} failed with errors: {critical_errors}"
                        if step_callback:
                            step_callback(f"❌ {agent.name} failed")
                        break
                
                if step_callback:
                    step_callback(f"✅ {agent.name} completed")
                
                if progress_callback:
                    progress_callback((idx + 1) / total_agents)
                    
            except Exception as e:
                result.add_error(f"{agent.name}_exception", str(e))
                result.success = False
                if step_callback:
                    step_callback(f"❌ {agent.name} raised exception: {str(e)}")
                break
        
        if result.success:
            result.message = "All agents completed successfully"
        else:
            result.message = "Some agents failed during execution"
        
        return result
    
    def get_agent_by_name(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None
    
    def reset_all_agents(self) -> None:
        """Reset all sub-agents."""
        for agent in self.agents:
            agent.reset()
        self.reset()

