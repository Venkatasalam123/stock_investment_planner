"""Base agent class for all agents in the system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class AgentState:
    """State container for agent execution."""
    data: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result container from agent execution."""
    success: bool
    state: AgentState
    message: str = ""
    
    def add_error(self, key: str, error: str) -> None:
        """Add an error to the result."""
        self.state.errors[key] = error
        self.success = False
    
    def add_data(self, key: str, value: Any) -> None:
        """Add data to the result."""
        self.state.data[key] = value
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """Get data from the result."""
        return self.state.data.get(key, default)


class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.state = AgentState()
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        Execute the agent's main task.
        
        Args:
            context: Shared context from previous agents
            
        Returns:
            AgentResult with success status and state
        """
        pass
    
    def reset(self) -> None:
        """Reset agent state."""
        self.state = AgentState()
    
    def validate_context(self, context: Dict[str, Any], required_keys: list[str]) -> bool:
        """Validate that context contains required keys."""
        missing = [key for key in required_keys if key not in context]
        if missing:
            raise ValueError(f"Missing required context keys: {missing}")
        return True

