"""Base agent class for agentic RAG."""

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the agentic RAG system.
    
    Each agent is responsible for a specific decision-making task:
    - Query rewriter: decides which query technique to use
    - Grader: evaluates document relevance
    - Generator: selects prompting strategy
    """
    
    def __init__(self, llm, config=None):
        """
        Initialize agent.
        
        Args:
            llm: LangChain LLM instance for decision-making
            config: Configuration dictionary for the agent
        """
        self.llm = llm
        self.config = config or {}
    
    @abstractmethod
    async def execute(self, state):
        """
        Execute the agent on the given state.
        
        Args:
            state: Current agent state (AgenticRAGState)
            
        Returns:
            Updated state dictionary
        """
        pass
