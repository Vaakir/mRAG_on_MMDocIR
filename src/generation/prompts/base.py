"""
Base class for prompting strategies.

Each prompting strategy defines how to construct and use prompts for answer generation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class PromptStrategy(ABC):
    """
    Abstract base class for prompting strategies.
    
    All strategies inherit from this and implement:
    - generate(): Main method to generate answers with the strategy's prompts
    - Potentially override system prompt construction, LLM call patterns, etc.
    """
    
    def __init__(self, generator, config: Dict[str, Any] = None):
        """
        Initialize a prompting strategy.
        
        Args:
            generator: BaselineGenerator instance with chat() and generate() methods
            config: Dictionary with strategy-specific configuration
                   Examples:
                   - role_type: type of expert role (e.g., 'financial_analyst')
                   - cot_steps: number of reasoning steps for CoT
                   - ensemble_count: number of outputs for ensemble
        """
        self.generator = generator
        self.config = config or {}
    
    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        """
        Generate an answer using this prompting strategy.
        
        Args:
            question: User's question
            context: Retrieved context/documents
        
        Returns:
            Generated answer string
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this strategy.

        Returns:
            System prompt string
        """
        pass

    def generate_with_images(self, question: str, image_paths: List[str], text_context: str = "") -> str:
        """
        Generate an answer using this strategy with image support.

        Default implementation calls the generator's VLM with this strategy's
        system prompt. Subclasses can override for strategy-specific behaviour
        (e.g. CoT extracts the answer section after generation).

        Args:
            question: User's question
            image_paths: List of absolute paths to images
            text_context: Text context from retrieved chunks

        Returns:
            Generated answer string
        """
        return self.generator.generate_with_images(
            question, image_paths, text_context,
            system_prompt=self.get_system_prompt(),
        )
