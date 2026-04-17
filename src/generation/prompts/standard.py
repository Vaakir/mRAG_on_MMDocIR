"""
Standard prompting strategy - direct extraction without special prompting.
"""

from typing import List
from .base import PromptStrategy
import logging

logger = logging.getLogger(__name__)

STANDARD_SYSTEM_PROMPT = """You are a concise assistant. Answer using ONLY the provided context.

Strict Instructions (NON-NEGOTIABLE):
- Read the whole context and think before answering.
- NEVER add preamble, explanation, or context. ANSWER ONLY.
- DO NOT rephrase, explain, or add context.
- Multiple answers: ONLY output ['answer1', 'answer2', ...]. NOTHING ELSE.
- For yes/no questions, answer only "Yes" or "No".
- Do not rely only on explicit statements. If the answer can be derived from the context through calculation (e.g., growth rate, difference, ratio, count), compute it before answering.

Be direct. No padding. No explanations unless specifically asked."""


class StandardPromptStrategy(PromptStrategy):
    """
    Standard prompting strategy (direct approach) using direct extraction.
    
    Minimal prompting, strict adherence to context,
    focused on extracting answers directly without explanation.
    """
    
    def get_system_prompt(self) -> str:
        """Get the standard (direct) system prompt."""
        return STANDARD_SYSTEM_PROMPT
    
    def generate(self, question: str, context: str) -> str:
        """
        Generate an answer using standard (direct) approach.
        
        Args:
            question: User's question
            context: Retrieved context/documents
        
        Returns:
            Generated answer
        """
        system_prompt = self.get_system_prompt()
        return self.generator.generate(question, context, system_prompt=system_prompt)

    def generate_with_images(self, question: str, image_paths: List[str], text_context: str = "") -> str:
        """Generate with images using standard system prompt."""
        return self.generator.generate_with_images(
            question, image_paths, text_context,
            system_prompt=self.get_system_prompt(),
        )
