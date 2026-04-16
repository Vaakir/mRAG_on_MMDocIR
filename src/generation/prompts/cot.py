"""
Chain-of-Thought (CoT) prompting strategy - explicit step-by-step reasoning.

The LLM is instructed to think step-by-step before providing the final answer,
making reasoning explicit and improving accuracy for complex questions.
"""

from typing import List
from .base import PromptStrategy
import logging

logger = logging.getLogger(__name__)

# Chain-of-Thought system prompt
COT_SYSTEM_PROMPT = """You are a careful, methodical assistant that thinks through problems step-by-step.

Your task: Answer the question based ONLY on the provided context.

Think Before You Answer (MANDATORY):
1. Read the entire context carefully
2. Identify ALL relevant information for answering the question
3. List the key facts and evidence you found
4. Reason through how these facts connect to the question
5. Verify your answer against the context
6. Only then provide your final answer

Answer Format:
<REASONING>
[Your step-by-step thinking here]
[List relevant facts found]
[Explain your logical reasoning]
[Note any gaps or uncertainties]
</REASONING>

<ANSWER>
[Your final, concise answer here - ONLY based on context]
</ANSWER>

Strict Rules:
- ONLY use information from the provided context
- NEVER add information not in the context
- Be explicit about your reasoning so errors can be caught
- If the answer cannot be found in context, state in your answer: "I cannot find the answer in the provided context"
- For yes/no questions: answer only "Yes" or "No" (nothing else)
- For factual answers: give only the fact/value, no preamble (e.g., "538" not "The answer is 538")
- For list answers: format as numbered list, nothing else
- For calculations: give only the final result in the ANSWER section
- NO complete sentences in the answer section, keep it minimalist"""


class ChainOfThoughtPromptStrategy(PromptStrategy):
    """
    Chain-of-Thought (CoT) prompting strategy for explicit step-by-step reasoning.
    
    Configuration options:
    - show_reasoning: Whether to include REASONING section in output (True by default)
                     If False, only <ANSWER> section is returned
    """
    
    def __init__(self, generator, config=None):
        super().__init__(generator, config)
        self.show_reasoning = self.config.get('show_reasoning', False)
    
    def get_system_prompt(self) -> str:
        """Get the Chain-of-Thought system prompt."""
        return COT_SYSTEM_PROMPT
    
    def generate(self, question: str, context: str) -> str:
        """
        Generate an answer using Chain-of-Thought reasoning.
        
        The LLM will explicitly show its reasoning before providing the answer.
        
        Args:
            question: User's question
            context: Retrieved context/documents
        
        Returns:
            Generated answer with reasoning (can be filtered to extract just answer)
        """
        system_prompt = self.get_system_prompt()
        logger.debug("Generating with Chain-of-Thought reasoning")
        
        # think=False: the <REASONING> block IS the reasoning, no need for model's internal thinking too
        full_response = self.generator.generate(question, context, system_prompt=system_prompt, think=False)
        
        # If show_reasoning is False, extract only the answer section
        if not self.show_reasoning:
            return self._extract_answer_only(full_response)
        
        return full_response
    
    def generate_with_images(self, question: str, image_paths: List[str], text_context: str = "") -> str:
        """Generate with images, then extract answer if show_reasoning is False."""
        full_response = self.generator.generate_with_images(
            question, image_paths, text_context,
            system_prompt=self.get_system_prompt(), think=False,
        )
        if not self.show_reasoning:
            return self._extract_answer_only(full_response)
        return full_response

    def _extract_answer_only(self, response: str) -> str:
        """
        Extract only the answer from the full CoT response.
        
        Tries multiple strategies:
        1. Look for <ANSWER> tags (with or without closing tag)
        2. Look for "ANSWER:" keyword
        3. Return the last non-empty paragraph (usually the final answer)
        
        Args:
            response: Full response with REASONING and ANSWER sections
        
        Returns:
            Just the answer content
        """
        try:
            # Strategy: Extract content between <ANSWER> and </ANSWER> tags
            if '<ANSWER>' in response and '</ANSWER>' in response:
                start = response.find('<ANSWER>') + len('<ANSWER>')
                end = response.find('</ANSWER>')
                answer = response[start:end].strip()
                if answer:
                    logger.debug("Extracted answer from <ANSWER>...</ANSWER> tags")
                    return answer
            
            # Strategy: Extract from <ANSWER> to end of string (tag not closed)
            if '<ANSWER>' in response:
                start = response.find('<ANSWER>') + len('<ANSWER>')
                answer = response[start:].strip()
                if answer:
                    logger.debug("Extracted answer from <ANSWER> tag (no closing tag)")
                    return answer
            
            # Strategy: Look for "ANSWER:" or similar keywords
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if any(keyword in line.upper() for keyword in ['ANSWER:', 'FINAL ANSWER:', 'ANSWER IS:']):
                    # Extract everything after this line
                    remaining = '\n'.join(lines[i+1:]).strip()
                    if remaining:
                        logger.debug("Extracted answer using 'ANSWER:' keyword")
                        return remaining
            
            # Strategy: Return the last substantial paragraph
            paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
            if paragraphs:
                last_para = paragraphs[-1]
                # Don't return if it looks like it's still reasoning (contains reasoning keywords)
                reasoning_keywords = ['reasoning', 'think', 'consider', 'analyzing', 'looking at', 'document', 'mention']
                keyword_count = sum(1 for keyword in reasoning_keywords if keyword in last_para.lower())
                if keyword_count < len(reasoning_keywords) * 0.3:
                    logger.debug("Extracted answer as last paragraph")
                    return last_para
            
            # Fallback: return the full response
            logger.warning("Could not extract answer clearly, returning full response")
            return response
            
        except Exception as e:
            logger.warning(f"Error extracting answer section: {e}")
            return response
