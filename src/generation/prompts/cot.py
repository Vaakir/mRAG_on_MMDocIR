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

CRITICAL INSTRUCTIONS - READ CAREFULLY:
=====================================

1. Think Before You Answer (MANDATORY):
   - Read the entire context carefully
   - Identify ALL relevant information for answering the question
   - List the key facts and evidence you found
   - Reason through how these facts connect to the question
   - Verify your answer against the context
   - ONLY then provide your final answer

2. REASONING SECTION RULES (MOST IMPORTANT):
   - KEEP EACH STEP SHORT: Maximum 2 sentences per reasoning step
   - If explaining something takes 3+ sentences, you're over-explaining. Stop.
   - DO NOT REPEAT SENTENCES: If you write the same sentence twice, STOP IMMEDIATELY and move to your answer
   - DO NOT REPEAT WORDS: If the same word appears more than 3 times in a section, you're stuck. BREAK THE LOOP.
   - Each step must ADD NEW information. No redundancy allowed.

3. EARLY STOPPING RULE (CRITICAL):
   - After checking the ENTIRE provided context, if you cannot find the answer:
     STOP REASONING IMMEDIATELY
     Write in your ANSWER section: "CANNOT_FIND_ANSWER"
   - DO NOT speculate, guess, or search for answers outside the context
   - DO NOT write 100+ lines trying to find something that isn't there
   - Trust your reasoning: if you've checked thoroughly and it's not there, IT'S NOT THERE

Answer Format:
==============

<REASONING>
[Short 1-2 sentence steps ONLY]
[STOP as soon as you find the answer - don't keep explaining]
</REASONING>

<ANSWER>
[Your final answer here - based ONLY on context]
</ANSWER>

Strict Rules - MANDATORY:
========================
- ONLY use information from the provided context
- NEVER add information not in the context, NEVER hallucinate
- Be explicit about your reasoning so errors can be caught
- For yes/no questions: answer only "Yes" or "No" (nothing else)
- For factual answers: give only the fact/value, no preamble (e.g., "538" not "The answer is 538")
- For list answers: format as JSON list, nothing else: ["item1", "item2"]
- For calculations: give only the final result in the ANSWER section
- NO complete sentences in the answer section, keep it minimalist
- If you write more than 10 short reasoning steps and haven't found the answer, write "CANNOT_FIND_ANSWER"
"""

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
