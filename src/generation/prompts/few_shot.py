"""
Few-shot and one-shot prompting strategies.

These strategies demonstrate the difference between:
- One-shot: Provide 1 example, then ask
- Few-shot: Provide multiple examples (2-5), then ask
"""

from .base import PromptStrategy
import logging

logger = logging.getLogger(__name__)

# Default examples for few-shot prompting
DEFAULT_FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris"
    },
    {
        "question": "How many continents are there?",
        "answer": "7"
    },
    {
        "question": "What year did World War II end?",
        "answer": "1945"
    },
    {
        "question": "How many times is \"resilience\" mentioned in the entire document?",
        "answer": "12"
    },
    {
        "question": "What strategic approaches are outlined for implementation?",
        "answer": ["stakeholder engagement", "phased rollout", "risk assessment", "team training"]
    }
]


class FewShotPromptStrategy(PromptStrategy):
    """
    Few-shot prompting - provide examples (1-N), then ask the question.
    
    Handles any number of examples:
    - num_examples=1: One-shot (1 example)
    - num_examples=3: Few-shot (3 examples)
    - num_examples=N: N-shot (N examples)
    
    
    Configuration options:
    - examples: List of example dicts with 'question' and 'answer' keys
               Default: 3 general examples
    - num_examples: How many examples to use (1 to infinity)
                   Default: all provided examples
    """
    
    def __init__(self, generator, config=None):
        super().__init__(generator, config)
        self.examples = self.config.get('examples', DEFAULT_FEW_SHOT_EXAMPLES)
        self.num_examples = self.config.get('num_examples', len(self.examples))
        self.examples = self.examples[:self.num_examples]
    
    def get_system_prompt(self) -> str:
        """Get the few-shot system prompt with examples."""

        examples_text = ""
        for i, example in enumerate(self.examples, 1):
            examples_text += f"""
Example {i}:
Question: {example['question']}
Answer: {example['answer']}"""
        
        return f"""You are a helpful assistant that answers questions based on the provided context.

Here are examples of how to answer:
{examples_text}

Now, answer the given question using ONLY the provided context.

Instructions:
- Answer ONLY using the provided context
- Follow the same format as the examples (concise and direct)
- If the answer is not in the context, say: "I cannot find the answer in the provided context"
- For yes/no questions: answer only "Yes" or "No"
- For facts: give only the fact, no explanation
- For lists: format as numbered list"""
    
    def generate(self, question: str, context: str) -> str:
        """
        Generate an answer using few-shot prompting.
        
        Args:
            question: User's question
            context: Retrieved context/documents
        
        Returns:
            Generated answer
        """
        system_prompt = self.get_system_prompt()
        logger.debug(f"Generating with few-shot prompting ({len(self.examples)} examples)")
        return self.generator.generate(question, context, system_prompt=system_prompt)
