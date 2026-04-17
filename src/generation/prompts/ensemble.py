"""
Ensemble prompting strategy - generate multiple answers and aggregate results.

Features:
1. Multiple prompts: Generate with different roles/strategies, aggregate intelligently
2. Self-consistency: Generate multiple CoT outputs, aggregate intelligently
3. Flexible aggregation: judge (select best), combine (synthesize), embedding_similarity
4. Configurable strategies: Pick which strategies to use and how many
5. Temperature control: Per-strategy temperature settings
"""

from .base import PromptStrategy
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class EnsemblePromptStrategy(PromptStrategy):
    """
    Advanced ensemble prompting strategy with configurable behavior.
    
    Supports multiple modes and aggregation strategies:
    1. multi_prompt: Use different roles/strategies, aggregate using judge/combine/embedding_similarity
    2. self_consistency: Use same strategy multiple times with variance, aggregate
    
    Configuration (via config dict):
    - mode: 'multi_prompt' or 'self_consistency' (default: 'multi_prompt')
    - ensemble_size: Number of different outputs to generate (default: 4)
    - strategies: List of strategies to use (default: ['cot', 'few_shot', 'financial_analyst_role', 'standard'])
    - temperatures: Dict mapping strategy names to temperatures
    - aggregation_method: 'judge', 'combine', 'embedding_similarity'
    - embedder: Optional TextEmbedder instance for semantic similarity (used with embedding_similarity method)
    - include_strategy_metadata: Whether to include which strategy generated answer
    - verbose_logging: Enable detailed logging
    
    Users can configure via config.py PROMPTING_STRATEGY_CONFIG:
    ```python
    # In config.py
    PROMPTING_STRATEGY: str = "ensemble"
    PROMPTING_STRATEGY_CONFIG: Dict[str, Any] = {
        'mode': 'multi_prompt',
        'ensemble_size': 3,
        'aggregation_method': 'combine',
        'strategies': ['cot', 'standard', 'few_shot'],
        'temperatures': {'cot': 0.6, 'standard': 0.5, 'few_shot': 0.5}
    }
    ```
    """
    
    def __init__(self, generator, config=None):
        super().__init__(generator, config)
        
        # Use provided config or apply defaults
        if not self.config:
            self.config = {}
        
        # Parse configuration
        self.mode = self.config.get('mode', 'multi_prompt')
        self.ensemble_size = self.config.get('ensemble_size', 4)
        self.temperature = self.config.get('temperature', 0.7)
        self.strategies = self.config.get('strategies', [
            'cot',
            'few_shot',
            'financial_analyst_role',
            'standard'
        ])
        self.temperatures = self.config.get('temperatures', {})
        self.aggregation_method = self.config.get('aggregation_method', 'combine')
        self.include_strategy_metadata = self.config.get('include_strategy_metadata', False)
        self.verbose_logging = self.config.get('verbose_logging', False)
        self.embedder = self.config.get('embedder', None)
        
        if self.mode not in ['multi_prompt', 'self_consistency']:
            logger.warning(f"Unknown ensemble mode '{self.mode}', using 'multi_prompt'")
            self.mode = 'multi_prompt'
        
        if self.aggregation_method not in ['judge', 'combine', 'embedding_similarity']:
            logger.warning(f"Unknown aggregation method '{self.aggregation_method}', using 'combine'")
            self.aggregation_method = 'combine'
    
    def get_system_prompt(self) -> str:
        """Ensemble doesn't use a single system prompt (uses sub-strategies)."""
        return "Ensemble strategy - uses multiple sub-strategies"
    
    def generate(self, question: str, context: str) -> str:
        """
        Generate answers using ensemble approach and aggregate them.

        Args:
            question: User's question
            context: Retrieved context/documents

        Returns:
            Aggregated answer from multiple prompting approaches
        """
        log_msg = f"Starting ensemble generation (mode={self.mode}, ensemble_size={self.ensemble_size}, aggregation={self.aggregation_method})"
        if self.verbose_logging:
            logger.info(log_msg)

        if self.mode == 'multi_prompt':
            return self._generate_multi_prompt(question, context)
        else:  # self_consistency
            return self._generate_self_consistency(question, context)

    def generate_with_images(self, question: str, image_paths: List[str], text_context: str = "") -> str:
        """Generate with images using ensemble of sub-strategies."""
        log_msg = f"Starting ensemble VLM generation (mode={self.mode}, ensemble_size={self.ensemble_size})"
        if self.verbose_logging:
            logger.info(log_msg)

        if self.mode == 'multi_prompt':
            return self._generate_multi_prompt_with_images(question, image_paths, text_context)
        else:
            return self._generate_self_consistency_with_images(question, image_paths, text_context)
    
    def _generate_multi_prompt(self, question: str, context: str) -> str:
        """
        Generate answers using multiple different prompting strategies.
        
        Uses different roles/approaches and aggregates intelligently.
        """
        all_answers = []
        strategy_map = self._get_strategy_map()
        
        strategies_to_use = self.strategies[:self.ensemble_size]
        
        if self.verbose_logging:
            logger.info(f"Using strategies: {strategies_to_use}")
        
        for i, strategy_name in enumerate(strategies_to_use, 1):
            try:
                temp = self.temperatures.get(strategy_name, 0.5)
                
                if self.verbose_logging:
                    logger.debug(f"Ensemble: Generating [{i}/{len(strategies_to_use)}] with {strategy_name} (temp={temp})")
                
                if strategy_name not in strategy_map:
                    logger.warning(f"Unknown strategy '{strategy_name}', skipping")
                    continue
                
                strategy = strategy_map[strategy_name](temp)
                answer = strategy.generate(question, context)
                
                all_answers.append({
                    'strategy': strategy_name,
                    'answer': answer,
                    'temperature': temp
                })
            except Exception as e:
                logger.error(f"Error in ensemble strategy '{strategy_name}': {e}")
                continue
        
        if not all_answers:
            logger.warning("No answers generated in ensemble - returning empty")
            return "Error: Could not generate answers with any strategy"
        
        return self._aggregate_answers(all_answers, question, context)
    
    def _generate_self_consistency(self, question: str, context: str) -> str:
        """
        Generate multiple outputs from the same strategy with temperature variance.
        
        Useful for sampling different reasoning paths using CoT.
        """
        from .cot import ChainOfThoughtPromptStrategy
        
        all_answers = []
        
        if self.verbose_logging:
            logger.info(f"Self-consistency: Generating {self.ensemble_size} samples with temperature={self.temperature}")
        
        for i in range(self.ensemble_size):
            try:
                strategy = ChainOfThoughtPromptStrategy(
                    self.generator,
                    {'show_reasoning': False, 'temperature': self.temperature}
                )
                answer = strategy.generate(question, context)
                all_answers.append({
                    'sample': i + 1,
                    'answer': answer,
                })
                
                if self.verbose_logging:
                    logger.debug(f"Self-consistency: Generated sample {i+1}/{self.ensemble_size}")
            except Exception as e:
                logger.error(f"Error generating sample {i+1}: {e}")
                continue
        
        if not all_answers:
            logger.warning("No answers generated in self-consistency")
            return "Error: Could not generate answers"
        
        return self._aggregate_answers(all_answers, question, context)
    
    def _generate_multi_prompt_with_images(self, question: str, image_paths: List[str], text_context: str) -> str:
        """Multi-prompt ensemble with VLM: each sub-strategy calls generate_with_images."""
        all_answers = []
        strategy_map = self._get_strategy_map()
        strategies_to_use = self.strategies[:self.ensemble_size]

        for i, strategy_name in enumerate(strategies_to_use, 1):
            try:
                temp = self.temperatures.get(strategy_name, 0.5)
                if strategy_name not in strategy_map:
                    continue
                strategy = strategy_map[strategy_name](temp)
                answer = strategy.generate_with_images(question, image_paths, text_context)
                all_answers.append({'strategy': strategy_name, 'answer': answer, 'temperature': temp})
            except Exception as e:
                logger.error(f"Error in ensemble VLM strategy '{strategy_name}': {e}")

        if not all_answers:
            return "Error: Could not generate answers with any strategy"
        return self._aggregate_answers(all_answers, question, text_context)

    def _generate_self_consistency_with_images(self, question: str, image_paths: List[str], text_context: str) -> str:
        """Self-consistency ensemble with VLM: same CoT strategy, multiple samples."""
        from .cot import ChainOfThoughtPromptStrategy
        all_answers = []

        for i in range(self.ensemble_size):
            try:
                strategy = ChainOfThoughtPromptStrategy(
                    self.generator, {'show_reasoning': False, 'temperature': self.temperature}
                )
                answer = strategy.generate_with_images(question, image_paths, text_context)
                all_answers.append({'sample': i + 1, 'answer': answer})
            except Exception as e:
                logger.error(f"Error generating VLM sample {i+1}: {e}")

        if not all_answers:
            return "Error: Could not generate answers"
        return self._aggregate_answers(all_answers, question, text_context)

    def _get_strategy_map(self) -> Dict[str, callable]:
        """
        Get mapping of strategy names to factory functions using Strategy Pattern.

        
        Returns:
            Dict mapping strategy names to factory functions (temp -> strategy instance)
        """
        from .standard import StandardPromptStrategy
        from .role import RolePromptStrategy
        from .cot import ChainOfThoughtPromptStrategy
        from .few_shot import FewShotPromptStrategy
        
        # Strategy registry
        registry = {
            'standard': (StandardPromptStrategy, {}),
            'one_shot': (FewShotPromptStrategy, {'num_examples': 1}),
            'few_shot': (FewShotPromptStrategy, {'num_examples': 3}),
            'cot': (ChainOfThoughtPromptStrategy, {'show_reasoning': False}),
            'financial_analyst_role': (RolePromptStrategy, {'role_type': 'financial_analyst'}),
            'researcher_role': (RolePromptStrategy, {'role_type': 'researcher'}),
            'data_analyst_role': (RolePromptStrategy, {'role_type': 'data_analyst'}),
        }
        
        def create_factory(strategy_class, base_config):
            def factory(temperature):
                config = {**base_config, 'temperature': temperature}
                return strategy_class(self.generator, config)
            return factory
        
        return {
            name: create_factory(cls, cfg) 
            for name, (cls, cfg) in registry.items()
        }
    
    def _aggregate_answers(self, answers: List[Dict[str, Any]], 
                          question: str, context: str) -> str:
        """
        Aggregate multiple answers using configured aggregation method.
        
        Args:
            answers: List of answer dicts with 'answer' key
            question: Original question
            context: Original context
        
        Returns:
            Aggregated answer string
        """
        aggregators = {
            'judge': lambda: self._judge_best(answers, question, context),
            'combine': lambda: self._combine_answers(answers, question, context),
            'embedding_similarity': lambda: self._embedding_similarity(answers, question, context),
        }
        
        if self.aggregation_method not in aggregators:
            logger.warning(f"Unknown aggregation method {self.aggregation_method}")
            return answers[0]['answer'] if answers else ""
        
        return aggregators[self.aggregation_method]()
    

    def _judge_best(self, answers: List[Dict[str, Any]], 
                   question: str, context: str) -> str:
        """
        Use LLM as judge to select the best answer from alternatives.
        
        The judge evaluates all answers and selects the one that best
        addresses the question with the provided context.
        """
        if len(answers) == 1:
            return answers[0]['answer']
        
        judge_prompt = self._build_judge_prompt(answers, question, context)
        
        if self.verbose_logging:
            logger.info(f"Ensemble: Using judge to select best from {len(answers)} answers")
        
        try:
            judgment = self.generator.generate(
                question,
                context,
                system_prompt=judge_prompt
            )
            
            if self.verbose_logging:
                logger.debug(f"Judge output: {judgment[:200]}...")
            
            selected_idx = self._extract_selected_index(judgment, len(answers))
            
            if selected_idx is not None:
                selected_answer = answers[selected_idx]['answer']
                if self.include_strategy_metadata:
                    strategy = answers[selected_idx].get('strategy', 'unknown')
                    return f"[Selected via judge - Strategy: {strategy}]\n\n{selected_answer}"
                return selected_answer
            return judgment
                
        except Exception as e:
            logger.error(f"Error in judge aggregation: {e}")
            return answers[0]['answer'] if answers else "Unable to judge answers"
    
    def _combine_answers(self, answers: List[Dict[str, Any]], 
                        question: str, context: str) -> str:
        """
        Use LLM to synthesize/combine all answers into one cohesive response.
        
        The synthesizer reads all answers and creates a unified answer that
        incorporates insights from all strategies.
        """
        if len(answers) == 1:
            return answers[0]['answer']
        
        synthesis_prompt = self._build_synthesis_prompt(answers, question, context)
        
        if self.verbose_logging:
            logger.info(f"Ensemble: Synthesizing {len(answers)} answers into one")
        
        try:
            synthesized = self.generator.generate(
                question,
                context,
                system_prompt=synthesis_prompt
            )
            
            if self.verbose_logging:
                logger.debug(f"Synthesized answer generated ({len(synthesized)} chars)")
            
            return synthesized
            
        except Exception as e:
            logger.error(f"Error in synthesis aggregation: {e}")
            return answers[0]['answer'] if answers else "Unable to synthesize answers"
    
    def _embedding_similarity(self, answers: List[Dict[str, Any]], 
                             question: str, context: str) -> str:
        """
        Select answer with highest semantic similarity to question and context.
        
        Scores each answer based on:
        - Similarity to question embedding (60% weight)
        - Similarity to context embedding (40% weight)
        """
        if not self.embedder:
            logger.warning("Embedding similarity selected but no embedder provided, returning first answer")
            return answers[0]['answer']
        
        if len(answers) == 1:
            return answers[0]['answer']
        
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Embed question and context
            question_emb = self.embedder.embed_query(question).reshape(1, -1)
            context_emb = self.embedder.embed_query(context).reshape(1, -1)
            
            # Embed all answers
            answer_texts = [a['answer'] for a in answers]
            answer_embs = self.embedder.embed_texts(answer_texts)
            
            # Calculate similarities
            question_sims = cosine_similarity(question_emb, answer_embs)[0]
            context_sims = cosine_similarity(context_emb, answer_embs)[0]
            
            # Weighted score: 60% question align, 40% context grounding
            scores = 0.6 * question_sims + 0.4 * context_sims
            best_idx = np.argmax(scores)
            
            if self.verbose_logging:
                best_score = scores[best_idx]
                logger.info(f"Ensemble: Selected answer via embedding similarity (score: {best_score:.3f})")
            
            if self.include_strategy_metadata:
                strategy = answers[best_idx].get('strategy', 'unknown')
                return f"[Selected via embedding similarity - Strategy: {strategy}]\n\n{answers[best_idx]['answer']}"
            
            return answers[best_idx]['answer']
            
        except Exception as e:
            logger.error(f"Error in embedding similarity: {e}")
            return answers[0]['answer'] if answers else "Unable to score answers"
    

    def _format_answers_for_prompt(self, answers: List[Dict[str, Any]], 
                                   bracket_format: str = "Answer {i}") -> str:
        """
        Format answers as a numbered list for prompts.
        
        Args:
            answers: List of answer dicts with 'answer' and optional 'strategy' keys
            bracket_format: Format string for numbering (default: "Answer {i}")
            
        Returns:
            Formatted string with numbered answers
        """
        lines = []
        for i, a in enumerate(answers, 1):
            strategy = a.get('strategy', f'Strategy {i}')
            bracket = bracket_format.format(i=i) if '{i}' in bracket_format else f"Answer {i}"
            lines.extend([
                f"\n{bracket}: {strategy}",
                a['answer'].strip()
            ])
        return "\n".join(lines)
    
    def _build_judge_prompt(self, answers: List[Dict[str, Any]], 
                           question: str, context: str) -> str:
        """Build a prompt for the judge to evaluate and select the best answer."""
        formatted_answers = self._format_answers_for_prompt(answers, bracket_format="[Answer {i}]")
        
        return (
            f"You are an expert evaluator. Select the SINGLE BEST answer from the options below.\n\n"
            f"Question: {question}\n\n"
            f"Context: {context}\n\n"
            f"Answer Options:{formatted_answers}\n\n"
            f"Output ONLY the answer number (e.g., '1' or '[Answer 1]'). No explanation needed."
        )
    
    def _build_synthesis_prompt(self, answers: List[Dict[str, Any]], 
                               question: str, context: str) -> str:
        """Build a prompt for LLM to synthesize all answers into one concise response."""
        formatted_answers = self._format_answers_for_prompt(answers)
        
        return (
            f"You are an expert synthesizer. Combine the best insights from all the answers below "
            f"into a single, direct answer to the question.\n\n"
            f"Question: {question}\n\n"
            f"Context: {context}\n\n"
            f"Answers from different strategies:{formatted_answers}\n\n"
            f"Requirements:\n"
            f"- Synthesize insights from all answers\n"
            f"- Keep it concise and direct\n"
            f"- Answer the question only, no fluff like 'The answer is...'\n"
            f"- No repetition\n\n"
            f"Synthesized Answer:"
        )
    
    def _extract_selected_index(self, judge_text: str, num_answers: int) -> Optional[int]:
        """
        Extract which answer the judge selected from their output.
        
        Args:
            judge_text: The judge's output text
            num_answers: Total number of answers to choose from
            
        Returns:
            Index (0-based) of selected answer, or None if unclear
        """
        import re
        
        judge_lower = judge_text.lower().strip()
        
        # Look for patterns like "answer 1", "[1]", "option 1", etc.
        patterns = [
            r'answer\s+(\d+)',
            r'\[(\d+)\]',
            r'option\s+(\d+)',
            r'#(\d+)',
            r'the\s+(?:best|correct|appropriate)\s+(?:answer|option)?\s+is\s+(\d+)',
            r'^(\d+)$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, judge_lower)
            if match:
                idx = int(match.group(1))
                # Convert to 0-based index
                if 1 <= idx <= num_answers:
                    return idx - 1
        
        return None
    


