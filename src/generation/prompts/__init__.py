"""
Factory function and utilities for creating prompting strategies.
"""

from typing import Dict, Any
from .base import PromptStrategy
from .standard import StandardPromptStrategy
from .role import RolePromptStrategy
from .cot import ChainOfThoughtPromptStrategy
from .ensemble import EnsemblePromptStrategy
from .few_shot import FewShotPromptStrategy
import logging

logger = logging.getLogger(__name__)


def get_prompt_strategy(
    strategy_name: str,
    generator,
    config: Dict[str, Any] = None
) -> PromptStrategy:
    """
    Factory function to get the appropriate prompting strategy based on name.
    
    Args:
        strategy_name: Name of the strategy
                      ('standard', 'few_shot', 'role', 'cot', 'ensemble')
        generator: BaselineGenerator instance
        config: Configuration dict for the strategy
    
    Returns:
        PromptStrategy subclass instance
    
    Raises:
        ValueError: If strategy name is not recognized
    
    Examples:
        >>> strategy = get_prompt_strategy('cot', generator)
        >>> answer = strategy.generate(question, context)
        
        >>> config = {'role_type': 'financial_analyst'}
        >>> strategy = get_prompt_strategy('role', generator, config)
        
        >>> config = {'examples': [{'question': 'Q1?', 'answer': 'A1'}, ...]}
        >>> strategy = get_prompt_strategy('few_shot', generator, config)
    """
    strategies = {
        'standard': StandardPromptStrategy,
        'few_shot': FewShotPromptStrategy,
        'role': RolePromptStrategy,
        'cot': ChainOfThoughtPromptStrategy,
        'ensemble': EnsemblePromptStrategy,
    }
    
    strategy_name = strategy_name.lower().strip()
    
    if strategy_name not in strategies:
        available = ', '.join(strategies.keys())
        raise ValueError(
            f"Unknown prompting strategy '{strategy_name}'. "
            f"Available strategies: {available}"
        )
    
    StrategyClass = strategies[strategy_name]
    return StrategyClass(generator, config)


__all__ = [
    'PromptStrategy',
    'StandardPromptStrategy',
    'FewShotPromptStrategy',
    'RolePromptStrategy',
    'ChainOfThoughtPromptStrategy',
    'EnsemblePromptStrategy',
    'get_prompt_strategy',
]
