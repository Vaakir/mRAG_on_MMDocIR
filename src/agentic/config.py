"""Configuration for System 3 agentic RAG (optional - can use AdvancedConfig directly)."""

from dataclasses import dataclass, field
from typing import Dict, Any
from config.config import AdvancedConfig


@dataclass
class AgenticConfig(AdvancedConfig):
    """Configuration specific to agentic RAG (System 3)."""
    
    # ===== AGENT SETTINGS =====
    MAX_RETRIES: int = 2
    """Maximum number of retry iterations for query rewriting"""
    
    RETRY_ON_LOW_CONFIDENCE: bool = True
    """Whether to retry if grader confidence is below threshold"""
    
    GRADER_CONFIDENCE_THRESHOLD: float = 0.6
    """Minimum confidence threshold for document relevance"""
    
    AGENT_DECISION_LOGGING: bool = True
    """Whether to log all agent decisions for analysis"""
