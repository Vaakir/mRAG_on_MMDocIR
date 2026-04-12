"""Pydantic models for structured LLM outputs."""

from pydantic import BaseModel, Field
from typing import Literal, List


class QueryRewriterDecision(BaseModel):
    """Structured output from query rewriter agent."""
    
    technique: Literal[
        "standard", "multi_query", "rag_fusion", "step_back",
        "hyde", "query_decomposition", "query_rewriting", "query_expansion"
    ] = Field(description="Which QueryTechnique to apply (one of 8 System 2 techniques)")
    
    reasoning: str = Field(description="Why this technique?")
    
    rewritten_queries: List[str] = Field(
        default_factory=list,
        description="List of rewritten/expanded query variants from the technique"
    )


class DocumentGrade(BaseModel):
    """Structured output from document grader."""
    
    relevant: str = Field(
        description="'yes' if documents are relevant, 'no' otherwise"
    )
    
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in decision (0.0-1.0)"
    )
    
    reasoning: str = Field(description="Why this grade?")
    
    num_relevant: int = Field(
        ge=0, le=20,
        description="How many of the retrieved docs are relevant?"
    )


class RetrieverDecision(BaseModel):
    """Structured output from retriever agent."""
    
    k: int = Field(ge=1, le=20, description="Number of documents to retrieve")
    
    method: Literal["hybrid", "dense", "bm25"] = Field(
        description="Retrieval method to use"
    )
    
    reasoning: str = Field(description="Why this method and k?")
    
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in decision"
    )


class GeneratorDecision(BaseModel):
    """Structured output from generator agent."""
    
    strategy: Literal["standard", "few_shot", "cot", "role", "ensemble"] = Field(
        description="Prompting strategy to use"
    )
    
    reasoning: str = Field(description="Why this strategy?")
    
    needs_more_context: bool = Field(
        default=False,
        description="Does generator need more documents?"
    )
    
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in answer quality"
    )
