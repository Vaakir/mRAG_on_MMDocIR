"""Node implementations for the agentic graph."""

import logging
import json
from typing import Literal, Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage
from .state import AgenticRAGState
from ..tools.output_parser import (
    QueryRewriterDecision,
    DocumentGrade,
    RetrieverDecision,
    GeneratorDecision
)

logger = logging.getLogger(__name__)


# ============================================================================
# QUERY REWRITER NODE
# Decides which QueryTechnique to use based on the question
# ============================================================================

def make_query_rewriter_node(llm, query_techniques_dict, config: Dict[str, Any]):
    """
    Factory function to create the query rewriter node.
    
    The query rewriter agent decides which of the 8 QueryTechnique classes
    to use based on the user's question.
    
    Args:
        llm: LangChain LLM instance (for decision-making)
        query_techniques_dict: Dict mapping technique names to QueryTechnique instances
        config: Configuration dict
        
    Returns:
        Node function for LangGraph
    """
    
    def query_rewriter_node(state: AgenticRAGState):
        """
        Query Rewriter Agent: Decides which technique to use for rewriting/retrieving the query.
        
        Available techniques:
        - standard: No modification (baseline)
        - multi_query: Generate multiple paraphrases and retrieve for each
        - rag_fusion: Paraphrases with RRF fusion
        - step_back: Abstract to broader concept, then retrieve
        - hyde: Generate hypothetical documents matching the query
        - query_decomposition: Break into sub-questions
        - query_rewriting: Improve grammar and clarity
        - query_expansion: Add synonyms and related terms
        """
        
        # Handle both AgenticRAGState and dict
        if isinstance(state, dict):
            question = state.get('original_question', '')
            retry_count = state.get('retry_count', 0)
            last_technique = state.get('last_technique_used', '')
        else:
            question = state.original_question
            retry_count = state.retry_count
            last_technique = state.last_technique_used
        
        logger.info(f"\n{'='*80}")
        logger.info(f"QUERY REWRITER AGENT (Attempt {retry_count + 1})")
        logger.info(f"{'='*80}")
        logger.info(f"Question: {question}")
        if last_technique:
            logger.info(f"Last technique (to avoid on retry): {last_technique}")
        
        # Build prompt for LLM to decide which technique to use
        available_techniques = list(query_techniques_dict.keys())
        
        prompt = f"""You are a query planning agent. Your job is to decide which query technique
to use for improved retrieval.

Available techniques:
- standard: No modification (baseline)
- multi_query: Generate multiple paraphrases and retrieve for each (use for ambiguous or multi-faceted questions)
- rag_fusion: Paraphrases with RRF fusion (good for complex questions)
- step_back: Abstract to broader concept, then retrieve (good for specific queries needing context)
- hyde: Generate hypothetical documents matching the query (good for semantic searches)
- query_decomposition: Break into sub-questions (good for multi-part questions)
- query_rewriting: Improve grammar and clarity (good for poorly-phrased questions)
- query_expansion: Add synonyms and related terms (good for narrow searches)

Question: {question}

Retry attempt: {retry_count}
Last technique used: {last_technique if last_technique else "None"}

On retry, prefer a DIFFERENT technique than last time (e.g., if last was multi_query, try step_back or query_decomposition).

Decide which technique to use and explain why.

Respond with ONLY valid JSON (no markdown code block, no extra text):
{{
    "technique": "one of the 8 listed above",
    "reasoning": "why you chose this technique"
}}"""
        
        # Call LLM to get decision
        try:
            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse as JSON
            try:
                decision_dict = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code block
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                    decision_dict = json.loads(json_str)
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                    decision_dict = json.loads(json_str)
                else:
                    raise ValueError(f"Cannot parse LLM response as JSON: {response_text}")
            
            decision = QueryRewriterDecision(**decision_dict)
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM decision: {e}, falling back to 'standard'")
            decision = QueryRewriterDecision(
                technique="standard",
                reasoning="Error in parsing, using baseline",
                rewritten_queries=[question]
            )
        
        logger.info(f"Chose technique: {decision.technique}")
        logger.info(f"Reasoning: {decision.reasoning}")
        
        # Apply the chosen query technique
        technique = query_techniques_dict.get(decision.technique)
        if not technique:
            logger.warning(f"Technique {decision.technique} not found, using 'standard'")
            technique = query_techniques_dict['standard']
        
        # Retrieve using the technique
        retrieved_docs = technique.retrieve(question, top_k=config.get('TOP_K', 5))
        
        # Format retrieved text
        retrieved_text = "\n\n".join([
            f"[Document {i+1}]:\n{doc.get('text', '')}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # Get existing agent_decisions safely (handle both dict and AgenticRAGState)
        if isinstance(state, dict):
            existing_decisions = state.get('agent_decisions') or {}
        else:
            existing_decisions = state.agent_decisions or {}
        
        # Update state
        return {
            "rewritten_queries": [question],  # Store the final queries used
            "retrieved_documents": retrieved_docs,
            "retrieved_text": retrieved_text,
            "last_technique_used": decision.technique,
            "agent_decisions": {
                **existing_decisions,
                "query_rewriter": {
                    "technique": decision.technique,
                    "reasoning": decision.reasoning,
                    "rewritten_queries": [question]
                }
            }
        }
    
    return query_rewriter_node


# ============================================================================
# GRADER NODE
# Decides if retrieved documents are relevant to the question
# ============================================================================

def make_grader_node(llm, config: Dict[str, Any]):
    """
    Factory function to create the grader node.
    
    The grader agent evaluates whether retrieved documents are relevant
    to the user's question.
    
    Args:
        llm: LangChain LLM instance
        config: Configuration dict
        
    Returns:
        Node function for LangGraph
    """
    
    def grader_node(state: AgenticRAGState):
        """
        Grader Agent: Evaluate relevance of retrieved documents.
        """
        
        # Handle both AgenticRAGState and dict
        if isinstance(state, dict):
            question = state.get('original_question', '')
            retrieved_docs = state.get('retrieved_documents') or []
        else:
            question = state.original_question
            retrieved_docs = state.retrieved_documents or []
        
        logger.info(f"\n{'='*80}")
        logger.info("GRADER AGENT")
        logger.info(f"{'='*80}")
        logger.info(f"Evaluating {len(retrieved_docs)} documents")
        
        if not retrieved_docs:
            logger.warning("No documents to grade")
            return {
                "grade_decision": "no",
                "grade_score": "no",
                "grade_confidence": 0.0,
                "grade_reasoning": "No documents retrieved"
            }
        
        # Build grading prompt
        doc_text = "\n".join([
            f"[Doc {i+1}]: {doc.get('text', '')[:500]}..."
            for i, doc in enumerate(retrieved_docs[:5])  # Grade top 5
        ])
        
        prompt = f"""You are a document relevance grader.

Question: {question}

Retrieved Documents (showing first 500 chars each):
{doc_text}

Are these documents relevant to answering the question?
Consider:
- Do they contain information that could help answer the question?
- Are they related to the topic?
- Would they help generate a good answer?

Respond with ONLY valid JSON (no markdown):
{{
    "relevant": "yes" or "no",
    "confidence": 0.0-1.0,
    "reasoning": "why or why not",
    "num_relevant": 0-5
}}"""
        
        try:
            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON
            try:
                grade_dict = json.loads(response_text)
            except json.JSONDecodeError:
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                    grade_dict = json.loads(json_str)
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                    grade_dict = json.loads(json_str)
                else:
                    raise ValueError(f"Cannot parse: {response_text}")
            
            grade = DocumentGrade(**grade_dict)
            
        except Exception as e:
            logger.warning(f"Failed to grade documents: {e}, assuming relevant")
            grade = DocumentGrade(
                relevant="yes",
                confidence=0.5,
                reasoning="Error in grading",
                num_relevant=len(retrieved_docs)
            )
        
        logger.info(f"Grade: {grade.relevant} (confidence: {grade.confidence})")
        logger.info(f"Reasoning: {grade.reasoning}")
        
        # Get existing agent_decisions and retry count safely (handle both dict and AgenticRAGState)
        if isinstance(state, dict):
            existing_decisions = state.get('agent_decisions') or {}
            current_retry_count = state.get('retry_count', 0)
        else:
            existing_decisions = state.agent_decisions or {}
            current_retry_count = state.retry_count
        
        # Increment retry count if documents are not relevant (will retry)
        new_retry_count = current_retry_count + 1 if grade.relevant == "no" else current_retry_count
        
        return {
            "grade_decision": grade.relevant,
            "grade_score": grade.relevant,
            "grade_confidence": grade.confidence,
            "grade_reasoning": grade.reasoning,
            "retry_count": new_retry_count,
            "agent_decisions": {
                **existing_decisions,
                "grader": {
                    "relevant": grade.relevant,
                    "confidence": grade.confidence,
                    "num_relevant": grade.num_relevant,
                    "reasoning": grade.reasoning
                }
            }
        }
    
    return grader_node


# ============================================================================
# GENERATOR NODE
# Decides which prompting strategy to use and generates answer
# ============================================================================

def make_generator_node(llm, generator, config: Dict[str, Any]):
    """
    Factory function to create the generator node.
    
    The generator agent decides which prompting strategy to use
    and generates the final answer.
    
    Args:
        llm: LangChain LLM instance (for decision-making)
        generator: BaselineGenerator instance (for answer generation)
        config: Configuration dict
        
    Returns:
        Node function for LangGraph
    """
    
    def generator_node(state: AgenticRAGState):
        """
        Generator Agent: Decide prompting strategy and generate answer.
        """
        
        # Handle both AgenticRAGState and dict
        if isinstance(state, dict):
            question = state.get('original_question', '')
            context = state.get('retrieved_text', '')
            grade_decision = state.get('grade_decision', '')
            retry_count = state.get('retry_count', 0)
            max_retries = state.get('max_retries', 2)
        else:
            question = state.original_question
            context = state.retrieved_text
            grade_decision = state.grade_decision
            retry_count = state.retry_count
            max_retries = state.max_retries
        
        # Build prompt for strategy selection
        prompt = f"""You are an expert in selecting prompting strategies.

Question: {question}
Context available: {"Yes" if context else "No"}
Documents relevant: {grade_decision}

Which prompting strategy would work best?

Available strategies:
- standard: Direct extraction from context without explanation
- cot: Chain-of-thought reasoning with step-by-step logic (good for complex questions)
- few_shot: Example-based reasoning
- role: Expert role assumption (e.g., financial analyst) - good for domain-specific q's
- ensemble: Multiple strategies combined

Choose based on:
- Question complexity
- Whether reasoning is needed
- Domain specificity

Respond with ONLY valid JSON:
{{
    "strategy": "standard|cot|few_shot|role|ensemble",
    "reasoning": "why this strategy",
    "needs_more_context": false,
    "confidence": 0.0-1.0
}}"""
        
        try:
            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON
            try:
                strategy_dict = json.loads(response_text)
            except json.JSONDecodeError:
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                    strategy_dict = json.loads(json_str)
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                    strategy_dict = json.loads(json_str)
                else:
                    raise ValueError(f"Cannot parse: {response_text}")
            
            strategy_decision = GeneratorDecision(**strategy_dict)
            
        except Exception as e:
            logger.warning(f"Failed to decide strategy: {e}, using 'standard'")
            strategy_decision = GeneratorDecision(
                strategy="standard",
                reasoning="Error in selection, using default",
                needs_more_context=False,
                confidence=0.5
            )
        
        logger.info(f"Chose strategy: {strategy_decision.strategy}")
        logger.info(f"Confidence: {strategy_decision.confidence}")
        
        # Generate answer using the chosen strategy
        from generation.prompts import get_prompt_strategy
        
        strategy = get_prompt_strategy(
            strategy_decision.strategy,
            generator,
            {}
        )
        
        answer = strategy.generate(question, context)
        
        logger.info(f"Generated answer length: {len(answer)} chars")
        
        # Get existing agent_decisions safely (handle both dict and AgenticRAGState)
        if isinstance(state, dict):
            existing_decisions = state.get('agent_decisions') or {}
        else:
            existing_decisions = state.agent_decisions or {}
        
        return {
            "generated_answer": answer,
            "chosen_prompting_strategy": strategy_decision.strategy,
            "generation_confidence": strategy_decision.confidence,
            "agent_decisions": {
                **existing_decisions,
                "generator": {
                    "strategy": strategy_decision.strategy,
                    "reasoning": strategy_decision.reasoning,
                    "confidence": strategy_decision.confidence
                }
            }
        }
    
    return generator_node


# ============================================================================
# ROUTING FUNCTION
# Decides whether to continue or retry
# ============================================================================

def route_after_grading(state: AgenticRAGState) -> Literal["generator", "query_rewriter"]:
    """
    Route after grading: if documents not relevant and retries left, go back to rewriter.
    Otherwise, go to generator.
    """
    
    # Handle both AgenticRAGState and dict
    if isinstance(state, dict):
        grade_decision = state.get('grade_decision', '')
        retry_count = state.get('retry_count', 0)
        max_retries = state.get('max_retries', 2)
    else:
        grade_decision = state.grade_decision
        retry_count = state.retry_count
        max_retries = state.max_retries
    
    if (grade_decision == "no" and retry_count < max_retries):
        logger.info(f"Routing to query_rewriter for retry {retry_count + 1}")
        return "query_rewriter"
    
    logger.info("Routing to generator for answer generation")
    return "generator"
