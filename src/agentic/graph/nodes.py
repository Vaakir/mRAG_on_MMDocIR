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
from generation.prompts import get_prompt_strategy

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTION: Clean LLM responses
# ============================================================================

def clean_and_extract_json(response_text: str) -> dict:
    """
    Clean LLM response and extract JSON.
    
    Handles:
    - Removes <think> and </think> XML tags
    - Extracts JSON from markdown code blocks
    - Returns first valid JSON object found
    
    Args:
        response_text: Raw LLM response text
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        json.JSONDecodeError: If no valid JSON found
    """
    # Remove XML-style thinking tags
    cleaned = response_text.replace("<think>", "").replace("</think>", "").strip()
    
    # Try direct JSON parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Try markdown code blocks
    if "```json" in cleaned:
        json_str = cleaned.split("```json")[1].split("```")[0].strip()
        return json.loads(json_str)
    elif "```" in cleaned:
        json_str = cleaned.split("```")[1].split("```")[0].strip()
        return json.loads(json_str)
    
    # Last resort: try to find first { ... } JSON object
    start = cleaned.find("{")
    if start != -1:
        bracket_count = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                bracket_count += 1
            elif cleaned[i] == "}":
                bracket_count -= 1
                if bracket_count == 0:
                    json_str = cleaned[start:i+1]
                    return json.loads(json_str)
    
    # If all else fails, raise error
    raise json.JSONDecodeError(f"Cannot extract valid JSON from: {response_text}", response_text, 0)


# ============================================================================
# QUERY REWRITER NODE
# Decides which QueryTechnique to use based on the question
# ============================================================================

def make_query_rewriter_node(llm, query_techniques_dict, config: Dict[str, Any]):
    """
    Factory function to create the query rewriter node.
    
    The query rewriter agent decides which of the 8 QueryTechnique classes
    to use, based on the user's question.
    
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
        
        prompt = f"""You are a query planning agent. Your job is to decide which query technique to use for better retrieval.

QUESTION: {question}

AVAILABLE TECHNIQUES (choose exactly ONE):
1. "standard": No modification (baseline retrieval)
2. "multi_query": Generate multiple paraphrases and retrieve for each (ambiguous/multi-faceted questions)
3. "rag_fusion": Paraphrases with RRF fusion (complex questions)
4. "step_back": Abstract to broader concept, then retrieve (specific queries needing context)
5. "hyde": Generate hypothetical documents (semantic searches)
6. "query_decomposition": Break into sub-questions (multi-part questions)
7. "query_rewriting": Improve grammar and clarity (poorly-phrased questions)
8. "query_expansion": Add synonyms and related terms (narrow searches, entity searches)

RETRY CONTEXT:
- Attempt number: {retry_count + 1}
- Last technique used: {last_technique if last_technique else "None (first attempt)"}
- On retry: Choose a DIFFERENT technique than last time

DECISION:
Think about the question type, then select the best technique.

REQUIRED: Return ONLY a JSON object with exactly 2 fields:
1. "technique": Must be exactly one of the 8 techniques listed above (string)
2. "reasoning": Explain why this technique will help (string)

Return ONLY this JSON (no markdown, no code blocks, no extra text):
{{"technique": "standard", "reasoning": "This is a straightforward factual question"}}"""
        
        # Call LLM to get decision on which technique to use
        try:
            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response) # Handle different response types
            
            # Try to parse as JSON
            decision_dict = clean_and_extract_json(response_text) # This will raise an error if parsing fails, which we catch below
            decision = QueryRewriterDecision(**decision_dict) # Validate and create Pydantic model (also checks if technique is valid)
            
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
        if not technique: # This should not happen due to Pydantic validation, but we check just in case
            logger.warning(f"Technique {decision.technique} not found, using 'standard'")
            technique = query_techniques_dict['standard']
        
        # Retrieve documents using the technique
        retrieved_docs = technique.retrieve(question, top_k=config.get('TOP_K', 5))
        
        # Format the retrieved text
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
            "last_technique_used": decision.technique, # Store last technique for retry logic
            "agent_decisions": { # Store the decision details for analysis
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
        
        if not retrieved_docs: # No documents to grade; return not relevant
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
        
        prompt = f"""You are a document relevance grader. Your job is to assess if retrieved documents are relevant to the question.

QUESTION:
{question}

RETRIEVED DOCUMENTS (first 500 chars each):
{doc_text}

EVALUATION CRITERIA:
1. Does the content contain information to answer the question?
2. Is the content related to the topic?
3. Would this help generate a good answer?

REQUIRED: Return ONLY a JSON object with ALL four fields:
1. "relevant": Must be exactly "yes" or "no" (string)
2. "confidence": Must be a number from 0.0 to 1.0 (e.g., 0.95 for 95% confident)
3. "num_relevant": Must be an integer from 0 to {len(retrieved_docs)} (count of relevant docs)
4. "reasoning": Explain why you made this judgment (string)

Return ONLY this JSON (no markdown, no code blocks, no extra text):
{{"relevant": "yes/no", "confidence": 0.0to1.0, "num_relevant": 0-{len(retrieved_docs)}, "reasoning": "explanation"}}"""
        
        try:
            response = llm.invoke(prompt) # Call LLM to get grading decision
            response_text = response.content if hasattr(response, 'content') else str(response) # Handle different response types
            
            # Parse JSON
            grade_dict = clean_and_extract_json(response_text)
            grade = DocumentGrade(**grade_dict) # Validate and create Pydantic model (also checks field types and values)
            
        except Exception as e:
            logger.warning(f"Failed to grade documents: {e}, assuming relevant")
            grade = DocumentGrade( # Default to relevant if grading fails, to avoid blocking generation
                relevant="yes",
                confidence=0.5,
                reasoning="Error in grading",
                num_relevant=len(retrieved_docs)
            )
        
        logger.info(f"Grade: {grade.relevant} (confidence: {grade.confidence})")
        logger.info(f"Reasoning: {grade.reasoning}")
        
        # Get existing agent_decisions and retry count safely (handle both dict and AgenticRAGState)
        if isinstance(state, dict):
            existing_decisions = state.get('agent_decisions') or {} # Existing decisions from previous nodes
            current_retry_count = state.get('retry_count', 0)       # Current retry count for this question
        else:
            existing_decisions = state.agent_decisions or {} # Existing decisions from previous nodes
            current_retry_count = state.retry_count          # Current retry count for this question
        
        # Increment retry count if documents are not relevant (will retry)
        new_retry_count = current_retry_count + 1 if grade.relevant == "no" else current_retry_count
        
        return {
            "grade_decision": grade.relevant, # Store the relevance decision for routing
            "grade_score": grade.relevant,    # Store the same relevance decision as a score for clarity
            "grade_confidence": grade.confidence, # Store the confidence in the grading decision
            "grade_reasoning": grade.reasoning, # Store the reasoning for the grading decision
            "retry_count": new_retry_count, # Update retry count in state (increment if documents are not relevant)
            "agent_decisions": { # Store the decision details for analysis
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
            question = state.get('original_question', '') # Original question (not rewritten, for generation)
            context = state.get('retrieved_text', '')     # Formatted retrieved text to use as context for generation
            grade_decision = state.get('grade_decision', '') # Grader's decision on the relevance (to inform strategy choice)
            retry_count = state.get('retry_count', 0)
            max_retries = state.get('max_retries', 2)
        else:
            question = state.original_question
            context = state.retrieved_text
            grade_decision = state.grade_decision
            retry_count = state.retry_count
            max_retries = state.max_retries
        
        # Build prompt for strategy selection
        prompt = f"""You are an expert in selecting prompting strategies for answer generation.

CONTEXT:
- Question: {question}
- Context available: {"Yes" if context else "No"}
- Documents relevant: {grade_decision}

AVAILABLE STRATEGIES:
1. "standard": Direct extraction from context without explanation (factual, simple)
2. "cot": Chain-of-thought reasoning with step-by-step logic (complex, requires reasoning)
3. "few_shot": Example-based reasoning (learn from patterns)
4. "role": Expert role assumption (financial analyst, researcher, etc.) (domain-specific)
5. "ensemble": Multiple strategies combined (highest quality but slower)

SELECTION CRITERIA:
- Question complexity (simple vs. complex)
- Whether reasoning/explanation is needed
- Domain specificity (general vs. specialized)
- Available context quality

REQUIRED: Return ONLY a JSON object with ALL four fields:
1. "strategy": Must be exactly one of: "standard", "cot", "few_shot", "role", or "ensemble"
2. "reasoning": Explain your choice (why this strategy works best)
3. "needs_more_context": Boolean true or false (do we need more context?)
4. "confidence": A number from 0.0 to 1.0 (how confident are you?)

Return ONLY this JSON (no markdown, no code blocks, no extra text):
{{"strategy": "standard/cot/few_shot/role/ensemble", "reasoning": "explanation", "needs_more_context": false, "confidence": 0.9}}"""
        
        try:
            response = llm.invoke(prompt) # Call LLM to get strategy decision
            response_text = response.content if hasattr(response, 'content') else str(response) # Handle different response types
            
            # Parse JSON
            strategy_dict = clean_and_extract_json(response_text)
            strategy_decision = GeneratorDecision(**strategy_dict) # Validate and create Pydantic model (also checks if strategy is valid and that confidence is in range)
            
        except Exception as e:
            logger.warning(f"Failed to decide strategy: {e}, using 'standard'")
            strategy_decision = GeneratorDecision( # Default to standard strategy if decision fails, to ensure continuation of generation
                strategy="standard",
                reasoning="Error in selection, using default",
                needs_more_context=False,
                confidence=0.5
            )
        
        logger.info(f"Chose strategy: {strategy_decision.strategy}")
        logger.info(f"Confidence: {strategy_decision.confidence}")
        
        # Generate answer using the chosen strategy        
        strategy = get_prompt_strategy( # Get the actual prompting strategy function based on the decision
            strategy_decision.strategy,
            generator,
            {}
        )
        
        answer = strategy.generate(question, context) # Generate the answer using the selected strategy
        
        logger.info(f"Generated answer length: {len(answer)} chars")
        
        # Get existing agent_decisions safely (handle both dict and AgenticRAGState)
        if isinstance(state, dict):
            existing_decisions = state.get('agent_decisions') or {} # Existing decisions from previous nodes
        else:
            existing_decisions = state.agent_decisions or {} # Existing decisions from previous nodes
        
        return {
            "generated_answer": answer, # Store the generated answer in state
            "chosen_prompting_strategy": strategy_decision.strategy, # Store the chosen strategy for reporting
            "generation_confidence": strategy_decision.confidence, # Store the confidence in the generation quality
            "agent_decisions": { # Store the decision details for analysis
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
    Route after grading: if documents are not relevant and retries left, go back to rewriter.
    Otherwise, go to generator.
    """
    
    # Handle both AgenticRAGState and dict
    if isinstance(state, dict):
        grade_decision = state.get('grade_decision', '') # Grader's decision on the relevance
        retry_count = state.get('retry_count', 0)        # Current retry count for this question
        max_retries = state.get('max_retries', 2)        # Maximum retries allowed before forcing generation
    else:
        grade_decision = state.grade_decision
        retry_count = state.retry_count
        max_retries = state.max_retries
    
    if (grade_decision == "no" and retry_count < max_retries): # If documents are not relevant and we have retries left, go back to query rewriter
        logger.info(f"Routing to query_rewriter for retry {retry_count + 1}")
        return "query_rewriter"
    
    logger.info("Routing to generator for answer generation")
    return "generator"
