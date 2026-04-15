"""Node implementations for the agentic graph."""

import logging
import json
from typing import Literal, Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage
from .state import AgenticRAGState
from ..tools.output_parser import (
    QueryRewriterDecision,
    DocumentGrade,
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

def make_query_rewriter_node(agent_llm, query_techniques_dict, config: Dict[str, Any]):
    """
    Factory function to create the query rewriter node.
    
    The query rewriter agent decides which of the 8 QueryTechnique classes
    to use, based on the user's question.
    
    Args:
        agent_llm: Lightweight LLM instance for decision-making (qwen3-vl:8b)
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
        
        print(f"\n{'='*80}")
        print(f"QUERY REWRITER AGENT (Attempt {retry_count + 1})")
        print(f"{'='*80}")
        print(f"Question: {question}")
        if last_technique:
            print(f"Last technique (to avoid on retry): {last_technique}")
        
        # Build prompt for LLM to decide which technique to use
        available_techniques = list(query_techniques_dict.keys())
        
        prompt = f"""Choose ONE query technique for this question:
Question: {question}

Techniques:
1. standard - baseline retrieval
2. multi_query - multiple paraphrases
3. rag_fusion - paraphrases with fusion
4. step_back - abstract to broader concept
5. hyde - hypothetical documents
6. query_decomposition - break into sub-questions
7. query_rewriting - improve phrasing
8. query_expansion - add synonyms

Last technique: {last_technique if last_technique else "none"}
Attempt: {retry_count + 1}

RESPOND WITH ONLY THIS JSON:
{{"technique": "<one of above>", "reasoning": "<short explanation>"}}

NO OTHER TEXT. ONLY JSON."""
        
        # Call agent LLM to get decision on which technique to use
        try:
            response = agent_llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response) # Handle different response types
            
            # Try to parse as JSON
            decision_dict = clean_and_extract_json(response_text) # This will raise an error if parsing fails, which we catch below
            decision = QueryRewriterDecision(**decision_dict) # Validate and create Pydantic model (also checks if technique is valid)
            
        except Exception as e:
            print(f"Failed to parse LLM decision: {e}, falling back to 'standard'")
            decision = QueryRewriterDecision(
                technique="standard",
                reasoning="Error in parsing, using baseline",
                rewritten_queries=[question]
            )
        
        print(f"Chose technique: {decision.technique}")
        print(f"Reasoning: {decision.reasoning}")
        
        # Apply the chosen query technique
        technique = query_techniques_dict.get(decision.technique)
        if not technique: # This should not happen due to Pydantic validation, but we check just in case
            print(f"Technique {decision.technique} not found, using 'standard'")
            technique = query_techniques_dict['standard']
        
        # Retrieve documents using the technique
        retrieved_docs = technique.retrieve(question, top_k=config.get('TOP_K', 5))
        
        # Format the retrieved text
        retrieved_text = "\n\n".join([
            f"[Document {i+1}]:\n{doc.get('text', '')}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        print(f"Retrieved {len(retrieved_docs)} documents")
        
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

def make_grader_node(agent_llm, config: Dict[str, Any]):
    """
    Factory function to create the grader node.
    
    The grader agent evaluates whether retrieved documents are relevant
    to the user's question. Enhanced version evaluates ALL documents
    with multi-criteria scoring (not just top 3).
    
    Args:
        agent_llm: Lightweight LLM instance for decision-making (qwen3-vl:8b)
        config: Configuration dict
        
    Returns:
        Node function for LangGraph
    """
    
    def grader_node(state: AgenticRAGState):
        """
        Grader Agent: Evaluate relevance of retrieved documents with multi-criteria scoring.
        
        Enhanced to:
        - Evaluate ALL retrieved documents (not just top 3)
        - Use multi-criteria grading (answer containment, concept matching, context type)
        - Provide detailed breakdown of relevant vs irrelevant docs
        - Better count actual relevant documents
        """
        
        # Handle both AgenticRAGState and dict
        if isinstance(state, dict):
            question = state.get('original_question', '')
            retrieved_docs = state.get('retrieved_documents') or []
        else:
            question = state.original_question
            retrieved_docs = state.retrieved_documents or []
        
        print(f"\n{'='*80}")
        print("GRADER AGENT")
        print(f"{'='*80}")
        print(f"Evaluating {len(retrieved_docs)} documents for relevance")
        
        if not retrieved_docs: # No documents to grade; return not relevant
            print("No documents to grade")
            return {
                "grade_decision": "no",
                "grade_score": "no",
                "grade_confidence": 0.0,
                "grade_reasoning": "No documents retrieved"
            }
        
        # Build grading prompt that evaluates ALL documents with multi-criteria approach
        # Show summaries of all docs, ask for detailed grading
        doc_summaries = []
        for i, doc in enumerate(retrieved_docs):
            doc_text = doc.get('text', '')
            # Truncate to 250 chars per doc to fit more docs in context
            summary = doc_text[:250] + "..." if len(doc_text) > 250 else doc_text
            doc_summaries.append(f"[Doc {i+1}]: {summary}")
        
        doc_text = "\n\n".join(doc_summaries)
        
        prompt = f"""You are an expert evaluator. Grade these documents on relevance to the question.

Question: {question}

Documents ({len(retrieved_docs)} total):
{doc_text}

For EACH document, evaluate:
1. Does it contain answer content for the question?
2. Does it mention key concepts from the question?
3. Is the context type (table, text, image caption, etc.) appropriate?

Then decide:
- Which document numbers are RELEVANT? (e.g., [1, 3, 5])
- Which are PARTIALLY relevant? (e.g., [2, 4])
- Overall decision: yes|no (is there enough relevant content overall?)
- Confidence: 0.0-1.0 (how sure are you?)
- Reasoning: specific details on what makes docs relevant/irrelevant

RESPOND WITH ONLY THIS JSON:
{{
    "relevant": "yes",
    "confidence": 0.85,
    "relevant_docs": [1, 3, 5],
    "partial_docs": [2],
    "irrelevant_docs": [4],
    "reasoning": "Docs 1, 3, 5 directly contain answer content about [topic]. Doc 2 is tangentially related. Doc 4 is irrelevant. Strong confidence in relevance."
}}

NO OTHER TEXT. ONLY JSON."""
        
        try:
            response = agent_llm.invoke(prompt) # Call agent LLM to get detailed grading decision
            response_text = response.content if hasattr(response, 'content') else str(response) # Handle different response types
            
            # Parse JSON
            grade_dict = clean_and_extract_json(response_text)
            
            # Calculate num_relevant from relevant_docs list (counts the documents the LLM marked as relevant)
            relevant_docs_list = grade_dict.get('relevant_docs', [])
            num_relevant = len(relevant_docs_list)
            
            # Validate and create Pydantic model
            grade = DocumentGrade(**{
                'relevant': grade_dict.get('relevant', 'no'),
                'confidence': grade_dict.get('confidence', 0.5),
                'reasoning': grade_dict.get('reasoning', ''),
                'num_relevant': num_relevant  # Derived from the relevant_docs list
            })
            
            # Extract list of relevant document indices (1-based from LLM) for filtering
            relevant_doc_indices = grade_dict.get('relevant_docs', [])
            
            # Log detailed grading information
            print(f"Grade: {grade.relevant} (confidence: {grade.confidence})")
            print(f"Relevant docs: {len(relevant_doc_indices)}/{len(retrieved_docs)} (indices: {relevant_doc_indices})")
            if 'partial_docs' in grade_dict:
                print(f"Partially relevant docs: {grade_dict['partial_docs']}")
            print(f"Reasoning: {grade.reasoning}")
            
        except Exception as e:
            print(f"Failed to grade documents: {e}, assuming relevant")
            grade = DocumentGrade( # Default to relevant if grading fails, to avoid blocking generation
                relevant="yes",
                confidence=0.5,
                reasoning="Error in advanced grading; defaulting to relevant",
                num_relevant=len(retrieved_docs)
            )
            # If parsing fails, treat all docs as relevant
            relevant_doc_indices = list(range(1, len(retrieved_docs) + 1))
        
        print(f"Grade: {grade.relevant} (confidence: {grade.confidence})")
        print(f"Reasoning: {grade.reasoning}")
        
        # Get existing agent_decisions and retry count safely (handle both dict and AgenticRAGState)
        if isinstance(state, dict):
            existing_decisions = state.get('agent_decisions') or {} # Existing decisions from previous nodes
            current_retry_count = state.get('retry_count', 0)       # Current retry count for this question
        else:
            existing_decisions = state.agent_decisions or {} # Existing decisions from previous nodes
            current_retry_count = state.retry_count          # Current retry count for this question
        
        # Increment retry count if documents are not relevant, or confidence is too low (will retry)
        # This must match the retry conditions in route_after_grading()
        confidence_threshold = 0.6  # Must match the threshold in route_after_grading()
        confidence_too_low = grade.confidence < confidence_threshold
        should_retry = grade.relevant == "no" or confidence_too_low
        new_retry_count = current_retry_count + 1 if should_retry else current_retry_count
        
        # Only return documents that were marked as relevant by the grader
        # relevant_doc_indices are 1-based from LLM response, convert to 0-based for list indexing
        if relevant_doc_indices:
            filtered_docs = [retrieved_docs[i-1] for i in relevant_doc_indices if i > 0 and i <= len(retrieved_docs)]
            print(f"Filtered {len(retrieved_docs)} docs --> {len(filtered_docs)} relevant docs")
        else:
            # If parsing failed, keep all docs
            filtered_docs = retrieved_docs
        
        return {
            "retrieved_documents": filtered_docs,  # Return only those documents marked as relevant by the grader
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
                    "num_docs_evaluated": len(retrieved_docs),
                    "reasoning": grade.reasoning
                }
            }
        }
    
    return grader_node


# ============================================================================
# GENERATOR NODE
# Decides which prompting strategy to use and generates answer
# ============================================================================

def make_generator_node(agent_llm, generator, config: Dict[str, Any]):
    """
    Factory function to create the generator node.
    
    The generator agent decides which prompting strategy to use
    and then uses the heavy LLM (via generator) for final answer generation.
    
    Args:
        agent_llm: Lightweight LLM instance for strategy decision-making (qwen3-vl:8b)
        generator: BaselineGenerator instance (uses qwen3:32b for answer generation)
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
            grade_confidence = state.get('grade_confidence', 0.0) # Grader's confidence in relevance (to inform caution level)
            retry_count = state.get('retry_count', 0)
            max_retries = state.get('max_retries', 2)
        else:
            question = state.original_question
            context = state.retrieved_text
            grade_decision = state.grade_decision
            grade_confidence = state.grade_confidence
            retry_count = state.retry_count
            max_retries = state.max_retries
        
        # Build prompt for strategy selection
        prompt = f"""Choose a generation strategy:
Question: {question}
Context available: {"yes" if context else "no"}
Documents relevant: {grade_decision}
Grader confidence: {grade_confidence:.2f} (scale 0.0-1.0)

Strategies:
1. standard - direct extraction (use when docs are highly relevant)
2. cot - step-by-step reasoning (use for complex questions)
3. few_shot - example-based (use when patterns are clear)
4. role - expert perspective (use for domain-specific topics)
5. ensemble - combined approach (use when uncertain)

Select the best strategy based on question complexity, context quality, and grader confidence.
If grader confidence < 0.6, consider using ensemble or cot for more reasoning.

RESPOND WITH ONLY THIS JSON:
{{"strategy": "standard", "reasoning": "straightforward factual question", "needs_more_context": false, "confidence": 0.9}}

NO OTHER TEXT. ONLY JSON."""
        
        try:
            response = agent_llm.invoke(prompt) # Call agent LLM to get strategy decision
            response_text = response.content if hasattr(response, 'content') else str(response) # Handle different response types
            
            # Parse JSON
            strategy_dict = clean_and_extract_json(response_text)
            strategy_decision = GeneratorDecision(**strategy_dict) # Validate and create Pydantic model (also checks if strategy is valid and that confidence is in range)
            
        except Exception as e:
            print(f"Failed to decide strategy: {e}, using 'standard'")
            strategy_decision = GeneratorDecision( # Default to standard strategy if decision fails, to ensure continuation of generation
                strategy="standard",
                reasoning="Error in selection, using default",
                needs_more_context=False,
                confidence=0.5
            )
        
        print(f"Chose strategy: {strategy_decision.strategy}")
        print(f"Confidence: {strategy_decision.confidence}")
        
        # Generate answer using the chosen strategy        
        strategy = get_prompt_strategy( # Get the actual prompting strategy function based on the decision
            strategy_decision.strategy,
            generator,
            {}
        )
        
        answer = strategy.generate(question, context) # Generate the answer using the selected strategy
        
        print(f"Generated answer length: {len(answer)} chars")
        
        # Get existing agent_decisions and retrieved_documents safely (handle both dict and AgenticRAGState)
        if isinstance(state, dict):
            existing_decisions = state.get('agent_decisions') or {} # Existing decisions from previous nodes
            retrieved_docs = state.get('retrieved_documents') or []  # Get retrieved documents from state to preserve them
        else:
            existing_decisions = state.agent_decisions or {} # Existing decisions from previous nodes
            retrieved_docs = state.retrieved_documents or []  # Get retrieved documents from state to preserve them
        
        return {
            "retrieved_documents": retrieved_docs,  # PRESERVE: explicitly return retrieved documents to prevent them from being cleared
            "generated_answer": answer, # Store the generated answer in state
            "chosen_prompting_strategy": strategy_decision.strategy, # Store the chosen strategy for reporting
            "generation_confidence": strategy_decision.confidence, # Store the confidence in the generation quality
            "grade_confidence": grade_confidence,  # PASS: Forward the grader's confidence 
            "agent_decisions": { # Store the decision details for analysis
                **existing_decisions,
                "generator": {
                    "strategy": strategy_decision.strategy,
                    "reasoning": strategy_decision.reasoning,
                    "confidence": strategy_decision.confidence,
                    "grader_confidence": grade_confidence  # Include grader confidence in decisions log
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
    Route after grading: if documents are not relevant OR confidence is low, and there are retries left, go back to rewriter.
    Otherwise, go to generator.
    
    The retry_count is incremented by the grader when it returns "no", so:
    - retry_count = 1 means "first retry attempt was just made"
    - max_retries = 1 means "allow up to 1 retry attempt"
    - We use <= to allow the current retry to proceed
    """
    
    # Handle both AgenticRAGState and dict
    if isinstance(state, dict):
        grade_decision = state.get('grade_decision', '') # Grader's decision on the relevance
        grade_confidence = state.get('grade_confidence', 0.0) # Grader's confidence in the decision
        retry_count = state.get('retry_count', 0)        # Current retry count for this question
        max_retries = state.get('max_retries', 2)        # Maximum retries allowed before forcing generation
    else:
        grade_decision = state.grade_decision
        grade_confidence = state.grade_confidence
        retry_count = state.retry_count
        max_retries = state.max_retries
    
    # Check both grade_decision and confidence threshold
    # Retry if: (1) documents marked not relevant, or (2) confidence too low (< 0.6) AND there are retries remaining
    confidence_threshold = 0.6  # Minimum acceptable confidence
    confidence_too_low = grade_confidence < confidence_threshold
    decision_negative = grade_decision == "no"
    retries_remaining = retry_count <= max_retries
    
    if ((decision_negative or confidence_too_low) and retries_remaining):
        reason = "grade=no" if decision_negative else f"confidence={grade_confidence:.2f} < {confidence_threshold}"
        print(f"Routing to query_rewriter for retry (attempt {retry_count + 1}): {reason}")
        return "query_rewriter"
    
    print(f"Routing to generator for answer generation (retries exhausted or high confidence)")
    return "generator"
