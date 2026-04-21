"""Node implementations for the agentic graph."""

import logging
import json
from typing import Literal, Dict, Any

from .state import AgenticRAGState
from ..tools.output_parser import (
    QueryRewriterDecision,
    DocumentGrade,
    GeneratorDecision
)
from generation.prompts import get_prompt_strategy
from retrieval_techniques import MultimodalRetriever
from generation.answer_validator import validate_answer_format
from config.config import DATA_DIR
from pathlib import Path
from generation.generator import VisionGenerator

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTION: Clean LLM responses
# ============================================================================

def clean_and_extract_json(response_text: str) -> dict:
    """
    Cleaning the LLM response and extract JSON.
    
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
# - Decides which QueryTechnique to use based on the question
# ============================================================================

def make_query_rewriter_node(agent_llm, retriever, query_techniques_dict, config: Dict[str, Any]):
    """
    Factory function to create the query rewriter node.
    
    The query rewriter agent decides which of the 8 QueryTechnique classes
    to use, based on the user's question.
    
    Args:
        agent_llm: Lightweight LLM instance for decision-making
        retriever: Retriever instance (HybridRetriever, MultimodalRetriever, or basic Retriever)
                   If None, falls back to using query techniques directly
        query_techniques_dict: Dict mapping technique names to QueryTechnique instances
        config: Configuration dict
        
    Returns:
        Node function for LangGraph
    """
    
    def query_rewriter_node(state: AgenticRAGState):
        """
        Query Rewriter Agent: Decides which technique to use for rewriting/retrieving.
        
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
        
        # Build prompt for LLM to decide which query technique to use
        # Create technique descriptions
        techniques_info = {
            "standard": "baseline retrieval (use when question is clear and well-formed)",
            "multi_query": "multiple paraphrases (use when question can be asked multiple ways)",
            "rag_fusion": "paraphrases with fusion (use when you need to combine results from multiple reformulations)",
            "step_back": "abstract to broader concept (use when question is too specific and needs foundational context)",
            "hyde": "hypothetical documents (use when searching for rare/niche information)",
            "query_decomposition": "break into sub-questions (use when question has multiple parts or compound structure)",
            "query_rewriting": "improve phrasing (use when question has grammar issues, unclear wording, or lacks structure)",
            "query_expansion": "add synonyms (use when question uses domain-specific/niche terminology)"
        }
        
        # On retry, add clear warning about last technique
        if last_technique and retry_count > 0:
            avoid_instruction = f"\nMUST AVOID: '{last_technique.upper()}' was already tried and failed. Choose a DIFFERENT technique.\n"
            strategy_note = f"Since {last_technique} didn't work, try a fundamentally different approach."
        else:
            avoid_instruction = ""
            strategy_note = ""
        
        # Build technique list description
        technique_list = "\n".join([
            f"- {name}: {desc}"
            for name, desc in techniques_info.items()
            if not (last_technique and name.lower() == last_technique.lower() and retry_count > 0)
        ])
        
        prompt = f"""You are deciding which query technique to use for information retrieval.

Question: {question}
Attempt: {retry_count + 1}{avoid_instruction}

Available Query Techniques:
{technique_list}

Selection Guidance:
- Simple/clear questions → standard or multi_query
- Ambiguous phrasing → multi_query or query_rewriting
- Multi-part questions (with "and") → query_decomposition or query_expansion
- Complex/abstract questions → step_back
- Rare/niche topics → hyde or query_expansion
- Poorly worded questions → query_rewriting

{strategy_note}

CRITICAL: Your response MUST be ONLY a valid JSON object:
{{"technique":"<technique_name>","reasoning":"<brief explanation>"}}

Do NOT include any text before or after the JSON, no code blocks, no extra text.
Choose a technique name from the list above."""
        
        # Call agent LLM to get a decision on which query technique to use
        try:
            response = agent_llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse as JSON
            decision_dict = clean_and_extract_json(response_text)
            decision = QueryRewriterDecision(**decision_dict)
            
        except Exception as e:
            print(f"Failed to parse LLM decision: {e}, falling back to 'standard'")
            decision = QueryRewriterDecision(
                technique="standard", # Default to standard if parsing fails
                reasoning="Error in parsing, using baseline",
                rewritten_queries=[question]
            )
        
        print(f"Chose technique: {decision.technique}")
        print(f"Reasoning: {decision.reasoning}")
        
        # Retrieve documents using either MultimodalRetriever or selected technique
        top_k = config.get('TOP_K', 5)
        
        # Get the selected query technique object
        technique = query_techniques_dict.get(decision.technique)
        if not technique:
            print(f"Technique {decision.technique} not found, using 'standard'")
            technique = query_techniques_dict['standard'] # Fallback to standard if technique not found
        
        print(f"Using query technique: {decision.technique}")
        
        # If we have a MultimodalRetriever, pass the agent's technique decision to it. Otherwise, use the technique directly
        if isinstance(retriever, MultimodalRetriever):
            print("Using MultimodalRetriever for image-aware retrieval...")
            # Pass the agent's decision to MultimodalRetriever, which will use it for text retrieval while handling multimodal (image) retrieval internally
            retrieved_docs = retriever.retrieve(question, top_k=top_k, technique=technique)
        else:
            # Direct retrieval without MultimodalRetriever
            retrieved_docs = technique.retrieve(question, top_k=top_k)
        
        # Detect image types in retrieved documents
        image_types = {"page_image", "figure", "evidence"}
        detected_image_types = []
        image_paths = []
        for doc in retrieved_docs: # Check retrieved documents for any image types (for routing to image-aware generator if needed)
            doc_type = doc.get("payload", {}).get("type") # Check metadata for type (e.g., "page_image", "figure", "evidence")
            if doc_type in image_types:
                if doc_type not in detected_image_types: # Only add unique image types to the list
                    detected_image_types.append(doc_type)
                # Collect image paths for potential image generation
                img_path = doc.get("payload", {}).get("image_path")
                if img_path:
                    # Resolve image path: Qdrant stores paths relative to DATA_DIR
                    # Prepend DATA_DIR to make absolute path for image encoding                    
                    img_path_obj = Path(img_path)
                    if not img_path_obj.is_absolute():
                        # Relative path from DATA_DIR, resolve it
                        resolved_path = DATA_DIR / img_path
                    else:
                        # Already absolute
                        resolved_path = img_path_obj
                    
                    image_paths.append(str(resolved_path))
        
        # Format (concatenate) the retrieved text (including all retrieved documents) for the Generator's context
        retrieved_text = "\n\n".join([
            f"[Document {i+1}]:\n{doc.get('text', '')}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        print(f"Retrieved {len(retrieved_docs)} documents || ({len(detected_image_types)} image types)")
        
        # Get existing agent_decisions safely
        if isinstance(state, dict):
            existing_decisions = state.get('agent_decisions') or {}
        else:
            existing_decisions = state.agent_decisions or {}
        
        # Update state with retrieved documents and image information
        return {
            "rewritten_queries": [question],
            "retrieved_documents": retrieved_docs,
            "retrieved_text": retrieved_text,
            "detected_image_types": detected_image_types,   # Track image types for generator routing
            "has_images": len(detected_image_types) > 0,    # Flag for image-aware generation
            "image_paths": image_paths,                     # Store image paths for generation
            "last_technique_used": decision.technique,
            "agent_decisions": {
                **existing_decisions,
                "query_rewriter": {
                    "technique": decision.technique,
                    "reasoning": decision.reasoning,
                    "rewritten_queries": [question],
                    "has_images": len(detected_image_types) > 0, # Track if images were detected in the retrieved documents
                    "num_images": len(image_paths)               # Track number of images retrieved for potential image-aware generation
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
    to the user's question. (Enhanced version evaluates ALL documents
    with multi-criteria scoring (not just top 3).)
    
    Args:
        agent_llm: Lightweight LLM instance for decision-making
        config: Configuration dict
        
    Returns:
        Node function for LangGraph
    """
    
    def grader_node(state: AgenticRAGState):
        """
        Grader Agent: Evaluate relevance of retrieved documents with multi-criteria scoring.
        
        The Grader Agent:
        - Evaluate ALL retrieved documents
        - Use multi-criteria grading (answer containment, concept matching, context type)
        - Provide detailed breakdown of relevant vs irrelevant docs
        - Counts actual relevant documents
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
            doc_text = doc.get('text', '') # Use full text for grading, not truncated summary, to give the Grader enough context to evaluate relevance
            doc_type = doc.get('metadata', {}).get('type', 'text') # Check metadata for type (e.g., "page_image", "figure", "evidence") to identify multimodal content
            
            if doc_type in {"page_image", "figure", "evidence"}:
                # Inform the grader that this is visual content, so it marks it as relevant rather than discarding empty/short text
                summary = f"[{doc_type}] A retrieved image that is visually relevant to the user's question."
                doc_summaries.append(f"[Doc {i+1}]: {summary}")
            else:
                # Giving the Grader enough context to evaluate whether the chunk actually contains the answer (solving the Truncation Mismatch)
                summary = doc_text
                doc_summaries.append(f"[Doc {i+1}]: {summary}")
        
        doc_text = "\n\n".join(doc_summaries) # This represents the retrieved information the Grader has to evaluate for relevance
        
        prompt = f"""You are a strict evaluator. Grade these documents on whether they CONTAIN THE SPECIFIC ANSWER TO THE QUESTION.

Question: {question}

Documents ({len(retrieved_docs)} total):
{doc_text}

CRITICAL EVALUATION RULE:
For EACH document, attempt to EXTRACT the specific answer:
1. State: "Can I extract the specific answer to the question from this document?"
2. If YES: Show the exact extracted value (number, name, list, etc.)
3. If NO: Explain why (vague reference, missing data, wrong section, off-topic, etc.)

RELEVANCE CRITERIA (STRICT):
- RELEVANT (mark doc as relevant ONLY IF you successfully extracted the specific answer)
- PARTIALLY RELEVANT (if document discusses topic but lacks the exact answer needed)
- IRRELEVANT (if extraction impossible or answer not in document)

Then output:
- Which documents are RELEVANT (has extractable answer)?
- Which are PARTIALLY relevant (topic-related but incomplete)?
- Overall decision: yes|no (enough relevant docs for answering?)
- Confidence: 0.0-1.0 (how confident in answer being derivable?)
- Reasoning: Show what you extracted from each relevant doc

CRITICAL: Your response MUST be ONLY a valid JSON object. Do NOT include:
- Any text before the JSON
- Any text after the JSON
- Code block markers (```, ```json)
- Explanations or reasoning outside JSON
- Any markdown formatting

Output ONLY the raw JSON starting with {{ and ending with }}, like this:
{{"relevant":"yes","confidence":0.85,"relevant_docs":[1,3],"partial_docs":[2,4],"irrelevant_docs":[5],"reasoning":"Doc 1: extracted '$11407M FY2021 debt'. Doc 3: extracted 'total debt $11.4B'. Doc 2: mentions debt but not FY2021..."}}"""
        
        try:
            response = agent_llm.invoke(prompt) # Call agent LLM to get THE detailed grading decision
            response_text = response.content if hasattr(response, 'content') else str(response) # Handle different response types
            
            # Parse JSON
            grade_dict = clean_and_extract_json(response_text)
            
            # Handle LLM hallucination: if 'relevant' field is a list, convert to string
            relevant_val = grade_dict.get('relevant', 'no')
            if isinstance(relevant_val, list):
                # LLM hallucinated a list; convert to string representation
                relevant_val = 'yes' if relevant_val else 'no'
            
            # Calculate num_relevant from relevant_docs list (i.e. counts the documents that the LLM marked as relevant)
            relevant_docs_list = grade_dict.get('relevant_docs', [])
            num_relevant = len(relevant_docs_list)
            
            # Validate and create Pydantic model
            grade = DocumentGrade(**{
                'relevant': relevant_val,
                'confidence': grade_dict.get('confidence', 0.5),
                'reasoning': grade_dict.get('reasoning', ''),
                'num_relevant': num_relevant 
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
            grade = DocumentGrade( # Default to 'relevant' if grading fails (to avoid blocking generation)
                relevant="yes",
                confidence=0.5,
                reasoning="Error in advanced grading; defaulting to relevant",
                num_relevant=len(retrieved_docs)
            )
            # If parsing fails, treat all docs as relevant
            relevant_doc_indices = list(range(1, len(retrieved_docs) + 1))
        
        # Get existing agent_decisions and retry count safely (handle both dict and AgenticRAGState)
        if isinstance(state, dict):
            existing_decisions = state.get('agent_decisions') or {} # Existing decisions from previous nodes
            current_retry_count = state.get('retry_count', 0)       # Current retry count for this question
        else:
            existing_decisions = state.agent_decisions or {} # Existing decisions from previous nodes
            current_retry_count = state.retry_count          # Current retry count for this question
        
        # Increment retry count if documents are not relevant, or if confidence is too low (will retry)
        # This must match the retry conditions in route_after_grading()
        confidence_threshold = config.get('GRADER_CONFIDENCE_THRESHOLD', 0.6)
        retry_on_low_confidence = config.get('RETRY_ON_LOW_CONFIDENCE', True)
        confidence_too_low = grade.confidence < confidence_threshold if retry_on_low_confidence else False
        should_retry = grade.relevant == "no" or confidence_too_low
        new_retry_count = current_retry_count + 1 if should_retry else current_retry_count
        
        # Only return those documents that were marked as relevant by the grader
        if relevant_doc_indices:
            filtered_docs = []
            for i, doc in enumerate(retrieved_docs): # Go through all retrieved docs and filter based on LLM grading (but always keep multimodal docs (images, figures) regardless of relevance to avoid blocking image-aware generation)
                doc_type = doc.get('metadata', {}).get('type', 'text')
                # Always retain multimodal documents (images, figures) or texts explicitly marked relevant
                if doc_type in {"page_image", "figure", "evidence"} or (i + 1) in relevant_doc_indices:
                    filtered_docs.append(doc)
            print(f"Filtered {len(retrieved_docs)} docs --> {len(filtered_docs)} relevant docs (including all retrieved images)")
        else:
            # If parsing failed, keep all docs
            filtered_docs = retrieved_docs
            
        # Format the newly filtered text so that the Generator only receives relevant docs;
        # otherwise it will still use the old un-filtered 'retrieved_text' from the Query Rewriter.
        filtered_text = "\n\n".join([
            f"[Document {i+1}]:\n{doc.get('text', '')}"
            for i, doc in enumerate(filtered_docs)
        ])
        
        return {
            "retrieved_documents": filtered_docs,  # Return only those documents marked as relevant by the grader
            "retrieved_text": filtered_text,       # Return the updated prompt context for the Generator
            "grade_decision": grade.relevant, # Store the relevance decision for routing purposes
            "grade_score": grade.relevant,    # Store the same relevance decision as a score for clarity
            "grade_confidence": grade.confidence, # Store the confidence in the grading decision
            "grade_reasoning": grade.reasoning, # Store the reasoning for the grading decision
            "retry_count": new_retry_count, # Update retry count in state (increment if documents are not relevant)
            "confidence_threshold": confidence_threshold,  # Pass config to routing function
            "retry_on_low_confidence": retry_on_low_confidence,  # Pass config to routing function
            # Preserve image-related fields from query rewriter
            "detected_image_types": state.get('detected_image_types') if isinstance(state, dict) else (state.detected_image_types or []),
            "has_images": state.get('has_images') if isinstance(state, dict) else (state.has_images or False),
            "image_paths": state.get('image_paths') if isinstance(state, dict) else (state.image_paths or []),
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
    
    The generator agent decides which prompting strategy to use,
    and then uses the LLM (via generator) for final answer generation.
    Supports both text-only and image-aware generation.
    
    Args:
        agent_llm: Lightweight LLM instance for strategy decision-making 
        generator: BaselineGenerator or VisionGenerator instance for answer generation
        config: Configuration dict
        
    Returns:
        Node function for LangGraph
    """
    
    def generator_node(state: AgenticRAGState):
        """
        Generator Agent: Decide prompting strategy, and generate answer (with optional image support).
        """
        
        print(f"\n{'='*80}")
        print("Generator AGENT")
        print(f"{'='*80}")
        
        # Handle both AgenticRAGState and dict
        if isinstance(state, dict):
            question = state.get('original_question', '')
            context = state.get('retrieved_text', '')
            grade_decision = state.get('grade_decision', '')
            grade_confidence = state.get('grade_confidence', 0.0)
            detected_image_types = state.get('detected_image_types', [])
            image_paths = state.get('image_paths', [])
            retrieved_docs = state.get('retrieved_documents', [])
        else:
            question = state.original_question
            context = state.retrieved_text
            grade_decision = state.grade_decision
            grade_confidence = state.grade_confidence
            detected_image_types = state.detected_image_types or []
            image_paths = state.image_paths or []
            retrieved_docs = state.retrieved_documents or []
        
        # Build prompt for strategy selection
        has_images_context = " (IMAGES AVAILABLE)" if detected_image_types else ""
        prompt = f"""Choose a generation strategy:
Question: {question}{has_images_context}
Context available: {"yes" if context else "no"}
Documents relevant: {grade_decision}
Grader confidence: {grade_confidence:.2f} (scale 0.0-1.0)

Strategies:
1. standard - direct extraction (use when docs are highly relevant)
2. cot - step-by-step reasoning (use for complex questions)
3. few_shot - example-based (use when patterns are clear)
4. role - expert perspective (use for domain-specific topics)
5. ensemble - combined approach (use when uncertain)

If you choose 'role' strategy, also specify a role_type (pick the MOST appropriate):
- financial_analyst: for financial/numerical questions
- researcher: for evidence-based questions requiring citations
- data_analyst: for data extraction and pattern questions
- domain_expert: for field-specific interpretation
- technical_writer: for technical/structured communication

Select the best strategy based on question complexity, context quality, and grader confidence.
If grader confidence < 0.6, consider using ensemble or cot for more reasoning.

CRITICAL: Your response MUST be ONLY a valid JSON object. Do NOT include:
- Any text before the JSON
- Any text after the JSON
- Code block markers (```, ```json)
- Explanations or reasoning outside JSON

Output raw JSON only, like: {{"strategy":"standard","role_type":"financial_analyst","reasoning":"straightforward","needs_more_context":false,"confidence":0.9}}"""
        
        try:
            response = agent_llm.invoke(prompt) # Call agent LLM to decide on prompting strategy
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON
            strategy_dict = clean_and_extract_json(response_text)   
            strategy_decision = GeneratorDecision(**strategy_dict) # Validate and create Pydantic model for the strategy decision
            
        except Exception as e:
            print(f"Failed to decide strategy: {e}, using 'standard'")
            strategy_decision = GeneratorDecision( # Default to 'standard' strategy if parsing fails
                strategy="standard",
                reasoning="Error in selection, using default",
                needs_more_context=False,
                confidence=0.5
            )
        
        print(f"Chose strategy: {strategy_decision.strategy}")
        if strategy_decision.strategy == "role" and strategy_decision.role_type: 
            print(f"Role type: {strategy_decision.role_type}")
        print(f"Confidence: {strategy_decision.confidence}")
        
        # Build config for the strategy (include role_type if we're using role prompt strategy)
        strategy_config = {}
        if strategy_decision.strategy == "role":
            # Use provided role_type or default to financial_analyst
            strategy_config['role_type'] = strategy_decision.role_type or "financial_analyst"
        
        # Format rules are now handled by strategy system prompts (CoT, Standard, etc.)
        # Each strategy.generate() internally applies its own system prompt with complete format guidance.
        # This eliminates redundancy and removes instruction conflicts that caused hallucinations.
        enhanced_context = context or ""
        
        # Build the strategy once (works for both image and text generation)
        # The strategy object handles system_prompt internally via get_system_prompt()
        strategy = get_prompt_strategy(
            strategy_decision.strategy,
            generator,
            strategy_config
        )
        
        # CHECK FOR IMAGE HANDLING
        # If we have images and the generator supports vision, use image-aware generation
        use_image_generation = (
            detected_image_types and  # Only consider image generation if we detected image types in the retrieved documents
            image_paths and # Ensure we have valid image paths to pass to the generator
            isinstance(generator, VisionGenerator) and # Generator must be VisionGenerator to handle images
            hasattr(generator, 'generate_with_images') # The generator must have a method to handle image generation
        )
        
        if use_image_generation:
            print(f"Using image-aware generation (detected: {detected_image_types})")
            try:
                # Use strategy's generate_with_images(), which handles system_prompt internally
                answer = strategy.generate_with_images(
                    question=question,
                    image_paths=image_paths,
                    text_context=enhanced_context
                )
                print(f"Generated answer from images: {len(answer)} chars")
                generation_method = "vision" # Track that we used the vision generation path for analysis
            except Exception as e:
                print(f"Image generation failed: {e}, falling back to text generation")
                # Fallback to text-only generation
                answer = strategy.generate(question, enhanced_context) # This will use the same strategy but without images, allowing us to still get an answer even if image generation fails
                print(f"Generated answer from text: {len(answer)} chars")
                generation_method = "text-fallback" # Track that we attempted vision generation but had to fallback to text generation
        else:
            # Text-only generation
            answer = strategy.generate(question, enhanced_context)
            print(f"Generated answer: {len(answer)} chars")
            generation_method = "text" # Track that we used the text generation path
        
        # Get existing agent_decisions safely
        if isinstance(state, dict):
            existing_decisions = state.get('agent_decisions') or {}
        else:
            existing_decisions = state.agent_decisions or {}
        
        # ===== ANSWER FORMAT VALIDATION =====
        # Validate that answer matches expected output format (count, percentage, list, etc.)
        # IMPORTANT: Store BOTH raw and validated answers for proper evaluation metrics
        raw_answer = answer  # Save raw answer before validation
        
        is_valid_format, corrected_answer = validate_answer_format(
            answer=answer,
            question=question,
            log_issues=True
        )
        
        if not is_valid_format:
            print("Answer format validation: FAILED - using original answer")
            validation_status = "format_invalid"
            validated_answer = raw_answer  # Use raw if validation failed
        else:   
            print("Answer format validation: PASSED")
            validated_answer = corrected_answer  # Use corrected answer if valid
            validation_status = "format_valid"
        
        return {
            "retrieved_documents": retrieved_docs,  # Preserve retrieved documents
            "generated_answer": validated_answer,  # Use validated for pipeline (for grading/display)
            "raw_answer": raw_answer,  # Store raw answer for evaluation metrics
            "validated_answer": validated_answer,  # Store validated answer for evaluation metrics
            "chosen_prompting_strategy": strategy_decision.strategy,
            "generation_confidence": strategy_decision.confidence, # Confidence in the chosen strategy (not the same as grader confidence)
            "grade_confidence": grade_confidence,
            "answer_format_validation": validation_status,  # Track validation status
            "agent_decisions": {
                **existing_decisions,
                "generator": {
                    "strategy": strategy_decision.strategy,
                    "reasoning": strategy_decision.reasoning,
                    "confidence": strategy_decision.confidence,
                    "grader_confidence": grade_confidence,
                    "has_images": use_image_generation,
                    "generation_method": generation_method,
                    "num_images_used": len(image_paths) if use_image_generation else 0,
                    "answer_format_validation": validation_status
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
    Route after grading: if documents are not relevant OR confidence is low, and THERE ARE retries left, go back to rewriter.
    Otherwise, go to generator.
    
    The retry_count is incremented by the grader when it returns "no", so:
    - retry_count = 1 means "first retry attempt was just made"
    - max_retries = 1 means "allow up to 1 retry attempt"
    """
    
    # Handle both AgenticRAGState and dict
    if isinstance(state, dict):
        grade_decision = state.get('grade_decision', '') # Grader's decision on the relevance ("yes" or "no")
        grade_confidence = state.get('grade_confidence', 0.0) # Grader's confidence in the decision
        retry_count = state.get('retry_count', 0)        # Current retry count for this question
        max_retries = state.get('max_retries', 2)        # Maximum retries allowed before forcing generation
    else:
        grade_decision = state.grade_decision
        grade_confidence = state.grade_confidence
        retry_count = state.retry_count
        max_retries = state.max_retries
    
    # Check both grade_decision and confidence threshold
    # Retry if: (1) documents marked not relevant, or (2) confidence too low AND there are retries remaining
    # Note: Config values are passed via state from the grader node
    confidence_threshold = state.get('confidence_threshold', 0.6) if isinstance(state, dict) else getattr(state, 'confidence_threshold', 0.6)
    retry_on_low_confidence = state.get('retry_on_low_confidence', True) if isinstance(state, dict) else getattr(state, 'retry_on_low_confidence', True)
    confidence_too_low = grade_confidence < confidence_threshold if retry_on_low_confidence else False
    decision_negative = grade_decision == "no"
    retries_remaining = retry_count <= max_retries
    
    if ((decision_negative or confidence_too_low) and retries_remaining):
        reason = "grade=no" if decision_negative else f"confidence={grade_confidence:.2f} < {confidence_threshold}"
        print(f"Routing to query_rewriter for retry (attempt {retry_count + 1}): {reason}")
        return "query_rewriter"
    
    print(f"Routing to generator for answer generation (retries exhausted or high confidence)")
    return "generator"