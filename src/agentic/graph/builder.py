"""Build the LangGraph StateGraph for agentic RAG."""

import logging
from typing import Literal, Dict, Any

from langgraph.graph import StateGraph, START, END
from .state import AgenticRAGState
from .nodes import (
    make_query_rewriter_node,
    make_grader_node,
    make_generator_node,
    route_after_grading
)

logger = logging.getLogger(__name__)


def build_agentic_graph(
    agent_llm,
    embedder,
    retriever,
    generator,
    query_techniques_dict,
    config: Dict[str, Any] = None
):
    """
    Build the agentic RAG graph using LangGraph.
    
    Args:
        agent_llm: Lightweight LLM instance for agent decisions (Query Rewriter, Grader, Generator strategy)
        embedder: TextEmbedder instance
        retriever: HybridRetriever instance
        generator: BaselineGenerator instance (uses heavy LLM for answer generation)
        query_techniques_dict: Dict of QueryTechnique instances keyed by name
        config: Configuration dict
        
    Returns:
        Compiled LangGraph StateGraph
    """
    # If no config provided, use empty dict (nodes will handle defaults)
    if config is None:
        config = {}
    
    logger.info("Building agentic RAG graph...")
    
    # Create graph
    graph = StateGraph(AgenticRAGState)
    
    # Create node factories (closures that capture dependencies)
    query_rewriter = make_query_rewriter_node(agent_llm, query_techniques_dict, config)
    grader = make_grader_node(agent_llm, config)
    generator_node = make_generator_node(agent_llm, generator, config)
    
    # Add nodes to the graph
    graph.add_node("query_rewriter", query_rewriter)
    graph.add_node("grader", grader)
    graph.add_node("generator", generator_node)
    
    # Add edges, with conditional routing after grader
    graph.add_edge(START, "query_rewriter") # Start -> query_rewriter    
    graph.add_edge("query_rewriter", "grader") # query_rewriter -> grader
    graph.add_conditional_edges( # grader -> (conditional routing)
        "grader",
        route_after_grading,
        {
            "query_rewriter": "query_rewriter",
            "generator": "generator"
        }
    )
    graph.add_edge("generator", END) # generator -> END
    
    # Compile with async support
    compiled_graph = graph.compile()
    
    logger.info("Agentic RAG graph built successfully")
    
    return compiled_graph
