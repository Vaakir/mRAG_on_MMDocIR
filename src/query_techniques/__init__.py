from typing import Dict, Any
from .base import QueryTechnique
from .standard import StandardRetrieval
from .multi_query import MultiQueryRetrieval
from .rag_fusion import RAGFusionRetrieval
from .step_back import StepBackRetrieval
from .hyde import HyDERetrieval
from .query_decomposition import QueryDecompositionRetrieval
from .query_rewriting import QueryRewritingRetrieval
from .query_expansion import QueryExpansionRetrieval


def get_query_technique(
    technique_name: str,
    embedder,
    retriever,
    generator,
    config: Dict[str, Any] = None
) -> QueryTechnique:
    """
    Factory function to get the appropriate query technique based on name.
    
    Args:
        technique_name: Name of the technique ('standard', 'multi_query', 'rag_fusion', etc.)
        embedder: TextEmbedder instance
        retriever: HybridRetriever instance
        generator: BaselineGenerator instance
        config: Configuration dict for the technique
    
    Returns:
        QueryTechnique subclass instance
    
    Raises:
        ValueError: If technique name is not recognized
    """
    techniques = {
        'standard': StandardRetrieval,
        'multi_query': MultiQueryRetrieval,
        'rag_fusion': RAGFusionRetrieval,
        'step_back': StepBackRetrieval,
        'hyde': HyDERetrieval,
        'query_decomposition': QueryDecompositionRetrieval,
        'query_rewriting': QueryRewritingRetrieval,
        'query_expansion': QueryExpansionRetrieval,
    }
    
    technique_name = technique_name.lower().strip()
    
    if technique_name not in techniques:
        available = ', '.join(techniques.keys())
        raise ValueError(f"Unknown query technique '{technique_name}'. Available techniques: {available}")
    
    technique_class = techniques[technique_name]
    return technique_class(embedder, retriever, generator, config or {})


__all__ = [
    'get_query_technique',
    'StandardRetrieval',
    'MultiQueryRetrieval',
    'RAGFusionRetrieval',
    'StepBackRetrieval',
    'HyDERetrieval',
    'QueryDecompositionRetrieval',
    'QueryRewritingRetrieval',
    'QueryExpansionRetrieval',
]
