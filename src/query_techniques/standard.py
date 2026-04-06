from typing import Any, Dict, List
from .base import QueryTechnique


class StandardRetrieval(QueryTechnique):
    """
    Baseline retrieval strategy.

    Passes the input question directly to the retriever without any
    query transformation.
    """

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k results for the given question.

        Args:
            question: Input query
            top_k: Number of results to return

        Returns:
            List of retrieved chunks with id, text, score, and metadata.
        """

        # Direct retrieval with no query modification
        return self.retriever.retrieve(question, top_k=top_k)