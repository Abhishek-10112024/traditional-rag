"""Reranking module using cross-encoders."""

from typing import List, Tuple, Dict, Optional
from sentence_transformers import CrossEncoder
from loguru import logger
from config import settings
import os

# Suppress transformers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Reranker:
    """Rerank documents using cross-encoder models."""
    
    def __init__(self, model_name: str = None):
        """Initialize reranker.
        
        Args:
            model_name: Name of the cross-encoder model
        """
        self.model_name = model_name or settings.RERANKER_MODEL
        logger.info(f"Loading reranker model: {self.model_name}")
        
        try:
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Reranker model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading reranker model: {e}")
            raise
    
    def rerank(self,
               query: str,
               documents: List[str],
               metadatas: List[Dict] = None,
               top_k: int = None,
               batch_size: int = None,
               score_threshold: float = None) -> List[Tuple[str, float, Dict]]:
        """Rerank documents based on relevance to query.
        
        Args:
            query: Query text
            documents: List of document texts
            metadatas: Optional metadata for documents
            top_k: Number of results to return
            batch_size: Batch size for processing
            score_threshold: Minimum score threshold
            
        Returns:
            List of tuples (text, score, metadata) sorted by score
        """
        if not documents:
            logger.warning("Empty documents list provided to rerank()")
            return []
        
        top_k = top_k or settings.TOP_K_RERANK
        batch_size = batch_size or settings.RERANKER_BATCH_SIZE
        score_threshold = score_threshold if score_threshold is not None else settings.RERANKER_SCORE_THRESHOLD
        
        if metadatas is None:
            metadatas = [{"id": i} for i in range(len(documents))]
        
        try:
            logger.info(f"Reranking {len(documents)} documents...")
            
            # Prepare query-document pairs
            pairs = [[query, doc] for doc in documents]
            
            # Score documents
            scores = self.model.predict(pairs, batch_size=batch_size, convert_to_numpy=True)
            
            # Combine with metadata
            results = [
                (doc, float(score), meta)
                for doc, score, meta in zip(documents, scores, metadatas)
            ]
            
            # Filter by threshold
            results = [(d, s, m) for d, s, m in results if s >= score_threshold]
            
            # Sort by score descending
            results = sorted(results, key=lambda x: x[1], reverse=True)
            
            # Return top-k
            # If nothing passes threshold → fallback BEFORE slicing
            if len(results) == 0:
                logger.warning("Reranker returned 0 results → fallback to top documents without threshold")
                
                # Recompute without threshold
                results = [
                    (doc, float(score), meta)
                    for doc, score, meta in zip(documents, scores, metadatas)
                ]
                
                results = sorted(results, key=lambda x: x[1], reverse=True)

            # Return top-k (AFTER fallback)
            results = results[:top_k]

            logger.debug(f"Reranking returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error reranking documents: {e}")
            raise


# Global reranker instance
_reranker: Reranker = None


def get_reranker() -> Reranker:
    """Get or create reranker instance (singleton)."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
