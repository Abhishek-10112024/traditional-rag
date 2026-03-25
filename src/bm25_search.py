"""BM25 Search implementation."""

from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from loguru import logger
import re
from dataclasses import dataclass


@dataclass
class BM25Result:
    """BM25 search result."""
    text: str
    score: float
    metadata: Dict = None


class BM25Search:
    """BM25 search engine for sparse retrieval."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 search.
        
        Args:
            k1: BM25 parameter for term frequency saturation
            b: BM25 parameter for length normalization
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        self.metadata = []
        logger.info("BM25 search initialized")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25.
        
        Args:
            text: Text to preprocess
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Tokenize
        tokens = text.split()
        return tokens
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to BM25 index.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        if not documents:
            logger.warning("Empty documents list provided to add_documents()")
            return
        
        self.documents = documents
        self.metadata = metadata or [{"id": i} for i in range(len(documents))]
        
        # Tokenize documents
        tokenized_docs = [self._preprocess_text(doc) for doc in documents]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)
        logger.info(f"BM25 index created with {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for documents similar to query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of tuples (document, score)
        """
        if self.bm25 is None:
            logger.warning("BM25 index not initialized. Add documents first.")
            return []
        
        # Tokenize query
        tokenized_query = self._preprocess_text(query)
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = [
            (self.documents[i], float(scores[i]), self.metadata[i])
            for i in top_indices
            if scores[i] > 0
        ]
        
        logger.debug(f"BM25 search returned {len(results)} results")
        return results
    
    def reset(self):
        """Reset the BM25 index."""
        self.bm25 = None
        self.documents = []
        self.metadata = []
        logger.info("BM25 index reset")
