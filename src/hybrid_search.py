"""Hybrid search combining BM25 and vector search."""

from typing import List, Tuple, Dict, Optional
from loguru import logger
from config import settings
from src.bm25_search import BM25Search
from src.vectorstore import VectorStore
from src.embeddings import get_embedding_generator
import numpy as np


class HybridSearch:
    """Hybrid search combining BM25 (sparse) and vector (dense) retrieval."""
    
    def __init__(self, vector_weight: float = None, bm25_weight: float = None):
        """Initialize hybrid search.
        
        Args:
            vector_weight: Weight for vector search (0-1)
            bm25_weight: Weight for BM25 search (0-1)
        """
        self.vector_weight = vector_weight or settings.VECTOR_WEIGHT
        self.bm25_weight = bm25_weight or settings.BM25_WEIGHT
        
        # Normalize weights
        total = self.vector_weight + self.bm25_weight
        self.vector_weight = self.vector_weight / total
        self.bm25_weight = self.bm25_weight / total
        
        logger.info(f"Hybrid search initialized with vector_weight={self.vector_weight:.2f}, "
                   f"bm25_weight={self.bm25_weight:.2f}")
        
        # Initialize components
        self.vector_store = VectorStore()
        self.bm25_search = BM25Search()
        self.embedding_generator = get_embedding_generator()
        
        self.documents = []
        self.metadatas = []
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """Add documents to both vector and BM25 indexes.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
        """
        if not documents:
            logger.warning("Empty documents list provided to add_documents()")
            return
        
        self.documents = documents
        self.metadatas = metadatas or [{"id": i} for i in range(len(documents))]
        
        logger.info(f"Adding {len(documents)} documents to hybrid search...")
        
        try:
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_generator.embed(
                documents,
                show_progress_bar=True
            )
            
            # Add to vector store
            logger.info("Adding to vector store...")
            self.vector_store.add_documents(documents, self.metadatas, embeddings)
            
            # Add to BM25
            logger.info("Building BM25 index...")
            self.bm25_search.add_documents(documents, self.metadatas)
            
            logger.info(f"Successfully added {len(documents)} documents to hybrid search")
        except Exception as e:
            logger.error(f"Error adding documents to hybrid search: {e}")
            raise
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[str, float, Dict]]:
        """Perform hybrid search.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of tuples (text, score, metadata) sorted by combined score
        """
        top_k = top_k or settings.TOP_K_RETRIEVAL
        
        # Check if documents exist in vector store (they may be persisted)
        vector_store_count = self.vector_store.get_document_count()
        if not self.documents and vector_store_count == 0:
            logger.warning("No documents available for search")
            return []
        
        try:
            # Parallel search
            logger.debug(f"Performing hybrid search for query: {query[:100]}...")
            
            # Vector search
            query_embedding = self.embedding_generator.embed_query(query)
            vector_results = self.vector_store.search(query_embedding, top_k=top_k)
            
            # BM25 search (only if we have in-memory documents)
            if self.documents:
                bm25_results = self.bm25_search.search(query, top_k=top_k)
            else:
                bm25_results = []
            
            # Combine results with weighted scoring
            combined_scores: Dict[str, Tuple[str, Dict, float]] = {}
            
            # Add vector search results
            for doc, score, metadata in vector_results:
                doc_id = metadata.get("id", doc)
                if doc_id not in combined_scores:
                    combined_scores[doc_id] = (doc, metadata, 0.0)
                _, meta, combined_score = combined_scores[doc_id]
                combined_scores[doc_id] = (doc, meta, combined_score + score * self.vector_weight)
            
            # Add BM25 results
            for doc, score, metadata in bm25_results:
                doc_id = metadata.get("id", doc)
                if doc_id not in combined_scores:
                    combined_scores[doc_id] = (doc, metadata, 0.0)
                _, meta, combined_score = combined_scores[doc_id]
                combined_scores[doc_id] = (doc, meta, combined_score + score * self.bm25_weight)
            
            # Sort by combined score and return top-k
            sorted_results = sorted(
                combined_scores.values(),
                key=lambda x: x[2],
                reverse=True
            )[:top_k]
            
            logger.debug(f"Hybrid search returned {len(sorted_results)} results")
            return sorted_results
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            raise
    
    def get_document_count(self) -> int:
        """Get number of documents in hybrid search."""
        return self.vector_store.get_document_count()
    
    def clear(self):
        """Clear both indexes."""
        self.vector_store.clear_collection()
        self.bm25_search.reset()
        self.documents = []
        self.metadatas = []
        logger.info("Hybrid search cleared")
