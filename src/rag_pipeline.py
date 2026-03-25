"""Main RAG Pipeline."""

from typing import List, Dict, Tuple, Optional
from loguru import logger
from config import settings
from src.hybrid_search import HybridSearch
from src.reranker import get_reranker
from src.llm import get_llm_instance
import hashlib
import pickle
import json
from pathlib import Path


class RAGPipeline:
    """Complete RAG (Retrieval-Augmented Generation) pipeline."""
    
    def __init__(self, use_reranking: bool = None, use_hybrid_search: bool = None):
        """Initialize RAG pipeline.
        
        Args:
            use_reranking: Whether to use reranking
            use_hybrid_search: Whether to use hybrid search
        """
        self.use_reranking = use_reranking if use_reranking is not None else settings.RERANKING_ENABLED
        self.use_hybrid_search = use_hybrid_search if use_hybrid_search is not None else settings.HYBRID_SEARCH_ENABLED
        
        logger.info(f"Initializing RAG Pipeline (reranking={self.use_reranking}, "
                   f"hybrid_search={self.use_hybrid_search})")
        
        # Initialize components
        self.hybrid_search = HybridSearch() if self.use_hybrid_search else None
        self.reranker = get_reranker() if self.use_reranking else None
        self.llm = get_llm_instance()
        
        # Cache for responses
        self.cache = {}
        self.cache_enabled = settings.CACHE_ENABLED
        
        logger.info("RAG Pipeline initialized successfully")
    
    def _get_cache_key(self, query: str, context: str) -> str:
        """Generate cache key for query+context.
        
        Args:
            query: Query text
            context: Context text
            
        Returns:
            Cache key hash
        """
        combined = f"{query}|||{context}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
        return prompt
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """Add documents to RAG pipeline.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for documents
        """
        if not documents:
            logger.warning("Empty documents list provided to add_documents()")
            return
        
        logger.info(f"Adding {len(documents)} documents to RAG pipeline...")
        
        try:
            if self.use_hybrid_search and self.hybrid_search:
                self.hybrid_search.add_documents(documents, metadatas)
            
            logger.info(f"Successfully added {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float, Dict]]:
        """Retrieve relevant documents for query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of tuples (text, score, metadata)
        """
        top_k = top_k or settings.TOP_K_RETRIEVAL
        
        try:
            logger.debug(f"Retrieving {top_k} documents for query: {query[:100]}...")
            
            if not self.use_hybrid_search or not self.hybrid_search:
                logger.warning("Hybrid search not enabled")
                return []
            
            # Retrieve documents
            retrieved = self.hybrid_search.search(query, top_k=top_k)
            
            logger.debug(f"Retrieved {len(retrieved)} documents")
            return retrieved
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise
    
    def rerank_documents(self,
                        query: str,
                        documents: List[str],
                        metadatas: List[Dict] = None,
                        top_k: int = None) -> List[Tuple[str, float, Dict]]:
        """Rerank documents using cross-encoder.
        
        Args:
            query: Query text
            documents: Documents to rerank
            metadatas: Document metadata
            top_k: Number of results to return
            
        Returns:
            List of tuples (text, score, metadata)
        """
        if not self.use_reranking or not self.reranker:
            logger.warning("Reranking not enabled")
            return [(d, 1.0, m) for d, m in zip(documents, metadatas or [{"id": i} for i in range(len(documents))])]
        
        try:
            reranked = self.reranker.rerank(query, documents, metadatas, top_k=top_k)
            logger.debug(f"Reranked to {len(reranked)} documents")
            return reranked
        except Exception as e:
            logger.error(f"Error reranking documents: {e}")
            raise
    
    def generate(self, context: str, query: str, use_cache: bool = True) -> str:
        """Generate answer using LLM.
        
        Args:
            context: Retrieved context
            query: User query
            use_cache: Whether to use cache
            
        Returns:
            Generated answer
        """
        try:
            # Check cache
            if use_cache and self.cache_enabled:
                cache_key = self._get_cache_key(query, context)
                if cache_key in self.cache:
                    logger.debug("Cache hit!")
                    return self.cache[cache_key]
            
            # Build prompt
            prompt = self._build_prompt(query, context)
            
            # Generate
            logger.debug("Generating answer with LLM...")
            answer = self.llm.generate(prompt)
            
            # Cache result
            if use_cache and self.cache_enabled:
                cache_key = self._get_cache_key(query, context)
                self.cache[cache_key] = answer
            
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def query(self, query: str, top_k_retrieval: int = None, top_k_rerank: int = None) -> Dict:
        """Full RAG query pipeline.
        
        Args:
            query: User query
            top_k_retrieval: Number of documents to retrieve
            top_k_rerank: Number of documents after reranking
            
        Returns:
            Dictionary with query, context, and answer
        """
        top_k_retrieval = top_k_retrieval or settings.TOP_K_RETRIEVAL
        top_k_rerank = top_k_rerank or settings.TOP_K_RERANK
        
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Step 1: Retrieve
            retrieved = self.retrieve(query, top_k=top_k_retrieval)
            
            if not retrieved:
                logger.warning("No documents retrieved")
                return {
                    "query": query,
                    "context": "",
                    "retrieved_docs": [],
                    "answer": "No relevant documents found.",
                }
            
            # Step 2: Rerank
            if self.use_reranking:
                documents = [doc for doc, _, _ in retrieved]
                metadatas = [meta for _, _, meta in retrieved]
                reranked = self.rerank_documents(query, documents, metadatas, top_k=top_k_rerank)
            else:
                reranked = retrieved[:top_k_rerank]
            
            # Step 3: Build context
            context = "\n\n".join([f"[{i+1}] {doc}" for i, (doc, _, _) in enumerate(reranked)])
            
            # Step 4: Generate
            answer = self.generate(context, query)
            
            logger.info("Query processing completed")
            
            return {
                "query": query,
                "context": context,
                "retrieved_docs": [{"text": doc, "score": score, "metadata": meta} 
                                  for doc, score, meta in reranked],
                "answer": answer,
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def clear_cache(self):
        """Clear response cache."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def save_documents(self, filepath: str):
        """Save indexed documents to file.
        
        Args:
            filepath: Path to save file
        """
        try:
            if self.hybrid_search:
                data = {
                    "documents": self.hybrid_search.documents,
                    "metadatas": self.hybrid_search.metadatas,
                }
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f"Documents saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving documents: {e}")
            raise
    
    def load_documents(self, filepath: str):
        """Load documents from file.
        
        Args:
            filepath: Path to load file
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.add_documents(data["documents"], data["metadatas"])
            logger.info(f"Documents loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
