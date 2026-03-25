"""Embedding generation module."""

import hashlib
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
from config import settings
import os

# Suppress transformers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EmbeddingGenerator:
    """Generate embeddings using SentenceTransformers."""
    
    def __init__(self, model_name: str = None):
        """Initialize embedding generator.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded successfully. Dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def embed(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            show_progress_bar: Show progress bar
            
        Returns:
            Array of embeddings
        """
        if not texts:
            logger.warning("Empty text list provided to embed()")
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=show_progress_bar,
                batch_size=settings.BATCH_SIZE,
                convert_to_numpy=True
            )
            logger.debug(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return self.embed([query])[0]
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim


def _get_embedding_hash(texts: List[str]) -> str:
    """Generate hash for embeddings cache key."""
    combined = "".join(sorted(texts))
    return hashlib.md5(combined.encode()).hexdigest()


# Global embedding generator instance
_embedding_generator: EmbeddingGenerator = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create embedding generator instance (singleton)."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator
