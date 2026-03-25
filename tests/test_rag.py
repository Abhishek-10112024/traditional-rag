"""Basic tests for RAG pipeline."""

import sys
from pathlib import Path
import pytest

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings import EmbeddingGenerator
from src.bm25_search import BM25Search
from src.vectorstore import VectorStore


class TestEmbeddings:
    """Test embeddings module."""
    
    def test_embedding_generation(self):
        """Test embedding generation."""
        generator = EmbeddingGenerator()
        
        texts = ["Hello world", "How are you"]
        embeddings = generator.embed(texts)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == generator.embedding_dim
    
    def test_single_embedding(self):
        """Test single text embedding."""
        generator = EmbeddingGenerator()
        
        embedding = generator.embed_query("Test query")
        
        assert len(embedding) == generator.embedding_dim


class TestBM25:
    """Test BM25 search."""
    
    def test_bm25_indexing(self):
        """Test BM25 document indexing."""
        bm25 = BM25Search()
        
        documents = [
            "The quick brown fox jumps over the lazy dog",
            "Python is a programming language",
            "Machine learning is powerful"
        ]
        
        bm25.add_documents(documents)
        
        assert bm25.bm25 is not None
    
    def test_bm25_search(self):
        """Test BM25 search functionality."""
        bm25 = BM25Search()
        
        documents = [
            "The quick brown fox jumps over the lazy dog",
            "Python is a programming language",
            "Machine learning is powerful"
        ]
        
        bm25.add_documents(documents)
        results = bm25.search("python programming", top_k=2)
        
        assert len(results) > 0


class TestVectorStore:
    """Test vector store."""
    
    def test_vectorstore_add_documents(self):
        """Test adding documents to vector store."""
        vs = VectorStore()
        
        documents = [
            "Test document 1",
            "Test document 2"
        ]
        
        vs.add_documents(documents)
        
        assert vs.get_document_count() > 0
    
    def test_vectorstore_cleanup(self):
        """Test vector store cleanup."""
        vs = VectorStore()
        vs.clear_collection()
        
        assert vs.get_document_count() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
