"""Vector store abstraction using Chroma."""

from typing import List, Dict, Tuple, Optional
import chromadb
from loguru import logger
from config import settings
import numpy as np
import os


class VectorStore:
    """Vector store using Chroma for similarity search."""
    
    def __init__(self, persist_directory: str = None, collection_name: str = None):
        """Initialize vector store.
        
        Args:
            persist_directory: Directory to persist vector store
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory or settings.VECTORSTORE_PATH
        self.collection_name = collection_name or settings.COLLECTION_NAME
        
        logger.info(f"Initializing Chroma with persist directory: {self.persist_directory}")
        
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize Chroma client with persistence (new API)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"Vector store initialized. Collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None, 
                      embeddings: np.ndarray = None):
        """Add documents to vector store.
        
        Args:
            texts: List of document texts
            metadatas: Optional metadata for each document
            embeddings: Optional precomputed embeddings
        """
        if not texts:
            logger.warning("Empty texts list provided to add_documents()")
            return
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{"id": i} for i in range(len(texts))]
        
        # Generate IDs
        ids = [f"doc_{i}" for i in range(len(texts))]
        
        try:
            if embeddings is not None:
                # Add with precomputed embeddings
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings.tolist()
                )
            else:
                # Chroma will compute embeddings automatically
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
            logger.info(f"Added {len(texts)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10,
               where: Dict = None) -> List[Tuple[str, float, Dict]]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            where: Optional filter metadata
            
        Returns:
            List of tuples (text, score, metadata)
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where
            )
            
            # Format results
            documents = results["documents"][0] if results["documents"] else []
            distances = results["distances"][0] if results["distances"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            
            # Convert distances to similarity scores (cosine distance to similarity)
            # For cosine distance, similarity = 1 - distance
            similarities = [1 - float(d) for d in distances]
            
            formatted_results = [
                (doc, score, meta)
                for doc, score, meta in zip(documents, similarities, metadatas)
            ]
            
            logger.debug(f"Vector search returned {len(formatted_results)} results")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
    
    def get_document_count(self) -> int:
        """Get number of documents in collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    def delete_collection(self):
        """Delete the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def clear_collection(self):
        """Clear all documents from collection."""
        try:
            count = self.collection.count()
            if count > 0:
                all_ids = self.collection.get()["ids"]
                self.collection.delete(ids=all_ids)
            logger.info("Collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
    
    def persist(self):
        """Persist the vector store to disk."""
        try:
            self.client.persist()
            logger.info("Vector store persisted to disk")
        except Exception as e:
            logger.warning(f"Error persisting vector store: {e}")
