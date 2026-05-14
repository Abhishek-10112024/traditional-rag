"""Configuration file for RAG application."""

import os
from pathlib import Path
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings
from typing import Optional

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
CACHE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


class Settings(BaseSettings):
    """Application settings."""
    
    # Model configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Fast, free embedding model
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Free reranker
    
    # LLM Configuration
    LLM_TYPE: str = "ollama"  # Options: "ollama" (local), "huggingface" (cloud), "transformers" (fallback)
    
    # Ollama (local only, requires Ollama running on localhost:11434)
    OLLAMA_MODEL: str = "mistral"  # Smaller, faster model (2GB vs 4GB for mistral)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # HuggingFace Inference API (cloud-based, works on HuggingFace Spaces)
    HUGGINGFACE_MODEL: str = "microsoft/phi-2"  # Lightweight, works with free HF Inference API
    HUGGINGFACE_API_TOKEN: str = os.getenv("HF_TOKEN", "")  # Set via environment variable
    
    # Vector Store Configuration
    VECTORSTORE_TYPE: str = "chroma"  # Options: "chroma", "faiss"
    VECTORSTORE_PATH: str = str(CACHE_DIR / "vectorstore")
    COLLECTION_NAME: str = "documents"
    
    # Search Configuration
    HYBRID_SEARCH_ENABLED: bool = True
    VECTOR_WEIGHT: float = 0.8
    BM25_WEIGHT: float = 0.2
    TOP_K_RETRIEVAL: int = 20  # Higher recall for long research documents
    TOP_K_RERANK: int = 8      # Keep enough context after reranking
    
    # Reranking Configuration
    RERANKING_ENABLED: bool = True
    RERANKER_BATCH_SIZE: int = 32
    RERANKER_SCORE_THRESHOLD: float = 0.05
    
    # Chunking Configuration
    CHUNK_SIZE: int = 900      # Word-based chunks for research-paper style documents
    CHUNK_OVERLAP: int = 150
    
    # Caching Configuration
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 1800  # 30 minutes in seconds
    
    # Batch Processing
    BATCH_SIZE: int = 32
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = str(LOGS_DIR / "app.log")
    
    # API Configuration
    MAX_CONTEXT_LENGTH: int = 5000
    TEMPERATURE: float = 0.25
    TOP_P: float = 0.9
    MAX_NEW_TOKENS: int = 800
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance."""
    return settings
