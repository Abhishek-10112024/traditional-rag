"""Ingest documents into the RAG system."""

import argparse
import sys
from pathlib import Path
from loguru import logger

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag_pipeline import RAGPipeline
from config import settings


def load_documents(file_paths: list) -> list:
    """Load documents from files.
    
    Args:
        file_paths: List of file paths
        
    Returns:
        List of documents
    """
    documents = []
    
    for file_path in file_paths:
        try:
            content = None
            # Try multiple encodings for robust file reading
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            
            if content is None:
                logger.warning(f"Could not decode {file_path} with any supported encoding - skipping")
                continue
            
            # Split by paragraphs
            chunks = [p.strip() for p in content.split("\n\n") if p.strip()]
            documents.extend(chunks)
            logger.info(f"Loaded {len(chunks)} chunks from {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    return documents


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ingest documents into RAG system")
    parser.add_argument("files", nargs="+", help="File paths to ingest")
    parser.add_argument("--output", default="cache/indexed_docs.pkl", help="Output file path")
    
    args = parser.parse_args()
    
    try:
        logger.info(f"Loading documents from {len(args.files)} files...")
        documents = load_documents(args.files)
        
        if not documents:
            logger.error("No documents loaded")
            return
        
        logger.info(f"Total chunks: {len(documents)}")
        
        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline()
        
        # Add documents
        logger.info("Indexing documents...")
        rag_pipeline.add_documents(documents)
        
        # Save
        logger.info(f"Saving indexed documents to {args.output}...")
        rag_pipeline.save_documents(args.output)
        
        logger.info("✅ Document ingestion completed successfully!")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
