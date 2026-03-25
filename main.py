"""Main example script for RAG pipeline."""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag_pipeline import RAGPipeline
from loguru import logger


def main():
    """Main example."""
    logger.info("Starting RAG Pipeline Example")
    
    # Sample documents
    documents = [
        """Machine learning is a subset of artificial intelligence that focuses on 
        the development of algorithms and statistical models that enable computers 
        to improve their performance on tasks through experience.""",
        
        """Deep learning is a subset of machine learning methods based on 
        artificial neural networks with representation learning. Learning can be 
        supervised, semi-supervised or unsupervised.""",
        
        """Natural Language Processing (NLP) is a subfield of linguistics, 
        computer science, and artificial intelligence concerned with the 
        interactions between computers and human language.""",
        
        """Retrieval-Augmented Generation (RAG) is a technique that combines 
        information retrieval with generative models to produce more accurate 
        and contextually relevant responses.""",
        
        """Vector embeddings are mathematical representations of text or other 
        data that capture semantic meaning, allowing for similarity comparisons 
        and efficient retrieval.""",
    ]
    
    try:
        # Initialize RAG pipeline
        logger.info("Initializing RAG Pipeline...")
        rag = RAGPipeline()
        
        # Add documents
        logger.info(f"Adding {len(documents)} documents...")
        rag.add_documents(documents)
        
        # Example queries
        queries = [
            "What is machine learning?",
            "How does RAG work?",
            "What are embeddings?",
        ]
        
        # Process queries
        for query in queries:
            logger.info(f"\n{'='*60}")
            logger.info(f"Query: {query}")
            logger.info(f"{'='*60}")
            
            try:
                result = rag.query(query)
                
                print(f"\nQuery: {result['query']}")
                print(f"\nAnswer:\n{result['answer']}")
                print(f"\nRetrieved Documents:")
                for i, doc in enumerate(result['retrieved_docs'], 1):
                    print(f"\n[{i}] (Score: {doc['score']:.3f})")
                    print(f"    {doc['text'][:200]}...")
            
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"Error: {e}")
        
        logger.info("\nExample completed successfully!")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
