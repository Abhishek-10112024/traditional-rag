"""RAGAS evaluation framework for RAG systems."""

from typing import List, Dict, Tuple
from loguru import logger
import json
from pathlib import Path
import pandas as pd

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not installed. Install with: pip install ragas")

from datasets import Dataset


class RAGEvaluator:
    """Evaluate RAG system using RAGAS metrics."""
    
    def __init__(self):
        """Initialize evaluator."""
        if not RAGAS_AVAILABLE:
            logger.warning("RAGAS not available. Some metrics will be skipped.")
        
        logger.info("RAG Evaluator initialized")
    
    def evaluate_response(self,
                         query: str,
                         context: str,
                         ground_truth: str,
                         generated_answer: str) -> Dict:
        """Evaluate a single query-response pair.
        
        Args:
            query: Query text
            context: Retrieved context
            ground_truth: Expected answer
            generated_answer: LLM generated answer
            
        Returns:
            Dictionary with evaluation scores
        """
        if not RAGAS_AVAILABLE:
            logger.warning("Cannot evaluate: RAGAS not installed")
            return {}
        
        try:
            logger.info(f"Evaluating response for query: {query[:50]}...")
            
            # Create dataset
            data = {
                "question": [query],
                "contexts": [[line.strip() for line in context.split("\n") if line.strip()]],
                "answer": [generated_answer],
                "ground_truth": [ground_truth],
            }
            
            dataset = Dataset.from_dict(data)
            
            # Evaluate
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ]
            )
            
            # Extract scores
            scores = {
                "faithfulness": float(result["faithfulness"][0]) if "faithfulness" in result else 0.0,
                "answer_relevancy": float(result["answer_relevancy"][0]) if "answer_relevancy" in result else 0.0,
                "context_precision": float(result["context_precision"][0]) if "context_precision" in result else 0.0,
                "context_recall": float(result["context_recall"][0]) if "context_recall" in result else 0.0,
            }
            
            # Calculate average
            scores["average"] = sum(scores.values()) / len(scores)
            
            logger.debug(f"Evaluation scores: {scores}")
            return scores
        
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return {}
    
    def evaluate_batch(self,
                      queries: List[str],
                      contexts: List[str],
                      ground_truths: List[str],
                      generated_answers: List[str]) -> Dict:
        """Evaluate multiple query-response pairs.
        
        Args:
            queries: List of queries
            contexts: List of contexts
            ground_truths: List of expected answers
            generated_answers: List of generated answers
            
        Returns:
            Dictionary with evaluation results
        """
        if not RAGAS_AVAILABLE:
            logger.warning("Cannot evaluate: RAGAS not installed")
            return {}
        
        if not (len(queries) == len(contexts) == len(ground_truths) == len(generated_answers)):
            raise ValueError("All input lists must have the same length")
        
        try:
            logger.info(f"Batch evaluating {len(queries)} query-response pairs...")
            
            # Create dataset
            data = {
                "question": queries,
                "contexts": [
                    [line.strip() for line in ctx.split("\n") if line.strip()]
                    for ctx in contexts
                ],
                "answer": generated_answers,
                "ground_truth": ground_truths,
            }
            
            dataset = Dataset.from_dict(data)
            
            # Evaluate
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ]
            )
            
            # Extract results
            results = {
                "queries": queries,
                "faithfulness": result["faithfulness"].tolist(),
                "answer_relevancy": result["answer_relevancy"].tolist(),
                "context_precision": result["context_precision"].tolist(),
                "context_recall": result["context_recall"].tolist(),
                "summary": {
                    "avg_faithfulness": float(result["faithfulness"].mean()),
                    "avg_answer_relevancy": float(result["answer_relevancy"].mean()),
                    "avg_context_precision": float(result["context_precision"].mean()),
                    "avg_context_recall": float(result["context_recall"].mean()),
                }
            }
            
            logger.info(f"Batch evaluation completed. Average scores: {results['summary']}")
            return results
        
        except Exception as e:
            logger.error(f"Error in batch evaluation: {e}")
            return {}
    
    def save_results(self, results: Dict, filepath: str):
        """Save evaluation results to file.
        
        Args:
            results: Evaluation results
            filepath: Path to save file
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def create_report(self, results: Dict) -> str:
        """Create a human-readable evaluation report.
        
        Args:
            results: Evaluation results
            
        Returns:
            Formatted report string
        """
        if not results or "summary" not in results:
            return "No evaluation results available"
        
        summary = results["summary"]
        
        report = f"""
RAG System Evaluation Report
============================

Metrics Summary:
- Faithfulness (Ground Truth Consistency):        {summary['avg_faithfulness']:.3f}
- Answer Relevancy (Query Relevance):             {summary['avg_answer_relevancy']:.3f}
- Context Precision (Correct Context):            {summary['avg_context_precision']:.3f}
- Context Recall (Complete Context):              {summary['avg_context_recall']:.3f}

Overall Score: {(sum([summary['avg_faithfulness'], summary['avg_answer_relevancy'], summary['avg_context_precision'], summary['avg_context_recall']]) / 4):.3f}

Interpretation:
- Faithfulness: How factually consistent the answer is with the context
- Answer Relevancy: How relevant the generated answer is to the query
- Context Precision: How precise the retrieved context is (no irrelevant content)
- Context Recall: How complete the retrieved context is (no missing relevant content)
"""
        return report


# Global evaluator instance
_evaluator: RAGEvaluator = None


def get_evaluator() -> RAGEvaluator:
    """Get or create evaluator instance (singleton)."""
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGEvaluator()
    return _evaluator
