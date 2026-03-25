"""Additional evaluation metrics for RAG systems."""

from typing import List, Dict, Tuple
from loguru import logger
import numpy as np
from difflib import SequenceMatcher


class AdditionalMetrics:
    """Additional metrics beyond RAGAS."""
    
    @staticmethod
    def bleu_score(reference: str, generated: str) -> float:
        """Calculate BLEU-like similarity score.
        
        Args:
            reference: Reference text
            generated: Generated text
            
        Returns:
            Similarity score (0-1)
        """
        try:
            ratio = SequenceMatcher(None, reference.lower(), generated.lower()).ratio()
            return float(ratio)
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            return 0.0
    
    @staticmethod
    def retrieval_precision(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate retrieval precision.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_docs: List of relevant documents
            
        Returns:
            Precision score (0-1)
        """
        if not retrieved_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        retrieved_relevant = sum(1 for doc in retrieved_docs if doc in relevant_set)
        
        return retrieved_relevant / len(retrieved_docs)
    
    @staticmethod
    def retrieval_recall(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate retrieval recall.
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_docs: List of relevant documents
            
        Returns:
            Recall score (0-1)
        """
        if not relevant_docs:
            return 0.0
        
        retrieved_set = set(retrieved_docs)
        relevant_found = sum(1 for doc in relevant_docs if doc in retrieved_set)
        
        return relevant_found / len(relevant_docs)
    
    @staticmethod
    def mrr_score(ranked_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Mean Reciprocal Rank.
        
        Args:
            ranked_docs: Ranked list of documents
            relevant_docs: List of relevant documents
            
        Returns:
            MRR score
        """
        relevant_set = set(relevant_docs)
        
        for i, doc in enumerate(ranked_docs, 1):
            if doc in relevant_set:
                return 1.0 / i
        
        return 0.0
    
    @staticmethod
    def ndcg_score(ranked_scores: List[float], relevant_indices: List[int]) -> float:
        """Calculate Normalized Discounted Cumulative Gain.
        
        Args:
            ranked_scores: Relevance scores of ranked documents
            relevant_indices: Indices of relevant documents
            
        Returns:
            NDCG score
        """
        dcg = 0.0
        idcg = 0.0
        
        # Calculate DCG
        for i, score in enumerate(ranked_scores, 1):
            if i - 1 in relevant_indices:
                dcg += score / np.log2(i + 1)
        
        # Calculate IDCG (ideal ranking)
        ideal_scores = sorted(ranked_scores, reverse=True)
        for i, score in enumerate(ideal_scores, 1):
            if len(relevant_indices) > 0:
                idcg += 1.0 / np.log2(i + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def latency_score(response_time: float, threshold: float = 3.0) -> float:
        """Calculate latency score.
        
        Args:
            response_time: Response time in seconds
            threshold: Target time threshold in seconds
            
        Returns:
            Score (0-1), higher is better
        """
        if response_time <= threshold:
            return 1.0
        else:
            return max(0.0, 1.0 - (response_time - threshold) / threshold)


class MetricsCalculator:
    """Calculate comprehensive metrics for RAG evaluation."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metrics = AdditionalMetrics()
        logger.info("Metrics calculator initialized")
    
    def calculate_retrieval_metrics(self,
                                   retrieved_docs: List[str],
                                   relevant_docs: List[str]) -> Dict:
        """Calculate retrieval evaluation metrics.
        
        Args:
            retrieved_docs: Retrieved documents
            relevant_docs: Relevant documents
            
        Returns:
            Dictionary of metrics
        """
        return {
            "precision": self.metrics.retrieval_precision(retrieved_docs, relevant_docs),
            "recall": self.metrics.retrieval_recall(retrieved_docs, relevant_docs),
            "mrr": self.metrics.mrr_score(retrieved_docs, relevant_docs),
        }
    
    def calculate_generation_metrics(self,
                                    reference: str,
                                    generated: str) -> Dict:
        """Calculate generation evaluation metrics.
        
        Args:
            reference: Reference answer
            generated: Generated answer
            
        Returns:
            Dictionary of metrics
        """
        return {
            "similarity": self.metrics.bleu_score(reference, generated),
        }
    
    def calculate_latency_score(self, response_time: float) -> float:
        """Calculate latency score.
        
        Args:
            response_time: Response time in seconds
            
        Returns:
            Latency score (0-1)
        """
        return self.metrics.latency_score(response_time)
