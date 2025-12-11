from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import CrossEncoder
from config.settings import settings


class RerankerService:
    """Service for reranking retrieved documents with dynamic thresholding"""
    
    def __init__(self):
        self.model = CrossEncoder(settings.reranker_model)
        # Base threshold - will be adjusted dynamically
        self.base_threshold = 0.2
    
    def _calculate_dynamic_threshold(self, scores: List[float]) -> Tuple[float, str]:
        """
        Calculate a dynamic threshold based on score distribution.
        
        Returns:
            Tuple of (threshold, reason)
        """
        if not scores:
            return self.base_threshold, "no_scores"
        
        scores_array = np.array(scores)
        max_score = float(np.max(scores_array))
        mean_score = float(np.mean(scores_array))
        std_score = float(np.std(scores_array))
        
        # If there's a clear winner (large gap between top scores)
        if len(scores) >= 2:
            sorted_scores = np.sort(scores_array)[::-1]
            top_gap = sorted_scores[0] - sorted_scores[1]
            
            # If top result is significantly better than second
            if top_gap > 0.3:
                # Only return top result - use score just below top
                return sorted_scores[0] - 0.01, "clear_winner"
        
        # If scores are very high overall
        if mean_score > 0.5:
            # Be more selective - use higher threshold
            threshold = max(mean_score - std_score * 0.5, self.base_threshold)
            return threshold, "high_quality_results"
        
        # If scores are spread out (high std)
        if std_score > 0.2:
            # Use mean as threshold to filter out low scores
            threshold = max(mean_score, self.base_threshold)
            return threshold, "high_variance"
        
        # If all scores are low
        if max_score < 0.3:
            # Lower threshold to get at least some results
            threshold = max_score * 0.5
            return threshold, "low_quality_all"
        
        # Default: use adaptive threshold based on distribution
        # Keep results within 1 std of the mean
        threshold = max(mean_score - std_score, self.base_threshold)
        return threshold, "adaptive"
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: int = 5,
        apply_threshold: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query with dynamic thresholding.
        
        Args:
            query: The search query
            documents: List of retrieved documents with 'text' field
            top_k: Maximum number of top documents to return
            apply_threshold: Whether to apply dynamic threshold filtering
            
        Returns:
            Reranked list of documents with scores and threshold info
        """
        if not documents:
            return []
        
        # Prepare pairs for reranking
        pairs = [(query, doc['text']) for doc in documents]

        # Get scores from cross-encoder
        scores = self.model.predict(pairs)
        scores_list = [float(s) for s in scores]

        # Add scores to documents
        for doc, score in zip(documents, scores_list):
            doc['rerank_score'] = score
        
        # Sort by rerank score
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        if apply_threshold:
            # Calculate dynamic threshold
            threshold, reason = self._calculate_dynamic_threshold(scores_list)
            
            # Filter by threshold
            filtered = [doc for doc in reranked if doc['rerank_score'] >= threshold]
            
            # Add threshold info to first result for debugging
            if filtered:
                filtered[0]['threshold_used'] = threshold
                filtered[0]['threshold_reason'] = reason
            
            # Return at least 1 result if any exist, up to top_k
            if filtered:
                return filtered[:top_k]
            elif reranked:
                # Fallback: return top result even if below threshold
                reranked[0]['threshold_used'] = threshold
                reranked[0]['threshold_reason'] = "fallback_below_threshold"
                return reranked[:1]
        
        return reranked[:top_k]
    
    def get_score_statistics(self, documents: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get statistics about rerank scores for debugging/analysis"""
        scores = [doc.get('rerank_score', 0) for doc in documents if 'rerank_score' in doc]
        
        if not scores:
            return {}
        
        scores_array = np.array(scores)
        return {
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'median': float(np.median(scores_array))
        }
