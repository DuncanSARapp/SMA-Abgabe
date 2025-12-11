from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from config.settings import settings


class RerankerService:
    """Service for reranking retrieved documents"""
    
    def __init__(self):
        self.model = CrossEncoder(settings.reranker_model)
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query
        
        Args:
            query: The search query
            documents: List of retrieved documents with 'text' field
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of top_k documents with scores
        """
        if not documents:
            return []
        
        # Prepare pairs for reranking
        pairs = [(query, doc['text']) for doc in documents]

        # Get scores from cross-encoder
        scores = self.model.predict(pairs)

        # Add scores to documents and sort
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by rerank score and return top_k
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_k]
