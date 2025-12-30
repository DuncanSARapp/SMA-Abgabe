import logging
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import CrossEncoder

from .settings import settings
from .embeddings import get_optimal_device

logger = logging.getLogger(__name__)

BASE_THRESHOLD = 0.2

def _calculate_dynamic_threshold(scores: List[float]) -> tuple[float, str] | tuple[
    np.ndarray[tuple[Any, ...], np.dtype[Any]], str]:
    if not scores:
        return BASE_THRESHOLD, "no_scores"

    scores_array = np.array(scores)
    max_score = float(np.max(scores_array))
    mean_score = float(np.mean(scores_array))
    std_score = float(np.std(scores_array))

    if len(scores) >= 2:
        sorted_scores = np.sort(scores_array)[::-1]
        top_gap = sorted_scores[0] - sorted_scores[1]

        if top_gap > 0.3:
            return sorted_scores[0] - 0.01, "clear_winner"

    if mean_score > 0.5:
        threshold = max(mean_score - std_score * 0.5, BASE_THRESHOLD)
        return threshold, "high_quality_results"

    if std_score > 0.2:
        threshold = max(mean_score, BASE_THRESHOLD)
        return threshold, "high_variance"

    if max_score < 0.3:
        threshold = max_score * 0.5
        return threshold, "low_quality_all"

    threshold = max(mean_score - std_score, BASE_THRESHOLD)
    return threshold, "adaptive"


def get_score_statistics(documents: List[Dict[str, Any]]) -> Dict[str, float]:
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


class RerankerService:
    _instance = None

    @classmethod
    def get_instance(cls) -> "RerankerService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        device = get_optimal_device()
        logger.info(f"Loading reranker model on device: {device}")

        self.model = CrossEncoder(
            settings.reranker_model,
            device=device
        )

        logger.info(f"Reranker model loaded: {settings.reranker_model}")

    def rerank(
            self,
            query: str,
            documents: List[Dict[str, Any]],
            top_k: int = 5,
            apply_threshold: bool = True
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        pairs = [(query, doc['text']) for doc in documents]
        scores = self.model.predict(pairs)
        scores_list = [float(s) for s in scores]

        for doc, score in zip(documents, scores_list):
            doc['rerank_score'] = score

        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

        if apply_threshold:
            threshold, reason = _calculate_dynamic_threshold(scores_list)
            filtered = [doc for doc in reranked if doc['rerank_score'] >= threshold]

            if filtered:
                filtered[0]['threshold_used'] = threshold
                filtered[0]['threshold_reason'] = reason
                return filtered[:top_k]
            elif reranked:
                reranked[0]['threshold_used'] = threshold
                reranked[0]['threshold_reason'] = "fallback_below_threshold"
                return reranked[:1]

        return reranked[:top_k]
