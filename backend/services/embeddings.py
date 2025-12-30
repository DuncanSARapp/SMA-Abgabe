import hashlib
import logging
import math
import re
from collections import Counter, OrderedDict
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import torch

from .settings import settings

logger = logging.getLogger(__name__)

# TODO
def get_optimal_device() -> str:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device = 'cuda'

    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        logger.info(f"ROCm available: HIP version {torch.version.hip}")
        return 'cuda'

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Apple MPS (Metal) available")
        return 'mps'

    logger.info("No GPU acceleration available, using CPU")
    return 'cpu'


def _make_key(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


class LRUCache:
    """Simple LRU Cache for embeddings."""

    def __init__(self, max_size: int = 10000):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, text: str) -> Optional[List[float]]:
        key = _make_key(text)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, text: str, embedding: List[float]) -> None:
        key = _make_key(text)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = embedding

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "size": len(self.cache),
            "max_size": self.max_size
        }


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r'\b[a-zA-ZäöüÄÖÜß]+\b', text)
    return [t for t in tokens if len(t) > 2]


class SparseEmbedding:
    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size

    def _hash_token(self, token: str) -> int:
        return hash(token) % self.vocab_size

    def embed(self, text: str) -> Dict[str, Any]:
        tokens = tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}

        term_frequencies = Counter(tokens)
        total_tokens = len(tokens)

        indices = []
        values = []

        for token, count in term_frequencies.items():
            idx = self._hash_token(token)
            tf_score = 1 + math.log(count) if count > 0 else 0
            score = tf_score / math.sqrt(total_tokens)
            indices.append(idx)
            values.append(float(score))

        sorted_pairs = sorted(zip(indices, values), key=lambda x: x[0])

        deduped = {}
        for idx, val in sorted_pairs:
            if idx not in deduped or val > deduped[idx]:
                deduped[idx] = val

        return {
            "indices": list(deduped.keys()),
            "values": list(deduped.values())
        }


class EmbeddingService:
    _instance = None

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        device = get_optimal_device()
        logger.info(f"Loading embedding model on device: {device}")
        self.model = SentenceTransformer(
            settings.embedding_model,
            device=device,
            cache_folder=settings.models_cache_dir
        )
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.sparse_model = SparseEmbedding(vocab_size=30000)
        self.cache = LRUCache(max_size=settings.embedding_cache_size)

        logger.info(f"Embedding model loaded: {settings.embedding_model} (dim={self.dimension})")
        logger.info(f"Embedding cache enabled: {settings.embedding_cache_size} entries")

    def embed_text(self, text: str) -> List[float]:
        """Embed text with LRU caching."""
        cached = self.cache.get(text)
        if cached is not None:
            return cached

        embedding = self.model.encode(text).tolist()
        self.cache.put(text, embedding)
        return embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts with caching."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            new_embeddings = self.model.encode(uncached_texts).tolist()
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                self.cache.put(texts[idx], embedding)

        return embeddings

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        return self.cache.get_stats()

    def embed_sparse(self, text: str) -> Dict[str, Any]:
        return self.sparse_model.embed(text)

    def embed_sparse_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.sparse_model.embed(text) for text in texts]
