import os
import pickle
import uuid
import logging
import re
import math
from collections import Counter
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    SparseVectorParams, SparseIndexParams,
    SparseVector, Prefetch, FusionQuery, Fusion
)
from config.settings import settings


class SparseEmbedding:
    """
    Lightweight TF-IDF based sparse embedding generator.
    No external dependencies needed - much faster to install than fastembed.
    """
    
    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.word_to_idx: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count = 0
        self._initialized = False
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - lowercase and split on non-alphanumeric"""
        text = text.lower()
        # Keep German umlauts and common characters
        tokens = re.findall(r'\b[a-zA-ZäöüÄÖÜß]+\b', text)
        # Filter very short tokens
        return [t for t in tokens if len(t) > 2]
    
    def _hash_token(self, token: str) -> int:
        """Hash token to vocab index for consistent sparse vector indices"""
        return hash(token) % self.vocab_size
    
    def embed(self, text: str) -> Dict[str, Any]:
        """
        Generate sparse embedding using term frequency with position weighting.
        Returns dict with 'indices' and 'values' for Qdrant SparseVector.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}
        
        # Calculate term frequencies
        tf = Counter(tokens)
        total_tokens = len(tokens)
        
        indices = []
        values = []
        
        for token, count in tf.items():
            idx = self._hash_token(token)
            # TF with log normalization
            tf_score = 1 + math.log(count) if count > 0 else 0
            # Normalize by document length
            score = tf_score / math.sqrt(total_tokens)
            
            indices.append(idx)
            values.append(float(score))
        
        # Sort by index for consistency
        sorted_pairs = sorted(zip(indices, values), key=lambda x: x[0])
        
        # Remove duplicates (keep max value for same index due to hash collisions)
        deduped = {}
        for idx, val in sorted_pairs:
            if idx not in deduped or val > deduped[idx]:
                deduped[idx] = val
        
        return {
            "indices": list(deduped.keys()),
            "values": list(deduped.values())
        }


class EmbeddingService:
    """Service for generating embeddings using local models"""
    
    def __init__(self):
        self.model = SentenceTransformer(settings.embedding_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        # Lightweight sparse embedding (no heavy dependencies)
        self.sparse_model = SparseEmbedding(vocab_size=30000)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.model.encode(text).tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return self.model.encode(texts).tolist()
    
    def embed_sparse(self, text: str) -> Dict[str, Any]:
        """Generate sparse TF-based embedding for keyword matching"""
        return self.sparse_model.embed(text)
    
    def embed_sparse_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Generate sparse embeddings for multiple texts"""
        return [self.sparse_model.embed(text) for text in texts]


logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for managing Qdrant vector store with hybrid search"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.embedding_service = embedding_service
        self.default_collection = settings.qdrant_collection_name
        self.collection_prefix = settings.qdrant_collection_prefix
    
    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------
    def collection_name_for_document(self, document_id: int) -> str:
        return f"{self.collection_prefix}{document_id}"

    def collection_exists(self, collection_name: str) -> bool:
        if not collection_name:
            return False
        try:
            self.client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def ensure_collection(self, collection_name: str):
        """Ensure a hybrid collection exists for a document."""
        if not collection_name:
            return
        try:
            info = self.client.get_collection(collection_name)
            vectors_config = getattr(info.config.params, "vectors", None)

            has_dense = False
            if isinstance(vectors_config, dict):
                has_dense = "dense" in vectors_config
            elif hasattr(vectors_config, "get"):
                has_dense = vectors_config.get("dense") is not None

            sparse_config = getattr(info.config.params, "sparse_vectors", None)
            has_sparse = bool(sparse_config)

            if not has_dense or not has_sparse:
                logger.info("Recreating collection %s for hybrid support", collection_name)
                self._create_hybrid_collection(collection_name)
                return

            current_size = None
            if isinstance(vectors_config, dict) and "dense" in vectors_config:
                dense_config = vectors_config["dense"]
                if hasattr(dense_config, "size"):
                    current_size = dense_config.size

            if current_size and current_size != self.embedding_service.dimension:
                logger.info(
                    "Recreating collection %s due to dimension change %s -> %s",
                    collection_name,
                    current_size,
                    self.embedding_service.dimension,
                )
                self._create_hybrid_collection(collection_name)

        except Exception as exc:
            logger.info("Creating collection %s (%s)", collection_name, exc)
            self._create_hybrid_collection(collection_name)

    def _create_hybrid_collection(self, collection_name: str):
        """Create or recreate a hybrid collection with dense + sparse vectors."""
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.embedding_service.dimension,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            }
        )

    def reset_collection(self, collection_name: str):
        if not collection_name:
            return
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self._create_hybrid_collection(collection_name)

    def delete_collection(self, collection_name: str):
        if not collection_name or not self.collection_exists(collection_name):
            return
        self.client.delete_collection(collection_name)

    def cleanup_orphaned_collections(self, valid_collections: set[str]) -> None:
        try:
            collection_list = self.client.get_collections().collections
        except Exception as exc:
            logger.warning("Unable to list Qdrant collections: %s", exc)
            return

        for collection in collection_list:
            name = getattr(collection, "name", "")
            if not name or not name.startswith(self.collection_prefix):
                continue
            if name not in valid_collections:
                logger.info("Deleting orphaned Qdrant collection %s", name)
                try:
                    self.client.delete_collection(name)
                except Exception as exc:
                    logger.warning("Failed to delete collection %s: %s", name, exc)

    def build_collection_map(self, documents: List[Any]) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        for document in documents:
            doc_id = getattr(document, "id", None)
            if doc_id is None:
                continue
            collection_name = getattr(document, "collection_name", None) or self.collection_name_for_document(doc_id)
            mapping[doc_id] = collection_name
        return mapping
    
    def add_documents(
        self,
        doc_id: int,
        chunks: List[Dict[str, Any]],
        collection_name: str,
        document_name: str | None = None
    ) -> None:
        """Add document chunks to the specified collection (dense + sparse)."""
        self.ensure_collection(collection_name)
        points = []
        for idx, chunk in enumerate(chunks):
            dense_embedding = self.embedding_service.embed_text(chunk['text'])
            sparse_embedding = self.embedding_service.embed_sparse(chunk['text'])
            chunk_id = chunk.get('chunk_id', idx)
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_embedding,
                    "sparse": SparseVector(
                        indices=sparse_embedding["indices"],
                        values=sparse_embedding["values"]
                    )
                },
                payload={
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'text': chunk['text'],
                    'parent_id': chunk.get('parent_id'),
                    'document_name': document_name or chunk.get('document_name', ''),
                    'section': chunk.get('section', ''),
                    'position': chunk.get('position', 'middle'),
                    'chunk_index': idx,
                    'total_chunks': len(chunks)
                }
            )
            points.append(point)
        
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
        except Exception as exc:
            if "vector" in str(exc).lower() or "size" in str(exc).lower():
                logger.warning("Collection %s had schema issues; recreating", collection_name)
                self._create_hybrid_collection(collection_name)
                self.client.upsert(collection_name=collection_name, points=points)
            else:
                raise
    
    def document_exists(self, collection_name: str) -> bool:
        if not self.collection_exists(collection_name):
            return False
        try:
            results, _ = self.client.scroll(
                collection_name=collection_name,
                limit=1
            )
            return len(results) > 0
        except Exception as exc:
            logger.warning("Failed to check collection %s: %s", collection_name, exc)
            return False

    def delete_document(self, collection_name: str) -> None:
        self.delete_collection(collection_name)

    def search(self, query: str, doc_collection_map: Dict[int, str], top_k: int = 20) -> List[Dict[str, Any]]:
        """Hybrid search across the selected per-document collections."""
        if not doc_collection_map:
            return []

        dense_embedding = self.embedding_service.embed_text(query)
        sparse_embedding = self.embedding_service.embed_sparse(query)
        combined_results: List[Dict[str, Any]] = []

        per_collection_limit = max(top_k, 5)

        for doc_id, collection_name in doc_collection_map.items():
            if not self.collection_exists(collection_name):
                continue
            try:
                results = self.client.query_points(
                    collection_name=collection_name,
                    prefetch=[
                        Prefetch(query=dense_embedding, using="dense", limit=per_collection_limit * 2),
                        Prefetch(
                            query=SparseVector(
                                indices=sparse_embedding["indices"],
                                values=sparse_embedding["values"]
                            ),
                            using="sparse",
                            limit=per_collection_limit * 2
                        )
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=per_collection_limit
                )
            except Exception as exc:
                logger.warning("Query failed for collection %s: %s", collection_name, exc)
                continue

            for hit in results.points:
                combined_results.append({
                    'text': hit.payload['text'],
                    'doc_id': hit.payload.get('doc_id', doc_id),
                    'chunk_id': hit.payload['chunk_id'],
                    'parent_id': hit.payload.get('parent_id'),
                    'document_name': hit.payload.get('document_name', ''),
                    'section': hit.payload.get('section', ''),
                    'position': hit.payload.get('position', ''),
                    'score': hit.score
                })

        combined_results.sort(key=lambda item: item['score'], reverse=True)
        return combined_results[:top_k]
    
    def search_dense_only(self, query: str, collection_name: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Search using only dense vectors (semantic similarity) within a collection."""
        if not self.collection_exists(collection_name):
            return []
        query_embedding = self.embedding_service.embed_text(query)
        results = self.client.search(
            collection_name=collection_name,
            query_vector=("dense", query_embedding),
            limit=top_k
        )
        return [
            {
                'text': hit.payload['text'],
                'doc_id': hit.payload['doc_id'],
                'chunk_id': hit.payload['chunk_id'],
                'parent_id': hit.payload.get('parent_id'),
                'document_name': hit.payload.get('document_name', ''),
                'section': hit.payload.get('section', ''),
                'position': hit.payload.get('position', ''),
                'score': hit.score
            }
            for hit in results
        ]
    
    def get_metadata_chunks_for_docs(self, doc_collection_map: Dict[int, str]) -> List[Dict[str, Any]]:
        """Retrieve metadata chunks for the selected documents."""
        if not doc_collection_map:
            return []

        metadata_chunks: List[Dict[str, Any]] = []
        for doc_id, collection_name in doc_collection_map.items():
            if not self.collection_exists(collection_name):
                continue
            try:
                results, _ = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter={
                        "must": [
                            {"key": "section", "match": {"value": "Document Metadata"}}
                        ]
                    },
                    limit=2,
                    with_payload=True
                )
            except Exception as exc:
                logger.warning("Failed to retrieve metadata for doc %s: %s", doc_id, exc)
                continue

            for point in results:
                metadata_chunks.append({
                    'text': point.payload['text'],
                    'doc_id': doc_id,
                    'chunk_id': point.payload.get('chunk_id', 0),
                    'parent_id': point.payload.get('parent_id', 0),
                    'document_name': point.payload.get('document_name', ''),
                    'section': point.payload.get('section', ''),
                    'position': point.payload.get('position', ''),
                    'score': 0.0,
                    'is_metadata_injection': True
                })
        return metadata_chunks
    
    def search_sparse_only(self, query: str, collection_name: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Search using only sparse vectors within a collection."""
        if not self.collection_exists(collection_name):
            return []
        sparse_embedding = self.embedding_service.embed_sparse(query)
        results = self.client.search(
            collection_name=collection_name,
            query_vector=(
                "sparse",
                SparseVector(
                    indices=sparse_embedding["indices"],
                    values=sparse_embedding["values"]
                )
            ),
            limit=top_k
        )
        return [
            {
                'text': hit.payload['text'],
                'doc_id': hit.payload['doc_id'],
                'chunk_id': hit.payload['chunk_id'],
                'parent_id': hit.payload.get('parent_id'),
                'document_name': hit.payload.get('document_name', ''),
                'section': hit.payload.get('section', ''),
                'position': hit.payload.get('position', ''),
                'score': hit.score
            }
            for hit in results
        ]
