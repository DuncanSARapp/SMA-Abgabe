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
        self.collection_name = settings.qdrant_collection_name
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection with hybrid vectors (dense + sparse) if needed"""
        try:
            info = self.client.get_collection(self.collection_name)
            vectors_config = getattr(info.config.params, "vectors", None)

            # Check if we have the new hybrid setup with named vectors
            has_dense = False
            has_sparse = False
            
            if isinstance(vectors_config, dict):
                has_dense = "dense" in vectors_config
            elif hasattr(vectors_config, "get"):
                has_dense = vectors_config.get("dense") is not None
            
            sparse_config = getattr(info.config.params, "sparse_vectors", None)
            if sparse_config:
                has_sparse = True
            
            # If we don't have hybrid setup, recreate
            if not has_dense or not has_sparse:
                logger.info("Recreating collection %s for hybrid search support", self.collection_name)
                self._create_hybrid_collection()
                return
                
            # Check dense vector dimension
            current_size = None
            if isinstance(vectors_config, dict) and "dense" in vectors_config:
                dense_config = vectors_config["dense"]
                if hasattr(dense_config, "size"):
                    current_size = dense_config.size
            
            if current_size and current_size != self.embedding_service.dimension:
                logger.info(
                    "Recreating collection %s to adjust dimensionality %s -> %s",
                    self.collection_name,
                    current_size,
                    self.embedding_service.dimension,
                )
                self._create_hybrid_collection()
                
        except Exception as e:
            logger.info("Creating new hybrid collection %s: %s", self.collection_name, e)
            self._create_hybrid_collection()
    
    def _create_hybrid_collection(self):
        """Create a collection with both dense and sparse vectors"""
        self.client.recreate_collection(
            collection_name=self.collection_name,
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
    
    def add_documents(self, doc_id: int, chunks: List[Dict[str, Any]], document_name: str = None) -> None:
        """Add document chunks to vector store with both dense and sparse embeddings"""
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
                collection_name=self.collection_name,
                points=points
            )
        except Exception as exc:
            if "vector" in str(exc).lower() or "size" in str(exc).lower():
                logger.warning("Qdrant collection issue detected; recreating collection and retrying")
                self._create_hybrid_collection()
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            else:
                raise
    
    def document_exists(self, doc_id: int) -> bool:
        """Check if a document has any chunks in the vector store"""
        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {"key": "doc_id", "match": {"value": doc_id}}
                    ]
                },
                limit=1
            )
            return len(results) > 0
        except Exception as e:
            logger.warning("Failed to check document existence in Qdrant: %s", e)
            return False

    def delete_document(self, doc_id: int) -> None:
        """Delete all chunks for a document from the vector store"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector={
                    "filter": {
                        "must": [
                            {"key": "doc_id", "match": {"value": doc_id}}
                        ]
                    }
                }
            )
        except Exception as e:
            logger.warning("Failed to delete document %s from Qdrant: %s", doc_id, e)

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Hybrid search combining dense and sparse (BM25) vectors using RRF fusion"""
        dense_embedding = self.embedding_service.embed_text(query)
        sparse_embedding = self.embedding_service.embed_sparse(query)
        
        # Use Qdrant's query API with prefetch for hybrid search
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_embedding,
                    using="dense",
                    limit=top_k * 2
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_embedding["indices"],
                        values=sparse_embedding["values"]
                    ),
                    using="sparse",
                    limit=top_k * 2
                )
            ],
            query=FusionQuery(fusion=Fusion.RRF),
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
            for hit in results.points
        ]
    
    def search_dense_only(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Search using only dense vectors (semantic similarity)"""
        query_embedding = self.embedding_service.embed_text(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
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
    
    def get_metadata_chunks_for_docs(self, doc_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Retrieve metadata chunks for specific documents.
        Metadata chunks have section='Document Metadata' or position='metadata'.
        """
        if not doc_ids:
            return []
        
        try:
            # Query for metadata chunks of the specified documents
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {"key": "doc_id", "match": {"any": doc_ids}},
                        {"key": "section", "match": {"value": "Document Metadata"}}
                    ]
                },
                limit=len(doc_ids) * 2,  # Allow for some buffer
                with_payload=True
            )
            
            return [
                {
                    'text': point.payload['text'],
                    'doc_id': point.payload['doc_id'],
                    'chunk_id': point.payload.get('chunk_id', 0),
                    'parent_id': point.payload.get('parent_id', 0),
                    'document_name': point.payload.get('document_name', ''),
                    'section': point.payload.get('section', ''),
                    'position': point.payload.get('position', ''),
                    'score': 0.0,  # No search score for direct retrieval
                    'is_metadata_injection': True  # Mark as injected
                }
                for point in results
            ]
        except Exception as e:
            logger.warning("Failed to retrieve metadata chunks: %s", e)
            return []
    
    def search_sparse_only(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Search using only sparse vectors (keyword/BM25 similarity)"""
        sparse_embedding = self.embedding_service.embed_sparse(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=("sparse", SparseVector(
                indices=sparse_embedding["indices"],
                values=sparse_embedding["values"]
            )),
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
