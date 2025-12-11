import os
import pickle
import uuid
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config.settings import settings


class EmbeddingService:
    """Service for generating embeddings using local models"""
    
    def __init__(self):
        self.model = SentenceTransformer(settings.embedding_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.model.encode(text).tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return self.model.encode(texts).tolist()


logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for managing Qdrant vector store"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.embedding_service = embedding_service
        self.collection_name = settings.qdrant_collection_name
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if missing or update mismatched dimensions"""
        try:
            info = self.client.get_collection(self.collection_name)
            vectors_config = getattr(info.config.params, "vectors", None)

            current_size = None
            if hasattr(vectors_config, "size"):
                current_size = vectors_config.size
            elif hasattr(vectors_config, "default") and vectors_config.default is not None:
                current_size = getattr(vectors_config.default, "size", None)
            elif hasattr(vectors_config, "dict"):
                # Fall back to dict representation provided by pydantic models
                vc_dict = vectors_config.dict()
                if isinstance(vc_dict, dict):
                    if "size" in vc_dict:
                        current_size = vc_dict.get("size")
                    elif "default" in vc_dict and isinstance(vc_dict["default"], dict):
                        current_size = vc_dict["default"].get("size")

            if current_size is None:
                logger.warning("Unable to determine existing Qdrant vector size; recreating collection %s", self.collection_name)
                raise ValueError("Unknown vector size")

            if current_size != self.embedding_service.dimension:
                logger.info(
                    "Recreating Qdrant collection %s to adjust dimensionality %s -> %s",
                    self.collection_name,
                    current_size,
                    self.embedding_service.dimension,
                )
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_service.dimension,
                        distance=Distance.COSINE
                    )
                )
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_service.dimension,
                    distance=Distance.COSINE
                )
            )
    
    def add_documents(self, doc_id: int, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to vector store"""
        points = []
        for idx, chunk in enumerate(chunks):
            embedding = self.embedding_service.embed_text(chunk['text'])
            chunk_id = chunk.get('chunk_id', idx)
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'text': chunk['text'],
                    'parent_id': chunk.get('parent_id')
                }
            )
            points.append(point)
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        except Exception as exc:
            if "vector dimension" in str(exc).lower() or "size" in str(exc).lower():
                logger.warning("Qdrant collection dimension mismatch detected; recreating collection and retrying")
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_service.dimension,
                        distance=Distance.COSINE
                    )
                )
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            else:
                raise
    
    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        query_embedding = self.embedding_service.embed_text(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        return [
            {
                'text': hit.payload['text'],
                'doc_id': hit.payload['doc_id'],
                'chunk_id': hit.payload['chunk_id'],
                'parent_id': hit.payload.get('parent_id'),
                'score': hit.score
            }
            for hit in results
        ]
