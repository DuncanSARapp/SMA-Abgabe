import os
import pickle
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


class VectorStoreService:
    """Service for managing Qdrant vector store"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.embedding_service = embedding_service
        self.collection_name = settings.qdrant_collection_name
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            self.client.get_collection(self.collection_name)
        except Exception as e:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_service.dimension,
                    distance=Distance.COSINE
                )
            )
    
    def add_documents(self, doc_id: int, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to vector store"""
        points = []
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_service.embed_text(chunk['text'])
            point = PointStruct(
                id=f"{doc_id}_{i}",
                vector=embedding,
                payload={
                    'doc_id': doc_id,
                    'chunk_id': i,
                    'text': chunk['text'],
                    'parent_id': chunk.get('parent_id')
                }
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
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
