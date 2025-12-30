import logging
import uuid
from typing import List, Dict, Any, Set

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseIndexParams,
    SparseVector, Prefetch,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType,
    PayloadSchemaType, Filter, FieldCondition, MatchValue
)

from .settings import settings
from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    pass

class CollectionNotFoundError(VectorStoreError):
    pass


class VectorStoreService:
    def __init__(self, embedding_service: EmbeddingService):
        logger.info(f"Initializing Qdrant client with gRPC: {settings.qdrant_prefer_grpc}")
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            grpc_port=settings.qdrant_grpc_port,
            prefer_grpc=settings.qdrant_prefer_grpc,
            timeout=60
        )
        self.embedding_service = embedding_service
        self.collection_prefix = settings.qdrant_collection_prefix

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

    def delete_collection(self, collection_name: str) -> None:
        if not collection_name:
            return
        try:
            if self.collection_exists(collection_name):
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted collection: {collection_name}")
        except Exception as exc:
            logger.warning(f"Could not delete collection {collection_name}: {exc}")

    def ensure_collection(self, collection_name: str) -> None:
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
                logger.info(f"Recreating collection {collection_name} for hybrid support")
                self._create_hybrid_collection(collection_name)
                return

            current_size = None
            if isinstance(vectors_config, dict) and "dense" in vectors_config:
                dense_config = vectors_config["dense"]
                if hasattr(dense_config, "size"):
                    current_size = dense_config.size

            if current_size and current_size != self.embedding_service.dimension:
                logger.info(
                    f"Recreating collection {collection_name} due to dimension change "
                    f"{current_size} -> {self.embedding_service.dimension}"
                )
                self._create_hybrid_collection(collection_name)

        except Exception as exc:
            logger.info(f"Creating collection {collection_name}: {exc}")
            self._create_hybrid_collection(collection_name)

    def _create_hybrid_collection(self, collection_name: str) -> None:
        logger.info(f"Creating hybrid collection with Scalar Quantization: {collection_name}")

        if self.collection_exists(collection_name):
            try:
                self.client.delete_collection(collection_name)
            except Exception as exc:
                logger.warning(f"Failed to delete existing collection {collection_name}: {exc}")



        self.client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=self.embedding_service.dimension,
                distance=Distance.COSINE,
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True
                    )
                )
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        }
        )
        self._create_payload_indexes(collection_name)

    def _create_payload_indexes(self, collection_name: str) -> None:
        """Create payload indexes for frequently filtered fields."""
        try:
            logger.info(f"Creating payload indexes for {collection_name}")
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="doc_id",
                field_schema=PayloadSchemaType.INTEGER
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="section",
                field_schema=PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="parent_id",
                field_schema=PayloadSchemaType.INTEGER
            )
            logger.info(f"Payload indexes created successfully for {collection_name}")
        except Exception as exc:
            logger.warning(f"Failed to create payload indexes for {collection_name}: {exc}")

    def reset_collection(self, collection_name: str) -> None:
        if not collection_name:
            return
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self._create_hybrid_collection(collection_name)

    def cleanup_orphaned_collections(self, valid_collections: Set[str]) -> None:
        try:
            collection_list = self.client.get_collections().collections
        except Exception as exc:
            logger.warning(f"Unable to list Qdrant collections: {exc}")
            return

        for collection in collection_list:
            name = getattr(collection, "name", "")
            if not name or not name.startswith(self.collection_prefix):
                continue
            if name not in valid_collections:
                logger.info(f"Deleting orphaned Qdrant collection {name}")
                try:
                    self.client.delete_collection(name)
                except Exception as exc:
                    logger.warning(f"Failed to delete collection {name}: {exc}")

    def build_collection_map(self, documents: List[Any]) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        for document in documents:
            doc_id = getattr(document, "id", None)
            if doc_id is None:
                continue
            collection_name = (
                getattr(document, "collection_name", None)
                or self.collection_name_for_document(doc_id)
            )
            mapping[doc_id] = collection_name
        return mapping

    def add_documents(
            self,
            doc_id: int,
            chunks: List[Dict[str, Any]],
            collection_name: str,
            document_name: str = None
    ) -> None:
        logger.info(f"Adding {len(chunks)} chunks to collection {collection_name} for doc_id {doc_id}")

        if not collection_name:
            raise VectorStoreError(f"Cannot add documents: empty collection_name for doc_id {doc_id}")

        try:
            self.ensure_collection(collection_name)
            logger.debug(f"Collection {collection_name} ensured/created successfully")
        except Exception as exc:
            logger.error(f"Failed to ensure collection {collection_name}: {exc}", exc_info=True)
            raise VectorStoreError(f"Failed to ensure collection {collection_name}: {exc}")

        points = []

        try:
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

            logger.debug(f"Created {len(points)} points for collection {collection_name}")

        except Exception as exc:
            logger.error(f"Failed to create points for collection {collection_name}: {exc}", exc_info=True)
            raise VectorStoreError(f"Failed to create points: {exc}")

        try:
            logger.debug(f"Upserting {len(points)} points to collection {collection_name}")
            self.client.upsert(collection_name=collection_name, points=points)
            logger.info(f"Successfully added {len(points)} points to collection {collection_name}")

        except Exception as exc:
            if "vector" in str(exc).lower() or "size" in str(exc).lower():
                logger.warning(
                    f"Collection {collection_name} had schema issues (vector/size error), recreating collection"
                )
                try:
                    self._create_hybrid_collection(collection_name)
                    self.client.upsert(collection_name=collection_name, points=points)
                    logger.info(f"Successfully added {len(points)} points after recreating collection {collection_name}")
                except Exception as retry_exc:
                    logger.error(
                        f"Failed to add documents after recreating collection {collection_name}: {retry_exc}",
                        exc_info=True
                    )
                    raise VectorStoreError(f"Failed to add documents after recreating collection: {retry_exc}")
            else:
                logger.error(
                    f"Failed to upsert documents to collection {collection_name}: {type(exc).__name__}: {exc}",
                    exc_info=True
                )
                raise VectorStoreError(f"Failed to add documents to {collection_name}: {exc}")

    def document_exists(self, collection_name: str) -> bool:
        if not self.collection_exists(collection_name):
            return False
        try:
            results, _ = self.client.scroll(collection_name=collection_name, limit=1)
            return len(results) > 0
        except Exception as exc:
            logger.warning(f"Failed to check collection {collection_name}: {exc}")
            return False

    def delete_document(self, collection_name) -> None:
        self.delete_collection(collection_name)

    def search(self, query: str, doc_collection_map: Dict[int, str], top_k: int = 20) -> List[Dict[str, Any]]:
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
                # RRF Fusion mit beiden Vektoren
                from qdrant_client.models import Fusion

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
                    query=dense_embedding,
                    using="dense",
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
                    'chunk_index': hit.payload.get('chunk_index'),
                    'total_chunks': hit.payload.get('total_chunks'),
                    'score': hit.score
                })

        combined_results.sort(key=lambda item: item['score'], reverse=True)
        return combined_results[:top_k]

    def search_dense_only(
            self,
            query: str,
            collection_name: str,
            top_k: int = 20
    ) -> List[Dict[str, Any]]:
        if not self.collection_exists(collection_name):
            return []

        query_embedding = self.embedding_service.embed_text(query)
        results = self.client.search(  # type: ignore[attr-defined]
            collection_name=collection_name,
            query_vector=("dense", query_embedding),
            limit=top_k
        )

        return [self._hit_to_dict(hit) for hit in results]

    def search_sparse_only(
            self,
            query: str,
            collection_name: str,
            top_k: int = 20
    ) -> List[Dict[str, Any]]:
        if not self.collection_exists(collection_name):
            return []

        sparse_embedding = self.embedding_service.embed_sparse(query)
        results = self.client.search(  # type: ignore[attr-defined]
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

        return [self._hit_to_dict(hit) for hit in results]

    def get_metadata_chunks_for_docs(
            self,
            doc_collection_map: Dict[int, str]
    ) -> List[Dict[str, Any]]:
        if not doc_collection_map:
            return []

        metadata_chunks: List[Dict[str, Any]] = []

        for doc_id, collection_name in doc_collection_map.items():
            if not self.collection_exists(collection_name):
                continue

            try:
                results, _ = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="section",
                                match=MatchValue(value="Document Metadata")
                            )
                        ]
                    ),
                    limit=2,
                    with_payload=True
                )
            except Exception as exc:
                logger.warning(f"Failed to retrieve metadata for doc {doc_id}: {exc}")
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

    @staticmethod
    def _hit_to_dict(hit) -> Dict[str, Any]:
        return {
            'text': hit.payload['text'],
            'doc_id': hit.payload['doc_id'],
            'chunk_id': hit.payload['chunk_id'],
            'parent_id': hit.payload.get('parent_id'),
            'document_name': hit.payload.get('document_name', ''),
            'section': hit.payload.get('section', ''),
            'position': hit.payload.get('position', ''),
            'chunk_index': hit.payload.get('chunk_index'),
            'total_chunks': hit.payload.get('total_chunks'),
            'score': hit.score
        }