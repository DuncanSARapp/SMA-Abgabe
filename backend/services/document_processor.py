import logging
import os
import pickle
from typing import List, Dict, Any, Optional

from docling_core.transforms.chunker import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from .settings import settings

logger = logging.getLogger(__name__)


class MarkdownTableSerializerProvider(ChunkingSerializerProvider):
    """Serialisiert Tabellen als Markdown (RAG-optimiert)"""

    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )


def load_parent_document(pickle_path: Optional[str], parent_id: int) -> str:
    """
    Lädt Parent-Dokument aus Pickle-Datei
    Wird von rag_service.py für Neighbor-Expansion benötigt
    """
    if not pickle_path:
        return ""

    try:
        with open(pickle_path, 'rb') as f:
            parent_docs = pickle.load(f)

        if parent_id < len(parent_docs):
            return parent_docs[parent_id]
        return ""

    except FileNotFoundError:
        logger.error(f"Parent document file not found: {pickle_path}")
        return ""
    except Exception as exc:
        logger.error(f"Error loading parent document from {pickle_path}: {exc}")
        return ""


class DocumentProcessor:
    """
    Verarbeitet Dokumente mit Parent-Child-Chunking
    Nutzt Markdown Table Serializer für bessere Tabellen-Darstellung
    """

    def __init__(self):
        embed_model = settings.embedding_model

        # HuggingFace Tokenizer für Chunking
        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(embed_model),
            max_tokens=settings.chunk_size,
        )

        # HybridChunker mit Markdown Table Serializer
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            serializer_provider=MarkdownTableSerializerProvider(),
        )

        logger.info(
            f"DocumentProcessor initialized with HybridChunker "
            f"(max_tokens={settings.chunk_size}, markdown_tables=True)"
        )

    def process_document(
            self,
            doc_id: int,
            text: str,
            pickle_path: str,
            document_name: str = "",
            metadata_chunk: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        parent_size = settings.parent_chunk_size
        parent_overlap = settings.parent_chunk_overlap or settings.chunk_overlap

        parent_docs = []
        for start in range(0, len(text), parent_size - parent_overlap):
            parent_chunk = text[start:start + parent_size]
            if parent_chunk.strip():
                parent_docs.append(parent_chunk)

        # Metadata-Chunk als Parent 0 (falls vorhanden)
        if metadata_chunk:
            parent_docs_with_meta = [metadata_chunk] + parent_docs
        else:
            parent_docs_with_meta = parent_docs

        # Speichere Parent-Dokumente für Neighbor-Expansion
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        with open(pickle_path, 'wb') as f:
            pickle.dump(parent_docs_with_meta, f)

        chunks = []
        chunk_counter = 0

        # Metadata-Chunk
        if metadata_chunk:
            chunks.append({
                'text': metadata_chunk,
                'parent_id': 0,
                'doc_id': doc_id,
                'document_name': document_name,
                'section': 'Document Metadata',
                'position': 'metadata',
                'chunk_index': chunk_counter,
                'is_metadata': True
            })
            parent_offset = 1
            chunk_counter += 1
        else:
            parent_offset = 0

        # Child-Level: Kleine Chunks (für Retrieval)
        child_size = settings.child_chunk_size or settings.chunk_size
        child_overlap = settings.child_chunk_overlap or settings.chunk_overlap

        for parent_id, parent_text in enumerate(parent_docs):
            # Erstelle Child-Chunks aus jedem Parent
            for start in range(0, len(parent_text), child_size - child_overlap):
                child_text = parent_text[start:start + child_size]

                if child_text.strip():
                    chunks.append({
                        'text': child_text,
                        'parent_id': parent_id + parent_offset,
                        'doc_id': doc_id,
                        'document_name': document_name,
                        'section': 'Body',
                        'position': 'middle',
                        'chunk_index': chunk_counter,
                        'is_metadata': False
                    })
                    chunk_counter += 1

        logger.info(
            f"Processed document {document_name}: "
            f"{len(parent_docs)} parent chunks, {len(chunks)} child chunks"
            f"{' (with metadata)' if metadata_chunk else ''}"
        )

        return chunks