import os
import pickle
import logging
from typing import List, Dict, Any, Tuple
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Service for processing and chunking documents with metadata extraction"""
    
    def __init__(self):
        parent_overlap = getattr(settings, "parent_chunk_overlap", None) or settings.chunk_overlap
        child_overlap = getattr(settings, "child_chunk_overlap", None) or settings.chunk_overlap

        self.heading_keys = ["h1", "h2", "h3", "h4", "h5", "h6"]
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
                ("#####", "h5"),
                ("######", "h6"),
            ]
        )
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.parent_chunk_size,
            chunk_overlap=parent_overlap,
            length_function=len,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=child_overlap,
            length_function=len,
        )
    
    def _segment_sections(self, text: str) -> List[Dict[str, str]]:
        """Segment document into sections based on markdown headings."""
        documents = self.header_splitter.split_text(text)
        sections: List[Dict[str, str]] = []

        for doc in documents:
            content = doc.page_content.strip()
            if not content:
                continue

            metadata = doc.metadata or {}
            heading_parts = [
                metadata.get(level)
                for level in self.heading_keys
                if metadata.get(level)
            ]
            heading_path = " / ".join(heading_parts) if heading_parts else "Body"

            sections.append({
                "heading": heading_path,
                "content": content,
            })

        if not sections:
            sections.append({"heading": "Body", "content": text.strip()})

        return sections

    def _build_parent_chunks(
        self, sections: List[Dict[str, str]]
    ) -> Tuple[List[str], List[str]]:
        """Build parent chunks while respecting section boundaries."""
        parent_docs: List[str] = []
        parent_sections: List[str] = []

        for section in sections:
            content = section.get("content", "").strip()
            heading_path = section.get("heading", "Body")

            if not content:
                continue

            splits = self.parent_splitter.split_text(content)
            for split in splits:
                chunk_text = (
                    f"{heading_path}\n\n{split}".strip()
                    if heading_path != "Body"
                    else split.strip()
                )
                parent_docs.append(chunk_text)
                parent_sections.append(heading_path)

        return parent_docs, parent_sections
    
    def _determine_position(self, chunk_index: int, total_chunks: int) -> str:
        """Determine if chunk is at beginning, middle, or end of document"""
        if total_chunks <= 1:
            return "full"
        
        relative_pos = chunk_index / total_chunks
        if relative_pos < 0.2:
            return "beginning"
        elif relative_pos > 0.8:
            return "end"
        else:
            return "middle"
    
    
    def process_document(
        self, 
        doc_id: int, 
        text: str, 
        pickle_path: str,
        document_name: str = "",
        metadata_chunk: str = None
    ) -> List[Dict[str, Any]]:
        """
        Process document using parent-child chunking strategy with metadata
        
        Args:
            doc_id: Document ID
            text: Full document text
            pickle_path: Path to save parent documents
            document_name: Original filename for metadata
            metadata_chunk: Optional pre-extracted metadata chunk to prepend
            
        Returns:
            List of child chunks with parent_id references and metadata
        """
        # Extract section structure (works best for markdown, but provides fallback for others)
        sections = self._segment_sections(text)
        
        # Create parent documents while keeping heading metadata
        parent_docs, parent_sections = self._build_parent_chunks(sections)
        
        # If we have a metadata chunk, prepend it as parent_id -1 (special metadata parent)
        if metadata_chunk:
            parent_docs_with_meta = [metadata_chunk] + parent_docs
        else:
            parent_docs_with_meta = parent_docs
        
        # Save parent documents to pickle file (including metadata chunk)
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        with open(pickle_path, 'wb') as f:
            pickle.dump(parent_docs_with_meta, f)
        
        # Create child chunks with metadata
        chunks = []
        chunk_counter = 0
        
        # Add metadata chunk as a special searchable chunk (if present)
        if metadata_chunk:
            chunks.append({
                'text': metadata_chunk,
                'parent_id': 0,  # Metadata is parent 0
                'doc_id': doc_id,
                'document_name': document_name,
                'section': 'Document Metadata',
                'position': 'metadata',
                'chunk_index': chunk_counter,
                'chunk_id': -1,  # Ensure metadata chunk has unique chunk_id
                'is_metadata': True
            })
            # Offset for regular chunks
            parent_offset = 1
            chunk_counter += 1
        else:
            parent_offset = 0
        
        total_parent_docs = len(parent_docs)
        
        for parent_id, parent_text in enumerate(parent_docs):
            child_texts = self.child_splitter.split_text(parent_text)
            
            section = (
                parent_sections[parent_id]
                if parent_id < len(parent_sections)
                else "Body"
            )
            position = self._determine_position(parent_id, total_parent_docs)

            for child_text in child_texts:
                chunks.append({
                    'text': child_text,
                    'parent_id': parent_id + parent_offset,  # Offset by 1 if metadata exists
                    'doc_id': doc_id,
                    'document_name': document_name,
                    'section': section,
                    'position': position,
                    'chunk_index': chunk_counter,
                    'is_metadata': False
                })
                chunk_counter += 1
        
        logger.info(
            "Processed document %s: %d parent chunks, %d child chunks%s",
            document_name, len(parent_docs), len(chunks),
            " (with metadata)" if metadata_chunk else ""
        )
        
        return chunks
    
    def load_parent_document(self, pickle_path: str, parent_id: int) -> str:
        """Load a specific parent document from pickle file"""
        try:
            with open(pickle_path, 'rb') as f:
                parent_docs = pickle.load(f)
            return parent_docs[parent_id] if parent_id < len(parent_docs) else ""
        except Exception as e:
            logger.error(f"Error loading parent document from {pickle_path}: {e}")
            return ""
