import os
import pickle
import re
import logging
from typing import List, Dict, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Service for processing and chunking documents with metadata extraction"""
    
    def __init__(self):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.parent_chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
    
    def _extract_sections_from_markdown(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract sections from markdown text with their headings.
        Returns list of (section_heading, section_content) tuples.
        """
        # Pattern to match markdown headings (# to ######)
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')
        sections = []
        current_heading = "Introduction"
        current_content = []
        
        for line in lines:
            match = re.match(heading_pattern, line)
            if match:
                # Save previous section
                if current_content:
                    sections.append((current_heading, '\n'.join(current_content)))
                current_heading = match.group(2).strip()
                current_content = [line]
            else:
                current_content.append(line)
        
        # Don't forget the last section
        if current_content:
            sections.append((current_heading, '\n'.join(current_content)))
        
        return sections
    
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
    
    def _find_section_for_text(self, text: str, sections: List[Tuple[str, str]]) -> str:
        """Find which section a text chunk belongs to"""
        for heading, content in sections:
            if text[:100] in content:  # Check if start of chunk is in section
                return heading
        return "Unknown"
    
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
        sections = self._extract_sections_from_markdown(text)
        
        # Create parent documents
        parent_docs = self.parent_splitter.split_text(text)
        
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
        
        # Add metadata chunk as a special searchable chunk (if present)
        if metadata_chunk:
            chunks.append({
                'text': metadata_chunk,
                'parent_id': 0,  # Metadata is parent 0
                'doc_id': doc_id,
                'document_name': document_name,
                'section': 'Document Metadata',
                'position': 'metadata',
                'chunk_index': 0,
                'chunk_id': -1,  # Ensure metadata chunk has unique chunk_id
                'is_metadata': True
            })
            # Offset for regular chunks
            parent_offset = 1
        else:
            parent_offset = 0
        
        total_parent_docs = len(parent_docs)
        
        for parent_id, parent_text in enumerate(parent_docs):
            child_texts = self.child_splitter.split_text(parent_text)
            
            # Find section for this parent
            section = self._find_section_for_text(parent_text, sections)
            
            for child_idx, child_text in enumerate(child_texts):
                global_chunk_index = sum(
                    len(self.child_splitter.split_text(parent_docs[i])) 
                    for i in range(parent_id)
                ) + child_idx + (1 if metadata_chunk else 0)
                
                chunks.append({
                    'text': child_text,
                    'parent_id': parent_id + parent_offset,  # Offset by 1 if metadata exists
                    'doc_id': doc_id,
                    'document_name': document_name,
                    'section': section,
                    'position': self._determine_position(parent_id, total_parent_docs),
                    'chunk_index': global_chunk_index,
                    'is_metadata': False
                })
        
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
