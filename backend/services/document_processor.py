import os
import pickle
import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Service for processing and chunking documents"""
    
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
    
    def process_document(self, doc_id: int, text: str, pickle_path: str) -> List[Dict[str, Any]]:
        """
        Process document using parent-child chunking strategy
        
        Args:
            doc_id: Document ID
            text: Full document text
            pickle_path: Path to save parent documents
            
        Returns:
            List of child chunks with parent_id references
        """
        # Create parent documents
        parent_docs = self.parent_splitter.split_text(text)
        
        # Save parent documents to pickle file
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        with open(pickle_path, 'wb') as f:
            pickle.dump(parent_docs, f)
        
        # Create child chunks
        chunks = []
        for parent_id, parent_text in enumerate(parent_docs):
            child_texts = self.child_splitter.split_text(parent_text)
            for child_text in child_texts:
                chunks.append({
                    'text': child_text,
                    'parent_id': parent_id,
                    'doc_id': doc_id
                })
        
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
