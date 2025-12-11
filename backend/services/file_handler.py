import os
import shutil
from typing import BinaryIO, Dict, Any, Optional
from pypdf import PdfReader
from docx import Document as DocxDocument


class FileHandler:
    """Utility for handling different file types"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    @staticmethod
    def extract_pdf_metadata(file_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF file (author, title, subject, etc.)"""
        try:
            reader = PdfReader(file_path)
            metadata = reader.metadata
            
            if metadata:
                return {
                    "title": metadata.get("/Title", "") or "",
                    "author": metadata.get("/Author", "") or "",
                    "subject": metadata.get("/Subject", "") or "",
                    "creator": metadata.get("/Creator", "") or "",
                    "producer": metadata.get("/Producer", "") or "",
                    "creation_date": str(metadata.get("/CreationDate", "")) or "",
                    "num_pages": len(reader.pages)
                }
        except Exception:
            pass
        
        return {"num_pages": 0}
    
    @staticmethod
    def extract_first_pages_text(file_path: str, num_pages: int = 2, max_chars: int = 3000) -> str:
        """Extract text from the first N pages of a PDF for metadata extraction"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            try:
                reader = PdfReader(file_path)
                text = ""
                for i, page in enumerate(reader.pages[:num_pages]):
                    text += page.extract_text() + "\n"
                    if len(text) > max_chars:
                        break
                return text[:max_chars]
            except Exception:
                return ""
        else:
            # For other file types, just return the beginning
            full_text = FileHandler.extract_text(file_path)
            return full_text[:max_chars]
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX file"""
        doc = DocxDocument(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from TXT file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    @staticmethod
    def extract_text(file_path: str) -> str:
        """Extract text based on file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return FileHandler.extract_text_from_pdf(file_path)
        elif ext == '.docx':
            return FileHandler.extract_text_from_docx(file_path)
        elif ext in ['.txt', '.md']:
            return FileHandler.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    @staticmethod
    def save_upload(file: BinaryIO, filename: str, upload_dir: str) -> str:
        """Save uploaded file to disk"""
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file, buffer)
        
        return file_path
