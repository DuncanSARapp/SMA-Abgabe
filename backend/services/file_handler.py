import os
import shutil
from typing import BinaryIO
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
