import logging
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from config.settings import settings

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Service for extracting document metadata using LLM"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.llm_api_key,
            openai_api_base=settings.llm_api_base,
            temperature=0,  # Deterministic for metadata extraction
            streaming=False
        )
    
    def extract_metadata_from_text(
        self, 
        first_pages_text: str, 
        filename: str,
        pdf_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Extract structured metadata from document text using LLM.
        
        Args:
            first_pages_text: Text from the first 1-2 pages of the document
            filename: Original filename
            pdf_metadata: Optional PDF metadata dict
            
        Returns:
            Dict with extracted metadata fields
        """
        # Build context from PDF metadata if available
        pdf_context = ""
        if pdf_metadata:
            pdf_parts = []
            if pdf_metadata.get("title"):
                pdf_parts.append(f"PDF Title: {pdf_metadata['title']}")
            if pdf_metadata.get("author"):
                pdf_parts.append(f"PDF Author: {pdf_metadata['author']}")
            if pdf_metadata.get("subject"):
                pdf_parts.append(f"PDF Subject: {pdf_metadata['subject']}")
            if pdf_metadata.get("num_pages"):
                pdf_parts.append(f"Total Pages: {pdf_metadata['num_pages']}")
            if pdf_parts:
                pdf_context = "\n\nPDF Metadata:\n" + "\n".join(pdf_parts)
        
        messages = [
            SystemMessage(content="""You are a document metadata extractor. Analyze the provided document text and extract key metadata.

Extract the following information if available:
- Title: The title of the document/paper/article
- Author(s): Names of all authors (comma-separated)
- Institution(s): Universities, companies, or organizations
- Date/Year: Publication or creation date
- Abstract: A brief summary (if explicitly present)
- Keywords: Key topics or terms
- Document Type: paper, thesis, report, article, manual, etc.

IMPORTANT RULES:
1. Only extract information that is EXPLICITLY stated in the text
2. If information is not found, use "Not found" for that field
3. For authors, list ALL names you can find
4. Be precise - don't guess or infer

Respond in this exact format (keep the field names exactly as shown):
Title: [extracted title or "Not found"]
Author(s): [names or "Not found"]
Institution(s): [names or "Not found"]
Date/Year: [date or "Not found"]
Abstract: [abstract text or "Not found"]
Keywords: [keywords or "Not found"]
Document Type: [type or "Not found"]"""),
            HumanMessage(content=f"Filename: {filename}{pdf_context}\n\nDocument text (first pages):\n\n{first_pages_text}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            return self._parse_metadata_response(response.content, filename)
        except Exception as e:
            logger.error(f"Failed to extract metadata via LLM: {e}")
            return self._create_fallback_metadata(filename, pdf_metadata)
    
    def _parse_metadata_response(self, response: str, filename: str) -> Dict[str, str]:
        """Parse the LLM response into a structured dict"""
        metadata = {
            "title": "Not found",
            "authors": "Not found",
            "institutions": "Not found",
            "date": "Not found",
            "abstract": "Not found",
            "keywords": "Not found",
            "document_type": "Not found",
            "filename": filename
        }
        
        field_mapping = {
            "title:": "title",
            "author(s):": "authors",
            "author:": "authors",
            "institution(s):": "institutions",
            "institution:": "institutions",
            "date/year:": "date",
            "date:": "date",
            "year:": "date",
            "abstract:": "abstract",
            "keywords:": "keywords",
            "document type:": "document_type",
            "type:": "document_type"
        }
        
        lines = response.strip().split('\n')
        current_field = None
        current_value = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if this line starts a new field
            matched_field = None
            for prefix, field_name in field_mapping.items():
                if line_lower.startswith(prefix):
                    # Save previous field
                    if current_field and current_value:
                        metadata[current_field] = ' '.join(current_value).strip()
                    
                    # Start new field
                    current_field = field_name
                    value_part = line[len(prefix):].strip()
                    current_value = [value_part] if value_part else []
                    matched_field = field_name
                    break
            
            # If no field matched, this might be a continuation
            if not matched_field and current_field and line.strip():
                current_value.append(line.strip())
        
        # Don't forget the last field
        if current_field and current_value:
            metadata[current_field] = ' '.join(current_value).strip()
        
        return metadata
    
    def _create_fallback_metadata(
        self, 
        filename: str, 
        pdf_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Create fallback metadata when LLM extraction fails"""
        metadata = {
            "title": "Not found",
            "authors": "Not found",
            "institutions": "Not found",
            "date": "Not found",
            "abstract": "Not found",
            "keywords": "Not found",
            "document_type": "Not found",
            "filename": filename
        }
        
        if pdf_metadata:
            if pdf_metadata.get("title"):
                metadata["title"] = pdf_metadata["title"]
            if pdf_metadata.get("author"):
                metadata["authors"] = pdf_metadata["author"]
        
        return metadata
    
    def create_metadata_chunk(self, metadata: Dict[str, str], document_name: str) -> str:
        """
        Create a searchable text chunk from extracted metadata.
        This chunk will be added to the vector store for retrieval.
        """
        parts = [
            "=== DOCUMENT METADATA ===",
            f"Filename: {document_name}",
        ]
        
        if metadata.get("title") and metadata["title"] != "Not found":
            parts.append(f"Title: {metadata['title']}")
        
        if metadata.get("authors") and metadata["authors"] != "Not found":
            parts.append(f"Author(s): {metadata['authors']}")
            # Add alternative phrasings for better retrieval
            parts.append(f"This document was written by: {metadata['authors']}")
            parts.append(f"The author of this paper is: {metadata['authors']}")
        
        if metadata.get("institutions") and metadata["institutions"] != "Not found":
            parts.append(f"Institution(s): {metadata['institutions']}")
            parts.append(f"Affiliation: {metadata['institutions']}")
        
        if metadata.get("date") and metadata["date"] != "Not found":
            parts.append(f"Date/Year: {metadata['date']}")
            parts.append(f"Published: {metadata['date']}")
        
        if metadata.get("document_type") and metadata["document_type"] != "Not found":
            parts.append(f"Document Type: {metadata['document_type']}")
        
        if metadata.get("keywords") and metadata["keywords"] != "Not found":
            parts.append(f"Keywords: {metadata['keywords']}")
        
        if metadata.get("abstract") and metadata["abstract"] != "Not found":
            parts.append(f"\nAbstract:\n{metadata['abstract']}")
        
        parts.append("=== END METADATA ===")
        
        return "\n".join(parts)
