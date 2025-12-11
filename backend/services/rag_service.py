import os
from typing import List, Dict, Any, Iterator
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from config.settings import settings
from services.embeddings import VectorStoreService
from services.reranker import RerankerService
from services.document_processor import DocumentProcessor
from sqlalchemy.orm import Session
from models.database import Document


class RAGService:
    """Service for RAG-based question answering"""
    
    def __init__(
        self,
        vector_store: VectorStoreService,
        reranker: RerankerService,
        doc_processor: DocumentProcessor
    ):
        self.vector_store = vector_store
        self.reranker = reranker
        self.doc_processor = doc_processor
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.llm_api_key,
            openai_api_base=settings.llm_api_base,
            temperature=0.7,
            streaming=True
        )
    
    def retrieve_and_rerank(self, query: str, db: Session) -> tuple[List[str], List[Dict[str, str]]]:
        """
        Retrieve relevant chunks, rerank them, and return parent documents
        
        Returns:
            Tuple of (parent_contexts, source_descriptions)
        """
        # Step 1: Retrieve child chunks from vector store
        retrieved_chunks = self.vector_store.search(query, top_k=settings.top_k_retrieval)
        
        if not retrieved_chunks:
            return [], []
        
        # Step 2: Rerank chunks
        reranked_chunks = self.reranker.rerank(query, retrieved_chunks, top_k=settings.top_k_rerank)
        
        # Step 3: Load parent documents
        parent_contexts: List[str] = []
        sources: List[Dict[str, str]] = []
        seen_parents = set()
        
        for chunk in reranked_chunks:
            doc_id = chunk['doc_id']
            parent_id = chunk.get('parent_id')
            
            # Avoid duplicate parents
            parent_key = f"{doc_id}_{parent_id}"
            if parent_key in seen_parents:
                continue
            seen_parents.add(parent_key)
            
            # Get document from database
            document = db.query(Document).filter(Document.id == doc_id).first()
            if not document or not document.pickle_path:
                continue
            
            # Load parent document from pickle
            parent_text = self.doc_processor.load_parent_document(
                document.pickle_path, 
                parent_id
            )
            
            if parent_text:
                parent_contexts.append(parent_text)
                sources.append({
                    "label": f"{document.filename} (chunk {parent_id})",
                    "content": parent_text.strip()
                })
        
        return parent_contexts, sources
    
    def _build_messages(self, query: str, contexts: List[str], chat_history: List[Dict[str, str]] | None = None) -> List[Any]:
        """Build messages for the LLM including context and history."""
        context_str = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])

        messages: List[Any] = [
            SystemMessage(content=(
                "You are a helpful assistant that answers questions based on the provided context. "
                "Use the context to answer the question accurately. If the context doesn't contain "
                "enough information to answer the question, say so."
            ))
        ]

        if chat_history:
            for msg in chat_history[-5:]:
                if msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    messages.append(AIMessage(content=msg['content']))

        user_message = f"Context:\n{context_str}\n\nQuestion: {query}"
        messages.append(HumanMessage(content=user_message))
        return messages

    def generate_answer(self, query: str, contexts: List[str], chat_history: List[Dict[str, str]] = None) -> str:
        """
        Generate answer using LLM based on retrieved contexts
        
        Args:
            query: User question
            contexts: List of relevant context strings
            chat_history: Optional list of previous messages
            
        Returns:
            Generated answer
        """
        messages = self._build_messages(query, contexts, chat_history)
        response = self.llm.invoke(messages)
        return response.content

    def generate_answer_stream(
        self,
        query: str,
        contexts: List[str],
        chat_history: List[Dict[str, str]] | None = None
    ) -> Iterator[str]:
        """Stream answer tokens from the LLM."""
        messages = self._build_messages(query, contexts, chat_history)
        for chunk in self.llm.stream(messages):
            text = ""
            if hasattr(chunk, "message") and getattr(chunk.message, "content", None):
                text = chunk.message.content
            elif hasattr(chunk, "content") and chunk.content:
                text = chunk.content
            elif hasattr(chunk, "delta") and isinstance(chunk.delta, dict):
                text = chunk.delta.get("content", "")

            if text:
                yield text
    
    def query(self, query: str, db: Session, chat_history: List[Dict[str, str]] = None) -> tuple[str, List[str]]:
        """
        Main RAG query method
        
        Returns:
            Tuple of (answer, sources)
        """
        # Retrieve and rerank
        contexts, sources = self.retrieve_and_rerank(query, db)
        
        if not contexts:
            return "I couldn't find relevant information in the documents to answer your question.", []
        
        # Generate answer
        answer = self.generate_answer(query, contexts, chat_history)
        
        return answer, sources
