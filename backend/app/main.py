from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db, init_db
from models.database import Chat, Message, Document
from models.schemas import (
    ChatCreate, ChatResponse, MessageResponse, 
    QueryRequest, QueryResponse, DocumentUploadResponse
)
from config.settings import settings
from services.embeddings import EmbeddingService, VectorStoreService
from services.reranker import RerankerService
from services.document_processor import DocumentProcessor
from services.rag_service import RAGService
from services.file_handler import FileHandler

# Initialize services
embedding_service = None
vector_store_service = None
reranker_service = None
doc_processor = None
rag_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global embedding_service, vector_store_service, reranker_service, doc_processor, rag_service
    
    # Startup
    init_db()
    os.makedirs(settings.data_dir, exist_ok=True)
    
    embedding_service = EmbeddingService()
    vector_store_service = VectorStoreService(embedding_service)
    reranker_service = RerankerService()
    doc_processor = DocumentProcessor()
    rag_service = RAGService(vector_store_service, reranker_service, doc_processor)
    
    yield
    
    # Shutdown (cleanup if needed)


# Initialize FastAPI app
app = FastAPI(title="RAG System API", version="1.0.0", lifespan=lifespan)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "RAG System API"}


# Chat endpoints
@app.post("/chats", response_model=ChatResponse)
async def create_chat(chat: ChatCreate, db: Session = Depends(get_db)):
    """Create a new chat"""
    db_chat = Chat(title=chat.title)
    db.add(db_chat)
    db.commit()
    db.refresh(db_chat)
    return db_chat


@app.get("/chats", response_model=List[ChatResponse])
async def list_chats(db: Session = Depends(get_db)):
    """List all chats"""
    chats = db.query(Chat).order_by(Chat.updated_at.desc()).all()
    return chats


@app.get("/chats/{chat_id}", response_model=ChatResponse)
async def get_chat(chat_id: int, db: Session = Depends(get_db)):
    """Get a specific chat"""
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: int, db: Session = Depends(get_db)):
    """Delete a chat"""
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    db.delete(chat)
    db.commit()
    return {"status": "deleted"}


# Message endpoints
@app.get("/chats/{chat_id}/messages", response_model=List[MessageResponse])
async def get_messages(chat_id: int, db: Session = Depends(get_db)):
    """Get all messages for a chat"""
    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.created_at).all()
    return messages


# Document endpoints
@app.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process a document"""
    try:
        # Save file
        upload_dir = os.path.join(settings.data_dir, "uploads")
        file_path = FileHandler.save_upload(file.file, file.filename, upload_dir)
        
        # Create document record
        db_document = Document(
            filename=file.filename,
            file_path=file_path,
            processed=False
        )
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        
        # Extract text
        text = FileHandler.extract_text(file_path)
        
        # Process document
        pickle_path = os.path.join(settings.data_dir, "pickles", f"doc_{db_document.id}.pkl")
        chunks = doc_processor.process_document(db_document.id, text, pickle_path)
        
        # Add to vector store
        vector_store_service.add_documents(db_document.id, chunks)
        
        # Update document record
        db_document.pickle_path = pickle_path
        db_document.processed = True
        db_document.num_chunks = len(chunks)
        db.commit()
        db.refresh(db_document)
        
        return db_document
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.get("/documents", response_model=List[DocumentUploadResponse])
async def list_documents(db: Session = Depends(get_db)):
    """List all uploaded documents"""
    documents = db.query(Document).order_by(Document.uploaded_at.desc()).all()
    return documents


# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, db: Session = Depends(get_db)):
    """Query documents using RAG"""
    try:
        # Verify chat exists
        chat = db.query(Chat).filter(Chat.id == request.chat_id).first()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Get chat history
        messages = db.query(Message).filter(Message.chat_id == request.chat_id).order_by(Message.created_at).all()
        chat_history = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        # Save user message
        user_message = Message(
            chat_id=request.chat_id,
            content=request.query,
            role="user"
        )
        db.add(user_message)
        db.commit()
        
        # Query RAG system
        answer, sources = rag_service.query(request.query, db, chat_history)
        
        # Save assistant message
        assistant_message = Message(
            chat_id=request.chat_id,
            content=answer,
            role="assistant"
        )
        db.add(assistant_message)
        db.commit()
        db.refresh(assistant_message)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            message_id=assistant_message.id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
