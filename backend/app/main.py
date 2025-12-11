from contextlib import asynccontextmanager
import json
import logging
import os
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Dict

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
from services.metadata_extractor import MetadataExtractor

# Initialize services
embedding_service = None
vector_store_service = None
reranker_service = None
doc_processor = None
rag_service = None
metadata_extractor = None

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global embedding_service, vector_store_service, reranker_service, doc_processor, rag_service, metadata_extractor
    
    # Startup
    init_db()
    os.makedirs(settings.data_dir, exist_ok=True)
    
    embedding_service = EmbeddingService()
    vector_store_service = VectorStoreService(embedding_service)
    reranker_service = RerankerService()
    doc_processor = DocumentProcessor()
    rag_service = RAGService(vector_store_service, reranker_service, doc_processor)
    metadata_extractor = MetadataExtractor()
    
    # Sync document status with Qdrant on startup
    sync_documents_with_qdrant()
    
    yield
    
    # Shutdown (cleanup if needed)


def sync_documents_with_qdrant():
    """Mark documents as unprocessed if they're not in Qdrant anymore"""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        documents = db.query(Document).filter(Document.processed == True).all()
        synced_count = 0
        for doc in documents:
            if not vector_store_service.document_exists(doc.id):
                logger.info("Document %s (%s) not found in Qdrant, marking as unprocessed", doc.id, doc.filename)
                doc.processed = False
                doc.num_chunks = 0
                synced_count += 1
        if synced_count > 0:
            db.commit()
            logger.info("Synced %d documents with Qdrant status", synced_count)
    except Exception as e:
        logger.exception("Failed to sync documents with Qdrant: %s", e)
        db.rollback()
    finally:
        db.close()


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
        
        # Extract metadata using LLM
        logger.info("Extracting metadata for document %s", file.filename)
        metadata_chunk = None
        try:
            # Get first pages for metadata extraction
            first_pages_text = FileHandler.extract_first_pages_text(file_path, num_pages=2)
            
            # Get PDF metadata if available
            pdf_metadata = None
            if file.filename.lower().endswith('.pdf'):
                pdf_metadata = FileHandler.extract_pdf_metadata(file_path)
            
            # Extract structured metadata via LLM
            extracted_metadata = metadata_extractor.extract_metadata_from_text(
                first_pages_text, 
                file.filename,
                pdf_metadata
            )
            
            # Create searchable metadata chunk
            metadata_chunk = metadata_extractor.create_metadata_chunk(
                extracted_metadata, 
                file.filename
            )
            logger.info("Extracted metadata for %s: %s", file.filename, extracted_metadata)
        except Exception as meta_error:
            logger.warning("Failed to extract metadata for %s: %s", file.filename, meta_error)
            # Continue without metadata - it's not critical
        
        # Process document with metadata chunk
        pickle_path = os.path.join(settings.data_dir, "pickles", f"doc_{db_document.id}.pkl")
        chunks = doc_processor.process_document(
            db_document.id, 
            text, 
            pickle_path,
            document_name=file.filename,
            metadata_chunk=metadata_chunk
        )
        
        # Add to vector store with document name
        vector_store_service.add_documents(db_document.id, chunks, document_name=file.filename)
        
        # Update document record
        db_document.pickle_path = pickle_path
        db_document.processed = True
        db_document.num_chunks = len(chunks)
        db.commit()
        db.refresh(db_document)
        
        return db_document
        
    except Exception as e:
        db.rollback()
        logger.exception("Failed to process document %s: %s", file.filename, e)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.get("/documents", response_model=List[DocumentUploadResponse])
async def list_documents(db: Session = Depends(get_db)):
    """List all uploaded documents"""
    documents = db.query(Document).order_by(Document.uploaded_at.desc()).all()
    return documents


@app.post("/documents/{doc_id}/reprocess", response_model=DocumentUploadResponse)
async def reprocess_document(doc_id: int, db: Session = Depends(get_db)):
    """Reprocess a document that is no longer in Qdrant"""
    document = db.query(Document).filter(Document.id == doc_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if not document.file_path or not os.path.exists(document.file_path):
        raise HTTPException(status_code=400, detail="Document file not found on disk")
    
    try:
        # Extract text
        text = FileHandler.extract_text(document.file_path)
        
        # Extract metadata using LLM
        logger.info("Extracting metadata for reprocessed document %s", document.filename)
        metadata_chunk = None
        try:
            first_pages_text = FileHandler.extract_first_pages_text(document.file_path, num_pages=2)
            pdf_metadata = None
            if document.filename.lower().endswith('.pdf'):
                pdf_metadata = FileHandler.extract_pdf_metadata(document.file_path)
            
            extracted_metadata = metadata_extractor.extract_metadata_from_text(
                first_pages_text, 
                document.filename,
                pdf_metadata
            )
            metadata_chunk = metadata_extractor.create_metadata_chunk(
                extracted_metadata, 
                document.filename
            )
        except Exception as meta_error:
            logger.warning("Failed to extract metadata for %s: %s", document.filename, meta_error)
        
        # Process document with metadata
        pickle_path = os.path.join(settings.data_dir, "pickles", f"doc_{document.id}.pkl")
        chunks = doc_processor.process_document(
            document.id, 
            text, 
            pickle_path,
            document_name=document.filename,
            metadata_chunk=metadata_chunk
        )
        
        # Add to vector store with document name
        vector_store_service.add_documents(document.id, chunks, document_name=document.filename)
        
        # Update document record
        document.pickle_path = pickle_path
        document.processed = True
        document.num_chunks = len(chunks)
        db.commit()
        db.refresh(document)
        
        return document
        
    except Exception as e:
        db.rollback()
        logger.exception("Failed to reprocess document %s: %s", document.filename, e)
        raise HTTPException(status_code=500, detail=f"Error reprocessing document: {str(e)}")


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: int, db: Session = Depends(get_db)):
    """Delete a document from database and Qdrant"""
    document = db.query(Document).filter(Document.id == doc_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Delete from Qdrant
        vector_store_service.delete_document(doc_id)
        
        # Delete pickle file if exists
        if document.pickle_path and os.path.exists(document.pickle_path):
            os.remove(document.pickle_path)
        
        # Delete uploaded file if exists
        if document.file_path and os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        # Delete from database
        db.delete(document)
        db.commit()
        
        return {"status": "deleted"}
        
    except Exception as e:
        db.rollback()
        logger.exception("Failed to delete document %s: %s", doc_id, e)
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


# Query endpoint
def _format_sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


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


@app.post("/query/stream")
async def query_documents_stream(request: QueryRequest, db: Session = Depends(get_db)):
    """Stream query responses for incremental rendering on the frontend."""
    import queue
    import threading
    
    try:
        chat = db.query(Chat).filter(Chat.id == request.chat_id).first()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        messages = db.query(Message).filter(Message.chat_id == request.chat_id).order_by(Message.created_at).all()
        chat_history = [{"role": msg.role, "content": msg.content} for msg in messages]

        user_message = Message(
            chat_id=request.chat_id,
            content=request.query,
            role="user"
        )
        db.add(user_message)
        db.commit()

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception("Failed to prepare streaming response: %s", e)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

    # Queue for real-time thinking events
    thinking_queue: queue.Queue = queue.Queue()
    retrieval_result = {"contexts": [], "sources": [], "thinking_steps": [], "error": None}
    retrieval_done = threading.Event()

    def run_retrieval():
        """Run retrieval in background thread, pushing thinking events to queue."""
        try:
            def on_thinking(step):
                thinking_queue.put(("thinking", step))
            
            contexts, sources, thinking_steps = rag_service.multi_query_retrieve_and_rerank(
                request.query, db, on_thinking=on_thinking
            )
            retrieval_result["contexts"] = contexts
            retrieval_result["sources"] = sources
            retrieval_result["thinking_steps"] = thinking_steps
        except Exception as e:
            retrieval_result["error"] = str(e)
        finally:
            retrieval_done.set()
            thinking_queue.put(("done", None))

    # Start retrieval in background
    retrieval_thread = threading.Thread(target=run_retrieval)
    retrieval_thread.start()

    def event_generator():
        accumulated_answer = ""
        
        try:
            # First, stream thinking events as they come in real-time
            while True:
                try:
                    event_type, data = thinking_queue.get(timeout=0.1)
                    if event_type == "done":
                        break
                    elif event_type == "thinking":
                        yield _format_sse_event({
                            "type": "thinking",
                            "step": data
                        })
                except queue.Empty:
                    # Check if retrieval is done
                    if retrieval_done.is_set():
                        # Drain any remaining events
                        while not thinking_queue.empty():
                            try:
                                event_type, data = thinking_queue.get_nowait()
                                if event_type == "thinking":
                                    yield _format_sse_event({
                                        "type": "thinking",
                                        "step": data
                                    })
                            except queue.Empty:
                                break
                        break
            
            # Wait for retrieval thread to complete
            retrieval_thread.join(timeout=60)
            
            if retrieval_result["error"]:
                raise Exception(retrieval_result["error"])
            
            contexts = retrieval_result["contexts"]
            sources = retrieval_result["sources"]
            
            if not contexts:
                answer_text = "I couldn't find relevant information in the documents to answer your question."
                assistant_message = Message(
                    chat_id=request.chat_id,
                    content=answer_text,
                    role="assistant"
                )
                db.add(assistant_message)
                db.commit()
                db.refresh(assistant_message)
                yield _format_sse_event({
                    "type": "end",
                    "content": answer_text,
                    "sources": [],
                    "message_id": assistant_message.id
                })
                return

            for token in rag_service.generate_answer_stream(request.query, contexts, chat_history):
                if not token:
                    continue
                accumulated_answer += token
                yield _format_sse_event({
                    "type": "chunk",
                    "content": token
                })

            assistant_message = Message(
                chat_id=request.chat_id,
                content=accumulated_answer,
                role="assistant"
            )
            db.add(assistant_message)
            db.commit()
            db.refresh(assistant_message)

            yield _format_sse_event({
                "type": "end",
                "content": accumulated_answer,
                "sources": sources,
                "message_id": assistant_message.id
            })

        except Exception as stream_error:
            db.rollback()
            logger.exception("Streaming query failed: %s", stream_error)
            yield _format_sse_event({
                "type": "error",
                "message": "Error processing query"
            })

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
