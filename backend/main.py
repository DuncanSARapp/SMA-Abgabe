import threading
from contextlib import asynccontextmanager
import json
import logging
import os
import queue
import uvicorn
from typing import List, Dict, Optional
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import InstrumentedAttribute

from db.session import init_db, get_db, SessionLocal
from services.settings import settings
from db.models import Chat, Message, Document
from models.schemas import (
    ChatCreate, ChatResponse, MessageResponse,
    QueryRequest, DocumentUploadResponse,
    DocumentPreferenceUpdate
)
from services.embeddings import EmbeddingService
from services.vector_store import VectorStoreService
from services.reranker import RerankerService
from services.document_processor import DocumentProcessor
from services.rag_service import RAGService
from services.file_handler import FileHandler
from services.metadata_extractor import MetadataExtractor, create_metadata_chunk

logger = logging.getLogger(__name__)

embedding_service: Optional[EmbeddingService] = None
vector_store_service: Optional[VectorStoreService] = None
reranker_service: Optional[RerankerService] = None
doc_processor: Optional[DocumentProcessor] = None
rag_service: Optional[RAGService] = None
metadata_extractor: Optional[MetadataExtractor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_service, vector_store_service, reranker_service
    global doc_processor, rag_service, metadata_extractor

    logger.info("Initializing RAG System services...")

    init_db()
    settings.ensure_directories()

    embedding_service = EmbeddingService.get_instance()
    vector_store_service = VectorStoreService(embedding_service)
    reranker_service = RerankerService.get_instance()
    doc_processor = DocumentProcessor()
    rag_service = RAGService(vector_store_service, reranker_service, doc_processor)
    metadata_extractor = MetadataExtractor()

    _sync_documents_with_qdrant()

    logger.info("RAG System initialized successfully")

    yield

    logger.info("Shutting down RAG System...")


def _sync_documents_with_qdrant() -> None:
    db = SessionLocal()
    try:
        documents = db.query(Document).all()
        synced_count = 0
        valid_collections: set[str] = set()

        for doc in documents:
            collection_name = doc.collection_name
            if collection_name:
                valid_collections.add(collection_name)

            if doc.processed and not vector_store_service.document_exists(collection_name):
                logger.warning(
                    f"Document {doc.id} ({doc.filename}) missing in Qdrant, marking as unprocessed"
                )
                doc.processed = False
                doc.num_chunks = 0
                synced_count += 1

        if synced_count > 0:
            db.commit()
            logger.info(f"Synced {synced_count} documents with Qdrant")

        vector_store_service.cleanup_orphaned_collections(valid_collections)

    except Exception as exc:
        logger.exception(f"Failed to sync documents with Qdrant: {exc}")
        db.rollback()
    finally:
        db.close()

app = FastAPI(
    title="RAG System API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_active_doc_collection_map(db: Session) -> Dict[int, str]:
    active_documents = db.query(Document).filter(
        Document.processed == True,
        Document.query_enabled == True
    ).all()
    return vector_store_service.build_collection_map(active_documents)


def _format_sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _extract_metadata_for_document(file_path: str, filename: str) -> Optional[str]:
    try:
        logger.info(f"Extracting metadata for document: {filename}")

        first_pages_text = FileHandler.extract_first_pages_text(file_path, num_pages=2)

        pdf_metadata = None
        if filename.lower().endswith('.pdf'):
            pdf_metadata = FileHandler.extract_pdf_metadata(file_path)

        extracted_metadata = metadata_extractor.extract_metadata_from_text(
            first_pages_text,
            filename,
            pdf_metadata
        )

        metadata_chunk = create_metadata_chunk(extracted_metadata, filename)

        logger.info(f"Extracted metadata for {filename}: {extracted_metadata}")
        return metadata_chunk

    except Exception as exc:
        logger.warning(f"Failed to extract metadata for {filename}: {exc}")
        return None


def _process_document_pipeline(
        document: Document,
        file_path: str
) -> Document:
    logger.info(f"Starting document processing pipeline for doc_id={document.id}, filename={document.filename}")

    try:
        logger.debug(f"Extracting text from {file_path}")
        text = FileHandler.extract_text(file_path)
        logger.info(f"Extracted {len(text)} characters from {document.filename}")

        metadata_chunk = _extract_metadata_for_document(file_path, document.filename)

        pickle_path = os.path.join(settings.pickle_dir, f"doc_{document.id}.pkl")
        collection_name = document.collection_name

        logger.info(f"Processing document with collection_name={collection_name}, pickle_path={pickle_path}")

        if not collection_name:
            raise ValueError(f"Invalid collection_name for document {document.id}: {collection_name}")

        chunks = doc_processor.process_document(
            document.id,
            text,
            pickle_path,
            document_name=document.filename,
            metadata_chunk=metadata_chunk
        )
        logger.info(f"Document processing created {len(chunks)} chunks for {document.filename}")

        logger.info(f"Resetting collection {collection_name}")
        vector_store_service.reset_collection(collection_name)

        logger.info(f"Adding {len(chunks)} chunks to collection {collection_name}")
        vector_store_service.add_documents(
            document.id,
            chunks,
            collection_name,
            document_name=document.filename
        )

        document.pickle_path = pickle_path
        document.processed = True
        document.num_chunks = len(chunks)

        logger.info(f"Successfully completed processing for doc_id={document.id}, {len(chunks)} chunks in {collection_name}")

        return document

    except Exception as exc:
        logger.error(
            f"Document processing pipeline failed for doc_id={document.id}, filename={document.filename}: "
            f"{type(exc).__name__}: {exc}",
            exc_info=True
        )
        raise


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "RAG System API", "version": "1.0.0"}


@app.post("/chats", response_model=ChatResponse, tags=["Chats"])
async def create_chat(chat: ChatCreate, db: Session = Depends(get_db)):
    db_chat = Chat(title=chat.title)
    db.add(db_chat)
    db.commit()
    db.refresh(db_chat)
    return db_chat


@app.get("/chats", response_model=List[ChatResponse], tags=["Chats"])
async def list_chats(db: Session = Depends(get_db)):
    return db.query(Chat).order_by(Chat.updated_at.desc()).all()


@app.get("/chats/{chat_id}", response_model=ChatResponse, tags=["Chats"])
async def get_chat(chat_id: int, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@app.delete("/chats/{chat_id}", tags=["Chats"])
async def delete_chat(chat_id: int, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    db.delete(chat)
    db.commit()
    return {"status": "deleted"}


@app.get("/chats/{chat_id}/messages", response_model=List[MessageResponse], tags=["Messages"])
async def get_messages(chat_id: int, db: Session = Depends(get_db)):
    return (
        db.query(Message)
        .filter(Message.chat_id == chat_id)
        .order_by(Message.created_at)
        .all()
    )


@app.post("/documents", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(
        file: UploadFile = File(...),
        db: Session = Depends(get_db)
):
    logger.info(f"Received document upload request: {file.filename}")

    try:
        logger.debug(f"Saving uploaded file {file.filename} to {settings.upload_dir}")
        file_path = FileHandler.save_upload(file.file, file.filename, settings.upload_dir)
        logger.info(f"File saved to {file_path}")

        db_document = Document(
            filename=file.filename,
            file_path=file_path,
            processed=False
        )
        db.add(db_document)
        db.commit()
        db.refresh(db_document)

        logger.info(f"Document created in database with id={db_document.id}, collection_name={db_document.collection_name}")

        db_document = _process_document_pipeline(db_document, file_path)

        db.commit()
        db.refresh(db_document)

        logger.info(f"Successfully uploaded and processed document {file.filename} with id={db_document.id}")

        return db_document

    except Exception as exc:
        db.rollback()
        logger.error(
            f"Failed to upload/process document {file.filename}: {type(exc).__name__}: {exc}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(exc)}"
        )


@app.get("/documents", response_model=List[DocumentUploadResponse], tags=["Documents"])
async def list_documents(db: Session = Depends(get_db)):
    return db.query(Document).order_by(Document.uploaded_at.desc()).all()


@app.post("/documents/{doc_id}/reprocess", response_model=DocumentUploadResponse, tags=["Documents"])
async def reprocess_document(doc_id: int, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == doc_id).first()  # type: ignore[arg-type]
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if not document.file_path or not os.path.exists(document.file_path):  # type: ignore[arg-type]
        raise HTTPException(status_code=400, detail="Document file not found on disk")

    try:
        document = _process_document_pipeline(document, document.file_path)  # type: ignore[arg-type]
        db.commit()
        db.refresh(document)
        return document

    except Exception as exc:
        db.rollback()
        logger.exception(f"Failed to reprocess document {document.filename}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reprocessing document: {str(exc)}"
        )


@app.patch("/documents/{doc_id}/preferences", response_model=DocumentUploadResponse, tags=["Documents"])
async def update_document_preferences(
        doc_id: int,
        preferences: DocumentPreferenceUpdate,
        db: Session = Depends(get_db)
):
    document = db.query(Document).filter(Document.id == doc_id).first()  # type: ignore[arg-type]
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    document.query_enabled = preferences.query_enabled
    db.commit()
    db.refresh(document)

    return document

@app.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: int, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == doc_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    collection_name = document.collection_name
    pickle_path = document.pickle_path
    file_path = document.file_path

    # Lösche aus Datenbank zuerst (wichtigster Schritt)
    try:
        db.delete(document)
        db.commit()
        logger.info(f"Deleted document {doc_id} from database")
    except Exception as exc:
        db.rollback()
        logger.exception(f"Failed to delete document {doc_id} from database: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document from database: {str(exc)}"
        )

    # Cleanup - Fehler hier sind nicht kritisch
    try:
        vector_store_service.delete_document(collection_name)
        logger.info(f"Deleted collection {collection_name}")
    except Exception as exc:
        logger.warning(f"Failed to delete collection {collection_name}: {exc}")

    try:
        FileHandler.delete_file(pickle_path)
        logger.info(f"Deleted pickle file {pickle_path}")
    except Exception as exc:
        logger.warning(f"Failed to delete pickle {pickle_path}: {exc}")

    try:
        FileHandler.delete_file(file_path)
        logger.info(f"Deleted file {file_path}")
    except Exception as exc:
        logger.warning(f"Failed to delete file {file_path}: {exc}")

    return {"status": "deleted"}

@app.post("/query/stream", tags=["Query"])
async def query_documents_stream(request: QueryRequest, db: Session = Depends(get_db)):
    try:
        chat = db.query(Chat).filter(Chat.id == request.chat_id).first()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        messages = (
            db.query(Message)
            .filter(Message.chat_id == request.chat_id)
            .order_by(Message.created_at)
            .all()
        )
        chat_history: list[dict[str, InstrumentedAttribute[str]]] = [{"role": msg.role, "content": msg.content} for msg in messages]

        doc_collection_map = _get_active_doc_collection_map(db)
        if not doc_collection_map:
            raise HTTPException(
                status_code=400,
                detail="No active documents selected for querying."
            )

        user_message = Message(
            chat_id=request.chat_id,
            content=request.query,
            role="user"
        )
        db.add(user_message)
        db.commit()

    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        logger.exception(f"Failed to prepare streaming response: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(exc)}"
        )

    thinking_queue: queue.Queue = queue.Queue()
    retrieval_result = {
        "contexts": [],
        "sources": [],
        "thinking_steps": [],
        "error": None
    }
    retrieval_done = threading.Event()

    def run_retrieval():
        try:
            def on_thinking(step):
                thinking_queue.put(("thinking", step))

            contexts, sources, thinking_steps = rag_service.multi_query_retrieve_and_rerank(
                request.query,
                db,
                doc_collection_map,
                on_thinking=on_thinking
            )

            retrieval_result["contexts"] = contexts
            retrieval_result["sources"] = sources
            retrieval_result["thinking_steps"] = thinking_steps

        except Exception as exception:
            logger.exception(f"Retrieval failed: {exception}")
            retrieval_result["error"] = str(exception)
        finally:
            retrieval_done.set()
            thinking_queue.put(("done", None))
    if retrieval_result["error"]:
        raise Exception(retrieval_result["error"])

    retrieval_thread = threading.Thread(target=run_retrieval, daemon=True)
    retrieval_thread.start()

    def event_generator():
        accumulated_answer = ""
        try:
            while True:
                try:
                    event_type, data = thinking_queue.get(timeout=0.1)
                    if event_type == "done":
                        break
                    elif event_type == "thinking":
                        yield _format_sse_event({"type": "thinking", "step": data})
                except queue.Empty:
                    if retrieval_done.is_set():
                        while not thinking_queue.empty():
                            try:
                                event_type, data = thinking_queue.get_nowait()
                                if event_type == "thinking":
                                    yield _format_sse_event({"type": "thinking", "step": data})
                            except queue.Empty:
                                break
                        break

            retrieval_thread.join(timeout=60)
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
                yield _format_sse_event({"type": "chunk", "content": token})

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
            logger.exception(f"Streaming query failed: {stream_error}")
            yield _format_sse_event({"type": "error", "message": "Error processing query"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
