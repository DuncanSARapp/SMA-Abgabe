import threading
from contextlib import asynccontextmanager
import json
import logging
import os
import queue
import time
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

    logger.info("🚀 Initializing RAG System services...")

    init_db()
    settings.ensure_directories()

    embedding_service = EmbeddingService.get_instance()
    vector_store_service = VectorStoreService(embedding_service)
    reranker_service = RerankerService.get_instance()
    doc_processor = DocumentProcessor()
    rag_service = RAGService(vector_store_service, reranker_service, doc_processor)
    metadata_extractor = MetadataExtractor()

    _sync_documents_with_qdrant()

    logger.info("✅ RAG System initialized successfully")

    yield

    logger.info("👋 Shutting down RAG System...")


def _sync_documents_with_qdrant() -> None:
    db = SessionLocal()
    try:
        documents = db.query(Document).all()
        synced_count = 0
        valid_collections: set[str] = set()

        logger.info(f"🔄 Syncing {len(documents)} documents with Qdrant...")

        for doc in documents:
            collection_name = doc.collection_name
            if collection_name:
                valid_collections.add(collection_name)

            if doc.processed and not vector_store_service.document_exists(collection_name):
                logger.warning(
                    f"⚠️  Document {doc.id} ({doc.filename}) missing in Qdrant, marking as unprocessed"
                )
                doc.processed = False
                doc.num_chunks = 0
                synced_count += 1

        if synced_count > 0:
            db.commit()
            logger.info(f"🔄 Synced {synced_count} documents with Qdrant")

        vector_store_service.cleanup_orphaned_collections(valid_collections)
        logger.info(f"✅ Document sync complete ({len(documents)} documents, {len(valid_collections)} collections)")

    except Exception as exc:
        logger.exception(f"❌ Failed to sync documents with Qdrant: {exc}")
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
        logger.info(f"📋 [METADATA] Starting metadata extraction for: {filename}")
        start_time = time.time()

        # Extrahiere erste Seiten
        logger.info(f"   → Reading first 2 pages for metadata...")
        first_pages_text = FileHandler.extract_first_pages_text(file_path, num_pages=2)
        logger.info(f"   → Extracted {len(first_pages_text)} characters from first pages")

        # PDF Metadaten
        pdf_metadata = None
        if filename.lower().endswith('.pdf'):
            logger.info(f"   → Extracting PDF metadata...")
            pdf_metadata = FileHandler.extract_pdf_metadata(file_path)
            if pdf_metadata:
                logger.info(f"   → PDF metadata: {pdf_metadata.get('num_pages', 0)} pages, "
                            f"title='{pdf_metadata.get('title', 'N/A')}'")

        # LLM-basierte Metadata-Extraktion
        logger.info(f"   → Running LLM-based metadata extraction...")
        extracted_metadata = metadata_extractor.extract_metadata_from_text(
            first_pages_text,
            filename,
            pdf_metadata
        )

        metadata_chunk = create_metadata_chunk(extracted_metadata, filename)

        elapsed = time.time() - start_time
        logger.info(f"✅ [METADATA] Extraction complete in {elapsed:.1f}s")
        logger.info(f"   → Title: {extracted_metadata.get('title', 'N/A')}")
        logger.info(f"   → Author: {extracted_metadata.get('author', 'N/A')}")
        logger.info(f"   → Type: {extracted_metadata.get('document_type', 'N/A')}")

        return metadata_chunk

    except Exception as exc:
        logger.warning(f"⚠️  [METADATA] Extraction failed for {filename}: {exc}")
        return None


def _process_document_pipeline(
        document: Document,
        file_path: str
) -> Document:
    pipeline_start = time.time()

    logger.info("=" * 80)
    logger.info(f"📄 [PIPELINE START] Processing document")
    logger.info(f"   • Document ID: {document.id}")
    logger.info(f"   • Filename: {document.filename}")
    logger.info(f"   • Collection: {document.collection_name}")
    logger.info("=" * 80)

    try:
        logger.info(f"🔤 [STEP 1/5] Text Extraction")
        text_start = time.time()

        text = FileHandler.extract_text(file_path)

        text_elapsed = time.time() - text_start
        logger.info(f"✅ [STEP 1/5] Text extracted in {text_elapsed:.1f}s")
        logger.info(f"   → Characters: {len(text):,}")
        logger.info(f"   → Words: ~{len(text.split()):,}")
        logger.info(f"   → Lines: ~{text.count(chr(10)):,}")

        # SCHRITT 2: Metadata-Extraktion
        logger.info(f"📋 [STEP 2/5] Metadata Extraction")
        metadata_start = time.time()

        metadata_chunk = _extract_metadata_for_document(file_path, document.filename)

        metadata_elapsed = time.time() - metadata_start
        if metadata_chunk:
            logger.info(f"✅ [STEP 2/5] Metadata extracted in {metadata_elapsed:.1f}s")
        else:
            logger.info(f"⚠️  [STEP 2/5] No metadata extracted ({metadata_elapsed:.1f}s)")

        # SCHRITT 3: Chunking
        logger.info(f"✂️  [STEP 3/5] Document Chunking")
        chunk_start = time.time()

        pickle_path = os.path.join(settings.pickle_dir, f"doc_{document.id}.pkl")
        collection_name = document.collection_name

        if not collection_name:
            raise ValueError(f"Invalid collection_name for document {document.id}: {collection_name}")

        logger.info(f"   → Chunking with parent-child strategy...")
        logger.info(f"   → Parent size: {settings.parent_chunk_size} tokens")
        logger.info(f"   → Child size: {settings.child_chunk_size or settings.chunk_size} tokens")

        chunks = doc_processor.process_document(
            document.id,
            text,
            pickle_path=pickle_path,
            document_name=document.filename,
            metadata_chunk=metadata_chunk
        )

        chunk_elapsed = time.time() - chunk_start
        logger.info(f"✅ [STEP 3/5] Chunking complete in {chunk_elapsed:.1f}s")
        logger.info(f"   → Total chunks: {len(chunks)}")
        logger.info(f"   → Metadata chunks: {sum(1 for c in chunks if c.get('is_metadata'))}")
        logger.info(f"   → Content chunks: {sum(1 for c in chunks if not c.get('is_metadata'))}")
        logger.info(f"   → Pickle saved: {pickle_path}")

        # SCHRITT 4: Vektorisierung & Speicherung
        logger.info(f"🔢 [STEP 4/5] Vector Embedding & Storage")
        vector_start = time.time()

        logger.info(f"   → Resetting collection '{collection_name}'...")
        vector_store_service.reset_collection(collection_name)

        logger.info(f"   → Generating embeddings for {len(chunks)} chunks...")
        logger.info(f"   → Embedding model: {settings.embedding_model}")

        vector_store_service.add_documents(
            document.id,
            chunks,
            collection_name,
            document_name=document.filename
        )

        vector_elapsed = time.time() - vector_start
        logger.info(f"✅ [STEP 4/5] Vectors stored in {vector_elapsed:.1f}s")
        logger.info(f"   → Collection: {collection_name}")
        logger.info(f"   → Vectors: {len(chunks)} embeddings")

        # SCHRITT 5: Datenbank-Update
        logger.info(f"💾 [STEP 5/5] Database Update")

        document.pickle_path = pickle_path
        document.processed = True
        document.num_chunks = len(chunks)

        # Gesamt-Statistik
        pipeline_elapsed = time.time() - pipeline_start

        logger.info("=" * 80)
        logger.info(f"✅ [PIPELINE COMPLETE] Document processing successful!")
        logger.info(f"   • Total time: {pipeline_elapsed:.1f}s")
        logger.info(f"   • Text extraction: {text_elapsed:.1f}s ({text_elapsed / pipeline_elapsed * 100:.0f}%)")
        logger.info(f"   • Metadata: {metadata_elapsed:.1f}s ({metadata_elapsed / pipeline_elapsed * 100:.0f}%)")
        logger.info(f"   • Chunking: {chunk_elapsed:.1f}s ({chunk_elapsed / pipeline_elapsed * 100:.0f}%)")
        logger.info(f"   • Vectorization: {vector_elapsed:.1f}s ({vector_elapsed / pipeline_elapsed * 100:.0f}%)")
        logger.info(f"   • Document ID: {document.id}")
        logger.info(f"   • Chunks created: {len(chunks)}")
        logger.info(f"   • Ready for queries: YES")
        logger.info("=" * 80)

        return document

    except Exception as exc:
        pipeline_elapsed = time.time() - pipeline_start
        logger.error("=" * 80)
        logger.error(f"❌ [PIPELINE FAILED] Document processing failed after {pipeline_elapsed:.1f}s")
        logger.error(f"   • Document ID: {document.id}")
        logger.error(f"   • Filename: {document.filename}")
        logger.error(f"   • Error: {type(exc).__name__}: {exc}")
        logger.error("=" * 80)
        logger.exception("Full traceback:")
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
    logger.info(f"💬 Created new chat: {db_chat.id} - '{db_chat.title}'")
    return db_chat


@app.get("/chats", response_model=List[ChatResponse], tags=["Chats"])
async def list_chats(db: Session = Depends(get_db)):
    chats = db.query(Chat).order_by(Chat.updated_at.desc()).all()
    logger.info(f"📋 Listed {len(chats)} chats")
    return chats


@app.get("/chats/{chat_id}", response_model=ChatResponse, tags=["Chats"])
async def get_chat(chat_id: int, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        logger.warning(f"⚠️  Chat {chat_id} not found")
        raise HTTPException(status_code=404, detail="Chat not found")
    logger.info(f"📖 Retrieved chat: {chat_id}")
    return chat


@app.delete("/chats/{chat_id}", tags=["Chats"])
async def delete_chat(chat_id: int, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    db.delete(chat)
    db.commit()
    logger.info(f"🗑️  Deleted chat: {chat_id}")
    return {"status": "deleted"}


@app.get("/chats/{chat_id}/messages", response_model=List[MessageResponse], tags=["Messages"])
async def get_messages(chat_id: int, db: Session = Depends(get_db)):
    messages = (
        db.query(Message)
        .filter(Message.chat_id == chat_id)
        .order_by(Message.created_at)
        .all()
    )
    logger.info(f"💬 Retrieved {len(messages)} messages for chat {chat_id}")
    return messages


@app.post("/documents", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(
        file: UploadFile = File(...),
        db: Session = Depends(get_db)
):
    logger.info(f"📤 Upload request received: {file.filename} ({file.content_type})")

    try:
        # File speichern
        logger.info(f"💾 Saving file to disk...")
        file_path = FileHandler.save_upload(file.file, file.filename, settings.upload_dir)
        file_size = os.path.getsize(file_path)
        logger.info(f"✅ File saved: {file_path} ({file_size:,} bytes)")

        # DB Eintrag erstellen
        logger.info(f"💾 Creating database entry...")
        db_document = Document(
            filename=file.filename,
            file_path=file_path,
            processed=False
        )
        db.add(db_document)
        db.commit()
        db.refresh(db_document)

        logger.info(f"✅ Document entry created:")
        logger.info(f"   • ID: {db_document.id}")
        logger.info(f"   • Collection: {db_document.collection_name}")

        db_document = _process_document_pipeline(db_document, file_path)

        db.commit()
        db.refresh(db_document)

        logger.info(f"🎉 Document upload complete: {file.filename} (ID: {db_document.id})")

        return db_document

    except Exception as exc:
        db.rollback()
        logger.error(f"❌ Upload failed for {file.filename}: {type(exc).__name__}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(exc)}"
        )


@app.get("/documents", response_model=List[DocumentUploadResponse], tags=["Documents"])
async def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.uploaded_at.desc()).all()
    logger.info(f"📚 Listed {len(docs)} documents")
    return docs


@app.post("/documents/{doc_id}/reprocess", response_model=DocumentUploadResponse, tags=["Documents"])
async def reprocess_document(doc_id: int, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == doc_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if not document.file_path or not os.path.exists(document.file_path):
        raise HTTPException(status_code=400, detail="Document file not found on disk")

    logger.info(f"🔄 Reprocessing document {doc_id}: {document.filename}")

    try:
        document = _process_document_pipeline(document, document.file_path)
        db.commit()
        db.refresh(document)
        logger.info(f"✅ Reprocessing complete for doc {doc_id}")
        return document

    except Exception as exc:
        db.rollback()
        logger.exception(f"❌ Reprocessing failed for document {document.filename}: {exc}")
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
    document = db.query(Document).filter(Document.id == doc_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    old_status = document.query_enabled
    document.query_enabled = preferences.query_enabled
    db.commit()
    db.refresh(document)

    status_text = "enabled" if preferences.query_enabled else "disabled"
    logger.info(f"⚙️  Document {doc_id} query {status_text} (was: {old_status})")

    return document


@app.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: int, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == doc_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    logger.info(f"🗑️  Deleting document {doc_id}: {document.filename}")

    collection_name = document.collection_name
    pickle_path = document.pickle_path
    file_path = document.file_path

    # DB-Eintrag löschen
    try:
        db.delete(document)
        db.commit()
        logger.info(f"   ✅ Deleted from database")
    except Exception as exc:
        db.rollback()
        logger.exception(f"   ❌ Database deletion failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document from database: {str(exc)}"
        )

    # Cleanup (Fehler nicht kritisch)
    try:
        vector_store_service.delete_document(collection_name)
        logger.info(f"   ✅ Deleted collection: {collection_name}")
    except Exception as exc:
        logger.warning(f"   ⚠️  Collection deletion failed: {exc}")

    try:
        FileHandler.delete_file(pickle_path)
        logger.info(f"   ✅ Deleted pickle file")
    except Exception as exc:
        logger.warning(f"   ⚠️  Pickle deletion failed: {exc}")

    try:
        FileHandler.delete_file(file_path)
        logger.info(f"   ✅ Deleted source file")
    except Exception as exc:
        logger.warning(f"   ⚠️  File deletion failed: {exc}")

    logger.info(f"✅ Document {doc_id} deletion complete")
    return {"status": "deleted"}


@app.post("/query/stream", tags=["Query"])
async def query_documents_stream(request: QueryRequest, db: Session = Depends(get_db)):
    logger.info(f"🔍 Query received: '{request.query[:100]}...' (chat_id: {request.chat_id})")

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
        chat_history: list[dict[str, InstrumentedAttribute[str]]] = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]
        logger.info(f"   → Chat history: {len(messages)} messages")

        doc_collection_map = _get_active_doc_collection_map(db)
        if not doc_collection_map:
            logger.warning(f"   ⚠️  No active documents for query")
            raise HTTPException(
                status_code=400,
                detail="No active documents selected for querying."
            )

        logger.info(f"   → Active documents: {len(doc_collection_map)}")

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
        logger.exception(f"❌ Query preparation failed: {exc}")
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
            logger.info(f"🔄 Starting retrieval thread...")

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

            logger.info(f"✅ Retrieval complete: {len(contexts)} contexts, {len(sources)} sources")

        except Exception as exception:
            logger.exception(f"❌ Retrieval failed: {exception}")
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
            # Warte auf Retrieval
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
                logger.warning(f"⚠️  No contexts found for query")
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

            # Generiere Antwort
            logger.info(f"🤖 Generating answer with {len(contexts)} contexts...")

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

            logger.info(f"✅ Answer generated: {len(accumulated_answer)} chars")

            yield _format_sse_event({
                "type": "end",
                "content": accumulated_answer,
                "sources": sources,
                "message_id": assistant_message.id
            })

        except Exception as stream_error:
            db.rollback()
            logger.exception(f"❌ Streaming failed: {stream_error}")
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