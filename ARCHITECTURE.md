# Architecture Overview

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                            │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP
                         ▼
                  ┌─────────────┐
                  │   Frontend  │
                  │  (Nginx)    │
                  │  Port 3000  │
                  └──────┬──────┘
                         │ API Calls
                         ▼
                  ┌─────────────┐
                  │   Backend   │
                  │  (FastAPI)  │
                  │  Port 8000  │
                  └──────┬──────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
 ┌──────────┐    ┌──────────┐    ┌──────────┐
 │PostgreSQL│    │  Qdrant  │    │ LLM API  │
 │ Port 5432│    │Port 6333 │    │ (Remote) │
 └──────────┘    └──────────┘    └──────────┘
```

## Data Flow

### Document Upload Flow

```
1. User uploads document (PDF/DOCX/TXT/MD)
                │
                ▼
2. Backend receives file
                │
                ▼
3. Extract text from document
                │
                ▼
4. Create parent chunks (2000 tokens)
                │
                ▼
5. Save parent chunks to pickle file
                │
                ▼
6. Split into child chunks (1000 tokens)
                │
                ▼
7. Generate embeddings (local model)
                │
                ▼
8. Store embeddings in Qdrant
                │
                ▼
9. Update PostgreSQL with document metadata
```

### Query Flow

```
1. User sends query in chat
                │
                ▼
2. Backend generates query embedding
                │
                ▼
3. Vector search in Qdrant (retrieve top 20 child chunks)
                │
                ▼
4. Rerank child chunks (local reranker → top 5)
                │
                ▼
5. Load parent documents from pickle files
                │
                ▼
6. Send context + query to LLM API
                │
                ▼
7. Receive LLM response
                │
                ▼
8. Save message to PostgreSQL
                │
                ▼
9. Return response to user
```

## Service Details

### Backend (Python/FastAPI)

**Responsibilities:**
- REST API endpoints
- Document processing
- Vector store management
- RAG pipeline orchestration
- Database operations

**Key Components:**
- `app/main.py`: FastAPI application and routes
- `services/embeddings.py`: Embedding generation and vector store
- `services/reranker.py`: Document reranking
- `services/document_processor.py`: Text chunking and pickle management
- `services/rag_service.py`: RAG pipeline coordination
- `services/file_handler.py`: File format handling
- `models/database.py`: SQLAlchemy models
- `models/schemas.py`: Pydantic schemas

**Dependencies:**
- FastAPI for API framework
- LangChain for RAG orchestration
- Sentence-Transformers for embeddings
- Qdrant client for vector store
- SQLAlchemy for database ORM

### Frontend (HTML/CSS/JavaScript)

**Responsibilities:**
- User interface
- Chat management
- Document upload
- Real-time messaging

**Features:**
- Multi-chat sidebar
- Message history
- Document list panel
- Responsive design
- Loading states and notifications

### PostgreSQL

**Purpose:** Persistent storage for structured data

**Tables:**
- `chats`: Chat sessions
- `messages`: Chat messages (user and assistant)
- `documents`: Uploaded document metadata

### Qdrant

**Purpose:** Vector store for semantic search

**Data:**
- Child chunk embeddings (1000 token chunks)
- Metadata (doc_id, parent_id, text)

### LLM API

**Purpose:** Natural language generation

**Supported Providers:**
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Azure OpenAI
- Local models (Ollama)
- Any OpenAI-compatible API

## Parent Document Retriever Strategy

### Why Parent-Child Chunking?

**Problem:** Traditional RAG systems chunk documents into fixed sizes. Small chunks lack context, while large chunks are imprecise.

**Solution:** Two-level chunking strategy:

1. **Parent Documents** (2000 tokens)
   - Larger chunks with more context
   - Stored in pickle files (`.pkl`)
   - Not embedded directly

2. **Child Chunks** (1000 tokens)
   - Smaller, more precise chunks
   - Embedded in vector store
   - Reference their parent document

### Benefits

✅ **Precise retrieval** via child chunks  
✅ **Rich context** from parent documents  
✅ **Better LLM responses** with full context  
✅ **Efficient storage** (only child embeddings)

### Example

Document: "Machine learning is a subset of artificial intelligence..."

```
Parent Document 1 (2000 tokens):
├── Child Chunk 1.1 (1000 tokens) → embedded
└── Child Chunk 1.2 (1000 tokens) → embedded

Parent Document 2 (2000 tokens):
├── Child Chunk 2.1 (1000 tokens) → embedded
├── Child Chunk 2.2 (1000 tokens) → embedded
└── Child Chunk 2.3 (1000 tokens) → embedded
```

Query: "What is machine learning?"
1. Retrieves Child Chunk 1.1 (matches well)
2. Returns Parent Document 1 (full context)

## Models

### Embedding Model
- **Name:** all-MiniLM-L6-v2
- **Size:** ~80 MB
- **Dimensions:** 384
- **Type:** Sentence Transformer
- **Speed:** ~1-2 seconds per document (CPU)

### Reranker Model
- **Name:** cross-encoder/ms-marco-MiniLM-L-6-v2
- **Size:** ~80 MB
- **Type:** Cross Encoder
- **Speed:** <100ms for 20 documents (CPU)
- **Purpose:** Improve retrieval relevance

### LLM
- **Type:** API-based (configurable)
- **Default:** GPT-3.5-turbo
- **Location:** Remote (requires API key)

## Security Considerations

⚠️ **No Authentication:** System has no auth by design  
⚠️ **No Authorization:** All users can access all data  
⚠️ **No Encryption:** Data stored in plain text  

**For Production:**
- Add authentication (JWT, OAuth)
- Implement authorization (user-specific data)
- Enable HTTPS
- Encrypt sensitive data
- Add rate limiting
- Input validation and sanitization

## Scalability Considerations

### Current Limitations
- Single backend instance
- No load balancing
- No caching layer
- CPU-bound embeddings

### Future Improvements
- Multiple backend replicas
- Redis caching
- GPU acceleration for embeddings
- Message queue for async processing
- CDN for frontend assets

## Configuration

All configurable in `backend/config/settings.py`:

- Chunk sizes
- Retrieval parameters (top_k)
- Model names
- Database URLs
- API endpoints

## Development vs Production

### Development (Current Setup)
- Volume mounts for hot reload
- Debug mode enabled
- No HTTPS
- Simple passwords

### Production Recommendations
- Build images without volume mounts
- Disable debug mode
- Use HTTPS (nginx + Let's Encrypt)
- Use secrets management
- Set resource limits
- Enable monitoring and logging
- Add health checks
- Implement backups
