# SMA-Abgabe - RAG Chat System

Repository für die SMA-Abgabe bezüglich des Mini-Rag-Systems

## Overview

A Docker-Compose-based RAG (Retrieval-Augmented Generation) system with:
- **PostgreSQL** for chat persistence
- **Qdrant** vector store for document embeddings
- **Python Backend** with FastAPI and LangChain
- **Web Frontend** for multi-chat interface and document upload
- **Parent Document Retriever** with pickle files
- **Local Embedding Model** (mixedbread-ai/deepset-mxbai-embed-de-large-v1)
- **Local Reranker** (BAAI/bge-reranker-v2-m3 cross-encoder)
- **API-based LLM** (OpenAI-compatible)

## Features

- 📚 Upload documents (PDF, DOCX, TXT, MD)
- 💬 Multiple persistent chats
- 🔍 RAG-based question answering
- 📊 Parent-child document chunking strategy
- 🎯 Reranking for improved relevance
- 🌐 Web-based user interface
- ✅ Document-specific Qdrant collections with per-file query toggles
- 🔒 No authentication (as requested)

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Frontend   │────▶│   Backend    │────▶│  PostgreSQL │
│  (Nginx)    │     │  (FastAPI)   │     │             │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ├────▶ Qdrant (Vector Store)
                           │
                           └────▶ LLM API (OpenAI/Claude/etc.)
```

### Components

1. **Backend (Python/FastAPI)**
   - Document processing and chunking
   - Local embeddings (mixedbread-ai/deepset-mxbai-embed-de-large-v1)
   - Local reranker (BAAI/bge-reranker-v2-m3)
   - Parent document retriever with pickle storage
   - RAG pipeline with LangChain
   - API-based LLM integration

2. **Frontend (HTML/CSS/JS)**
   - Multi-chat interface
   - Document upload
   - Real-time messaging
   - Document management

3. **Infrastructure**
   - PostgreSQL for chat/message persistence
   - Qdrant for vector storage
   - Docker Compose orchestration

## Prerequisites

- Docker and Docker Compose
- LLM API key (OpenAI, Anthropic, or other OpenAI-compatible API)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/DuncanSARapp/SMA-Abgabe.git
cd SMA-Abgabe
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your LLM API key:

```env
LLM_API_KEY=your-api-key-here
LLM_API_BASE=https://api.openai.com/v1
LLM_MODEL=gpt-3.5-turbo
```

**Supported LLM Providers:**

- **OpenAI**: Default configuration
- **Anthropic Claude**:
  ```env
  LLM_API_BASE=https://api.anthropic.com/v1
  LLM_MODEL=claude-3-sonnet-20240229
  ```
- **Local (Ollama)**:
  ```env
  LLM_API_BASE=http://host.docker.internal:11434/v1
  LLM_MODEL=llama2
  ```

### 3. Start the services

```bash
docker-compose up --build
```

This will:
- Build the backend and frontend containers
- Start PostgreSQL and Qdrant
- Download embedding and reranker models
- Initialize the database

### 4. Access the application

Open your browser and navigate to:
```
http://localhost:3000
```

API documentation (Swagger UI):
```
http://localhost:8000/docs
```

## Usage

### 1. Upload Documents

1. Click "Upload Document" button
2. Select a PDF, DOCX, TXT, or MD file
3. Wait for processing (documents are chunked and embedded)

### 2. Create a Chat

1. Click "New Chat" button
2. Enter a chat title
3. Start asking questions!

### 3. Ask Questions

1. Select a chat from the sidebar
2. Type your question about the uploaded documents
3. Get AI-generated answers with source citations

### 4. Manage Chats

- View all chats in the sidebar
- Switch between chats
- Delete chats when no longer needed

### 5. Control Document Participation

Each uploaded document now owns its own Qdrant collection. Use the square toggle next to the trash icon in the documents panel to decide whether a document should be included in retrieval:

1. Green checkmark = document is searchable.
2. Gray square = document is excluded from queries (chunks stay on disk for later reactivation).

Disabled documents are ignored by the retriever, which keeps responses scoped to the sources you explicitly selected.

## Technical Details

### Parent Document Retriever

The system uses a two-level chunking strategy:

1. **Parent Documents** (2000 tokens): Large chunks stored in pickle files
2. **Child Chunks** (1000 tokens): Smaller chunks embedded in Qdrant

**Retrieval Process:**
1. User query is embedded
2. Top-K child chunks retrieved from Qdrant (K=20)
3. Chunks are reranked for relevance (top 5)
4. Parent documents are loaded from pickle files
5. If there is remaining room (less than 5 sources), the system automatically appends the next section and then the previous section from the same document so that complete headings stay together (configurable).
6. LLM generates answer based on parent contexts

### Document Parsing

- **PDFs** are parsed via [Docling](https://github.com/DS4SD/docling) to preserve headings, tables, and other structural cues before chunking. The Docling output is converted to Markdown so the semantic chunker can align child chunks with their parent sections.
- If Docling is unavailable or `USE_DOCLING_PARSER=false`, the system falls back to `pypdf`-based text extraction.
- DOCX/TXT/MD inputs continue to use their original parsers.

### Vector Store Layout

- Every document is stored in its own Qdrant collection named `doc_<document_id>`.
- Startup synchronization marks database entries as unprocessed if their collection is missing and removes orphaned collections that have no corresponding document.
- The UI toggle updates the `query_enabled` flag per document so that retrieval only touches the collections you explicitly selected.

### Models

- **Embedding**: `mixedbread-ai/deepset-mxbai-embed-de-large-v1` (local, ≈1.3GB, 1024-dim)
- **Reranker**: `BAAI/bge-reranker-v2-m3` (local, ≈1.4GB)
- **LLM**: API-based (configurable)

### API Endpoints

- `POST /chats` - Create new chat
- `GET /chats` - List all chats
- `GET /chats/{id}` - Get chat details
- `DELETE /chats/{id}` - Delete chat
- `GET /chats/{id}/messages` - Get chat messages
- `POST /documents` - Upload document
- `GET /documents` - List documents
- `POST /query` - Query RAG system

## Development

### Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI application
│   │   └── database.py       # Database configuration
│   ├── models/
│   │   ├── database.py       # SQLAlchemy models
│   │   └── schemas.py        # Pydantic schemas
│   ├── services/
│   │   ├── embeddings.py     # Embedding & vector store
│   │   ├── reranker.py       # Reranking service
│   │   ├── document_processor.py  # Document chunking
│   │   ├── rag_service.py    # RAG pipeline
│   │   └── file_handler.py   # File processing
│   ├── config/
│   │   └── settings.py       # Configuration
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   ├── nginx.conf
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

### Running in Development Mode

Backend with hot reload:
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend (serve with any static server):
```bash
cd frontend
python -m http.server 3000
```

### Configuration

Edit `backend/config/settings.py` to customize:
- Chunk sizes
- Retrieval parameters
- Model names
- Data directories

## Troubleshooting

### Issue: Backend can't connect to PostgreSQL

**Solution**: Wait for PostgreSQL healthcheck to pass before starting backend. Docker Compose handles this automatically.

### Issue: Out of memory when downloading models

**Solution**: Increase Docker memory limit to at least 4GB.

### Issue: LLM API errors

**Solution**: 
- Verify your API key in `.env`
- Check API base URL is correct
- Ensure you have API credits

### Issue: Document processing fails

**Solution**:
- Check file format is supported (PDF, DOCX, TXT, MD)
- Ensure file is not corrupted
- Check backend logs: `docker-compose logs backend`

### Issue: Column "query_enabled" does not exist

**Solution**: Existing PostgreSQL databases need one new column. Run:

```sql
ALTER TABLE documents ADD COLUMN IF NOT EXISTS query_enabled BOOLEAN DEFAULT TRUE;
```

Restart the backend afterwards so FastAPI picks up the new field.

## Data Persistence

All data is persisted in Docker volumes:
- `postgres_data`: Chat and message history
- `qdrant_data`: Vector embeddings
- `backend_data`: Uploaded files and pickle files

To reset all data:
```bash
docker-compose down -v
```

## Performance Considerations

- **Embedding Model**: Runs on CPU, ~1-2 seconds per document
- **Reranker**: Fast inference, <100ms for 20 documents
- **LLM**: Depends on API provider latency
- **Vector Search**: Sub-second with Qdrant

## Security Note

⚠️ This system has **no authentication or authorization** as requested. Do not expose it to the public internet without adding security measures.

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

## Support

For issues and questions, please open a GitHub issue.
