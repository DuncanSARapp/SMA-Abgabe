# RAG Chat System

Docker-based RAG (Retrieval-Augmented Generation) system with parent-child chunking, local embeddings, and API-based LLM integration.

## Features

- Multi-format document upload (PDF, DOCX, TXT, MD)
- Multiple persistent chats
- RAG-based question answering with reranking
- Parent-child chunking strategy
- Web-based interface

## Architecture
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

## Prerequisites

- Docker and Docker Compose
- LLM API key (OpenAI, Anthropic, or OpenAI-compatible)

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/DuncanSARapp/SMA-Abgabe.git
cd SMA-Abgabe
```

### 2. Configure Environment
```bash
cp .env.example .env
```

Edit `.env`:
```env
LLM_API_KEY=your-api-key-here
LLM_API_BASE=https://api.openai.com/v1
LLM_MODEL=gpt-3.5-turbo
```

### 3. Run Setup Script
```bash
./setup.sh
```

The `setup.sh` script validates prerequisites, creates a virtual environment, installs dependencies, validates Docker configuration, and starts all services.

### 4. Access Application

Frontend: http://localhost:3000  
API Docs: http://localhost:8000/docs

## Models

**Embedding:** mixedbread-ai/deepset-mxbai-embed-de-large-v1 (1024 dims)  
**Reranker:** BAAI/bge-reranker-v2-m3 (multilingual)  
**LLM:** API-based (configurable)

## Configuration

Edit `backend/config/settings.py`:
- Chunk sizes
- Retrieval parameters
- Model names
- Database URLs
