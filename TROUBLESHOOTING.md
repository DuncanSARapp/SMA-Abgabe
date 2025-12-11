# Troubleshooting Guide

## Common Issues and Solutions

### 1. Backend Connection Issues

#### Problem: Backend can't connect to PostgreSQL
```
SQLALCHEMY ERROR: could not connect to server
```

**Solution:**
- Wait a few seconds for PostgreSQL to fully initialize
- Check if PostgreSQL is healthy: `docker compose ps`
- View PostgreSQL logs: `docker compose logs postgres`
- Restart services: `docker compose restart backend`

#### Problem: Backend can't connect to Qdrant
```
QdrantException: Could not connect to Qdrant
```

**Solution:**
- Check Qdrant status: `docker compose ps qdrant`
- View Qdrant logs: `docker compose logs qdrant`
- Restart Qdrant: `docker compose restart qdrant`

### 2. LLM API Issues

#### Problem: Invalid API Key
```
Error 401: Invalid authentication credentials
```

**Solution:**
- Verify your API key in `.env` is correct
- Ensure no extra spaces or quotes around the key
- Check you have credits/quota remaining with your LLM provider
- Restart backend after changing `.env`: `docker compose restart backend`

#### Problem: API Rate Limit
```
Error 429: Rate limit exceeded
```

**Solution:**
- Wait a few minutes and try again
- Check your API plan limits
- Consider using a different model with higher limits

#### Problem: Model Not Found
```
Error 404: Model not found
```

**Solution:**
- Verify the model name in `.env` matches your provider's available models
- Check your API plan has access to that model
- Common model names:
  - OpenAI: `gpt-3.5-turbo`, `gpt-4`
  - Anthropic: `claude-3-sonnet-20240229`

### 3. Document Upload Issues

#### Problem: Document processing fails
```
Error processing document: Unsupported file type
```

**Solution:**
- Ensure file is one of: PDF, DOCX, TXT, MD
- Check file is not corrupted
- Try a smaller document first
- View backend logs: `docker compose logs backend`

#### Problem: PDF text extraction issues
```
No text extracted from PDF
```

**Solution:**
- PDF might be scanned images (use OCR separately)
- PDF might be password protected
- Try converting to TXT first

#### Problem: Out of memory during processing
```
MemoryError or OOMKilled
```

**Solution:**
- Increase Docker memory limit (Settings → Resources)
- Process smaller documents
- Reduce chunk sizes in `backend/config/settings.py`

### 4. Frontend Issues

#### Problem: Frontend can't connect to backend
```
Failed to fetch
```

**Solution:**
- Ensure backend is running: `docker compose ps backend`
- Check backend logs: `docker compose logs backend`
- Try accessing backend directly: `http://localhost:8000/docs`
- Clear browser cache and reload

#### Problem: Blank page or JavaScript errors
```
Script error or blank page
```

**Solution:**
- Open browser console (F12) to see errors
- Hard refresh: Ctrl+Shift+R (or Cmd+Shift+R on Mac)
- Try a different browser
- Check frontend logs: `docker compose logs frontend`

### 5. Performance Issues

#### Problem: Slow document processing

**Causes & Solutions:**
- **Large documents**: Break into smaller files
- **CPU-bound**: Embeddings run on CPU, be patient
- **Too many chunks**: Increase chunk size in settings

#### Problem: Slow query responses

**Causes & Solutions:**
- **LLM API latency**: Try a faster model or local model
- **Too many retrievals**: Reduce `top_k_retrieval` in settings
- **Network issues**: Check internet connection

### 6. Docker Issues

#### Problem: Port already in use
```
Error: port is already allocated
```

**Solution:**
- Check what's using the port: `lsof -i :8000` (or :3000, :5432, :6333)
- Stop the conflicting service
- Or change ports in `docker-compose.yml`

#### Problem: Cannot connect to Docker daemon
```
Cannot connect to the Docker daemon
```

**Solution:**
- Ensure Docker Desktop is running
- On Linux, start Docker: `sudo systemctl start docker`
- Check Docker status: `docker info`

#### Problem: Build fails with network errors
```
failed to fetch from archive
```

**Solution:**
- Check internet connection
- Try again (temporary network issue)
- Clear Docker build cache: `docker compose build --no-cache`

### 7. Data Issues

#### Problem: Lost all chats/documents

**Solution:**
- If you ran `docker compose down -v`, volumes were deleted
- This is permanent - there's no backup by default
- To preserve data: use `docker compose down` without `-v`

#### Problem: Database migration errors
```
SQLALCHEMY ERROR: table already exists
```

**Solution:**
- This is usually harmless on first run
- If persistent, reset database:
  ```bash
  docker compose down -v
  docker compose up -d
  ```

### 8. Model Download Issues

#### Problem: Models fail to download during build
```
Error downloading model
```

**Solution:**
- Check internet connection
- Increase Docker build timeout
- Build without cache: `docker compose build --no-cache backend`
- Download models manually:
  ```bash
  docker compose run backend python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('mixedbread-ai/deepset-mxbai-embed-de-large-v1')"
  docker compose run backend python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
  ```

## Getting Help

### 1. Check Logs

View all logs:
```bash
docker compose logs
```

View specific service:
```bash
docker compose logs backend
docker compose logs frontend
docker compose logs postgres
docker compose logs qdrant
```

Follow logs in real-time:
```bash
docker compose logs -f backend
```

### 2. Check Service Status

```bash
docker compose ps
```

All services should show "Up" status.

### 3. Restart Services

Restart all:
```bash
docker compose restart
```

Restart specific service:
```bash
docker compose restart backend
```

### 4. Complete Reset

If all else fails:
```bash
# Stop and remove everything
docker compose down -v

# Rebuild and start fresh
docker compose up --build -d
```

### 5. Debug Mode

Run backend in foreground to see detailed logs:
```bash
docker compose stop backend
docker compose run --rm -p 8000:8000 backend
```

## Still Having Issues?

1. Check the GitHub Issues page
2. Open a new issue with:
   - Description of the problem
   - Steps to reproduce
   - Relevant log output
   - Your environment (OS, Docker version)
