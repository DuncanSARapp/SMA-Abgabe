FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    libgomp1 \
    libopenblas-dev \
    libomp-dev \
    libgl1 \
    libglib2.0-0 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir onnxruntime rapidocr

ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/app/models/sentence-transformers
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

RUN mkdir -p /app/models/transformers /app/models/sentence-transformers

RUN echo "📥 Downloading Embedding Model..." && \
    python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('mixedbread-ai/deepset-mxbai-embed-de-large-v1')" && \
    echo "✅ Embedding Model cached"

RUN echo "📥 Downloading Reranker Model..." && \
    python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('BAAI/bge-reranker-v2-m3')" && \
    echo "✅ Reranker Model cached"

RUN echo "📥 Downloading VLM Model (GraniteDocling)..." && \
    python -c "from transformers import AutoModel, AutoTokenizer; \
    model_id = 'ibm-granite/granite-docling-258M'; \
    AutoTokenizer.from_pretrained(model_id); \
    AutoModel.from_pretrained(model_id)" && \
    echo "✅ VLM Model cached"

RUN python -c "import transformers; \
    print(f'✅ Transformers: {transformers.__version__}'); \
    assert transformers.__version__ >= '4.51.0', 'Transformers version must be >= 4.51.0'"

COPY . .

RUN mkdir -p /app/data/uploads /app/data/pickles /db/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]