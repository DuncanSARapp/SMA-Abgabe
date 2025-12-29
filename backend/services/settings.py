import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://raguser:ragpass@localhost:5432/ragdb"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = True
    qdrant_collection_prefix: str = "doc_"

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = ""
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    llm_timeout: float = 60.0

    embedding_model: str = "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_cache_size: int = 10000
    use_docling_parser: bool = True
    ocr_engine: str = "rapidocr" # Options: "rapidocr", "paddleocr", "easyocr", "tesseract"
    use_gpu_for_ocr: bool = False

    chunk_size: int = 1000
    chunk_overlap: int = 180
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 400
    child_chunk_size: int = 400
    child_chunk_overlap: int = 80

    enable_neighbor_expansion: bool = True
    neighbor_expansion_window: int = 4
    top_k_retrieval: int = 20
    top_k_rerank: int = 6

    query_expansion_cache_size: int = 1000
    query_expansion_cache_ttl: int = 3600

    data_dir: str = "/app/data"
    models_cache_dir: str = "/app/models"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"
        protected_namespaces = ()


    @property
    def upload_dir(self) -> str:
        return os.path.join(self.data_dir, "uploads")

    @property
    def pickle_dir(self) -> str:
        return os.path.join(self.data_dir, "pickles")

    def get_active_provider(self) -> str:
        if self.anthropic_api_key:
            return "anthropic"
        elif self.openai_api_key:
            return "openai"
        else:
            return "ollama"

    def ensure_directories(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.pickle_dir, exist_ok=True)
        os.makedirs(self.models_cache_dir, exist_ok=True)

settings = Settings()