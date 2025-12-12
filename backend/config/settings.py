from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://raguser:ragpass@localhost:5432/ragdb"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents"
    qdrant_collection_prefix: str = "doc_"
    
    llm_api_key: str = ""
    llm_api_base: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-3.5-turbo"
    
    embedding_model: str = "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    parent_chunk_size: int = 2000
    
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    
    data_dir: str = "/app/data"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
