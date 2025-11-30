"""Configuration management for RAGBrain - Extensible provider architecture"""

from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Application settings with support for multiple providers"""

    # ===== LLM Provider =====
    llm_provider: Literal["anthropic", "openai", "ollama", "fallback"] = "fallback"  # Default to fallback (Ollama -> Anthropic)
    llm_model: str | None = None

    # ===== Ollama Configuration =====
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_embedding_model: str = "mxbai-embed-large"

    # ===== Fallback Provider Configuration =====
    fallback_primary: Literal["ollama", "openai", "anthropic"] = "ollama"
    fallback_secondary: Literal["ollama", "openai", "anthropic"] = "anthropic"

    # ===== Embedding Provider =====
    embedding_provider: Literal["openai", "cohere", "ollama", "mixedbread", "fallback"] = "openai"
    embedding_model: str | None = None
    embedding_dimension: int = 1536

    # ===== Embedding Fallback Configuration =====
    embedding_fallback_primary: Literal["ollama", "openai", "cohere", "mixedbread"] = "ollama"
    embedding_fallback_secondary: Literal["ollama", "openai", "cohere", "mixedbread"] = "mixedbread"

    # ===== Vector Database Provider =====
    vectordb_provider: Literal["qdrant", "pinecone", "chroma"] = "qdrant"

    # ===== Mixedbread Configuration =====
    mixedbread_api_key: str | None = None
    mixedbread_model: str = "mxbai-embed-large-v1"

    # ===== API Keys =====
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    cohere_api_key: str | None = None

    # ===== Qdrant Configuration =====
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "ragbrain"

    # ===== Namespace Configuration =====
    default_namespace: str | None = None  # Optional default namespace for multi-user setups
    namespace_provider: Literal["sqlite", "redis"] = "sqlite"  # Namespace registry provider
    namespace_db_path: str = "data/namespaces.db"  # Path to SQLite database for namespaces

    # ===== Redis Configuration (for namespace provider) =====
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str | None = None
    redis_db: int = 0  # Redis database number (0-15)

    # ===== Pinecone Configuration =====
    pinecone_api_key: str | None = None
    pinecone_index: str = "ragbrain"
    pinecone_namespace: str | None = None
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # ===== Chroma Configuration =====
    chroma_collection: str = "ragbrain"
    # For embedded mode (local)
    chroma_persist_directory: str | None = None  # Default: ./data/chroma
    # For client/server mode (HTTP)
    chroma_host: str | None = None  # e.g., "localhost" or "chroma.example.com"
    chroma_port: int = 8000
    chroma_ssl: bool = False
    chroma_api_key: str | None = None

    # ===== Reranker Configuration =====
    reranker_provider: Literal["simple", "ollama", "cohere", "none"] = "simple"  # Default to simple local reranker
    reranker_model: str = "qllama/bge-reranker-v2-m3"  # For Ollama reranker
    reranker_dedupe_threshold: float = 0.85  # For simple reranker: similarity threshold for deduplication

    # ===== Chunking Configuration =====
    # Available strategies: recursive, markdown, semantic, character, transcript
    default_chunking_strategy: Literal["recursive", "markdown", "semantic", "character", "transcript"] = "recursive"
    chunk_size: int = 2000  # Target size for content (before metadata prefix)
    chunk_overlap: int = 200  # Overlap between chunks for context continuity
    chunk_max_size: int = 2500  # Hard limit including metadata prefix
    chunk_metadata_reserve: int = 300  # Reserved space for prepended metadata (e.g., speaker, topic)

    # ===== Application Settings =====
    log_level: str = "info"
    upload_dir: str = "uploads"
    queue_dir: str = "/data/queue"  # Shared directory for pending uploads from dropbox

    class Config:
        env_file = ".env"
        case_sensitive = False

    def get_llm_model(self) -> str:
        """Get LLM model name based on provider"""
        if self.llm_model:
            return self.llm_model

        defaults = {
            "anthropic": "claude-3-7-sonnet-20250219",
            "openai": "gpt-4-turbo-preview",
            "ollama": self.ollama_model,
            "fallback": self.ollama_model,  # Primary model for fallback
        }
        return defaults.get(self.llm_provider, "claude-3-7-sonnet-20250219")

    def get_embedding_model(self) -> str:
        """Get embedding model name based on provider"""
        if self.embedding_model:
            return self.embedding_model

        defaults = {
            "openai": "text-embedding-3-small",
            "cohere": "embed-english-v3.0",
            "ollama": self.ollama_embedding_model,
            "mixedbread": self.mixedbread_model,
            "fallback": self.ollama_embedding_model,  # Primary model for fallback
        }
        return defaults.get(self.embedding_provider, "text-embedding-3-small")

    def get_embedding_dimensions(self) -> int:
        """Get embedding dimensions based on provider/model"""
        dimension_map = {
            "openai": {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
            },
            "cohere": {
                "embed-english-v3.0": 1024,
            },
            "ollama": {
                "mxbai-embed-large": 1024,
                "nomic-embed-text": 768,
                "all-minilm": 384,
                "snowflake-arctic-embed": 1024,
                "bge-large": 1024,
                "bge-m3": 1024,
            },
            "mixedbread": {
                "mxbai-embed-large-v1": 1024,
                "deepset-mxbai-embed-de-large-v1": 1024,
                "mxbai-embed-2d-large-v1": 1024,
            },
        }

        # For fallback, use primary provider's dimensions
        provider = self.embedding_provider
        if provider == "fallback":
            provider = self.embedding_fallback_primary

        provider_models = dimension_map.get(provider, {})
        model = self.get_embedding_model()
        return provider_models.get(model, self.embedding_dimension)


# Global settings instance
settings = Settings()
