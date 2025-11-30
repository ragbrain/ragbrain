"""Tests for configuration module"""

import pytest
import os
from unittest.mock import patch


class TestSettings:
    """Tests for Settings configuration class"""

    def test_default_values(self):
        """Test default configuration values"""
        from ragbrain.config import Settings

        # Create settings with minimal config
        settings = Settings(
            anthropic_api_key="test",
            openai_api_key="test"
        )

        # Check defaults
        assert settings.llm_provider == "fallback"
        assert settings.embedding_provider == "openai"
        assert settings.vectordb_provider == "qdrant"
        assert settings.default_chunking_strategy == "recursive"
        assert settings.chunk_size == 2000
        assert settings.chunk_overlap == 200

    def test_qdrant_defaults(self):
        """Test Qdrant default configuration"""
        from ragbrain.config import Settings

        settings = Settings()
        assert settings.qdrant_url == "http://localhost:6333"
        assert settings.qdrant_collection == "ragbrain"

    def test_ollama_defaults(self):
        """Test Ollama default configuration"""
        from ragbrain.config import Settings

        settings = Settings()
        assert settings.ollama_url == "http://localhost:11434"
        assert settings.ollama_model == "llama3.2"

    def test_get_llm_model_default(self):
        """Test default LLM model selection"""
        from ragbrain.config import Settings

        # Anthropic provider
        settings = Settings(llm_provider="anthropic")
        assert "claude" in settings.get_llm_model()

        # OpenAI provider
        settings = Settings(llm_provider="openai")
        assert "gpt" in settings.get_llm_model()

        # Ollama provider
        settings = Settings(llm_provider="ollama")
        assert settings.get_llm_model() == "llama3.2"

    def test_get_llm_model_custom(self):
        """Test custom LLM model override"""
        from ragbrain.config import Settings

        settings = Settings(llm_provider="anthropic", llm_model="claude-3-opus")
        assert settings.get_llm_model() == "claude-3-opus"

    def test_get_embedding_model_default(self):
        """Test default embedding model selection"""
        from ragbrain.config import Settings

        # OpenAI provider
        settings = Settings(embedding_provider="openai")
        assert "embedding" in settings.get_embedding_model()

        # Cohere provider
        settings = Settings(embedding_provider="cohere")
        assert "embed" in settings.get_embedding_model()

    def test_get_embedding_model_custom(self):
        """Test custom embedding model override"""
        from ragbrain.config import Settings

        settings = Settings(embedding_provider="openai", embedding_model="custom-model")
        assert settings.get_embedding_model() == "custom-model"

    def test_get_embedding_dimensions_openai(self):
        """Test embedding dimensions for OpenAI models"""
        from ragbrain.config import Settings

        # text-embedding-3-small
        settings = Settings(embedding_provider="openai")
        settings.embedding_model = None  # Force default
        assert settings.get_embedding_dimensions() == 1536

    def test_get_embedding_dimensions_cohere(self):
        """Test embedding dimensions for Cohere models"""
        from ragbrain.config import Settings

        settings = Settings(embedding_provider="cohere")
        settings.embedding_model = "embed-english-v3.0"
        assert settings.get_embedding_dimensions() == 1024

    def test_get_embedding_dimensions_fallback(self):
        """Test embedding dimensions fallback to default"""
        from ragbrain.config import Settings

        settings = Settings(
            embedding_provider="openai",
            embedding_model="unknown-model",
            embedding_dimension=2048
        )
        assert settings.get_embedding_dimensions() == 2048

    def test_namespace_defaults(self):
        """Test namespace configuration defaults"""
        from ragbrain.config import Settings

        settings = Settings()
        assert settings.default_namespace is None
        assert settings.namespace_provider == "sqlite"
        assert settings.namespace_db_path == "data/namespaces.db"

    def test_chroma_configuration(self):
        """Test Chroma configuration options"""
        from ragbrain.config import Settings

        settings = Settings(
            vectordb_provider="chroma",
            chroma_collection="test-collection",
            chroma_persist_directory="/data/chroma",
            chroma_host="localhost",
            chroma_port=8001,
            chroma_ssl=True
        )

        assert settings.chroma_collection == "test-collection"
        assert settings.chroma_persist_directory == "/data/chroma"
        assert settings.chroma_host == "localhost"
        assert settings.chroma_port == 8001
        assert settings.chroma_ssl is True

    def test_pinecone_configuration(self):
        """Test Pinecone configuration options"""
        from ragbrain.config import Settings

        settings = Settings(
            vectordb_provider="pinecone",
            pinecone_api_key="test-key",
            pinecone_index="test-index",
            pinecone_cloud="gcp",
            pinecone_region="us-central1"
        )

        assert settings.pinecone_api_key == "test-key"
        assert settings.pinecone_index == "test-index"
        assert settings.pinecone_cloud == "gcp"
        assert settings.pinecone_region == "us-central1"

    def test_fallback_provider_configuration(self):
        """Test fallback provider configuration"""
        from ragbrain.config import Settings

        settings = Settings(
            llm_provider="fallback",
            fallback_primary="ollama",
            fallback_secondary="anthropic"
        )

        assert settings.llm_provider == "fallback"
        assert settings.fallback_primary == "ollama"
        assert settings.fallback_secondary == "anthropic"

    def test_api_keys_optional(self):
        """Test that API keys are optional"""
        from ragbrain.config import Settings

        settings = Settings()
        assert settings.anthropic_api_key is None
        assert settings.openai_api_key is None
        assert settings.cohere_api_key is None

    def test_log_level_default(self):
        """Test default log level"""
        from ragbrain.config import Settings

        settings = Settings()
        assert settings.log_level == "info"

    def test_upload_dir_default(self):
        """Test default upload directory"""
        from ragbrain.config import Settings

        settings = Settings()
        assert settings.upload_dir == "uploads"


class TestGlobalSettings:
    """Tests for global settings instance"""

    def test_global_settings_exists(self):
        """Test that global settings instance exists"""
        from ragbrain.config import settings

        assert settings is not None

    def test_global_settings_is_settings_instance(self):
        """Test that global settings is a Settings instance"""
        from ragbrain.config import settings, Settings

        assert isinstance(settings, Settings)
