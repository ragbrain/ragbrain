"""Pytest fixtures and configuration for RAGBrain tests"""

import pytest
import os
import tempfile
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any


# Set up test environment variables before importing modules
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")


@pytest.fixture
def sample_text():
    """Sample text for chunking tests"""
    return """# Introduction

This is a sample document for testing the RAGBrain chunking strategies.
It contains multiple paragraphs and sections.

## Section 1: Overview

The RAGBrain system is designed to help you organize and query your personal knowledge base.
It uses advanced AI techniques to understand and retrieve information.

This paragraph provides more details about the system architecture.
It includes information about embeddings, vector databases, and LLM synthesis.

## Section 2: Features

- Feature 1: Document ingestion
- Feature 2: Semantic search
- Feature 3: AI-powered answers

### Subsection 2.1: Document Ingestion

The document ingestion pipeline supports multiple file formats including PDF, EPUB, DOCX, and Markdown.

## Conclusion

RAGBrain provides a powerful way to build your personal knowledge base."""


@pytest.fixture
def long_sample_text():
    """Longer sample text for testing chunk overlap"""
    paragraphs = []
    for i in range(50):
        paragraphs.append(
            f"This is paragraph {i+1}. It contains important information about topic {i+1}. "
            f"The content is designed to test chunking behavior with longer documents. "
            f"Each paragraph should be substantial enough to test overlap handling."
        )
    return "\n\n".join(paragraphs)


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider"""
    provider = MagicMock()
    provider.embed.return_value = [0.1] * 1536
    provider.embed_batch.return_value = [[0.1] * 1536, [0.2] * 1536]
    provider.get_dimensions.return_value = 1536
    provider.get_name.return_value = "MockEmbeddingProvider"
    return provider


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider"""
    provider = MagicMock()
    provider.generate.return_value = "This is a generated response."
    provider.generate_with_context.return_value = "Based on the context, here is my answer."
    provider.get_name.return_value = "MockLLMProvider"
    return provider


@pytest.fixture
def mock_vectordb_provider():
    """Mock vector database provider"""
    provider = MagicMock()
    provider.insert.return_value = ["id1", "id2", "id3"]
    provider.search.return_value = [
        {
            "content": "Sample content 1",
            "metadata": {"filename": "test.pdf", "chunk_index": 0},
            "score": 0.95
        },
        {
            "content": "Sample content 2",
            "metadata": {"filename": "test.pdf", "chunk_index": 1},
            "score": 0.85
        }
    ]
    provider.delete.return_value = True
    provider.get_collection_info.return_value = {"vectors_count": 100}
    provider.get_name.return_value = "MockVectorDBProvider"
    return provider


@pytest.fixture
def mock_namespace_provider():
    """Mock namespace provider"""
    provider = MagicMock()
    provider.create.return_value = {
        "id": "test-ns",
        "name": "Test Namespace",
        "description": "A test namespace"
    }
    provider.get.return_value = {
        "id": "test-ns",
        "name": "Test Namespace",
        "description": "A test namespace"
    }
    provider.list.return_value = [
        {"id": "ns1", "name": "Namespace 1"},
        {"id": "ns2", "name": "Namespace 2"}
    ]
    provider.exists.return_value = True
    provider.get_name.return_value = "MockNamespaceProvider"
    return provider


@pytest.fixture
def test_settings():
    """Test settings configuration"""
    from ragbrain.config import Settings

    return Settings(
        llm_provider="anthropic",
        embedding_provider="openai",
        vectordb_provider="qdrant",
        anthropic_api_key="test-key",
        openai_api_key="test-key",
        qdrant_url="http://localhost:6333",
        chunk_size=500,
        chunk_overlap=50
    )


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
