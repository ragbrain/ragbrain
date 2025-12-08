"""Tests for provider factories and base classes"""

import pytest
from unittest.mock import MagicMock, patch
from ragbrain.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    VectorDBProvider,
    NamespaceProvider
)
from ragbrain.providers.factories import (
    EmbeddingProviderFactory,
    LLMProviderFactory,
    VectorDBProviderFactory,
    NamespaceProviderFactory
)


class TestEmbeddingProviderBase:
    """Tests for EmbeddingProvider abstract base class"""

    def test_cannot_instantiate_abstract(self):
        """Should not be able to instantiate abstract class directly"""
        with pytest.raises(TypeError):
            EmbeddingProvider()

    def test_concrete_implementation_required_methods(self):
        """Concrete implementation must implement all abstract methods"""

        class MinimalEmbeddingProvider(EmbeddingProvider):
            def embed(self, text):
                return [0.1] * 10

            def embed_batch(self, texts):
                return [[0.1] * 10 for _ in texts]

            def get_dimensions(self):
                return 10

        provider = MinimalEmbeddingProvider()
        assert provider.embed("test") == [0.1] * 10
        assert provider.embed_batch(["a", "b"]) == [[0.1] * 10, [0.1] * 10]
        assert provider.get_dimensions() == 10

    def test_get_name_default(self):
        """get_name should return class name by default"""

        class TestEmbeddingProvider(EmbeddingProvider):
            def embed(self, text):
                return []

            def embed_batch(self, texts):
                return []

            def get_dimensions(self):
                return 0

        provider = TestEmbeddingProvider()
        assert provider.get_name() == "TestEmbeddingProvider"


class TestLLMProviderBase:
    """Tests for LLMProvider abstract base class"""

    def test_cannot_instantiate_abstract(self):
        """Should not be able to instantiate abstract class directly"""
        with pytest.raises(TypeError):
            LLMProvider()

    def test_concrete_implementation_required_methods(self):
        """Concrete implementation must implement all abstract methods"""

        class MinimalLLMProvider(LLMProvider):
            def generate(self, prompt, **kwargs):
                return "Generated text"

            def generate_with_context(self, query, context, **kwargs):
                return "Context-based answer"

        provider = MinimalLLMProvider()
        assert provider.generate("test") == "Generated text"
        assert provider.generate_with_context("q", []) == "Context-based answer"

    def test_get_name_default(self):
        """get_name should return class name by default"""

        class TestLLMProvider(LLMProvider):
            def generate(self, prompt, **kwargs):
                return ""

            def generate_with_context(self, query, context, **kwargs):
                return ""

        provider = TestLLMProvider()
        assert provider.get_name() == "TestLLMProvider"


class TestVectorDBProviderBase:
    """Tests for VectorDBProvider abstract base class"""

    def test_cannot_instantiate_abstract(self):
        """Should not be able to instantiate abstract class directly"""
        with pytest.raises(TypeError):
            VectorDBProvider()

    def test_concrete_implementation_required_methods(self):
        """Concrete implementation must implement all abstract methods"""

        class MinimalVectorDBProvider(VectorDBProvider):
            def insert(self, vectors, texts, metadatas=None, ids=None, namespace=None):
                return ["id1"]

            def search(self, query_vector, top_k=5, filter=None, namespace=None):
                return []

            def delete(self, ids, namespace=None):
                return True

            def get_collection_info(self):
                return {"count": 0}

        provider = MinimalVectorDBProvider()
        assert provider.insert([[0.1]], ["text"]) == ["id1"]
        assert provider.search([0.1]) == []
        assert provider.delete(["id1"]) is True
        assert provider.get_collection_info() == {"count": 0}

    def test_delete_by_metadata_not_implemented(self):
        """delete_by_metadata raises NotImplementedError by default"""

        class MinimalVectorDBProvider(VectorDBProvider):
            def insert(self, vectors, texts, metadatas=None, ids=None, namespace=None):
                return []

            def search(self, query_vector, top_k=5, filter=None, namespace=None):
                return []

            def delete(self, ids, namespace=None):
                return True

            def get_collection_info(self):
                return {}

        provider = MinimalVectorDBProvider()
        with pytest.raises(NotImplementedError):
            provider.delete_by_metadata("field", "value")


class TestNamespaceProviderBase:
    """Tests for NamespaceProvider abstract base class"""

    def test_cannot_instantiate_abstract(self):
        """Should not be able to instantiate abstract class directly"""
        with pytest.raises(TypeError):
            NamespaceProvider()

    def test_concrete_implementation_required_methods(self):
        """Concrete implementation must implement all abstract methods"""

        class MinimalNamespaceProvider(NamespaceProvider):
            def create(self, id, name, description="", parent_id=None, metadata=None):
                return {"id": id, "name": name}

            def get(self, id):
                return {"id": id}

            def list(self, parent_id=None, include_children=False):
                return []

            def update(self, id, name=None, description=None, parent_id=None, metadata=None):
                return {"id": id}

            def delete(self, id, cascade=False):
                return True

            def get_tree(self, root_id=None):
                return []

            def exists(self, id):
                return True

        provider = MinimalNamespaceProvider()
        assert provider.create("ns1", "Namespace 1") == {"id": "ns1", "name": "Namespace 1"}
        assert provider.get("ns1") == {"id": "ns1"}
        assert provider.list() == []
        assert provider.exists("ns1") is True


class TestEmbeddingProviderFactory:
    """Tests for EmbeddingProviderFactory"""

    def test_register_provider(self):
        """Should register a provider"""

        class TestProvider(EmbeddingProvider):
            def __init__(self, settings):
                pass

            def embed(self, text):
                return []

            def embed_batch(self, texts):
                return []

            def get_dimensions(self):
                return 0

        EmbeddingProviderFactory.register("test_embedding", TestProvider)
        assert "test_embedding" in EmbeddingProviderFactory.get_available_providers()

        # Clean up
        del EmbeddingProviderFactory._providers["test_embedding"]

    def test_create_unknown_provider(self):
        """Should raise validation error for unknown provider"""
        from ragbrain.config import Settings
        from pydantic import ValidationError

        # Pydantic validates provider names, so invalid names raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            settings = Settings(embedding_provider="nonexistent")
        assert "embedding_provider" in str(exc_info.value)

    def test_get_available_providers(self):
        """Should list available providers"""
        # Import to trigger registration
        import ragbrain.providers.embeddings  # noqa

        providers = EmbeddingProviderFactory.get_available_providers()
        assert isinstance(providers, list)
        assert "openai" in providers


class TestLLMProviderFactory:
    """Tests for LLMProviderFactory"""

    def test_register_provider(self):
        """Should register a provider"""

        class TestLLM(LLMProvider):
            def __init__(self, settings):
                pass

            def generate(self, prompt, **kwargs):
                return ""

            def generate_with_context(self, query, context, **kwargs):
                return ""

        LLMProviderFactory.register("test_llm", TestLLM)
        assert "test_llm" in LLMProviderFactory.get_available_providers()

        # Clean up
        del LLMProviderFactory._providers["test_llm"]

    def test_create_unknown_provider(self):
        """Should raise validation error for unknown provider"""
        from ragbrain.config import Settings
        from pydantic import ValidationError

        # Pydantic validates provider names, so invalid names raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            settings = Settings(llm_provider="nonexistent")
        assert "llm_provider" in str(exc_info.value)

    def test_get_available_providers(self):
        """Should list available providers"""
        import ragbrain.providers.llm  # noqa

        providers = LLMProviderFactory.get_available_providers()
        assert isinstance(providers, list)
        assert "anthropic" in providers


class TestVectorDBProviderFactory:
    """Tests for VectorDBProviderFactory"""

    def test_register_provider(self):
        """Should register a provider"""

        class TestVectorDB(VectorDBProvider):
            def __init__(self, settings):
                pass

            def insert(self, vectors, texts, metadatas=None, ids=None, namespace=None):
                return []

            def search(self, query_vector, top_k=5, filter=None, namespace=None):
                return []

            def delete(self, ids, namespace=None):
                return True

            def get_collection_info(self):
                return {}

        VectorDBProviderFactory.register("test_vectordb", TestVectorDB)
        assert "test_vectordb" in VectorDBProviderFactory.get_available_providers()

        # Clean up
        del VectorDBProviderFactory._providers["test_vectordb"]

    def test_create_unknown_provider(self):
        """Should raise validation error for unknown provider"""
        from ragbrain.config import Settings
        from pydantic import ValidationError

        # Pydantic validates provider names, so invalid names raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            settings = Settings(vectordb_provider="nonexistent")
        assert "vectordb_provider" in str(exc_info.value)

    def test_get_available_providers(self):
        """Should list available providers"""
        import ragbrain.providers.vectordb  # noqa

        providers = VectorDBProviderFactory.get_available_providers()
        assert isinstance(providers, list)
        assert "qdrant" in providers


class TestNamespaceProviderFactory:
    """Tests for NamespaceProviderFactory"""

    def test_register_provider(self):
        """Should register a provider"""

        class TestNamespace(NamespaceProvider):
            def __init__(self, settings):
                pass

            def create(self, id, name, description="", parent_id=None, metadata=None):
                return {}

            def get(self, id):
                return None

            def list(self, parent_id=None, include_children=False):
                return []

            def update(self, id, name=None, description=None, parent_id=None, metadata=None):
                return None

            def delete(self, id, cascade=False):
                return True

            def get_tree(self, root_id=None):
                return []

            def exists(self, id):
                return False

        NamespaceProviderFactory.register("test_namespace", TestNamespace)
        assert "test_namespace" in NamespaceProviderFactory.get_available_providers()

        # Clean up
        del NamespaceProviderFactory._providers["test_namespace"]

    def test_create_unknown_provider(self):
        """Should raise validation error for unknown provider"""
        from ragbrain.config import Settings
        from pydantic import ValidationError

        # Pydantic validates provider names, so invalid names raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            settings = Settings(namespace_provider="nonexistent")
        assert "namespace_provider" in str(exc_info.value)
