"""Provider factories - Factory pattern for creating providers"""

from typing import Dict, Type, Optional
from ragbrain.config import Settings
from .base import EmbeddingProvider, LLMProvider, VectorDBProvider, NamespaceProvider
from .reranker.base import RerankerProvider
import logging

logger = logging.getLogger(__name__)


class EmbeddingProviderFactory:
    """Factory for creating embedding providers"""

    _providers: Dict[str, Type[EmbeddingProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[EmbeddingProvider]):
        """
        Register a new embedding provider

        Args:
            name: Provider name (e.g., 'openai', 'cohere')
            provider_class: Provider class
        """
        cls._providers[name] = provider_class
        logger.info(f"Registered embedding provider: {name}")

    @classmethod
    def create(cls, settings: Settings) -> EmbeddingProvider:
        """
        Create embedding provider based on settings

        Args:
            settings: Application settings

        Returns:
            Embedding provider instance
        """
        provider_name = settings.embedding_provider
        provider_class = cls._providers.get(provider_name)

        if not provider_class:
            available = ', '.join(cls._providers.keys())
            raise ValueError(
                f"Unknown embedding provider: {provider_name}. "
                f"Available: {available}"
            )

        logger.info(f"Creating embedding provider: {provider_name}")
        return provider_class(settings)

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available providers"""
        return list(cls._providers.keys())


class LLMProviderFactory:
    """Factory for creating LLM providers"""

    _providers: Dict[str, Type[LLMProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[LLMProvider]):
        """
        Register a new LLM provider

        Args:
            name: Provider name (e.g., 'anthropic', 'openai')
            provider_class: Provider class
        """
        cls._providers[name] = provider_class
        logger.info(f"Registered LLM provider: {name}")

    @classmethod
    def create(cls, settings: Settings) -> LLMProvider:
        """
        Create LLM provider based on settings

        Args:
            settings: Application settings

        Returns:
            LLM provider instance
        """
        provider_name = settings.llm_provider
        provider_class = cls._providers.get(provider_name)

        if not provider_class:
            available = ', '.join(cls._providers.keys())
            raise ValueError(
                f"Unknown LLM provider: {provider_name}. "
                f"Available: {available}"
            )

        logger.info(f"Creating LLM provider: {provider_name}")
        return provider_class(settings)

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available providers"""
        return list(cls._providers.keys())


class VectorDBProviderFactory:
    """Factory for creating vector database providers"""

    _providers: Dict[str, Type[VectorDBProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[VectorDBProvider]):
        """
        Register a new vector DB provider

        Args:
            name: Provider name (e.g., 'qdrant', 'pinecone')
            provider_class: Provider class
        """
        cls._providers[name] = provider_class
        logger.info(f"Registered vector DB provider: {name}")

    @classmethod
    def create(cls, settings: Settings) -> VectorDBProvider:
        """
        Create vector DB provider based on settings

        Args:
            settings: Application settings

        Returns:
            Vector DB provider instance
        """
        provider_name = settings.vectordb_provider
        provider_class = cls._providers.get(provider_name)

        if not provider_class:
            available = ', '.join(cls._providers.keys())
            raise ValueError(
                f"Unknown vector DB provider: {provider_name}. "
                f"Available: {available}"
            )

        logger.info(f"Creating vector DB provider: {provider_name}")
        return provider_class(settings)

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available providers"""
        return list(cls._providers.keys())


class NamespaceProviderFactory:
    """Factory for creating namespace registry providers"""

    _providers: Dict[str, Type[NamespaceProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[NamespaceProvider]):
        """
        Register a new namespace provider

        Args:
            name: Provider name (e.g., 'sqlite', 'postgresql')
            provider_class: Provider class
        """
        cls._providers[name] = provider_class
        logger.info(f"Registered namespace provider: {name}")

    @classmethod
    def create(cls, settings: Settings) -> NamespaceProvider:
        """
        Create namespace provider based on settings

        Args:
            settings: Application settings

        Returns:
            Namespace provider instance
        """
        provider_name = settings.namespace_provider
        provider_class = cls._providers.get(provider_name)

        if not provider_class:
            available = ', '.join(cls._providers.keys())
            raise ValueError(
                f"Unknown namespace provider: {provider_name}. "
                f"Available: {available}"
            )

        logger.info(f"Creating namespace provider: {provider_name}")
        return provider_class(settings)

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available providers"""
        return list(cls._providers.keys())


class RerankerProviderFactory:
    """Factory for creating reranker providers"""

    @classmethod
    def create(cls, settings: Settings) -> Optional[RerankerProvider]:
        """
        Create reranker provider based on settings

        Args:
            settings: Application settings

        Returns:
            Reranker provider instance, or None if disabled
        """
        provider_name = settings.reranker_provider

        if provider_name == "none":
            logger.info("Reranking disabled")
            return None

        if provider_name == "simple":
            from .reranker.simple import SimpleReranker
            logger.info("Creating simple local reranker")
            return SimpleReranker(dedupe_threshold=settings.reranker_dedupe_threshold)

        if provider_name == "ollama":
            from .reranker.ollama import OllamaReranker
            logger.info(f"Creating Ollama reranker with model {settings.reranker_model}")
            return OllamaReranker(
                base_url=settings.ollama_url,
                model=settings.reranker_model
            )

        if provider_name == "cohere":
            from .reranker.cohere import CohereReranker
            if not settings.cohere_api_key:
                logger.warning("Cohere API key not set, falling back to simple reranker")
                from .reranker.simple import SimpleReranker
                return SimpleReranker(dedupe_threshold=settings.reranker_dedupe_threshold)
            logger.info("Creating Cohere reranker")
            return CohereReranker(api_key=settings.cohere_api_key)

        raise ValueError(f"Unknown reranker provider: {provider_name}")
