"""Embedding providers - Auto-register all providers"""

from ragbrain.providers.factories import EmbeddingProviderFactory
from .openai import OpenAIEmbeddingProvider
from .ollama import OllamaEmbeddingProvider
from .mixedbread import MixedbreadEmbeddingProvider
from .fallback import FallbackEmbeddingProvider

# Register providers
EmbeddingProviderFactory.register('openai', OpenAIEmbeddingProvider)
EmbeddingProviderFactory.register('ollama', OllamaEmbeddingProvider)
EmbeddingProviderFactory.register('mixedbread', MixedbreadEmbeddingProvider)
EmbeddingProviderFactory.register('fallback', FallbackEmbeddingProvider)

# Optional providers (only register if dependencies available)
try:
    from .cohere import CohereEmbeddingProvider
    EmbeddingProviderFactory.register('cohere', CohereEmbeddingProvider)
except ImportError:
    pass

# Bedrock provider (optional - requires boto3)
try:
    from .bedrock import BedrockEmbeddingProvider
    EmbeddingProviderFactory.register('bedrock', BedrockEmbeddingProvider)
except ImportError:
    pass

__all__ = [
    'OpenAIEmbeddingProvider',
    'OllamaEmbeddingProvider',
    'MixedbreadEmbeddingProvider',
    'FallbackEmbeddingProvider',
]
