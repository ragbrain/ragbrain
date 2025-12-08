"""Provider package - Extensible provider architecture for RAGBrain"""

from .base import EmbeddingProvider, LLMProvider, VectorDBProvider, NamespaceProvider
from .factories import (
    EmbeddingProviderFactory,
    LLMProviderFactory,
    VectorDBProviderFactory,
    NamespaceProviderFactory,
    RerankerProviderFactory
)

# Register namespace providers
from .namespace.sqlite import SQLiteNamespaceProvider
NamespaceProviderFactory.register("sqlite", SQLiteNamespaceProvider)

# Redis provider - only import if redis package is installed
try:
    import redis as _redis_check  # noqa: F401
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

if _REDIS_AVAILABLE:
    from .namespace.redis_provider import RedisNamespaceProvider
    NamespaceProviderFactory.register("redis", RedisNamespaceProvider)

# DynamoDB provider - only import if boto3 is available
try:
    import boto3 as _boto3_check  # noqa: F401
    from .namespace.dynamodb import DynamoDBNamespaceProvider
    NamespaceProviderFactory.register("dynamodb", DynamoDBNamespaceProvider)
except ImportError:
    pass

__all__ = [
    'EmbeddingProvider',
    'LLMProvider',
    'VectorDBProvider',
    'NamespaceProvider',
    'EmbeddingProviderFactory',
    'LLMProviderFactory',
    'VectorDBProviderFactory',
    'NamespaceProviderFactory',
    'RerankerProviderFactory',
]
