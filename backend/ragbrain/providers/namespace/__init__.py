"""Namespace providers - Registry for namespace management"""

from ragbrain.providers.factories import NamespaceProviderFactory
from .sqlite import SQLiteNamespaceProvider

# Register providers
NamespaceProviderFactory.register('sqlite', SQLiteNamespaceProvider)

__all__ = ["SQLiteNamespaceProvider"]

# Redis provider is optional
try:
    from .redis_provider import RedisNamespaceProvider
    NamespaceProviderFactory.register('redis', RedisNamespaceProvider)
    __all__.append("RedisNamespaceProvider")
except ImportError:
    pass

# DynamoDB provider (optional - requires boto3)
try:
    from .dynamodb import DynamoDBNamespaceProvider
    NamespaceProviderFactory.register('dynamodb', DynamoDBNamespaceProvider)
    __all__.append("DynamoDBNamespaceProvider")
except ImportError:
    pass
