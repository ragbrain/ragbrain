"""Namespace providers - Registry for namespace management"""

from .sqlite import SQLiteNamespaceProvider

__all__ = ["SQLiteNamespaceProvider"]

# Redis provider is optional
try:
    from .redis_provider import RedisNamespaceProvider
    __all__.append("RedisNamespaceProvider")
except ImportError:
    pass
