"""Vector DB providers - Auto-register all providers"""

from ragbrain.providers.factories import VectorDBProviderFactory
from .qdrant import QdrantVectorDBProvider

# Register providers
VectorDBProviderFactory.register('qdrant', QdrantVectorDBProvider)

# Optional providers (only register if dependencies available)
try:
    from .pinecone_provider import PineconeVectorDBProvider
    VectorDBProviderFactory.register('pinecone', PineconeVectorDBProvider)
except ImportError:
    pass

try:
    from .chroma import ChromaVectorDBProvider
    VectorDBProviderFactory.register('chroma', ChromaVectorDBProvider)
except ImportError:
    pass

# S3 Vectors provider (optional - requires boto3)
try:
    from .s3vectors import S3VectorsProvider
    VectorDBProviderFactory.register('s3vectors', S3VectorsProvider)
except ImportError:
    pass

__all__ = ['QdrantVectorDBProvider']
