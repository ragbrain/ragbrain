"""Cohere embedding provider"""

from typing import List
from ragbrain.providers.base import EmbeddingProvider
from ragbrain.config import Settings
import cohere


class CohereEmbeddingProvider(EmbeddingProvider):
    """Cohere embedding provider"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = cohere.Client(api_key=settings.cohere_api_key)
        self.model = settings.get_embedding_model()
        self._dimensions = 1024  # Cohere embed-english-v3.0 is 1024

    def embed(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        response = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="search_document"
        )
        return response.embeddings[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        response = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="search_document"
        )
        return response.embeddings

    def get_dimensions(self) -> int:
        """Get embedding dimensions"""
        return self._dimensions
