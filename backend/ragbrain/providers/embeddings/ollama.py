"""Ollama embedding provider"""

from typing import List
import httpx
import logging

from ragbrain.providers.base import EmbeddingProvider
from ragbrain.config import Settings

logger = logging.getLogger(__name__)

# Known embedding model dimensions
OLLAMA_EMBEDDING_DIMENSIONS = {
    "mxbai-embed-large": 1024,
    "nomic-embed-text": 768,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
    "bge-large": 1024,
    "bge-m3": 1024,
}


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider for local embeddings"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.ollama_url.rstrip('/')
        self.model = settings.ollama_embedding_model
        self._dimensions = OLLAMA_EMBEDDING_DIMENSIONS.get(
            self.model,
            settings.embedding_dimension
        )
        self.timeout = 60.0  # Embeddings can take a moment on first load

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Call Ollama embeddings API"""
        embeddings = []

        for text in texts:
            response = httpx.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            embeddings.append(data["embedding"])

        return embeddings

    def embed(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        try:
            result = self._call_api([text])
            return result[0]
        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            return self._call_api(texts)
        except Exception as e:
            logger.error(f"Ollama batch embedding failed: {e}")
            raise

    def get_dimensions(self) -> int:
        """Get embedding dimensions"""
        return self._dimensions

    def get_name(self) -> str:
        """Get provider name"""
        return f"ollama/{self.model}"

    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
