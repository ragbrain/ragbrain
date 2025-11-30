"""Mixedbread embedding provider"""

from typing import List
import httpx
import logging

from ragbrain.providers.base import EmbeddingProvider
from ragbrain.config import Settings

logger = logging.getLogger(__name__)

# Model dimensions
MIXEDBREAD_DIMENSIONS = {
    "mxbai-embed-large-v1": 1024,
    "mixedbread-ai/mxbai-embed-large-v1": 1024,
    "deepset-mxbai-embed-de-large-v1": 1024,
    "mxbai-embed-2d-large-v1": 1024,
}


class MixedbreadEmbeddingProvider(EmbeddingProvider):
    """Mixedbread embedding provider (mixedbread.ai API)"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.mixedbread_api_key
        self.model = settings.mixedbread_model
        self.base_url = "https://api.mixedbread.com/v1"
        self._dimensions = MIXEDBREAD_DIMENSIONS.get(self.model, 1024)
        self.timeout = 60.0

        if not self.api_key:
            raise ValueError("MIXEDBREAD_API_KEY is required for Mixedbread provider")

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Call Mixedbread embeddings API"""
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": texts,
                "normalized": True,
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        # Extract embeddings from response
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings

    def embed(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        try:
            result = self._call_api([text])
            return result[0]
        except Exception as e:
            logger.error(f"Mixedbread embedding failed: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            return self._call_api(texts)
        except Exception as e:
            logger.error(f"Mixedbread batch embedding failed: {e}")
            raise

    def get_dimensions(self) -> int:
        """Get embedding dimensions"""
        return self._dimensions

    def get_name(self) -> str:
        """Get provider name"""
        return f"mixedbread/{self.model}"
