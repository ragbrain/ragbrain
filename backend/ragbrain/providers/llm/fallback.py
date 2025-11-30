"""Fallback LLM provider - tries primary provider first, then falls back to secondary"""

from typing import List, Dict, Any
import logging
import httpx
from ragbrain.providers.base import LLMProvider
from ragbrain.config import Settings
from .ollama import OllamaLLMProvider
from .anthropic import AnthropicLLMProvider
from .openai_llm import OpenAILLMProvider

logger = logging.getLogger(__name__)

# Provider class mapping
PROVIDER_CLASSES = {
    "ollama": OllamaLLMProvider,
    "anthropic": AnthropicLLMProvider,
    "openai": OpenAILLMProvider,
}


class FallbackLLMProvider(LLMProvider):
    """LLM provider that tries primary first, falls back to secondary"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._primary_name = settings.fallback_primary
        self._secondary_name = settings.fallback_secondary

        # Create provider instances
        self._primary = PROVIDER_CLASSES[self._primary_name](settings)
        self._secondary = PROVIDER_CLASSES[self._secondary_name](settings)

        self._active_provider, self._active_name, self._status = self._select_provider()

    def _check_provider_health(self, provider_name: str) -> tuple[bool, str]:
        """Check if a provider is available"""
        if provider_name == "ollama":
            return self._check_ollama_health()
        # For API providers, assume available if we have credentials
        # (actual errors will surface at request time)
        elif provider_name == "anthropic":
            if self.settings.anthropic_api_key:
                return True, "ok"
            return False, "no API key configured"
        elif provider_name == "openai":
            if self.settings.openai_api_key:
                return True, "ok"
            return False, "no API key configured"
        return False, f"unknown provider: {provider_name}"

    def _check_ollama_health(self) -> tuple[bool, str]:
        """Check if Ollama is available"""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.settings.ollama_url}/api/tags")
                if response.status_code == 200:
                    return True, "ok"
                else:
                    return False, f"status_code={response.status_code}"
        except httpx.ConnectError:
            return False, f"connection_refused at {self.settings.ollama_url}"
        except httpx.TimeoutException:
            return False, f"timeout connecting to {self.settings.ollama_url}"
        except Exception as e:
            return False, str(e)

    def _select_provider(self) -> tuple[LLMProvider, str, str]:
        """Select provider based on availability"""
        # Try primary first
        available, reason = self._check_provider_health(self._primary_name)
        if available:
            logger.info(f"{self._primary_name} is available, using as primary LLM provider")
            return self._primary, self._primary_name, "ok"

        # Fall back to secondary
        logger.info(f"{self._primary_name} unavailable ({reason}), falling back to {self._secondary_name}")
        sec_available, sec_reason = self._check_provider_health(self._secondary_name)
        if sec_available:
            return self._secondary, self._secondary_name, f"{self._primary_name} unavailable: {reason}"

        # Both failed - still use secondary and let it fail at request time
        logger.warning(f"Both providers unavailable: {self._primary_name}={reason}, {self._secondary_name}={sec_reason}")
        return self._secondary, self._secondary_name, f"both unavailable: {self._primary_name}={reason}, {self._secondary_name}={sec_reason}"

    def get_name(self) -> str:
        """Get provider name with active provider info"""
        if self._status == "ok":
            return f"FallbackLLMProvider(active={self._active_name})"
        else:
            return f"FallbackLLMProvider(active={self._active_name}, reason={self._status})"

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt using active provider"""
        return self._active_provider.generate(prompt, **kwargs)

    def generate_with_context(
        self,
        query: str,
        context: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """Generate answer with context using active provider"""
        return self._active_provider.generate_with_context(query, context, **kwargs)
