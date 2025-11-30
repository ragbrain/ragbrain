"""LLM providers - Auto-register all providers"""

from ragbrain.providers.factories import LLMProviderFactory
from .anthropic import AnthropicLLMProvider
from .openai_llm import OpenAILLMProvider
from .ollama import OllamaLLMProvider
from .fallback import FallbackLLMProvider

# Register providers
LLMProviderFactory.register('anthropic', AnthropicLLMProvider)
LLMProviderFactory.register('openai', OpenAILLMProvider)
LLMProviderFactory.register('ollama', OllamaLLMProvider)
LLMProviderFactory.register('fallback', FallbackLLMProvider)

__all__ = ['AnthropicLLMProvider', 'OpenAILLMProvider', 'OllamaLLMProvider', 'FallbackLLMProvider']
