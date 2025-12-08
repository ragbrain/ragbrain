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

# Bedrock provider (optional - requires boto3)
try:
    from .bedrock import BedrockLLMProvider
    LLMProviderFactory.register('bedrock', BedrockLLMProvider)
except ImportError:
    pass

__all__ = ['AnthropicLLMProvider', 'OpenAILLMProvider', 'OllamaLLMProvider', 'FallbackLLMProvider']
