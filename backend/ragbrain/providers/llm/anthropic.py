"""Anthropic Claude LLM provider"""

from typing import List, Dict, Any
from ragbrain.providers.base import LLMProvider
from ragbrain.config import Settings
from anthropic import Anthropic


class AnthropicLLMProvider(LLMProvider):
    """Anthropic Claude LLM provider"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.get_llm_model()

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        max_tokens = kwargs.get('max_tokens', 1024)
        temperature = kwargs.get('temperature', 0)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return message.content[0].text

    def generate_with_context(
        self,
        query: str,
        context: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """Generate answer with context"""
        # Build context string
        context_str = "\n\n".join([
            f"[Source {i+1}]\n{chunk['content']}"
            for i, chunk in enumerate(context)
        ])

        # Build prompt
        prompt = f"""You are a helpful AI assistant with access to the user's knowledge base.

Answer the question directly using the information below. Do not mention "context", "provided information", or reference where the information came from - just answer naturally as if you know this information. If you don't have enough information to answer, simply say you don't have that information.

Information:
{context_str}

Question: {query}

Answer:"""

        return self.generate(prompt, **kwargs)
