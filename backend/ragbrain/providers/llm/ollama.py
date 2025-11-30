"""Ollama LLM provider"""

from typing import List, Dict, Any
import httpx
from ragbrain.providers.base import LLMProvider
from ragbrain.config import Settings


class OllamaLLMProvider(LLMProvider):
    """Ollama local LLM provider"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.ollama_url
        self.model = settings.ollama_model

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        temperature = kwargs.get('temperature', 0)

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature
                    }
                }
            )
            response.raise_for_status()
            return response.json()["response"]

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
