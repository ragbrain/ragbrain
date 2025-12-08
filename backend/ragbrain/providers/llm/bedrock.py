"""AWS Bedrock LLM provider - Native AWS LLM access via IAM"""

import json
import logging
from typing import List, Dict, Any

import boto3
from botocore.exceptions import ClientError

from ragbrain.providers.base import LLMProvider
from ragbrain.config import Settings

logger = logging.getLogger(__name__)


class BedrockLLMProvider(LLMProvider):
    """AWS Bedrock LLM provider for Lambda-native deployments

    Required IAM permissions:
    - bedrock:InvokeModel

    Note: The model must be enabled in your AWS account. Visit the Bedrock
    console to enable specific models before use.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=settings.aws_region
        )
        self.model_id = settings.bedrock_llm_model
        logger.info(f"Bedrock LLM provider initialized: {self.model_id}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt using Bedrock"""
        max_tokens = kwargs.get('max_tokens', 1024)
        temperature = kwargs.get('temperature', 0)

        # Determine model family and format request accordingly
        if 'anthropic' in self.model_id.lower():
            return self._generate_anthropic(prompt, max_tokens, temperature)
        elif 'titan' in self.model_id.lower():
            return self._generate_titan(prompt, max_tokens, temperature)
        elif 'meta' in self.model_id.lower():
            return self._generate_llama(prompt, max_tokens, temperature)
        else:
            # Default to Anthropic format
            return self._generate_anthropic(prompt, max_tokens, temperature)

    def _invoke_model(self, body: dict) -> dict:
        """Invoke Bedrock model with error handling"""
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )
            return json.loads(response['body'].read())
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'AccessDeniedException':
                raise RuntimeError(
                    f"Access denied to Bedrock model '{self.model_id}'. "
                    "Ensure the model is enabled in your AWS account and IAM permissions are configured."
                ) from e
            elif error_code == 'ThrottlingException':
                raise RuntimeError(
                    f"Bedrock request throttled for model '{self.model_id}'. "
                    "Consider implementing retry logic or reducing request rate."
                ) from e
            elif error_code == 'ModelNotReadyException':
                raise RuntimeError(
                    f"Bedrock model '{self.model_id}' is not ready. Please try again later."
                ) from e
            elif error_code == 'ValidationException':
                raise ValueError(
                    f"Invalid request to Bedrock model '{self.model_id}': {e.response['Error']['Message']}"
                ) from e
            else:
                logger.error(f"Bedrock error ({error_code}): {e}")
                raise

    def _generate_anthropic(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Anthropic Claude models on Bedrock"""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response_body = self._invoke_model(body)
        return response_body['content'][0]['text']

    def _generate_titan(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Amazon Titan models"""
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": 0.9
            }
        }

        response_body = self._invoke_model(body)
        return response_body['results'][0]['outputText']

    def _generate_llama(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using Meta Llama models on Bedrock"""
        body = {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": 0.9
        }

        response_body = self._invoke_model(body)
        return response_body['generation']

    def generate_with_context(
        self,
        query: str,
        context: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """Generate answer with context (RAG)"""
        # Build context string
        context_str = "\n\n".join([
            f"[Source {i+1}]\n{chunk['content']}"
            for i, chunk in enumerate(context)
        ])

        # Build prompt (same pattern as Anthropic provider)
        prompt = f"""You are a helpful AI assistant with access to the user's knowledge base.

Answer the question directly using the information below. Do not mention "context", "provided information", or reference where the information came from - just answer naturally as if you know this information. If you don't have enough information to answer, simply say you don't have that information.

Information:
{context_str}

Question: {query}

Answer:"""

        return self.generate(prompt, **kwargs)
