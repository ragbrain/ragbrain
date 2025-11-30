# Contributing

## Setup

```bash
# Clone and setup
git clone https://github.com/ragbrain/ragbrain.git
cd ragbrain/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy env and add API keys
cp .env.example .env

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Run backend
uvicorn ragbrain.api.main:app --reload --port 8000
```

## Tests

```bash
cd backend
pytest                                    # all tests
pytest --cov=ragbrain --cov-report=html   # with coverage
pytest tests/test_chunking.py -v          # specific file
```

## Adding Providers

RAGBrain uses a factory pattern. Implement the base class and register it.

### Embedding Provider

```python
# backend/ragbrain/providers/embeddings/your_provider.py
from ragbrain.providers.base import EmbeddingProvider
from ragbrain.providers.factories import EmbeddingProviderFactory

class YourEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings):
        self.settings = settings

    def embed(self, text: str) -> List[float]:
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass

    def get_dimensions(self) -> int:
        pass

EmbeddingProviderFactory.register("your_provider", YourEmbeddingProvider)
```

### LLM Provider

```python
# backend/ragbrain/providers/llm/your_provider.py
from ragbrain.providers.base import LLMProvider
from ragbrain.providers.factories import LLMProviderFactory

class YourLLMProvider(LLMProvider):
    def __init__(self, settings):
        self.settings = settings

    def generate(self, prompt: str, **kwargs) -> str:
        pass

    def generate_with_context(self, query: str, context: List[Dict], **kwargs) -> str:
        pass

LLMProviderFactory.register("your_provider", YourLLMProvider)
```

### Vector DB Provider

```python
# backend/ragbrain/providers/vectordb/your_provider.py
from ragbrain.providers.base import VectorDBProvider
from ragbrain.providers.factories import VectorDBProviderFactory

class YourVectorDBProvider(VectorDBProvider):
    def __init__(self, settings):
        self.settings = settings

    def insert(self, vectors, texts, metadatas=None, ids=None, namespace=None):
        pass

    def search(self, query_vector, top_k=5, filter=None, namespace=None):
        pass

VectorDBProviderFactory.register("your_provider", YourVectorDBProvider)
```

### Chunking Strategy

```python
# backend/ragbrain/chunking/your_strategy.py
from ragbrain.chunking.base import ChunkingStrategy, Chunk
from ragbrain.chunking.factory import ChunkingStrategyFactory

class YourChunkingStrategy(ChunkingStrategy):
    def chunk(self, text: str, chunk_size: int = 2000, chunk_overlap: int = 200, **kwargs) -> List[Chunk]:
        chunks = []
        # your logic here
        return [
            Chunk(text=chunk_text, index=i, metadata={"strategy": "your_strategy"})
            for i, chunk_text in enumerate(chunks)
        ]

ChunkingStrategyFactory.register("your_strategy", YourChunkingStrategy)
```
