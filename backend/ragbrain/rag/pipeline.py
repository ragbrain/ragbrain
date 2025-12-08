"""
RAG Pipeline - Using Factory Pattern

Extensible RAG pipeline using the factory pattern.
Allows swapping providers via configuration without code changes.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
import threading
from ragbrain.config import Settings, settings
from ragbrain.providers import (
    EmbeddingProviderFactory,
    LLMProviderFactory,
    VectorDBProviderFactory,
    RerankerProviderFactory
)
from ragbrain.chunking import ChunkingStrategyFactory
# Auto-register providers
import ragbrain.providers.embeddings  # noqa
import ragbrain.providers.llm  # noqa
import ragbrain.providers.vectordb  # noqa
import ragbrain.chunking.strategies  # noqa

import logging

logger = logging.getLogger(__name__)

# Auto-select chunking strategy based on file extension
CHUNKING_BY_EXTENSION = {
    'docx': 'hierarchical',
    'pdf': 'hierarchical',
    'pptx': 'hierarchical',
    'xlsx': 'hierarchical',
    'md': 'markdown',
    'markdown': 'markdown',
    'txt': 'recursive',
    'vtt': 'transcript',
    'srt': 'transcript',
    'epub': 'hierarchical',
}


def get_chunking_strategy_for_file(filename: str) -> str:
    """
    Get the best chunking strategy for a file based on extension.

    Args:
        filename: Name of the file

    Returns:
        Chunking strategy name (hierarchical, markdown, recursive, transcript)
    """
    ext = Path(filename).suffix.lower().lstrip('.')
    return CHUNKING_BY_EXTENSION.get(ext, 'recursive')


class RAGPipeline:
    """
    Extensible RAG pipeline using factory pattern

    Supports:
    - Multiple embedding providers (OpenAI, Cohere, etc.)
    - Multiple LLM providers (Anthropic, OpenAI, etc.)
    - Multiple vector DBs (Qdrant, Pinecone, Chroma, etc.)
    - Swappable via environment variables
    """

    def __init__(self, config: Optional[Settings] = None):
        self.config = config or settings
        self._embedding_provider = None
        self._llm_provider = None
        self._vectordb_provider = None
        self._reranker_provider = None
        self._reranker_initialized = False

    @property
    def embedding_provider(self):
        """Lazy-load embedding provider"""
        if self._embedding_provider is None:
            self._embedding_provider = EmbeddingProviderFactory.create(self.config)
            logger.info(f"Initialized embedding provider: {self._embedding_provider.get_name()}")
        return self._embedding_provider

    @property
    def llm_provider(self):
        """Lazy-load LLM provider"""
        if self._llm_provider is None:
            self._llm_provider = LLMProviderFactory.create(self.config)
            logger.info(f"Initialized LLM provider: {self._llm_provider.get_name()}")
        return self._llm_provider

    @property
    def vectordb_provider(self):
        """Lazy-load vector DB provider"""
        if self._vectordb_provider is None:
            self._vectordb_provider = VectorDBProviderFactory.create(self.config)
            logger.info(f"Initialized vector DB provider: {self._vectordb_provider.get_name()}")
        return self._vectordb_provider

    @property
    def reranker_provider(self):
        """Lazy-load reranker provider (can be None if disabled)"""
        if not self._reranker_initialized:
            self._reranker_provider = RerankerProviderFactory.create(self.config)
            self._reranker_initialized = True
            if self._reranker_provider:
                logger.info(f"Initialized reranker provider: {self._reranker_provider.get_name()}")
        return self._reranker_provider

    def ingest_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunking_strategy: str = "recursive",
        namespace: Optional[str] = None,
        prepend_metadata: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Ingest text into the knowledge base

        Args:
            text: Text to ingest
            metadata: Optional metadata
            chunking_strategy: Strategy for chunking (recursive, markdown, etc.)
            namespace: Optional namespace for isolation (multi-user/multi-project)
            prepend_metadata: List of metadata keys to prepend to each chunk text.
                              This embeds the metadata into the vector for better semantic search.
                              Example: ["speaker", "topic"] would prepend "Speaker: X\nTopic: Y\n\n"

        Returns:
            Result dictionary with chunks_created count
        """
        logger.info(f"Ingesting text (length: {len(text)}, strategy: {chunking_strategy}, namespace: {namespace})")

        # Build metadata prefix first so we can account for its size during chunking
        metadata_prefix = ""
        if prepend_metadata and metadata:
            prefix_lines = []
            for key in prepend_metadata:
                if key in metadata:
                    # Convert key to title case for readability (e.g., "speaker" -> "Speaker")
                    label = key.replace("_", " ").title()
                    prefix_lines.append(f"{label}: {metadata[key]}")
            if prefix_lines:
                metadata_prefix = "\n".join(prefix_lines) + "\n\n"
                logger.info(f"Prepending metadata to chunks: {prepend_metadata}")

        # Calculate effective chunk size accounting for metadata prefix
        prefix_len = len(metadata_prefix)
        if prefix_len > 0:
            # Reduce chunk size to leave room for metadata prefix
            effective_chunk_size = min(
                self.config.chunk_size,
                self.config.chunk_max_size - prefix_len
            )
            # Warn if prefix is larger than reserved space
            if prefix_len > self.config.chunk_metadata_reserve:
                logger.warning(
                    f"Metadata prefix ({prefix_len} chars) exceeds reserved space "
                    f"({self.config.chunk_metadata_reserve} chars). Consider increasing "
                    f"CHUNK_METADATA_RESERVE or reducing metadata."
                )
        else:
            effective_chunk_size = self.config.chunk_size

        # Use chunking strategy factory with adjusted size
        strategy = ChunkingStrategyFactory.create(chunking_strategy)
        chunk_objects = strategy.chunk(
            text,
            chunk_size=effective_chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        # Extract text from chunk objects
        chunks = [chunk.text for chunk in chunk_objects]

        logger.info(f"Created {len(chunks)} chunks (effective size: {effective_chunk_size})")

        # Prepend metadata to each chunk for embedding
        chunks_for_embedding = [metadata_prefix + chunk for chunk in chunks] if metadata_prefix else chunks

        # Generate embeddings (with metadata prefix for better semantic search)
        embeddings = self.embedding_provider.embed_batch(chunks_for_embedding)

        # Prepare metadata with chunk linking
        from datetime import datetime, timezone
        metadata = metadata or {}
        doc_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        metadatas = [
            {
                **metadata,
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "created_at": created_at
            }
            for i, _ in enumerate(chunks)
        ]

        # Use provided namespace or default from config
        ns = namespace or self.config.default_namespace

        # Insert into vector DB (store enriched chunks if metadata was prepended)
        ids = self.vectordb_provider.insert(
            vectors=embeddings,
            texts=chunks_for_embedding,
            metadatas=metadatas,
            namespace=ns
        )

        # Create document summary for catalog/discovery
        # Generate a readable filename if not provided
        if metadata and metadata.get("filename"):
            filename = metadata["filename"]
        else:
            # Create timestamp-based filename with text preview for captures
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            preview = text[:30].replace("\n", " ").strip()
            if len(text) > 30:
                preview += "..."
            filename = f"Capture {timestamp} - {preview}"
        self._create_document_summary(
            doc_id=doc_id,
            filename=filename,
            namespace=ns,
            chunks=chunks,
            chunk_metadatas=metadatas,
            created_at=created_at,
            metadata=metadata
        )

        return {
            "chunks_created": len(chunks),
            "success": True,
            "ids": ids,
            "namespace": ns,
            "doc_id": doc_id
        }

    def ingest_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunking_strategy: str = "auto",
        namespace: Optional[str] = None,
        prepend_metadata: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Ingest a file into the knowledge base with structure-aware processing.

        Uses Docling for hierarchical chunking when appropriate, preserving
        document structure like headings and sections.

        Args:
            file_path: Path to the document file
            metadata: Optional metadata
            chunking_strategy: Strategy for chunking. Use "auto" to select
                              based on file type, or specify explicitly.
            namespace: Optional namespace for isolation
            prepend_metadata: List of metadata keys to prepend to chunks

        Returns:
            Result dictionary with chunks_created count
        """
        path = Path(file_path)
        filename = path.name

        # Auto-select chunking strategy based on file type
        if chunking_strategy == "auto":
            chunking_strategy = get_chunking_strategy_for_file(filename)
            logger.info(f"Auto-selected chunking strategy: {chunking_strategy} for {filename}")

        logger.info(f"Ingesting file: {filename} (strategy: {chunking_strategy}, namespace: {namespace})")

        # Build metadata prefix
        metadata_prefix = ""
        if prepend_metadata and metadata:
            prefix_lines = []
            for key in prepend_metadata:
                if key in metadata:
                    label = key.replace("_", " ").title()
                    prefix_lines.append(f"{label}: {metadata[key]}")
            if prefix_lines:
                metadata_prefix = "\n".join(prefix_lines) + "\n\n"
                logger.info(f"Prepending metadata to chunks: {prepend_metadata}")

        # Calculate effective chunk size
        prefix_len = len(metadata_prefix)
        if prefix_len > 0:
            effective_chunk_size = min(
                self.config.chunk_size,
                self.config.chunk_max_size - prefix_len
            )
        else:
            effective_chunk_size = self.config.chunk_size

        # Use chunking strategy factory
        strategy = ChunkingStrategyFactory.create(chunking_strategy)

        # For hierarchical chunking, pass the file path directly
        if chunking_strategy == "hierarchical":
            chunk_objects = strategy.chunk(
                "",  # Empty text, file_path is used
                chunk_size=effective_chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                file_path=str(path)
            )
        else:
            # For other strategies, load text first
            from ragbrain.loaders import load_document
            text = load_document(str(path), filename)
            chunk_objects = strategy.chunk(
                text,
                chunk_size=effective_chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )

        # Extract text and merge chunk-level metadata
        chunks = []
        chunk_metadatas = []
        for chunk in chunk_objects:
            chunks.append(chunk.text)
            # Include any heading metadata from hierarchical chunking
            chunk_metadatas.append(chunk.metadata)

        logger.info(f"Created {len(chunks)} chunks from {filename}")

        # Prepend metadata to chunks for embedding
        chunks_for_embedding = [metadata_prefix + chunk for chunk in chunks] if metadata_prefix else chunks

        # Generate embeddings
        embeddings = self.embedding_provider.embed_batch(chunks_for_embedding)

        # Prepare metadata
        from datetime import datetime, timezone
        metadata = metadata or {}
        doc_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        metadatas = []
        for i, chunk_meta in enumerate(chunk_metadatas):
            meta = {
                **metadata,
                **chunk_meta,  # Include heading info from hierarchical chunking
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "created_at": created_at,
                "filename": filename,
            }
            metadatas.append(meta)

        # Use provided namespace or default from config
        ns = namespace or self.config.default_namespace

        # Insert into vector DB
        ids = self.vectordb_provider.insert(
            vectors=embeddings,
            texts=chunks_for_embedding,
            metadatas=metadatas,
            namespace=ns
        )

        # Create document summary record for fast catalog and semantic discovery
        self._create_document_summary(
            doc_id=doc_id,
            filename=filename,
            namespace=ns,
            chunks=chunks,
            chunk_metadatas=chunk_metadatas,
            created_at=created_at,
            metadata=metadata
        )

        return {
            "chunks_created": len(chunks),
            "success": True,
            "ids": ids,
            "namespace": ns,
            "doc_id": doc_id,
            "chunking_strategy": chunking_strategy
        }

    def _create_document_summary(
        self,
        doc_id: str,
        filename: str,
        namespace: str,
        chunks: List[str],
        chunk_metadatas: List[Dict[str, Any]],
        created_at: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Create a document summary record in the vector DB for fast catalog
        and semantic discovery.

        The summary record has _type: "document_summary" to distinguish
        it from regular chunks. It contains:
        - Document info (filename, namespace, doc_id)
        - Headings extracted from chunks (for semantic search)
        - First ~1500 chars of content (for topic matching)
        - Chunk count for display

        Args:
            doc_id: The document ID
            filename: Original filename
            namespace: Namespace the document belongs to
            chunks: List of chunk texts
            chunk_metadatas: Metadata from each chunk (may contain headings)
            created_at: ISO timestamp
            metadata: Original document metadata
        """
        try:
            # Extract unique headings from chunk metadata (from hierarchical chunking)
            headings = []
            seen_headings = set()
            for chunk_meta in chunk_metadatas:
                for heading in chunk_meta.get("headings", []):
                    if heading and heading not in seen_headings:
                        headings.append(heading)
                        seen_headings.add(heading)

            # Build summary text for semantic search
            # Format: Document: {filename}\nNamespace: {namespace}\nHeadings: {...}\n\n{first 500 chars}
            summary_parts = [
                f"Document: {filename}",
                f"Namespace: {namespace}"
            ]

            if headings:
                summary_parts.append(f"Headings: {', '.join(headings[:20])}")  # Limit to 20 headings

            # Add first ~1500 chars of content from first chunks for better semantic matching
            content_preview = ""
            char_count = 0
            for chunk in chunks:
                remaining = 1500 - char_count
                if remaining <= 0:
                    break
                content_preview += chunk[:remaining] + " "
                char_count += len(chunk[:remaining])

            if content_preview.strip():
                summary_parts.append("")  # Empty line before content
                summary_parts.append(content_preview.strip())

            summary_text = "\n".join(summary_parts)

            # Generate embedding for summary
            summary_embedding = self.embedding_provider.embed(summary_text)

            # Create summary record - use a new UUID (Qdrant requires valid UUID or integer)
            summary_id = str(uuid.uuid4())
            summary_metadata = {
                "_type": "document_summary",
                "doc_id": doc_id,
                "filename": filename,
                "namespace": namespace,
                "headings": headings[:50],  # Store up to 50 headings
                "chunk_count": len(chunks),
                "created_at": created_at,
                **(metadata or {})
            }

            # Insert summary record
            self.vectordb_provider.insert(
                vectors=[summary_embedding],
                texts=[summary_text],
                metadatas=[summary_metadata],
                ids=[summary_id],
                namespace=namespace
            )

            logger.info(f"Created document summary for {filename} (doc_id: {doc_id}, headings: {len(headings)})")

        except Exception as e:
            # Don't fail the whole ingestion if summary creation fails
            logger.error(f"Failed to create document summary for {filename}: {e}")

    def get_available_chunking_strategies(self) -> List[str]:
        """Get list of available chunking strategies"""
        return ChunkingStrategyFactory.get_available_strategies()

    def query(
        self,
        question: str,
        top_k: int = 5,
        synthesize: bool = True,
        namespace: Optional[str] = None,
        rerank: bool = False
    ) -> Dict[str, Any]:
        """
        Query the knowledge base

        Args:
            question: User question
            top_k: Number of results to retrieve
            synthesize: Whether to use LLM synthesis
            namespace: Optional namespace to search within
            rerank: Whether to rerank results for better relevance

        Returns:
            Dictionary with question, answer (if synthesize=True), and sources
        """
        logger.info(f"Querying: {question} (synthesize={synthesize}, namespace={namespace}, rerank={rerank})")

        # Generate query embedding (uses embed_query for providers that differentiate)
        query_embedding = self.embedding_provider.embed_query(question)

        # Use provided namespace or default from config
        ns = namespace or self.config.default_namespace

        # Search vector DB - fetch more results if reranking
        search_top_k = top_k * 3 if rerank else top_k
        results = self.vectordb_provider.search(
            query_vector=query_embedding,
            top_k=search_top_k,
            namespace=ns
        )

        # Format sources - include namespace in metadata for downstream consumers
        sources = [
            {
                "content": result["content"],
                "metadata": {**result["metadata"], "namespace": result.get("namespace", "default")},
                "score": result.get("score", 0)
            }
            for result in results
        ]

        # Apply reranking if requested
        if rerank and sources and self.reranker_provider:
            logger.info(f"Reranking {len(sources)} results")
            sources = self.reranker_provider.rerank(question, sources, top_k=top_k)
        elif rerank and not self.reranker_provider:
            logger.warning("Reranking requested but no reranker configured")
            sources = sources[:top_k]
        else:
            sources = sources[:top_k]

        response = {
            "question": question,
            "sources": sources,
            "namespace": ns,
            "reranked": rerank and self.reranker_provider is not None
        }

        # Optionally synthesize answer
        if synthesize and sources:
            answer = self.llm_provider.generate_with_context(
                query=question,
                context=sources
            )
            response["answer"] = answer

        return response

    def search(
        self,
        query: str,
        top_k: int = 5,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search without LLM synthesis (faster)

        Args:
            query: Search query
            top_k: Number of results
            namespace: Optional namespace to search within

        Returns:
            Search results
        """
        return self.query(query, top_k=top_k, synthesize=False, namespace=namespace)

    def get_providers_info(self) -> Dict[str, str]:
        """Get information about current providers"""
        return {
            "embedding_provider": self.embedding_provider.get_name(),
            "llm_provider": self.llm_provider.get_name(),
            "vectordb_provider": self.vectordb_provider.get_name(),
            "embedding_dimensions": self.embedding_provider.get_dimensions(),
        }


# Global pipeline instance with thread-safe initialization
_pipeline: Optional[RAGPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> RAGPipeline:
    """Get or create global pipeline instance (thread-safe)"""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            # Double-check pattern
            if _pipeline is None:
                _pipeline = RAGPipeline()
    return _pipeline
