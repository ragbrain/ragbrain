"""Base provider interfaces - Abstract base classes for all providers"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_dimensions(self) -> int:
        """
        Get the dimension size of embeddings

        Returns:
            Number of dimensions in embedding vectors
        """
        pass

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a search query.

        Some embedding models (e.g., Cohere) use different parameters for
        queries vs documents. Override this method if your provider needs
        different behavior for query embeddings.

        Args:
            text: Query text to embed

        Returns:
            List of floats representing the embedding vector
        """
        return self.embed(text)

    def get_name(self) -> str:
        """Get provider name"""
        return self.__class__.__name__


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt

        Args:
            prompt: Input prompt
            **kwargs: Provider-specific parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_with_context(
        self,
        query: str,
        context: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """
        Generate answer given query and context

        Args:
            query: User query
            context: List of context chunks with metadata
            **kwargs: Provider-specific parameters

        Returns:
            Generated answer
        """
        pass

    def get_name(self) -> str:
        """Get provider name"""
        return self.__class__.__name__


class NamespaceProvider(ABC):
    """Abstract base class for namespace registry providers"""

    @abstractmethod
    def create(
        self,
        id: str,
        name: str,
        description: str = "",
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new namespace

        Args:
            id: Unique namespace ID (slug, e.g., 'mba/finance/corporate-finance')
            name: Display name (e.g., 'Corporate Finance')
            description: What belongs in this namespace
            parent_id: Parent namespace ID for hierarchy
            metadata: Additional metadata (tags, icon, color, etc.)

        Returns:
            Created namespace record
        """
        pass

    @abstractmethod
    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Get a namespace by ID

        Args:
            id: Namespace ID

        Returns:
            Namespace record or None if not found
        """
        pass

    @abstractmethod
    def list(
        self,
        parent_id: Optional[str] = None,
        include_children: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List namespaces

        Args:
            parent_id: Filter by parent (None for root namespaces)
            include_children: If True, recursively include children

        Returns:
            List of namespace records
        """
        pass

    @abstractmethod
    def update(
        self,
        id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update a namespace

        Args:
            id: Namespace ID to update
            name: New display name (if provided)
            description: New description (if provided)
            parent_id: New parent ID (if provided)
            metadata: New metadata (merged with existing if provided)

        Returns:
            Updated namespace record or None if not found
        """
        pass

    @abstractmethod
    def delete(self, id: str, cascade: bool = False) -> bool:
        """
        Delete a namespace

        Args:
            id: Namespace ID to delete
            cascade: If True, delete children; if False, fail if children exist

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def get_tree(self, root_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get namespace hierarchy as a tree

        Args:
            root_id: Start from this namespace (None for full tree)

        Returns:
            List of namespaces with nested 'children' arrays
        """
        pass

    @abstractmethod
    def exists(self, id: str) -> bool:
        """
        Check if a namespace exists

        Args:
            id: Namespace ID

        Returns:
            True if exists
        """
        pass

    def get_name(self) -> str:
        """Get provider name"""
        return self.__class__.__name__


class VectorDBProvider(ABC):
    """Abstract base class for vector database providers"""

    @abstractmethod
    def insert(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None
    ) -> List[str]:
        """
        Insert vectors into the database

        Args:
            vectors: List of embedding vectors
            texts: Corresponding text chunks
            metadatas: Optional metadata for each vector
            ids: Optional IDs for each vector (generated if not provided)
            namespace: Optional namespace for isolation (multi-user/multi-project)

        Returns:
            List of inserted vector IDs
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filter
            namespace: Optional namespace to search within

        Returns:
            List of results with text, metadata, and scores
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str], namespace: Optional[str] = None) -> bool:
        """
        Delete vectors by IDs

        Args:
            ids: List of vector IDs to delete
            namespace: Optional namespace to delete from

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection

        Returns:
            Dictionary with collection stats
        """
        pass

    def delete_by_metadata(self, field: str, value: str, namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete vectors by metadata field value

        Args:
            field: Metadata field name (e.g., 'filename')
            value: Value to match
            namespace: Optional namespace filter

        Returns:
            Dictionary with deleted count and IDs
        """
        raise NotImplementedError("delete_by_metadata not implemented for this provider")

    def get_name(self) -> str:
        """Get provider name"""
        return self.__class__.__name__
