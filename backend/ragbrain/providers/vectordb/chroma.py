"""Chroma vector database provider"""

from typing import List, Dict, Any, Optional
from ragbrain.providers.base import VectorDBProvider
from ragbrain.config import Settings
import uuid

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class ChromaVectorDBProvider(VectorDBProvider):
    """
    Chroma vector database provider

    Supports both:
    - Local/embedded mode (default)
    - Client/server mode (HTTP)
    """

    def __init__(self, settings: Settings):
        if not CHROMA_AVAILABLE:
            raise ImportError("Chroma package not installed. Run: pip install chromadb")

        self.settings = settings
        self.collection_name = settings.chroma_collection
        self.dimensions = settings.embedding_dimension

        # Initialize Chroma client
        if settings.chroma_host:
            # Client/server mode
            self.client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port or 8000,
                ssl=settings.chroma_ssl or False,
                headers={"Authorization": f"Bearer {settings.chroma_api_key}"} if settings.chroma_api_key else None
            )
        else:
            # Embedded mode (local persistence)
            persist_directory = settings.chroma_persist_directory or "./data/chroma"
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False)
            )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

    def insert(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None
    ) -> List[str]:
        """Insert vectors into Chroma"""
        if not ids:
            ids = [str(uuid.uuid4()) for _ in vectors]

        if not metadatas:
            metadatas = [{} for _ in vectors]

        # Chroma requires string values in metadata
        sanitized_metadatas = []
        for metadata in metadatas:
            sanitized = {"namespace": namespace or "default"}  # Add namespace to metadata
            for key, value in metadata.items():
                # Convert all values to strings (Chroma requirement)
                if isinstance(value, (list, dict)):
                    sanitized[key] = str(value)
                elif value is not None:
                    sanitized[key] = str(value)
            sanitized_metadatas.append(sanitized)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=sanitized_metadatas
        )

        return ids

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        # Chroma's query format
        where = {}

        # Add namespace filter
        if namespace:
            where["namespace"] = {"$eq": namespace}

        # Add custom filters
        if filter:
            for key, value in filter.items():
                where[key] = {"$eq": value}

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where if where else None,
            include=["documents", "metadatas", "distances"]
        )

        # Convert results to standard format
        formatted_results = []
        if results['ids'] and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                # Chroma returns distances, convert to similarity scores
                # Distance is L2, so lower is better
                # Convert to 0-1 similarity score (1 = identical, 0 = very different)
                distance = results['distances'][0][i]
                score = 1.0 / (1.0 + distance)  # Simple conversion

                # Remove namespace from metadata
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                metadata = {k: v for k, v in metadata.items() if k != "namespace"}

                formatted_results.append({
                    "id": results['ids'][0][i],
                    "score": score,
                    "content": results['documents'][0][i],
                    "metadata": metadata
                })

        return formatted_results

    def delete(self, ids: List[str], namespace: Optional[str] = None) -> bool:
        """Delete vectors by IDs"""
        # Note: Chroma delete by IDs doesn't filter by namespace
        # If namespace filtering is critical, would need to search first
        self.collection.delete(ids=ids)
        return True

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
        # Build where clause
        where = {field: {"$eq": value}}
        if namespace:
            where["namespace"] = {"$eq": namespace}

        # First, find matching IDs
        results = self.collection.get(
            where=where,
            include=[]  # Only need IDs
        )

        deleted_ids = results['ids'] if results['ids'] else []

        if deleted_ids:
            self.collection.delete(ids=deleted_ids)

        return {
            "deleted_count": len(deleted_ids),
            "deleted_ids": deleted_ids
        }

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        count = self.collection.count()

        return {
            "name": self.collection_name,
            "total_vectors": count,
            "dimensions": self.dimensions,
            "mode": "http" if self.settings.chroma_host else "embedded"
        }
