"""Qdrant vector database provider"""

from typing import List, Dict, Any, Optional
from ragbrain.providers.base import VectorDBProvider
from ragbrain.config import Settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid


class QdrantVectorDBProvider(VectorDBProvider):
    """Qdrant vector database provider"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.collection_name = settings.qdrant_collection
        self.dimensions = settings.embedding_dimension

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimensions,
                    distance=Distance.COSINE
                )
            )

    def insert(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None
    ) -> List[str]:
        """Insert vectors into Qdrant"""
        if not ids:
            ids = [str(uuid.uuid4()) for _ in vectors]

        if not metadatas:
            metadatas = [{} for _ in vectors]

        points = [
            PointStruct(
                id=id_,
                vector=vector,
                payload={
                    "text": text,
                    "namespace": namespace or "default",
                    **metadata
                }
            )
            for id_, vector, text, metadata in zip(ids, vectors, texts, metadatas)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return ids

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors

        Supports namespace wildcards:
        - "books/*" matches "books/fiction", "books/nonfiction", etc.
        - "books" matches exactly "books"
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Build filter with namespace
        conditions = []

        # Exclude document summaries from search results
        # Use must_not to filter out records where _type = "document_summary"
        must_not_conditions = [
            FieldCondition(key="_type", match=MatchValue(value="document_summary"))
        ]

        # Add namespace filter (supports wildcards like "books/*")
        # By default, searching a namespace also includes all sub-namespaces
        # Use exact:namespace for exact match only
        namespace_prefix = None
        if namespace:
            if namespace.startswith("exact:"):
                # Exact match only (no sub-namespaces)
                exact_ns = namespace[6:]  # Remove "exact:" prefix
                conditions.append(
                    FieldCondition(key="namespace", match=MatchValue(value=exact_ns))
                )
            elif namespace.endswith("/*"):
                # Explicit wildcard - post-filter after search
                namespace_prefix = namespace[:-2]
            else:
                # Default: include sub-namespaces via post-filtering
                namespace_prefix = namespace

        # Add custom filters
        if filter:
            for key, value in filter.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        # Build filter: must conditions AND must_not conditions
        search_filter = Filter(
            must=conditions if conditions else None,
            must_not=must_not_conditions
        )

        # Increase limit if using namespace prefix to allow for post-filtering
        search_limit = top_k * 3 if namespace_prefix else top_k

        # Use query_points for newer qdrant-client versions
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=search_limit,
            query_filter=search_filter
        ).points

        # Post-filter for namespace prefix (includes sub-namespaces)
        if namespace_prefix:
            filtered_results = [
                hit for hit in results
                if hit.payload.get("namespace", "").startswith(namespace_prefix + "/") or
                   hit.payload.get("namespace", "") == namespace_prefix
            ]
            results = filtered_results[:top_k]

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "content": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k not in ("text", "namespace")},
                "namespace": hit.payload.get("namespace", "default")
            }
            for hit in results
        ]

    def delete(self, ids: List[str], namespace: Optional[str] = None) -> bool:
        """Delete vectors by IDs"""
        # Note: Qdrant doesn't support namespace in delete by IDs
        # If namespace filtering is critical, would need to search first then delete
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids
        )
        return True

    def delete_by_metadata(self, field: str, value: str, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Delete vectors by metadata field value (e.g., filename)"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        conditions = [FieldCondition(key=field, match=MatchValue(value=value))]
        if namespace:
            conditions.append(FieldCondition(key="namespace", match=MatchValue(value=namespace)))

        # First, find all matching points to get count
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=conditions),
            limit=10000,  # Reasonable upper bound
            with_payload=False,
            with_vectors=False
        )

        if not points:
            return {"deleted": 0, "ids": []}

        ids = [p.id for p in points]

        # Delete the points
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids
        )

        return {"deleted": len(ids), "ids": ids}

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": getattr(info, 'vectors_count', None),
            "points_count": getattr(info, 'points_count', None),
            "status": getattr(info, 'status', None)
        }
