"""Document management endpoints"""

from fastapi import APIRouter, HTTPException, Query, Path
from ragbrain.rag.pipeline import get_pipeline
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/documents")
async def list_documents(
    namespace: str | None = Query(None, description="Optional namespace filter"),
    extension: str | None = Query(None, description="Filter by file extension (e.g., 'pdf', 'txt')"),
    orphaned: bool = Query(False, description="Show only chunks without doc_id (legacy data)"),
    use_summaries: bool = Query(True, description="Use fast summary-based listing (set False for legacy scan)")
):
    """
    List all unique documents in the knowledge base.

    Uses document summary records for fast O(1) catalog lookups.
    Falls back to chunk scanning for documents without summaries.

    Returns doc_id, filename, created_at, chunk count, namespace, and headings.

    Optional filters:
    - namespace: Filter by namespace
    - extension: Filter by file extension (e.g., 'pdf', 'txt', 'md')
    - orphaned: If true, show chunks that don't have a doc_id (pre-UUID data)
    - use_summaries: If true (default), use fast summary-based listing
    """
    try:
        pipeline = get_pipeline()
        vectordb = pipeline.vectordb_provider

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Handle orphaned chunks separately (requires full scan)
        if orphaned:
            return await _list_orphaned_chunks(vectordb, namespace)

        # Use summary-based listing (fast path)
        if use_summaries:
            # Build filter for document summaries
            conditions = [
                FieldCondition(key="_type", match=MatchValue(value="document_summary"))
            ]
            if namespace:
                conditions.append(
                    FieldCondition(key="namespace", match=MatchValue(value=namespace))
                )

            scroll_filter = Filter(must=conditions)

            # Normalize extension filter
            ext_filter = None
            if extension:
                ext_filter = extension.lower().lstrip('.')

            # Scroll through summaries (much faster than all chunks)
            documents = []
            offset = None

            while True:
                points, offset = vectordb.client.scroll(
                    collection_name=vectordb.collection_name,
                    scroll_filter=scroll_filter,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                for point in points:
                    payload = point.payload
                    filename = payload.get("filename")

                    # Apply extension filter
                    if ext_filter and filename:
                        if not filename.lower().endswith(f'.{ext_filter}'):
                            continue
                    elif ext_filter and not filename:
                        continue

                    documents.append({
                        "doc_id": payload.get("doc_id"),
                        "filename": filename,
                        "namespace": payload.get("namespace", "default"),
                        "chunk_count": payload.get("chunk_count"),
                        "created_at": payload.get("created_at"),
                        "headings": payload.get("headings", [])
                    })

                if offset is None:
                    break

            # Sort by created_at (newest first)
            documents.sort(key=lambda x: x.get("created_at") or "", reverse=True)

            return {
                "documents": documents,
                "count": len(documents),
                "source": "summaries"
            }

        # Legacy scan-based listing (fallback)
        return await _list_documents_legacy(vectordb, namespace, extension)

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _list_orphaned_chunks(vectordb, namespace: str | None):
    """List chunks without doc_id (legacy data)"""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    scroll_filter = None
    if namespace:
        scroll_filter = Filter(
            must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))]
        )

    orphaned_chunks = {}
    offset = None

    while True:
        points, offset = vectordb.client.scroll(
            collection_name=vectordb.collection_name,
            scroll_filter=scroll_filter,
            limit=1000,
            offset=offset,
            with_payload=["doc_id", "filename", "namespace", "_type"],
            with_vectors=False
        )

        for point in points:
            # Skip summary records
            if point.payload.get("_type") == "document_summary":
                continue

            doc_id = point.payload.get("doc_id")
            if doc_id:
                continue

            filename = point.payload.get("filename")
            key = filename or f"orphan_{point.id}"
            if key not in orphaned_chunks:
                orphaned_chunks[key] = {
                    "doc_id": None,
                    "filename": filename,
                    "namespace": point.payload.get("namespace", "default"),
                    "chunk_count": 0,
                    "point_ids": [],
                    "created_at": None
                }
            orphaned_chunks[key]["chunk_count"] += 1
            orphaned_chunks[key]["point_ids"].append(str(point.id))

        if offset is None:
            break

    return {
        "orphaned_chunks": list(orphaned_chunks.values()),
        "count": len(orphaned_chunks)
    }


async def _list_documents_legacy(vectordb, namespace: str | None, extension: str | None):
    """Legacy document listing by scanning all chunks"""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    scroll_filter = None
    if namespace:
        scroll_filter = Filter(
            must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))]
        )

    ext_filter = None
    if extension:
        ext_filter = extension.lower().lstrip('.')

    docs = {}
    offset = None

    while True:
        points, offset = vectordb.client.scroll(
            collection_name=vectordb.collection_name,
            scroll_filter=scroll_filter,
            limit=1000,
            offset=offset,
            with_payload=["doc_id", "filename", "namespace", "total_chunks", "created_at", "_type"],
            with_vectors=False
        )

        for point in points:
            # Skip summary records
            if point.payload.get("_type") == "document_summary":
                continue

            doc_id = point.payload.get("doc_id")
            filename = point.payload.get("filename")

            if ext_filter and filename:
                if not filename.lower().endswith(f'.{ext_filter}'):
                    continue
            elif ext_filter and not filename:
                continue

            if doc_id and doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "filename": filename,
                    "namespace": point.payload.get("namespace", "default"),
                    "total_chunks": point.payload.get("total_chunks"),
                    "created_at": point.payload.get("created_at")
                }

        if offset is None:
            break

    documents = sorted(
        docs.values(),
        key=lambda x: x.get("created_at") or "",
        reverse=True
    )

    return {
        "documents": documents,
        "count": len(documents),
        "source": "legacy_scan"
    }


@router.get("/documents/discover")
async def discover_documents(
    query: str = Query(..., description="Semantic search query to find relevant documents"),
    namespace: str | None = Query(None, description="Optional namespace filter (supports wildcards like 'mba/*')"),
    top_k: int = Query(10, ge=1, le=50, description="Number of documents to return")
):
    """
    Semantic discovery of documents.

    Uses vector search over document summaries to find documents matching a topic,
    concept, or natural language query. This is useful for:
    - Finding documents about a specific topic ("documents about leadership")
    - Discovering related documents ("notes on valuation methods")
    - Exploring what's in the knowledge base ("chapters on conflict resolution")

    Returns documents ranked by semantic similarity to the query.
    """
    try:
        pipeline = get_pipeline()
        vectordb = pipeline.vectordb_provider

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Generate embedding for query
        query_embedding = pipeline.embedding_provider.embed(query)

        # Build filter for document summaries only
        conditions = [
            FieldCondition(key="_type", match=MatchValue(value="document_summary"))
        ]

        # Handle namespace filter (exact match, wildcard handled after search)
        namespace_prefix = None
        if namespace:
            if namespace.endswith("/*"):
                namespace_prefix = namespace[:-2]  # For post-filtering
            else:
                conditions.append(
                    FieldCondition(key="namespace", match=MatchValue(value=namespace))
                )

        search_filter = Filter(must=conditions)

        # Search with higher limit if using wildcard namespace
        search_limit = top_k * 3 if namespace_prefix else top_k

        # Perform semantic search over summaries
        results = vectordb.client.query_points(
            collection_name=vectordb.collection_name,
            query=query_embedding,
            limit=search_limit,
            query_filter=search_filter
        ).points

        # Post-filter for namespace wildcard
        if namespace_prefix:
            results = [
                hit for hit in results
                if hit.payload.get("namespace", "").startswith(namespace_prefix + "/") or
                   hit.payload.get("namespace", "") == namespace_prefix
            ][:top_k]

        # Format response
        documents = []
        for hit in results:
            payload = hit.payload
            documents.append({
                "doc_id": payload.get("doc_id"),
                "filename": payload.get("filename"),
                "namespace": payload.get("namespace", "default"),
                "headings": payload.get("headings", []),
                "chunk_count": payload.get("chunk_count"),
                "created_at": payload.get("created_at"),
                "score": hit.score,
                "summary_preview": payload.get("text", "")[:300]  # First 300 chars of summary
            })

        return {
            "query": query,
            "documents": documents,
            "count": len(documents)
        }

    except Exception as e:
        logger.error(f"Failed to discover documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# NOTE: Static routes must come before parameterized routes
@router.get("/documents/chunks")
async def get_chunks_by_ids(
    point_ids: str = Query(..., description="Comma-separated list of point IDs")
):
    """
    Retrieve chunk text content by point IDs.

    Useful for inspecting orphaned chunks that don't have doc_id.
    Pass the point_ids from the orphaned chunks response.

    Example: /api/documents/chunks?point_ids=abc123,def456,ghi789
    """
    ids = [pid.strip() for pid in point_ids.split(",") if pid.strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="point_ids cannot be empty")

    try:
        pipeline = get_pipeline()
        vectordb = pipeline.vectordb_provider

        # Retrieve points by ID
        points = vectordb.client.retrieve(
            collection_name=vectordb.collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=False
        )

        chunks = []
        for point in points:
            chunks.append({
                "point_id": str(point.id),
                "text": point.payload.get("text", ""),
                "filename": point.payload.get("filename"),
                "namespace": point.payload.get("namespace", "default"),
                "chunk_index": point.payload.get("chunk_index"),
                "doc_id": point.payload.get("doc_id"),
                "created_at": point.payload.get("created_at")
            })

        # Sort by chunk_index if available
        chunks.sort(key=lambda x: x.get("chunk_index") or 0)

        return {
            "chunks": chunks,
            "count": len(chunks),
            "reconstructed_text": "\n\n".join(c["text"] for c in chunks if c["text"])
        }
    except Exception as e:
        logger.error(f"Failed to get chunks by IDs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/orphaned")
async def delete_orphaned_chunks(
    filename: str | None = Query(None, description="Delete orphaned chunks for specific filename"),
    all_orphaned: bool = Query(False, description="Delete ALL orphaned chunks (use with caution)")
):
    """
    Delete orphaned chunks (chunks without doc_id).

    Either specify a filename or set all_orphaned=true to delete all.
    """
    if not filename and not all_orphaned:
        raise HTTPException(
            status_code=400,
            detail="Must specify either 'filename' or 'all_orphaned=true'"
        )

    try:
        pipeline = get_pipeline()
        vectordb = pipeline.vectordb_provider

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Find orphaned chunks by scrolling and filtering in Python
        # (Qdrant's IsNull condition isn't reliable for missing fields)
        orphan_ids = []
        offset = None

        while True:
            points, offset = vectordb.client.scroll(
                collection_name=vectordb.collection_name,
                limit=1000,
                offset=offset,
                with_payload=["doc_id", "filename"],
                with_vectors=False
            )

            for point in points:
                doc_id = point.payload.get("doc_id")
                point_filename = point.payload.get("filename")

                # Check if orphaned (no doc_id)
                if doc_id is None:
                    # If filtering by filename, check it matches
                    if filename and point_filename != filename:
                        continue
                    orphan_ids.append(point.id)

            if offset is None:
                break

        if not orphan_ids:
            return {
                "success": True,
                "chunks_deleted": 0,
                "message": "No orphaned chunks found"
            }

        # Delete
        vectordb.client.delete(
            collection_name=vectordb.collection_name,
            points_selector=orphan_ids
        )

        logger.info(f"Deleted {len(orphan_ids)} orphaned chunks" + (f" for filename: {filename}" if filename else ""))

        return {
            "success": True,
            "chunks_deleted": len(orphan_ids),
            "filename": filename
        }
    except Exception as e:
        logger.error(f"Failed to delete orphaned chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/id/{doc_id}")
async def get_document(
    doc_id: str = Path(..., description="Document ID (UUID)")
):
    """
    Get a document by doc_id, including reconstructed text from all chunks.

    Chunks are returned in order and can be joined to reconstruct the original text.
    """
    try:
        pipeline = get_pipeline()
        vectordb = pipeline.vectordb_provider

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Find all chunks for this doc_id
        points, _ = vectordb.client.scroll(
            collection_name=vectordb.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            limit=10000,
            with_payload=True,
            with_vectors=False
        )

        if not points:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        # Sort chunks by index
        chunks = sorted(
            [(p.payload.get("chunk_index", 0), p.payload.get("text", ""), p.payload) for p in points],
            key=lambda x: x[0]
        )

        # Extract metadata from first chunk
        first_payload = chunks[0][2]

        return {
            "doc_id": doc_id,
            "filename": first_payload.get("filename"),
            "namespace": first_payload.get("namespace", "default"),
            "created_at": first_payload.get("created_at"),
            "total_chunks": len(chunks),
            "chunks": [
                {"index": idx, "text": text}
                for idx, text, _ in chunks
            ],
            "reconstructed_text": "\n\n".join(text for _, text, _ in chunks)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/id/{doc_id}")
async def delete_document_by_id(
    doc_id: str = Path(..., description="Document ID (UUID) to delete")
):
    """
    Delete all chunks and summary record associated with a doc_id.
    """
    try:
        pipeline = get_pipeline()
        vectordb = pipeline.vectordb_provider

        # Delete chunks by doc_id
        result = vectordb.delete_by_metadata(
            field="doc_id",
            value=doc_id
        )

        # Also explicitly delete the summary record by its ID
        summary_id = f"summary_{doc_id}"
        try:
            vectordb.delete(ids=[summary_id])
        except Exception:
            pass  # Summary may not exist for older documents

        if result["deleted"] == 0:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        logger.info(f"Deleted {result['deleted']} chunks + summary for doc_id: {doc_id}")

        return {
            "success": True,
            "doc_id": doc_id,
            "chunks_deleted": result["deleted"]
        }
    except HTTPException:
        raise
    except NotImplementedError:
        raise HTTPException(
            status_code=501,
            detail="delete_by_metadata not implemented for current vector DB provider"
        )
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents")
async def delete_document_by_filename(
    filename: str = Query(..., description="Filename to delete"),
    namespace: str | None = Query(None, description="Optional namespace filter")
):
    """
    Delete all chunks associated with a filename.

    This allows re-ingesting a file without duplicates.
    """
    try:
        pipeline = get_pipeline()
        result = pipeline.vectordb_provider.delete_by_metadata(
            field="filename",
            value=filename,
            namespace=namespace
        )

        logger.info(f"Deleted {result['deleted']} chunks for filename: {filename}")

        return {
            "success": True,
            "filename": filename,
            "namespace": namespace,
            "chunks_deleted": result["deleted"]
        }
    except NotImplementedError:
        raise HTTPException(
            status_code=501,
            detail="delete_by_metadata not implemented for current vector DB provider"
        )
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/migrate-summaries")
async def migrate_document_summaries(
    namespace: str | None = Query(None, description="Only migrate documents in this namespace"),
    dry_run: bool = Query(False, description="Show what would be migrated without making changes")
):
    """
    Create document summary records for existing documents that don't have them.

    This migration scans all chunks, groups them by doc_id, and creates
    summary records for documents that don't already have one.

    The summary records enable fast document listing and semantic discovery.
    """
    from collections import defaultdict
    from datetime import datetime, timezone

    try:
        pipeline = get_pipeline()
        vectordb = pipeline.vectordb_provider

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # First, collect all existing summary doc_ids
        existing_summaries = set()
        offset = None

        while True:
            points, offset = vectordb.client.scroll(
                collection_name=vectordb.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="_type", match=MatchValue(value="document_summary"))]
                ),
                limit=1000,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False
            )

            for point in points:
                doc_id = point.payload.get("doc_id")
                if doc_id:
                    existing_summaries.add(doc_id)

            if offset is None:
                break

        # Now scan all chunks and group by doc_id
        documents = defaultdict(lambda: {
            "chunks": [],
            "filename": None,
            "namespace": None,
            "created_at": None,
            "headings": []
        })

        scroll_filter = None
        if namespace:
            scroll_filter = Filter(
                must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))]
            )

        offset = None

        while True:
            points, offset = vectordb.client.scroll(
                collection_name=vectordb.collection_name,
                scroll_filter=scroll_filter,
                limit=1000,
                offset=offset,
                with_payload=["doc_id", "filename", "namespace", "text", "created_at", "headings", "_type"],
                with_vectors=False
            )

            for point in points:
                if point.payload.get("_type") == "document_summary":
                    continue

                doc_id = point.payload.get("doc_id")
                if not doc_id or doc_id in existing_summaries:
                    continue

                doc = documents[doc_id]
                doc["chunks"].append(point.payload.get("text", ""))
                doc["filename"] = doc["filename"] or point.payload.get("filename")
                doc["namespace"] = doc["namespace"] or point.payload.get("namespace", "default")
                doc["created_at"] = doc["created_at"] or point.payload.get("created_at")

                chunk_headings = point.payload.get("headings", [])
                if chunk_headings:
                    for h in chunk_headings:
                        if h and h not in doc["headings"]:
                            doc["headings"].append(h)

            if offset is None:
                break

        if not documents:
            return {
                "success": True,
                "message": "No migration needed - all documents have summaries",
                "existing_summaries": len(existing_summaries),
                "migrated": 0
            }

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "existing_summaries": len(existing_summaries),
                "would_migrate": len(documents),
                "documents": [
                    {"doc_id": doc_id, "filename": doc["filename"], "chunks": len(doc["chunks"])}
                    for doc_id, doc in list(documents.items())[:50]
                ]
            }

        # Create summaries
        created = 0
        errors = []

        for doc_id, doc in documents.items():
            try:
                summary_parts = [
                    f"Document: {doc['filename'] or 'Unknown'}",
                    f"Namespace: {doc['namespace']}"
                ]

                if doc["headings"]:
                    summary_parts.append(f"Headings: {', '.join(doc['headings'][:20])}")

                content_preview = ""
                char_count = 0
                for chunk in doc["chunks"]:
                    remaining = 1500 - char_count
                    if remaining <= 0:
                        break
                    content_preview += chunk[:remaining] + " "
                    char_count += len(chunk[:remaining])

                if content_preview.strip():
                    summary_parts.append("")
                    summary_parts.append(content_preview.strip())

                summary_text = "\n".join(summary_parts)
                summary_embedding = pipeline.embedding_provider.embed(summary_text)

                import uuid
                summary_id = str(uuid.uuid4())
                summary_metadata = {
                    "_type": "document_summary",
                    "doc_id": doc_id,
                    "filename": doc["filename"],
                    "namespace": doc["namespace"],
                    "headings": doc["headings"][:50],
                    "chunk_count": len(doc["chunks"]),
                    "created_at": doc["created_at"] or datetime.now(timezone.utc).isoformat(),
                }

                vectordb.insert(
                    vectors=[summary_embedding],
                    texts=[summary_text],
                    metadatas=[summary_metadata],
                    ids=[summary_id],
                    namespace=doc["namespace"]
                )

                created += 1

            except Exception as e:
                errors.append({"doc_id": doc_id, "error": str(e)})
                logger.error(f"Failed to create summary for {doc_id}: {e}")

        logger.info(f"Migration complete: created {created} summaries, {len(errors)} errors")

        return {
            "success": len(errors) == 0,
            "existing_summaries": len(existing_summaries),
            "migrated": created,
            "errors": errors[:10] if errors else []
        }

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
