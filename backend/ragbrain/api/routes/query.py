"""Query endpoint for searching knowledge base"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ragbrain.rag.pipeline import get_pipeline
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str
    top_k: int = 20
    synthesize: bool = True  # If False, return search results without LLM
    namespace: str | None = None
    rerank: bool = True  # Rerank results for better relevance and deduplication


@router.post("/query")
async def query_knowledge(request: QueryRequest):
    """
    Query your knowledge base with natural language

    - synthesize=True: Returns AI-synthesized answer from relevant chunks
    - synthesize=False: Returns raw search results (faster, no LLM cost)
    - namespace: Optional namespace to search within
    """
    try:
        pipeline = get_pipeline()

        result = pipeline.query(
            question=request.query,
            top_k=request.top_k,
            synthesize=request.synthesize,
            namespace=request.namespace,
            rerank=request.rerank
        )

        return result
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
