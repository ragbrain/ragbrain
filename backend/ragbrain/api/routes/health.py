"""Health check endpoint"""

from fastapi import APIRouter
from ragbrain.config import settings
from ragbrain.rag.pipeline import get_pipeline
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Try to initialize pipeline and providers
        pipeline = get_pipeline()
        provider_info = pipeline.get_providers_info()

        return {
            "status": "healthy",
            "providers": provider_info,
            "vectordb_provider": settings.vectordb_provider
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        # Don't expose detailed error messages to clients
        return {
            "status": "unhealthy",
            "error": "Service initialization failed"
        }
