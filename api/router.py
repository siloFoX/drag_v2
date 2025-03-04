"""
Main API router that combines all endpoint routers.
"""
from fastapi import APIRouter

from api.endpoints import health, prediction, streaming

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router)
api_router.include_router(prediction.router)
api_router.include_router(streaming.router)