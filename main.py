"""
Main application entry point.
"""
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from core.config import STATIC_DIR, HOST, PORT
from core.logging import logger
from api.router import api_router
from services.trt_service import lifespan_handler

def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="TensorRT Inference Server",
        description="TensorRT model inference API with memory pooling",
        version="1.0.0",
        lifespan=lifespan_handler  # 통합된 라이프사이클 핸들러 사용
    )
    
    # Mount static files
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    
    # Include API router
    app.include_router(api_router)
    
    return app

# Create application instance
app = create_application()

if __name__ == "__main__":
    logger.info(f"Starting TensorRT inference server at {HOST}:{PORT}...")
    
    # Start server
    uvicorn.run(app, host=HOST, port=PORT)