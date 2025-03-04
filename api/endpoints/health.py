"""
Health check and diagnostic endpoints.
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

import json

from services.trt_service import trt_services
from models.responses import HealthResponse, ModelInfo
from core.logging import logger
from core.config import MODEL_PATHs, USE_CUDA_STREAM

router = APIRouter(
    prefix="",
    tags=["health"],
)

@router.get("/health", response_model=HealthResponse)
async def health_check() -> Dict[str, Any]:
    """Health check endpoint to verify service status."""
    for service in trt_services.values() : 
        if service.is_initialized():
            return {"status": "healthy"}
        else:
            return {"status": "unhealthy", "message": "TensorRT server not initialized"}

@router.get("/model/info")
async def model_info() -> Dict[str, Any]:
    """Get model information."""
    for service in trt_services.values() :
        if not service.is_initialized() :
            raise HTTPException(status_code=503, detail="TensorRT server not initialized")
    
    try:
        # Get input and output details
        inputs = ""
        outputs = ""
        for service in trt_services.values() :
            inputs += json.dumps(service.get_input_details())
            outputs += json.dumps(service.get_output_details())
        
        return {
            "model_path": MODEL_PATHs,
            "inputs": inputs,
            "outputs": outputs,
            "use_cuda_stream": USE_CUDA_STREAM
        }
    
    except Exception as e:
        logger.error(f"Error retrieving model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")