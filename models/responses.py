"""
Pydantic models for response standardization.
"""
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

class DetectionResult(BaseModel):
    """Model for a single detection result."""
    class_id: int
    score: float
    box: List[float]

class ModelInfo(BaseModel):
    """Model for TensorRT model information."""
    shape: List[int]
    dtype: str

class PredictionResponse(BaseModel):
    """Response model for prediction requests."""
    success: bool
    inference_time_ms: float
    total_time_ms: float
    results: Dict[str, Any]

class BatchResultItem(BaseModel):
    """Model for a single item in a batch prediction response."""
    filename: str
    inference_time_ms: Optional[float] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction requests."""
    success: bool
    total_files: int
    processed_files: int
    total_inference_time_ms: float
    results: List[BatchResultItem]

class StreamInfo(BaseModel):
    """Model for stream information."""
    stream_id: str
    rtsp_url: str
    status: str
    resolution: Optional[str] = None
    fps: Optional[float] = None

class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    message: Optional[str] = None