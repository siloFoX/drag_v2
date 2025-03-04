"""
Pydantic models for request validation.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator

class PredictionOptions(BaseModel):
    """Options for prediction requests."""
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Confidence threshold for detections")

class JsonPredictionRequest(BaseModel):
    """Request model for JSON-based prediction."""
    inputs: Dict[str, List[float]]
    options: Optional[PredictionOptions] = None

class StreamRequest(BaseModel):
    """Request model for adding a new stream."""
    stream_id: str
    rtsp_url: str
    
    @validator('rtsp_url')
    def validate_url(cls, v):
        """Validate that the URL has an acceptable protocol."""
        if not v.startswith(("rtsp://", "http://", "https://", "file://")):
            raise ValueError("URL must start with rtsp://, http://, https://, or file://")
        return v