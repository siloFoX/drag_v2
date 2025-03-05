"""
Endpoints for video streaming and management.
"""
from fastapi import APIRouter, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from typing import Dict, Any, Optional, List, AsyncGenerator

from core.config import TEMPLATES_DIR
from core.logging import logger
from services.stream_service import stream_service
from services.trt_service import trt_services
from services.image_service import image_service
from utils.visualization import visualization_utils
from models.requests import StreamRequest
from models.responses import StreamInfo

router = APIRouter(
    prefix="",
    tags=["streaming"],
)

# Initialize templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@router.get("/video-config", response_class=HTMLResponse)
async def video_config(request: Request):
    """RTSP stream configuration page."""
    streams = stream_service.get_all_streams()
    return templates.TemplateResponse(
        "video_config.html", 
        {
            "request": request,
            "streams": streams
        }
    )

@router.post("/api/streams")
async def add_stream(stream_id: str = Form(...), rtsp_url: str = Form(...)):
    """Add a new RTSP stream.
    
    Args:
        stream_id: Stream identifier
        rtsp_url: RTSP URL
    """
    if not stream_id or not rtsp_url:
        raise HTTPException(status_code=400, detail="Stream ID and RTSP URL are required")
    
    if not rtsp_url.startswith(("rtsp://", "http://", "https://", "file://")):
        raise HTTPException(status_code=400, detail="Invalid URL format")
    
    success = stream_service.add_stream(stream_id, rtsp_url)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to add stream")
    
    return {"status": "success", "message": f"Stream '{stream_id}' added"}

@router.delete("/api/streams/{stream_id}")
async def remove_stream(stream_id: str):
    """Remove an RTSP stream.
    
    Args:
        stream_id: Stream identifier
    """
    success = stream_service.remove_stream(stream_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Stream '{stream_id}' not found")
    
    return {"status": "success", "message": f"Stream '{stream_id}' removed"}

@router.get("/api/streams")
async def get_streams():
    """Get status of all streams."""
    streams = stream_service.get_all_streams()
    return streams

@router.get("/video", response_class=HTMLResponse)
async def video_page(request: Request, stream_id: Optional[str] = None):
    """Video stream viewing page.
    
    Args:
        stream_id: Stream identifier (if None, show all streams)
    """
    streams = stream_service.get_all_streams()
    
    # Check if specified stream exists
    if stream_id and not stream_service.stream_exists(stream_id):
        return templates.TemplateResponse(
            "video.html", 
            {
                "request": request,
                "error": f"Stream '{stream_id}' not found",
                "streams": streams,
                "selected_stream": None
            }
        )
    
    return templates.TemplateResponse(
        "video.html", 
        {
            "request": request,
            "streams": streams,
            "selected_stream": stream_id
        }
    )

@router.get("/api/video-feed/{stream_id}")
async def video_feed(stream_id: str):
    """Video feed endpoint (MJPEG stream).
    
    Args:
        stream_id: Stream identifier
    """
    if not stream_service.stream_exists(stream_id):
        raise HTTPException(status_code=404, detail=f"Stream '{stream_id}' not found")
    
    # 중요: 코루틴이 아닌 제너레이터를 직접 반환
    async def frame_generator():
        async for frame in stream_service.generate_frames(stream_id, process_frame):
            yield frame
    
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

async def process_frame(frame):
    """Process a single frame with the object detection model."""
    trt_service = trt_services.get("gauge_detect")
    if frame is None or not trt_service.is_initialized():
        return frame
    
    try:
        # Get input shape
        input_shape = trt_service.get_first_input_shape()
        
        # Preprocess frame
        preprocessed_data, preprocess_info = image_service.preprocess_image(frame, input_shape)
        
        # Run inference
        outputs, inference_time = trt_service.predict(preprocessed_data)
        
        # Visualize results
        processed_frame = visualization_utils.visualize_frame(frame, outputs, preprocess_info)
        
        # Add inference time display
        import cv2
        cv2.putText(
            processed_frame,
            f"Inference: {inference_time:.1f} ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        return processed_frame
    
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return frame