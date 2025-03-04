"""
RTSP stream management service.
"""
import time
import cv2
import asyncio
import numpy as np
from typing import Dict, Optional, List, AsyncGenerator, Callable, Any

from core.logging import logger
from module.rtsp_stream_processor import RTSPStreamManager

class FPSLimiter:
    """Helper class for limiting FPS."""
    
    def __init__(self, max_fps=60):
        """Initialize FPS limiter.
        
        Args:
            max_fps: Maximum frames per second
        """
        self.max_fps = max_fps
        self.frame_time = 1.0 / max_fps
        self.last_frame_time = 0
    
    def should_process(self) -> bool:
        """Determine whether to process a frame at the current time.
        
        Returns:
            bool: True if should process, False otherwise
        """
        current_time = time.time()
        time_elapsed = current_time - self.last_frame_time
        
        if time_elapsed >= self.frame_time:
            self.last_frame_time = current_time
            return True
        
        return False

class StreamService:
    """Service for managing video streams."""
    
    def __init__(self):
        """Initialize the stream service."""
        self.stream_manager = RTSPStreamManager()
    
    def cleanup(self):
        """Clean up all stream resources."""
        if self.stream_manager:
            self.stream_manager.cleanup()
    
    def add_stream(self, stream_id: str, rtsp_url: str) -> bool:
        """Add a new stream.
        
        Args:
            stream_id: Unique identifier for the stream
            rtsp_url: RTSP URL to connect to
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.stream_manager.add_stream(stream_id, rtsp_url)
    
    def remove_stream(self, stream_id: str) -> bool:
        """Remove a stream.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.stream_manager.remove_stream(stream_id)
    
    def get_stream(self, stream_id: str):
        """Get a specific stream processor.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            Stream processor object or None if not found
        """
        return self.stream_manager.get_stream(stream_id)
    
    def get_all_streams(self) -> Dict[str, Dict]:
        """Get all stream information.
        
        Returns:
            Dict mapping stream IDs to stream info dictionaries
        """
        return self.stream_manager.get_all_streams()
    
    def stream_exists(self, stream_id: str) -> bool:
        """Check if a stream exists.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            bool: True if exists, False otherwise
        """
        return stream_id in self.stream_manager.streams
    
    async def generate_frames(self, stream_id: str, process_func: Optional[Callable] = None) -> AsyncGenerator[bytes, None]:
        """Frame generator for MJPEG streaming.
        
        Args:
            stream_id: Stream identifier
            process_func: Optional function to process frames
            
        Yields:
            MJPEG frame chunks
        """
        try:
            processor = self.get_stream(stream_id)
            if not processor:
                logger.error(f"Stream '{stream_id}' not found")
                return
            
            fps_limiter = FPSLimiter(max_fps=15)  # Limit to max 15 FPS
            
            while True:
                # FPS limiting
                if not fps_limiter.should_process():
                    await asyncio.sleep(0.01)  # Short delay
                    continue
                
                # Get frame
                frame = processor.get_frame()
                if frame is None:
                    await asyncio.sleep(0.1)  # Wait if no frame
                    continue
                
                # Process frame if needed
                if process_func and callable(process_func):
                    try:
                        # 중요: process_func이 코루틴인 경우 await로 실행
                        if asyncio.iscoroutinefunction(process_func):
                            frame = await process_func(frame)
                        else:
                            frame = process_func(frame)
                    except Exception as e:
                        logger.error(f"Error processing frame: {e}")
                
                # JPEG encoding
                success, encoded_image = cv2.imencode('.jpg', frame)
                if not success:
                    logger.error("Failed to encode image")
                    await asyncio.sleep(0.1)
                    continue
                
                # Yield frame in MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       encoded_image.tobytes() + 
                       b'\r\n')
                
                # Short delay to reduce CPU usage
                await asyncio.sleep(0.01)
        
        except Exception as e:
            logger.error(f"Error generating frames: {e}")

# Create a singleton instance
stream_service = StreamService()