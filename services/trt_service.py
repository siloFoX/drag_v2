"""
TensorRT inference service with memory pooling.
"""
import numpy as np
import time
import asyncio
from typing import Dict, Tuple, List, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI
import sys

from core.config import MODEL_PATH, USE_CUDA_STREAM
from core.logging import logger
from module.trt_server_with_memory_pool import TRTServerWithMemoryPool

class TensorRTService:
    """Service for managing TensorRT engine and inference."""
    
    def __init__(self):
        """Initialize the TensorRT service."""
        self.trt_server = None
        self.model_path = MODEL_PATH
        self.use_cuda_stream = USE_CUDA_STREAM
        
    def initialize(self) -> bool:
        """Initialize the TensorRT server with the configured model.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            logger.info(f"Initializing TensorRT server... Model: {self.model_path}")
            
            # Initialize TensorRT server
            self.trt_server = TRTServerWithMemoryPool(self.model_path, use_stream=self.use_cuda_stream)
            
            # Validate engine
            if not self.trt_server.check_engine():
                raise RuntimeError("TensorRT engine validation failed")
            
            # Warm up
            self.trt_server.warm_up(num_runs=3)
            
            logger.info("TensorRT server initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing TensorRT server: {e}")
            return False
    
    async def release(self) -> None:
        """Release TensorRT server resources."""
        if self.trt_server:
            logger.info("Releasing TensorRT server resources...")
            
            # 중요: 비동기 함수 호출을 위한 이벤트 루프 확인
            try:
                # 비동기 작업이 완료될 때까지 기다리기
                pending_tasks = [task for task in asyncio.all_tasks() 
                               if not task.done() and task.get_name().startswith('process_stream')]
                
                if pending_tasks:
                    logger.info(f"Waiting for {len(pending_tasks)} pending stream processing tasks to complete...")
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error waiting for pending tasks: {e}")
            
            # 메모리 해제
            self.trt_server.release()
            self.trt_server = None
            logger.info("TensorRT server resources released")
    
    def is_initialized(self) -> bool:
        """Check if the TensorRT server is initialized.
        
        Returns:
            bool: True if initialized, False otherwise.
        """
        return self.trt_server is not None
    
    def predict(self, input_data: np.ndarray) -> Tuple[Dict[str, np.ndarray], float]:
        """Run inference on input data.
        
        Args:
            input_data: Preprocessed input data for the model.
            
        Returns:
            Tuple containing:
                - Dict of output tensor names to numpy arrays
                - Inference time in milliseconds
        
        Raises:
            RuntimeError: If the TensorRT server is not initialized.
        """
        if not self.trt_server:
            raise RuntimeError("TensorRT server not initialized")
            
        return self.trt_server.predict(input_data)
    
    def get_input_details(self) -> Dict[str, Dict[str, Any]]:
        """Get details about model input bindings.
        
        Returns:
            Dict mapping input names to details (shape, dtype).
            
        Raises:
            RuntimeError: If the TensorRT server is not initialized.
        """
        if not self.trt_server:
            raise RuntimeError("TensorRT server not initialized")
            
        shapes = self.trt_server.get_buffer_shapes()
        dtypes = self.trt_server.get_buffer_dtypes()
        
        inputs = {}
        for idx in self.trt_server.input_binding_idxs:
            name = self.trt_server.engine.get_binding_name(idx)
            shape_list = [int(dim) for dim in shapes[name]]
            
            inputs[name] = {
                "shape": shape_list,
                "dtype": str(dtypes[name])
            }
            
        return inputs
    
    def get_output_details(self) -> Dict[str, Dict[str, Any]]:
        """Get details about model output bindings.
        
        Returns:
            Dict mapping output names to details (shape, dtype).
            
        Raises:
            RuntimeError: If the TensorRT server is not initialized.
        """
        if not self.trt_server:
            raise RuntimeError("TensorRT server not initialized")
            
        shapes = self.trt_server.get_buffer_shapes()
        dtypes = self.trt_server.get_buffer_dtypes()
        
        outputs = {}
        for idx in self.trt_server.output_binding_idxs:
            name = self.trt_server.engine.get_binding_name(idx)
            shape_list = [int(dim) for dim in shapes[name]]
            
            outputs[name] = {
                "shape": shape_list,
                "dtype": str(dtypes[name])
            }
            
        return outputs
    
    def get_first_input_shape(self) -> Tuple:
        """Get the shape of the first input binding.
        
        Returns:
            Tuple representing the input shape.
            
        Raises:
            RuntimeError: If the TensorRT server is not initialized.
        """
        if not self.trt_server:
            raise RuntimeError("TensorRT server not initialized")
            
        first_input_idx = self.trt_server.input_binding_idxs[0]
        input_name = self.trt_server.engine.get_binding_name(first_input_idx)
        return self.trt_server.binding_shapes[input_name]
    
    def get_first_input_name(self) -> str:
        """Get the name of the first input binding.
        
        Returns:
            String name of the first input.
            
        Raises:
            RuntimeError: If the TensorRT server is not initialized.
        """
        if not self.trt_server:
            raise RuntimeError("TensorRT server not initialized")
            
        first_input_idx = self.trt_server.input_binding_idxs[0]
        return self.trt_server.engine.get_binding_name(first_input_idx)

# Create a singleton instance
trt_service = TensorRTService()

@asynccontextmanager
async def lifespan_handler(app: FastAPI):
    """Application lifecycle management.
    
    Handles initialization during startup and cleanup during shutdown.
    
    Args:
        app: FastAPI application instance.
    """
    # Startup code (server initialization)
    success = trt_service.initialize()
    if not success:
        logger.error("Failed to initialize TensorRT server, exiting...")
        sys.exit(1)
    
    yield  # Application running
    
    # Shutdown code (resource release)
    await trt_service.release()  # 비동기 해제 함수 호출