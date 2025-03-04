"""
Endpoints for image prediction.
"""
import time
import numpy as np
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any, Optional

from core.config import TEMPLATES_DIR
from core.logging import logger
from services.trt_service import trt_services
from services.image_service import image_service
from utils.visualization import visualization_utils
from models.requests import JsonPredictionRequest
from models.responses import PredictionResponse, BatchPredictionResponse, BatchResultItem

router = APIRouter(
    prefix="",
    tags=["prediction"],
)

# Initialize templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with image upload form."""
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@router.post("/", response_class=HTMLResponse)
async def process_image(request: Request, file: UploadFile = File(...)):
    """Process uploaded image and render results."""
    for service in trt_services.values():
        if not service.is_initialized():
            return templates.TemplateResponse(
                "index.html", 
                {
                    "request": request, 
                    "error": "TensorRT server not initialized"
                }
            )
    
    try:
        # Validate file extension
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            return templates.TemplateResponse(
                "index.html", 
                {
                    "request": request, 
                    "error": "Unsupported file format. Only PNG, JPG, JPEG, BMP are supported."
                }
            )
        
        # Read file
        content = await file.read()
        logger.info(f"Image uploaded: {file.filename}, size: {len(content)} bytes")
        
        # Get input binding details
        trt_service = trt_services.get("gauge_detect")
        input_shape = trt_service.get_first_input_shape()
        logger.info(f"Model input shape: {input_shape}")
        
        # Preprocess image
        logger.info("Starting image preprocessing")
        preprocessed_data, preprocess_info = image_service.preprocess_image(content, input_shape)
        logger.info(f"Preprocessed data shape: {preprocessed_data.shape}, type: {preprocessed_data.dtype}")
        
        # Run inference
        logger.info("Starting inference")
        start_time = time.time()
        outputs, inference_time = trt_service.predict(preprocessed_data)
        total_time = (time.time() - start_time) * 1000  # ms
        logger.info(f"Inference complete: time={inference_time:.2f}ms, outputs={len(outputs)}")
        
        # Log output details for debugging
        if outputs:
            logger.info(f"Output keys: {list(outputs.keys())}")
            for k, v in outputs.items():
                if isinstance(v, np.ndarray):
                    logger.info(f"  {k}: shape={v.shape}, type={v.dtype}")
                    
                    # Log detection details
                    if k == 'det_boxes' and 'num_dets' in outputs and 'det_scores' in outputs:
                        num_dets = outputs['num_dets'][0][0]
                        logger.info(f"Number of detections: {num_dets}")
                        for i in range(min(int(num_dets), 10)):  # Log up to 10
                            box = outputs['det_boxes'][0][i]
                            score = outputs['det_scores'][0][i]
                            if 'det_classes' in outputs:
                                class_id = outputs['det_classes'][0][i]
                                logger.info(f"  Object {i}: class={class_id}, score={score}, box={box}")
                            else:
                                logger.info(f"  Object {i}: score={score}, box={box}")
        
        # Encode original image to base64
        logger.info("Starting original image encoding")
        original_base64 = image_service.image_to_base64(content)
        
        # Visualize results
        logger.info("Starting result visualization")
        base64_img = visualization_utils.visualize_detection(content, outputs, preprocess_info)
        logger.info("Result visualization complete")
        
        # Render result page
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "original_image": original_base64,
                "result_image": base64_img,
                "result": image_service.postprocess_output(outputs),
                "inference_time": inference_time,
                "total_time": total_time,
                "filename": file.filename
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "error": f"Error processing image: {str(e)}"
            }
        )

@router.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """API endpoint for single image prediction."""
    for service in trt_services.values():
        if not service.is_initialized():
            raise HTTPException(status_code=503, detail="TensorRT server not initialized")
    
    try:
        # Validate file extension
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            raise HTTPException(status_code=400, detail="Unsupported file format. Only PNG, JPG, JPEG, BMP are supported.")
        
        # Read file
        content = await file.read()
        
        # Get input shape
        trt_service = trt_services.get("gauge_detect")
        input_shape = trt_service.get_first_input_shape()
        
        # Preprocess image
        preprocessed_data, _ = image_service.preprocess_image(content, input_shape)
        
        # Run inference
        start_time = time.time()
        outputs, inference_time = trt_service.predict(preprocessed_data)
        total_time = (time.time() - start_time) * 1000  # ms
        
        # Postprocess results
        results = image_service.postprocess_output(outputs)
        
        return {
            "success": True,
            "inference_time_ms": inference_time,
            "total_time_ms": total_time,
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

# @router.post("/predict/batch", response_model=BatchPredictionResponse)
# async def predict_batch(files: List[UploadFile] = File(...)):
#     """API endpoint for batch image prediction."""
#     if not trt_service.is_initialized():
#         raise HTTPException(status_code=503, detail="TensorRT server not initialized")
    
#     if len(files) == 0:
#         raise HTTPException(status_code=400, detail="At least one file is required")
    
#     try:
#         results = []
#         total_inference_time = 0
        
#         # Get input shape
#         input_shape = trt_service.get_first_input_shape()
        
#         # Process each file
#         for file in files:
#             # Validate file extension
#             if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
#                 logger.warning(f"Unsupported file format: {file.filename}")
#                 results.append({
#                     "filename": file.filename,
#                     "error": "Unsupported file format"
#                 })
#                 continue
            
#             try:
#                 # Read file
#                 content = await file.read()
                
#                 # Preprocess image
#                 preprocessed_data, _ = image_service.preprocess_image(content, input_shape)
                
#                 # Run inference
#                 outputs, inference_time = trt_service.predict(preprocessed_data)
#                 total_inference_time += inference_time
                
#                 # Postprocess results
#                 processed_results = image_service.postprocess_output(outputs)
                
#                 results.append({
#                     "filename": file.filename,
#                     "inference_time_ms": inference_time,
#                     "results": processed_results
#                 })
                
#             except Exception as e:
#                 logger.error(f"Error processing file {file.filename}: {e}")
#                 results.append({
#                     "filename": file.filename,
#                     "error": str(e)
#                 })
        
#         return {
#             "success": True,
#             "total_files": len(files),
#             "processed_files": len(results),
#             "total_inference_time_ms": total_inference_time,
#             "results": results
#         }
    
#     except Exception as e:
#         logger.error(f"Error during batch inference: {e}")
#         raise HTTPException(status_code=500, detail=f"Error during batch inference: {str(e)}")

# @router.post("/predict/json", response_model=PredictionResponse)
# async def predict_json(data: JsonPredictionRequest):
#     """API endpoint for JSON-based prediction."""
#     if not trt_service.is_initialized():
#         raise HTTPException(status_code=503, detail="TensorRT server not initialized")
    
#     try:
#         # Get first input binding name
#         default_input_name = trt_service.get_first_input_name()
        
#         # Prepare input data
#         inputs = data.inputs
#         input_name = list(inputs.keys())[0] if inputs else default_input_name
#         input_data = inputs.get(input_name)
        
#         if not input_data:
#             raise HTTPException(status_code=400, detail=f"No data for input '{input_name}'")
        
#         # Convert to NumPy array
#         try:
#             np_data = np.array(input_data, dtype=np.float32)
#         except Exception as e:
#             raise HTTPException(status_code=400, detail=f"Error converting input data to NumPy array: {str(e)}")
        
#         # Match input shape
#         expected_shape = trt_service.get_first_input_shape()
#         try:
#             if np_data.shape != expected_shape:
#                 logger.warning(f"Input shape mismatch. Expected: {expected_shape}, Got: {np_data.shape}. Attempting resize...")
#                 np_data = np.resize(np_data, expected_shape)
#         except Exception as e:
#             raise HTTPException(status_code=400, detail=f"Error resizing input data: {str(e)}")
        
#         # Run inference
#         start_time = time.time()
#         outputs, inference_time = trt_service.predict(np_data)
#         total_time = (time.time() - start_time) * 1000  # ms
        
#         # Postprocess results
#         results = image_service.postprocess_output(outputs)
        
#         return {
#             "success": True,
#             "inference_time_ms": inference_time,
#             "total_time_ms": total_time,
#             "results": results
#         }
    
#     except HTTPException:
#         raise
    
#     except Exception as e:
#         logger.error(f"Error during JSON inference: {e}")
#         raise HTTPException(status_code=500, detail=f"Error during JSON inference: {str(e)}")