"""
Visualization utilities for detection results.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw
import base64
from io import BytesIO
from typing import Dict, Optional, List, Tuple, Any

from core.logging import logger
from core.config import CONFIDENCE_THRESHOLD

class VisualizationUtils:
    """Utilities for visualizing detection results."""
    
    @staticmethod
    def visualize_detection(image_bytes: bytes, raw_outputs: Optional[Dict] = None, 
                           preprocess_info: Optional[Dict] = None, 
                           conf_threshold: float = CONFIDENCE_THRESHOLD) -> str:
        """Visualize YOLO-style object detection results.
        
        Args:
            image_bytes: Image bytes data
            raw_outputs: Raw model outputs (dictionary)
            preprocess_info: Information from preprocessing (for coordinate conversion)
            conf_threshold: Confidence threshold for showing detections
            
        Returns:
            Base64-encoded result image
        """
        try:
            # Load image from byte stream
            img = Image.open(BytesIO(image_bytes))
            width, height = img.size
            logger.info(f"Visualization image size: {width}x{height}")
            
            # Create drawing object
            draw = ImageDraw.Draw(img)
            
            # Return early if no results
            if raw_outputs is None or not isinstance(raw_outputs, dict):
                logger.info("No model output or not in dictionary format")
                buffered = BytesIO()
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.save(buffered, format="JPEG", quality=95)
                return base64.b64encode(buffered.getvalue()).decode()
            
            # Extract data from model output
            outputs = None
            if 'outputs' in raw_outputs and isinstance(raw_outputs['outputs'], dict):
                outputs = raw_outputs['outputs']
            else:
                outputs = raw_outputs
            
            # Check for required keys
            required_keys = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
            if not all(key in outputs for key in required_keys):
                logger.info(f"Missing required keys. Required: {required_keys}, Present: {list(outputs.keys())}")
                buffered = BytesIO()
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.save(buffered, format="JPEG", quality=95)
                return base64.b64encode(buffered.getvalue()).decode()
            
            # Number of detections
            num_detections = int(outputs['num_dets'][0][0])
            logger.info(f"Number of detected objects: {num_detections}")
            
            # Check and set preprocessing info
            if preprocess_info is None:
                preprocess_info = {
                    'original_width': width,
                    'original_height': height,
                    'target_width': width,
                    'target_height': height,
                    'new_width': width,
                    'new_height': height,
                    'ratio': 1.0
                }
                logger.info("No preprocessing info, using defaults")
            
            # Original image size
            original_width = preprocess_info['original_width']
            original_height = preprocess_info['original_height']
            
            # Model input size
            target_width = preprocess_info['target_width']
            target_height = preprocess_info['target_height']
            
            # Actual resized image size
            new_width = preprocess_info['new_width']
            new_height = preprocess_info['new_height']
            
            # Process each detection
            detected_count = 0
            for i in range(num_detections):
                # Confidence score
                score = float(outputs['det_scores'][0][i])
                
                # Process only if above threshold
                if score >= conf_threshold:
                    # Bounding box
                    box = outputs['det_boxes'][0][i]
                    # Class ID
                    class_id = int(outputs['det_classes'][0][i])
                    
                    # Box coordinates
                    x1, y1, x2, y2 = box
                    
                    # Check if coordinates are normalized
                    is_normalized = all(0 <= coord <= 1.0 for coord in [x1, y1, x2, y2])
                    
                    logger.info(f"Original bounding box: {box}, Normalized: {is_normalized}")
                    
                    # Convert coordinates (model input space -> original image space)
                    if is_normalized:
                        # Convert normalized coordinates (0~1) to model input size
                        x1_model = x1 * target_width
                        y1_model = y1 * target_height
                        x2_model = x2 * target_width
                        y2_model = y2 * target_height
                    else:
                        # Use as-is if already in pixel units
                        x1_model, y1_model, x2_model, y2_model = x1, y1, x2, y2
                    
                    # Check and clip to padded area
                    x1_model = min(max(x1_model, 0), new_width)
                    y1_model = min(max(y1_model, 0), new_height)
                    x2_model = min(max(x2_model, 0), new_width)
                    y2_model = min(max(y2_model, 0), new_height)
                    
                    # Convert from model input space to original image space
                    ratio = preprocess_info['ratio']
                    
                    # Remove padding and convert to original size
                    x1_orig = x1_model / ratio
                    y1_orig = y1_model / ratio
                    x2_orig = x2_model / ratio
                    y2_orig = y2_model / ratio
                    
                    # Ensure coordinates don't exceed original image size
                    x1_orig = max(0, min(x1_orig, original_width))
                    y1_orig = max(0, min(y1_orig, original_height))
                    x2_orig = max(0, min(x2_orig, original_width))
                    y2_orig = max(0, min(y2_orig, original_height))
                    
                    # Convert to integers for PIL
                    x1_draw, y1_draw = int(x1_orig), int(y1_orig)
                    x2_draw, y2_draw = int(x2_orig), int(y2_orig)
                    
                    logger.info(f"Converted bounding box: ({x1_draw}, {y1_draw}, {x2_draw}, {y2_draw})")
                    
                    # Skip if bounding box is too small
                    if x2_draw - x1_draw < 5 or y2_draw - y1_draw < 5:
                        logger.info(f"Bounding box too small, skipping: {x2_draw-x1_draw}x{y2_draw-y1_draw}")
                        continue
                    
                    # Bounding box color (different color for each class)
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                    color_idx = class_id % len(colors)
                    color = colors[color_idx]
                    
                    # Draw bounding box
                    draw.rectangle(
                        [x1_draw, y1_draw, x2_draw, y2_draw], 
                        outline=color, 
                        width=3
                    )
                    
                    # Label text
                    label = f"Class {class_id}: {score:.2f}"
                    
                    # Measure text size (PIL version compatibility)
                    try:
                        # PIL 9.0.0 and above
                        font = ImageDraw.getfont()
                        text_bbox = draw.textbbox((0, 0), label, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                    except (AttributeError, TypeError):
                        # For older versions
                        if hasattr(draw, 'textsize'):
                            text_width, text_height = draw.textsize(label)
                        else:
                            text_width, text_height = 100, 15  # Estimated value
                    
                    # Draw label background
                    draw.rectangle(
                        [x1_draw, y1_draw - text_height - 4, x1_draw + text_width, y1_draw], 
                        fill=color
                    )
                    
                    # Draw label text
                    draw.text(
                        [x1_draw, y1_draw - text_height - 2], 
                        label, 
                        fill=(255, 255, 255)  # White text
                    )
                    
                    detected_count += 1
                    
            logger.info(f"Number of visualized objects: {detected_count}")
            
            # Encode result image to base64
            buffered = BytesIO()
            
            # Convert RGBA to RGB if needed
            if img.mode == 'RGBA':
                img = img.convert('RGB')
                
            img.save(buffered, format="JPEG", quality=95)
            base64_img = base64.b64encode(buffered.getvalue()).decode()
            
            return base64_img
            
        except Exception as e:
            logger.error(f"Error during result visualization: {e}", exc_info=True)
            # Return original image on error
            return base64.b64encode(image_bytes).decode()

    @staticmethod
    def visualize_frame(frame: np.ndarray, outputs: Dict, 
                        preprocess_info: Optional[Dict] = None,
                        conf_threshold: float = CONFIDENCE_THRESHOLD) -> np.ndarray:
        """Visualize object detection results on a frame.
        
        Args:
            frame: Input image frame
            outputs: Model output results
            preprocess_info: Information from preprocessing (for coordinate conversion)
            conf_threshold: Confidence threshold for showing detections
            
        Returns:
            Visualized image frame
        """
        try:
            # Return original frame if no results
            if not outputs:
                return frame
            
            # Make a copy of the frame
            vis_frame = frame.copy()
            height, width = vis_frame.shape[:2]
            
            # Check for required keys
            required_keys = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
            if not all(key in outputs for key in required_keys):
                return vis_frame
            
            # Number of detections
            num_detections = int(outputs['num_dets'][0][0])
            
            # Check and set preprocessing info
            if preprocess_info is None:
                preprocess_info = {
                    'original_width': width,
                    'original_height': height,
                    'target_width': width,
                    'target_height': height,
                    'new_width': width,
                    'new_height': height,
                    'ratio': 1.0
                }
            
            # Original image size
            original_width = preprocess_info['original_width']
            original_height = preprocess_info['original_height']
            
            # Model input size
            target_width = preprocess_info['target_width']
            target_height = preprocess_info['target_height']
            
            # Actual resized image size
            new_width = preprocess_info['new_width']
            new_height = preprocess_info['new_height']
            
            # Process each detection
            for i in range(num_detections):
                # Confidence score
                score = float(outputs['det_scores'][0][i])
                
                # Process only if above threshold
                if score >= conf_threshold:
                    # Bounding box
                    box = outputs['det_boxes'][0][i]
                    # Class ID
                    class_id = int(outputs['det_classes'][0][i])
                    
                    # Box coordinates
                    x1, y1, x2, y2 = box
                    
                    # Check if coordinates are normalized
                    is_normalized = all(0 <= coord <= 1.0 for coord in [x1, y1, x2, y2])
                    
                    # Convert coordinates (model input space -> original image space)
                    if is_normalized:
                        # Convert normalized coordinates (0~1) to model input size
                        x1_model = x1 * target_width
                        y1_model = y1 * target_height
                        x2_model = x2 * target_width
                        y2_model = y2 * target_height
                    else:
                        # Use as-is if already in pixel units
                        x1_model, y1_model, x2_model, y2_model = x1, y1, x2, y2
                    
                    # Check and clip to padded area
                    x1_model = min(max(x1_model, 0), new_width)
                    y1_model = min(max(y1_model, 0), new_height)
                    x2_model = min(max(x2_model, 0), new_width)
                    y2_model = min(max(y2_model, 0), new_height)
                    
                    # Convert from model input space to original image space
                    ratio = preprocess_info['ratio']
                    
                    # Remove padding and convert to original size
                    x1_orig = x1_model / ratio
                    y1_orig = y1_model / ratio
                    x2_orig = x2_model / ratio
                    y2_orig = y2_model / ratio
                    
                    # Ensure coordinates don't exceed original image size
                    x1_orig = max(0, min(x1_orig, original_width))
                    y1_orig = max(0, min(y1_orig, original_height))
                    x2_orig = max(0, min(x2_orig, original_width))
                    y2_orig = max(0, min(y2_orig, original_height))
                    
                    # Convert to integers for OpenCV
                    x1_draw, y1_draw = int(x1_orig), int(y1_orig)
                    x2_draw, y2_draw = int(x2_orig), int(y2_orig)
                    
                    # Skip if bounding box is too small
                    if x2_draw - x1_draw < 5 or y2_draw - y1_draw < 5:
                        continue
                    
                    # Bounding box color (different color for each class)
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                    color_idx = class_id % len(colors)
                    # OpenCV uses BGR order
                    b, g, r = colors[color_idx]
                    color = (b, g, r)
                    
                    # Draw bounding box
                    cv2.rectangle(vis_frame, (x1_draw, y1_draw), (x2_draw, y2_draw), color, 2)
                    
                    # Label text
                    label = f"Class {class_id}: {score:.2f}"
                    
                    # Draw label background
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(
                        vis_frame,
                        (x1_draw, y1_draw - text_size[1] - 4),
                        (x1_draw + text_size[0], y1_draw),
                        color,
                        -1  # Filled rectangle
                    )
                    
                    # Draw label text
                    cv2.putText(
                        vis_frame,
                        label,
                        (x1_draw, y1_draw - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),  # White text
                        2
                    )
            
            return vis_frame
            
        except Exception as e:
            logger.error(f"Error during frame visualization: {e}")
            return frame

# Create a singleton instance
visualization_utils = VisualizationUtils()