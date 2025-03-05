"""
Image preprocessing and postprocessing service.
"""
import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO
from typing import Dict, Tuple, Any, List, Optional

from core.logging import logger
from core.config import CONFIDENCE_THRESHOLD

class ImageService:
    """Service for handling image preprocessing and postprocessing."""
    
    @staticmethod
    def preprocess_image(image_data, input_shape) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess image for model input.
        
        Args:
            image_data: Input image data (bytes or numpy array)
            input_shape: Expected input shape for the model
            
        Returns:
            Tuple containing:
                - preprocessed_data: Preprocessed numpy array
                - preprocess_info: Information about preprocessing for later use
                
        Raises:
            Exception: If preprocessing fails
        """
        try:
            # Decode image bytes if needed
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = image_data

            # Get original image size
            original_height, original_width = img.shape[:2]
            
            # Determine target size from input shape
            if len(input_shape) == 4:  # [batch_size, channels, height, width]
                target_height, target_width = input_shape[2], input_shape[3]
            else:
                # Use default size for YOLO models
                target_height, target_width = 640, 640
                
            # Calculate ratio for maintaining aspect ratio
            ratio = min(target_width / original_width, target_height / original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            
            # Resize image (maintain aspect ratio)
            resized = cv2.resize(img, (new_width, new_height))
            
            # Apply padding (black)
            padded_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            padded_img[:new_height, :new_width] = resized
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            img_normalized = img_rgb.astype(np.float32) / 255.0
            
            # Convert NHWC to NCHW format if needed
            if len(input_shape) == 4 and input_shape[1] in [1, 3]:  # Check for NCHW format
                img_nchw = np.transpose(img_normalized, (2, 0, 1))
                preprocessed_data = np.expand_dims(img_nchw, axis=0)
            else:
                preprocessed_data = np.expand_dims(img_normalized, axis=0)
            
            # Store preprocessing info for postprocessing
            preprocess_info = {
                'original_width': original_width,
                'original_height': original_height,
                'target_width': target_width,
                'target_height': target_height,
                'new_width': new_width,
                'new_height': new_height,
                'ratio': ratio
            }
            
            return preprocessed_data, preprocess_info
        
        except Exception as e:
            logger.error(f"Error during image preprocessing: {e}")
            raise

    @staticmethod
    def postprocess_output(outputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Process model output.
        
        Args:
            outputs: Model output dictionary
            
        Returns:
            Dict containing processed results
        """
        results = {}
        
        try:
            # Process each output tensor
            for name, tensor in outputs.items():
                # Convert NumPy arrays to lists
                if isinstance(tensor, np.ndarray):
                    # For large tensors, include only shape and summary info
                    if tensor.size > 100:
                        results[name] = {
                            "shape": tensor.shape,
                            "dtype": str(tensor.dtype),
                            "min": float(tensor.min()),
                            "max": float(tensor.max()),
                            "mean": float(tensor.mean()),
                            "sample": tensor.flatten()[:10].tolist()
                        }
                    else:
                        # For small tensors, include all data
                        results[name] = tensor.tolist()
            
            return results
        
        except Exception as e:
            logger.error(f"Error during output postprocessing: {e}")
            return {"error": f"Error during output postprocessing: {str(e)}"}
    
    @staticmethod
    def image_to_base64(image_data: bytes, format: str = "JPEG", quality: int = 95) -> str:
        """Convert image data to base64 string.
        
        Args:
            image_data: Image bytes
            format: Output image format
            quality: JPEG quality (1-100)
            
        Returns:
            Base64-encoded image string
        """
        try:
            img = Image.open(BytesIO(image_data))
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            buffered = BytesIO()
            img.save(buffered, format=format, quality=quality)
            return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            # Fall back to direct encoding of the original bytes
            return base64.b64encode(image_data).decode()
    
    @staticmethod
    def is_centerbox_format(box: List[float], img_width: int, img_height: int) -> bool:
        """Check if the box is in center-size format ([center_x, center_y, width, height]).
        
        Args:
            box: Bounding box coordinates
            img_width: Image width
            img_height: Image height
            
        Returns:
            bool: True if center-size format, False if coordinate format
        """
        if len(box) != 4:
            return False
        
        x1, y1, x2, y2 = box
        
        # If coordinates are normalized (0~1 range)
        if 0 <= x1 <= 1.0 and 0 <= y1 <= 1.0 and 0 <= x2 <= 1.0 and 0 <= y2 <= 1.0:
            # In center-size format, x2(width) and y2(height) are usually < 1
            # In coordinate format, typically x1 < x2, y1 < y2
            if x1 >= x2 or y1 >= y2:
                return True
        else:
            # For pixel coordinates
            # In center-size format, width and height are usually < image_size/2
            if x2 < img_width/2 and y2 < img_height/2:
                return True
        
        return False
    
    @staticmethod
    def convert_to_original_coordinates(box_coords, preprocess_info):
        """
        모델 출력 좌표를 원본 이미지 좌표로 변환
        
        Args:
            box_coords: 모델 출력 좌표 [x1, y1, x2, y2]
            preprocess_info: 전처리 정보
            
        Returns:
            원본 이미지 좌표 [x1, y1, x2, y2]
        """
        # 입력 좌표 형태 확인
        if len(box_coords) != 4:
            raise ValueError("좌표는 [x1, y1, x2, y2] 형태여야 합니다")
        
        x1, y1, x2, y2 = box_coords
        
        # 전처리 정보에서 관련 값 추출
        ratio = preprocess_info['ratio']
        new_width = preprocess_info['new_width']
        new_height = preprocess_info['new_height']
        original_width = preprocess_info['original_width']
        original_height = preprocess_info['original_height']
        
        # 좌표가 정규화되었는지 확인 (0~1 범위)
        is_normalized = all(0 <= coord <= 1.0 for coord in [x1, y1, x2, y2])
        
        # 좌표 변환
        if is_normalized:
            # 정규화된 좌표를 모델 입력 크기로 변환
            x1_model = x1 * preprocess_info['target_width']
            y1_model = y1 * preprocess_info['target_height']
            x2_model = x2 * preprocess_info['target_width']
            y2_model = y2 * preprocess_info['target_height']
        else:
            # 이미 픽셀 단위인 경우 그대로 사용
            x1_model, y1_model, x2_model, y2_model = x1, y1, x2, y2
        
        # 패딩된 영역 내에 좌표가 있는지 확인
        x1_model = min(max(x1_model, 0), new_width)
        y1_model = min(max(y1_model, 0), new_height)
        x2_model = min(max(x2_model, 0), new_width)
        y2_model = min(max(y2_model, 0), new_height)
        
        # 모델 좌표를 원본 이미지 좌표로 변환
        x1_orig = x1_model / ratio
        y1_orig = y1_model / ratio
        x2_orig = x2_model / ratio
        y2_orig = y2_model / ratio
        
        # 원본 이미지 범위 내 확인
        x1_orig = max(0, min(x1_orig, original_width))
        y1_orig = max(0, min(y1_orig, original_height))
        x2_orig = max(0, min(x2_orig, original_width))
        y2_orig = max(0, min(y2_orig, original_height))
        
        return [x1_orig, y1_orig, x2_orig, y2_orig]

# Create a singleton instance
image_service = ImageService()