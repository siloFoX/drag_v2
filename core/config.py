"""
Application configuration settings.
"""
import os

# Define base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Create directories if they don't exist
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Server configuration
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 5000))

# Model configuration
# TODO : 여러 모델을 사용할 수 있도록 환경 변수를 설정해야함
# TODO : Memory pooling을 사용할 수 있도록 설정해야함
MODEL_PATHs = ["models/trt/gauge_detection_20240718.trt", "models/trt/gauge_feature_detection.trt", "models/trt/digital_gauge_ocr_20240807.trt", "models/trt/digital_gauge_segmentation_20240801.trt"]
GAUGE_DETECT_MODEL_PATH = os.environ.get("TRT_MODEL_PATH", "models/trt/gauge_detection_20240718.trt")
GAUGE_FEATURE_MODEL_PATH = os.environ.get("TRT_MODEL_PATH", "models/trt/gauge_feature_detection.trt")
DIGITAL_OCR_MODEL_PATH = os.environ.get("TRT_MODEL_PATH", "models/trt/digital_gauge_ocr_20240807.trt")
DIGITAL_GAUGE_SEGMENTATION_MODEL_PATH = os.environ.get("TRT_MODEL_PATH", "models/trt/digital_gauge_segmentation_20240801.trt")

USE_CUDA_STREAM = os.environ.get("USE_CUDA_STREAM", "True").lower() == "true"

# Detection settings
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.25"))