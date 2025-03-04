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
MODEL_PATH = os.environ.get("TRT_MODEL_PATH", "models/trt/20240718_gauge_detection.trt")
USE_CUDA_STREAM = os.environ.get("USE_CUDA_STREAM", "True").lower() == "true"

# Detection settings
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.25"))