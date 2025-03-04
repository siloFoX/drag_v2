# drag_v2/server.py

import os
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi import Request, Form
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import logging
from typing import Dict, List, Optional
import base64
import io
from PIL import Image
import cv2
import time
import asyncio

# 내 TensorRT 서버 모듈 가져오기
from module.trt_server_with_memory_pool import TRTServerWithMemoryPool
from module.rtsp_stream_processor import RTSPStreamManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 템플릿 및 정적 파일 디렉토리 설정
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# 디렉토리가 없으면 생성
os.makedirs(templates_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

# 템플릿 및 정적 파일 설정
templates = Jinja2Templates(directory=templates_dir)

from contextlib import asynccontextmanager

stream_manager = RTSPStreamManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    global trt_server, stream_manager
    
    # 시작 코드 (서버 초기화)
    try:
        logger.info(f"TensorRT 서버 초기화 중... 모델: {MODEL_PATH}")
        
        # TensorRT 서버 초기화
        trt_server = TRTServerWithMemoryPool(MODEL_PATH, use_stream=USE_CUDA_STREAM)
        
        # 엔진 검증
        if not trt_server.check_engine():
            raise RuntimeError("TensorRT 엔진 검증 실패")
        
        # 웜업 실행
        trt_server.warm_up(num_runs=3)
        
        logger.info("TensorRT 서버 초기화 완료")
        
    except Exception as e:
        logger.error(f"TensorRT 서버 초기화 중 오류: {e}")
        # 시작 이벤트에서 오류가 발생하면 앱이 실행되지 않아야 하므로 종료
        import sys
        sys.exit(1)
    
    yield  # 애플리케이션 실행 중

    if stream_manager : 
        stream_manager.cleanup()
    
    # 종료 코드 (자원 해제)
    if trt_server:
        logger.info("TensorRT 서버 자원 해제 중...")
        trt_server.release()
        trt_server = None
        logger.info("TensorRT 서버 자원 해제 완료")

# 글로벌 변수로 TRT 서버 인스턴스 선언
trt_server = None

# FastAPI 앱 생성
app = FastAPI(
    title="TensorRT 추론 서버",
    description="메모리 풀을 활용한 TensorRT 모델 추론 API",
    version="1.0.0",
    lifespan=lifespan
)

# FastAPI 앱에 정적 파일 마운트 (app 정의 뒤에 추가)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 모델 설정 (환경 변수 또는 기본값 사용)
# TODO : 여러 모델을 사용할 수 있게 해야함
MODEL_PATH = os.environ.get("TRT_MODEL_PATH", "models/trt/20240718_gauge_detection.trt")
USE_CUDA_STREAM = os.environ.get("USE_CUDA_STREAM", "True").lower() == "true"

# 1. 입력 이미지 전처리 부분 수정 - 적절한 리사이징과 이미지 처리
def preprocess_image(image_data, input_shape):
    """
    이미지 전처리 함수 개선
    """
    try:
        # 바이트 데이터인 경우 이미지로 디코딩
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = image_data

        # 원본 이미지 크기 로깅
        original_height, original_width = img.shape[:2]
        # logger.info(f"원본 이미지 크기: {original_width}x{original_height}")
        
        # 입력 형상에 맞게 크기 조정
        if len(input_shape) == 4:  # [batch_size, channels, height, width]
            target_height, target_width = input_shape[2], input_shape[3]
        else:
            # 기본값 대신 YOLOv4/v5 일반적인 입력 크기 사용
            target_height, target_width = 640, 640
            
        # logger.info(f"모델 입력 크기: {target_width}x{target_height}")
        
        # 원본 이미지의 비율 유지를 위한 패딩 처리
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # 크기 조정 (비율 유지)
        resized = cv2.resize(img, (new_width, new_height))
        
        # 패딩 적용 (검은색)
        padded_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        padded_img[:new_height, :new_width] = resized
        
        # BGR -> RGB 변환
        img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        
        # 정규화: [0, 255] -> [0, 1]
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # NHWC -> NCHW 변환 (배치 차원 추가)
        if len(input_shape) == 4 and input_shape[1] in [1, 3]:  # NCHW 형식 확인
            img_nchw = np.transpose(img_normalized, (2, 0, 1))
            preprocessed_data = np.expand_dims(img_nchw, axis=0)
        else:
            preprocessed_data = np.expand_dims(img_normalized, axis=0)
        
        # 전처리 정보 저장 (후처리에서 사용)
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
        logger.error(f"이미지 전처리 중 오류: {e}")
        raise

def postprocess_output(outputs):
    """
    모델 출력 후처리 함수
    
    Args:
        outputs: 모델의 출력 딕셔너리
        
    Returns:
        results: 후처리된 결과
    """
    results = {}
    
    try:
        # 각 출력에 대한 후처리 로직
        for name, tensor in outputs.items():
            # 기본적으로 NumPy 배열을 리스트로 변환
            if isinstance(tensor, np.ndarray):
                # 큰 텐서의 경우 형상과 요약 정보만 포함
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
                    # 작은 텐서는 전체 데이터 포함
                    results[name] = tensor.tolist()
        
        return results
    
    except Exception as e:
        logger.error(f"출력 후처리 중 오류: {e}")
        return {"error": f"출력 후처리 중 오류: {str(e)}"}
    
# YOLO 모델의 바운딩 박스 좌표 출력 형식에 따른 수정
# 3. 시각화 함수 개선 - 전처리 정보를 활용한 좌표 변환 적용
def visualize_detection(image_bytes, raw_outputs=None, preprocess_info=None):
    """
    YOLO 스타일 객체 탐지 결과 시각화 (게이지 탐지 모델용)
    
    Args:
        image_bytes: 이미지 바이트 데이터
        raw_outputs: 원본 모델 출력 (딕셔너리)
        preprocess_info: 전처리 과정에서 생성된 정보 (크기 변환 등)
        
    Returns:
        base64_img: base64로 인코딩된 결과 이미지
    """
    try:
        # 바이트 스트림에서 이미지 로드
        img = Image.open(BytesIO(image_bytes))
        width, height = img.size
        logger.info(f"시각화 이미지 크기: {width}x{height}")
        
        # 그리기 객체 생성
        draw = ImageDraw.Draw(img)
        
        # 결과 없을 경우 조기 반환
        if raw_outputs is None or not isinstance(raw_outputs, dict):
            logger.info("모델 출력이 없거나 딕셔너리 형식이 아닙니다")
            buffered = BytesIO()
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode()
        
        # 모델 출력에서 데이터 추출
        outputs = None
        if 'outputs' in raw_outputs and isinstance(raw_outputs['outputs'], dict):
            outputs = raw_outputs['outputs']
        else:
            outputs = raw_outputs
        
        # 필요한 키 확인
        required_keys = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
        if not all(key in outputs for key in required_keys):
            logger.info(f"필요한 키가 누락되었습니다. 필요: {required_keys}, 존재: {list(outputs.keys())}")
            buffered = BytesIO()
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode()
        
        # 탐지 개수
        num_detections = int(outputs['num_dets'][0][0])
        logger.info(f"탐지된 객체 수: {num_detections}")
        
        # 점수 임계값 (0.25 이상만 표시)
        conf_threshold = 0.25
        
        # 전처리 정보 확인 및 설정
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
            logger.info("전처리 정보가 없어 기본값 사용")
        
        # 원본 이미지 크기
        original_width = preprocess_info['original_width']
        original_height = preprocess_info['original_height']
        
        # 모델 입력 크기
        target_width = preprocess_info['target_width']
        target_height = preprocess_info['target_height']
        
        # 실제 리사이징된 이미지 크기
        new_width = preprocess_info['new_width']
        new_height = preprocess_info['new_height']
        
        # 각 탐지 결과 처리
        detected_count = 0
        for i in range(num_detections):
            # 신뢰도 점수
            score = float(outputs['det_scores'][0][i])
            
            # 점수가 임계값 이상인 경우만 처리
            if score >= conf_threshold:
                # 바운딩 박스
                box = outputs['det_boxes'][0][i]
                # 클래스 ID
                class_id = int(outputs['det_classes'][0][i])
                
                # 박스 좌표
                x1, y1, x2, y2 = box
                
                # 좌표 정규화 여부 확인
                is_normalized = all(0 <= coord <= 1.0 for coord in [x1, y1, x2, y2])
                
                logger.info(f"원본 바운딩 박스: {box}, 정규화됨: {is_normalized}")
                
                # 좌표 변환 (모델 입력 공간 -> 원본 이미지 공간)
                if is_normalized:
                    # 정규화된 좌표 (0~1)를 모델 입력 크기에 맞게 변환
                    x1_model = x1 * target_width
                    y1_model = y1 * target_height
                    x2_model = x2 * target_width
                    y2_model = y2 * target_height
                else:
                    # 이미 픽셀 단위인 경우 그대로 사용
                    x1_model, y1_model, x2_model, y2_model = x1, y1, x2, y2
                
                # 패딩이 적용된 영역 내에 있는지 확인 및 자르기
                x1_model = min(max(x1_model, 0), new_width)
                y1_model = min(max(y1_model, 0), new_height)
                x2_model = min(max(x2_model, 0), new_width)
                y2_model = min(max(y2_model, 0), new_height)
                
                # 모델 입력 공간에서 원본 이미지 공간으로 좌표 변환
                ratio = preprocess_info['ratio']
                
                # 패딩 제거 후 원본 크기로 변환
                x1_orig = x1_model / ratio
                y1_orig = y1_model / ratio
                x2_orig = x2_model / ratio
                y2_orig = y2_model / ratio
                
                # 원본 이미지 크기를 초과하지 않도록 확인
                x1_orig = max(0, min(x1_orig, original_width))
                y1_orig = max(0, min(y1_orig, original_height))
                x2_orig = max(0, min(x2_orig, original_width))
                y2_orig = max(0, min(y2_orig, original_height))
                
                # PIL 이미지에 맞게 변환된 좌표를 정수로 변환
                x1_draw, y1_draw = int(x1_orig), int(y1_orig)
                x2_draw, y2_draw = int(x2_orig), int(y2_orig)
                
                logger.info(f"변환된 바운딩 박스: ({x1_draw}, {y1_draw}, {x2_draw}, {y2_draw})")
                
                # 바운딩 박스 크기가 너무 작으면 건너뛰기
                if x2_draw - x1_draw < 5 or y2_draw - y1_draw < 5:
                    logger.info(f"바운딩 박스가 너무 작아 건너뜀: {x2_draw-x1_draw}x{y2_draw-y1_draw}")
                    continue
                
                # 바운딩 박스 색상 (클래스별 다른 색상)
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                color_idx = class_id % len(colors)
                color = colors[color_idx]
                
                # 바운딩 박스 그리기
                draw.rectangle(
                    [x1_draw, y1_draw, x2_draw, y2_draw], 
                    outline=color, 
                    width=3
                )
                
                # 라벨 텍스트
                label = f"Class {class_id}: {score:.2f}"
                
                # 텍스트 크기 측정 (PIL 버전 호환성)
                try:
                    # PIL 9.0.0 이상
                    font = ImageDraw.getfont()
                    text_bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                except (AttributeError, TypeError):
                    # 이전 버전 호환
                    if hasattr(draw, 'textsize'):
                        text_width, text_height = draw.textsize(label)
                    else:
                        text_width, text_height = 100, 15  # 추정값
                
                # 라벨 배경 그리기
                draw.rectangle(
                    [x1_draw, y1_draw - text_height - 4, x1_draw + text_width, y1_draw], 
                    fill=color
                )
                
                # 라벨 텍스트 그리기
                draw.text(
                    [x1_draw, y1_draw - text_height - 2], 
                    label, 
                    fill=(255, 255, 255)  # 흰색 텍스트
                )
                
                detected_count += 1
                
        logger.info(f"시각화된 객체 수: {detected_count}")
        
        # 결과 이미지를 base64로 인코딩
        buffered = BytesIO()
        
        # RGBA 모드인 경우 RGB로 변환
        if img.mode == 'RGBA':
            img = img.convert('RGB')
            
        img.save(buffered, format="JPEG", quality=95)
        base64_img = base64.b64encode(buffered.getvalue()).decode()
        
        return base64_img
        
    except Exception as e:
        logger.error(f"결과 시각화 중 오류: {e}", exc_info=True)
        # 오류 발생 시 원본 이미지 반환
        return base64.b64encode(image_bytes).decode()

# YOLO 중심점-크기 형식인지 확인하는 헬퍼 함수
def is_centerbox_format(box, img_width, img_height):
    """
    박스가 중심점-크기 형식([center_x, center_y, width, height])인지 확인
    
    Args:
        box: 바운딩 박스 좌표
        img_width: 이미지 너비
        img_height: 이미지 높이
        
    Returns:
        bool: 중심점-크기 형식이면 True, 좌표 형식이면 False
    """
    if len(box) != 4:
        return False
    
    x1, y1, x2, y2 = box
    
    # 좌표가 정규화된 경우 (0~1 범위)
    if 0 <= x1 <= 1.0 and 0 <= y1 <= 1.0 and 0 <= x2 <= 1.0 and 0 <= y2 <= 1.0:
        # 중심점-크기 형식의 경우 x2(width)와 y2(height)는 
        # 일반적으로 1보다 작은 값이어야 함
        # 좌표 형식의 경우 일반적으로 x1 < x2, y1 < y2
        
        # x1이 x2보다 크거나 같은 경우 (일반적인 좌표 형식이라면 이런 경우는 없음)
        # 또는 y1이 y2보다 크거나 같은 경우
        if x1 >= x2 or y1 >= y2:
            return True
    else:
        # 픽셀 좌표인 경우
        # 중심점-크기 형식에서 width, height는 일반적으로 이미지 크기보다 작아야 함
        if x2 < img_width/2 and y2 < img_height/2:
            return True
    
    return False

# 5. 비디오 프레임 시각화 함수도 수정 (visualize_frame)
def visualize_frame(frame: np.ndarray, outputs: Dict, preprocess_info=None) -> np.ndarray:
    """
    프레임에 객체 탐지 결과 시각화
    
    Args:
        frame: 입력 이미지 프레임
        outputs: 모델 출력 결과
        preprocess_info: 전처리 과정에서 생성된 정보 (크기 변환 등)
        
    Returns:
        시각화된 이미지 프레임
    """
    try:
        # 결과 없으면 원본 프레임 반환
        if not outputs:
            return frame
        
        # 프레임 복사
        vis_frame = frame.copy()
        height, width = vis_frame.shape[:2]
        
        # 필요한 키 확인
        required_keys = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
        if not all(key in outputs for key in required_keys):
            return vis_frame
        
        # 탐지 개수
        num_detections = int(outputs['num_dets'][0][0])
        
        # 점수 임계값 (0.25 이상만 표시)
        conf_threshold = 0.25
        
        # 전처리 정보 확인 및 설정
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
        
        # 원본 이미지 크기
        original_width = preprocess_info['original_width']
        original_height = preprocess_info['original_height']
        
        # 모델 입력 크기
        target_width = preprocess_info['target_width']
        target_height = preprocess_info['target_height']
        
        # 실제 리사이징된 이미지 크기
        new_width = preprocess_info['new_width']
        new_height = preprocess_info['new_height']
        
        # 각 탐지 결과 처리
        for i in range(num_detections):
            # 신뢰도 점수
            score = float(outputs['det_scores'][0][i])
            
            # 점수가 임계값 이상인 경우만 처리
            if score >= conf_threshold:
                # 바운딩 박스
                box = outputs['det_boxes'][0][i]
                # 클래스 ID
                class_id = int(outputs['det_classes'][0][i])
                
                # 박스 좌표
                x1, y1, x2, y2 = box
                
                # 좌표 정규화 여부 확인
                is_normalized = all(0 <= coord <= 1.0 for coord in [x1, y1, x2, y2])
                
                # 좌표 변환 (모델 입력 공간 -> 원본 이미지 공간)
                if is_normalized:
                    # 정규화된 좌표 (0~1)를 모델 입력 크기에 맞게 변환
                    x1_model = x1 * target_width
                    y1_model = y1 * target_height
                    x2_model = x2 * target_width
                    y2_model = y2 * target_height
                else:
                    # 이미 픽셀 단위인 경우 그대로 사용
                    x1_model, y1_model, x2_model, y2_model = x1, y1, x2, y2
                
                # 패딩이 적용된 영역 내에 있는지 확인 및 자르기
                x1_model = min(max(x1_model, 0), new_width)
                y1_model = min(max(y1_model, 0), new_height)
                x2_model = min(max(x2_model, 0), new_width)
                y2_model = min(max(y2_model, 0), new_height)
                
                # 모델 입력 공간에서 원본 이미지 공간으로 좌표 변환
                ratio = preprocess_info['ratio']
                
                # 패딩 제거 후 원본 크기로 변환
                x1_orig = x1_model / ratio
                y1_orig = y1_model / ratio
                x2_orig = x2_model / ratio
                y2_orig = y2_model / ratio
                
                # 원본 이미지 크기를 초과하지 않도록 확인
                x1_orig = max(0, min(x1_orig, original_width))
                y1_orig = max(0, min(y1_orig, original_height))
                x2_orig = max(0, min(x2_orig, original_width))
                y2_orig = max(0, min(y2_orig, original_height))
                
                # OpenCV 이미지에 맞게 변환된 좌표를 정수로 변환
                x1_draw, y1_draw = int(x1_orig), int(y1_orig)
                x2_draw, y2_draw = int(x2_orig), int(y2_orig)
                
                # 바운딩 박스 크기가 너무 작으면 건너뛰기
                if x2_draw - x1_draw < 5 or y2_draw - y1_draw < 5:
                    continue
                
                # 바운딩 박스 색상 (클래스별 다른 색상)
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                color_idx = class_id % len(colors)
                # OpenCV는 BGR 순서
                b, g, r = colors[color_idx]
                color = (b, g, r)
                
                # 바운딩 박스 그리기
                cv2.rectangle(vis_frame, (x1_draw, y1_draw), (x2_draw, y2_draw), color, 2)
                
                # 라벨 텍스트
                label = f"Class {class_id}: {score:.2f}"
                
                # 라벨 배경 그리기
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(
                    vis_frame,
                    (x1_draw, y1_draw - text_size[1] - 4),
                    (x1_draw + text_size[0], y1_draw),
                    color,
                    -1  # 채워진 사각형
                )
                
                # 라벨 텍스트 그리기
                cv2.putText(
                    vis_frame,
                    label,
                    (x1_draw, y1_draw - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),  # 흰색 텍스트
                    2
                )
        
        return vis_frame
        
    except Exception as e:
        logger.error(f"프레임 시각화 중 오류: {e}")
        return frame

# 홈페이지 핸들러 (GET 및 POST 모두 처리)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """홈페이지 렌더링 (이미지 업로드 폼)"""
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# 2. 전처리 정보를 predict 함수에 전달하고 시각화에 사용하기 위한 수정
@app.post("/", response_class=HTMLResponse)
async def process_image(request: Request, file: UploadFile = File(...)):
    """이미지 업로드 및 추론 처리"""
    if not trt_server:
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "error": "TensorRT 서버가 초기화되지 않았습니다"
            }
        )
    
    try:
        # 파일 확장자 확인
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            return templates.TemplateResponse(
                "index.html", 
                {
                    "request": request, 
                    "error": "지원되지 않는 파일 형식입니다. PNG, JPG, JPEG, BMP만 지원합니다."
                }
            )
        
        # 파일 읽기
        content = await file.read()
        logger.info(f"이미지 업로드됨: {file.filename}, 크기: {len(content)} 바이트")
        
        # 입력 바인딩 이름 가져오기 (첫 번째 입력 사용)
        first_input_idx = trt_server.input_binding_idxs[0]
        input_name = trt_server.engine.get_binding_name(first_input_idx)
        input_shape = trt_server.binding_shapes[input_name]
        logger.info(f"모델 입력 정보: 이름={input_name}, 형상={input_shape}")
        
        # 이미지 전처리 - 전처리 정보도 함께 받아옴
        logger.info("이미지 전처리 시작")
        preprocessed_data, preprocess_info = preprocess_image(content, input_shape)
        logger.info(f"전처리된 데이터 형상: {preprocessed_data.shape}, 타입: {preprocessed_data.dtype}")
        logger.info(f"전처리 정보: {preprocess_info}")
        
        # 추론 실행
        logger.info("추론 실행 시작")
        start_time = time.time()
        outputs, inference_time = trt_server.predict(preprocessed_data)
        total_time = (time.time() - start_time) * 1000  # ms
        logger.info(f"추론 완료: 시간={inference_time:.2f}ms, 출력 개수={len(outputs)}")
        
        # 디버깅을 위한 출력 내용 로깅
        if outputs:
            logger.info(f"출력 키: {list(outputs.keys())}")
            for k, v in outputs.items():
                if isinstance(v, np.ndarray):
                    logger.info(f"  {k}: 형상={v.shape}, 타입={v.dtype}")
                    # 샘플 값 몇 개 출력
                    if v.size > 0:
                        flat_array = v.flatten()
                        sample_values = flat_array[:min(10, flat_array.size)]
                        logger.info(f"  {k} 샘플 값: {sample_values}")
                    
                    # 감지된 객체 정보 자세히 출력
                    if k == 'det_boxes' and 'num_dets' in outputs and 'det_scores' in outputs:
                        num_dets = outputs['num_dets'][0][0]
                        logger.info(f"감지된 객체 수: {num_dets}")
                        for i in range(min(num_dets, 10)):  # 최대 10개까지만 출력
                            box = outputs['det_boxes'][0][i]
                            score = outputs['det_scores'][0][i]
                            if 'det_classes' in outputs:
                                class_id = outputs['det_classes'][0][i]
                                logger.info(f"  객체 {i}: 클래스={class_id}, 점수={score}, 박스={box}")
                            else:
                                logger.info(f"  객체 {i}: 점수={score}, 박스={box}")
        
        # 원본 이미지를 RGB로 변환하여 base64 인코딩
        logger.info("원본 이미지 인코딩 시작")
        original_base64 = None
        try:
            img = Image.open(BytesIO(content))
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            original_base64 = base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            logger.error(f"원본 이미지 인코딩 중 오류: {e}")
            original_base64 = base64.b64encode(content).decode()
        
        # 이미지 시각화 - 전처리 정보 함께 전달
        logger.info("결과 시각화 시작")
        base64_img = visualize_detection(content, outputs, preprocess_info)
        logger.info("결과 시각화 완료")
        
        # 결과 페이지 렌더링
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "original_image": original_base64,
                "result_image": base64_img,
                "result": postprocess_output(outputs),
                "inference_time": inference_time,
                "total_time": total_time,
                "filename": file.filename
            }
        )
    
    except Exception as e:
        logger.error(f"이미지 처리 중 오류: {e}", exc_info=True)
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "error": f"이미지 처리 중 오류: {str(e)}"
            }
        )
    
@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    if trt_server:
        return {"status": "healthy"}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "TensorRT 서버가 초기화되지 않았습니다"}
        )

@app.get("/model/info")
async def model_info():
    """모델 정보 조회 엔드포인트"""
    if not trt_server:
        raise HTTPException(status_code=503, detail="TensorRT 서버가 초기화되지 않았습니다")
    
    try:
        # 바인딩 정보 수집
        shapes = trt_server.get_buffer_shapes()
        dtypes = trt_server.get_buffer_dtypes()
        
        # 입력 및 출력 정보 구성
        inputs = {}
        outputs = {}
        
        for idx in trt_server.input_binding_idxs:
            name = trt_server.engine.get_binding_name(idx)
            # TensorRT 또는 NumPy 형상을 Python 리스트로 변환
            shape_list = [int(dim) for dim in shapes[name]]
            
            inputs[name] = {
                "shape": shape_list,
                "dtype": str(dtypes[name])
            }
        
        for idx in trt_server.output_binding_idxs:
            name = trt_server.engine.get_binding_name(idx)
            # TensorRT 또는 NumPy 형상을 Python 리스트로 변환
            shape_list = [int(dim) for dim in shapes[name]]
            
            outputs[name] = {
                "shape": shape_list,
                "dtype": str(dtypes[name])
            }
        
        return {
            "model_path": MODEL_PATH,
            "inputs": inputs,
            "outputs": outputs,
            "use_cuda_stream": USE_CUDA_STREAM
        }
    
    except Exception as e:
        logger.error(f"모델 정보 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"모델 정보 조회 중 오류: {str(e)}")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    이미지 파일을 받아 추론 수행
    
    Args:
        file: 업로드된 이미지 파일
        
    Returns:
        결과: 추론 결과 및 시간
    """
    if not trt_server:
        raise HTTPException(status_code=503, detail="TensorRT 서버가 초기화되지 않았습니다")
    
    try:
        # 파일 확장자 확인
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            raise HTTPException(status_code=400, detail="지원되지 않는 파일 형식입니다. PNG, JPG, JPEG, BMP만 지원합니다.")
        
        # 파일 읽기
        content = await file.read()
        
        # 입력 바인딩 이름 가져오기 (첫 번째 입력 사용)
        first_input_idx = trt_server.input_binding_idxs[0]
        input_name = trt_server.engine.get_binding_name(first_input_idx)
        input_shape = trt_server.binding_shapes[input_name]
        
        # 이미지 전처리
        preprocessed_data = preprocess_image(content, input_shape)
        
        # 추론 실행
        start_time = time.time()
        outputs, inference_time = trt_server.predict(preprocessed_data)
        total_time = (time.time() - start_time) * 1000  # ms
        
        # 결과 후처리
        results = postprocess_output(outputs)
        
        return {
            "success": True,
            "inference_time_ms": inference_time,
            "total_time_ms": total_time,
            "results": results
        }
    
    except Exception as e:
        logger.error(f"추론 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"추론 중 오류: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    여러 이미지 파일을 받아 배치 추론 수행
    
    Args:
        files: 업로드된 이미지 파일 리스트
        
    Returns:
        결과: 추론 결과 및 시간
    """
    if not trt_server:
        raise HTTPException(status_code=503, detail="TensorRT 서버가 초기화되지 않았습니다")
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="최소 하나 이상의 파일이 필요합니다")
    
    try:
        results = []
        total_inference_time = 0
        
        # 입력 바인딩 이름 가져오기 (첫 번째 입력 사용)
        first_input_idx = trt_server.input_binding_idxs[0]
        input_name = trt_server.engine.get_binding_name(first_input_idx)
        input_shape = trt_server.binding_shapes[input_name]
        
        # 각 파일에 대해 처리
        for file in files:
            # 파일 확장자 확인
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                logger.warning(f"지원되지 않는 파일 형식: {file.filename}")
                results.append({
                    "filename": file.filename,
                    "error": "지원되지 않는 파일 형식"
                })
                continue
            
            # 파일 읽기
            content = await file.read()
            
            # 이미지 전처리
            try:
                preprocessed_data = preprocess_image(content, input_shape)
                
                # 추론 실행
                outputs, inference_time = trt_server.predict(preprocessed_data)
                total_inference_time += inference_time
                
                # 결과 후처리
                processed_results = postprocess_output(outputs)
                
                results.append({
                    "filename": file.filename,
                    "inference_time_ms": inference_time,
                    "results": processed_results
                })
                
            except Exception as e:
                logger.error(f"파일 {file.filename} 처리 중 오류: {e}")
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "total_files": len(files),
            "processed_files": len(results),
            "total_inference_time_ms": total_inference_time,
            "results": results
        }
    
    except Exception as e:
        logger.error(f"배치 추론 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"배치 추론 중 오류: {str(e)}")

@app.post("/predict/json")
async def predict_json(data: Dict):
    """
    JSON 형식의 데이터로 추론 수행
    
    Args:
        data: 입력 데이터 (형식: {"inputs": {"input_name": [...data...]}, "options": {...}})
        
    Returns:
        결과: 추론 결과 및 시간
    """
    if not trt_server:
        raise HTTPException(status_code=503, detail="TensorRT 서버가 초기화되지 않았습니다")
    
    try:
        # 입력 데이터 검증
        if "inputs" not in data:
            raise HTTPException(status_code=400, detail="요청에 'inputs' 필드가 없습니다")
        
        inputs = data["inputs"]
        options = data.get("options", {})
        
        # 첫 번째 입력 바인딩 가져오기
        first_input_idx = trt_server.input_binding_idxs[0]
        default_input_name = trt_server.engine.get_binding_name(first_input_idx)
        
        # 입력 데이터 준비
        input_name = list(inputs.keys())[0] if inputs else default_input_name
        input_data = inputs.get(input_name)
        
        if not input_data:
            raise HTTPException(status_code=400, detail=f"입력 '{input_name}'에 대한 데이터가 없습니다")
        
        # NumPy 배열로 변환
        try:
            np_data = np.array(input_data, dtype=np.float32)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"입력 데이터를 NumPy 배열로 변환하는 중 오류: {str(e)}")
        
        # 입력 형상 맞추기
        expected_shape = trt_server.binding_shapes[input_name]
        try:
            if np_data.shape != expected_shape:
                logger.warning(f"입력 형상 불일치. 예상: {expected_shape}, 실제: {np_data.shape}. 크기 조정 시도...")
                np_data = np.resize(np_data, expected_shape)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"입력 형상 조정 중 오류: {str(e)}")
        
        # 추론 실행
        start_time = time.time()
        outputs, inference_time = trt_server.predict(np_data)
        total_time = (time.time() - start_time) * 1000  # ms
        
        # 결과 후처리
        results = postprocess_output(outputs)
        
        return {
            "success": True,
            "inference_time_ms": inference_time,
            "total_time_ms": total_time,
            "results": results
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"JSON 추론 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"JSON 추론 중 오류: {str(e)}")
    
# RTSP 스트림 설정 페이지
@app.get("/video-config", response_class=HTMLResponse)
async def video_config(request: Request):
    """RTSP 스트림 설정 페이지"""
    streams = stream_manager.get_all_streams()
    return templates.TemplateResponse(
        "video_config.html", 
        {
            "request": request,
            "streams": streams
        }
    )

# RTSP 스트림 추가 엔드포인트
@app.post("/api/streams")
async def add_stream(stream_id: str = Form(...), rtsp_url: str = Form(...)):
    """
    RTSP 스트림 추가
    
    Args:
        stream_id: 스트림 식별자
        rtsp_url: RTSP URL
    """
    if not stream_id or not rtsp_url:
        raise HTTPException(status_code=400, detail="스트림 ID와 RTSP URL이 필요합니다")
    
    if not rtsp_url.startswith(("rtsp://", "http://", "https://", "file://")):
        raise HTTPException(status_code=400, detail="유효한 URL 형식이 아닙니다")
    
    success = stream_manager.add_stream(stream_id, rtsp_url)
    if not success:
        raise HTTPException(status_code=500, detail="스트림 추가 실패")
    
    return {"status": "success", "message": f"스트림 '{stream_id}' 추가됨"}

# RTSP 스트림 제거 엔드포인트
@app.delete("/api/streams/{stream_id}")
async def remove_stream(stream_id: str):
    """
    RTSP 스트림 제거
    
    Args:
        stream_id: 스트림 식별자
    """
    success = stream_manager.remove_stream(stream_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"스트림 '{stream_id}'를 찾을 수 없습니다")
    
    return {"status": "success", "message": f"스트림 '{stream_id}' 제거됨"}

# 모든 스트림 상태 조회 엔드포인트
@app.get("/api/streams")
async def get_streams():
    """모든 스트림 상태 조회"""
    streams = stream_manager.get_all_streams()
    return streams

# 비디오 스트림 시청 페이지
@app.get("/video", response_class=HTMLResponse)
async def video_page(request: Request, stream_id: str = None):
    """
    비디오 스트림 시청 페이지
    
    Args:
        stream_id: 스트림 식별자 (없으면 모든 스트림 목록 표시)
    """
    streams = stream_manager.get_all_streams()
    
    # 특정 스트림 ID가 지정되었으나 존재하지 않는 경우
    if stream_id and stream_id not in streams:
        return templates.TemplateResponse(
            "video.html", 
            {
                "request": request,
                "error": f"스트림 '{stream_id}'를 찾을 수 없습니다",
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

# 비디오 스트림 엔드포인트 (MJPEG 스트림)
@app.get("/api/video-feed/{stream_id}")
async def video_feed(stream_id: str):
    """
    비디오 피드 엔드포인트 (MJPEG 스트림)
    
    Args:
        stream_id: 스트림 식별자
    """
    if stream_id not in stream_manager.streams:
        raise HTTPException(status_code=404, detail=f"스트림 '{stream_id}'를 찾을 수 없습니다")
    
    return StreamingResponse(
        generate_frames(stream_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# 프레임 생성 제너레이터 함수
async def generate_frames(stream_id: str):
    """
    MJPEG 스트리밍을 위한 프레임 생성 제너레이터
    
    Args:
        stream_id: 스트림 식별자
    """
    try:
        processor = stream_manager.get_stream(stream_id)
        if not processor:
            logger.error(f"스트림 '{stream_id}'를 찾을 수 없습니다")
            return
        
        fps_limiter = FPSLimiter(max_fps=15)  # 최대 15 FPS로 제한
        
        while True:
            # FPS 제한
            if not fps_limiter.should_process():
                await asyncio.sleep(0.01)  # 짧은 지연
                continue
            
            # 프레임 가져오기
            frame = processor.get_frame()
            if frame is None:
                await asyncio.sleep(0.1)  # 프레임이 없으면 대기
                continue
            
            # 모델 추론 실행 (객체 검출)
            try:
                processed_frame = await process_frame_with_model(frame)
            except Exception as e:
                logger.error(f"프레임 처리 중 오류: {e}")
                processed_frame = frame  # 오류 시 원본 프레임 사용
            
            # JPEG 인코딩
            success, encoded_image = cv2.imencode('.jpg', processed_frame)
            if not success:
                logger.error("이미지 인코딩 실패")
                await asyncio.sleep(0.1)
                continue
            
            # MJPEG 형식으로 프레임 제공
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + 
                   encoded_image.tobytes() + 
                   b'\r\n')
            
            # 짧은 지연으로 CPU 사용률 감소
            await asyncio.sleep(0.01)
    
    except Exception as e:
        logger.error(f"프레임 생성 중 오류: {e}")
        return

# FPS 제한 클래스
class FPSLimiter:
    """FPS 제한을 위한 헬퍼 클래스"""
    
    def __init__(self, max_fps=60):
        self.max_fps = max_fps
        self.frame_time = 1.0 / max_fps
        self.last_frame_time = 0
    
    def should_process(self):
        """
        현재 시간에 프레임 처리 여부 결정
        
        Returns:
            처리 여부 (True/False)
        """
        current_time = time.time()
        time_elapsed = current_time - self.last_frame_time
        
        if time_elapsed >= self.frame_time:
            self.last_frame_time = current_time
            return True
        
        return False

# 4. 비디오 프레임 처리 함수도 수정 (process_frame_with_model)
async def process_frame_with_model(frame: np.ndarray) -> np.ndarray:
    """
    이미지 프레임에 객체 탐지 모델 적용
    
    Args:
        frame: 입력 이미지 프레임
        
    Returns:
        처리된 이미지 프레임
    """
    if frame is None or trt_server is None:
        return frame
    
    try:
        # 입력 바인딩 이름 가져오기
        first_input_idx = trt_server.input_binding_idxs[0]
        input_name = trt_server.engine.get_binding_name(first_input_idx)
        input_shape = trt_server.binding_shapes[input_name]
        
        # 이미지 전처리 - 전처리 정보도 함께 받아옴
        preprocessed_data, preprocess_info = preprocess_image(frame, input_shape)
        
        # 모델 추론 실행
        outputs, inference_time = trt_server.predict(preprocessed_data)
        
        # 결과 시각화 - 전처리 정보 함께 전달
        processed_frame = visualize_frame(frame, outputs, preprocess_info)
        
        # 추론 시간 표시
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
        logger.error(f"프레임 처리 중 오류: {e}")
        return frame
    

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"TensorRT 추론 서버를 {host}:{port}에서 시작합니다...")
    
    # 서버 시작
    uvicorn.run(app, host=host, port=port)