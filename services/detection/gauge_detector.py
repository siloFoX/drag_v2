# services/detection/gauge_detector.py
import cv2
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from core.config import GAUGE_DETECT_MODEL_PATH
from core.logging import logger
from core.exceptions import DetectionError, ModelNotInitializedError
from services.trt_service import model_manager

class GaugeDetector:
    """게이지 검출 및 분류를 위한 서비스 클래스"""
    
    def __init__(self, model_id: str = "gauge_detect", conf_thres: float = 0.25):
        """
        게이지 검출기 초기화
        
        Args:
            model_id: 사용할 모델 ID
            conf_thres: 검출 신뢰도 임계값
        """
        self.model_id = model_id
        self.conf_thres = conf_thres
        self.engine = None
        self.initialize()
    
    def initialize(self) -> bool:
        """게이지 검출 엔진 초기화"""
        try:
            # 기존 서비스 인스턴스가 있는지 확인
            self.engine = model_manager.get_service(self.model_id)
            
            # 서비스가 이미 존재하고 초기화되어 있으면 사용
            if self.engine and self.engine.is_initialized():
                logger.info(f"게이지 검출 엔진({self.model_id})이 이미 초기화되어 있습니다.")
                return True
            
            # 서비스가 존재하지 않거나 초기화되지 않은 경우, 등록 또는 재초기화 시도
            if not self.engine:
                logger.info(f"게이지 검출 엔진({self.model_id})을 새로 등록합니다.")
                success = model_manager.register_model(
                    self.model_id, 
                    model_path=GAUGE_DETECT_MODEL_PATH,
                    enable_metrics=True
                )
                
                if success:
                    self.engine = model_manager.get_service(self.model_id)
                    logger.info(f"게이지 검출 엔진({self.model_id}) 등록 성공")
                else:
                    logger.error(f"게이지 검출 엔진({self.model_id}) 등록 실패")
                    return False
            else:
                # 엔진이 존재하지만 초기화되지 않은 경우 재초기화
                logger.info(f"게이지 검출 엔진({self.model_id})을 재초기화합니다.")
                success = self.engine.initialize()
                if not success:
                    logger.error(f"게이지 검출 엔진({self.model_id}) 재초기화 실패")
                    return False
            
            # 최종 확인
            if not self.engine or not self.engine.is_initialized():
                logger.error(f"게이지 검출 엔진({self.model_id})이 초기화되지 않았습니다.")
                return False
                
            logger.info(f"게이지 검출 엔진({self.model_id}) 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"게이지 검출기 초기화 오류: {e}")
            return False
    
    def detect_gauges(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        이미지에서 게이지 검출 및 분류
        
        Args:
            image: 입력 이미지 (NumPy 배열)
            
        Returns:
            List[Dict]: 검출된 게이지 목록, 각 게이지는 다음 키를 포함:
                - id: 게이지 ID
                - box: 바운딩 박스 [x1, y1, x2, y2]
                - score: 검출 신뢰도
                - type: 게이지 유형 ("analog" 또는 "digital")
                - cropped_image: 잘라낸 게이지 이미지
        """
        if not self.engine:
            if not self.initialize():
                raise ModelNotInitializedError("게이지 검출 모델이 초기화되지 않았습니다")
        
        try:
            # 원본 이미지 복사 (시각화에 사용)
            original_image = image.copy()
            
            # 이미지 전처리
            preprocessed_data, preprocess_info = self._preprocess_image(image)
            
            # 추론 실행
            start_time = time.time()
            outputs, inference_time = self.engine.predict(preprocessed_data)
            
            # 검출 결과 추출
            gauges = self._process_detections(
                outputs, 
                original_image,
                preprocess_info
            )
            
            logger.info(f"게이지 {len(gauges)}개 검출 완료, 처리 시간: {inference_time:.2f}ms")
            return gauges
            
        except Exception as e:
            raise DetectionError(f"게이지 검출 중 오류 발생: {e}")
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        이미지 전처리
        
        Args:
            image: 입력 이미지
            
        Returns:
            Tuple: (전처리된 이미지, 전처리 정보)
        """
        # 원본 이미지 크기
        original_height, original_width = image.shape[:2]
        
        # 엔진 입력 형상 가져오기
        input_shape = self.engine.get_first_input_shape()
        
        if len(input_shape) == 4:  # [batch_size, channels, height, width]
            target_height, target_width = input_shape[2], input_shape[3]
        else:
            # 기본값 (YOLO 모델 표준 입력 크기)
            target_height, target_width = 640, 640
        
        # 비율 계산
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # 이미지 크기 조정
        resized = cv2.resize(image, (new_width, new_height))
        
        # 패딩 적용
        padded_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        padded_img[:new_height, :new_width] = resized
        
        # BGR → RGB 변환
        img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        
        # 정규화 (0-255 → 0-1)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # NCHW 형식 (배치 차원 추가)
        if len(input_shape) == 4 and input_shape[1] in [1, 3]:
            img_nchw = np.transpose(img_normalized, (2, 0, 1))
            preprocessed_data = np.expand_dims(img_nchw, axis=0)
        else:
            preprocessed_data = np.expand_dims(img_normalized, axis=0)
        
        # 전처리 정보 저장
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
    
    def _process_detections(
        self, outputs: Dict[str, np.ndarray], 
        original_image: np.ndarray,
        preprocess_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        검출 결과 처리
        
        Args:
            outputs: 모델 출력
            original_image: 원본 이미지
            preprocess_info: 전처리 정보
            
        Returns:
            List[Dict]: 검출된 게이지 목록
        """
        # 필요한 키 확인
        required_keys = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
        if not all(key in outputs for key in required_keys):
            logger.warning(f"검출 결과에 필요한 키가 누락됨. 필요: {required_keys}, 존재: {list(outputs.keys())}")
            return []
        
        # 검출 개수
        num_detections = int(outputs['num_dets'][0][0])
        
        # 비어 있으면 빈 결과 반환
        if num_detections == 0:
            return []
        
        # 원본 이미지 크기
        original_width = preprocess_info['original_width']
        original_height = preprocess_info['original_height']
        
        # 모델 입력 크기
        target_width = preprocess_info['target_width']
        target_height = preprocess_info['target_height']
        
        # 실제 리사이징된 이미지 크기
        new_width = preprocess_info['new_width']
        new_height = preprocess_info['new_height']
        
        # 비율
        ratio = preprocess_info['ratio']
        
        # 결과 저장용 리스트
        gauges = []
        
        # 검출 클래스 매핑
        class_names = {
            0: "analog_gauge",
            1: "digital_gauge"
        }
        
        # 각 검출 결과 처리
        for i in range(num_detections):
            # 신뢰도 점수
            score = float(outputs['det_scores'][0][i])
            
            # 신뢰도 임계값 확인
            if score < self.conf_thres:
                continue
            
            # 클래스 ID
            class_id = int(outputs['det_classes'][0][i])
            
            # 박스 좌표
            box = outputs['det_boxes'][0][i]
            x1, y1, x2, y2 = box
            
            # 좌표 정규화 여부 확인
            is_normalized = all(0 <= coord <= 1.0 for coord in [x1, y1, x2, y2])
            
            # 좌표 변환 (모델 입력 공간 → 원본 이미지 공간)
            if is_normalized:
                # 정규화된 좌표를 모델 입력 크기로 변환
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
            x1_orig = x1_model / ratio
            y1_orig = y1_model / ratio
            x2_orig = x2_model / ratio
            y2_orig = y2_model / ratio
            
            # 원본 이미지 크기를 초과하지 않도록 확인
            x1_orig = max(0, min(x1_orig, original_width))
            y1_orig = max(0, min(y1_orig, original_height))
            x2_orig = max(0, min(x2_orig, original_width))
            y2_orig = max(0, min(y2_orig, original_height))
            
            # 정수로 변환
            x1_orig, y1_orig, x2_orig, y2_orig = map(int, [x1_orig, y1_orig, x2_orig, y2_orig])
            
            # 박스 크기가 너무 작으면 건너뛰기
            if x2_orig - x1_orig < 20 or y2_orig - y1_orig < 20:
                logger.debug(f"너무 작은 게이지 무시: {x2_orig-x1_orig}x{y2_orig-y1_orig}")
                continue
            
            # 게이지 종류 결정
            gauge_type = "analog" if "analog" in class_names.get(class_id, "") else "digital"
            
            # 게이지 이미지 잘라내기 (약간의 패딩 추가)
            pad = 10
            pad_x1 = max(0, x1_orig - pad)
            pad_y1 = max(0, y1_orig - pad)
            pad_x2 = min(original_width, x2_orig + pad)
            pad_y2 = min(original_height, y2_orig + pad)
            
            cropped_image = original_image[pad_y1:pad_y2, pad_x1:pad_x2]
            
            # 결과 추가
            gauges.append({
                "id": i,
                "box": [x1_orig, y1_orig, x2_orig, y2_orig],
                "score": score,
                "class_id": class_id,
                "type": gauge_type,
                "cropped_image": cropped_image
            })
        
        return gauges
    
    def visualize_detection(self, image: np.ndarray, gauges: List[Dict[str, Any]]) -> np.ndarray:
        """
        검출 결과 시각화
        
        Args:
            image: 원본 이미지
            gauges: 검출된 게이지 목록
            
        Returns:
            np.ndarray: 시각화된 이미지
        """
        # 이미지 복사
        vis_image = image.copy()
        
        # 각 게이지 시각화
        for gauge in gauges:
            # 바운딩 박스
            box = gauge["box"]
            x1, y1, x2, y2 = map(int, box)
            
            # 색상 설정 (아날로그: 초록색, 디지털: 파란색)
            color = (0, 255, 0) if gauge["type"] == "analog" else (255, 0, 0)
            
            # 바운딩 박스 그리기
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 라벨 텍스트
            label = f"{gauge['type']} {gauge['score']:.2f}"
            
            # 텍스트 크기 측정
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # 라벨 배경 그리기
            cv2.rectangle(
                vis_image,
                (x1, y1 - text_size[1] - 5),
                (x1 + text_size[0], y1),
                color,
                -1  # 채워진 사각형
            )
            
            # 라벨 텍스트 그리기
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # 흰색 텍스트
                2
            )
            
            # 게이지 ID 표시
            cv2.putText(
                vis_image,
                f"ID: {gauge['id']}",
                (x1, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        return vis_image

# 게이지 검출기 싱글톤 인스턴스
gauge_detector = GaugeDetector()