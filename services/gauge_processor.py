"""
게이지 감지 및 처리를 위한 통합 서비스
drag_v2/services/gauge_processor.py
"""
import cv2
import numpy as np
from typing import List, Dict, Any

from core.logging import logger
from core.exceptions import ProcessingError
from core.config import GAUGE_DETECT_MODEL_PATH, GAUGE_FEATURE_MODEL_PATH
from services.trt_service import model_manager
from services.detection.gauge_detector import gauge_detector
from services.processing.analog_gauge_processor import analog_gauge_processor

# 디지털 게이지 프로세서는 향후 구현 예정
# from services.processing.digital_gauge_processor import digital_gauge_processor

class GaugeProcessor:
    """
    이미지에서 게이지를 감지하고 값을 판독하는 통합 프로세서
    """
    
    def __init__(self):
        """게이지 프로세서 초기화"""
        self.initialize()
        
    # gauge_processor.py에서 통합 초기화 관리
    def initialize(self) -> bool:
        """게이지 감지 및 처리 모듈 초기화"""
        try:
            # 전역 모델 매니저 참조
            global model_manager
            
            # 1. 게이지 감지기 초기화 - 이미 초기화된 경우 재사용
            detector_service = model_manager.get_service("gauge_detect")
            if detector_service and detector_service.is_initialized():
                logger.info("게이지 감지기 모델이 이미 초기화되어 있습니다")
                detector_initialized = True
            else:
                logger.info("게이지 감지기 모델 초기화 중...")
                detector_initialized = model_manager.register_model(
                    "gauge_detect", 
                    GAUGE_DETECT_MODEL_PATH,
                    enable_metrics=True
                )
                if not detector_initialized:
                    logger.error("게이지 감지기 모델 초기화 실패")
            
            # 2. 아날로그 게이지 프로세서 초기화 - 이미 초기화된 경우 재사용
            analog_service = model_manager.get_service("gauge_feature")
            if analog_service and analog_service.is_initialized():
                logger.info("아날로그 게이지 특징 모델이 이미 초기화되어 있습니다")
                analog_initialized = True
            else:
                logger.info("아날로그 게이지 특징 모델 초기화 중...")
                analog_initialized = model_manager.register_model(
                    "gauge_feature", 
                    GAUGE_FEATURE_MODEL_PATH,
                    enable_metrics=True
                )
                if not analog_initialized:
                    logger.error("아날로그 게이지 특징 모델 초기화 실패")
                
            # 디지털 게이지 프로세서는 향후 구현
            digital_initialized = True
            
            # 초기화 상태 확인 - 모델 서비스 참조 얻기
            detector_service = model_manager.get_service("gauge_detect")
            analog_service = model_manager.get_service("gauge_feature")
            
            # 서비스 초기화 상태 확인
            detector_ok = detector_service and detector_service.is_initialized()
            analog_ok = analog_service and analog_service.is_initialized()
            
            # 최종 성공 여부 판단
            success = detector_ok and analog_ok and digital_initialized
            
            if success:
                logger.info("게이지 프로세서 초기화 완료")
            else:
                logger.error("게이지 프로세서 초기화 실패")
                
            return success
            
        except Exception as e:
            logger.error(f"게이지 프로세서 초기화 중 오류: {e}")
            return False
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        이미지에서 게이지를 감지하고 값을 판독
        
        Args:
            image: 입력 이미지
            
        Returns:
            Dict: 처리 결과가 포함된 딕셔너리
                - status: 처리 상태 ('성공' 또는 '실패')
                - message: 상태 메시지
                - gauges: 감지된 게이지 목록
                    - id: 게이지 ID
                    - type: 게이지 유형 ('analog' 또는 'digital')
                    - value: 판독된 값
                    - unit: 단위
                    - confidence: 신뢰도
                    - box: 바운딩 박스 좌표 [x1, y1, x2, y2]
                - visualization: 시각화 이미지 (선택 사항)
        """
        start_time = cv2.getTickCount()
        
        result = {
            "status": "실패",
            "message": "",
            "gauges": [],
            "count": 0,
            "processing_time_ms": 0
        }
        
        try:
            # CUDA 에러 방지를 위한 안전 모드 - 에러 복구 옵션
            cuda_error_occurred = False
            detected_gauges = []
            
            try:
                # 1. 게이지 감지
                detected_gauges = gauge_detector.detect_gauges(image)
                
            except Exception as e:
                error_str = str(e).lower()
                if "cuda" in error_str or "resource handle" in error_str:
                    cuda_error_occurred = True
                    logger.error(f"CUDA 오류 발생: {e}, 감지 단계 건너뛰기")
                    # CUDA 에러 시 빈 결과로 계속 진행
                    result["message"] = "CUDA 리소스 오류, 감지 단계 건너뛰기"
                    return self._finalize_result(result, start_time)
                else:
                    # 다른 일반 오류는 전파
                    logger.error(f"게이지 감지 중 오류: {e}")
                    result["message"] = f"감지 중 오류: {str(e)}"
                    return self._finalize_result(result, start_time)
            
            if not detected_gauges:
                result["message"] = "이미지에서 게이지가 감지되지 않았습니다"
                return self._finalize_result(result, start_time)
            
            # 2. 각 게이지 처리
            for gauge in detected_gauges:
                gauge_id = gauge["id"]
                gauge_type = gauge["type"]
                cropped_image = gauge["cropped_image"]
                
                # 기본 게이지 정보
                gauge_result = {
                    "id": gauge_id,
                    "type": gauge_type,
                    "box": gauge["box"],
                    "value": None,
                    "unit": "",
                    "confidence": 0.0,
                    "status": "실패"
                }
                
                # 3. 게이지 유형에 따른 처리
                if gauge_type == "analog":
                    # 아날로그 게이지 처리
                    try:
                        processing_result = analog_gauge_processor.process_gauge(cropped_image)
                        
                        # 처리 결과 업데이트
                        if processing_result["status"] == "성공":
                            gauge_result.update({
                                "value": processing_result["value"],
                                "unit": processing_result["unit"],
                                "confidence": processing_result["confidence"],
                                "needle_angle": processing_result["needle_angle"],
                                "status": "성공"
                            })
                            
                            # 디버그 이미지가 필요하면 추가
                            if "debug_images" in processing_result:
                                gauge_result["debug_images"] = processing_result["debug_images"]
                        else:
                            gauge_result["message"] = "아날로그 게이지 처리 실패"
                            
                    except Exception as e:
                        logger.error(f"아날로그 게이지 {gauge_id} 처리 중 오류: {e}")
                        gauge_result["message"] = f"처리 오류: {str(e)}"
                    
                elif gauge_type == "digital":
                    # 디지털 게이지 처리 (향후 구현)
                    gauge_result["message"] = "디지털 게이지 처리는 아직 구현되지 않았습니다"
                
                # 결과 추가
                result["gauges"].append(gauge_result)
            
            # 전체 결과 업데이트
            result["count"] = len(detected_gauges)
            
            # 하나 이상의 게이지가 성공적으로 처리된 경우 성공 상태로 설정
            if any(gauge["status"] == "성공" for gauge in result["gauges"]):
                result["status"] = "성공"
                result["message"] = f"{result['count']}개의 게이지 중 일부 처리 성공"
            else:
                result["message"] = f"{result['count']}개의 게이지를 감지했으나 처리 실패"
            
            # 4. 시각화 (선택 사항)
            result["visualization"] = self._visualize_results(image.copy(), result["gauges"])
            
            return self._finalize_result(result, start_time)
            
        except Exception as e:
            # 전체 처리 과정에서 오류 발생
            logger.error(f"게이지 처리 중 오류: {e}")
            result["message"] = f"처리 중 오류가 발생했습니다: {str(e)}"
            return self._finalize_result(result, start_time)
    
    def _finalize_result(self, result: Dict[str, Any], start_time: int) -> Dict[str, Any]:
        """처리 시간 계산 및 최종 결과 반환"""
        end_time = cv2.getTickCount()
        processing_time = (end_time - start_time) / cv2.getTickFrequency() * 1000  # 밀리초
        result["processing_time_ms"] = round(processing_time, 2)
        return result
    
    def _visualize_results(self, image: np.ndarray, gauges: List[Dict[str, Any]]) -> np.ndarray:
        """처리 결과를 시각적으로 표시"""
        for gauge in gauges:
            box = gauge["box"]
            x1, y1, x2, y2 = map(int, box)
            
            # 성공적으로 처리된 게이지는 녹색, 실패한 게이지는 빨간색으로 표시
            color = (0, 255, 0) if gauge["status"] == "성공" else (0, 0, 255)
            
            # 바운딩 박스 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 게이지 유형 및 ID 표시
            gauge_type = gauge["type"]
            gauge_id = gauge["id"]
            cv2.putText(image, f"{gauge_type} #{gauge_id}", (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 값 표시 (성공한 경우만)
            if gauge["status"] == "성공" and gauge.get("value") is not None:
                value_text = f"{gauge['value']:.1f} {gauge['unit']}"
                cv2.putText(image, value_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return image

# 싱글톤 인스턴스 생성
gauge_processor = GaugeProcessor()