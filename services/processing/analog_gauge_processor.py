"""
아날로그 게이지 처리 및 값 판독 서비스.
drag_v2/services/processing/analog_gauge_processor.py
"""
import cv2
import numpy as np
import math
import time
from math import atan2, degrees
import torch
from typing import Dict, List, Tuple, Any, Optional
import asyncio

from core.config import GAUGE_FEATURE_MODEL_PATH
from core.logging import logger
from core.exceptions import ProcessingError, ModelNotInitializedError
from services.trt_service import model_manager
import easyocr

class AnalogGaugeProcessor:
    """아날로그 게이지 처리 및 값 판독을 위한 서비스 클래스"""
    
    def __init__(self, model_id: str = "gauge_feature", conf_thres: float = 0.25, 
                 unit_list: List[str] = ["C", "F", "bar", "psi", "kPa", "MPa", "%", "V", "A", "kW", "Hz", "c"],
                 enable_metrics: bool = True):
        """
        아날로그 게이지 처리기 초기화
        
        Args:
            model_id: 바늘 및 특징점 감지에 사용할 모델 ID
            conf_thres: 검출 신뢰도 임계값
            unit_list: 인식할 단위 목록
            enable_metrics: 성능 지표 수집 여부
        """
        self.model_id = model_id
        self.conf_thres = conf_thres
        self.unit_list = unit_list
        self.enable_metrics = enable_metrics
        self.reader = None
        
        # 성능 측정 관련 변수
        self.processing_times = []
        self.last_reset_time = time.time()
        self.total_processed = 0
        
        # 초기화
        self.initialize()
    
    def initialize(self) -> bool:
        """특징점 검출 엔진 및 OCR 초기화"""
        try:
            # OCR 초기화 (GPU 사용 가능 시 GPU 활용)
            self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
            
            # 모델 매니저에 모델이 등록되어 있는지 확인
            service = model_manager.get_service(self.model_id)
            if service and service.is_initialized():
                logger.info(f"모델 '{self.model_id}'이(가) 이미 초기화되어 있습니다.")
                return True
            
            # 이미 등록된 모델이 있지만 초기화되지 않은 경우, 재등록 시도
            if service:
                logger.info(f"모델 '{self.model_id}'이(가) 등록되었지만 초기화되지 않았습니다. 재등록 시도 중...")
                asyncio.create_task(model_manager.release_model(self.model_id))
            
            # 모델 등록 시도
            logger.info(f"모델 '{self.model_id}'을(를) 등록 중...")
            success = model_manager.register_model(
                self.model_id, 
                GAUGE_FEATURE_MODEL_PATH,
                self.enable_metrics
            )
            
            if not success:
                logger.error(f"모델 '{self.model_id}' 등록 실패")
                return False
            
            logger.info("아날로그 게이지 처리기 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"아날로그 게이지 처리기 초기화 오류: {e}")
            return False
    
    def get_service(self):
        """현재 모델의 TensorRT 서비스 가져오기"""
        return model_manager.get_service(self.model_id)
    
    def is_initialized(self) -> bool:
        """서비스가 초기화되었는지 확인"""
        service = self.get_service()
        return service is not None and service.is_initialized()
    
    async def release(self) -> None:
        """리소스 해제"""
        try:
            # OCR 리더 해제
            self.reader = None
            
            # 모델 해제는 모델 매니저를 통해 처리
            await model_manager.release_model(self.model_id)
            
            logger.info(f"아날로그 게이지 처리기 리소스 해제 완료")
            
        except Exception as e:
            logger.error(f"아날로그 게이지 처리기 해제 중 오류: {e}")
    
    def process_gauge(self, gauge_image: np.ndarray) -> Dict[str, Any]:
        """
        아날로그 게이지 이미지 처리 및 값 판독
        
        Args:
            gauge_image: 크롭된 게이지 이미지
            
        Returns:
            Dict: 처리 결과 (값, 단위, 신뢰도 등)
        """
        if not self.is_initialized():
            raise ModelNotInitializedError("아날로그 게이지 처리 모델이 초기화되지 않았습니다")
        
        start_time = time.time()
        
        try:
            # 결과 저장용 딕셔너리
            result = {
                "value": None,
                "unit": None,
                "confidence": 0.0,
                "needle_angle": None,
                "status": "실패",
                "processing_time_ms": 0.0,
                "debug_images": {}
            }
            
            # 1. 타원(게이지 본체) 검출 및 변환
            ellipses, image_with_ellipses = self._ellipse_finder(gauge_image)
            result["debug_images"]["ellipses"] = image_with_ellipses
            
            # 게이지 본체 없을 경우 예외 처리
            if len(ellipses) == 0:
                logger.warning("게이지 본체(타원) 감지 실패")
                return self._finalize_result(result, start_time)
            
            # 최적의 타원 선택 및 원형 변환
            largest_ellipse = self._find_largest_ellipse(ellipses)
            if largest_ellipse is None:
                logger.warning("타원 필터링 후 적합한 게이지 본체 없음")
                return self._finalize_result(result, start_time)
            
            # 원형 변환
            circle_image, center_x, center_y = self._crop_and_transform_ellipse(gauge_image, largest_ellipse)
            result["debug_images"]["circle"] = circle_image
            
            # 2. 특징점 검출 (바늘 기점, 끝점 등)
            points, image_with_points = self._detect_gauge_features(circle_image)
            result["debug_images"]["features"] = image_with_points
            
            # 3. OCR을 통한 숫자 및 단위 인식
            unit, numbers, image_with_numbers, image_with_valid_numbers = self._detect_numbers_and_units(
                circle_image, center_x, center_y
            )
            result["debug_images"]["numbers"] = image_with_numbers
            result["debug_images"]["valid_numbers"] = image_with_valid_numbers
            
            # 숫자가 충분히 감지되지 않은 경우
            if len(numbers) < 2:
                logger.warning(f"감지된 숫자가 부족합니다: {len(numbers)}개")
                return self._finalize_result(result, start_time)
            
            # 4. 바늘 각도 계산
            first_number = list(numbers.keys())[0]
            first_number_pos = numbers[first_number]
            first_number_angle = self._calculate_angle(
                first_number_pos[0], first_number_pos[1], center_x, center_y
            )
            
            # 바늘 각도 찾기 (위치에 따라 다른 방법 사용)
            if points.get('base') and points.get('tip'):
                # 바늘 기점과 끝점이 모두 감지된 경우
                needle_angle, needle_image = self._calculate_needle_angle_from_points(
                    circle_image, points, first_number_angle
                )
                result["debug_images"]["needle"] = needle_image
            else:
                # Hough 변환으로 바늘 찾기
                needle_angle, needle_image, edge_image = self._find_needle_angle(
                    circle_image, center_x, center_y, first_number_angle
                )
                result["debug_images"]["needle"] = needle_image
                result["debug_images"]["edges"] = edge_image
            
            if needle_angle is None:
                logger.warning("바늘 각도 계산 실패")
                return self._finalize_result(result, start_time)
            
            # 5. 게이지 값 계산
            gauge_value, unit_str = self._calculate_gauge_value(
                circle_image, unit, numbers, needle_angle, first_number_angle, first_number, 
                center_x, center_y
            )
            
            if gauge_value is not None:
                result["value"] = gauge_value
                result["unit"] = unit_str if unit_str else ""
                result["needle_angle"] = needle_angle
                result["status"] = "성공"
                result["confidence"] = self._calculate_confidence(
                    len(numbers), needle_angle, points
                )
            
            return self._finalize_result(result, start_time)
            
        except Exception as e:
            logger.error(f"아날로그 게이지 처리 중 오류: {e}")
            # 실패 결과 생성 및 처리 시간 기록
            failed_result = {
                "value": None,
                "unit": None,
                "confidence": 0.0,
                "needle_angle": None,
                "status": "실패",
                "error_message": str(e),
                "processing_time_ms": 0.0,
                "debug_images": {}
            }
            return self._finalize_result(failed_result, start_time)
    
    def _finalize_result(self, result: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """결과 처리를 마무리하고 처리 시간 계산"""
        # 처리 시간 계산 (밀리초)
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        # 성능 통계 업데이트
        self.processing_times.append(processing_time)
        self.total_processed += 1
        
        # 최대 1000개 샘플만 보관
        if len(self.processing_times) > 1000:
            self.processing_times.pop(0)
            
        # 100회 처리마다 통계 로깅
        if self.total_processed % 100 == 0:
            self._log_performance_stats()
        
        return result
    
    def _log_performance_stats(self) -> None:
        """성능 통계를 로깅"""
        if not self.processing_times:
            return
            
        avg_time = sum(self.processing_times) / len(self.processing_times)
        max_time = max(self.processing_times)
        min_time = min(self.processing_times)
        
        elapsed_time = time.time() - self.last_reset_time
        throughput = len(self.processing_times) / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(f"아날로그 게이지 처리 성능: "
                   f"평균={avg_time:.2f}ms, 최대={max_time:.2f}ms, "
                   f"최소={min_time:.2f}ms, 처리량={throughput:.2f}개/초")
        
        # 통계 재설정
        self.processing_times = []
        self.last_reset_time = time.time()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        if not self.processing_times:
            return {
                "status": "no_data",
                "total_processed": self.total_processed
            }
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        max_time = max(self.processing_times)
        min_time = min(self.processing_times)
        
        uptime = time.time() - self.last_reset_time
        throughput = len(self.processing_times) / uptime if uptime > 0 else 0
        
        return {
            "processing_time_ms": {
                "avg": round(avg_time, 2),
                "min": round(min_time, 2),
                "max": round(max_time, 2)
            },
            "throughput": round(throughput, 2),
            "total_processed": self.total_processed,
            "samples_count": len(self.processing_times),
            "uptime_seconds": round(uptime, 2)
        }
    
    def _detect_gauge_features(self, image: np.ndarray) -> Tuple[Dict[str, Tuple[int, int]], np.ndarray]:
        """게이지 특징점 (바늘 시작점, 끝점, 최소값, 최대값) 검출"""
        service = self.get_service()
        if not service:
            logger.warning(f"모델 '{self.model_id}'에 대한 서비스를 찾을 수 없습니다")
            return {}, image.copy()
        
        # 모델 입력 형상 가져오기
        input_shape = service.get_first_input_shape()
        
        # 이미지 전처리
        from services.image_service import image_service
        preprocessed_data, preprocess_info = image_service.preprocess_image(image, input_shape)
        
        # 추론 실행
        outputs, inference_time = service.predict(preprocessed_data)
        
        # 결과 처리
        points = {'base': None, 'tip': None, 'minimum': None, 'maximum': None}
        image_with_points = image.copy()
        
        # 검출 개수가 있는지 확인
        if 'num_dets' in outputs and outputs['num_dets'][0][0] > 0:
            # 검출된 개수
            num_detections = int(outputs['num_dets'][0][0])
            
            # 클래스별 최고 신뢰도 검출 저장
            best_detections = {}  # 여기서 초기화
            
            for i in range(num_detections):
                # 클래스 및 신뢰도 가져오기
                class_id = int(outputs['det_classes'][0][i]) if 'det_classes' in outputs else 0
                score = float(outputs['det_scores'][0][i]) if 'det_scores' in outputs else 0
                
                # 신뢰도 임계값 확인
                if score < self.conf_thres:
                    continue
                
                # 클래스명 매핑
                class_names = {0: 'base', 1: 'tip', 2: 'minimum', 3: 'maximum'}
                if class_id not in class_names:
                    continue
                    
                class_name = class_names[class_id]
                
                # 현재 클래스의 최고 신뢰도 검출인지 확인
                if class_name not in best_detections or score > best_detections[class_name]['score']:
                    # 박스 좌표
                    if 'det_boxes' in outputs and i < len(outputs['det_boxes'][0]):
                        box = outputs['det_boxes'][0][i]
                        x1, y1, x2, y2 = box
                        
                        # 전처리 정보로 좌표 변환
                        original_coords = image_service.convert_to_original_coordinates(
                            [x1, y1, x2, y2], preprocess_info
                        )
                        x1, y1, x2, y2 = map(int, original_coords)
                        
                        # 중심점 계산
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        # 최고 신뢰도 검출 업데이트
                        best_detections[class_name] = {
                            'box': (x1, y1, x2, y2),
                            'center': (cx, cy),
                            'score': score
                        }
        
        # 최고 신뢰도 검출 결과를 points에 저장
        for class_name, detection in best_detections.items() if 'best_detections' in locals() else []:
            points[class_name] = detection['center']
            
            # 시각화
            x1, y1, x2, y2 = detection['box']
            cv2.rectangle(image_with_points, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_with_points, f"{class_name}: {detection['score']:.2f}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return points, image_with_points
    
    # 나머지 메서드(ellipse_finder, find_largest_ellipse 등)는 동일하게 유지
    # 이전 코드에서 그대로 사용하면 됩니다
    
    def _ellipse_finder(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        이미지에서 타원(게이지 본체) 검출
        
        Args:
            image: 입력 이미지
            
        Returns:
            검출된 타원 목록과 시각화 이미지
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # 모폴로지 연산 (노이즈 제거)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 엣지 검출
        grad = cv2.Sobel(morphed, cv2.CV_16S, 0, 1)
        abs_grad = cv2.convertScaleAbs(grad)
        edges = cv2.addWeighted(morphed, 0.5, abs_grad, 0.5, 0)
        
        # 윤곽 검출 및 타원 피팅
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = 150
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:10]
        
        ellipses = []
        for contour in filtered_contours:
            if len(contour) >= 5:  # 타원 피팅에는 최소 5개 점이 필요
                try:
                    ellipse = cv2.fitEllipse(contour)
                    ellipses.append(ellipse)
                except:
                    continue
        
        # 시각화
        image_with_ellipses = image.copy()
        for ellipse in ellipses:
            cv2.ellipse(image_with_ellipses, ellipse, (0, 255, 0), 2)
        
        return ellipses, image_with_ellipses
    
    def _find_largest_ellipse(self, ellipses: List) -> Optional[Tuple]:
        """
        가장 큰 타원을 찾고 필터링 적용
        
        Args:
            ellipses: 검출된 타원 목록
            
        Returns:
            적합한 가장 큰 타원
        """
        if not ellipses:
            return None
        
        # 타원 형태 및 위치에 따른 필터링
        filtered_ellipses = []
        for ellipse in ellipses:
            ((center_x, center_y), (axis1, axis2), angle) = ellipse
            
            # S_f(형태 파라미터), A_f(면적 파라미터), C_f(중심 이동) 계산
            S_f = abs(axis1 - axis2) / (axis1 + axis2)  # 원에 가까울수록 0에 가까움
            
            # 너무 찌그러진 타원 거부 (0.6 이상이면 타원이 너무 찌그러진 것)
            if S_f < 0.6:
                filtered_ellipses.append(ellipse)
        
        if not filtered_ellipses:
            return None
        
        # 가장 큰 타원 선택
        largest_ellipse = max(filtered_ellipses, 
                              key=lambda e: e[1][0] * e[1][1])  # axis1 * axis2가 가장 큰 것
        
        return largest_ellipse
    
    def _crop_and_transform_ellipse(self, image: np.ndarray, ellipse: Tuple) -> Tuple[np.ndarray, int, int]:
        """
        타원 영역을 잘라내고 원형으로 변환
        
        Args:
            image: 원본 이미지
            ellipse: 타원 정보 ((center_x, center_y), (axis1, axis2), angle)
            
        Returns:
            원형 변환된 이미지, 중심 x, 중심 y
        """
        # 타원 정보 추출
        ((center_x, center_y), (axis1, axis2), angle) = ellipse
        
        # 마스크 생성
        mask = np.zeros_like(image)
        cv2.ellipse(mask, ellipse, (255, 255, 255), -1)
        
        # 타원 영역 바운딩 사각형 계산
        ellipse_poly = cv2.ellipse2Poly(
            (int(center_x), int(center_y)), 
            (int(axis1 / 2), int(axis2 / 2)), 
            int(angle), 0, 360, 1
        )
        x, y, w, h = cv2.boundingRect(ellipse_poly)
        
        # 이미지 경계 확인
        x, y = max(0, x), max(0, y)
        w, h = min(image.shape[1] - x, w), min(image.shape[0] - y, h)
        
        # 타원 영역 잘라내기
        cropped_img = image[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]
        cropped_img = cv2.bitwise_and(cropped_img, cropped_mask)
        
        # 타원을 원으로 변환 (비율 유지)
        scale_factor = min(axis1, axis2) / max(axis1, axis2)
        if axis1 > axis2:
            scaled_img = cv2.resize(cropped_img, None, fx=scale_factor, fy=1, interpolation=cv2.INTER_AREA)
        else:
            scaled_img = cv2.resize(cropped_img, None, fx=1, fy=scale_factor, interpolation=cv2.INTER_AREA)
        
        # 정사각형 캔버스에 중앙 정렬
        diameter = int(max(axis1, axis2) * scale_factor)
        result_size = diameter
        result = np.zeros((result_size, result_size, 3), dtype=np.uint8)
        
        # 중앙 정렬을 위한 오프셋 계산
        x_offset = max((result_size - scaled_img.shape[1]) // 2, 0)
        y_offset = max((result_size - scaled_img.shape[0]) // 2, 0)
        
        # 대상 영역이 결과 이미지에 맞는지 확인
        target_y_range = y_offset + min(scaled_img.shape[0], result_size - y_offset)
        target_x_range = x_offset + min(scaled_img.shape[1], result_size - x_offset)
        
        # 경계 내에서 작업
        result[y_offset:target_y_range, x_offset:target_x_range] = scaled_img[:target_y_range-y_offset, :target_x_range-x_offset]
        
        # 중심 좌표 계산
        center_x = result.shape[1] // 2
        center_y = result.shape[0] // 2
        
        return result, center_x, center_y
    
    def _detect_numbers_and_units(self, image: np.ndarray, center_x: int, center_y: int) -> Tuple:
        """
        OCR을 사용하여 게이지의 숫자와 단위 감지
        
        Args:
            image: 게이지 이미지
            center_x: 게이지 중심 x 좌표
            center_y: 게이지 중심 y 좌표
            
        Returns:
            unit: 감지된 단위 {단위: 좌표}
            numbers: 감지된 숫자 {숫자: 좌표}
            image_with_detections: 모든 감지 표시된 이미지
            image_with_valid_numbers: 유효한 숫자만 표시된 이미지
        """
        if not self.reader:
            logger.warning("OCR 리더가 초기화되지 않았습니다.")
            return {}, {}, image.copy(), image.copy()
        
        try:
            # 이미지 복사
            img_all = image.copy()
            img_valid = image.copy()
            
            # OCR 실행
            results = self.reader.readtext(image)
            
            unit = {}
            numbers = {}
            valid_box = {}
            
            for box, text, score in results:
                # 바운딩 박스 좌표
                top_left = tuple(map(int, box[0]))
                bottom_right = tuple(map(int, box[2]))
                
                # 텍스트 중심 좌표
                cx = (top_left[0] + bottom_right[0]) / 2
                cy = (top_left[1] + bottom_right[1]) / 2
                coordinates = (cx, cy)
                box_corners = (top_left, bottom_right)
                
                # 텍스트 길이가 너무 길면 스킵 (숫자 또는 단위만 관심 있음)
                if 0 < len(text) < 5:
                    # 단위 체크
                    if text in self.unit_list:
                        unit[text] = coordinates
                        cv2.rectangle(img_valid, top_left, bottom_right, (255, 0, 0), 2)
                        cv2.putText(img_valid, text, (top_left[0], top_left[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    else:
                        # 텍스트 정규화 ('o'를 '0'으로, 첫 자리 'l' 또는 'I'를 '1'로)
                        transformed = []
                        i = 0
                        while i < len(text):
                            char = text[i]
                            if char == 'o':
                                # 연속된 'o'를 '0'으로 변환
                                count = 0
                                while i < len(text) and text[i] == 'o':
                                    count += 1
                                    i += 1
                                transformed.append('0' * count)
                                continue
                            elif (char == 'l' or char == 'I') and i == 0:
                                transformed.append('1')
                            else:
                                transformed.append(char)
                            i += 1
                        text = ''.join(transformed)
                    
                        # 숫자인 경우만 추가
                        if text.isdigit():
                            numbers[text] = coordinates
                            # 바운딩 박스 비율에 따른 필터링 (너무 넓거나 좁은 박스 제외)
                            box_width = bottom_right[0] - top_left[0]
                            box_height = bottom_right[1] - top_left[1]
                            if 0.4 < box_width/box_height < 4:
                                valid_box[text] = box_corners
                
                # 모든 감지 결과 시각화
                cv2.rectangle(img_all, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(img_all, text, (top_left[0], top_left[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 유효한 숫자만 시각화
            for text, corners in valid_box.items():
                top_left, bottom_right = corners
                cv2.rectangle(img_valid, top_left, bottom_right, (255, 0, 0), 2)
                cv2.putText(img_valid, text, (top_left[0], top_left[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # 각도에 따라 숫자 정렬
            if numbers:
                # 각 숫자의 각도 계산
                angles = {}
                for text, (x, y) in numbers.items():
                    dx = x - center_x
                    dy = y - center_y
                    angle = degrees(atan2(dy, dx))
                    
                    # 각도 정규화
                    if angle > 0:
                        angle -= 360
                    adjusted_angle = (angle + 270) % 360
                    angles[text] = adjusted_angle
                
                # 각도에 따라 정렬
                sorted_numbers = sorted(angles.items(), key=lambda item: item[1])
                
                # 오름차순이 아닌 숫자 제거
                sorted_texts = [num[0] for num in sorted_numbers]
                number_values = [int(num) for num in sorted_texts]
                
                indices_to_remove = []
                for i in range(1, len(number_values)):
                    if number_values[i] < number_values[i-1]:
                        indices_to_remove.append(i)
                
                for index in sorted(indices_to_remove, reverse=True):
                    del sorted_numbers[index]
                
                # 정렬된 숫자로 numbers 딕셔너리 업데이트
                new_numbers = {}
                for text, angle in sorted_numbers:
                    new_numbers[text] = numbers[text]
                
                numbers = new_numbers
            
            return unit, numbers, img_all, img_valid
            
        except Exception as e:
            logger.error(f"숫자 및 단위 인식 중 오류: {e}")
            return {}, {}, image.copy(), image.copy()
    
    def _calculate_angle(self, x: float, y: float, center_x: float, center_y: float) -> float:
        """
        중심점을 기준으로 좌표의 각도 계산
        
        Args:
            x, y: 대상 좌표
            center_x, center_y: 중심점 좌표
            
        Returns:
            각도 (도)
        """
        dx = x - center_x
        dy = y - center_y
        angle = degrees(atan2(dy, dx))
        
        # 각도 정규화
        while angle >= 360:
            angle -= 360
        while angle < 0:
            angle += 360
            
        return angle
    
    def _calculate_needle_angle_from_points(self, image: np.ndarray, points: Dict, low_num_angle: float) -> Tuple[float, np.ndarray]:
        """
        감지된 바늘 기점과 끝점을 사용하여 바늘 각도 계산
        
        Args:
            image: 게이지 이미지
            points: 감지된 특징점 딕셔너리
            low_num_angle: 가장 낮은 숫자의 각도
            
        Returns:
            바늘 각도, 시각화 이미지
        """
        base_x, base_y = points['base']
        tip_x, tip_y = points['tip']
        
        # 바늘 각도 계산
        needle_angle = degrees(atan2(tip_y - base_y, tip_x - base_x))
        
        # 각도 정규화
        needle_angle = self._normalize_angle(needle_angle)
        low_num_angle = self._normalize_angle(low_num_angle)
        
        # 각도 차이 계산 (가장 낮은 숫자를 기준으로)
        needle_angle = (needle_angle - low_num_angle)
        
        # 각도가 -120도 미만이면 방향이 반대로 계산된 것으로 가정하고 조정
        if needle_angle < -120:
            needle_angle += 360
        
        # 시각화
        result_image = image.copy()
        cv2.line(result_image, (base_x, base_y), (tip_x, tip_y), (0, 0, 255), 2)
        cv2.circle(result_image, (base_x, base_y), 5, (255, 0, 0), -1)
        
        return needle_angle, result_image
    
    def _find_needle_angle(self, image: np.ndarray, center_x: int, center_y: int, low_num_angle: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        이미지 처리 기법으로 바늘 각도 찾기
        
        Args:
            image: 게이지 이미지
            center_x, center_y: 게이지 중심 좌표
            low_num_angle: 가장 낮은 숫자의 각도
            
        Returns:
            바늘 각도, 결과 이미지, 엣지 이미지
        """
        # 이미지 처리
        color_image = image.copy()
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred_image = cv2.GaussianBlur(image_gray, (5, 5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY_INV, 11, 7)
        edges = cv2.Canny(adaptive_thresh, 50, 150, apertureSize=5)
        
        # 허프 변환으로 선 검출
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=50, maxLineGap=4)
        
        # 엣지 이미지 색상 변환 (시각화용)
        edge_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        if lines is None:
            logger.warning("바늘 감지를 위한 선 검출 실패")
            return None, color_image, edge_image
        
        # 중심 근처의 선 필터링
        diameter = min(image.shape[1], image.shape[0])
        radius = diameter // 6
        circle_center = (center_x, center_y)
        
        # 중심 표시용 원 그리기
        cv2.circle(edge_image, circle_center, radius, (0, 0, 255), 2)
        
        best_lines = []
        
        # 중심 근처의 선 찾기
        for line in lines:
            for x1, y1, x2, y2 in line:
                # 선의 시작점이나 끝점이 중심 근처인 경우만 선택
                if (np.sqrt((x1 - center_x) ** 2 + (y1 - center_y) ** 2) <= radius or
                    np.sqrt((x2 - center_x) ** 2 + (y2 - center_y) ** 2) <= radius):
                    best_lines.append((x1, y1, x2, y2))
                    cv2.line(edge_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if not best_lines:
            logger.warning("중심 근처에 바늘로 추정되는 선이 없음")
            return None, color_image, edge_image
        
        # 중심에 가장 가까운 선 순으로 정렬
        best_lines = sorted(best_lines, key=lambda line: min(
            np.sqrt((line[0] - center_x) ** 2 + (line[1] - center_y) ** 2),
            np.sqrt((line[2] - center_x) ** 2 + (line[3] - center_y) ** 2)
        ))
        
        # 가장 가까운 두 선의 평균 계산 (바늘의 양쪽 가장자리일 것으로 가정)
        if len(best_lines) >= 2:
            avg_line = np.mean(best_lines[:2], axis=0).astype(int)
        else:
            avg_line = best_lines[0]
        
        # 선의 좌표
        x1, y1, x2, y2 = avg_line
        
        # 중심을 지나도록 선 조정
        if x2 < center_x:
            dx = center_x - x2
            x1, x2 = x1 + dx, x2 + dx
        else:
            dx = x2 - center_x
            x1, x2 = x1 - dx, x2 - dx
            
        if y2 < center_y:
            dy = center_y - y2
            y1, y2 = y1 + dy, y2 + dy
        else:
            dy = y2 - center_y
            y1, y2 = y1 - dy, y2 - dy
        
        # 바늘 각도 계산
        angle = degrees(atan2(y1 - center_y, x1 - center_x))
        needle_angle = (angle - low_num_angle)
        
        # 시각화
        cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 파란색: 평균선
        cv2.line(color_image, (center_x, center_y), (int(x1), int(y1)), (0, 0, 255), 4)  # 빨간색: 실제 바늘
        cv2.circle(color_image, (center_x, center_y), 5, (255, 0, 0), -1)  # 중심점
        
        # 엣지 이미지에도 평균선 표시
        cv2.line(edge_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        return needle_angle, color_image, edge_image
    
    def _calculate_gauge_value(self, image: np.ndarray, unit: Dict, numbers: Dict, 
                              needle_angle: float, low_num_angle: float, low_num: str,
                              center_x: int, center_y: int) -> Tuple[float, str]:
        """
        바늘 각도와 감지된 숫자를 기반으로 게이지 값 계산
        
        Args:
            image: 게이지 이미지
            unit: 감지된 단위
            numbers: 감지된 숫자 및 좌표
            needle_angle: 바늘 각도
            low_num_angle: 가장 낮은 숫자의 각도
            low_num: 가장 낮은 숫자
            center_x, center_y: 게이지 중심 좌표
            
        Returns:
            게이지 값, 단위
        """
        if len(numbers) < 2:
            logger.warning("게이지 값 계산을 위한 숫자가 충분하지 않음")
            return None, None
            
        # 숫자 목록 가져오기
        keys = list(numbers.keys())
        
        # 첫 번째와 두 번째 숫자 값
        val_1 = float(keys[0])
        coor_1 = numbers[keys[0]]
        
        val_2 = float(keys[1])
        coor_2 = numbers[keys[1]]
        
        # 중심에 대한 각도 계산
        theta_1 = (270 - math.degrees(math.atan2(center_y - coor_1[1], coor_1[0] - center_x)))
        theta_2 = (270 - math.degrees(math.atan2(center_y - coor_2[1], coor_2[0] - center_x)))
        
        # 각도 차이 계산 (360도 넘어가는 경우 처리)
        angle_diff = (theta_2 - theta_1 + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360
            
        # 각도당 값 변화율 계산
        dval = (val_2 - val_1) / angle_diff
        
        # 최종 게이지 값 계산
        gauge_value = int(low_num) + needle_angle * dval
        
        # 단위 처리
        unit_str = list(unit.keys())[0] if unit else ""
        
        return gauge_value, unit_str
    
    def _normalize_angle(self, angle: float) -> float:
        """
        각도를 0~360 범위로 정규화
        
        Args:
            angle: 각도
            
        Returns:
            정규화된 각도
        """
        while angle >= 360:
            angle -= 360
        while angle < 0:
            angle += 360
        return angle
    
    def _calculate_confidence(self, num_numbers: int, needle_angle: float, points: Dict) -> float:
        """
        결과의 신뢰도 계산
        
        Args:
            num_numbers: 감지된 숫자 개수
            needle_angle: 바늘 각도
            points: 감지된 특징점
            
        Returns:
            0.0~1.0 범위의 신뢰도
        """
        # 기본 신뢰도
        confidence = 0.5
        
        # 숫자가 많을수록 신뢰도 상승
        if num_numbers >= 5:
            confidence += 0.3
        elif num_numbers >= 3:
            confidence += 0.2
        elif num_numbers >= 2:
            confidence += 0.1
            
        # 바늘 각도가 매우 큰 경우 신뢰도 감소
        if abs(needle_angle) > 300:
            confidence -= 0.2
            
        # 특징점 검출 기반으로 신뢰도 조정
        if points.get('base') and points.get('tip'):
            confidence += 0.2
        elif points.get('base'):
            confidence += 0.1
            
        # 0.0~1.0 범위로 제한
        return max(0.0, min(1.0, confidence))

# 싱글톤 인스턴스
analog_gauge_processor = AnalogGaugeProcessor()
if not analog_gauge_processor.is_initialized():
    logger.warning("아날로그 게이지 처리기 초기화 실패! 일부 기능이 작동하지 않을 수 있습니다.")