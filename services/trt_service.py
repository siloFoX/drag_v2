"""
메모리 풀링, 다중 모델 지원 및 성능 모니터링 기능이 포함된 확장 TensorRT 추론 서비스.
"""
import numpy as np
import time
import asyncio
from typing import Dict, Tuple, List, Any, Optional, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import sys
import os
from datetime import datetime

from core.config import USE_CUDA_STREAM
from core.config import GAUGE_DETECT_MODEL_PATH as DEFAULT_MODEL_PATH
from core.config import GAUGE_FEATURE_MODEL_PATH, DIGITAL_OCR_MODEL_PATH, DIGITAL_GAUGE_SEGMENTATION_MODEL_PATH
from core.logging import logger
from services.trt_server_wrapper import SafeTRTServerWrapper
# 메트릭 수집 기능 구현
class MetricsCollector:
    """TensorRT 모델의 성능 지표를 수집하고 관리하는 간단한 클래스."""
    
    def __init__(self, model_name: str):
        """MetricsCollector를 초기화합니다.
        
        Args:
            model_name: 모니터링 대상 모델 이름
        """
        self.model_name = model_name
        self.inference_times = []
        self.total_times = []
        self.start_time = time.time()
        self.total_inferences = 0
        
    def record_inference(self, inference_time: float, total_time: float) -> None:
        """추론 성능을 기록합니다.
        
        Args:
            inference_time: 순수 TensorRT 추론 시간 (ms)
            total_time: 전체 처리 시간 (ms)
        """
        self.inference_times.append(inference_time)
        self.total_times.append(total_time)
        self.total_inferences += 1
        
        # 최대 1000개 샘플만 유지
        if len(self.inference_times) > 1000:
            self.inference_times.pop(0)
            self.total_times.pop(0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """수집된 성능 지표에 대한 통계를 반환합니다.
        
        Returns:
            Dict[str, Any]: 성능 통계 딕셔너리
        """
        if not self.inference_times:
            return {
                "model_name": self.model_name,
                "status": "no_data",
                "total_inferences": 0
            }
        
        # 기본 통계 계산
        avg_inference = sum(self.inference_times) / len(self.inference_times)
        avg_total = sum(self.total_times) / len(self.total_times)
        max_inference = max(self.inference_times)
        min_inference = min(self.inference_times)
        
        # 처리량 계산
        uptime = time.time() - self.start_time
        throughput = self.total_inferences / uptime if uptime > 0 else 0
        
        return {
            "model_name": self.model_name,
            "inference_time_ms": {
                "avg": round(avg_inference, 2),
                "min": round(min_inference, 2),
                "max": round(max_inference, 2)
            },
            "total_time_ms": {
                "avg": round(avg_total, 2)
            },
            "throughput": round(throughput, 2),
            "total_inferences": self.total_inferences,
            "samples_count": len(self.inference_times),
            "uptime_seconds": round(uptime, 2)
        }

class TensorRTService:
    """TensorRT 엔진 및 추론을 위한 확장 기능이 포함된 서비스."""
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, enable_metrics: bool = True):
        """TensorRT 서비스를 초기화합니다.
        
        Args:
            model_path: TensorRT 엔진 파일 경로
            enable_metrics: 성능 지표 수집 여부
        """
        self.trt_server = None
        self.model_path = model_path
        self.use_cuda_stream = USE_CUDA_STREAM
        self.model_name = model_path.split('/')[-1].split('.')[0]
        self.enable_metrics = enable_metrics
        
        # 성능 지표
        if self.enable_metrics:
            self.metrics = MetricsCollector(self.model_name)
            self.inference_times = []
            self.last_reset_time = time.time()
            self.inference_count = 0
        
    def initialize(self) -> bool:
        """구성된 모델로 TensorRT 서버를 초기화합니다.
        
        Returns:
            bool: 초기화가 성공적으로 완료되면 True, 그렇지 않으면 False.
        """
        try:
            logger.info(f"{self.model_name}용 TensorRT 서버 초기화 중... 모델: {self.model_path}")
            
            # 모델 파일이 존재하는지 확인
            if not os.path.exists(self.model_path):
                logger.error(f"모델 파일을 찾을 수 없음: {self.model_path}")
                return False
            
            # 안전한 래퍼로 TensorRT 서버 초기화
            self.trt_server = SafeTRTServerWrapper(self.model_path, use_stream=self.use_cuda_stream)
            
            # 서버 초기화
            if not self.trt_server.initialize():
                raise RuntimeError(f"{self.model_name} TensorRT 엔진 초기화 실패")
            
            # 초기 성능 향상을 위한 웜업 수행
            self._warm_up()
            
            logger.info(f"{self.model_name} TensorRT 서버 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"{self.model_name} TensorRT 서버 초기화 오류: {e}")
            return False
    
    def _warm_up(self, num_runs: int = 3) -> None:
        """더미 데이터로 추론을 실행하여 모델을 웜업합니다.
        
        Args:
            num_runs: 웜업 추론 실행 횟수
        """
        if not self.is_initialized():
            logger.warning(f"초기화되지 않은 모델 {self.model_name}를 웜업할 수 없음")
            return
            
        try:
            # 첫 번째 입력 바인딩의 입력 shape 가져오기
            input_shape = self.get_first_input_shape()
            
            # 더미 입력 데이터 생성
            dummy_input = np.random.random(input_shape).astype(np.float32)
            
            logger.info(f"{self.model_name} 모델 {num_runs}회 추론으로 웜업 중")
            
            # 웜업 추론 실행
            for i in range(num_runs):
                self.trt_server.predict(dummy_input)
                
            logger.info(f"{self.model_name} 모델 웜업 완료")
            
        except Exception as e:
            logger.error(f"{self.model_name} 모델 웜업 중 오류 발생: {e}")
    
    async def release(self) -> None:
        """TensorRT 서버 리소스를 안전하게 해제합니다."""
        if self.trt_server:
            logger.info(f"{self.model_name}용 TensorRT 서버 리소스 해제 중...")
            
            # 비동기 작업이 완료될 때까지 기다리기
            try:
                # 활성 스트림 처리 작업 확인
                pending_tasks = [task for task in asyncio.all_tasks() 
                               if not task.done() and task.get_name().startswith('process_stream')]
                
                if pending_tasks:
                    logger.info(f"{len(pending_tasks)}개의 대기 중인 스트림 처리 작업이 완료될 때까지 대기 중...")
                    try:
                        # 최대 3초 대기
                        done, pending = await asyncio.wait(pending_tasks, timeout=3.0, return_when=asyncio.ALL_COMPLETED)
                        if pending:
                            logger.warning(f"타임아웃 후 {len(pending)}개 작업이 여전히 대기 중")
                    except Exception as e:
                        logger.error(f"작업 대기 중 오류: {e}")
            except Exception as e:
                logger.error(f"대기 중인 작업 처리 중 오류: {e}")
            
            # 메모리 안전하게 해제
            try:
                logger.info(f"{self.model_name}에 대한 안전한 해제 메서드 호출")
                self.trt_server.release(safe_exit=True)
                self.trt_server = None
            except Exception as e:
                logger.error(f"{self.model_name} 서비스 해제 중 오류: {e}")
                # 강제로 참조 제거
                self.trt_server = None
            
            logger.info(f"{self.model_name}용 TensorRT 서버 리소스 해제됨")
    
    def is_initialized(self) -> bool:
        """TensorRT 서버가 초기화되었는지 확인합니다.
        
        Returns:
            bool: 초기화된 경우 True, 그렇지 않으면 False.
        """
        return self.trt_server is not None and hasattr(self.trt_server, '_initialized') and self.trt_server._initialized
    
    def predict(self, input_data: np.ndarray) -> Tuple[Dict[str, np.ndarray], float]:
        """입력 데이터에 대해 추론을 실행합니다.
        
        Args:
            input_data: 모델용으로 전처리된 입력 데이터.
            
        Returns:
            다음을 포함하는 튜플:
                - 출력 텐서 이름에서 numpy 배열로의 딕셔너리
                - 밀리초 단위의 추론 시간
        
        Raises:
            RuntimeError: TensorRT 서버가 초기화되지 않은 경우.
        """
        if not self.is_initialized():
            raise RuntimeError(f"{self.model_name} TensorRT 서버가 초기화되지 않음")
        
        start_time = time.time()
        result, inference_time = self.trt_server.predict(input_data)
        total_time = (time.time() - start_time) * 1000  # ms로 변환
        
        # 성능 메트릭 기록
        if self.enable_metrics:
            self.inference_times.append(inference_time)
            self.inference_count += 1
            self.metrics.record_inference(inference_time, total_time)
            
            # 주기적으로 성능 통계 로깅 (100회 추론마다)
            if self.inference_count % 100 == 0:
                self._log_performance_stats()
            
        return result, inference_time
    
    def _log_performance_stats(self) -> None:
        """성능 통계를 로깅합니다."""
        if not self.inference_times:
            return
            
        avg_time = sum(self.inference_times) / len(self.inference_times)
        max_time = max(self.inference_times)
        min_time = min(self.inference_times)
        
        elapsed_time = time.time() - self.last_reset_time
        throughput = len(self.inference_times) / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(f"{self.model_name} 성능 통계: "
                   f"평균={avg_time:.2f}ms, 최대={max_time:.2f}ms, "
                   f"최소={min_time:.2f}ms, 처리량={throughput:.2f} 추론/초")
        
        # 통계 재설정
        self.inference_times = []
        self.last_reset_time = time.time()
    
    def get_input_details(self) -> Dict[str, Dict[str, Any]]:
        """모델 입력 바인딩에 대한 세부 정보를 가져옵니다.
        
        Returns:
            입력 이름을 세부 정보(shape, dtype)에 매핑하는 딕셔너리.
            
        Raises:
            RuntimeError: TensorRT 서버가 초기화되지 않은 경우.
        """
        if not self.is_initialized():
            raise RuntimeError(f"{self.model_name} TensorRT 서버가 초기화되지 않음")
            
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
        """모델 출력 바인딩에 대한 세부 정보를 가져옵니다.
        
        Returns:
            출력 이름을 세부 정보(shape, dtype)에 매핑하는 딕셔너리.
            
        Raises:
            RuntimeError: TensorRT 서버가 초기화되지 않은 경우.
        """
        if not self.is_initialized():
            raise RuntimeError(f"{self.model_name} TensorRT 서버가 초기화되지 않음")
            
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
        """첫 번째 입력 바인딩의 shape을 가져옵니다.
        
        Returns:
            입력 shape를 나타내는 튜플.
            
        Raises:
            RuntimeError: TensorRT 서버가 초기화되지 않은 경우.
        """
        if not self.is_initialized():
            raise RuntimeError(f"{self.model_name} TensorRT 서버가 초기화되지 않음")
            
        first_input_idx = self.trt_server.input_binding_idxs[0]
        input_name = self.trt_server.engine.get_binding_name(first_input_idx)
        return self.trt_server.binding_shapes[input_name]
    
    def get_first_input_name(self) -> str:
        """첫 번째 입력 바인딩의 이름을 가져옵니다.
        
        Returns:
            첫 번째 입력의 문자열 이름.
            
        Raises:
            RuntimeError: TensorRT 서버가 초기화되지 않은 경우.
        """
        if not self.is_initialized():
            raise RuntimeError(f"{self.model_name} TensorRT 서버가 초기화되지 않음")
            
        first_input_idx = self.trt_server.input_binding_idxs[0]
        return self.trt_server.engine.get_binding_name(first_input_idx)
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델에 대한 정보를 반환합니다.
        
        Returns:
            모델 정보를 포함하는 딕셔너리
        """
        if not self.is_initialized():
            return {
                "model_name": self.model_name,
                "model_path": self.model_path,
                "status": "not_initialized"
            }
            
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "status": "initialized",
            "inputs": self.get_input_details(),
            "outputs": self.get_output_details()
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """모델의 성능 통계를 반환합니다.
        
        Returns:
            성능 통계를 포함하는 딕셔너리
        """
        if not self.enable_metrics or not self.is_initialized():
            return {"status": "metrics_disabled_or_not_initialized"}
            
        return self.metrics.get_statistics()
        
class TensorRTModelManager:
    """여러 TensorRT 모델 관리."""
    
    def __init__(self):
        """TensorRT 모델 관리자를 초기화합니다."""
        self.services = {}
        
    def register_model(self, name: str, model_path: str, enable_metrics: bool = True) -> bool:
        """새 모델을 등록하고 초기화합니다.
        
        Args:
            name: 서비스 이름
            model_path: TensorRT 엔진 파일 경로
            enable_metrics: 성능 지표 수집 여부
            
        Returns:
            bool: 등록 및 초기화가 성공하면 True
        """
        if name in self.services:
            logger.warning(f"모델 '{name}'이(가) 이미 등록되어 있습니다. 먼저 해제하십시오.")
            return False
            
        service = TensorRTService(model_path, enable_metrics)
        success = service.initialize()
        
        if success:
            self.services[name] = service
            logger.info(f"모델 '{name}'이(가) 성공적으로 등록되고 초기화되었습니다.")
        else:
            logger.error(f"모델 '{name}'을(를) 초기화하지 못했습니다.")
            
        return success
        
    def get_service(self, name: str) -> Optional[TensorRTService]:
        """이름으로 서비스를 가져옵니다.
        
        Args:
            name: 서비스 이름
            
        Returns:
            TensorRTService or None: 서비스가 있으면 서비스, 없으면 None
        """
        return self.services.get(name)
        
    async def release_all(self) -> None:
        """모든 등록된 모델을 해제합니다."""
        logger.info("모든 TensorRT 모델 해제 중...")
        
        for name, service in list(self.services.items()):
            try:
                logger.info(f"'{name}' 서비스 해제 중...")
                await service.release()
                del self.services[name]
            except Exception as e:
                logger.error(f"'{name}' 서비스 해제 중 오류: {e}")
                
        logger.info("모든 TensorRT 모델이 해제되었습니다.")
        
    async def release_model(self, name: str) -> bool:
        """특정 모델을 해제합니다.
        
        Args:
            name: 해제할 모델 이름
            
        Returns:
            bool: 성공하면 True
        """
        if name not in self.services:
            logger.warning(f"모델 '{name}'을(를) 찾을 수 없습니다.")
            return False
            
        try:
            await self.services[name].release()
            del self.services[name]
            logger.info(f"모델 '{name}'이(가) 성공적으로 해제되었습니다.")
            return True
        except Exception as e:
            logger.error(f"모델 '{name}' 해제 중 오류: {e}")
            return False
            
    def list_models(self) -> List[Dict[str, Any]]:
        """등록된 모든 모델과 그 상태를 나열합니다.
        
        Returns:
            List[Dict]: 모델 정보 리스트
        """
        return [
            {
                "name": name,
                "status": "initialized" if service.is_initialized() else "not_initialized",
                "model_path": service.model_path
            }
            for name, service in self.services.items()
        ]

# 모델 관리자 생성
model_manager = TensorRTModelManager()

# 기본 모델 등록
default_models = {
    "gauge_detect": DEFAULT_MODEL_PATH,
    "gauge_feature": GAUGE_FEATURE_MODEL_PATH,
    "digital_ocr": DIGITAL_OCR_MODEL_PATH,
    "digital_segmentation": DIGITAL_GAUGE_SEGMENTATION_MODEL_PATH
}

# 이전 버전과의 호환성을 위한 전역 변수
trt_service = None
trt_services = {}

@asynccontextmanager
async def lifespan_handler(app: FastAPI):
    """애플리케이션 라이프사이클 관리.
    
    시작 시 초기화 및 종료 시 정리를 처리합니다.
    
    Args:
        app: FastAPI 애플리케이션 인스턴스.
    """
    global trt_service, trt_services
    
    # 시작 코드(서버 초기화)
    success = True
    
    # 기본 모델 등록
    for name, path in default_models.items():
        model_success = model_manager.register_model(name, path)
        if not model_success:
            logger.error(f"{name} TensorRT 서비스 초기화 실패")
        success = success and model_success
        
        # 이전 버전과의 호환성을 위한 서비스 참조 저장
        if model_success:
            trt_services[name] = model_manager.get_service(name)
    
    # gauge_detect를 기본 서비스로 설정(이전 버전과의 호환성을 위해)
    if "gauge_detect" in trt_services:
        trt_service = trt_services["gauge_detect"]
    
    if not success:
        logger.error("모든 TensorRT 서비스를 초기화하지 못했습니다. 종료 중...")
        sys.exit(1)
    
    logger.info("모든 TensorRT 서비스가 성공적으로 초기화되었습니다.")
    
    yield  # 애플리케이션 실행 중
    
    # 종료 코드(리소스 해제)
    logger.info("모든 TensorRT 서비스 종료 중...")
    await model_manager.release_all()
    logger.info("모든 TensorRT 서비스가 해제되었습니다.")