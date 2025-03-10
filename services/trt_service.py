"""
효율적인 메모리 관리 및 중복 로드 방지 기능이 포함된 개선된 TensorRT 모델 관리자.
services/trt_service.py에 추가 또는 교체할 코드입니다.
"""
import numpy as np
import time
import asyncio
import os
from typing import Dict, Tuple, List, Any, Optional, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import sys
from datetime import datetime
import gc

from core.config import USE_CUDA_STREAM
from core.config import GAUGE_DETECT_MODEL_PATH as DEFAULT_MODEL_PATH
from core.config import GAUGE_FEATURE_MODEL_PATH, DIGITAL_OCR_MODEL_PATH, DIGITAL_GAUGE_SEGMENTATION_MODEL_PATH
from core.logging import logger
from services.trt_server_wrapper import SafeTRTServerWrapper
from services.metrics import MetricsCollector

class TensorRTServiceRegistry:
    """
    모델 경로 기반 레지스트리를 통해 동일한 모델을 중복 로드하지 않도록 관리하는 싱글톤 클래스
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TensorRTServiceRegistry, cls).__new__(cls)
            cls._instance.path_to_model = {}  # 모델 경로 -> 모델 이름 매핑
            cls._instance.model_ref_count = {}  # 모델 이름 -> 참조 카운트 매핑
        return cls._instance
    
    def register_model_path(self, model_name: str, model_path: str) -> str:
        """
        모델 경로를 레지스트리에 등록하고 해당 경로에 대한 참조 모델 이름 반환
        
        Args:
            model_name: 등록할 모델 이름
            model_path: 모델 파일 경로
            
        Returns:
            str: 해당 경로에 대응하는 모델 이름 (이미 등록된 경우 기존 이름 반환)
        """
        # 이미 경로가 등록되어 있는지 확인
        if model_path in self.path_to_model:
            existing_name = self.path_to_model[model_path]
            # 참조 카운트 증가
            self.model_ref_count[existing_name] += 1
            logger.info(f"모델 경로 '{model_path}'는 이미 '{existing_name}'으로 등록되어 있습니다. 참조 카운트: {self.model_ref_count[existing_name]}")
            return existing_name
        
        # 새 모델 등록
        self.path_to_model[model_path] = model_name
        self.model_ref_count[model_name] = 1
        logger.info(f"새 모델 '{model_name}' 경로 등록: {model_path}")
        return model_name
    
    def unregister_model(self, model_name: str) -> bool:
        """
        모델 참조 해제. 참조 카운트가 0이 되면 경로 매핑도 제거
        
        Args:
            model_name: 등록 해제할 모델 이름
            
        Returns:
            bool: 성공적으로 등록 해제되었으면 True
        """
        if model_name not in self.model_ref_count:
            logger.warning(f"모델 '{model_name}'은 레지스트리에 등록되어 있지 않습니다.")
            return False
        
        # 참조 카운트 감소
        self.model_ref_count[model_name] -= 1
        logger.info(f"모델 '{model_name}' 참조 카운트 감소: {self.model_ref_count[model_name]}")
        
        # 참조 카운트가 0이면 경로 매핑 제거
        if self.model_ref_count[model_name] <= 0:
            # 경로 매핑 제거
            paths_to_remove = [path for path, name in self.path_to_model.items() if name == model_name]
            for path in paths_to_remove:
                del self.path_to_model[path]
            
            # 참조 카운트 제거
            del self.model_ref_count[model_name]
            logger.info(f"모델 '{model_name}' 레지스트리에서 완전히 제거됨")
        
        return True
    
    def get_actual_model_name(self, model_path: str) -> Optional[str]:
        """
        모델 경로에 대응하는 실제 모델 이름 반환
        
        Args:
            model_path: 모델 파일 경로
            
        Returns:
            str or None: 등록된 모델 이름 또는 None
        """
        return self.path_to_model.get(model_path)
    
    def get_ref_count(self, model_name: str) -> int:
        """
        모델의 현재 참조 카운트 반환
        
        Args:
            model_name: 모델 이름
            
        Returns:
            int: 참조 카운트 (등록되지 않은 경우 0)
        """
        return self.model_ref_count.get(model_name, 0)
    
    def is_registered(self, model_path: str) -> bool:
        """
        모델 경로가 레지스트리에 등록되어 있는지 확인
        
        Args:
            model_path: 모델 파일 경로
            
        Returns:
            bool: 등록되어 있으면 True
        """
        return model_path in self.path_to_model

# 싱글톤 레지스트리 인스턴스
model_registry = TensorRTServiceRegistry()

class TensorRTService:
    """TensorRT 엔진 및 추론을 위한 서비스."""
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, enable_metrics: bool = True, model_name: str = None):
        """TensorRT 서비스를 초기화합니다.
        
        Args:
            model_path: TensorRT 엔진 파일 경로
            enable_metrics: 성능 지표 수집 여부
            model_name: 모델 이름 (None이면 경로에서 추출)
        """
        self.trt_server = None
        self.model_path = model_path
        self.use_cuda_stream = USE_CUDA_STREAM
        
        # 모델 파일 존재 확인
        if not os.path.exists(model_path):
            logger.error(f"모델 파일을 찾을 수 없음: {model_path}")
            self._initialized = False
            return
        
        # 모델 이름이 제공되지 않은 경우 파일 이름 사용
        if model_name is None:
            self.model_name = os.path.basename(model_path).split('.')[0]
        else:
            self.model_name = model_name
        
        self.enable_metrics = enable_metrics
        self._initialized = False
        
        # 레지스트리에 등록
        self.registered_name = model_registry.register_model_path(self.model_name, self.model_path)
        
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
            # 이미 초기화된 경우
            if self._initialized and self.trt_server:
                logger.info(f"{self.model_name} 모델이 이미 초기화되어 있습니다.")
                return True
            
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

            self._initialized = True
            logger.info(f"{self.model_name} TensorRT 서버 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"{self.model_name} TensorRT 서버 초기화 오류: {e}")
            self._initialized = False
            if self.trt_server:
                self.trt_server.release(safe_exit=True)
                self.trt_server = None
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

    # trt_service.py에서 종료 처리 개선
    async def release(self) -> None:
        """TensorRT 서버 리소스를 안전하게 해제합니다."""
        if not self._initialized or self.trt_server is None:
            return
        
        logger.info(f"{self.model_name}용 TensorRT 서버 리소스 해제 중...")
        
        try:
            # 모든 대기 중인 작업 완료 대기
            pending_tasks = [task for task in asyncio.all_tasks() 
                        if not task.done() and 'process_stream' in task.get_name()]
            
            if pending_tasks:
                logger.info(f"{len(pending_tasks)}개의 대기 중인 작업 완료 대기...")
                await asyncio.wait(pending_tasks, timeout=2.0)
            
            # 명시적 CUDA 동기화
            try:
                import pycuda.driver as cuda
                cuda.Context.synchronize()
            except Exception as e:
                logger.warning(f"CUDA 동기화 실패: {e}")
            
            # 서버 래퍼에 안전한 종료 신호 전달
            self.trt_server.release(safe_exit=True)
            self.trt_server = None
            
            # 최종 가비지 컬렉션 강제 실행
            gc.collect()
        except Exception as e:
            logger.error(f"리소스 해제 중 오류: {e}")
            self.trt_server = None
        
        self._initialized = False
    
    def is_initialized(self) -> bool:
        """TensorRT 서버가 초기화되었는지 확인합니다.
        
        Returns:
            bool: 초기화된 경우 True, 그렇지 않으면 False.
        """
        return self._initialized and self.trt_server is not None and hasattr(self.trt_server, '_initialized') and self.trt_server._initialized
    
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
            
        # 서버에서 직접 바인딩 형상 정보 얻기
        first_input_idx = self.trt_server.input_binding_idxs[0]
        input_name = self.trt_server.engine.get_binding_name(first_input_idx)
        
        # 바인딩 형상 반환
        return tuple(self.trt_server.binding_shapes[input_name])

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
        # 이미 등록된 서비스가 있고 초기화된 경우 성공 반환
        if name in self.services and self.services[name].is_initialized():
            logger.info(f"모델 '{name}'이(가) 이미 등록되고 초기화되어 있습니다.")
            return True
        
        # 이미 경로에 해당하는 모델이 로드되어 있는지 확인
        actual_model_name = model_registry.get_actual_model_name(model_path)
        if actual_model_name and actual_model_name != name:
            logger.info(f"모델 경로 '{model_path}'가 이미 '{actual_model_name}'로 로드되어 있습니다. '{name}'과 연결합니다.")
            
            # 동일한 경로에 대한 다른 서비스 인스턴스 찾기
            for existing_name, service in self.services.items():
                if service.model_path == model_path and service.is_initialized():
                    # 기존 서비스를 새 이름으로 등록
                    self.services[name] = service
                    logger.info(f"기존 로드된 모델을 '{name}' 이름으로 추가 등록했습니다.")
                    return True
        
        # 모델이 등록되어 있지만 초기화되지 않은 경우 재초기화 시도
        if name in self.services:
            logger.info(f"모델 '{name}'이(가) 등록되어 있지만 초기화되지 않았습니다. 재초기화 시도 중...")
            success = self.services[name].initialize()
            if success:
                logger.info(f"모델 '{name}'이(가) 성공적으로 재초기화되었습니다.")
            else:
                logger.error(f"모델 '{name}'을(를) 재초기화하지 못했습니다.")
                # 초기화 실패 시 서비스 제거
                del self.services[name]
            return success
        
        # 새로운 모델 생성 및 초기화
        service = TensorRTService(model_path, enable_metrics, model_name=name)
        success = service.initialize()
        
        if success:
            self.services[name] = service
            logger.info(f"모델 '{name}'이(가) 성공적으로 등록되고 초기화되었습니다.")
        else:
            logger.error(f"모델 '{name}'을(를) 초기화하지 못했습니다.")
            
        return success
    
    def get_or_create_service(self, name: str, model_path: str, enable_metrics: bool = True) -> Optional[TensorRTService]:
        """이름으로 서비스를 가져오거나, 없으면 생성합니다.
        
        Args:
            name: 서비스 이름
            model_path: 없을 경우 생성에 사용할 모델 경로
            enable_metrics: 성능 지표 수집 여부
            
        Returns:
            TensorRTService or None: 서비스 인스턴스 또는 생성 실패 시 None
        """
        # 이미 등록된 서비스가 있고 초기화된 경우 반환
        service = self.get_service(name)
        if service and service.is_initialized():
            return service
        
        # 없거나 초기화되지 않은 경우 등록 시도
        if self.register_model(name, model_path, enable_metrics):
            return self.get_service(name)
        
        return None
        
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
            except Exception as e:
                logger.error(f"'{name}' 서비스 해제 중 오류: {e}")
                
        # 서비스 목록 비우기
        self.services.clear()
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
        models_info = []
        
        for name, service in self.services.items():
            info = {
                "name": name,
                "status": "initialized" if service.is_initialized() else "not_initialized",
                "model_path": service.model_path,
                "ref_count": model_registry.get_ref_count(service.registered_name)
            }
            
            # 성능 지표 추가 (있는 경우)
            if service.enable_metrics and hasattr(service, 'metrics'):
                stats = service.metrics.get_statistics()
                if stats and stats.get("status") != "no_data":
                    info["performance"] = {
                        "avg_inference_time_ms": stats["inference_time_ms"]["avg"],
                        "throughput": stats["throughput"],
                        "total_inferences": stats["total_inferences"]
                    }
            
            models_info.append(info)
            
        return models_info


# 순차적 모델 초기화 함수 - CUDA 컨텍스트 충돌 방지
async def initialize_models(model_manager: TensorRTModelManager, models_config: Dict[str, str]) -> bool:
    """
    여러 모델을 순차적으로(한 번에 하나씩) 초기화
    
    Args:
        model_manager: 모델 관리자 인스턴스
        models_config: 모델 이름과 경로 매핑 딕셔너리
        
    Returns:
        bool: 핵심 모델이 성공적으로 초기화되면 True
    """
    logger.info("모든 TensorRT 모델 순차적으로 초기화 중...")
    
    # 각 모델을 하나씩 순차적으로 초기화
    for name, path in models_config.items():
        # 이미 초기화된 모델이면 건너뛰기
        if name in model_manager.services and model_manager.services[name].is_initialized():
            logger.info(f"모델 '{name}'이(가) 이미 초기화되어 있습니다")
            continue
            
        # 한 번에 하나의 모델만 로드하여 CUDA 컨텍스트 충돌 방지
        logger.info(f"모델 '{name}' 초기화 중...")
        try:
            success = model_manager.register_model(name, path)
            if success:
                logger.info(f"모델 '{name}' 초기화 성공")
            else:
                logger.error(f"모델 '{name}' 초기화 실패")
        except Exception as e:
            logger.error(f"모델 '{name}' 초기화 중 오류: {e}")
        
        # 각 모델 초기화 사이에 짧은 지연
        await asyncio.sleep(0.5)
    
    # 초기화 결과 확인
    initialized_models = [name for name, service in model_manager.services.items() 
                         if service.is_initialized()]
    logger.info(f"초기화된 모델: {initialized_models}")
    
    return len(initialized_models) > 0

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

# 기존 lifespan_handler에서 사용할 비동기 초기화 함수로 대체
@asynccontextmanager
async def lifespan_handler(app: FastAPI):
    """애플리케이션 라이프사이클 관리.
    
    시작 시 초기화 및 종료 시 정리를 처리합니다.
    
    Args:
        app: FastAPI 애플리케이션 인스턴스.
    """
    global trt_service, trt_services
    
    # 시작 코드(서버 초기화)
    
    # 기본 모델 등록
    success = await initialize_models(model_manager, default_models)
    
    # 이전 버전과의 호환성을 위한 서비스 참조 저장
    for name, service in model_manager.services.items():
        if service.is_initialized():
            trt_services[name] = service
    
    # gauge_detect를 기본 서비스로 설정(이전 버전과의 호환성을 위해)
    if "gauge_detect" in trt_services:
        trt_service = trt_services["gauge_detect"]
    
    if not success:
        logger.warning("일부 TensorRT 서비스를 초기화하지 못했습니다. 제한된 기능으로 계속 실행됩니다.")
        # 중요한 모델이 없으면 종료, 그렇지 않으면 계속 실행
        if "gauge_detect" not in trt_services or "gauge_feature" not in trt_services:
            logger.error("핵심 모델을 초기화하지 못했습니다. 종료 중...")
            sys.exit(1)
    else:
        logger.info("모든 TensorRT 서비스가 성공적으로 초기화되었습니다.")
    
    yield  # 애플리케이션 실행 중
    
    # 종료 코드(리소스 해제)
    logger.info("애플리케이션 종료 - 모든 리소스 정리 시작...")
    
    # 1. 모든 스트림 비활성화 및 정리
    try:
        from api.endpoints.streaming import active_streams, clean_stream_resources
        for stream_id in list(active_streams.keys()):
            logger.info(f"스트림 '{stream_id}' 비활성화")
            active_streams[stream_id] = False
            clean_stream_resources(stream_id)
        
        # 스트림 서비스 정리
        from services.stream_service import stream_service
        stream_service.cleanup()
    except Exception as e:
        logger.error(f"스트림 정리 중 오류: {e}")
    
    # 2. 최대 0.5초 대기
    await asyncio.sleep(0.5)
    
    # 3. 모든 모델 서비스 해제
    await model_manager.release_all()
    
    # 4. CUDA 컨텍스트 동기화 및 닫기
    try:
        import pycuda.driver as cuda
        cuda.Context.synchronize()
    except Exception as e:
        logger.warning(f"CUDA 동기화 중 오류: {e}")
    
    # 5. 메모리 정리
    trt_service = None
    trt_services = {}
    
    import gc
    gc.collect()
    
    logger.info("모든 TensorRT 서비스가 해제되었습니다.")