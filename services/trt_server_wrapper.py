"""
TensorRT 서버 래퍼 - 안전한 메모리 관리를 위한 추가 래퍼 레이어
"""
import time
import numpy as np
import gc
import threading
import signal
from typing import Dict, Tuple, Optional, Any

from core.logging import logger
from module.trt_server_with_memory_pool import TRTServerWithMemoryPool

class SafeTRTServerWrapper:
    """
    TensorRT 서버 래퍼 클래스 - Segmentation Fault 예방을 위한 안전한 메모리 관리
    """
    
    def __init__(self, model_path: str, use_stream: bool = True):
        """
        래퍼 초기화
        
        Args:
            model_path: TensorRT 엔진 파일 경로
            use_stream: CUDA 스트림 사용 여부
        """
        self.trt_server = None
        self.model_path = model_path
        self.use_stream = use_stream
        self.resources_lock = threading.Lock()  # 리소스 접근 동기화를 위한 락
        self._initialized = False
        self._prediction_count = 0  # 추론 횟수 추적
        
        # 원래 시그널 핸들러 저장
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        self._original_sigint = signal.getsignal(signal.SIGINT)
        
    def _setup_signal_handlers(self):
        """안전한 종료를 위한 시그널 핸들러 설정"""
        signal.signal(signal.SIGTERM, self._handle_exit_signal)
        signal.signal(signal.SIGINT, self._handle_exit_signal)
    
    def _restore_signal_handlers(self):
        """원래 시그널 핸들러 복원"""
        signal.signal(signal.SIGTERM, self._original_sigterm)
        signal.signal(signal.SIGINT, self._original_sigint)
        
    def _handle_exit_signal(self, signum, frame):
        """종료 시그널을 적절히 처리"""
        logger.info(f"종료 시그널 {signum} 수신, 안전하게 리소스 정리 중...")
        self.release(safe_exit=True)
        
        # 원래 핸들러 호출
        if signum == signal.SIGTERM and callable(self._original_sigterm):
            self._original_sigterm(signum, frame)
        elif signum == signal.SIGINT and callable(self._original_sigint):
            self._original_sigint(signum, frame)
    
    def initialize(self) -> bool:
        """
        TensorRT 서버 초기화
        
        Returns:
            초기화 성공 여부
        """
        try:
            with self.resources_lock:
                if self._initialized:
                    logger.info("TensorRT 서버가 이미 초기화되었습니다")
                    return True
                
                logger.info(f"TensorRT 서버 초기화... 모델: {self.model_path}")
                self.trt_server = TRTServerWithMemoryPool(self.model_path, use_stream=self.use_stream)
                
                # 엔진 검증
                if not self.trt_server.check_engine():
                    raise RuntimeError("TensorRT 엔진 검증 실패")
                
                # 웜업 실행
                self.trt_server.warm_up(num_runs=3)
                
                # 시그널 핸들러 설정
                self._setup_signal_handlers()
                
                self._initialized = True
                logger.info("TensorRT 서버 초기화 완료")
                return True
                
        except Exception as e:
            logger.error(f"TensorRT 서버 초기화 중 오류: {e}")
            self.release(safe_exit=True)  # 실패 시 리소스 정리
            return False
    
    def release(self, safe_exit: bool = False) -> None:
        """
        TensorRT 서버 리소스 해제
        
        Args:
            safe_exit: 안전한 종료 모드 여부
        """
        with self.resources_lock:
            if not self._initialized or self.trt_server is None:
                logger.info("TensorRT 서버가 초기화되지 않았거나 이미 해제되었습니다")
                return
            
            try:
                logger.info("TensorRT 서버 리소스 해제 시작...")
                
                # 시그널 핸들러 복원
                if safe_exit:
                    self._restore_signal_handlers()
                
                # 안전한 리소스 해제 시퀀스
                try:
                    # 메모리 풀 참조를 분리하기 전 CUDA 동기화
                    if hasattr(self.trt_server, 'cuda_stream') and self.trt_server.cuda_stream:
                        try:
                            import pycuda.driver as cuda
                            cuda.Context.synchronize()
                            logger.info("CUDA 컨텍스트 동기화 완료")
                        except (ImportError, Exception) as e:
                            logger.warning(f"CUDA 컨텍스트 동기화 실패: {e}")
                    
                    # 각 필드를 명시적으로 None으로 설정하여 해제
                    if hasattr(self.trt_server, 'release'):
                        self.trt_server.release()
                except Exception as e:
                    logger.error(f"TensorRT 서버 릴리스 중 오류: {e}")
                
                # 명시적 메모리 정리
                self.trt_server = None
                gc.collect()
                
                # 20ms 대기하여 리소스가 적절히 해제되도록 함
                time.sleep(0.02)
                
                self._initialized = False
                logger.info("TensorRT 서버 리소스 해제 완료")
            
            except Exception as e:
                logger.error(f"리소스 해제 중 예외 발생: {e}")
                # 최후의 수단으로 변수 초기화
                self.trt_server = None
                self._initialized = False
    
    def predict(self, input_data: np.ndarray) -> Tuple[Dict[str, np.ndarray], float]:
        """
        입력 데이터로 추론 실행
        
        Args:
            input_data: 전처리된 입력 데이터
            
        Returns:
            추론 결과와 추론 시간(ms)
            
        Raises:
            RuntimeError: TensorRT 서버가 초기화되지 않은 경우
        """
        if not self._initialized or self.trt_server is None:
            raise RuntimeError("TensorRT 서버가 초기화되지 않았습니다")
            
        # 리소스 락을 사용하지 않고 추론만 실행
        # 추론 자체는 TensorRT 내부에서 동기화됨
        result = self.trt_server.predict(input_data)
        
        # 추론 횟수 업데이트
        with self.resources_lock:
            self._prediction_count += 1
            
            # 안정성을 위해 1000회 추론마다 가비지 컬렉션 실행
            if self._prediction_count % 1000 == 0:
                gc.collect()
        
        return result
    
    def check_engine(self) -> bool:
        """엔진 유효성 확인"""
        if not self._initialized or self.trt_server is None:
            return False
        return self.trt_server.check_engine()
    
    def warm_up(self, num_runs: int = 3) -> None:
        """모델 웜업"""
        if not self._initialized or self.trt_server is None:
            raise RuntimeError("TensorRT 서버가 초기화되지 않았습니다")
        self.trt_server.warm_up(num_runs)
    
    def get_buffer_shapes(self) -> Dict:
        """입출력 버퍼 형상 정보 반환"""
        if not self._initialized or self.trt_server is None:
            raise RuntimeError("TensorRT 서버가 초기화되지 않았습니다")
        return self.trt_server.get_buffer_shapes()
    
    def get_buffer_dtypes(self) -> Dict:
        """입출력 버퍼 데이터 타입 정보 반환"""
        if not self._initialized or self.trt_server is None:
            raise RuntimeError("TensorRT 서버가 초기화되지 않았습니다")
        return self.trt_server.get_buffer_dtypes()
    
    @property
    def engine(self):
        """TensorRT 엔진 인스턴스 반환"""
        if not self._initialized or self.trt_server is None:
            raise RuntimeError("TensorRT 서버가 초기화되지 않았습니다")
        return self.trt_server.engine
    
    @property
    def binding_shapes(self) -> Dict:
        """바인딩 형상 정보 반환"""
        if not self._initialized or self.trt_server is None:
            raise RuntimeError("TensorRT 서버가 초기화되지 않았습니다")
        return self.trt_server.binding_shapes
    
    @property
    def input_binding_idxs(self) -> list:
        """입력 바인딩 인덱스 반환"""
        if not self._initialized or self.trt_server is None:
            raise RuntimeError("TensorRT 서버가 초기화되지 않았습니다")
        return self.trt_server.input_binding_idxs
    
    @property
    def output_binding_idxs(self) -> list:
        """출력 바인딩 인덱스 반환"""
        if not self._initialized or self.trt_server is None:
            raise RuntimeError("TensorRT 서버가 초기화되지 않았습니다")
        return self.trt_server.output_binding_idxs