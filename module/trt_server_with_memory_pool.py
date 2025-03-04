# drag_v2/module/trt_server_with_memory_pool.py

import os
import numpy as np
import tensorrt as trt
import time
import logging
import ctypes
import pycuda.driver as cuda
import pycuda.autoinit
import gc

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TRTMemoryPool:
    """TensorRT용 메모리 풀 클래스"""
    
    def __init__(self, binding_shapes, binding_dtypes):
        """
        메모리 풀 초기화
        
        Args:
            binding_shapes: 바인딩 형상 (딕셔너리: {'binding_name': shape, ...})
            binding_dtypes: 바인딩 데이터 타입 (딕셔너리: {'binding_name': dtype, ...})
        """
        self.device_buffers = {}  # 디바이스(GPU) 메모리 버퍼
        self.host_buffers = {}    # 호스트(CPU) 메모리 버퍼
        self.binding_shapes = binding_shapes
        self.binding_dtypes = binding_dtypes
        
        # 메모리 버퍼 초기화
        self._allocate_buffers()
        
    def _allocate_buffers(self):
        """모든 바인딩에 대한 메모리 할당"""
        for name, shape in self.binding_shapes.items():
            dtype = self.binding_dtypes[name]
            
            # 호스트 메모리 할당
            host_buffer = np.zeros(shape, dtype=dtype)
            self.host_buffers[name] = host_buffer
            
            # 디바이스 메모리 할당
            nbytes = host_buffer.nbytes
            device_buffer = cuda.mem_alloc(nbytes)
            self.device_buffers[name] = device_buffer
            
            logger.info(f"  메모리 풀 할당: {name}, 형상={shape}, 크기={nbytes/1024/1024:.2f} MB")
    
    def get_device_buffer(self, name):
        """디바이스 버퍼 반환"""
        if name not in self.device_buffers:
            raise KeyError(f"바인딩 '{name}'에 대한 디바이스 버퍼가 존재하지 않습니다")
        return self.device_buffers[name]
    
    def get_host_buffer(self, name):
        """호스트 버퍼 반환"""
        if name not in self.host_buffers:
            raise KeyError(f"바인딩 '{name}'에 대한 호스트 버퍼가 존재하지 않습니다")
        return self.host_buffers[name]
    
    def copy_to_device(self, name, data, stream=None):
        """
        호스트에서 디바이스로 데이터 복사
        
        Args:
            name: 바인딩 이름
            data: 복사할 데이터 (NumPy 배열)
            stream: CUDA 스트림 (None이면 동기 복사)
        """
        if name not in self.device_buffers:
            raise KeyError(f"바인딩 '{name}'에 대한 디바이스 버퍼가 존재하지 않습니다")
        
        # 데이터 유효성 검사
        expected_shape = self.binding_shapes[name]
        expected_dtype = self.binding_dtypes[name]
        
        if data.shape != expected_shape:
            logger.warning(f"입력 형상 불일치. 예상: {expected_shape}, 실제: {data.shape}")
            # 형상이 다른 경우 크기 조정
            data = np.resize(data, expected_shape)
        
        if data.dtype != expected_dtype:
            data = data.astype(expected_dtype)
        
        # 호스트 버퍼에 복사
        np.copyto(self.host_buffers[name], data)
        
        # 디바이스로 복사
        if stream:
            # 비동기 복사
            cuda.memcpy_htod_async(self.device_buffers[name], self.host_buffers[name], stream)
        else:
            # 동기 복사
            cuda.memcpy_htod(self.device_buffers[name], self.host_buffers[name])
    
    def copy_from_device(self, name, stream=None):
        """
        디바이스에서 호스트로 데이터 복사
        
        Args:
            name: 바인딩 이름
            stream: CUDA 스트림 (None이면 동기 복사)
            
        Returns:
            data: 복사된 데이터 (NumPy 배열)
        """
        if name not in self.device_buffers:
            raise KeyError(f"바인딩 '{name}'에 대한 디바이스 버퍼가 존재하지 않습니다")
        
        if stream:
            # 비동기 복사
            cuda.memcpy_dtoh_async(self.host_buffers[name], self.device_buffers[name], stream)
        else:
            # 동기 복사
            cuda.memcpy_dtoh(self.host_buffers[name], self.device_buffers[name])
        
        return self.host_buffers[name]
    
    def get_bindings(self):
        """
        바인딩 포인터 목록 반환
        
        Returns:
            bindings: 바인딩 포인터 목록
        """
        return [int(self.device_buffers[name]) for name in self.binding_shapes.keys()]
    
    def release(self):
        """메모리 해제"""
        logger.info("메모리 풀 해제 중...")
        
        # 디바이스 메모리 해제
        for name, buffer in self.device_buffers.items():
            try:
                buffer.free()
                logger.info(f"  디바이스 메모리 해제: {name}")
            except Exception as e:
                logger.error(f"  디바이스 메모리 '{name}' 해제 중 오류: {e}")
        
        self.device_buffers.clear()
        self.host_buffers.clear()
        
        # 가비지 컬렉션 강제 실행
        gc.collect()
        
        logger.info("메모리 풀 해제 완료")
    
    def __del__(self):
        """소멸자"""
        self.release()


class TRTServerWithMemoryPool:
    """
    메모리 풀을 활용한 TensorRT 서버 클래스
    """
    
    def __init__(self, model_path, plugin_path="/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so", use_stream=True):
        """
        TensorRT 서버 초기화
        
        Args:
            model_path: TensorRT 엔진 파일(.trt) 경로
            plugin_path: TensorRT 플러그인 라이브러리 경로
            use_stream: CUDA 스트림 사용 여부 (비동기 처리)
        """
        self.model_path = model_path
        self.use_stream = use_stream
        
        # TensorRT 플러그인 로드
        logger.info("TensorRT 플러그인 로드 중...")
        trt.init_libnvinfer_plugins(None, "")
        if os.path.exists(plugin_path):
            ctypes.CDLL(plugin_path)
            logger.info(f"✅ 플러그인 로드 성공: {plugin_path}")
        
        # TensorRT 로거 생성
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        
        # 엔진 로드
        self._load_engine()
        
        # 바인딩 정보 준비
        self._prepare_bindings()
        
        # CUDA 스트림 생성
        self.cuda_stream = cuda.Stream() if use_stream else None
        logger.info(f"CUDA 스트림 생성 완료 (스트림 사용 {'활성화' if self.use_stream else '비활성화'})")
        
        # 메모리 풀 초기화
        logger.info("메모리 풀 초기화 중...")
        self.memory_pool = TRTMemoryPool(self.binding_shapes, self.binding_dtypes)
        logger.info("메모리 풀 초기화 완료")
    
    def _load_engine(self):
        """TensorRT 엔진 파일 로드"""
        # 파일 존재 확인
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"TensorRT 모델 파일이 존재하지 않습니다: {self.model_path}")
        
        logger.info(f"엔진 파일 로드 중: {self.model_path}")
        
        # 엔진 파일 로드
        with open(self.model_path, "rb") as f:
            engine_data = f.read()
            logger.info(f"엔진 파일 크기: {len(engine_data) / (1024*1024):.2f} MB")
        
        # 직렬화된 엔진에서 런타임 엔진 생성
        self.runtime = trt.Runtime(self.trt_logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        
        if not self.engine:
            raise RuntimeError("TensorRT 엔진 로드 실패")
        
        logger.info("TensorRT 엔진 로드 성공!")
        
        # 실행 컨텍스트 생성
        self.context = self.engine.create_execution_context()
        logger.info("실행 컨텍스트 생성 완료")
    
    def _prepare_bindings(self):
        """바인딩 정보 준비"""
        self.input_binding_idxs = []
        self.output_binding_idxs = []
        self.binding_shapes = {}
        self.binding_dtypes = {}
        self.binding_names = []
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            self.binding_shapes[name] = shape
            self.binding_dtypes[name] = dtype
            self.binding_names.append(name)
            
            if self.engine.binding_is_input(i):
                self.input_binding_idxs.append(i)
                logger.info(f"입력 바인딩: {i}, 이름={name}, 형상={shape}, 타입={dtype}")
            else:
                self.output_binding_idxs.append(i)
                logger.info(f"출력 바인딩: {i}, 이름={name}, 형상={shape}, 타입={dtype}")
    
    def check_engine(self):
        """엔진 정보 출력 및 검증"""
        logger.info("엔진 정보 확인 중...")
        
        try:
            # 엔진 속성 출력
            logger.info(f"엔진 바인딩 수: {self.engine.num_bindings}")
            
            # 최대 배치 크기
            logger.info(f"최대 배치 크기: {self.engine.max_batch_size}")
            
            # 입출력 목록
            inputs = [self.engine.get_binding_name(i) for i in self.input_binding_idxs]
            outputs = [self.engine.get_binding_name(i) for i in self.output_binding_idxs]
            
            logger.info(f"입력 텐서: {inputs}")
            logger.info(f"출력 텐서: {outputs}")
            
            # 메모리 풀 상태 확인
            # DeviceAllocation 객체는 size 속성이 없으므로 메모리 크기는 호스트 버퍼에서 가져옴
            total_device_memory = sum(self.memory_pool.host_buffers[name].nbytes 
                                    for name in self.memory_pool.device_buffers.keys())
            logger.info(f"메모리 풀 상태: {len(self.memory_pool.device_buffers)}개 버퍼, " 
                       f"총 {total_device_memory/(1024*1024):.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"엔진 검증 중 오류: {e}")
            return False
    
    def predict(self, input_data=None):
        """
        모델 추론 실행 (메모리 풀 사용)
        
        Args:
            input_data: 입력 데이터 (None이면 랜덤 데이터 생성)
            
        Returns:
            output_dict: 출력 결과 딕셔너리
            inference_time: 추론 시간(ms)
        """
        try:
            # 입력 데이터 준비 및 디바이스로 복사
            for i in self.input_binding_idxs:
                binding_name = self.engine.get_binding_name(i)
                binding_shape = self.binding_shapes[binding_name]
                binding_dtype = self.binding_dtypes[binding_name]
                
                # 입력 데이터가 제공되지 않았으면 랜덤 데이터 생성
                if input_data is None:
                    # 랜덤 데이터 생성 (사전 할당된 호스트 버퍼에 직접 생성)
                    host_data = np.random.random(binding_shape).astype(binding_dtype)
                    logger.info(f"입력 '{binding_name}': 랜덤 데이터 생성, 형상={binding_shape}")
                else:
                    # 사용자 제공 데이터 사용
                    host_data = input_data
                    # logger.info(f"입력 '{binding_name}': 사용자 데이터 사용, 형상={host_data.shape}")
                
                # 입력 데이터를 메모리 풀에 복사
                self.memory_pool.copy_to_device(binding_name, host_data, self.cuda_stream)
            
            # 추론 실행
            # logger.info("추론 실행 중...")
            start_time = time.time()
            
            # 바인딩 목록 가져오기 (메모리 풀에서)
            bindings = self.memory_pool.get_bindings()
            
            if self.use_stream:
                # 비동기 실행
                self.context.execute_async_v2(
                    bindings=bindings,
                    stream_handle=self.cuda_stream.handle
                )
            else:
                # 동기 실행
                self.context.execute_v2(bindings=bindings)
            
            # 결과 수집
            outputs = {}
            
            for i in self.output_binding_idxs:
                binding_name = self.engine.get_binding_name(i)
                
                # 비동기 메모리 복사 (메모리 풀 사용)
                if self.use_stream:
                    # 비동기 복사
                    self.memory_pool.copy_from_device(binding_name, self.cuda_stream)
                else:
                    # 동기 복사
                    self.memory_pool.copy_from_device(binding_name)
            
            # 스트림 사용 시 스트림 동기화
            if self.use_stream:
                self.cuda_stream.synchronize()
            
            # 출력 데이터 수집
            for i in self.output_binding_idxs:
                binding_name = self.engine.get_binding_name(i)
                outputs[binding_name] = self.memory_pool.get_host_buffer(binding_name).copy()
            
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # ms
            
            # logger.info(f"추론 완료: {inference_time:.2f}ms")
            return outputs, inference_time
            
        except Exception as e:
            logger.error(f"추론 중 오류 발생: {e}", exc_info=True)
            raise
    
    def get_buffer_shapes(self):
        """바인딩 형상 정보 반환"""
        return self.binding_shapes
    
    def get_buffer_dtypes(self):
        """바인딩 데이터 타입 정보 반환"""
        return self.binding_dtypes
    
    def warm_up(self, num_runs=5):
        """
        모델 웜업 실행 (초기 추론 지연 감소)
        
        Args:
            num_runs: 웜업 실행 횟수
        """
        logger.info(f"모델 웜업 시작 ({num_runs}회)...")
        
        times = []
        for i in range(num_runs):
            try:
                _, inference_time = self.predict()
                times.append(inference_time)
                logger.info(f"  웜업 실행 {i+1}: {inference_time:.2f}ms")
            except Exception as e:
                logger.error(f"  웜업 실행 {i+1} 실패: {e}")
        
        if times:
            avg_time = sum(times) / len(times)
            logger.info(f"웜업 완료. 평균 실행 시간: {avg_time:.2f}ms")
    
    def release(self):
        """리소스 해제"""
        logger.info("리소스 해제 중...")
        
        # 메모리 풀 해제
        if hasattr(self, 'memory_pool'):
            self.memory_pool.release()
        
        # TensorRT 리소스 해제
        if hasattr(self, 'context') and self.context:
            del self.context
            self.context = None
        
        if hasattr(self, 'engine') and self.engine:
            del self.engine
            self.engine = None
        
        if hasattr(self, 'runtime') and self.runtime:
            del self.runtime
            self.runtime = None
        
        # 가비지 컬렉션 강제 실행
        gc.collect()
        
        logger.info("리소스 해제 완료")
    
    def __del__(self):
        """소멸자"""
        self.release()


# 사용 예시
if __name__ == "__main__":
    try:
        # 모델 경로
        model_path = "models/trt/20240718_gauge_detection.trt"
        
        # 파일 존재 확인
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            logger.info(f"모델 파일 크기: {file_size:.2f} MB")
            
            # 서버 초기화 (메모리 풀 사용)
            server = TRTServerWithMemoryPool(model_path, use_stream=True)
            
            # 엔진 검증
            server.check_engine()
            
            # 웜업 실행
            server.warm_up(num_runs=3)
            
            # 성능 테스트
            test_runs = 10
            logger.info(f"성능 테스트 시작 ({test_runs}회)...")
            
            times = []
            for i in range(test_runs):
                try:
                    _, inference_time = server.predict()
                    times.append(inference_time)
                    logger.info(f"  실행 {i+1}: {inference_time:.2f}ms")
                except Exception as e:
                    logger.error(f"  실행 {i+1} 실패: {e}")
            
            # 성능 통계
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                
                logger.info("\n=== 성능 통계 ===")
                logger.info(f"평균 추론 시간: {avg_time:.2f}ms")
                logger.info(f"최소 추론 시간: {min_time:.2f}ms")
                logger.info(f"최대 추론 시간: {max_time:.2f}ms")
            
            # 리소스 해제
            server.release()
            
        else:
            logger.error(f"모델 파일이 존재하지 않습니다: {model_path}")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)