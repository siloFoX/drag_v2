# drag_v2/module/trt_server.py

import os
import numpy as np
import tensorrt as trt
import time
import logging
import ctypes
import pycuda.driver as cuda
import pycuda.autoinit

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TRTServer:
    """
    TensorRT 서버 클래스 (스트림 처리 지원)
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
        
        # TensorRT 로거 생성 (INFO 레벨로 변경)
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        
        # 엔진 로드
        self._load_engine()
        
        # 입력 및 출력 바인딩 인덱스 저장
        self.input_binding_idxs = []
        self.output_binding_idxs = []
        self.binding_shapes = {}
        self.binding_dtypes = {}
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            self.binding_shapes[name] = shape
            self.binding_dtypes[name] = dtype
            
            if self.engine.binding_is_input(i):
                self.input_binding_idxs.append(i)
                logger.info(f"입력 바인딩: {i}, 이름={name}, 형상={shape}, 타입={dtype}")
            else:
                self.output_binding_idxs.append(i)
                logger.info(f"출력 바인딩: {i}, 이름={name}, 형상={shape}, 타입={dtype}")
        
        # CUDA 스트림 생성
        self.cuda_stream = cuda.Stream()
        logger.info(f"CUDA 스트림 생성 완료 (스트림 사용 {'활성화' if self.use_stream else '비활성화'})")
    
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
            
            return True
            
        except Exception as e:
            logger.error(f"엔진 검증 중 오류: {e}")
            return False
    
    def predict(self, input_data=None):
        """
        모델 추론 실행
        
        Args:
            input_data: 입력 데이터 (None이면 랜덤 데이터 생성)
            
        Returns:
            output_dict: 출력 결과 딕셔너리
            inference_time: 추론 시간(ms)
        """
        # 디바이스 메모리 버퍼 리스트
        device_buffers = []
        host_buffers = []
        bindings = []
        
        try:
            # 바인딩 메모리 초기화 (엔진의 모든 바인딩에 대해)
            for i in range(self.engine.num_bindings):
                binding_name = self.engine.get_binding_name(i)
                binding_shape = self.engine.get_binding_shape(i)
                binding_dtype = self.binding_dtypes[binding_name]
                
                # 입력 바인딩인 경우
                if self.engine.binding_is_input(i):
                    # 입력 데이터가 제공되지 않았으면 랜덤 데이터 생성
                    if input_data is None:
                        host_data = np.random.random(binding_shape).astype(binding_dtype)
                        logger.info(f"입력 '{binding_name}': 랜덤 데이터 생성, 형상={binding_shape}")
                    else:
                        host_data = input_data
                        logger.info(f"입력 '{binding_name}': 사용자 데이터 사용, 형상={host_data.shape}")
                    
                    # 입력 데이터의 형태 확인 및 조정
                    if host_data.shape != binding_shape:
                        logger.warning(f"입력 형상 불일치. 예상: {binding_shape}, 실제: {host_data.shape}")
                        host_data = np.resize(host_data, binding_shape)
                    
                    if host_data.dtype != binding_dtype:
                        host_data = host_data.astype(binding_dtype)
                    
                    # 디바이스 메모리 할당
                    device_mem = cuda.mem_alloc(host_data.nbytes)
                    
                    # 데이터를 디바이스로 복사 (동기 또는 비동기)
                    if self.use_stream:
                        cuda.memcpy_htod_async(device_mem, host_data, self.cuda_stream)
                    else:
                        cuda.memcpy_htod(device_mem, host_data)
                else:
                    # 출력 바인딩인 경우, 출력을 저장할 메모리 할당
                    host_data = np.zeros(binding_shape, dtype=binding_dtype)
                    device_mem = cuda.mem_alloc(host_data.nbytes)
                
                # 바인딩 목록 업데이트
                bindings.append(int(device_mem))
                device_buffers.append(device_mem)
                host_buffers.append(host_data)
            
            # 추론 실행
            logger.info("추론 실행 중...")
            start_time = time.time()
            
            if self.use_stream:
                # 비동기 실행 (스트림 사용)
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
                name = self.engine.get_binding_name(i)
                idx = i  # 바인딩 인덱스
                
                # 디바이스에서 호스트로 결과 복사
                if self.use_stream:
                    # 비동기 복사
                    cuda.memcpy_dtoh_async(host_buffers[idx], device_buffers[idx], self.cuda_stream)
                else:
                    # 동기 복사
                    cuda.memcpy_dtoh(host_buffers[idx], device_buffers[idx])
            
            # 스트림 사용 시 스트림 동기화 (수정된 부분)
            if self.use_stream:
                # 스트림 동기화 - 타임아웃 매개변수 없이 호출
                self.cuda_stream.synchronize()
            
            # 모든 출력 복사
            for i in self.output_binding_idxs:
                name = self.engine.get_binding_name(i)
                idx = i  # 바인딩 인덱스
                outputs[name] = host_buffers[idx].copy()
            
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # ms로 변환
            
            logger.info(f"추론 완료: {inference_time:.2f}ms")
            return outputs, inference_time
            
        except Exception as e:
            logger.error(f"추론 중 오류 발생: {e}", exc_info=True)
            raise
        
        finally:
            # CUDA 메모리 해제 (PyCUDA는 자동으로 처리하지만, 명시적으로 해제)
            for buf in device_buffers:
                try:
                    buf.free()
                except:
                    pass
    
    def predict_batch(self, batch_input_data, batch_size=None):
        """
        배치 추론 실행
        
        Args:
            batch_input_data: 입력 데이터 배치 (리스트 또는 배치 차원이 포함된 배열)
            batch_size: 배치 크기 (None이면 입력 데이터에서 추정)
            
        Returns:
            batch_outputs: 배치 출력 결과 (배치별 결과 리스트)
            total_time: 총 추론 시간(ms)
        """
        if batch_size is None and hasattr(batch_input_data, "shape"):
            batch_size = batch_input_data.shape[0]
        elif batch_size is None:
            batch_size = len(batch_input_data)
        
        logger.info(f"배치 추론 시작: 크기={batch_size}")
        
        batch_outputs = []
        start_time = time.time()
        
        # 엔진의 최대 배치 크기 확인
        engine_max_batch = self.engine.max_batch_size
        if engine_max_batch > 0 and batch_size > engine_max_batch:
            logger.warning(f"요청된 배치 크기({batch_size})가 엔진 최대 배치 크기({engine_max_batch})보다 큽니다.")
            logger.warning(f"배치를 {engine_max_batch} 크기 단위로 분할합니다.")
        
        # 엔진이 배치 처리를 지원하는 경우
        if engine_max_batch > 1:
            # 여기서 배치 처리 로직 구현
            # 실제 구현은 TensorRT 엔진의 배치 처리 방식에 따라 달라질 수 있음
            pass
        else:
            # 배치 처리가 지원되지 않는 경우, 순차적으로 처리
            for i in range(batch_size):
                try:
                    if hasattr(batch_input_data, "shape"):
                        # NumPy 배열 또는 유사한 형태
                        input_data = batch_input_data[i]
                    else:
                        # 리스트 또는 기타 시퀀스
                        input_data = batch_input_data[i]
                    
                    outputs, _ = self.predict(input_data)
                    batch_outputs.append(outputs)
                    
                except Exception as e:
                    logger.error(f"배치 항목 {i} 추론 중 오류: {e}")
                    # 오류가 발생한 항목에 대해 None 추가
                    batch_outputs.append(None)
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # ms로 변환
        logger.info(f"배치 추론 완료: {batch_size}개 항목, 총 시간={total_time:.2f}ms, 항목당 평균={total_time/batch_size:.2f}ms")
        
        return batch_outputs, total_time
    
    def release(self):
        """리소스 해제"""
        logger.info("리소스 해제 중...")
        
        if hasattr(self, 'context') and self.context:
            del self.context
            self.context = None
        
        if hasattr(self, 'engine') and self.engine:
            del self.engine
            self.engine = None
        
        if hasattr(self, 'runtime') and self.runtime:
            del self.runtime
            self.runtime = None
        
        logger.info("리소스 해제 완료")
    
    def __del__(self):
        """소멸자"""
        self.release()


def test_stream_performance(model_path):
    """스트림 사용 여부에 따른 성능 비교 테스트"""
    logger.info(f"== 스트림 성능 비교 테스트: {model_path} ==")
    
    # 스트림 비사용 서버
    logger.info("1. 스트림 비사용 모드로 초기화")
    non_stream_server = TRTServer(model_path, use_stream=False)
    
    if not non_stream_server.check_engine():
        logger.error("엔진 검증 실패")
        return
    
    # 스트림 사용 서버
    logger.info("2. 스트림 사용 모드로 초기화")
    stream_server = TRTServer(model_path, use_stream=True)
    
    # 테스트 횟수
    test_runs = 10
    
    # 비스트림 모드 테스트
    logger.info(f"비스트림 모드 테스트 ({test_runs}회)...")
    non_stream_times = []
    
    for i in range(test_runs):
        try:
            _, inference_time = non_stream_server.predict()
            non_stream_times.append(inference_time)
            logger.info(f"  실행 {i+1}: {inference_time:.2f}ms")
        except Exception as e:
            logger.error(f"  실행 {i+1} 실패: {e}")
    
    # 스트림 모드 테스트
    logger.info(f"스트림 모드 테스트 ({test_runs}회)...")
    stream_times = []
    
    for i in range(test_runs):
        try:
            _, inference_time = stream_server.predict()
            stream_times.append(inference_time)
            logger.info(f"  실행 {i+1}: {inference_time:.2f}ms")
        except Exception as e:
            logger.error(f"  실행 {i+1} 실패: {e}")
    
    # 결과 분석
    if non_stream_times and stream_times:
        avg_non_stream = sum(non_stream_times) / len(non_stream_times)
        avg_stream = sum(stream_times) / len(stream_times)
        
        logger.info("\n=== 성능 비교 결과 ===")
        logger.info(f"비스트림 모드 평균: {avg_non_stream:.2f}ms")
        logger.info(f"스트림 모드 평균: {avg_stream:.2f}ms")
        
        if avg_stream < avg_non_stream:
            improvement = (1 - avg_stream / avg_non_stream) * 100
            logger.info(f"스트림 모드가 {improvement:.2f}% 더 빠름")
        else:
            deterioration = (avg_stream / avg_non_stream - 1) * 100
            logger.info(f"스트림 모드가 {deterioration:.2f}% 더 느림")
    
    # 리소스 해제
    non_stream_server.release()
    stream_server.release()
    
    logger.info("== 테스트 완료 ==")


if __name__ == "__main__":
    try:
        # 모델 경로
        model_path = "models/20240718_gauge_detection.trt"
        
        # 파일 존재 확인
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            logger.info(f"모델 파일 크기: {file_size:.2f} MB")
            
            # 스트림 성능 테스트
            test_stream_performance(model_path)
        else:
            logger.error(f"모델 파일이 존재하지 않습니다: {model_path}")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)