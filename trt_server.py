import os
import numpy as np
import tensorrt as trt
import time
import logging
import ctypes
import pycuda.driver as cuda
import pycuda.autoinit
import gc
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TRTServer:
    """
    TensorRT 서버 클래스
    """
    
    def __init__(self, model_path, model_name=None, plugin_path="/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so"):
        """
        TensorRT 서버 초기화
        
        Args:
            model_path: TensorRT 엔진 파일(.trt) 경로
            model_name: 모델 이름 (로깅용)
            plugin_path: TensorRT 플러그인 라이브러리 경로
        """
        self.model_path = model_path
        self.model_name = model_name or os.path.basename(model_path)
        
        # TensorRT 플러그인 로드 (정적 메서드로 한 번만 실행되도록 함)
        if not hasattr(TRTServer, '_plugins_loaded'):
            logger.info("TensorRT 플러그인 로드 중...")
            trt.init_libnvinfer_plugins(None, "")
            if os.path.exists(plugin_path):
                ctypes.CDLL(plugin_path)
                logger.info(f"✅ 플러그인 로드 성공: {plugin_path}")
            TRTServer._plugins_loaded = True
        
        # TensorRT 로거 생성
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
                logger.info(f"[{self.model_name}] 입력 바인딩: {i}, 이름={name}, 형상={shape}, 타입={dtype}")
            else:
                self.output_binding_idxs.append(i)
                logger.info(f"[{self.model_name}] 출력 바인딩: {i}, 이름={name}, 형상={shape}, 타입={dtype}")
        
        # CUDA 스트림 생성
        self.cuda_stream = cuda.Stream()
    
    def _load_engine(self):
        """TensorRT 엔진 파일 로드"""
        # 파일 존재 확인
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"TensorRT 모델 파일이 존재하지 않습니다: {self.model_path}")
        
        logger.info(f"[{self.model_name}] 엔진 파일 로드 중: {self.model_path}")
        
        # 엔진 파일 로드
        with open(self.model_path, "rb") as f:
            engine_data = f.read()
            logger.info(f"[{self.model_name}] 엔진 파일 크기: {len(engine_data) / (1024*1024):.2f} MB")
        
        # 직렬화된 엔진에서 런타임 엔진 생성
        self.runtime = trt.Runtime(self.trt_logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        
        if not self.engine:
            raise RuntimeError(f"[{self.model_name}] TensorRT 엔진 로드 실패")
        
        logger.info(f"[{self.model_name}] TensorRT 엔진 로드 성공!")
        
        # 실행 컨텍스트 생성
        self.context = self.engine.create_execution_context()
        logger.info(f"[{self.model_name}] 실행 컨텍스트 생성 완료")
    
    def check_engine(self):
        """엔진 정보 출력 및 검증"""
        logger.info(f"[{self.model_name}] 엔진 정보 확인 중...")
        
        try:
            # 엔진 속성 출력
            logger.info(f"[{self.model_name}] 엔진 바인딩 수: {self.engine.num_bindings}")
            
            # 최대 배치 크기
            logger.info(f"[{self.model_name}] 최대 배치 크기: {self.engine.max_batch_size}")
            
            # 입출력 목록
            inputs = [self.engine.get_binding_name(i) for i in self.input_binding_idxs]
            outputs = [self.engine.get_binding_name(i) for i in self.output_binding_idxs]
            
            logger.info(f"[{self.model_name}] 입력 텐서: {inputs}")
            logger.info(f"[{self.model_name}] 출력 텐서: {outputs}")
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.model_name}] 엔진 검증 중 오류: {e}")
            return False
    
    def predict(self, input_data=None):
        """
        모델 추론 실행
        
        Args:
            input_data: 입력 데이터 (None이면 랜덤 데이터 생성)
            
        Returns:
            출력 결과 딕셔너리
        """
        # 디바이스 메모리 버퍼 리스트
        device_buffers = []
        host_buffers = []
        bindings = []
        
        try:
            # 바인딩 메모리 초기화
            for i in range(self.engine.num_bindings):
                binding_name = self.engine.get_binding_name(i)
                binding_shape = self.engine.get_binding_shape(i)
                binding_dtype = self.binding_dtypes[binding_name]
                
                # 입력 바인딩인 경우
                if self.engine.binding_is_input(i):
                    # 입력 데이터가 제공되지 않았으면 랜덤 데이터 생성
                    if input_data is None:
                        host_data = np.random.random(binding_shape).astype(binding_dtype)
                        logger.info(f"[{self.model_name}] 입력 '{binding_name}': 랜덤 데이터 생성, 형상={binding_shape}")
                    else:
                        host_data = input_data
                        logger.info(f"[{self.model_name}] 입력 '{binding_name}': 사용자 데이터 사용, 형상={host_data.shape}")
                    
                    # 데이터 형상 및 타입 확인
                    if host_data.shape != binding_shape:
                        logger.warning(f"[{self.model_name}] 입력 형상 불일치. 예상: {binding_shape}, 실제: {host_data.shape}")
                        host_data = np.resize(host_data, binding_shape)
                    
                    if host_data.dtype != binding_dtype:
                        host_data = host_data.astype(binding_dtype)
                    
                    # 디바이스 메모리 할당
                    device_mem = cuda.mem_alloc(host_data.nbytes)
                    
                    # 데이터를 디바이스로 복사
                    cuda.memcpy_htod(device_mem, host_data)
                else:
                    # 출력 바인딩인 경우
                    host_data = np.zeros(binding_shape, dtype=binding_dtype)
                    device_mem = cuda.mem_alloc(host_data.nbytes)
                
                # 바인딩 목록 업데이트
                bindings.append(int(device_mem))
                device_buffers.append(device_mem)
                host_buffers.append(host_data)
            
            # 추론 실행
            logger.info(f"[{self.model_name}] 추론 실행 중...")
            start_time = time.time()
            self.context.execute_v2(bindings=bindings)
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # ms로 변환
            logger.info(f"[{self.model_name}] 추론 완료: {inference_time:.2f} ms")
            
            # 결과 수집
            outputs = {}
            for i in self.output_binding_idxs:
                name = self.engine.get_binding_name(i)
                idx = i  # 바인딩 인덱스
                
                # 디바이스에서 호스트로 결과 복사
                cuda.memcpy_dtoh(host_buffers[idx], device_buffers[idx])
                
                # 딕셔너리에 결과 저장
                outputs[name] = host_buffers[idx].copy()
            
            return outputs, inference_time
            
        except Exception as e:
            logger.error(f"[{self.model_name}] 추론 중 오류 발생: {e}", exc_info=True)
            raise
        
        finally:
            # CUDA 메모리 해제
            for buf in device_buffers:
                try:
                    buf.free()
                except:
                    pass
    
    def release(self):
        """리소스 해제"""
        logger.info(f"[{self.model_name}] 리소스 해제 중...")
        
        if hasattr(self, 'context') and self.context:
            del self.context
            self.context = None
        
        if hasattr(self, 'engine') and self.engine:
            del self.engine
            self.engine = None
        
        if hasattr(self, 'runtime') and self.runtime:
            del self.runtime
            self.runtime = None
            
        # 가비지 컬렉션 명시적 실행
        gc.collect()
        
        logger.info(f"[{self.model_name}] 리소스 해제 완료")
    
    def __del__(self):
        """소멸자"""
        self.release()


def test_multiple_models(model_paths):
    """
    여러 TensorRT 모델 테스트
    
    Args:
        model_paths: 모델 경로 목록
    """
    logger.info(f"== 여러 모델 테스트 시작: 총 {len(model_paths)}개 ==")
    
    # 결과 저장 딕셔너리
    results = {}
    
    # 모든 모델 서버 인스턴스
    servers = {}
    
    try:
        # 1. 모든 모델 로딩
        for model_path in model_paths:
            model_name = os.path.basename(model_path)
            try:
                logger.info(f"=== {model_name} 로딩 시작 ===")
                server = TRTServer(model_path, model_name)
                servers[model_name] = server
                
                if server.check_engine():
                    logger.info(f"✅ {model_name} 엔진 검증 성공")
                else:
                    logger.error(f"❌ {model_name} 엔진 검증 실패")
                    
            except Exception as e:
                logger.error(f"❌ {model_name} 로딩 실패: {e}", exc_info=True)
        
        # 2. 각 모델 추론 실행 및 결과 저장
        logger.info("모든 모델 로딩 완료, 추론 테스트 시작...")
        
        for model_name, server in servers.items():
            try:
                logger.info(f"=== {model_name} 추론 시작 ===")
                outputs, inference_time = server.predict()
                
                # 결과 저장
                results[model_name] = {
                    "status": "성공",
                    "inference_time": inference_time,
                    "outputs": {}
                }
                
                # 결과 정보 출력 및 저장
                for name, output in outputs.items():
                    shape_str = str(output.shape)
                    dtype_str = str(output.dtype)
                    results[model_name]["outputs"][name] = {
                        "shape": shape_str,
                        "dtype": dtype_str
                    }
                    
                    if name == "num_dets" and output.size > 0:
                        results[model_name]["outputs"][name]["value"] = output[0][0]
                        logger.info(f"  {model_name} - 출력 '{name}': 탐지된 객체 수: {output[0][0]}")
                    
                    logger.info(f"  {model_name} - 출력 '{name}': 형상={shape_str}, 타입={dtype_str}")
                
            except Exception as e:
                logger.error(f"❌ {model_name} 추론 실패: {e}", exc_info=True)
                results[model_name] = {
                    "status": "실패",
                    "error": str(e)
                }
        
        # 3. 종합 결과 출력
        logger.info("\n=== 종합 결과 ===")
        for model_name, result in results.items():
            status_icon = "✅" if result["status"] == "성공" else "❌"
            if result["status"] == "성공":
                logger.info(f"{status_icon} {model_name}: 추론 시간 {result['inference_time']:.2f} ms")
            else:
                logger.info(f"{status_icon} {model_name}: 실패 - {result.get('error', '알 수 없는 오류')}")
        
    finally:
        # 리소스 해제
        for model_name, server in servers.items():
            try:
                server.release()
            except Exception as e:
                logger.error(f"{model_name} 리소스 해제 중 오류: {e}")
    
    logger.info("== 모든 모델 테스트 완료 ==")
    return results


if __name__ == "__main__":
    try:
        # 여러 모델 경로 설정
        model_paths = [
            "models/20240718_gauge_detection.trt",
            "models/4_feature_detection.trt",
            "models/20240801_digit_seg.trt",
            "models/20240807_total_yolo.trt"
        ]
        
        # 실제 존재하는 모델만 필터링
        valid_models = []
        for path in model_paths:
            if os.path.exists(path):
                file_size = os.path.getsize(path) / (1024 * 1024)  # MB
                logger.info(f"모델 파일 발견: {path} ({file_size:.2f} MB)")
                valid_models.append(path)
            else:
                logger.warning(f"모델 파일이 존재하지 않습니다: {path}")
        
        # 존재하는 모델이 없으면 종료
        if not valid_models:
            logger.error("테스트할 모델 파일이 없습니다. 경로를 확인하세요.")
            exit(1)
        
        # 여러 모델 테스트 실행
        results = test_multiple_models(valid_models)
        
        # [옵션] 결과를 파일로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"model_test_results_{timestamp}.txt"
        
        with open(result_file, "w") as f:
            f.write(f"=== TensorRT 다중 모델 테스트 결과 ({timestamp}) ===\n\n")
            for model_name, result in results.items():
                f.write(f"모델: {model_name}\n")
                f.write(f"상태: {result['status']}\n")
                if result["status"] == "성공":
                    f.write(f"추론 시간: {result['inference_time']:.2f} ms\n")
                    f.write("출력 정보:\n")
                    for output_name, info in result["outputs"].items():
                        f.write(f"  - {output_name}: 형상={info['shape']}, 타입={info['dtype']}\n")
                        if "value" in info:
                            f.write(f"    값: {info['value']}\n")
                else:
                    f.write(f"오류: {result.get('error', '알 수 없는 오류')}\n")
                f.write("\n")
        
        logger.info(f"테스트 결과가 '{result_file}' 파일에 저장되었습니다.")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)