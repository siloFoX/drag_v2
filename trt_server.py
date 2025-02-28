import os
import numpy as np
import tensorrt as trt
import time
import logging

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
    
    def __init__(self, model_path: str):
        """
        TensorRT 서버 초기화
        
        Args:
            model_path: TensorRT 엔진 파일(.trt) 경로
        """
        self.model_path = model_path
        
        # 플러그인 로드
        trt.init_libnvinfer_plugins(None, "")
        
        # TensorRT 로거 생성
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 엔진 파일 로드
        self._load_engine()
    
    def _load_engine(self):
        """TensorRT 엔진 파일 로드"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"TensorRT 모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        logger.info(f"엔진 파일 로드 중: {self.model_path}")
        with open(self.model_path, "rb") as f:
            engine_data = f.read()
            logger.info(f"엔진 파일 크기: {len(engine_data)} 바이트")
        
        # 런타임과 엔진 생성
        with trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            
            if not self.engine:
                raise RuntimeError("TensorRT 엔진 로드에 실패했습니다.")
        
        logger.info("TensorRT 엔진 로드 성공!")
        
        # 바인딩 정보 출력
        self.num_bindings = self.engine.num_bindings
        logger.info(f"바인딩 수: {self.num_bindings}")
        
        for i in range(self.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = self.engine.get_binding_dtype(i)
            is_input = self.engine.binding_is_input(i)
            
            logger.info(f"  바인딩 {i}: 이름={name}, "
                       f"타입={'입력' if is_input else '출력'}, "
                       f"데이터타입={dtype}, 형상={shape}")
    
    def check_engine(self):
        """엔진 정보 출력 및 검증"""
        if not self.engine:
            logger.error("엔진이 로드되지 않았습니다.")
            return False
        
        logger.info("엔진 검증 중...")
        
        # 엔진 속성 확인
        logger.info(f"  최대 배치 크기: {self.engine.max_batch_size}")
        logger.info(f"  최적화 프로필 수: {self.engine.num_optimization_profiles}")
        
        # 입출력 확인
        inputs = []
        outputs = []
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            is_input = self.engine.binding_is_input(i)
            
            if is_input:
                inputs.append(name)
            else:
                outputs.append(name)
        
        logger.info(f"  입력: {inputs}")
        logger.info(f"  출력: {outputs}")
        
        return True


def test_trt_engine(model_path):
    """TensorRT 엔진 테스트"""
    logger.info(f"== 최소 TensorRT 엔진 테스트: {model_path} ==")
    
    try:
        # 서버 초기화
        server = TRTServer(model_path)
        
        # 엔진 검증
        if server.check_engine():
            logger.info("엔진 검증 성공!")
        
        logger.info("== 테스트 완료 ==")
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}", exc_info=True)


if __name__ == "__main__":
    try:
        # 모델 경로
        model_path = "models/20240718_gauge_detection.trt"
        
        # 파일 존재 확인
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            logger.info(f"모델 파일 크기: {file_size:.2f} MB")
        else:
            logger.error(f"모델 파일이 존재하지 않습니다: {model_path}")
        
        # 플러그인 존재 확인
        plugin_path = "/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so"
        if os.path.exists(plugin_path):
            logger.info(f"플러그인 파일 발견: {plugin_path}")
        else:
            logger.warning(f"플러그인 파일 없음: {plugin_path}")
        
        # 엔진 테스트
        test_trt_engine(model_path)
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}", exc_info=True)