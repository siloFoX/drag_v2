import os
import sys
import numpy as np
import logging
import time
from datetime import datetime
import gc

# 상위 디렉토리 경로 추가 (drag_v2 디렉토리)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # drag_v2 디렉토리
sys.path.append(parent_dir)

# module 디렉토리에서 TRTServer 클래스 임포트
from module.trt_server import TRTServer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
                server = TRTServer(model_path)
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
                
                # 시간 측정 시작
                start_time = time.time()
                outputs = server.predict()
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000  # ms로 변환
                
                # 결과 저장
                results[model_name] = {
                    "status": "성공",
                    "inference_time": inference_time,
                    "outputs": {}
                }
                
                # 튜플 형태의 출력 처리 (predict 메서드가 dict가 아닌 (dict, inference_time) 튜플을 반환하는 경우)
                if isinstance(outputs, tuple):
                    if len(outputs) >= 1:
                        outputs_dict = outputs[0]  # 첫 번째 요소가 출력 딕셔너리
                        
                        # 만약 두 번째 요소가 있고 숫자라면 그것이 추론 시간일 수 있음
                        if len(outputs) >= 2 and isinstance(outputs[1], (int, float)):
                            inference_time = outputs[1]
                    else:
                        outputs_dict = {}
                else:
                    outputs_dict = outputs  # 이미 딕셔너리인 경우
                
                # 결과 정보 출력 및 저장
                logger.info(f"  {model_name} 추론 시간: {inference_time:.2f} ms")
                
                # outputs_dict가 있는 경우에만 항목 순회
                if isinstance(outputs_dict, dict):
                    for name, output in outputs_dict.items():
                        shape_str = str(output.shape)
                        dtype_str = str(output.dtype)
                        results[model_name]["outputs"][name] = {
                            "shape": shape_str,
                            "dtype": dtype_str
                        }
                        
                        if name == "num_dets" and output.size > 0:
                            results[model_name]["outputs"][name]["value"] = int(output[0][0])
                            logger.info(f"  {model_name} - 출력 '{name}': 탐지된 객체 수: {int(output[0][0])}")
                        
                        logger.info(f"  {model_name} - 출력 '{name}': 형상={shape_str}, 타입={dtype_str}")
                else:
                    # outputs_dict가 딕셔너리가 아닌 경우 로깅
                    logger.warning(f"  {model_name} - 출력이 예상된 형식이 아닙니다. 타입: {type(outputs_dict)}")
                    if isinstance(outputs_dict, np.ndarray):
                        logger.info(f"  {model_name} - 출력 형상: {outputs_dict.shape}, 타입: {outputs_dict.dtype}")
                        results[model_name]["outputs"]["output"] = {
                            "shape": str(outputs_dict.shape),
                            "dtype": str(outputs_dict.dtype)
                        }
                    else:
                        logger.info(f"  {model_name} - 출력: {outputs_dict}")
                
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
            os.path.join(parent_dir, "models/20240718_gauge_detection.trt"),
            os.path.join(parent_dir, "models/4_feature_detection.trt"),
            os.path.join(parent_dir, "models/20240801_digit_seg.trt"),
            os.path.join(parent_dir, "models/20240807_total_yolo.trt")
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
        result_file = os.path.join(current_dir, f"model_test_results_{timestamp}.txt")
        
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