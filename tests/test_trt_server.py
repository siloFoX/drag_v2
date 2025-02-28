#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorRT 서버 테스트 모듈
- 기본 TRTServer 테스트
- 메모리 풀 TRTServerWithMemoryPool 테스트
- 다중 모델 테스트
- 성능 비교 테스트
"""

import os
import sys
import numpy as np
import logging
import time
import argparse
from datetime import datetime
import gc
import matplotlib.pyplot as plt
from tabulate import tabulate

# 상위 디렉토리 경로 추가 (drag_v2 디렉토리)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # drag_v2 디렉토리
sys.path.append(parent_dir)

# module 디렉토리에서 TensorRT 서버 클래스들 임포트
try:
    from module.trt_server import TRTServer
    from module.trt_server_with_memory_pool import TRTServerWithMemoryPool
except ImportError:
    print("TRTServer 클래스를 찾을 수 없습니다. 모듈 경로를 확인하세요.")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TRTTest")


class TRTTester:
    """TensorRT 서버 테스트 클래스"""
    
    def __init__(self, models_dir="models", results_dir="test_results"):
        """
        테스터 초기화
        
        Args:
            models_dir: 모델 디렉토리 경로
            results_dir: 결과 저장 디렉토리
        """
        self.models_dir = os.path.join(parent_dir, models_dir)
        self.results_dir = os.path.join(current_dir, results_dir)
        
        # 결과 디렉토리 생성
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 가능한 모델 목록 조회
        self.available_models = self._get_available_models()
        logger.info(f"테스트 가능한 모델: {len(self.available_models)}개")
        for i, model in enumerate(self.available_models):
            model_size = os.path.getsize(model) / (1024*1024)
            logger.info(f"  {i+1}. {os.path.basename(model)} ({model_size:.2f} MB)")
    
    def _get_available_models(self):
        """사용 가능한 모델 파일 목록 조회"""
        models = []
        if os.path.exists(self.models_dir):
            for filename in os.listdir(self.models_dir):
                if filename.endswith('.trt'):
                    model_path = os.path.join(self.models_dir, filename)
                    models.append(model_path)
        return models
    
    def test_single_model(self, model_path, num_runs=10, use_memory_pool=False, use_stream=True, warmup=True):
        """
        단일 모델 테스트
        
        Args:
            model_path: 모델 경로
            num_runs: 테스트 실행 횟수
            use_memory_pool: 메모리 풀 사용 여부
            use_stream: CUDA 스트림 사용 여부
            warmup: 웜업 실행 여부
        
        Returns:
            results: 테스트 결과 딕셔너리
        """
        model_name = os.path.basename(model_path)
        logger.info(f"== 단일 모델 테스트: {model_name} ==")
        logger.info(f"설정: 메모리 풀={use_memory_pool}, 스트림={use_stream}, 웜업={warmup}")
        
        results = {
            "model_name": model_name,
            "memory_pool": use_memory_pool,
            "stream": use_stream,
            "warmup": warmup,
            "times": [],
            "outputs": None,
            "status": "성공",
            "error": None
        }
        
        try:
            # 서버 인스턴스 생성
            if use_memory_pool:
                server = TRTServerWithMemoryPool(model_path, use_stream=use_stream)
            else:
                server = TRTServer(model_path)
            
            # 엔진 검증
            if server.check_engine():
                logger.info(f"✅ {model_name} 엔진 검증 성공")
            else:
                logger.error(f"❌ {model_name} 엔진 검증 실패")
                results["status"] = "실패"
                results["error"] = "엔진 검증 실패"
                return results
            
            # 웜업 실행
            if warmup:
                logger.info("웜업 실행 중...")
                warmup_times = []
                
                for i in range(3):  # 3회 웜업
                    start_time = time.time()
                    outputs = server.predict()
                    end_time = time.time()
                    
                    # 결과 형식에 따라 다르게 처리
                    if isinstance(outputs, tuple) and len(outputs) >= 1:
                        outputs = outputs[0]  # 첫 번째 요소가 출력 딕셔너리
                        inference_time = (end_time - start_time) * 1000  # ms
                    else:
                        inference_time = (end_time - start_time) * 1000  # ms
                    
                    warmup_times.append(inference_time)
                    logger.info(f"  웜업 {i+1}: {inference_time:.2f}ms")
                
                avg_warmup_time = sum(warmup_times) / len(warmup_times)
                logger.info(f"웜업 완료. 평균: {avg_warmup_time:.2f}ms")
            
            # 성능 테스트
            logger.info(f"성능 테스트 시작 ({num_runs}회)...")
            times = []
            
            for i in range(num_runs):
                start_time = time.time()
                outputs = server.predict()
                end_time = time.time()
                
                # 결과 형식에 따라 다르게 처리
                if isinstance(outputs, tuple) and len(outputs) >= 1:
                    if i == 0:  # 첫 번째 실행의 출력만 저장
                        results["outputs"] = outputs[0]
                    inference_time = (end_time - start_time) * 1000  # ms
                else:
                    if i == 0:  # 첫 번째 실행의 출력만 저장
                        results["outputs"] = outputs
                    inference_time = (end_time - start_time) * 1000  # ms
                
                times.append(inference_time)
                logger.info(f"  실행 {i+1}: {inference_time:.2f}ms")
            
            results["times"] = times
            
            # 통계 계산
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            logger.info(f"평균 추론 시간: {avg_time:.2f}ms")
            logger.info(f"최소 추론 시간: {min_time:.2f}ms")
            logger.info(f"최대 추론 시간: {max_time:.2f}ms")
            
            # 리소스 해제
            server.release()
            
        except Exception as e:
            logger.error(f"테스트 중 오류 발생: {e}")
            results["status"] = "실패"
            results["error"] = str(e)
        
        logger.info(f"== {model_name} 테스트 완료 ==")
        return results
    
    def test_multiple_models(self, model_paths=None, use_memory_pool=False):
        """
        여러 TensorRT 모델 테스트
        
        Args:
            model_paths: 모델 경로 목록 (None이면
            use_memory_pool: 메모리 풀 사용 여부
        
        Returns:
            results: 테스트 결과 딕셔너리
        """
        # 모델 목록이 제공되지 않으면 모든 가능한 모델 사용
        if model_paths is None:
            model_paths = self.available_models
        
        logger.info(f"== 여러 모델 테스트 시작: 총 {len(model_paths)}개 ==")
        logger.info(f"메모리 풀 사용: {use_memory_pool}")
        
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
                    
                    if use_memory_pool:
                        server = TRTServerWithMemoryPool(model_path, use_stream=True)
                    else:
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
                    
                    # 결과 형식에 따라 다르게 처리
                    if isinstance(outputs, tuple) and len(outputs) >= 1:
                        outputs_dict = outputs[0]  # 첫 번째 요소가 출력 딕셔너리
                        if len(outputs) >= 2 and isinstance(outputs[1], (int, float)):
                            inference_time = outputs[1]
                        else:
                            inference_time = (end_time - start_time) * 1000  # ms
                    else:
                        outputs_dict = outputs  # 이미 딕셔너리인 경우
                        inference_time = (end_time - start_time) * 1000  # ms
                    
                    # 결과 저장
                    results[model_name] = {
                        "status": "성공",
                        "inference_time": inference_time,
                        "outputs": {}
                    }
                    
                    # 결과 정보 출력 및 저장
                    logger.info(f"  {model_name} 추론 시간: {inference_time:.2f} ms")
                    
                    if isinstance(outputs_dict, dict):
                        for name, output in outputs_dict.items():
                            shape_str = str(output.shape)
                            dtype_str = str(output.dtype)
                            results[model_name]["outputs"][name] = {
                                "shape": shape_str,
                                "dtype": dtype_str
                            }
                            
                            if name == "num_dets" and output.size > 0:
                                num_detections = int(output[0][0])
                                results[model_name]["outputs"][name]["value"] = num_detections
                                logger.info(f"  {model_name} - 출력 '{name}': 탐지된 객체 수: {num_detections}")
                            
                            logger.info(f"  {model_name} - 출력 '{name}': 형상={shape_str}, 타입={dtype_str}")
                    else:
                        # 출력이 딕셔너리가 아닌 경우
                        logger.warning(f"  {model_name} - 출력이 예상된 형식이 아닙니다. 타입: {type(outputs_dict)}")
                
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
    
    def compare_performance(self, model_path, num_runs=10):
        """
        Compare performance across different configurations
        
        Args:
            model_path: Path to the model to test
            num_runs: Number of test runs per configuration
                
        Returns:
            results: Dictionary with test results
        """
        model_name = os.path.basename(model_path)
        logger.info(f"== Performance Comparison Test: {model_name} ==")
        
        configs = [
            {"name": "Basic", "memory_pool": False, "stream": False, "warmup": False},
            {"name": "Stream", "memory_pool": False, "stream": True, "warmup": False},
            {"name": "Memory Pool", "memory_pool": True, "stream": False, "warmup": False},
            {"name": "Memory Pool + Stream", "memory_pool": True, "stream": True, "warmup": False},
            {"name": "Warmup + Memory Pool + Stream", "memory_pool": True, "stream": True, "warmup": True}
        ]
        
        results = {}
        
        for config in configs:
            logger.info(f"\n=== Configuration: {config['name']} ===")
            test_result = self.test_single_model(
                model_path, 
                num_runs=num_runs,
                use_memory_pool=config["memory_pool"],
                use_stream=config["stream"],
                warmup=config["warmup"]
            )
            
            if test_result["status"] == "성공":
                results[config["name"]] = {
                    "times": test_result["times"],
                    "avg_time": sum(test_result["times"]) / len(test_result["times"]),
                    "min_time": min(test_result["times"]),
                    "max_time": max(test_result["times"])
                }
            else:
                results[config["name"]] = {
                    "status": "Failed",
                    "error": test_result["error"]
                }
        
        # Results summary
        logger.info("\n=== Performance Comparison Results ===")
        table_data = []
        for config_name, result in results.items():
            if "avg_time" in result:
                table_data.append([
                    config_name,
                    f"{result['avg_time']:.2f}ms",
                    f"{result['min_time']:.2f}ms",
                    f"{result['max_time']:.2f}ms"
                ])
            else:
                table_data.append([
                    config_name, 
                    "Failed", 
                    "Failed",
                    "Failed"
                ])
        
        print(tabulate(
            table_data,
            headers=["Configuration", "Average Time", "Minimum Time", "Maximum Time"],
            tablefmt="grid"
        ))
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.results_dir, f"performance_compare_{timestamp}.txt")
        
        with open(result_file, "w") as f:
            f.write(f"=== Performance Comparison Results: {model_name} ({timestamp}) ===\n\n")
            f.write(tabulate(
                table_data,
                headers=["Configuration", "Average Time", "Minimum Time", "Maximum Time"],
                tablefmt="grid"
            ))
            f.write("\n\n=== Detailed Results ===\n")
            for config_name, result in results.items():
                f.write(f"\n[{config_name}]\n")
                if "avg_time" in result:
                    f.write(f"Average Time: {result['avg_time']:.2f}ms\n")
                    f.write(f"Minimum Time: {result['min_time']:.2f}ms\n")
                    f.write(f"Maximum Time: {result['max_time']:.2f}ms\n")
                    f.write("Individual Run Times:\n")
                    for i, t in enumerate(result["times"]):
                        f.write(f"  Run {i+1}: {t:.2f}ms\n")
                else:
                    f.write(f"Status: Failed\n")
                    f.write(f"Error: {result.get('error', 'Unknown')}\n")
        
        logger.info(f"Results saved to '{result_file}'")
        
        # Create graphs
        self._create_performance_graph(results, model_name, timestamp)
        
        return results
    
    def _create_performance_graph(self, results, model_name, timestamp):
        """Create performance comparison graph"""
        try:
            plt.figure(figsize=(12, 6))
            
            # Bar graph
            labels = []
            avg_times = []
            min_times = []
            max_times = []
            
            # Convert Korean configuration names to English
            config_name_mapping = {
                "기본": "Basic",
                "스트림": "Stream",
                "메모리 풀": "Memory Pool",
                "메모리 풀 + 스트림": "Memory Pool + Stream",
                "웜업 + 메모리 풀 + 스트림": "Warmup + Memory Pool + Stream"
            }
            
            for config_name, result in results.items():
                if "avg_time" in result:
                    # Convert Korean config name to English if it exists in the mapping
                    english_name = config_name_mapping.get(config_name, config_name)
                    labels.append(english_name)
                    avg_times.append(result["avg_time"])
                    min_times.append(result["min_time"])
                    max_times.append(result["max_time"])
            
            x = np.arange(len(labels))
            width = 0.25
            
            plt.bar(x - width, avg_times, width, label='Average Time')
            plt.bar(x, min_times, width, label='Minimum Time')
            plt.bar(x + width, max_times, width, label='Maximum Time')
            
            plt.xlabel('Configuration')
            plt.ylabel('Inference Time (ms)')
            plt.title(f'TensorRT Performance Comparison: {model_name}')
            plt.xticks(x, labels, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            
            # Save graph
            graph_file = os.path.join(self.results_dir, f"performance_graph_{timestamp}.png")
            plt.savefig(graph_file)
            logger.info(f"Graph saved to '{graph_file}'")
            
            # Line graph (time per run)
            plt.figure(figsize=(12, 6))
            
            for config_name, result in results.items():
                if "times" in result:
                    # Convert Korean config name to English if it exists in the mapping
                    english_name = config_name_mapping.get(config_name, config_name)
                    plt.plot(range(1, len(result["times"])+1), result["times"], marker='o', label=english_name)
            
            plt.xlabel('Run Number')
            plt.ylabel('Inference Time (ms)')
            plt.title(f'Inference Time per Run: {model_name}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Save line graph
            line_graph_file = os.path.join(self.results_dir, f"performance_line_graph_{timestamp}.png")
            plt.savefig(line_graph_file)
            logger.info(f"Run-by-run graph saved to '{line_graph_file}'")
            
        except Exception as e:
            logger.error(f"Error while creating graph: {e}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='TensorRT 서버 테스트')
    parser.add_argument('--test', choices=['single', 'multiple', 'compare', 'all'], default='all',
                        help='테스트 유형 (단일 모델, 다중 모델, 성능 비교, 모두)')
    parser.add_argument('--model', type=str, help='테스트할 모델 파일 (없으면 첫 번째 모델 사용)')
    parser.add_argument('--num-runs', type=int, default=10, help='테스트 실행 횟수')
    parser.add_argument('--memory-pool', action='store_true', help='메모리 풀 사용 여부')
    parser.add_argument('--stream', action='store_true', default=True, help='CUDA 스트림 사용 여부')
    
    args = parser.parse_args()
    
    # 테스터 초기화
    tester = TRTTester()
    
    # 테스트할 모델이 지정되지 않았으면 첫 번째 모델 사용
    if args.model and os.path.exists(args.model):
        test_model = args.model
    elif tester.available_models:
        test_model = tester.available_models[0]
    else:
        logger.error("테스트할 모델을 찾을 수 없습니다.")
        return
    
    if args.test == 'single' or args.test == 'all':
        # 단일 모델 테스트
        tester.test_single_model(
            test_model,
            num_runs=args.num_runs,
            use_memory_pool=args.memory_pool,
            use_stream=args.stream
        )
    
    if args.test == 'multiple' or args.test == 'all':
        # 다중 모델 테스트
        tester.test_multiple_models(use_memory_pool=args.memory_pool)
    
    if args.test == 'compare' or args.test == 'all':
        # 성능 비교 테스트
        tester.compare_performance(test_model, num_runs=args.num_runs)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("사용자에 의해 테스트가 중단되었습니다.")
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}", exc_info=True)