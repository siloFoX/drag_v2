"""
TensorRT 서비스의 성능 지표를 수집하고 분석하는 모듈입니다.
"""
import time
from typing import Dict, List, Any, Optional
from collections import deque
import threading
import numpy as np
from core.logging import logger

class MetricsCollector:
    """TensorRT 모델의 성능 지표를 수집하고 관리하는 클래스."""
    
    def __init__(self, model_name: str, max_history: int = 1000):
        """MetricsCollector를 초기화합니다.
        
        Args:
            model_name: 모니터링 대상 모델 이름
            max_history: 저장할 최대 지표 수
        """
        self.model_name = model_name
        self.max_history = max_history
        self.inference_times = deque(maxlen=max_history)
        self.total_times = deque(maxlen=max_history)
        self.start_time = time.time()
        self.total_inferences = 0
        self.lock = threading.Lock()
        
    def record_inference(self, inference_time: float, total_time: float) -> None:
        """추론 성능을 기록합니다.
        
        Args:
            inference_time: 순수 TensorRT 추론 시간 (ms)
            total_time: 전체 처리 시간 (ms)
        """
        with self.lock:
            self.inference_times.append(inference_time)
            self.total_times.append(total_time)
            self.total_inferences += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """수집된 성능 지표에 대한 통계를 반환합니다.
        
        Returns:
            Dict[str, Any]: 성능 통계 딕셔너리
        """
        with self.lock:
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
            
            # 백분위수 계산
            inference_array = np.array(self.inference_times)
            percentiles = np.percentile(inference_array, [50, 90, 95, 99])
            
            # 처리량 계산
            uptime = time.time() - self.start_time
            throughput = self.total_inferences / uptime if uptime > 0 else 0
            
            return {
                "model_name": self.model_name,
                "inference_time_ms": {
                    "avg": round(avg_inference, 2),
                    "min": round(min_inference, 2),
                    "max": round(max_inference, 2),
                    "p50": round(percentiles[0], 2),
                    "p90": round(percentiles[1], 2),
                    "p95": round(percentiles[2], 2),
                    "p99": round(percentiles[3], 2)
                },
                "total_time_ms": {
                    "avg": round(avg_total, 2)
                },
                "throughput": round(throughput, 2),
                "total_inferences": self.total_inferences,
                "samples_count": len(self.inference_times),
                "uptime_seconds": round(uptime, 2)
            }
    
    def reset(self) -> None:
        """측정된 모든 지표를 재설정합니다."""
        with self.lock:
            self.inference_times.clear()
            self.total_times.clear()
            self.start_time = time.time()
            self.total_inferences = 0
            
    def log_statistics(self) -> None:
        """현재 성능 통계를 로그로 기록합니다."""
        stats = self.get_statistics()
        
        if stats.get("status") == "no_data":
            logger.info(f"{self.model_name}: 아직 수집된 지표 없음")
            return
            
        logger.info(
            f"{self.model_name} 성능: "
            f"평균={stats['inference_time_ms']['avg']}ms, "
            f"최대={stats['inference_time_ms']['max']}ms, "
            f"P95={stats['inference_time_ms']['p95']}ms, "
            f"처리량={stats['throughput']}inf/s, "
            f"총 추론={stats['total_inferences']}"
        )


class GlobalMetricsManager:
    """시스템 전체 성능 지표를 관리하는 싱글톤 클래스."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalMetricsManager, cls).__new__(cls)
            cls._instance.collectors = {}
            cls._instance.lock = threading.Lock()
        return cls._instance
    
    def register_collector(self, name: str, collector: MetricsCollector) -> None:
        """새로운 지표 수집기를 등록합니다.
        
        Args:
            name: 수집기 이름
            collector: MetricsCollector 인스턴스
        """
        with self.lock:
            self.collectors[name] = collector
    
    def unregister_collector(self, name: str) -> None:
        """지표 수집기를 제거합니다.
        
        Args:
            name: 제거할 수집기 이름
        """
        with self.lock:
            if name in self.collectors:
                del self.collectors[name]
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """모든 등록된 수집기의 통계를 반환합니다.
        
        Returns:
            Dict: 모든 수집기의 통계 정보
        """
        with self.lock:
            return {
                name: collector.get_statistics() 
                for name, collector in self.collectors.items()
            }
    
    def reset_all(self) -> None:
        """모든 수집기를 재설정합니다."""
        with self.lock:
            for collector in self.collectors.values():
                collector.reset()
    
    def log_all_statistics(self) -> None:
        """모든 등록된 수집기의 통계를 로그로 기록합니다."""
        with self.lock:
            for collector in self.collectors.values():
                collector.log_statistics()

# 시스템 전체에서 사용할 수 있는 싱글톤 인스턴스
metrics_manager = GlobalMetricsManager()