# RTSP 스트림 처리를 위한 클래스 및 함수
import cv2
import threading
import time
import queue
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# RTSP 스트림 처리 클래스
class RTSPStreamProcessor:
    """RTSP 스트림을 처리하고 프레임을 큐에 저장하는 클래스"""
    
    def __init__(self, rtsp_url: str, buffer_size: int = 10, reconnect_threshold: int = 5):
        """
        초기화
        
        Args:
            rtsp_url: RTSP 스트림 URL
            buffer_size: 프레임 버퍼 크기
            reconnect_threshold: 연결 끊김 후 재연결 시도할 프레임 수
        """
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.reconnect_threshold = reconnect_threshold
        
        # 프레임 버퍼 큐
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        
        # 상태 변수
        self.is_running = False
        self.last_frame = None
        self.last_frame_time = 0
        self.frames_processed = 0
        self.failed_frames = 0
        self.cap = None
        
        # 스레드
        self.thread = None
        
        logger.info(f"RTSP 스트림 프로세서 초기화: {rtsp_url}")
    
    def start(self):
        """스트림 처리 시작"""
        if self.is_running:
            logger.warning("이미 실행 중인 스트림입니다")
            return False
        
        self.is_running = True
        self.thread = threading.Thread(target=self._process_stream, daemon=True)
        self.thread.start()
        logger.info(f"RTSP 스트림 처리 시작: {self.rtsp_url}")
        return True
    
    def stop(self):
        """스트림 처리 중지"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # 큐 비우기
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info(f"RTSP 스트림 처리 중지: {self.rtsp_url}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        가장 최근 프레임 반환
        
        Returns:
            프레임 이미지 또는 None (사용 가능한 프레임이 없는 경우)
        """
        try:
            # 논블로킹 방식으로 프레임 획득
            frame = self.frame_queue.get_nowait()
            self.frame_queue.task_done()
            self.last_frame = frame
            return frame
        except queue.Empty:
            # 큐가 비어있으면 마지막 프레임 반환
            return self.last_frame
    
    def get_status(self) -> Dict:
        """
        현재 스트림 상태 정보 반환
        
        Returns:
            상태 정보 딕셔너리
        """
        current_time = time.time()
        fps = 0
        
        if self.last_frame_time > 0:
            time_diff = current_time - self.last_frame_time
            if time_diff > 0:
                fps = 1.0 / time_diff
        
        return {
            "url": self.rtsp_url,
            "is_running": self.is_running,
            "frames_processed": self.frames_processed,
            "failed_frames": self.failed_frames,
            "fps": round(fps, 2),
            "queue_size": self.frame_queue.qsize(),
            "has_frame": self.last_frame is not None
        }
    
    def _process_stream(self):
        """스트림에서 프레임을 읽어 큐에 저장하는 내부 메서드"""
        logger.info(f"스트림 처리 스레드 시작: {self.rtsp_url}")
        
        # 카메라 연결
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        # RTSP 트랜스포트 모드 설정 (TCP 또는 UDP)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
        
        # 연결 확인
        if not self.cap.isOpened():
            logger.error(f"RTSP 스트림 연결 실패: {self.rtsp_url}")
            self.is_running = False
            return
        
        # 프레임 처리 루프
        consecutive_fails = 0
        
        while self.is_running:
            try:
                # 프레임 읽기
                ret, frame = self.cap.read()
                
                if not ret:
                    consecutive_fails += 1
                    logger.warning(f"프레임 읽기 실패 ({consecutive_fails}/{self.reconnect_threshold})")
                    self.failed_frames += 1
                    
                    # 연속 실패 임계값 초과 시 재연결
                    if consecutive_fails >= self.reconnect_threshold:
                        logger.info("연결 재시도 중...")
                        if self.cap:
                            self.cap.release()
                        
                        time.sleep(1.0)  # 약간의 지연
                        self.cap = cv2.VideoCapture(self.rtsp_url)
                        consecutive_fails = 0
                    
                    time.sleep(0.1)  # 작은 지연으로 CPU 사용률 감소
                    continue
                
                # 성공적으로 프레임을 읽었으므로 카운터 초기화
                consecutive_fails = 0
                self.frames_processed += 1
                self.last_frame_time = time.time()
                
                # 큐가 가득 찬 경우 가장 오래된 프레임 제거
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.task_done()
                    except queue.Empty:
                        pass
                
                # 프레임을 큐에 추가
                self.frame_queue.put(frame)
                
            except Exception as e:
                logger.error(f"프레임 처리 중 오류: {e}")
                self.failed_frames += 1
                time.sleep(0.1)
        
        # 종료 시 리소스 정리
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info(f"스트림 처리 스레드 종료: {self.rtsp_url}")


# RTSP 스트림 관리자 클래스
class RTSPStreamManager:
    """여러 RTSP 스트림을 관리하는 클래스"""
    
    def __init__(self):
        """초기화"""
        self.streams: Dict[str, RTSPStreamProcessor] = {}
        logger.info("RTSP 스트림 관리자 초기화")
    
    def add_stream(self, stream_id: str, rtsp_url: str) -> bool:
        """
        새 스트림 추가 및 시작
        
        Args:
            stream_id: 스트림 식별자
            rtsp_url: RTSP URL
            
        Returns:
            성공 여부
        """
        if stream_id in self.streams:
            logger.warning(f"이미 존재하는 스트림 ID: {stream_id}")
            return False
        
        try:
            processor = RTSPStreamProcessor(rtsp_url)
            if processor.start():
                self.streams[stream_id] = processor
                logger.info(f"스트림 추가 및 시작: {stream_id}, URL: {rtsp_url}")
                return True
            else:
                logger.error(f"스트림 시작 실패: {stream_id}, URL: {rtsp_url}")
                return False
            
        except Exception as e:
            logger.error(f"스트림 추가 중 오류: {e}")
            return False
    
    def remove_stream(self, stream_id: str) -> bool:
        """
        스트림 중지 및 제거
        
        Args:
            stream_id: 스트림 식별자
            
        Returns:
            성공 여부
        """
        if stream_id not in self.streams:
            logger.warning(f"존재하지 않는 스트림 ID: {stream_id}")
            return False
        
        try:
            processor = self.streams[stream_id]
            processor.stop()
            del self.streams[stream_id]
            logger.info(f"스트림 제거: {stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"스트림 제거 중 오류: {e}")
            return False
    
    def get_stream(self, stream_id: str) -> Optional[RTSPStreamProcessor]:
        """
        스트림 프로세서 가져오기
        
        Args:
            stream_id: 스트림 식별자
            
        Returns:
            스트림 프로세서 또는 None
        """
        return self.streams.get(stream_id)
    
    def get_frame(self, stream_id: str) -> Optional[np.ndarray]:
        """
        특정 스트림의 프레임 가져오기
        
        Args:
            stream_id: 스트림 식별자
            
        Returns:
            프레임 이미지 또는 None
        """
        processor = self.get_stream(stream_id)
        if processor:
            return processor.get_frame()
        return None
    
    def get_all_streams(self) -> Dict[str, Dict]:
        """
        모든 스트림 상태 정보 가져오기
        
        Returns:
            스트림 ID를 키로 하는 상태 정보 딕셔너리
        """
        return {
            stream_id: processor.get_status()
            for stream_id, processor in self.streams.items()
        }
    
    def cleanup(self):
        """모든 스트림 정리"""
        for stream_id in list(self.streams.keys()):
            self.remove_stream(stream_id)
        logger.info("모든 스트림 정리 완료")