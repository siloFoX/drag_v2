"""
Endpoints for video streaming and management.
"""
from fastapi import APIRouter, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from typing import Dict, Any, Optional, List, AsyncGenerator
from contextlib import asynccontextmanager

from core.config import TEMPLATES_DIR
from core.logging import logger
from services.stream_service import stream_service
from services.image_service import image_service
from services.gauge_processor import gauge_processor
from utils.visualization import visualization_utils
from models.requests import StreamRequest
from models.responses import StreamInfo

import threading
import cv2
import time

process_lock = threading.Lock()

# 스트림 활성 상태를 추적하기 위한 전역 변수 추가
active_streams = {}  # stream_id: active_status
# 스트림별 처리 결과를 저장하기 위한 변수
stream_results = {}  # stream_id: last_processed_results
# 스트림별 마지막 처리 시간
stream_process_times = {}  # stream_id: last_full_process_time
# 스트림별 프로세싱 락
stream_locks = {}  # stream_id: lock

# frame 성능 최적화를 위한 변수
process_interval = 1.0 # 1초마다 전체 게이지 처리

router = APIRouter(
    prefix="",
    tags=["streaming"],
)

# Initialize templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@router.get("/video-config", response_class=HTMLResponse)
async def video_config(request: Request):
    """RTSP stream configuration page."""
    streams = stream_service.get_all_streams()
    return templates.TemplateResponse(
        "video_config.html", 
        {
            "request": request,
            "streams": streams
        }
    )

@router.post("/api/streams")
async def add_stream(stream_id: str = Form(...), rtsp_url: str = Form(...)):
    """Add a new RTSP stream.
    
    Args:
        stream_id: Stream identifier
        rtsp_url: RTSP URL
    """
    if not stream_id or not rtsp_url:
        raise HTTPException(status_code=400, detail="Stream ID and RTSP URL are required")
    
    if not rtsp_url.startswith(("rtsp://", "http://", "https://", "file://")):
        raise HTTPException(status_code=400, detail="Invalid URL format")
    
    success = stream_service.add_stream(stream_id, rtsp_url)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to add stream")
    
    return {"status": "success", "message": f"Stream '{stream_id}' added"}

@router.delete("/api/streams/{stream_id}")
async def remove_stream(stream_id: str):
    """Remove an RTSP stream.
    
    Args:
        stream_id: Stream identifier
    """
    success = stream_service.remove_stream(stream_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Stream '{stream_id}' not found")
    
    return {"status": "success", "message": f"Stream '{stream_id}' removed"}

@router.get("/api/streams")
async def get_streams():
    """Get status of all streams."""
    streams = stream_service.get_all_streams()
    return streams

@router.get("/video", response_class=HTMLResponse)
async def video_page(request: Request, stream_id: Optional[str] = None):
    """Video stream viewing page.
    
    Args:
        stream_id: Stream identifier (if None, show all streams)
    """
    streams = stream_service.get_all_streams()
    
    # Check if specified stream exists
    if stream_id and not stream_service.stream_exists(stream_id):
        return templates.TemplateResponse(
            "video.html", 
            {
                "request": request,
                "error": f"Stream '{stream_id}' not found",
                "streams": streams,
                "selected_stream": None
            }
        )
    
    return templates.TemplateResponse(
        "video.html", 
        {
            "request": request,
            "streams": streams,
            "selected_stream": stream_id
        }
    )

@router.get("/api/video-feed/{stream_id}")
async def video_feed(stream_id: str):
    """Video feed endpoint (MJPEG stream).
    
    Args:
        stream_id: Stream identifier
    """
    if not stream_service.stream_exists(stream_id):
        raise HTTPException(status_code=404, detail=f"Stream '{stream_id}' not found")
    
    # 스트림별 락 생성 (없으면)
    if stream_id not in stream_locks:
        stream_locks[stream_id] = threading.Lock()
    
    # 스트림 활성화 표시
    active_streams[stream_id] = True
    # 스트림별 초기 처리 시간 설정
    stream_process_times[stream_id] = 0
    
    logger.info(f"Stream '{stream_id}' activated")
    
    @asynccontextmanager
    async def stream_lifecycle():
        # 스트림 시작 시
        try:
            yield
        finally:
            # 스트림 종료 시 정리 작업
            active_streams[stream_id] = False
            # 스트림 상태 로깅
            logger.info(f"Stream '{stream_id}' deactivated, cleaning up resources")

            # 락이 걸려있다면 해제
            if stream_id in stream_locks and stream_locks[stream_id].locked():
                try:
                    stream_locks[stream_id].release()
                except RuntimeError:
                    # 이미 해제된 경우 예외 무시
                    pass

            # 스트림별 결과 정리
            if stream_id in stream_results:
                del stream_results[stream_id]
            
            # 스트림별 처리 시간 정리
            if stream_id in stream_process_times:
                del stream_process_times[stream_id]
    
    # 생성기 함수 - stream_id를 프레임 처리 함수에 전달
    async def frame_generator():
        async with stream_lifecycle():
            # 수정된 부분: stream_id를 process_frame 함수에 전달하는 래퍼 함수
            async def process_frame_with_id(frame):
                return await process_frame(frame, stream_id)
            
            # 스트림 활성 상태 확인 추가
            async for frame in stream_service.generate_frames(stream_id, process_frame_with_id):
                # 스트림이 비활성화되면 즉시 중단
                if not active_streams.get(stream_id, False):
                    logger.info(f"Stream '{stream_id}' deactivated, stopping frame generation")
                    break
                yield frame
    
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# stream_id를 매개변수로 추가
async def process_frame(frame, stream_id):
    """
    스트림의 각 프레임을 처리하여 게이지 검출 및 값 판독
    
    Args:
        frame: 비디오 프레임
        stream_id: 스트림 식별자
        
    Returns:
        처리된 프레임
    """
    if frame is None:
        return frame
    
    # 스트림이 비활성화 상태면 처리하지 않음
    if not active_streams.get(stream_id, False):
        return frame
    
    # 스트림별 값 가져오기
    last_processed_results = stream_results.get(stream_id, None)
    last_full_process_time = stream_process_times.get(stream_id, 0)
    process_lock = stream_locks.get(stream_id)
    
    if process_lock is None:
        # 락이 없으면 생성
        process_lock = threading.Lock()
        stream_locks[stream_id] = process_lock
    
    # 원본 프레임 복사
    processed_frame = frame.copy()
    
    # 락 획득 시도 (비차단 모드) - 이미 처리 중이면 시각화만 수행
    if not process_lock.acquire(blocking=False):
        # 기존 결과로 시각화만 수행
        if last_processed_results and "gauges" in last_processed_results:
            try:
                # 이전 검출 결과로 시각화
                for gauge in last_processed_results["gauges"]:
                    box = gauge.get("box")
                    if not box:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 255, 0) if gauge.get("status") == "성공" else (0, 0, 255)
                    
                    # 바운딩 박스 그리기
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 정보 표시
                    gauge_type = gauge.get("type", "unknown")
                    gauge_id = gauge.get("id", 0)
                    cv2.putText(
                        processed_frame,
                        f"{gauge_type} #{gauge_id}", 
                        (x1, y1 - 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2
                    )
                    
                    # 값 표시
                    if gauge.get("status") == "성공" and gauge.get("value") is not None:
                        value_text = f"{gauge['value']:.1f} {gauge.get('unit', '')}"
                        cv2.putText(
                            processed_frame,
                            value_text, 
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2
                        )
            except Exception as e:
                logger.error(f"이전 결과로 시각화 중 오류: {e}")
        
        # 처리 정보 표시
        time_since_process = time.time() - last_full_process_time
        cv2.putText(
            processed_frame,
            f"Processing busy (skipping detection), Last: {time_since_process:.1f}s ago", 
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2
        )
        
        return processed_frame
    
    # 락 획득 성공 - 처리 시작
    try:
        current_time = time.time()
        
        # 일정 간격마다만 전체 게이지 처리 수행
        time_since_process = current_time - last_full_process_time
        if time_since_process >= process_interval:
            # 스트림이 여전히 활성 상태인지 재확인
            if active_streams.get(stream_id, False):
                try:
                    # 전체 게이지 처리
                    result = gauge_processor.process_image(frame)
                    
                    # 유효한 결과인지 확인
                    if result and isinstance(result, dict) and "gauges" in result:
                        stream_results[stream_id] = result
                        stream_process_times[stream_id] = current_time
                        
                        gauges_count = len(result["gauges"])
                        logger.info(f"[Stream {stream_id}] 게이지 {gauges_count}개 검출 완료, 처리 시간: {result.get('processing_time_ms', 0):.2f}ms")
                except Exception as e:
                    logger.error(f"[Stream {stream_id}] 게이지 처리 중 오류: {e}")
        
        # 시각화
        if stream_id in stream_results and "gauges" in stream_results[stream_id]:
            last_processed_results = stream_results[stream_id]
            
            # 시각화된 결과가 있으면 사용
            if "visualization" in last_processed_results:
                processed_frame = last_processed_results["visualization"].copy()
            else:
                # 직접 시각화
                for gauge in last_processed_results["gauges"]:
                    box = gauge.get("box")
                    if not box:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 255, 0) if gauge.get("status") == "성공" else (0, 0, 255)
                    
                    # 바운딩 박스 그리기
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 정보 표시
                    gauge_type = gauge.get("type", "unknown")
                    gauge_id = gauge.get("id", 0)
                    cv2.putText(
                        processed_frame,
                        f"{gauge_type} #{gauge_id}", 
                        (x1, y1 - 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2
                    )
                    
                    # 값 표시
                    if gauge.get("status") == "성공" and gauge.get("value") is not None:
                        value_text = f"{gauge['value']:.1f} {gauge.get('unit', '')}"
                        cv2.putText(
                            processed_frame,
                            value_text, 
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2
                        )
        
        # 처리 정보 표시
        cv2.putText(
            processed_frame,
            f"Stream: {stream_id} | Process interval: {process_interval}s, Last: {time_since_process:.1f}s ago", 
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )
        
        if stream_id in stream_results and "gauges" in stream_results[stream_id]:
            gauges_count = len(stream_results[stream_id].get("gauges", []))
            process_time = stream_results[stream_id].get("processing_time_ms", 0)
            
            cv2.putText(
                processed_frame,
                f"Detected: {gauges_count} gauges | Time: {process_time:.1f}ms",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )
        
        return processed_frame
        
    except Exception as e:
        logger.error(f"[Stream {stream_id}] 프레임 처리 중 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return frame
    
    finally:
        # 항상 락 해제
        process_lock.release()

# 스트림 서비스에 스트림 종료 이벤트 핸들러 추가
@router.delete("/api/streams/{stream_id}")
async def remove_stream(stream_id: str):
    """Remove an RTSP stream.
    
    Args:
        stream_id: Stream identifier
    """
    # 스트림 상태 비활성화
    active_streams[stream_id] = False
    
    # 스트림 리소스 정리
    clean_stream_resources(stream_id)
    
    success = stream_service.remove_stream(stream_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Stream '{stream_id}' not found")
    
    logger.info(f"Stream '{stream_id}' removed and resources cleaned up")
    return {"status": "success", "message": f"Stream '{stream_id}' removed"}

# 스트림 리소스 정리 함수 추가
def clean_stream_resources(stream_id):
    """스트림 관련 리소스 정리"""
    # 락 정리
    if stream_id in stream_locks:
        if stream_locks[stream_id].locked():
            try:
                stream_locks[stream_id].release()
            except RuntimeError:
                # 이미 해제된 경우 예외 무시
                pass
        del stream_locks[stream_id]
    
    # 결과 정리
    if stream_id in stream_results:
        del stream_results[stream_id]
    
    # 처리 시간 정리
    if stream_id in stream_process_times:
        del stream_process_times[stream_id]
    
    # 활성 상태 정리
    if stream_id in active_streams:
        del active_streams[stream_id]
    
    logger.info(f"Resources for stream '{stream_id}' cleaned up")

# 애플리케이션 종료 시 정리 작업 추가 (필요한 경우 lifespan_handler에 추가)
def cleanup_all_streams():
    """모든 스트림 리소스 정리"""
    for stream_id in list(active_streams.keys()):
        clean_stream_resources(stream_id)
    logger.info("All stream resources cleaned up")