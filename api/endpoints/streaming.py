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

import numpy as np
import threading
import asyncio
import cv2
import time
import gc

# 전역 처리 동기화를 위한 락 추가 (파일 상단에 추가)
global_processing_lock = threading.Lock()
# 결과의 수명 주기 추적을 위한 필드 추가
stream_results_ttl = {}  # stream_id: timestamp

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
    """RTSP 스트림 제거"""
    # 1. 먼저 활성 상태 비활성화 (중요)
    if stream_id in active_streams:
        active_streams[stream_id] = False
        logger.info(f"스트림 '{stream_id}' 비활성화")
    
    # 2. 짧은 대기 시간 추가 - 진행 중인 작업이 완료될 수 있도록
    await asyncio.sleep(0.1)
    
    # 3. 리소스 정리 (상세 정리 함수로 위임)
    clean_stream_resources(stream_id)
    
    # 4. 스트림 매니저에서 제거
    success = stream_service.remove_stream(stream_id)
    logger.info(f"스트림 '{stream_id}' 제거 완료")
    
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
    # 즉시 스트림 상태 확인 (입력 확인 전에)
    if stream_id not in active_streams or not active_streams.get(stream_id, False):
        return frame
    
    if frame is None:
        return frame
    
    # 스트림별 값 가져오기
    last_processed_results = stream_results.get(stream_id, None)
    last_full_process_time = stream_process_times.get(stream_id, 0)
    
    # 스트림 락 존재 확인
    if stream_id not in stream_locks:
        stream_locks[stream_id] = threading.Lock()
    process_lock = stream_locks[stream_id]
    
    # 처리 전 다시 한번 스트림 상태 확인
    if not active_streams.get(stream_id, False):
        return frame
    
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
    global_lock_acquired = False
    try:
        # 락 획득 후 상태 재확인
        if stream_id not in active_streams or not active_streams.get(stream_id, False):
            logger.info(f"락 획득 후 스트림 '{stream_id}' 비활성화 확인, 프레임 처리 중단")
            return frame
            
        current_time = time.time()
        time_since_process = current_time - last_full_process_time
        
        # 일정 간격마다만 전체 게이지 처리 수행
        if time_since_process >= process_interval:
            # 스트림 상태 재확인
            if not active_streams.get(stream_id, False):
                return frame
                
            # 전역 락 획득 시도 - CUDA 리소스 충돌 방지
            if global_processing_lock.acquire(blocking=False):
                global_lock_acquired = True
                
                # 결과 초기화 전 TTL 확인
                last_result_time = stream_results_ttl.get(stream_id, 0)
                
                # 결과가 10초 이상 경과했거나 없으면 완전 초기화
                if current_time - last_result_time > 10.0 or stream_id not in stream_results:
                    stream_results[stream_id] = {
                        "gauges": [],
                        "processing_time_ms": 0,
                        "status": "처리 중"
                    }
                
                try:
                    # 전체 게이지 처리
                    result = gauge_processor.process_image(frame)
                    
                    # 유효한 결과인지 확인
                    if result and isinstance(result, dict) and "gauges" in result:
                        stream_results[stream_id] = result
                        stream_process_times[stream_id] = current_time
                        stream_results_ttl[stream_id] = current_time
                        
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
        # CUDA 오류 특별 처리
        error_str = str(e).lower()
        if "cuda" in error_str or "resource handle" in error_str:
            logger.error(f"프레임 처리 중 CUDA 오류: {e}")
            # 오류 메시지를 프레임에 표시
            error_frame = frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                error_frame,
                "CUDA 리소스 오류 발생, 복구 중...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )
            # 가비지 컬렉션 실행
            gc.collect()
            return error_frame
        else:
            # 기타 오류는 기존대로 처리
            logger.error(f"프레임 처리 중 오류: {e}")
            return frame
    
    finally:
        # 항상 락 해제
        if process_lock.locked():
            try:
                process_lock.release()
            except RuntimeError:
                pass
                
        # 전역 락 해제
        if global_lock_acquired:
            try:
                global_processing_lock.release()
            except RuntimeError:
                pass

# 스트림 서비스에 스트림 종료 이벤트 핸들러 추가
@router.delete("/api/streams/{stream_id}")
async def remove_stream(stream_id: str):
    """Remove an RTSP stream.
    
    Args:
        stream_id: Stream identifier
    """
    logger.info(f"스트림 '{stream_id}' 제거 요청 받음")
    
    # 스트림 상태 즉시 비활성화
    if stream_id in active_streams:
        active_streams[stream_id] = False
        logger.info(f"스트림 '{stream_id}' 비활성화 완료")
    
    # 비동기 작업 처리를 위한 짧은 대기
    await asyncio.sleep(0.1)  # 100ms 대기
    
    # 스트림 리소스 정리
    clean_stream_resources(stream_id)
    
    # 처리 중인 프레임 작업 완료 대기
    # 타임아웃 0.5초 설정으로 최대 대기 시간 제한
    if stream_id in stream_locks and stream_locks[stream_id].locked():
        logger.info(f"스트림 '{stream_id}'의 프레임 처리 완료 대기 중...")
        await asyncio.sleep(0.5)
    
    # 스트림 매니저에서 제거
    success = stream_service.remove_stream(stream_id)
    if not success:
        logger.warning(f"스트림 '{stream_id}'를 스트림 매니저에서 찾을 수 없습니다")
        # 존재하지 않더라도 관련 데이터는 정리
        success = True
    
    logger.info(f"스트림 '{stream_id}' 제거 완료")
    return {"status": "success", "message": f"Stream '{stream_id}' removed"}

# 스트림 리소스 정리 함수 강화
def clean_stream_resources(stream_id):
    """스트림 관련 리소스 정리 (철저히)"""
    logger.info(f"스트림 '{stream_id}' 리소스 정리 중...")
    
    # 상태 업데이트 다시 확인
    if stream_id in active_streams:
        active_streams[stream_id] = False
    
    # 락 정리
    if stream_id in stream_locks:
        try:
            # 락이 걸려있으면 강제 해제 시도
            if stream_locks[stream_id].locked():
                logger.info(f"스트림 '{stream_id}'의 락이 걸려있어 해제 시도")
                stream_locks[stream_id].release()
        except Exception as e:
            logger.warning(f"스트림 '{stream_id}' 락 해제 중 오류 (무시됨): {e}")
        
        # 딕셔너리에서 제거
        del stream_locks[stream_id]
    
    # 결과 정리
    if stream_id in stream_results:
        del stream_results[stream_id]
    
    # 처리 시간 정리
    if stream_id in stream_process_times:
        del stream_process_times[stream_id]
    
    logger.info(f"스트림 '{stream_id}' 리소스 정리 완료")

# 애플리케이션 종료 시 정리 작업 추가 (필요한 경우 lifespan_handler에 추가)
def cleanup_all_streams():
    """모든 스트림 리소스 정리"""
    for stream_id in list(active_streams.keys()):
        clean_stream_resources(stream_id)
    logger.info("All stream resources cleaned up")