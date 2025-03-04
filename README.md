# Detection and Reading of Analog and digital Gauges (DRAG) ver 2 by silofox

- 2025-02-28 Start development for refactoring codes and serving real time in DRAG
- 2025-03-04 Refactoring DRAG and make safe in multi models

```
주요 업데이트
1. 안정성 및 오류 처리 개선

SafeTRTServerWrapper 클래스를 통한 더 안전한 TensorRT 엔진 관리
모든 작업에 대한 포괄적인 예외 처리 및 로깅
비동기 작업 처리 중 안전한 리소스 해제 메커니즘

2. 성능 지표 수집 및 모니터링

MetricsCollector 클래스를 통한 추론 성능 데이터 수집
평균, 최소, 최대 추론 시간, 처리량 등 실시간 모니터링
주기적인 성능 통계 로깅 및 분석

3. 다중 모델 관리 개선

TensorRTModelManager 클래스를 통한 효율적인 다중 모델 관리
동적으로 모델 등록, 초기화, 해제 기능
모델 상태 모니터링 및 진단 기능

4. 웜업 기능 통합

초기화 시 자동 웜업을 통한 콜드 스타트 성능 개선
사용자 정의 웜업 반복 횟수 설정 가능
```

# TensorRT 추론 서버

TensorRT 엔진을 사용하여 객체 감지 모델을 배포하는 고성능 추론 서버입니다. 메모리 풀링, 비동기 처리, RTSP 스트림 지원 등의 기능을 갖춘 모듈화된 FastAPI 애플리케이션입니다.

## 프로젝트 구조

```
drag_v2/
│
├── main.py                   # 애플리케이션 진입점
│
├── core/                     # 핵심 설정 모듈
│   ├── __init__.py
│   ├── config.py             # 앱 설정 및 상수
│   └── logging.py            # 로깅 설정
│
├── api/                      # API 라우트
│   ├── __init__.py
│   ├── router.py             # 모든 라우터를 통합하는 메인 라우터
│   ├── endpoints/
│   │   ├── __init__.py
│   │   ├── health.py         # 헬스 체크 및 진단 엔드포인트
│   │   ├── prediction.py     # 이미지 예측 관련 엔드포인트
│   │   └── streaming.py      # 비디오 스트리밍 관련 엔드포인트
│
├── services/                 # 비즈니스 로직
│   ├── __init__.py
│   ├── trt_service.py        # TensorRT 추론 서비스
│   ├── trt_server_wrapper.py # 안전한 TensorRT 엔진 래퍼
│   ├── metrics.py            # 성능 지표 수집 및 분석
│   ├── image_service.py      # 이미지 전처리/후처리
│   └── stream_service.py     # RTSP 스트림 처리
│
├── utils/                    # 헬퍼 유틸리티
│   ├── __init__.py
│   └── visualization.py      # 객체 감지 시각화
│
├── models/                   # 데이터 모델 (Pydantic)
│   ├── __init__.py
│   ├── requests.py           # 요청 모델
│   ├── responses.py          # 응답 모델
│   └── trt/                  # TensorRT 엔진 파일 디렉토리
│       └── 20240718_gauge_detection.trt  # 게이지 감지 모델
│
├── templates/                # HTML 템플릿
│   ├── index.html            # 메인 페이지 (이미지 업로드 및 결과 표시)
│   ├── video.html            # 비디오 스트림 보기 페이지
│   └── video_config.html     # 비디오 스트림 설정 페이지
│
├── static/                   # 정적 파일
│   ├── css/
│   └── js/
│
├── module/                   # 핵심 기능 모듈
│   ├── trt_server_with_memory_pool.py  # 메모리 풀링 기능이 있는 TensorRT 서버
│   └── rtsp_stream_processor.py        # RTSP 스트림 처리 모듈
│
├── boilerplate/              # 보일러플레이트 코드
│   ├── templates/            # 템플릿 파일 원본
│   └── configs/              # 기본 설정 파일
│
└── tests/                    # 테스트 코드
    ├── test_api.py           # API 테스트
    ├── test_services.py      # 서비스 테스트
    └── test_modules.py       # 모듈 테스트
```

## 모듈별 역할 설명

### 1. 코어 모듈 (`core/`)

- **config.py**: 애플리케이션 환경 설정, 모델 경로, 서버 포트 등 모든 설정 관리
- **logging.py**: 로깅 설정 및 전역 로거 제공

### 2. API 모듈 (`api/`)

- **router.py**: 모든 API 라우터를 통합하는 메인 라우터
- **endpoints/health.py**: 서버 상태 확인 및 TensorRT 모델 정보 제공 엔드포인트
- **endpoints/prediction.py**: 이미지 업로드, 객체 감지, 일괄 처리, JSON 기반 예측 엔드포인트
- **endpoints/streaming.py**: RTSP 스트림 관리, 비디오 피드 및 설정 관련 엔드포인트

### 3. 서비스 모듈 (`services/`)

- **trt_service.py**: TensorRT 엔진 관리, 다중 모델 지원, 성능 모니터링 기능이 추가된 향상된 추론 서비스
- **trt_server_wrapper.py**: 안전한 TensorRT 엔진 초기화 및 메모리 관리를 위한 래퍼 클래스
- **metrics.py**: 추론 시간, 처리량 등 성능 지표를 수집하고 분석하는 모듈
- **image_service.py**: 이미지 전처리(리사이징, 정규화, 패딩 등), 후처리, 좌표 변환 등 이미지 관련 기능
- **stream_service.py**: RTSP 스트림 관리, 프레임 처리, FPS 제한, MJPEG 스트리밍 등 스트림 관련 기능

### 4. 유틸리티 모듈 (`utils/`)

- **visualization.py**: 객체 감지 결과 시각화, 바운딩 박스 그리기, 레이블 표시 등 시각화 기능

### 5. 데이터 모델 (`models/`)

- **requests.py**: API 요청의 데이터 유효성 검사 및 타입 지정을 위한 Pydantic 모델
- **responses.py**: API 응답의 표준화 및 문서화를 위한 Pydantic 모델
- **trt/**: TensorRT 엔진 파일이 저장되는 디렉토리

### 6. HTML 템플릿 (`templates/`)

- **index.html**: 메인 페이지 - 이미지 업로드 및 객체 감지 결과 표시
- **video.html**: 비디오 스트림 보기 페이지 - RTSP 스트림 선택 및 실시간 객체 감지 결과 표시
- **video_config.html**: 비디오 스트림 설정 페이지 - RTSP URL 추가/삭제 등 스트림 관리

### 7. 핵심 기능 모듈 (`module/`)

- **trt_server_with_memory_pool.py**: 
  - TensorRT 엔진 관리 및 메모리 풀링을 통한 효율적인 GPU 메모리 사용
  - CUDA 스트림을 활용한 비동기 추론 처리
  - 입출력 텐서 바인딩 관리 및 추론 실행
  
- **rtsp_stream_processor.py**: 
  - RTSP 스트림 연결 및 프레임 캡처
  - 멀티스레딩을 통한 효율적인 스트림 처리
  - 스트림 상태 관리 및 리소스 정리

### 8. 보일러플레이트 (`boilerplate/`)

- 기본 템플릿, 설정, 스크립트 등 재사용 가능한 코드 저장소
- 새로운 인스턴스 배포 시 기본 설정으로 사용 가능

### 9. 테스트 (`tests/`)

- 단위 테스트, 통합 테스트, 부하 테스트 등 다양한 테스트 코드
- API, 서비스, 모듈별 테스트 구현

## 주요 기능 및 워크플로우

### 이미지 객체 감지 워크플로우

1. 클라이언트가 이미지를 업로드 (`/` 또는 `/predict` 엔드포인트)
2. `image_service`가 이미지를 전처리하여 모델 입력 형식에 맞게 변환
3. `trt_service`가 TensorRT 엔진을 사용하여 추론 실행
4. 결과를 후처리하여 바운딩 박스, 클래스, 신뢰도 점수 등 추출
5. `visualization_utils`를 사용하여 결과 시각화
6. 웹 페이지 또는 JSON 응답으로 결과 반환

### RTSP 스트리밍 워크플로우

1. 사용자가 RTSP URL 설정 (`/video-config` 페이지)
2. `stream_service`가 RTSP 스트림 연결 및 관리
3. 클라이언트가 비디오 스트림 요청 (`/video` 페이지)
4. 실시간으로 프레임 캡처 및 객체 감지 처리
5. MJPEG 형식으로 처리된 프레임 스트리밍

### 다중 모델 관리 워크플로우

1. 서버 시작 시 모든 등록된 모델 초기화
2. 각 모델은 독립적으로 메모리와 리소스 관리
3. 성능 지표 수집 및 모니터링 (각 모델별)
4. API 엔드포인트를 통한 모델 등록, 목록 조회, 해제 등 관리
5. 안전한 종료 시 모든 모델 리소스 비동기 해제

## 특징 및 장점

1. **모듈화된 구조**: 각 기능이 명확히 분리되어 유지보수 및 확장이 용이
2. **메모리 풀링**: GPU 메모리 재사용으로 메모리 효율성 향상 및 지연 시간 감소
3. **비동기 처리**: FastAPI의 비동기 기능을 활용한 높은 처리량
4. **RTSP 스트림 지원**: 실시간 비디오 스트림 처리 및 객체 감지
5. **성능 모니터링**: 실시간 추론 성능 측정 및 분석
6. **안전한 리소스 관리**: 비동기 작업 중에도 안전한 리소스 해제
7. **다중 모델 지원**: 여러 TensorRT 모델의 동시 관리 및 사용
8. **사용자 친화적 인터페이스**: 웹 기반 UI로 쉬운 사용 및 관리
9. **확장성**: 새로운 모델 및 기능을 쉽게 추가 가능

## 설치 및 실행

### 요구사항

- Python 3.8 이상
- NVIDIA GPU와 드라이버
- CUDA 11.0 이상
- TensorRT 8.0 이상

### 설치

```bash
# 저장소 클론
git clone https://github.com/your-repo/drag_v2.git
cd drag_v2

# 필요 패키지 설치
pip install -r requirements.txt

# 환경 변수 설정 (선택 사항)
export TRT_MODEL_PATH="models/trt/20240718_gauge_detection.trt"
export USE_CUDA_STREAM="True"
export PORT=5000
```

### 실행

```bash
# 서버 실행
python main.py
```

서버가 시작되면 `http://localhost:5000`에서 웹 인터페이스에 접속할 수 있습니다.

## API 문서

FastAPI의 자동 문서 기능을 통해 API 문서를 확인할 수 있습니다:
- Swagger UI: `http://localhost:5000/docs`
- ReDoc: `http://localhost:5000/redoc`

## 개발 가이드

### 새 모듈 추가

1. 적절한 디렉토리에 새 모듈 생성
2. `__init__.py`에 필요한 임포트 추가
3. 기존 서비스 또는 엔드포인트에 통합

### 새 모델 추가

1. TensorRT 엔진 파일을 `models/trt/` 디렉토리에 저장
2. `TensorRTModelManager`를 사용하여 모델 등록:

```python
from services.trt_service import model_manager

# 모델 등록
model_manager.register_model(
    name="new_model",
    model_path="models/trt/new_model.trt",
    enable_metrics=True
)

# 모델 사용
new_model_service = model_manager.get_service("new_model")
result, inference_time = new_model_service.predict(input_data)
```

### 성능 모니터링

성능 지표 수집을 활성화하고 API 엔드포인트를 통해 모니터링할 수 있습니다:

```python
# 성능 통계 조회
@app.get("/models/{model_name}/stats")
async def get_model_stats(model_name: str):
    service = model_manager.get_service(model_name)
    if not service:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    return service.get_performance_stats()
```

### 테스트

```bash
# 테스트 실행
pytest tests/
```

## 트러블슈팅

### 일반적인 문제

1. **TensorRT 엔진 초기화 실패**
   - 모델 경로 확인
   - GPU와 TensorRT 버전 호환성 확인
   - 로그에서 상세한 오류 메시지 확인

2. **RTSP 스트림 연결 문제**
   - RTSP URL 형식 확인
   - 네트워크 연결 및 방화벽 설정 확인

3. **메모리 부족 오류**
   - 배치 크기 감소
   - 입력 이미지 크기 축소
   - 동시에 관리하는 모델 수 제한

4. **성능 저하 문제**
   - 성능 지표 모니터링으로 병목 지점 파악
   - 웜업 반복 횟수 증가
   - 불필요한 모델 해제

## 라이센스

[라이센스 정보]

## 기여

[기여 가이드라인]