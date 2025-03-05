"""
애플리케이션 전반에서 사용되는 사용자 정의 예외 클래스.
+
데이터 처리 및 모델 초기화 관련 예외 클래스 구현
"""
import traceback
from typing import Optional, Any, Dict

class ModelNotInitializedError(Exception):
    """
    모델이 초기화되지 않았거나 로드되지 않은 상태에서 사용할 때 발생하는 예외
    
    Attributes:
        message (str): 에러 메시지
        model_id (str): 초기화되지 않은 모델 ID
        details (dict): 추가적인 오류 상세 정보
    """
    
    def __init__(self, message: str = None, model_id: str = None, details: Dict[str, Any] = None):
        self.message = message or "모델이 초기화되지 않았습니다"
        self.model_id = model_id
        self.details = details or {}
        
        # 모델 ID가 제공된 경우 메시지에 포함
        if model_id:
            self.message = f"{self.message} (모델 ID: {model_id})"
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """예외 정보를 딕셔너리로 변환"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "model_id": self.model_id,
            "details": self.details
        }


class ProcessingError(Exception):
    """
    데이터 처리 중 발생하는 예외
    
    Attributes:
        message (str): 에러 메시지
        process_stage (str): 오류가 발생한 처리 단계
        input_shape (tuple): 입력 데이터의 형상 (해당되는 경우)
        traceback_str (str): 스택 추적 문자열
        details (dict): 추가적인 오류 상세 정보
    """
    
    def __init__(self, message: str = None, process_stage: str = None, 
                 input_shape: tuple = None, details: Dict[str, Any] = None, 
                 capture_traceback: bool = True):
        self.message = message or "데이터 처리 중 오류가 발생했습니다"
        self.process_stage = process_stage
        self.input_shape = input_shape
        self.details = details or {}
        
        # 처리 단계가 제공된 경우 메시지에 포함
        if process_stage:
            self.message = f"{process_stage} 단계에서 {self.message}"
        
        # 스택 추적 캡처 (선택적)
        self.traceback_str = ""
        if capture_traceback:
            self.traceback_str = traceback.format_exc()
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """예외 정보를 딕셔너리로 변환"""
        result = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "process_stage": self.process_stage,
            "details": self.details
        }
        
        # 입력 형상이 있는 경우 추가
        if self.input_shape:
            result["input_shape"] = self.input_shape
        
        # 스택 추적이 있는 경우 추가
        if self.traceback_str:
            result["traceback"] = self.traceback_str
        
        return result
    
    @classmethod
    def from_exception(cls, exception: Exception, process_stage: str = None, 
                      input_shape: tuple = None, details: Dict[str, Any] = None) -> 'ProcessingError':
        """
        다른 예외로부터 ProcessingError 생성
        
        Args:
            exception: 원본 예외
            process_stage: 오류가 발생한 처리 단계
            input_shape: 입력 데이터의 형상 (해당되는 경우)
            details: 추가적인 오류 상세 정보
            
        Returns:
            ProcessingError 인스턴스
        """
        # 기존 예외 메시지 사용
        message = str(exception)
        
        # 세부 정보에 원본 예외 유형 추가
        if details is None:
            details = {}
        details['original_exception_type'] = exception.__class__.__name__
        
        return cls(message, process_stage, input_shape, details)

class BaseError(Exception):
    """모든 애플리케이션 예외의 기본 클래스"""
    
    def __init__(self, message: str = None):
        self.message = message or "애플리케이션 오류가 발생했습니다"
        super().__init__(self.message)

# 설정 관련 오류
class ConfigurationError(BaseError):
    """설정 오류"""
    
    def __init__(self, message: str = None):
        self.message = message or "설정 오류가 발생했습니다"
        super().__init__(self.message)

# 모델 관련 오류
class ModelError(BaseError):
    """모델 관련 기본 오류"""
    pass



class ModelLoadError(ModelError):
    """모델 로드 실패"""
    
    def __init__(self, message: str = None):
        self.message = message or "모델 로드에 실패했습니다"
        super().__init__(self.message)

# 추론 관련 오류
class InferenceError(BaseError):
    """추론 중 발생하는 오류"""
    
    def __init__(self, message: str = None):
        self.message = message or "추론 중 오류가 발생했습니다"
        super().__init__(self.message)

# 데이터 관련 오류
class DataError(BaseError):
    """데이터 관련 기본 오류"""
    pass

class DataFormatError(DataError):
    """데이터 형식 오류"""
    
    def __init__(self, message: str = None):
        self.message = message or "데이터 형식이 잘못되었습니다"
        super().__init__(self.message)

class InputValidationError(DataError):
    """입력 검증 오류"""
    
    def __init__(self, message: str = None):
        self.message = message or "입력값이 유효하지 않습니다"
        super().__init__(self.message)

# 검출(Detection) 관련 오류
class DetectionError(BaseError):
    """객체 검출 중 발생하는 오류"""
    
    def __init__(self, message: str = None):
        self.message = message or "객체 검출 중 오류가 발생했습니다"
        super().__init__(self.message)

# API 관련 오류
class APIError(BaseError):
    """API 호출 관련 오류"""
    
    def __init__(self, message: str = None):
        self.message = message or "API 호출 중 오류가 발생했습니다"
        super().__init__(self.message)

# 리소스 관련 오류
class ResourceError(BaseError):
    """리소스 관련 오류"""
    pass

class ResourceNotFoundError(ResourceError):
    """리소스를 찾을 수 없음"""
    
    def __init__(self, message: str = None):
        self.message = message or "요청한 리소스를 찾을 수 없습니다"
        super().__init__(self.message)

class ResourceExhaustedError(ResourceError):
    """리소스 고갈 (메모리 등)"""
    
    def __init__(self, message: str = None):
        self.message = message or "리소스가 고갈되었습니다"
        super().__init__(self.message)

# 시스템 관련 오류
class SystemError(BaseError):
    """시스템 관련 오류"""
    
    def __init__(self, message: str = None):
        self.message = message or "시스템 오류가 발생했습니다"
        super().__init__(self.message)