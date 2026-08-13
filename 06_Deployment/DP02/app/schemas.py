"""
API 입출력 스키마 정의
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    """모델 추론 요청 스키마"""
    pixel_values: list[float] = Field(
        ...,
        min_length=784,       # 28 * 28 = 784
        max_length=784,
        description="28x28 이미지의 픽셀 값 (784개). 0.0~1.0 범위.",
        examples=[[0.0] * 784],   # Swagger UI에 예시로 표시
    )
    return_probabilities: bool = Field(
        default=False,
        description="True로 설정하면 전체 클래스별 확률을 함께 반환합니다.",
    )


class PredictResponse(BaseModel):
    """모델 추론 응답 스키마"""
    label: int = Field(
        description="예측된 숫자 (0~9)",
    )
    confidence: float = Field(
        description="예측 확신도 (0.0~1.0)",
    )
    probabilities: Optional[list[float]] = Field(
        default=None,
        description="각 클래스(0~9)별 확률. return_probabilities=True일 때만 포함.",
    )
    model_version: str = Field(
        default="1.0.0",
        description="사용된 모델 버전",
    )


class HealthResponse(BaseModel):
    """헬스체크 응답 스키마"""
    status: str
    model_loaded: bool
