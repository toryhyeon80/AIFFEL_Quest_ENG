from pydantic import BaseModel, Field, field_validator
from typing import Optional

class PixelPredictRequest(BaseModel):
    pixels: list[list[float]] = Field(..., description="28x28 픽셀 배열")
    return_probabilities: bool = Field(default=False)
    @field_validator("pixels")
    @classmethod
    def validate_pixels(cls, v):
        if len(v) != 28:
            raise ValueError(f"28행이어야 합니다. 현재: {len(v)}행")
        for i, row in enumerate(v):
            if len(row) != 28:
                raise ValueError(f"각 행은 28열이어야 합니다. {i}번째 행: {len(row)}열")
        return v

class ImagePredictRequest(BaseModel):
    image_base64: str = Field(..., min_length=1)
    return_probabilities: bool = Field(default=False)

class PredictResponse(BaseModel):
    success: bool = Field(description="성공 여부")
    predicted_class: str = Field(description="예측 숫자 (0~9)")
    confidence: float = Field(description="확신도", ge=0.0, le=1.0)
    probabilities: Optional[dict[str, float]] = Field(default=None)
