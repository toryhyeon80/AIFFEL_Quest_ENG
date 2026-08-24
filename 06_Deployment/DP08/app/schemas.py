"""감정 분석 API 스키마"""
from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="분석할 한국어 문장",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "오늘 실적 발표가 기대 이상이라 주가가 크게 올랐다."}
            ]
        }
    }


class PredictResponse(BaseModel):
    success: bool
    label: str
    score: float
    user: Optional[str] = None
