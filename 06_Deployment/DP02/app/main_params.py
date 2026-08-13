"""
파라미터 방식 실습
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Parameter Examples")

# ===== Path 파라미터 =====

# 기본 사용: 중괄호 {}로 경로 변수를 선언합니다
@app.get("/models/{model_name}")
def get_model_info(model_name: str):
    """특정 모델의 정보를 반환합니다."""
    return {
        "model_name": model_name,
        "status": "running",
        "version": "1.0.0",
    }

# Path 파라미터에 int 타입 지정
@app.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: int):
    """특정 예측 결과를 조회합니다."""
    return {
        "prediction_id": prediction_id,
        "label": "긍정",
        "confidence": 0.92,
    }

# ===== Query 파라미터 =====

# 함수 인자 중 Path에 포함되지 않은 것은 자동으로 Query 파라미터가 됩니다
@app.get("/models")
def list_models(status: str = None, limit: int = 10):
    """
    모델 목록을 조회합니다.

    - status: 필터링 조건 (선택) — "running", "stopped" 등
    - limit: 반환할 최대 개수 (기본값: 10)
    """
    # 실제로는 DB에서 조회하겠지만, 여기서는 예시 데이터를 반환합니다
    models = [
        {"name": "sentiment-v1", "status": "running"},
        {"name": "image-clf-v2", "status": "running"},
        {"name": "ner-v1", "status": "stopped"},
    ]

    # status 필터링
    if status:
        models = [m for m in models if m["status"] == status]

    # limit 적용
    models = models[:limit]

    return {
        "total": len(models),
        "models": models,
    }

# ===== Request Body =====
from pydantic import BaseModel
from typing import Optional

# 입력 스키마 정의
class PredictRequest(BaseModel):
    text: str
    return_probabilities: bool = False    # 선택, 기본값 False

# 출력 스키마 정의
class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Optional[dict] = None

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    텍스트 감성 분석을 수행합니다.

    - text: 분석할 텍스트 (필수)
    - return_probabilities: 전체 확률을 반환할지 여부 (선택, 기본 False)
    """
    # 실제로는 모델 추론을 수행하겠지만, 여기서는 더미 결과를 반환합니다
    result = {
        "label": "긍정",
        "confidence": 0.92,
    }

    if request.return_probabilities:
        result["probabilities"] = {
            "긍정": 0.92,
            "부정": 0.05,
            "중립": 0.03,
        }

    return result
