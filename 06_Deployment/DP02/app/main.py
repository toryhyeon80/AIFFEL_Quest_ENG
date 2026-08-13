"""
Day 2 실습: 모델 추론 API 서버
"""

from fastapi import FastAPI, HTTPException
import torch

from app.model_utils import load_model, predict
from app.schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
)

# ===== FastAPI 앱 생성 =====
app = FastAPI(
    title="MNIST Prediction API",
    description="Day 2 실습: MNIST 숫자 분류 모델 추론 API",
    version="1.0.0",
)


# ===== 모델을 서버 시작 시 한 번만 로드 =====
# 모듈 레벨에서 로드하면 서버가 시작될 때 실행됩니다.
# 요청마다 로드하면 매번 수 초가 걸리므로, 반드시 한 번만 로드해야 합니다.
try:
    model = load_model("models/mnist_state_dict.pth")
    model_loaded = True
    print("✅ 모델 로드 완료")
except Exception as e:
    model = None
    model_loaded = False
    print(f"❌ 모델 로드 실패: {e}")


# ===== 엔드포인트 1: 헬스체크 =====
@app.get("/health", response_model=HealthResponse)
def health_check():
    """서버 상태와 모델 로드 여부를 확인합니다."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
    )


# ===== 엔드포인트 2: 모델 추론 =====
@app.post("/predict", response_model=PredictResponse, summary="MNIST 숫자 예측")
def predict_digit(request: PredictRequest):
    """
    28x28 이미지의 픽셀 값을 받아 숫자(0~9)를 예측합니다.

    - **pixel_values**: 784개의 float 리스트 (28x28 이미지)
    - **return_probabilities**: True로 설정하면 전체 확률 분포를 반환
    """
    # 1. 모델이 로드되었는지 확인
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="모델이 로드되지 않았습니다. 서버 로그를 확인하세요."
        )

    # 2. 입력 데이터를 텐서로 변환
    try:
        input_tensor = torch.tensor(request.pixel_values, dtype=torch.float32)
        input_tensor = input_tensor.reshape(1, 1, 28, 28)  # (batch, channel, H, W)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"입력 데이터를 텐서로 변환할 수 없습니다: {str(e)}"
        )

    # 3. 추론 실행
    try:
        result = predict(model, input_tensor)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"모델 추론 중 에러가 발생했습니다: {str(e)}"
        )

    # 4. 응답 생성
    response = PredictResponse(
        label=result["label"],
        confidence=result["confidence"],
        model_version="1.0.0",
    )

    # 5. 옵션: 확률 분포 포함
    if request.return_probabilities:
        response.probabilities = [round(p, 4) for p in result["probabilities"]]

    return response
