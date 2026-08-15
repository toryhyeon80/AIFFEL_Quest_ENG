"""
Day 3 - Day 2 API에 비동기 패턴 적용
app/main.py의 개선 버전입니다.
"""
import io
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException

from app.schemas import (
    PixelPredictRequest,
    ImagePredictRequest,
    PredictResponse,
)
from app.model_utils import load_model, predict, preprocess


# ===== 앱 생성 =====
app = FastAPI(
    title="MNIST Prediction API (Async)",
    description="비동기 처리가 적용된 MNIST 추론 API",
    version="2.0.0",
)

# ===== 추론 전용 스레드풀 =====
inference_executor = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="inference",
)

# ===== 모델 로드 =====
MODEL_PATH = "models/mnist_state_dict.pth"
model = load_model(MODEL_PATH)


# ===== 동기 추론 함수 (스레드풀에서 실행될 함수) =====
def run_inference(image_tensor: torch.Tensor) -> dict:
    """모델 추론을 수행합니다. 이 함수는 별도 스레드에서 실행됩니다."""
    return predict(model, image_tensor)


# ===== 엔드포인트 =====

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict/pixels", response_model=PredictResponse, tags=["Inference"])
async def predict_from_pixels(request: PixelPredictRequest):
    """비동기 버전: 픽셀 배열로 추론"""
    try:
        # 전처리 (가벼운 작업 — 이벤트 루프에서 직접 수행)
        pixel_array = np.array(request.pixels, dtype=np.float32)
        pixel_tensor = torch.from_numpy(pixel_array)
        pixel_tensor = (pixel_tensor - 0.1307) / 0.3081
        pixel_tensor = pixel_tensor.unsqueeze(0).unsqueeze(0)

        # 추론 (무거운 작업 — 별도 스레드에서 실행)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            inference_executor,
            run_inference,
            pixel_tensor,
        )

        return PredictResponse(
            success=True,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            probabilities=result["probabilities"] if request.return_probabilities else None,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"추론 실패: {str(e)}")


@app.post("/predict/image", response_model=PredictResponse, tags=["Inference"])
async def predict_from_image(request: ImagePredictRequest):
    """비동기 버전: Base64 이미지로 추론"""
    try:
        # 전처리 (가벼운 작업)
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        image_tensor = preprocess(image).unsqueeze(0)

        # 추론 (무거운 작업 — 별도 스레드에서 실행)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            inference_executor,
            run_inference,
            image_tensor,
        )

        return PredictResponse(
            success=True,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            probabilities=result["probabilities"] if request.return_probabilities else None,
        )

    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="유효하지 않은 Base64 문자열입니다.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 실패: {str(e)}")
