"""
Day 3 최종 버전 - 비동기 + 에러 핸들링 + 로깅
"""
import io
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException

from app.schemas import PixelPredictRequest, ImagePredictRequest, PredictResponse
from app.model_utils import load_model, predict, preprocess
from app.logger_config import setup_logger
from app.error_handlers import register_error_handlers
from app.middleware import RequestLoggingMiddleware


logger = setup_logger("ml_api")

app = FastAPI(
    title="MNIST Prediction API",
    description="비동기 처리, 에러 핸들링, 로깅이 적용된 MNIST 추론 API",
    version="3.0.0",
)

app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)

inference_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="inference")

MODEL_PATH = "models/mnist_state_dict.pth"
model = None


@app.on_event("startup")
async def startup():
    global model
    logger.info(f"모델 로드 중: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    logger.info("모델 로드 완료")


def run_inference(image_tensor: torch.Tensor) -> dict:
    """별도 스레드에서 실행되는 추론 함수"""
    if model is None:
        raise RuntimeError("모델이 로드되지 않았습니다")
    return predict(model, image_tensor)


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy" if model is not None else "loading",
        "model_loaded": model is not None,
    }


@app.get("/model/info", tags=["System"])
async def model_info():
    from app.model_utils import CLASS_NAMES
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "model_name": "SimpleClassifier",
        "model_path": MODEL_PATH,
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "total_parameters": total_params,
    }


@app.post("/predict/pixels", response_model=PredictResponse, tags=["Inference"])
async def predict_from_pixels(request: PixelPredictRequest):
    try:
        pixel_array = np.array(request.pixels, dtype=np.float32)
        pixel_tensor = torch.from_numpy(pixel_array)
        pixel_tensor = (pixel_tensor - 0.1307) / 0.3081
        pixel_tensor = pixel_tensor.unsqueeze(0).unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"전처리 실패: {str(e)}")

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(inference_executor, run_inference, pixel_tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {str(e)}")

    return PredictResponse(
        success=True,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"] if request.return_probabilities else None,
    )


@app.post("/predict/image", response_model=PredictResponse, tags=["Inference"])
async def predict_from_image(request: ImagePredictRequest):
    try:
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        image_tensor = preprocess(image).unsqueeze(0)
    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="유효하지 않은 Base64 문자열입니다.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 실패: {str(e)}")

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(inference_executor, run_inference, image_tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {str(e)}")

    return PredictResponse(
        success=True,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"] if request.return_probabilities else None,
    )