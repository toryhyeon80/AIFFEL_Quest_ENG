"""
DP08 — 한국어 감정 분석 FastAPI 서버 (노트북 baseline)
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Depends, HTTPException

from app.schemas import PredictRequest, PredictResponse
from app.model_service import load_model, predict, MODEL_NAME
from app.auth import verify_api_key
from app.logger_config import setup_logger
from app.error_handlers import register_error_handlers
from app.middleware import RequestLoggingMiddleware


logger = setup_logger("dp08_pipeline")
inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="predict")

_model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_loaded
    logger.info(f"모델 로드 중: {MODEL_NAME}")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(inference_executor, load_model)
    _model_loaded = True
    logger.info("모델 로드 완료")
    yield


app = FastAPI(
    title="KR Sentiment Analysis API",
    description="snunlp/KR-FinBert-SC 감정 분류 API (인증 필요)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy" if _model_loaded else "loading",
        "model": MODEL_NAME if _model_loaded else None,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Predict"])
async def predict_endpoint(
    request: PredictRequest,
    user: str = Depends(verify_api_key),
):
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    logger.info(f"predict 요청 — 사용자: {user}, len={len(request.text)}")

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            inference_executor,
            predict,
            request.text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {str(e)}") from e

    return PredictResponse(
        success=True,
        label=result["label"],
        score=result["score"],
        user=user,
    )
