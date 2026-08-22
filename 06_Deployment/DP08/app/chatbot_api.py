"""
DP08 - 서울 실내 추천 챗봇 FastAPI 서버
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Depends, HTTPException

from app.chatbot_schemas import ChatRequest, ChatResponse
from app.chatbot_model import IndoorChatbotModel, MODEL_CHOICES
from app.auth import verify_api_key
from app.logger_config import setup_logger
from app.error_handlers import register_error_handlers
from app.middleware import RequestLoggingMiddleware


logger = setup_logger("indoor_chatbot")
inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indoor")

chatbot: IndoorChatbotModel | None = None
_model_lock = asyncio.Lock()


def _default_model_key() -> str:
    key = os.getenv("INDOOR_CHATBOT_MODEL", "3B").strip()
    return key if key in MODEL_CHOICES else "3B"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot
    model_key = _default_model_key()
    logger.info(f"챗봇 모델 로드 중: {MODEL_CHOICES[model_key]} ({model_key})")
    chatbot = IndoorChatbotModel(model_key)
    logger.info("모델 로드 완료")
    yield


app = FastAPI(
    title="Seoul Indoor Recommender API",
    description="서울 실내 활동 추천 특화 멀티턴 챗봇 API (인증 필요)",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)


def run_chat(
    messages,
    max_new_tokens,
    temperature,
    strict_indoor,
    use_rag,
    rag_top_k,
):
    if chatbot is None:
        raise RuntimeError("모델이 로드되지 않았습니다")
    return chatbot.generate_response(
        messages=[m.model_dump() for m in messages],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        strict_indoor=strict_indoor,
        use_rag=use_rag,
        rag_top_k=rag_top_k,
    )


async def ensure_model(model_key: str) -> IndoorChatbotModel:
    """요청한 모델이 아니면 교체 로드합니다. (첫 전환은 수십 초~수분 소요)"""
    global chatbot
    async with _model_lock:
        if chatbot is None or chatbot.model_key != model_key:
            logger.info(f"모델 교체 로드: {model_key} → {MODEL_CHOICES[model_key]}")
            old = chatbot
            loop = asyncio.get_running_loop()
            chatbot = await loop.run_in_executor(
                None, lambda: IndoorChatbotModel(model_key)
            )
            if old is not None:
                try:
                    del old.model
                    del old.tokenizer
                except Exception:
                    pass
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            logger.info("모델 교체 완료")
        return chatbot

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy" if chatbot else "loading",
        "model": chatbot.model_name if chatbot else None,
        "model_key": chatbot.model_key if chatbot else None,
        "available_models": list(MODEL_CHOICES.keys()),
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    user: str = Depends(verify_api_key),
):
    try:
        await ensure_model(request.model_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 로드 실패: {e}") from e

    if chatbot is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    logger.info(
        f"채팅 요청 — 사용자: {user}, 메시지 수: {len(request.messages)}, "
        f"model={request.model_key}, strict={request.strict_indoor}, "
        f"rag={request.use_rag}"
    )

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            inference_executor,
            run_chat,
            request.messages,
            request.max_new_tokens,
            request.temperature,
            request.strict_indoor,
            request.use_rag,
            request.rag_top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"텍스트 생성 실패: {str(e)}") from e

    return ChatResponse(
        success=True,
        response=result["response"],
        model_name=chatbot.model_name,
        model_key=chatbot.model_key,
        user=user,
        retried=result["retried"],
        retry_reasons=result.get("retry_reasons", []),
        outdoor_hits=result["outdoor_hits"],
        suspicious_hits=result.get("suspicious_hits", []),
        place_hits=result.get("place_hits", []),
        excluded_places=result.get("excluded_places", []),
        rag_hits=result.get("rag_hits", []),
        rag_backend=result.get("rag_backend"),
    )
