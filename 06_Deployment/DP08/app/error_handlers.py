"""글로벌 에러 핸들러"""
import traceback
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("indoor_chatbot")


def register_error_handlers(app: FastAPI):
    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        logger.error(
            f"에러 발생: {type(exc).__name__}: {exc}\n"
            f"경로: {request.method} {request.url}\n"
            f"스택 트레이스:\n{traceback.format_exc()}"
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "서버 내부 오류가 발생했습니다.",
            },
        )
