"""
Day 3 - 섹션 4: 세 가지 동시 처리 방식 비교
"""
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI

app = FastAPI(title="Async Solution Demo")

INFERENCE_TIME = 3

# 커스텀 스레드풀 생성 (최대 4개 스레드)
inference_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="inference")


def heavy_inference():
    """동기 함수: 모델 추론을 시뮬레이션합니다."""
    time.sleep(INFERENCE_TIME)
    return {"result": "완료", "duration": INFERENCE_TIME}


# ===== 버전 1: async def + 동기 작업 (문제 있음) =====
@app.post("/predict/v1-blocking")
async def predict_v1():
    """❌ 이벤트 루프를 막습니다."""
    time.sleep(INFERENCE_TIME)
    return {"method": "v1-blocking", "duration": INFERENCE_TIME}


# ===== 버전 2: 일반 def (FastAPI 자동 스레드풀) =====
@app.post("/predict/v2-def")
def predict_v2():
    """⭕ FastAPI가 자동으로 별도 스레드에서 실행합니다."""
    time.sleep(INFERENCE_TIME)
    return {"method": "v2-def", "duration": INFERENCE_TIME}


# ===== 버전 3: async def + run_in_executor (권장) =====
@app.post("/predict/v3-executor")
async def predict_v3():
    """✅ 명시적으로 스레드풀에 위임합니다."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        inference_executor,    # 커스텀 스레드풀 사용
        heavy_inference,       # 실행할 동기 함수
    )
    return {"method": "v3-executor", **result}


@app.get("/health")
async def health():
    return {"status": "healthy"}
