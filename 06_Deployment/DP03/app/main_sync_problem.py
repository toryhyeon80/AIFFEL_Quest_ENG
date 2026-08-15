"""
Day 3 - 섹션 3: 동기 추론의 문제점을 보여주는 서버
두 가지 버전의 엔드포인트를 비교합니다.
"""
import time
from fastapi import FastAPI

app = FastAPI(title="Sync vs Async Problem Demo")

INFERENCE_TIME = 3   # 추론에 3초 걸린다고 가정

# ===== 버전 1: async def 안에서 동기 작업 (문제 있음) =====
@app.post("/predict/blocking")
async def predict_blocking():
    """
    ⚠️ 문제 버전: async def 안에서 time.sleep (동기 블로킹)
    이벤트 루프가 멈추므로, 동시 요청을 처리할 수 없습니다.
    """
    time.sleep(INFERENCE_TIME)   # 동기 블로킹 — 이벤트 루프가 멈춤
    return {"result": "완료", "method": "blocking", "duration": INFERENCE_TIME}


# ===== 버전 2: 일반 def (FastAPI가 스레드풀에서 실행) =====
@app.post("/predict/threadpool")
def predict_threadpool():
    """
    일반 def: FastAPI가 자동으로 별도 스레드에서 실행합니다.
    이벤트 루프는 블로킹되지 않지만, 스레드풀 크기에 제한이 있습니다.
    """
    time.sleep(INFERENCE_TIME)
    return {"result": "완료", "method": "threadpool", "duration": INFERENCE_TIME}


# 헬스체크: 서버가 응답 가능한 상태인지 확인용
@app.get("/health")
async def health():
    return {"status": "healthy"}
