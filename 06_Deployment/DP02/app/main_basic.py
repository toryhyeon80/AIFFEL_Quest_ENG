"""
최소한의 FastAPI 서버
"""
from fastapi import FastAPI

# FastAPI 인스턴스 생성
app = FastAPI(
    title="My First ML API",
    description="Day 2 실습: 첫 번째 FastAPI 서버",
    version="0.1.0",
)

# 엔드포인트 1: 헬스체크 (서버가 살아있는지 확인)
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# 엔드포인트 2: 루트 경로
@app.get("/")
def root():
    return {
        "message": "ML Model Serving API",
        "docs_url": "/docs",
    }
