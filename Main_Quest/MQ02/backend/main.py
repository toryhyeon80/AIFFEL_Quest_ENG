"""
backend/main.py — Phase 2 FastAPI MVP

실행:
  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

문서:
  http://localhost:8000/docs
"""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.inference import IMAGE_EXTENSIONS, engine, load_api_config

ROOT = Path(__file__).resolve().parent.parent
API_CONFIG = ROOT / "configs" / "api.yaml"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 기동 시 YOLO 가중치 1회 로드 (요청마다 reload 방지)."""
    engine.load(API_CONFIG)
    yield


app = FastAPI(
    title="Wind Turbine YOLO API",
    description="풍력 터빈 블레이드 Dirt/Damage 객체 탐지 API (YOLO11s)",
    version="0.1.0",
    lifespan=lifespan,
)

_api_cfg = load_api_config(API_CONFIG)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_api_cfg.get("cors_origins", ["http://localhost:3000"])),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/model")
def model_info() -> dict[str, Any]:
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")
    return engine.model_info()


@app.post("/api/v1/predict")
@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    이미지 1장 업로드 → BBox·클래스·confidence JSON 반환.
    응답 형식은 predict.py `result_to_dict` + 클래스별 summary.
    """
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    filename = file.filename or "upload.jpg"
    suffix = Path(filename).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. 허용: {', '.join(sorted(IMAGE_EXTENSIONS))}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        return engine.predict_path(tmp_path, source_label=filename)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
