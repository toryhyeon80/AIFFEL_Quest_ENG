"""KR-FinBert 감정 분류 pipeline 로드 및 추론"""
from __future__ import annotations

from transformers import pipeline

MODEL_NAME = "snunlp/KR-FinBert-SC"

_classifier = None


def load_model():
    """text-classification pipeline을 로드합니다."""
    global _classifier
    if _classifier is None:
        _classifier = pipeline("text-classification", model=MODEL_NAME)
    return _classifier


def predict(text: str) -> dict:
    """단일 문장 감정 분류 결과를 반환합니다."""
    clf = load_model()
    out = clf(text)[0]
    return {
        "label": out["label"],
        "score": float(out["score"]),
    }
