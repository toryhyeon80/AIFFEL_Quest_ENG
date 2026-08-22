"""
미니 RAG 골격 — places.json에서 질의와 관련된 장소를 top-k로 검색합니다.

벡터 DB/임베딩 모델 없이 키워드·태그 점수로 동작합니다.
나중에 sentence-transformers 등으로 retrieve()만 교체하면 됩니다.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLACES_PATH = ROOT / "data" / "places.json"


def _tokenize(text: str) -> set[str]:
    """한글/영문/숫자 토큰 + 원문 부분문자열 매칭용 정규화."""
    text = (text or "").lower()
    parts = set(re.findall(r"[0-9a-zA-Z가-힣]{2,}", text))
    for w in (
        "비", "비오는", "아이", "어린이", "데이트", "혼자", "가족",
        "한강", "산책", "공연", "영화", "카페", "전시", "뮤지컬",
        "식물", "온실", "보드게임", "방탈출", "클라이밍", "쇼핑",
    ):
        if w in text:
            parts.add(w)
    return parts


@lru_cache(maxsize=1)
def load_places(path: str | None = None) -> tuple[dict[str, Any], ...]:
    places_path = Path(path) if path else DEFAULT_PLACES_PATH
    with places_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("places.json must be a list")
    return tuple(data)


def _score_place(query: str, place: dict[str, Any]) -> float:
    q = _tokenize(query)
    if not q:
        return 0.0

    name = place.get("name", "")
    area = place.get("area", "")
    blurb = place.get("blurb", "")
    tags = place.get("tags") or []

    score = 0.0
    # 이름/지역 직접 포함
    for token in q:
        if token and token in name.lower():
            score += 5.0
        if token and token in area.lower():
            score += 3.0
        if token and token in blurb.lower():
            score += 1.5

    tag_text = " ".join(tags).lower()
    for tag in tags:
        t = tag.lower()
        if t in query.lower():
            score += 4.0
        for token in q:
            if token == t or token in t or t in token:
                score += 2.0

    # 의도 힌트
    intent_boost = {
        "데이트": ["데이트"],
        "아이": ["아이", "가족", "어린이"],
        "어린이": ["아이", "가족", "어린이"],
        "가족": ["가족", "아이", "어린이"],
        "혼자": ["혼자"],
        "비": ["비"],
        "공연": ["공연", "뮤지컬"],
        "뮤지컬": ["뮤지컬", "공연"],
        "영화": ["영화"],
        "카페": ["카페"],
        "전시": ["전시", "미술관", "박물관"],
        "식물": ["식물", "온실"],
        "온실": ["식물", "온실", "산책대안"],
        "한강": ["한강대안", "산책대안", "공연", "휴식", "식물"],
        "산책": ["한강대안", "산책대안", "휴식", "공연", "식물"],
        "방탈출": ["방탈출", "체험"],
        "보드": ["보드게임", "체험"],
        "클라이밍": ["클라이밍", "운동"],
    }
    for key, boost_tags in intent_boost.items():
        if key in query:
            for bt in boost_tags:
                if bt in tag_text or bt in name:
                    score += 3.0

    return score


def retrieve(query: str, top_k: int = 5, path: str | None = None) -> list[dict[str, Any]]:
    """질의와 관련도 높은 장소를 top_k개 반환합니다."""
    places = list(load_places(path))
    ranked = sorted(
        (( _score_place(query, p), p) for p in places),
        key=lambda x: x[0],
        reverse=True,
    )
    hits = [p for score, p in ranked if score > 0][:top_k]
    # 전부 0점이면 다양성을 위해 앞에서 top_k
    if not hits:
        hits = places[:top_k]
    return hits


def format_rag_context(places: list[dict[str, Any]]) -> str:
    if not places:
        return ""
    lines = ["[검색 컨텍스트 — 이번 질문에 우선 참고할 실내 장소]"]
    for i, p in enumerate(places, 1):
        tags = ", ".join(p.get("tags") or [])
        lines.append(
            f"{i}. {p.get('name')} ({p.get('area')}) | 태그: {tags}\n"
            f"   - {p.get('blurb')}"
        )
    lines.append(
        "위 검색 결과를 우선 활용해 추천하되, 목록에 없는 상호는 만들지 마세요."
    )
    return "\n".join(lines)


def rag_enabled() -> bool:
    return os.getenv("INDOOR_RAG", "1").strip() not in {"0", "false", "False", "no"}
