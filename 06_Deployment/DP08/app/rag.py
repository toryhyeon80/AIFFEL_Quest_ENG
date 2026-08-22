"""
미니 RAG — places.json에서 질의와 관련된 장소를 top-k로 검색합니다.

기본: sentence-transformers 임베딩(CPU) + 키워드 점수 하이브리드.
폴백: 키워드만 (`INDOOR_RAG_BACKEND=keyword` 또는 임베딩 로드 실패 시).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLACES_PATH = ROOT / "data" / "places.json"
CACHE_DIR = ROOT / "data" / ".cache"
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

logger = logging.getLogger("indoor_chatbot")

_LAST_BACKEND = "keyword"


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


def _place_text(place: dict[str, Any]) -> str:
    tags = ", ".join(place.get("tags") or [])
    return (
        f"{place.get('name', '')}. {place.get('area', '')}. "
        f"태그: {tags}. {tags}. {place.get('blurb', '')}"
    )


def _score_place(query: str, place: dict[str, Any]) -> float:
    q = _tokenize(query)
    if not q:
        return 0.0

    name = place.get("name", "")
    area = place.get("area", "")
    blurb = place.get("blurb", "")
    tags = place.get("tags") or []

    score = 0.0
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


def retrieve_keyword(
    query: str, top_k: int = 5, path: str | None = None
) -> list[dict[str, Any]]:
    """키워드·태그 점수 검색 (임베딩 폴백)."""
    places = list(load_places(path))
    ranked = sorted(
        ((_score_place(query, p), p) for p in places),
        key=lambda x: x[0],
        reverse=True,
    )
    hits = [p for score, p in ranked if score > 0][:top_k]
    if not hits:
        hits = places[:top_k]
    return hits


def _embed_model_name() -> str:
    return os.getenv("INDOOR_RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL).strip()


def _wanted_backend() -> str:
    return os.getenv("INDOOR_RAG_BACKEND", "embedding").strip().lower()


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    # Qwen(3B)가 MPS/CUDA를 쓰므로 임베딩은 CPU에 고정합니다.
    model = SentenceTransformer(_embed_model_name(), device="cpu")
    logger.info("RAG embedder loaded: %s (cpu)", _embed_model_name())
    return model


def _places_fingerprint(places: tuple[dict[str, Any], ...], model_name: str) -> str:
    payload = json.dumps(
        [_place_text(p) for p in places],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(f"{model_name}\n{payload}".encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=4)
def _place_matrix(path: str | None = None) -> tuple[tuple[dict[str, Any], ...], np.ndarray]:
    places = load_places(path)
    model_name = _embed_model_name()
    fp = _places_fingerprint(places, model_name)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"place_emb_{fp}.npy"

    if cache_path.exists():
        matrix = np.load(cache_path)
        if matrix.shape[0] == len(places):
            return places, matrix

    texts = [_place_text(p) for p in places]
    matrix = _embedder().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    matrix = np.asarray(matrix, dtype=np.float32)
    np.save(cache_path, matrix)
    return places, matrix


def retrieve_embedding(
    query: str, top_k: int = 5, path: str | None = None
) -> list[dict[str, Any]]:
    """임베딩 코사인 + 키워드 점수를 섞어 순위화합니다."""
    places, matrix = _place_matrix(path)
    q_vec = _embedder().encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    q_vec = np.asarray(q_vec, dtype=np.float32)[0]
    emb = matrix @ q_vec

    kw = np.array([_score_place(query, p) for p in places], dtype=np.float32)
    if kw.max() > 0:
        kw = kw / kw.max()

    scores = 0.6 * emb + 0.4 * kw

    query_l = query.lower()
    for i, place in enumerate(places):
        name = (place.get("name") or "").lower()
        core = name.split("(")[0].strip()
        if core and core in query_l:
            scores[i] += 0.12

    order = np.argsort(-scores)[:top_k]
    return [places[int(i)] for i in order]


def retrieve(query: str, top_k: int = 5, path: str | None = None) -> list[dict[str, Any]]:
    """질의와 관련도 높은 장소를 top_k개 반환합니다."""
    global _LAST_BACKEND
    wanted = _wanted_backend()
    if wanted in {"keyword", "keywords", "lexical"}:
        _LAST_BACKEND = "keyword"
        return retrieve_keyword(query, top_k=top_k, path=path)

    try:
        hits = retrieve_embedding(query, top_k=top_k, path=path)
        _LAST_BACKEND = "embedding"
        return hits
    except Exception as exc:
        logger.warning("embedding retrieve failed (%s); falling back to keyword", exc)
        _LAST_BACKEND = "keyword"
        return retrieve_keyword(query, top_k=top_k, path=path)


def last_retrieve_backend() -> str:
    return _LAST_BACKEND


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
