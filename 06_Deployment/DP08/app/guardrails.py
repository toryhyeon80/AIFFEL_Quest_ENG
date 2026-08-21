"""
실내 추천 가드레일 — 야외 키워드 / 의심 환각 상호 휴리스틱.
완벽한 검증이 아니라 데모용입니다.
"""
from app.prompts import (
    OUTDOOR_KEYWORDS,
    PLACE_BANK,
    SUSPICIOUS_FAKE_PATTERNS,
)


def looks_outdoor(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(kw.lower() in lowered for kw in OUTDOOR_KEYWORDS)


def outdoor_hits(text: str) -> list[str]:
    if not text:
        return []
    return [kw for kw in OUTDOOR_KEYWORDS if kw.lower() in text.lower()]


def looks_suspicious_place(text: str) -> bool:
    """모호·가짜로 자주 나오는 상호 패턴이 있는지 검사합니다."""
    if not text:
        return False
    return any(pat in text for pat in SUSPICIOUS_FAKE_PATTERNS)


def suspicious_hits(text: str) -> list[str]:
    if not text:
        return []
    return [pat for pat in SUSPICIOUS_FAKE_PATTERNS if pat in text]


def known_place_hits(text: str) -> list[str]:
    """응답에 등장한 검증 장소 목록 항목을 반환합니다."""
    if not text:
        return []
    hits = []
    for name in PLACE_BANK:
        # 목록 항목이 길어서 괄호 설명이 있으면 앞부분만으로도 매칭
        core = name.split("(")[0].split("·")[0].strip()
        if core and core in text:
            hits.append(name)
            continue
        if name in text:
            hits.append(name)
    # 중복 제거 (순서 유지)
    seen = set()
    unique = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            unique.append(h)
    return unique


def previously_recommended_places(messages: list[dict]) -> list[str]:
    """이전 assistant/bot 메시지에서 이미 추천된 목록 장소를 모읍니다."""
    seen: set[str] = set()
    ordered: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        if role not in ("bot", "assistant"):
            continue
        for place in known_place_hits(msg.get("content", "")):
            if place not in seen:
                seen.add(place)
                ordered.append(place)
    return ordered
