"""가드레일 단위 테스트 (모델 로드 없음)."""
from app.guardrails import (
    looks_outdoor,
    outdoor_hits,
    looks_suspicious_place,
    known_place_hits,
)


def test_detects_park():
    text = "주말에 한강공원에서 산책하는 걸 추천해요."
    assert looks_outdoor(text) is True
    assert "한강" in outdoor_hits(text) or "공원" in outdoor_hits(text)


def test_allows_museum():
    text = "국립현대미술관에서 전시를 보고, 근처 카페에서 쉬어보세요."
    assert looks_outdoor(text) is False
    assert outdoor_hits(text) == []


def test_suspicious_fake_venue():
    text = "서울 실내 체험관에서 따뜻한 분위기를 즐기세요."
    assert looks_suspicious_place(text) is True


def test_known_place_match():
    text = "국립중앙박물관과 별마당도서관이 좋아요."
    hits = known_place_hits(text)
    assert any("국립중앙박물관" in h for h in hits)
    assert any("별마당도서관" in h for h in hits)


def test_previously_recommended():
    from app.guardrails import previously_recommended_places

    messages = [
        {"role": "user", "content": "추천해줘"},
        {"role": "bot", "content": "1) 국립중앙박물관 — 전시\n2) 디뮤지엄 — 전시"},
        {"role": "user", "content": "다른 거"},
    ]
    prev = previously_recommended_places(messages)
    assert any("국립중앙박물관" in p for p in prev)
    assert any("디뮤지엄" in p for p in prev)


if __name__ == "__main__":
    test_detects_park()
    test_allows_museum()
    test_suspicious_fake_venue()
    test_known_place_match()
    test_previously_recommended()
    print("guardrail tests OK")
