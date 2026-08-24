"""미니 RAG 검색 스모크 테스트 (임베딩 로드 포함)."""
from app.rag import (
    retrieve,
    retrieve_keyword,
    format_rag_context,
    load_places,
    last_retrieve_backend,
)


def test_load_places():
    places = load_places()
    assert len(places) >= 40


def test_retrieve_date():
    hits = retrieve("주말에 서울에서 실내 데이트 코스 2개", top_k=5)
    names = [h["name"] for h in hits]
    assert hits
    print(f"date hits [{last_retrieve_backend()}]:", names)


def test_retrieve_hangang():
    hits = retrieve("한강 산책 대신 실내 대안", top_k=5)
    names = [h["name"] for h in hits]
    assert hits
    print(f"hangang hits [{last_retrieve_backend()}]:", names)
    ctx = format_rag_context(hits)
    assert "검색 컨텍스트" in ctx
    joined = " ".join(names)
    assert any(
        k in joined
        for k in ("온실", "식물원", "아쿠아", "예술의전당", "국립극장", "더현대", "DDP")
    )


def test_retrieve_kids():
    hits = retrieve("아이랑 갈 수 있는 서울 실내 체험", top_k=5)
    names = [h["name"] for h in hits]
    assert hits
    print(f"kids hits [{last_retrieve_backend()}]:", names)
    joined = " ".join(names)
    assert any(
        k in joined
        for k in ("키자니아", "상상나라", "애니메이션", "아쿠아", "코엑스", "롯데월드", "어린이")
    )


def test_keyword_fallback_still_works():
    hits = retrieve_keyword("한강 산책 대신 실내 대안", top_k=5)
    assert hits
    print("keyword hangang:", [h["name"] for h in hits])


def test_place_bank_sync():
    from app.prompts import PLACE_BANK

    places = load_places()
    assert len(PLACE_BANK) == len(places)
    assert PLACE_BANK[0] == places[0]["name"]


if __name__ == "__main__":
    test_load_places()
    test_retrieve_date()
    test_retrieve_hangang()
    test_retrieve_kids()
    test_keyword_fallback_still_works()
    test_place_bank_sync()
    print("rag smoke OK")
