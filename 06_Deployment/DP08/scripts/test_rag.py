"""미니 RAG 검색 스모크 테스트 (모델 로드 없음)."""
from app.rag import retrieve, format_rag_context, load_places


def test_load_places():
    places = load_places()
    assert len(places) >= 40


def test_retrieve_date():
    hits = retrieve("주말에 서울에서 실내 데이트 코스 2개", top_k=5)
    names = [h["name"] for h in hits]
    assert hits
    print("date hits:", names)


def test_retrieve_hangang():
    hits = retrieve("한강 산책 대신 실내 대안", top_k=5)
    names = [h["name"] for h in hits]
    assert hits
    # 한강/산책 대안 태그가 있는 후보가 포함되는 것이 이상적
    print("hangang hits:", names)
    ctx = format_rag_context(hits)
    assert "검색 컨텍스트" in ctx


def test_retrieve_kids():
    hits = retrieve("아이랑 갈 수 있는 서울 실내 체험", top_k=5)
    names = [h["name"] for h in hits]
    assert hits
    print("kids hits:", names)


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
    test_place_bank_sync()
    print("rag smoke OK")
