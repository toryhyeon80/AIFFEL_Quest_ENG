"""미니 RAG 검색 스모크 테스트 (모델 로드 없음)."""
from app.rag import retrieve, format_rag_context, load_places


def test_load_places():
    places = load_places()
    assert len(places) >= 20


def test_retrieve_date():
    hits = retrieve("주말에 서울에서 실내 데이트 코스 2개", top_k=5)
    names = [h["name"] for h in hits]
    assert hits
    # 데이트 태그가 있는 장소가 상위에 올 가능성이 큼
    print("date hits:", names)


def test_retrieve_hangang():
    hits = retrieve("한강 산책 대신 실내 대안", top_k=5)
    names = [h["name"] for h in hits]
    assert hits
    print("hangang hits:", names)
    ctx = format_rag_context(hits)
    assert "검색 컨텍스트" in ctx


if __name__ == "__main__":
    test_load_places()
    test_retrieve_date()
    test_retrieve_hangang()
    print("rag smoke OK")
