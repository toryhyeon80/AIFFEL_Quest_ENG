# DP08 — 서울 실내 추천 챗봇

Day 1~7을 바탕으로 만든 **자율 프로젝트**입니다.  
주제는 서울 **시내 관광 일반**이 아니라, **실내(indoors) 활동만 추천**하는 특화 챗봇입니다.

## Quick start

```bash
cd 06_Deployment/DP08
# 권장: DP07 .venv 재사용
source ../DP07/.venv/bin/activate

uvicorn app.chatbot_api:app --host 0.0.0.0 --port 8000
# 다른 터미널
streamlit run frontend/app_chatbot.py --server.port 8501
```

- API Key: `test-key-001`
- 기본 모델: **3B** (`INDOOR_CHATBOT_MODEL=1.5B`로 경량 전환 가능)
- 문서: [`DP08.md`](./DP08.md)
- 가드레일 테스트: `PYTHONPATH=. python scripts/test_guardrails.py`

## 핵심 코드

| 파일 | 역할 |
|------|------|
| `app/prompts.py` | 실내 전용 system prompt |
| `app/rag.py` | 미니 RAG (`places.json` top-k) |
| `app/guardrails.py` | 야외 키워드 휴리스틱 |
| `data/places.json` | 장소 메타 DB |
| `app/chatbot_model.py` | 생성 + 필요 시 1회 재생성 |
| `app/chatbot_api.py` | FastAPI + 모델 교체 로드 |
| `frontend/app_chatbot.py` | Streamlit UI |

## Peer review

제출/리뷰 템플릿은 `Contents/README.md`, `Final_Code/README.md`를 참고하세요.
