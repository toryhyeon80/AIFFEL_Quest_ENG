# Indoor extension — 서울 실내 추천 챗봇

루트 baseline(pipeline)과 분리된 **Day7 확장** 프로젝트입니다.

```bash
cd indoor
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001
streamlit run frontend/app.py --server.port 8502
```

- `POST /predict` — 단일 질문
- `POST /chat` — 멀티턴 + RAG + 가드레일
- 테스트: `PYTHONPATH=. python scripts/test_rag.py`

상세: [`../Final_Code/DP08.md`](../Final_Code/DP08.md)
