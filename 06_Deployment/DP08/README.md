# DP08 — 자율 프로젝트 (노트북 baseline + indoor 확장)

공식 [`DP08.ipynb`](./DP08.ipynb) 기준으로 **두 트랙**으로 구성했습니다.

| 트랙 | 설명 | 경로 |
|------|------|------|
| **Baseline (LMS 제출·데모)** | 한국어 감정 분석 `snunlp/KR-FinBert-SC`, `POST /predict` | 루트 `app/`, `frontend/app.py` |
| **Indoor 확장** | 서울 실내 추천 Qwen + RAG + 가드레일 | [`indoor/`](./indoor/) |

제출 문서: [`DP08.md`](./DP08.md)

## 1. Pipeline baseline (포트 8000)

```bash
cd 06_Deployment/DP08
source ../DP07/.venv/bin/activate   # 또는 python3 -m venv .venv
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000
# 다른 터미널
streamlit run frontend/app.py --server.port 8501
```

- API Key: `test-key-001`
- Swagger: http://localhost:8000/docs
- `POST /predict` — `{ "text": "..." }` → `{ "label", "score", "success" }`

## 2. Indoor chatbot 확장 (포트 8001)

```bash
cd 06_Deployment/DP08/indoor
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8001
# 다른 터미널
streamlit run frontend/app.py --server.port 8502
```

- `POST /predict` — 단일 질문 실내 추천
- `POST /chat` — 멀티턴 + RAG + 가드레일
- 테스트: `PYTHONPATH=. python scripts/test_rag.py`

## 노트북 연동 (Mac)

[`DP08.ipynb`](./DP08.ipynb) 은 **교안 + 데모** 용도입니다. baseline 코드는 이미 구현되어 있어 **전체 실행은 필수 아님** (제출은 [`DP08.md`](./DP08.md)).

Mac에서 노트북으로 데모할 때:
1. 커널: `../DP07/.venv` 선택
2. 서버 도우미 셀 → 섹션 4.2 서버 실행 → 4.3 API 테스트

터미널만 써도 됩니다: `uvicorn app.main:app --port 8000`

## Peer review

`Contents/README.md`, `Final_Code/README.md` 참고.
