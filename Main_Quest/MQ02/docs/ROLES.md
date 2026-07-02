# 팀 역할 분담 (Roles)

> **용도:** Data / AI / Service 역할 경계 · CODEOWNERS · PR 리뷰 기준  
> **관련:** [WORKFLOW.md](WORKFLOW.md) · [SUBMISSION.md](SUBMISSION.md)

---

## 역할 요약

| 역할 | 담당 | 주요 경로 | Colab |
| :--- | :--- | :--- | :---: |
| **Data Engineer** | 수집·라벨·분할·EDA | `data/`, `split_data.py`, `eda.py`, `notebooks/01_*` | 🟡 |
| **AI Engineer** | 학습·실험·평가 | `configs/`, `train.py`, `val.py`, `predict.py`, `runs/`, `notebooks/02_*` | ✅ |
| **Service / Demo** | API·UI·데모 | `backend/`, `app.py`, `configs/api.yaml`, `notebooks/03_*` | 🟡 |
| **PM / Docs** (선택) | 리포트·발표·Notion | `report.md`, `PRESENTATION.md`, `update_report.py` | — |

---

## Data Engineer

**목표:** 재현 가능한 Train/Val 데이터와 EDA 인사이트 제공

| 작업 | 산출물 | 스크립트 |
| :--- | :--- | :--- |
| Train/Val 분할 | `data/images/*`, `data/labels/*`, `runs/split_summary.yaml` | `split_data.py` |
| EDA | `runs/eda/*.png`, `report.md` auto:eda | `eda.py` |
| 데이터 정책 문서화 | Test 미구축 등 `report.md` §1 | `data/data.yaml` |

**PR 전 체크:** 라벨 형식 YOLO · stem 1:1 매칭 · `.env` 미포함

---

## AI Engineer

**목표:** Baseline → 실험 → 최종 모델 · 메트릭 · `runs/` 산출

| 작업 | 산출물 | 설정 |
| :--- | :--- | :--- |
| Baseline | `runs/detect/baseline/` | `configs/train_baseline.yaml` |
| 실험 (EXP) | `runs/detect/exp*/` | `configs/train_exp*.yaml` |
| 본학습 | `runs/detect/train/weights/best.pt` | `configs/train.yaml` |
| 재검증 | `runs/detect/val_final/` | `val.py` |
| 일괄 추론 | `runs/predict/val_batch/predictions.json` | `predict.py` |

**로컬 (M1):** `device: mps` · `configs/train.yaml`  
**Colab (GPU):** `device: 0` · `configs/train_colab.yaml`

**PR 전 체크:** `update_report.py` 실행 · Val 기준 수치 명시

---

## Service / Demo Developer

**목표:** 탐지 모델을 API·웹 데모로 노출

| 작업 | 산출물 | 실행 |
| :--- | :--- | :--- |
| REST API | `POST /api/v1/predict` | `uvicorn backend.main:app --port 8000` |
| Live Demo | Streamlit UI | `streamlit run app.py` → `:8501` |
| Colab 연동 테스트 | API 호출 노트북 | `notebooks/03_api_demo_colab.ipynb` |

**PR 전 체크:** `best.pt` 경로 · Swagger `/docs` 동작

---

## GitHub ID 등록 (Organization 권장)

1. GitHub **Organization** 생성 (또는 기존 Org 사용)
2. 팀원을 **Members**로 초대
3. **Teams** 생성: `data`, `ml`, `service`, `docs`
4. `.github/CODEOWNERS`에 `@org/team-*` 또는 `@username` 매핑
5. `main` 브랜치 **Branch protection**: PR 필수, 1 approval

---

## 발표·심사 멘트 (역할별 한 줄)

| 역할 | 멘트 |
| :--- | :--- |
| Data | 「13,470장 8:2 분할, EDA로 극소형 Damage 93% 확인」 |
| AI | 「Baseline 0.538 → 최종 0.575, EXP1 ablation으로 증강 필요성 입증」 |
| Service | 「FastAPI + Streamlit MVP, Colab/API 연동 가능 구조」 |
