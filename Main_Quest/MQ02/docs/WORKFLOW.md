# 팀 개발 · 연구 워크플로 (Workflow)

> **Single Source of Truth:** GitHub 레포 (Colab = GPU 실행 환경)  
> **관련:** [ROLES.md](ROLES.md) · [../template/README.md](../template/README.md)

---

## 1. 전체 파이프라인

```
[Data] split → EDA
         ↓
[AI]   train → val → predict
         ↓
[Service] FastAPI → Streamlit
         ↓
[Docs] update_report.py → report.md / Notion
```

---

## 2. Git 브랜치 전략

| 브랜치 | 용도 |
| :--- | :--- |
| `main` | 제출·발표용 안정 버전 (직접 push 금지 권장) |
| `develop` | 통합 (선택) |
| `feature/data-*` | Data Engineer |
| `feature/ml-*` | AI Engineer |
| `feature/service-*` | Service Developer |

**흐름:** `feature/*` → PR → 리뷰 → `main` merge

---

## 3. 로컬 vs Colab

| 환경 | device | 설정 파일 | 용도 |
| :--- | :--- | :--- | :--- |
| **Mac M1/M2** | `mps` | `configs/train.yaml` | 일상 학습·데모 |
| **Google Colab** | `0` (CUDA) | `configs/train_colab.yaml` | GPU 학습·대량 실험 |

### Colab 시작 (공통)

1. 레포 **Public** 또는 Colab에 GitHub 연동
2. 노트북 상단 **Open in Colab** 배지 또는 `notebooks/` 실행
3. 첫 셀: clone + `pip install -r requirements.txt`
4. **Secrets:** Notion 토큰 등 (`.env` Git 커밋 금지)

### Colab → GitHub 반영

| 산출물 | 권장 |
| :--- | :--- |
| `runs/` 전체 | 용량 큼 → **요약만** PR (`results.csv`, `report/assets/`) |
| `predictions.json` | `report/assets/predict/` 복사 (GitHub UI 가시성) |
| 노트북 실험 | `.ipynb` 정리 후 `notebooks/` PR |

---

## 4. API · Colab 연동

**로컬 FastAPI는 Colab에서 접근 불가** (`localhost` 한계)

| 방법 | 사용 시점 |
| :--- | :--- |
| **ngrok** `ngrok http 8000` | 발표·데모 직전 API 테스트 |
| **클라우드 배포** | 팀 상시 API |
| **Colab 직접 추론** | API 없이 `YOLO(best.pt)` |

Colab API 테스트: `notebooks/03_api_demo_colab.ipynb`  
로컬 API: `http://localhost:8000/docs`

---

## 5. 일일 명령어 치트시트

```bash
# Data
python3 split_data.py
python3 eda.py

# AI (Mac)
python3 train.py
python3 train.py --config configs/train_baseline.yaml --no-report
python3 val.py
python3 predict.py --source data/images/val --name val_batch

# AI (Colab)
python3 train.py --config configs/train_colab.yaml --no-report

# Service
python3 -m uvicorn backend.main:app --reload --port 8000
python3 -m streamlit run app.py

# Docs
python3 update_report.py
python3 update_notion.py --skip-eda
```

---

## 6. 산출물 · Git 정책

| 경로 | Git |
| :--- | :---: |
| `data/images`, `data/labels` | ❌ (용량) |
| `data/data.yaml` | ✅ |
| `runs/` | ✅ (과제 제출 시) · 또는 LFS |
| `report/assets/` | ✅ (GitHub 미리보기용) |
| `.env` | ❌ |

---

## 7. 새 프로젝트 시작

1. GitHub **Use this template** (또는 `template/` 폴더 구조 복사)
2. `docs/ROLES.md` 팀원 GitHub ID 갱신
3. `.github/CODEOWNERS` 담당자 수정
4. `configs/train_colab.yaml` 프로젝트에 맞게 조정

자세한 템플릿: [template/README.md](../template/README.md)
