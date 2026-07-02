# Project Template — 팀 OS 메인 틀

> **다음 프로젝트** 시작 시 이 구조를 복제하세요.  
> **원본 레퍼런스:** [wind-turbine-yolo](https://github.com/toryhyeon80/wind-turbine-yolo)

---

## 템플릿으로 새 프로젝트 만들기

### 방법 A: GitHub Template Repository (권장)

1. GitHub → **Settings** → **Template repository** ✅ (관리자)
2. 새 프로젝트: **Use this template** → Create repository
3. `docs/ROLES.md` 팀원·GitHub ID 갱신
4. `.github/CODEOWNERS` 담당자 수정

### 방법 B: 수동 복제

```bash
git clone https://github.com/toryhyeon80/wind-turbine-yolo.git my-new-project
cd my-new-project
rm -rf .git runs/
git init && git remote add origin <새-레포-URL>
```

---

## 표준 폴더 구조

```
project/
├── README.md
├── docs/
│   ├── ROLES.md
│   ├── WORKFLOW.md
│   └── SUBMISSION.md
├── configs/
│   ├── train.yaml          # Mac mps
│   ├── train_colab.yaml    # Colab cuda
│   ├── train_baseline.yaml
│   └── api.yaml
├── notebooks/              # Open in Colab
│   ├── 01_eda_colab.ipynb
│   ├── 02_train_colab.ipynb
│   └── 03_api_demo_colab.ipynb
├── scripts/                # (= 레포 루트 .py)
├── backend/
├── app.py
├── report/
│   └── assets/
├── .github/
│   ├── CODEOWNERS
│   └── pull_request_template.md
└── template/
    └── README.md           # 본 파일
```

---

## 역할별 첫 작업

| 역할 | Day 1 |
| :--- | :--- |
| Data | `split_data.py` · `eda.py` · `notebooks/01_*` |
| AI | `train_baseline.yaml` · `notebooks/02_*` |
| Service | `backend/` 스켈레톤 · `notebooks/03_*` |
| PM | `report.md` 마커 · Notion `.env.example` |

---

## Colab 배지 (README에 추가)

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ORG/REPO/blob/main/notebooks/02_train_colab.ipynb)
```

`ORG/REPO`를 실제 경로로 교체하세요.

---

## Organization 설정 체크리스트

- [ ] Org 생성 + Members 초대
- [ ] Teams: `data`, `ml`, `service`
- [ ] Branch protection on `main`
- [ ] CODEOWNERS 활성화
- [ ] (선택) GitHub Actions: lint / smoke test

---

## wind-turbine-yolo에서 검증된 패턴

| 패턴 | 파일 |
| :--- | :--- |
| YAML 설정 분리 | `configs/*.yaml`, `train.py` |
| 리포트 자동화 | `update_report.py`, `report/assets/` |
| Phase 1→2→3 | ML → FastAPI → Streamlit |
| EXP ablation | `train_exp1_small_minaug.yaml` |
| 에이전트 가이드 | `CLAUDE.md` |
