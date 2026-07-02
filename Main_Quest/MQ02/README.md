# 🌪️ Wind Turbine YOLO — 팀 OS 레퍼런스 프로젝트

풍력 터빈 블레이드 **Dirt/Damage** 객체 탐지 (YOLO11) + FastAPI + Streamlit

**GitHub:** https://github.com/toryhyeon80/wind-turbine-yolo  
**AIFFEL MQ02:** [Main_Quest/MQ02](https://github.com/toryhyeon80/AIFFEL_Quest_ENG/tree/main/Main_Quest/MQ02)

---

## 팀 OS (다음 프로젝트 템플릿)

| 문서 | 내용 |
| :--- | :--- |
| [docs/ROLES.md](docs/ROLES.md) | Data / AI / Service 역할 |
| [docs/WORKFLOW.md](docs/WORKFLOW.md) | Git · Colab · API 워크플로 |
| [docs/SUBMISSION.md](docs/SUBMISSION.md) | 제출 · Peer Review |
| [template/README.md](template/README.md) | 새 프로젝트 시작 가이드 |

---

## Colab

| 노트북 | 용도 |
| :--- | :--- |
| [01_eda_colab.ipynb](notebooks/01_eda_colab.ipynb) | EDA |
| [02_train_colab.ipynb](notebooks/02_train_colab.ipynb) | GPU 학습 |
| [03_api_demo_colab.ipynb](notebooks/03_api_demo_colab.ipynb) | API 테스트 |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/toryhyeon80/wind-turbine-yolo/blob/main/notebooks/02_train_colab.ipynb)

---

## 빠른 시작 (Mac M1)

```bash
pip3 install -r requirements.txt
python3 split_data.py          # 최초 1회
python3 eda.py
python3 train.py               # configs/train.yaml (mps)
python3 val.py
python3 -m streamlit run app.py   # http://localhost:8501
```

Colab: `python3 train.py --config configs/train_colab.yaml --no-report`

---

## 문서

- [report.md](report.md) — 기술 리포트
- [PRESENTATION.md](PRESENTATION.md) — 발표
- [CLAUDE.md](CLAUDE.md) — 에이전트 가이드
