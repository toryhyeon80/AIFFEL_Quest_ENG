# 제출 · Peer Review 가이드 (Submission)

> **AIFFEL MQ02:** `Main_Quest/MQ02/README.md` = Peer Review 템플릿 **유지**  
> **기술 리포트:** `report.md` · **발표:** `PRESENTATION.md`

---

## GitHub 제출 체크리스트

- [ ] 코드·`configs/`·스크립트
- [ ] `runs/` 또는 핵심 산출물 (`best.pt`, `results.csv`, 그래프)
- [ ] `report/assets/` (이미지·`predictions.json` 요약)
- [ ] `report.md` · `PRESENTATION.md`
- [ ] `.env` **미포함** (`.env.example`만)
- [ ] Peer Review `README.md` (MQ02 경로) 유지

---

## 레포 URL (본 프로젝트)

| 레포 | URL |
| :--- | :--- |
| 프로젝트 전용 | https://github.com/toryhyeon80/wind-turbine-yolo |
| AIFFEL MQ02 | https://github.com/toryhyeon80/AIFFEL_Quest_ENG/tree/main/Main_Quest/MQ02 |

---

## 팀원 복사 방법

| 목적 | 방법 |
| :--- | :--- |
| 같이 개발 | Org/Collaborator 초대 |
| 본인 계정 제출 URL | **Fork** 또는 mirror push |
| Colab만 사용 | Public 레포 + `notebooks/` Open in Colab |

---

## Peer Review (PRT) 작성 팁

| PRT 항목 | 우리 근거 파일 |
| :--- | :--- |
| 1. 완성된 코드 | `train.py`, `app.py`, `runs/` |
| 2. 핵심 주석 | `train.py` docstring, `configs/*.yaml` |
| 3. 디버깅·실험 | EXP1 ablation, `report.md` §3 |
| 4. 회고 | `report.md` 한계·로드맵 |
| 5. 코드 품질 | YAML 분리, `split_stats.py` 공용 모듈 |
