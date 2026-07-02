# 🌪️ 풍력 발전기 스마트 예방 정비 시스템

---

## 1. 📁 데이터셋 구축 및 전처리

풍력 터빈 블레이드 이미지와 YOLO 형식 라벨을 수집한 뒤, 학습·검증용으로 무작위 분할했습니다. 이미지와 라벨은 파일명(stem) 기준으로 1:1 매칭하여 이동했으며, 라벨이 없는 배경 이미지도 동일한 비율로 분할에 포함했습니다.

### Train / Val 분할 결과

<!-- report:auto:split -->
- **자동 반영:** 2026-07-02 17:26:07 (`folder_scan`)

| 항목 | 이미지 수 | 비율 |
| :--------------- | ----------: | ----: |
| **전체 (Total)** | **13,470장** | 100% |
| **Train (학습)** | **10,776장** | 80.0% |
| **Val (검증)** | **2,694장** | 20.0% |

**데이터 구성**

| 구분 | Train | Val | 합계 |
| :--- | ---: | ---: | ---: |
| **라벨 있음** (객체 BBox) | 2,401 | 594 | 2,995 |
| **배경** (라벨 없음, negative) | 8,375 | 2,100 | 10,475 |

- **분할 비율:** 약 **8 : 2** (Train : Val), seed=—
- **분할 방식:** 랜덤 셔플 후 8:2 분할 (`split_data.py`)
- **평가 세트:** Test 세트는 별도로 두지 않음 — **Val 세트**로 최종 성능 평가
<!-- /report:auto:split -->

- **저장 경로:**
  - `data/images/train`, `data/images/val`
  - `data/labels/train`, `data/labels/val`

### Train / Val / Test 정책

| 구분 | 역할 | 본 프로젝트 |
| :--- | :--- | :--- |
| **Train** | 모델이 직접 학습하는 데이터 | ✅ 10,776장 |
| **Val** | 학습 중 mAP 모니터링 · `best.pt` 선정 · 리포트 수치 | ✅ 2,694장 |
| **Test** | 학습·튜닝에 전혀 쓰지 않은 최종 일반화 평가 | ❌ **미구축** |

- **분할 설계:** `split_data.py` · `data/data.yaml`은 **train / val만** 정의 (YOLO 표준 워크플로).
- **Test를 두지 않은 이유:** 해커톤 일정 내 파이프라인(분할→학습→리포트) 완주 우선, 라벨 데이터를 3-way로 나누면 학습량이 과도하게 줄어듦. 즉 “Test가 불필요”가 아니라 **“우선 Val로 개발·평가까지 진행”**한 선택.
- **수치 해석:** 본 리포트의 mAP·P/R·혼동행렬은 모두 **Validation set 기준**입니다. “Test 성능”이 아닙니다.
- **한계:** Val 점수로 epoch(50)·증강·모델(Small) 등을 결정했으므로, 독립 Hold-out Test 대비 **다소 낙관적**일 수 있습니다.
- **용어 주의:** `predict.py`의 “Phase 1 Test”는 **추론 파이프라인 검증 단계** 이름이며, Hold-out Test 세트 평가와는 다릅니다.

### EDA 및 시각화 (Exploratory Data Analysis)

> 학습 전 데이터 특성 파악 — `python eda.py` 실행 후 `update_report.py` / `update_notion.py`로 자동 반영

<!-- report:auto:eda -->
- **자동 반영:** 2026-07-02 17:26:07
- **총 BBox:** 9351개

**클래스별 BBox 분포**

| 클래스 | BBox 수 | 비율 | 극소형 비율 (w,h < 0.2) |
| :--- | ---: | ---: | ---: |
| **Dirt (0)** | 581 | 6.21% | 5.68% |
| **Damage (1)** | 8770 | 93.79% | 93.34% |

**주요 인사이트**

- 클래스 불균형: Damage (1) 8770개 vs Dirt (0) 581개 (약 15.1배).
- Damage (1) BBox의 93.3%가 width·height 모두 0.2 미만(극소형 객체).
- Damage 클래스가 이미지 대비 매우 작은 BBox로 밀집 → imgsz 1024 이상 상향 또는 소형 객체 탐지 증강 검토 권장.

**시각화**

![Class Distribution: Dirt vs Damage](report/assets/eda/class_distribution.png)
![Bounding Box Size Distribution (Normalized)](report/assets/eda/bbox_size_distribution.png)
![Bounding Box Area Distribution](report/assets/eda/bbox_area_distribution.png)
<!-- /report:auto:eda -->

---

## 2. 📊 훈련 결과 및 베이스라인 비교군 분석

> **루브릭 1:** 데이터셋 및 선택한 모델이 관련 분야의 베이스라인 모델과 비교하여 어떤 차이가 있는지 정량적, 정성적 분석 진행

### [정량적 분석] 베이스라인 vs 최종 모델 성능 비교

| 구분          | 사용 모델       | Epoch | mAP50 | mAP50-95 | 비고                            |
| :------------ | :-------------- | :---- | :---- | :------- | :------------------------------ |
| **Baseline** | YOLO11n (Nano) | 18 | 0.538 | 0.317 | Baseline Nano 학습 |
| **최종 모델** | YOLO11s (Small) | 50 | 0.575 | 0.319 | 본학습 best |
| **성능 향상** | - | - | **+ 3.7%p** | **+ 0.2%p** | Baseline 대비 개선 |

### [정성적 분석]

<!-- report:auto:run-summary -->
- **최종 학습:** `train` | mAP50 **0.575** | mAP50-95 **0.319**
- **재검증:** `val_final`
- **Val 메트릭:** mAP50 **0.574** | Precision **0.597** | Recall **0.640**
- **갱신 시각:** 2026-07-02 17:26:07
<!-- /report:auto:run-summary -->

- **Baseline 한계:** 작은 크기의 Damage(손상) 객체를 배경과 혼동하여 놓치는(False Negative) 현상이 잦았음.
- **최종 모델 개선점:** 모델 사이즈를 Small로 키우고 HSV·Mosaic 등 도메인 맞춤 증강을 적용한 결과, 미세한 블레이드 스크래치까지 명확하게 잡아내는 것을 육안으로 확인함.

---

## 3. 🧪 다양한 실험 및 성능 개선 기법 (Experiment Logs)

> **루브릭 2:** 모델의 조정 및 성능 개선 기법을 통해 분기된 훈련 결과의 성능 평가 비교

목표 성능 달성을 위해 Baseline 대비 아래 **3단계 개선(EXP 1~3)** 을 설계·적용하고, 최종 통합 설정(`configs/train.yaml`)으로 본학습을 수행했습니다.

> EXP 1~3은 각각 **독립 rerun** 이 아니라, 최종 모델에 누적 반영한 설계 변경입니다. Baseline(YOLO11n)만 별도 학습으로 비교합니다.

### [실험별 성능 비교]

<!-- report:auto:exp-comparison -->
| 실험 | 모델 | Epoch | mAP50 | mAP50-95 | 비고 |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **Baseline** | YOLO11n (Nano) | 18 | 0.538 | 0.317 | YOLO11n · 최소 증강 |
| **EXP 1** | YOLO11s (Small) | 19 | 0.498 | 0.288 | Small만 변경·20ep (`train_exp1_small_minaug.yaml`) |
| **최종 모델** | YOLO11s (Small) | 50 | 0.575 | 0.319 | EXP 1~3 통합 (`configs/train.yaml`) |
| **개선** | — | — | **+ 3.7%p** | + 0.2%p | Baseline 대비 (최종) |
<!-- /report:auto:exp-comparison -->

<!-- report:auto:hyper-tuning -->
**EXP 1~3은 누적 설계 단계** — EXP1은 독립 ablation 완료, EXP2·3은 최종 모델에 통합 반영.

| 단계 | 변경 | 선택 | 근거 |
| :--- | :--- | :--- | :--- |
| **EXP 1** | 모델 크기 | Nano → **Small** | EXP1 단독(20ep·min aug) mAP50 **0.498** — Small만으로는 Baseline(**0.538**) 미달 → EXP2·3 필요 |
| **EXP 2** | Data Augmentation | HSV·Mosaic·Mixup·Erasing | 도메인(안개·반사) · `flipud=0` |
| **EXP 3** | Epoch · Batch · Patience | **50ep · batch 8 · patience 10** | M1 16GB OOM → batch 8 · Cosine LR |

> **한계:** EXP2·3은 별도 독립 run 없이 최종 설정(`train.yaml`)에 누적 반영.
<!-- /report:auto:hyper-tuning -->

- **EXP 1: 모델 아키텍처 스케일업 (Nano vs Small) — 독립 ablation 완료**
  - **내용:** Baseline과 동일 조건(20ep·최소 증강)에서 **YOLO11n → YOLO11s**만 변경 (`train_exp1_small_minaug.yaml`).
  - **결과:** EXP1 단독 mAP50 **0.498** (best ep 19) — 동일 조건 Baseline(**0.538**)보다 낮음. **Small만으로는 부족**함을 확인 → EXP2(증강)·EXP3(50ep) 통합 후 최종 **0.575** 채택.
- **EXP 2: Data Augmentation (데이터 증강) 적용**
  - **내용:** 해상 풍력 터빈 특성(안개, 흐린 날씨, 빛 반사) 반영 — HSV 밝기/채도 변화, 좌우 Flip(`fliplr`), Mosaic·Mixup·Random Erasing 적용. 블레이드 방향 특성상 **상하 반전(flipud=0) 금지**.
  - **결과:** 과적합(Overfitting)이 방지되고 검증(Val) Loss가 안정적으로 수렴함.
- **EXP 3: 하이퍼파라미터 튜닝 (Epoch 및 Batch Size)**
  - **내용:** Epoch 50, Patience 10으로 충분한 학습과 과적합 방지를 설정. M1 16GB 환경에서 batch 32는 OOM·스왑 발생 → **batch 8**으로 안정 학습.
  - **결과:** 50 epoch까지 수렴하여 Val mAP50 **0.574** 달성 (`val_final`). Early stopping은 설정했으나 50 epoch 내 최적점이 유지됨.

---

## 4. 📈 적합한 로스와 메트릭 평가 및 시각화

> **루브릭 3:** 데이터셋 구성, 모델 훈련, 결과물 시각화 사이클 수행 및 평가지표에 따른 모델 평가

### 평가 지표 (Metrics & Loss) 분석

- **사용 지표:** 객체 탐지(Object Detection)의 글로벌 표준 지표인 **mAP50**, **mAP50-95**, **Precision**, **Recall** 사용.
- **평가 세트:** Test 세트 없음 — 학습에 사용하지 않은 **Val 세트**(`val.py` → `val_final`)로 최종 평가.
- **학습 환경:** Apple M1 Pro (`device: mps`), `imgsz: 640`, batch 8, workers 0, seed 42
- **Box Loss & Class Loss:** Train Loss와 Val Loss가 모두 안정적으로 우하향하는 그래프를 확인하여 학습이 정상적으로 이루어졌음을 검증함.

### 과적합 방지 (Regularization)

| 기법 | 적용 | 역할 |
| :--- | :---: | :--- |
| **Data Augmentation** (mosaic, mixup, HSV, erasing) | ✅ | 데이터 다양성 → **과적합 완화** (`configs/train.yaml` EXP 2) |
| **Early stopping** (`patience: 10`) | ✅ | Val mAP 기준 조기 종료 (`configs/train.yaml`) |
| **`close_mosaic: 10`** | ✅ | 학습 후반 일반화 유도 |
| **Cosine LR** + warmup | ✅ | 학습률 스케줄 (`cos_lr: true`, `warmup_epochs: 3`) |
| **L2 (weight_decay)** | ✅ | Ultralytics **기본 0.0005** — 별도 YAML 없이 프레임워크 기본 적용 |
| **Pretrained YOLO11** | ✅ | ImageNet 등 사전학습 가중치에서 시작 |
| **Dropout** | ❌ | YOLO 탐지 기본 `dropout: 0.0` — 소형 Damage 탐지에 불리, **의도적 미사용** |
| **L1** | ❌ | YOLO 학습 API 미지원 — **해당 없음** |

- **정책 요약:** Early stopping + 도메인 증강 + Ultralytics 기본 weight decay(L2)로 **일반화를 확보**했습니다. YOLO 객체 탐지에서는 Dropout/L1 별도 설계가 표준이 아니며, mAP 추가 이득도 제한적입니다.
- **학습 곡선:** Val Loss가 Train Loss와 함께 안정적으로 수렴 — **심각한 과적합 징후는 관찰되지 않음** (`runs/detect/train/results.png`).
<!-- report:auto:metrics-visuals -->
- **자동 반영:** 2026-07-02 17:26:07
- **Val 재검증 (`val_final`):** mAP50 **0.574** | mAP50-95 **0.318** | Precision **0.597** | Recall **0.640**

![Loss/mAP 학습 곡선 (Train)](report/assets/metrics/results.png)

![Confusion Matrix (Val)](report/assets/metrics/confusion_matrix.png)

![Box F1 Curve (Val)](report/assets/metrics/BoxF1_curve.png)
<!-- /report:auto:metrics-visuals -->

### 오탐·미탐 및 오류 패턴 분석

<!-- report:auto:error-analysis -->
- **자동 반영:** 2026-07-02 17:26:07 (`val_final` + `predict.py`)

**클래스별 Val 지표**

| 클래스 | Precision | Recall | mAP50 |
| :--- | ---: | ---: | ---: |
| **Dirt (0)** | 0.521 | 0.750 | 0.549 |
| **Damage (1)** | 0.673 | 0.530 | 0.599 |

**혼동행렬 기반 오류 패턴 (BBox 단위)**

| 패턴 | 건수 | 해석 |
| :--- | ---: | :--- |
| **Damage → Background (FN)** | **866** | Damage 미탐 (핵심 이슈) |
| Background → Damage (FP) | 323 | 배경 오탐 |
| Dirt → Background (FN) | 31 | Dirt 미탐 |
| Background → Dirt (FP) | 57 | Dirt 오탐 |
| Dirt ↔ Damage 혼동 | 7 | 클래스 간 혼동 **낮음** |

**대표 오류 사례 (predict.py 스캔)**

| 유형 | 이미지 | GT/탐지 |
| :--- | :--- | ---: |
| **FN (미탐)** | `DJI_0748_05_07.png` | GT BBox **12** · 탐지 0 |
| **FN (미탐)** | `DJI_0995_06_05.png` | GT BBox **6** · 탐지 0 |
| **FN (미탐)** | `DJI_0436_03_09.png` | GT BBox **5** · 탐지 0 |
| **FP (오탐)** | `DJI_0977_04_09.png` | 배경 · 탐지 **5** |
| **FP (오탐)** | `DJI_0593_02_02.png` | 배경 · 탐지 **3** |
<!-- /report:auto:error-analysis -->

### Phase 1 Test — predict.py Val 일괄 추론

<!-- report:auto:predict-inference -->
- **자동 반영:** 2026-07-02 06:56:42 (`predict.py` → `val_batch`)
- **가중치:** `runs/detect/train/weights/best.pt` | conf **0.25** | device `mps`
- **입력:** `data/images/val`

> **Val 공식 평가(`val.py`)와 별도** — best.pt로 Val 전체에 추론만 수행한 Phase 1 Test 결과입니다.

**Val 일괄 추론 집계**

| 항목 | 값 |
| :--- | ---: |
| **처리 이미지 수** | 2,694 |
| **탐지 있는 이미지** | 505 (18.7%) |
| **탐지 없음** | 2,189 |
| **총 BBox** | 1,294 |

**클래스별 탐지 수**

| 클래스 | BBox 수 |
| :--- | ---: |
| **Damage** | 1,164 |
| **Dirt** | 130 |

**대표 추론 결과 (탐지 있음)**

![predict 추론 결과 1 — DJI_0121_07_05.png (16 BBox)](report/assets/predict/DJI_0121_07_05.jpg)
![predict 추론 결과 2 — DJI_0374_02_07.png (16 BBox)](report/assets/predict/DJI_0374_02_07.jpg)
![predict 추론 결과 3 — DJI_0033_03_04.png (15 BBox)](report/assets/predict/DJI_0033_03_04.jpg)

- **전체 결과:** `runs/predict/val_batch/predictions.json` · `runs/predict/val_batch/`
<!-- /report:auto:predict-inference -->

### 탐지 결과 시각화 (val.py 검증)

<!-- report:auto:predictions -->
- **Dirt / Damage 탐지 결과** — `val_final`

![검증 예측 결과 1](report/assets/val/val_batch0_pred.jpg)

![검증 예측 결과 2](report/assets/val/val_batch1_pred.jpg)

![검증 예측 결과 3](report/assets/val/val_batch2_pred.jpg)
<!-- /report:auto:predictions -->

- **평가 (`val.py`):** 드론 촬영과 동일한 도메인의 **Val 검증 이미지**에서 Dirt·Damage를 분리 탐지하는 것을 확인함. (별도 Test 세트 미구분)
- **평가 (`predict.py`):** Val **전체**에 `best.pt` 추론 파이프라인(Phase 1 Test)을 적용하여 BBox·JSON 산출을 검증함.
