# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 최승현
- 리뷰어 : 정슬기


# PRT(Peer Review Template)
- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - 중요! 해당 조건을 만족하는 부분을 캡쳐해 근거로 첨부

    **→ 충족.** `# *your code*` 빈칸 14곳이 모두 채워졌고, **모델 학습 → 저장 → FastAPI → Streamlit → 통합 테스트**까지 실행 출력이 노트북에 그대로 남아 있습니다.

    - 학습/평가 (셀 21~22): `Epoch 50/50 — Loss: 0.3894`, `테스트 MAE: $38,666`
    - 저장 (셀 24): `✅ 모델 저장: models/housing_model.pth (13.3 KB)` + `housing_preprocessing.json`
    - API 실행 (셀 36, 40):
      ```
      2026-08-19 12:39:27 INFO [housing_api] 모델 로드 완료
      서버 실행됨: http://127.0.0.1:8000
      상태 코드: 200 / 예측 가격: $180,799
      ```
    - Streamlit 대시보드 (셀 47, 50): `✅ 프론트엔드: http://localhost:8501` + 실행 화면 캡처 이미지 첨부됨
    - 통합 테스트 종합 (셀 64):
      ```
      ✅ 정상 요청: 다양한 입력에서 200과 예측 가격 반환
      ✅ 에러 처리: 잘못된 입력을 4xx로 거부, 서버 안 죽음
      ✅ 동시 처리: 8개 동시 요청 모두 200
      ✅ 헬스체크: 서버 상태 healthy
      🎉 4가지 테스트를 모두 통과했습니다.
      ```
    - 모델이 학습 방향을 제대로 잡았는지도 확인됩니다 (셀 56). 저소득 지역 $106,548 < 평균 $180,799 < 고소득 지역 $487,085 로 소득 피처에 대한 단조 반응이 나옵니다.

- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 확인
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드의 기능, 존재 이유, 작동 원리 등을 기술했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

    **→ 충족.** 이 프로젝트에서 가장 이해하기 어려운 곳은 **비동기 API에서 동기 추론을 스레드풀로 넘기는 부분**(`app/housing_api.py`)과 **학습-배포 사이의 전처리 계약**(`HousingPredictor.predict`)입니다. 두 곳 모두 "무엇을/왜/안 하면 어떻게 되는지"가 주석에 다 적혀 있습니다.

    **(1) `app/housing_api.py` — 스레드풀 + run_in_executor (셀 34)**
    ```python
    # 추론 전용 스레드풀 (Day 3에서 배운 패턴)
    # PyTorch 추론은 CPU를 오래 쓰는 동기 작업이라, async def 안에서 그대로 호출하면
    # 이벤트 루프가 멈춰 다른 /health 요청까지 지연됩니다.
    # ThreadPoolExecutor에 넘기면 루프는 비워 두고, 스레드에서 predict()만 실행합니다.
    # max_workers=4: 동시에 최대 4개 추론. thread_name_prefix는 로그에서 스레드를 구분할 때 사용.
    inference_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="housing")

    result = await loop.run_in_executor(
        inference_executor,   # 추론 전용 스레드풀
        predictor.predict,    # 실행할 동기 함수
        features,             # 함수에 전달할 인자
    )
    ```
    `run_in_executor(executor, func, *args)`의 인자 3개가 각각 무슨 역할인지까지 한 줄씩 달려 있어, 이 API를 처음 보는 사람도 "왜 `await` 뒤에 동기 함수가 오는지"를 바로 이해할 수 있습니다. `@app.on_event("startup")`에도 *"요청마다 torch.load 하면 수백 ms~수 초가 반복되므로 전역 predictor를 재사용"* 이라는 이유가 붙어 있어 설계 의도가 분명합니다.

    **(2) `HousingPredictor.predict()` — 피처 순서 + 정규화 (셀 26)**
    ```python
    def predict(self, features: dict) -> dict:
        """
        피처 딕셔너리를 받아 가격을 예측합니다.

        Args:
            features: {"MedInc": 3.5, "HouseAge": 25, ...}
        Returns:
            {"predicted_price": 2.35, "predicted_price_usd": 235000}
        """
        # API/Streamlit은 dict로 값을 보내므로 키 순서가 뒤섞일 수 있습니다.
        # 학습 때 쓴 feature_names 순서(MedInc, HouseAge, ... Longitude)로 리스트를 만들어야
        # 정규화 벡터 self.mean/self.std의 i번째 값과 같은 피처가 짝을 맞습니다.
        values = [features[name] for name in self.feature_names]
    ```
    Args/Returns가 포함된 docstring + "왜 dict를 그대로 쓰면 안 되는가"가 함께 적혀 있어, **가장 조용히 틀리기 쉬운 버그(순서 어긋남)** 를 주석만 읽어도 인지할 수 있었습니다.

    **(3) 데이터 처리 주석도 인상적**
    ```python
    # axis=0  → 샘플 축을 따라 평균/표준편차를 내므로 결과는 피처별 8개 값 (shape: (8,))
    # axis를 빼면 전체 숫자를 한 값으로 뭉개 버려서, 피처마다 스케일이 다른 문제를 못 고칩니다.
    # 테스트 셋의 평균/표준편차를 따로 구하면 안 됩니다. 그건 ... 데이터 누수(data leakage)입니다.
    train_mean = X_train.mean(axis=0)
    ```
    ```python
    # MSELoss는 두 텐서의 shape가 같아야 하므로, unsqueeze(1)로 마지막에 축 1개를 붙입니다.
    #   (16512,) → (16512, 1)   ↔  "열 벡터"로 만들어 예측값과 원소별 뺄셈이 되게 함
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    ```
    빈칸을 "정답으로 채웠다"가 아니라 **왜 그 값인지, 틀리면 무슨 일이 나는지**까지 남긴 점이 이 노트북의 가장 큰 강점이었습니다.

- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 프로젝트 평가 기준에 더해 추가적으로 수행한 나만의 시도,
    실험이 기록되어 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

    **→ 충족.** 디버깅 기록과 추가 시도가 모두 있습니다.

    **(1) 실제로 겪은 에러와 해결 (셀 67 회고)**
    > 로컬에서 막혔던 지점은 **모델 파일이 없을 때 API가 안 뜨고 `/health`가 Connection refused** 가 난 것이다. 학습 셀을 돌려 `housing_model.pth`를 만든 뒤에야 통합 테스트 4종이 통과했다.

    여기서 그치지 않고 **재발 방지책**까지 제안한 점이 좋았습니다 (셀 73).
    > **시작 실패**: `housing_model.pth`가 없으면 서버가 안 뜬다. health 체크를 "모델 파일 존재 여부"까지 포함하면 원인 파악이 빨라진다.

    **(2) 겪은 에러를 코드에 반영 — 에러 안내 추가 (셀 39)**
    ```python
    try:
        resp = requests.get("http://localhost:8000/health", timeout=3)
        print(f"헬스체크: {resp.json()}")
    except requests.exceptions.ConnectionError:
        print("❌ Connection refused: http://localhost:8000 에 서버가 없습니다.")
        print("   섹션 3.3의 serve_in_thread(...) 셀을 먼저 실행하세요.")
        print("   커널은 DP05 .venv 를 사용하세요. (Select Kernel → Jupyter Kernel → DP05 .venv)")
    ```
    원본 노트북은 `requests.get` 한 줄이었는데, 본인이 당한 에러를 그대로 안내 메시지로 만들어 두었습니다.

    **(3) 실행 환경(macOS 로컬)에 맞춘 추가 개선 — 원본 코드 대비 직접 손본 부분**
    - **서버 도우미 (셀 2)**: 원본에는 없던 `lsof` 기반 `_pids_on_port()` + `SIGTERM`을 추가해, 이전 셀이 남긴 좀비 프로세스가 포트 8000을 잡고 있어도 자동 정리되도록 했습니다. `install_signal_handlers = lambda: None`, 플랫폼별 이벤트 루프 분기(`win32` → `SelectorEventLoop`)까지 들어가 있습니다.
      ```python
      def _pids_on_port(port):
          """해당 포트를 LISTEN 중인 프로세스 PID 목록을 반환합니다."""
          out = subprocess.check_output(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], ...)
      ```
    - **iframe → 브라우저 (셀 38, 49)**: Colab 전제 코드를 `IN_COLAB` 분기로 바꾸고, 로컬에서는 `webbrowser.open()`으로 열도록 수정했습니다. 그리고 **그 이유까지 출력에 남겼습니다** — *"(Jupyter iframe은 localhost X-Frame 제한으로 비어 보일 수 있습니다)"*. 원인을 정확히 짚은 디버깅입니다.
    - **의존성 설치 (셀 4, 47)**: `!pip install`을 `importlib.util.find_spec()` 체크로 바꿔 이미 설치된 경우 건너뛰게 했습니다.
    - **`asyncio.get_event_loop()` → `get_running_loop()` (셀 34)**: 원본의 deprecated API를 최신 권장 방식으로 교체했습니다. 작지만 정확한 개선입니다.

    **(4) 추가 실험**: 셀 56에서 저소득/고소득/평균 3개 케이스로 예측값 방향성을 확인했고, 셀 60에서 8개 동시 요청의 개별 소요 시간(0.033~0.049초, 전체 0.06초)을 측정해 **`run_in_executor`가 실제로 병렬 처리되고 있음**을 수치로 증명했습니다.

- [x]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

    **→ 충족.** 회고가 이 제출물에서 가장 잘 쓰인 부분입니다. **배운 점 / 아쉬운 점 / 느낀 점**이 모두 있고, **Mermaid 플로우차트**로 실행 흐름도 그렸습니다.

    **(1) 실행 플로우 그래프 (셀 73)**
    ```mermaid
    flowchart LR
      User["사용자"] -->|숫자 입력| ST["Streamlit :8501"]
      ST -->|POST /predict JSON| API["FastAPI :8000"]
      API --> Schema["Pydantic HousingRequest"]
      Schema --> Exec["run_in_executor"]
      Exec --> Pred["HousingPredictor"]
      Pred --> Norm["mean/std 정규화"]
      Norm --> MLP["HousingModel"]
      MLP --> API
      API -->|predicted_price_usd| ST
      ST -->|st.metric| User
    ```
    요청이 통과하는 계층(검증 → 스레드풀 → 전처리 → 모델)이 순서대로 드러나서, 코드를 안 봐도 구조가 잡힙니다.

    **(2) 파일별 역할을 자기 말로 정리한 표 (셀 67)** — "내가 이해한 역할" 컬럼으로 6개 파일을 정리했고, 특히 `.pth`와 `.json`을 왜 **둘 다** 배포해야 하는지를 자기 문장으로 설명했습니다.

    **(3) 핵심을 관통하는 문장 (셀 70)**
    > 어려웠던 점은 "코드를 복사하면 된다"가 아니라 **어느 층이 무슨 계약을 지키는지**를 맞춰야 했다는 것이다. mean/std 축(`axis=0`), `y`의 `unsqueeze(1)`, 피처 리스트 순서, JSON 키, 위젯 min/max가 하나라도 어긋나면 **학습은 되는데 배포만 틀린다.**

    "학습은 되는데 배포만 틀린다" — 모델 배포 과목의 핵심을 한 줄로 요약한 문장이라 리뷰하면서 가장 인상 깊었습니다.

    **(4) 배운 점 / 아쉬운 점 / 느낀 점 (셀 73)**
    - 배운 점: *"배포 단위는 가중치만이 아니라 **전처리 통계 + 피처 순서 + 스키마 계약**이다."*
    - 아쉬운 점: *"MLP라 샘플에 따라 가격 오차가 커도, UI에는 확신도 없이 점추정치만 나간다."*, *"학습 RMSE/MAE를 대시보드에 붙여 모델 품질을 같이 보여 주지는 못했다."* — 자기 결과물의 한계를 정확히 인지하고 있습니다.
    - 느낀 점: *"Day 1~4에서 조각으로 배운 직렬화, 검증, 비동기, Streamlit이 데이터 형태가 바뀌어도 같은 자리에 꽂힌다."*

    **(5) 개선 방향 표 (셀 73)**: JSON mean/std → `sklearn.Pipeline`, 단순 MLP → XGBoost/LightGBM 비교, 가격만 표시 → SHAP 설명 등 **다음 액션이 구체적**입니다.

    **(6) 최종 체크포인트 Q1~Q5 (셀 74)** 도 모두 서술형으로 답변했습니다. 특히 Q4 답변에서 *"로드밸런서·오케스트레이터가 서버를 죽은 것으로 볼 수 있습니다"* 라고 실무 영향까지 확장한 점이 좋았습니다.

- [x]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화/모듈화했는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

    **→ 충족.** 관심사 분리가 깔끔하고, PEP8(snake_case 함수/변수, PascalCase 클래스, import 순서, 4칸 들여쓰기)을 잘 지켰습니다.

    **(1) 계층별 모듈화 — 각 파일이 한 가지 책임만 담당**
    ```
    app/housing_model.py    → 모델 구조 + 전처리 + 추론   (ML 로직, HTTP를 모름)
    app/housing_schemas.py  → 요청/응답 계약             (검증만 담당)
    app/housing_api.py      → 라우팅 + 동시성            (모델 내부를 모름)
    frontend/app_housing.py → UI                        (모델을 모름, HTTP만 앎)
    ```
    회고에도 *"프론트는 모델을 몰라도 된다. HTTP와 필드 이름만 맞으면 된다"* 라고 이 원칙을 명시했습니다.

    **(2) 상태를 클래스로 캡슐화 (`HousingPredictor`)**
    ```python
    class HousingPredictor:
        """모델 로드 + 전처리 + 추론을 캡슐화한 클래스"""
        def __init__(self, model_path: str, preprocessing_path: str):
    ```
    mean/std/feature_names/model을 인스턴스가 들고 있어서, API는 `predictor.predict(features)` 한 줄만 호출하면 됩니다. 전역 변수를 흩뿌리지 않았고, 타입 힌트(`model_path: str`, `-> dict`)도 붙어 있습니다.

    **(3) 중복 제거된 API 호출 래퍼 (`call_api`)**
    ```python
    def call_api(url, json_data=None, method="post"):
        ...
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError: ...
    except requests.exceptions.HTTPError as e: ...
    ```
    `/health`와 `/predict` 두 호출이 같은 함수를 쓰고, 예외 종류별로 사용자 메시지를 분기해 UI에서 에러 처리 코드가 반복되지 않습니다.

    **(4) 상수 분리**
    ```python
    MODEL_PATH = "models/housing_model.pth"
    PREPROCESS_PATH = "models/housing_preprocessing.json"
    API_BASE = "http://localhost:8000"
    ```
    경로/주소가 하드코딩으로 흩어지지 않고 모듈 상단에 모여 있습니다.

    **(5) 도우미 함수의 재사용성 (셀 2, 47)**: `serve_in_thread`, `stop_server`, `_port_open`, `run_streamlit`, `show_dashboard` 모두 docstring을 가진 범용 함수로 분리되어, 포트/스크립트만 바꾸면 다음 Day에도 그대로 쓸 수 있습니다.


# 회고(참고 링크 및 코드 개선)
```
# 리뷰어의 회고를 작성합니다.
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```

## 리뷰어 회고

승현님 노트북은 **"빈칸을 정답으로 채운 코드"가 아니라 "왜 그 정답인지 남긴 코드"** 였습니다. 특히 `axis=0`을 빼면 왜 안 되는지, `unsqueeze(1)`이 왜 필요한지를 shape 변화까지 적어 둔 주석은 제 노트북을 다시 보게 만들었습니다. 저는 값만 채우고 넘어간 부분이 많아서, 주석으로 스스로 검증하는 방식을 배워 갑니다.

또 하나 배운 건 **실행 환경에 맞춰 원본 코드를 고친 태도**입니다. Colab 전제로 쓰인 iframe 코드를 macOS 로컬에서 그냥 "안 되네" 하고 넘기지 않고, X-Frame 제한이라는 원인을 짚어 `webbrowser.open()`으로 바꾸고 그 이유를 출력에 남긴 점, `lsof`로 포트 점유 프로세스를 정리하는 로직을 추가한 점이 인상적이었습니다.

### 참고한 링크

- [FastAPI — Concurrency and async / await](https://fastapi.tiangolo.com/async/) : `async def` 안에서 블로킹 호출을 하면 왜 안 되는지, 언제 스레드풀로 넘겨야 하는지. 승현님의 `run_in_executor` 주석이 이 문서 내용과 정확히 일치하는지 확인하며 읽었습니다.
- [Python 3 — `loop.run_in_executor`](https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor) : `get_event_loop()`가 deprecated이고 `get_running_loop()`가 권장된다는 부분. 승현님이 이미 반영해 두신 걸 확인했습니다.
- [FastAPI — Lifespan Events](https://fastapi.tiangolo.com/advanced/events/) : `@app.on_event("startup")`이 deprecated이고 `lifespan` 컨텍스트 매니저가 권장된다는 내용. 아래 개선 제안의 근거입니다.
- [scikit-learn — `Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) : 회고에서 언급하신 "JSON mean/std → Pipeline" 개선 방향의 출발점.

### 개선 제안 (사소한 것들이라 감점 요소는 아닙니다)

**1) `predict()`의 `KeyError`가 500으로 나갑니다**

지금은 Pydantic이 앞에서 막아 주기 때문에 실제로는 안 터지지만, `HousingPredictor`를 다른 곳(배치 스크립트 등)에서 재사용하면 원인 파악이 어려운 500이 납니다.

```python
# 현재
values = [features[name] for name in self.feature_names]

# 제안 — 어떤 피처가 빠졌는지 메시지에 담기
missing = [n for n in self.feature_names if n not in features]
if missing:
    raise ValueError(f"필수 피처 누락: {missing}")
values = [features[name] for name in self.feature_names]
```

**2) `on_event("startup")` → `lifespan` (FastAPI 최신 권장)**

`get_running_loop()`는 이미 최신 방식으로 고치셨으니, 이 부분도 같이 맞추면 DeprecationWarning이 사라집니다.

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("주택 가격 모델 로드 중...")
    predictor = HousingPredictor(MODEL_PATH, PREPROCESS_PATH)
    logger.info("모델 로드 완료")
    yield
    inference_executor.shutdown(wait=True)   # 종료 시 스레드풀 정리도 함께

app = FastAPI(title="California Housing Price API", version="1.0.0", lifespan=lifespan)
```

**3) 회고에서 직접 제안하신 "모델 파일까지 보는 health" — 코드로 옮기면 이렇게 됩니다**

Connection refused로 헤매셨던 그 문제를 실제로 막아 주는 형태입니다.

```python
from pathlib import Path

@app.get("/health", tags=["System"])
async def health_check():
    files_ok = Path(MODEL_PATH).exists() and Path(PREPROCESS_PATH).exists()
    return {
        "status": "healthy" if predictor is not None else "loading",
        "model": "California Housing",
        "model_files": "ok" if files_ok else "missing",   # 원인이 바로 보임
    }
```

**4) 통합 테스트의 `pop` / 복원 패턴**

```python
# 현재 — 원본 dict를 건드렸다가 되돌림 (중간에 예외가 나면 "name"이 사라진 채 남음)
name = case.pop("name")
resp = requests.post(f"{API_BASE}/predict", json=case)
case["name"] = name

# 제안 — 원본을 건드리지 않음
name = case["name"]
payload = {k: v for k, v in case.items() if k != "name"}
resp = requests.post(f"{API_BASE}/predict", json=payload)
```

**5) 모델 성능 관련 관찰**

테스트 MAE $38,666(≈0.387)은 동작에는 문제가 없지만, 학습 loss가 50 에포크 시점에서 0.3894로 **아직 내려가는 중**입니다(10→50 에포크: 0.5661 → 0.4831 → 0.4392 → 0.4106 → 0.3894). 즉 아직 수렴 전이라 **EPOCHS를 늘리거나 Dropout(0.2)을 낮추면** 개선 여지가 있어 보입니다. 회고에서 아쉬워하신 "확신도 없는 점추정치" 문제와 함께, 검증 셋 loss 곡선을 같이 찍어 보면 과소적합/과적합 어느 쪽인지 판단하기 쉬울 것 같습니다.

전반적으로 **주석, 회고, 환경 대응 모두 배울 점이 많은 제출물**이었습니다. 고생 많으셨습니다 🙂
