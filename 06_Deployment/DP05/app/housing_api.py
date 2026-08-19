"""
Day 5 - 주택 가격 예측 FastAPI 서버
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException

from app.housing_schemas import HousingRequest, HousingResponse
from app.housing_model import HousingPredictor
from app.logger_config import setup_logger
from app.error_handlers import register_error_handlers
from app.middleware import RequestLoggingMiddleware


# ===== 설정 =====
logger = setup_logger("housing_api")

app = FastAPI(
    title="California Housing Price API",
    description="캘리포니아 주택 가격을 예측하는 API",
    version="1.0.0",
)

app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)

# 추론 전용 스레드풀 (Day 3에서 배운 패턴)
# PyTorch 추론은 CPU를 오래 쓰는 동기 작업이라, async def 안에서 그대로 호출하면
# 이벤트 루프가 멈춰 다른 /health 요청까지 지연됩니다.
# ThreadPoolExecutor에 넘기면 루프는 비워 두고, 스레드에서 predict()만 실행합니다.
# max_workers=4: 동시에 최대 4개 추론. thread_name_prefix는 로그에서 스레드를 구분할 때 사용.
inference_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="housing")  # *your code* — 스레드풀 생성

# ===== 모델 로드 =====
MODEL_PATH = "models/housing_model.pth"
PREPROCESS_PATH = "models/housing_preprocessing.json"
predictor = None


@app.on_event("startup")
async def startup():
    global predictor
    logger.info("주택 가격 모델 로드 중...")
    # 서버가 뜨는 시점에 모델·전처리 JSON을 한 번만 로드합니다.
    # 요청마다 torch.load 하면 수백 ms~수 초가 반복되므로, 전역 predictor를 재사용합니다.
    predictor = HousingPredictor(MODEL_PATH, PREPROCESS_PATH)  # *your code* — HousingPredictor 인스턴스 생성
    logger.info("모델 로드 완료")


# ===== 엔드포인트 =====

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy" if predictor is not None else "loading",
        "model": "California Housing",
    }


@app.post("/predict", response_model=HousingResponse, tags=["Prediction"])
async def predict_housing(request: HousingRequest):
    """주택 정보를 받아 가격을 예측합니다."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    # Pydantic 모델을 dict로 변환 (키 이름은 스키마 필드와 동일: MedInc, HouseAge, ...)
    # HousingPredictor.predict()는 dict를 받으므로 model_dump()가 다리 역할을 합니다.
    features = request.model_dump()  # *your code* — Pydantic 모델 → dict

    try:
        # 추론 (별도 스레드에서 실행 — Day 3 패턴)
        loop = asyncio.get_running_loop()
        # run_in_executor(executor, func, *args): func(*args)를 스레드풀에서 실행하고,
        # 끝날 때까지 await로 기다립니다. 그동안 이벤트 루프는 다른 요청을 받을 수 있습니다.
        result = await loop.run_in_executor(       # *your code* — run_in_executor 사용
            inference_executor,              # 추론 전용 스레드풀
            predictor.predict,              # 실행할 동기 함수
            features,              # 함수에 전달할 인자
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {str(e)}")

    return HousingResponse(
        success=True,
        predicted_price=result["predicted_price"],
        predicted_price_usd=result["predicted_price_usd"],
        input_features=features,
    )
