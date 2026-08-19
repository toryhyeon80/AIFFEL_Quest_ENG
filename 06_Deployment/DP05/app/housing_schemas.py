"""
Day 5 - 주택 가격 예측 API 스키마
"""
from pydantic import BaseModel, Field


class HousingRequest(BaseModel):
    """주택 가격 예측 요청"""
    # Field(...): 필수 필드. 값이 없으면 422 Unprocessable Entity.
    # gt=0 (greater than): 중위 소득은 0보다 커야 함. 0·음수는 스키마 단계에서 거절.
    MedInc: float = Field(..., gt=0, description="중위 소득")                        # *your code* — gt=0 설정
    # ge=0, le=100: 0년 이상 100년 이하. 데이터셋 HouseAge 상한이 약 52여도
    # 실무 입력은 여유 있게 100까지 허용하는 경우가 많습니다.
    HouseAge: float = Field(..., ge=0, le=100, description="주택 연식 (년)")          # *your code* — ge, le 범위
    AveRooms: float = Field(..., gt=0, description="평균 방 수")
    AveBedrms: float = Field(..., gt=0, description="평균 침실 수")
    Population: float = Field(..., gt=0, description="인구")
    AveOccup: float = Field(..., gt=0, description="평균 거주 인원")
    # 이 모델은 캘리포니아 주택 데이터로만 학습됨. 위도를 32~42로 막아
    # 학습 분포 밖의 좌표가 들어와 지리 피처가 의미 없게 쓰이지 않게 합니다.
    Latitude: float = Field(..., ge=32, le=42, description="위도 (캘리포니아 범위)")   # *your code* — 캘리포니아 위도 범위
    Longitude: float = Field(..., ge=-125, le=-114, description="경도 (캘리포니아 범위)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "MedInc": 3.5,
                    "HouseAge": 25.0,
                    "AveRooms": 5.0,
                    "AveBedrms": 1.0,
                    "Population": 1500.0,
                    "AveOccup": 3.0,
                    "Latitude": 37.5,
                    "Longitude": -122.0,
                }
            ]
        }
    }


class HousingResponse(BaseModel):
    """주택 가격 예측 응답"""
    success: bool = Field(description="요청 처리 성공 여부")
    predicted_price: float = Field(description="예측 가격 ($100,000 단위)")
    predicted_price_usd: int = Field(description="예측 가격 (USD)")
    input_features: dict = Field(description="입력된 피처 값")
