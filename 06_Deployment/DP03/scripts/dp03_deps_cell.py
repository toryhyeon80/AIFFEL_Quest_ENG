import os
import shutil
from pathlib import Path

# 필요한 폴더 생성
for d in ["app", "models", "data"]:
    os.makedirs(d, exist_ok=True)

MODEL_UTILS = r'''import torch
import torch.nn as nn
from torchvision import transforms

class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

CLASS_NAMES = [str(i) for i in range(10)]

def load_model(model_path, num_classes=10):
    model = SimpleClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)[0]
        idx = probs.argmax().item()
        return {
            "predicted_class": CLASS_NAMES[idx],
            "confidence": round(probs[idx].item(), 4),
            "probabilities": {CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(len(CLASS_NAMES))},
        }
'''

SCHEMAS = r'''from pydantic import BaseModel, Field, field_validator
from typing import Optional

class PixelPredictRequest(BaseModel):
    pixels: list[list[float]] = Field(..., description="28x28 픽셀 배열")
    return_probabilities: bool = Field(default=False)
    @field_validator("pixels")
    @classmethod
    def validate_pixels(cls, v):
        if len(v) != 28:
            raise ValueError(f"28행이어야 합니다. 현재: {len(v)}행")
        for i, row in enumerate(v):
            if len(row) != 28:
                raise ValueError(f"각 행은 28열이어야 합니다. {i}번째 행: {len(row)}열")
        return v

class ImagePredictRequest(BaseModel):
    image_base64: str = Field(..., min_length=1)
    return_probabilities: bool = Field(default=False)

class PredictResponse(BaseModel):
    success: bool = Field(description="성공 여부")
    predicted_class: str = Field(description="예측 숫자 (0~9)")
    confidence: float = Field(description="확신도", ge=0.0, le=1.0)
    probabilities: Optional[dict[str, float]] = Field(default=None)
'''

ERROR_HANDLERS = r'''import traceback, logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("ml_api")

def register_error_handlers(app: FastAPI):
    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        logger.error(f"에러: {type(exc).__name__}: {exc}\n경로: {request.method} {request.url}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"success": False, "error": "서버 내부 오류가 발생했습니다."})
'''

LOGGER_CONFIG = r'''import logging, sys

def setup_logger(name="ml_api", level="INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger
'''

MIDDLEWARE = r'''import time, logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("ml_api")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = round(time.time() - start, 3)
        msg = f"{request.method} {request.url.path} -> {response.status_code} ({duration}s)"
        if response.status_code >= 500: logger.error(msg)
        elif response.status_code >= 400: logger.warning(msg)
        else: logger.info(msg)
        response.headers["X-Process-Time"] = str(duration)
        return response
'''

# 1. app/model_utils.py (Day 1)
if not os.path.exists("app/model_utils.py"):
    print("⚠️ app/model_utils.py 없음 → 생성합니다.")
    with open("app/model_utils.py", "w", encoding="utf-8") as f:
        f.write(MODEL_UTILS)
    print("  ✅ app/model_utils.py 생성 완료")
else:
    print("✅ app/model_utils.py 있음")

# 2. app/schemas.py (Day 2)
if not os.path.exists("app/schemas.py"):
    print("⚠️ app/schemas.py 없음 → 생성합니다.")
    with open("app/schemas.py", "w", encoding="utf-8") as f:
        f.write(SCHEMAS)
    print("  ✅ app/schemas.py 생성 완료")
else:
    print("✅ app/schemas.py 있음")

# 3. app/error_handlers.py (Day 3 섹션 5)
if not os.path.exists("app/error_handlers.py"):
    print("⚠️ app/error_handlers.py 없음 → 생성합니다.")
    with open("app/error_handlers.py", "w", encoding="utf-8") as f:
        f.write(ERROR_HANDLERS)
    print("  ✅ app/error_handlers.py 생성 완료")
else:
    print("✅ app/error_handlers.py 있음")

# 4. app/logger_config.py (Day 3 섹션 5)
if not os.path.exists("app/logger_config.py"):
    print("⚠️ app/logger_config.py 없음 → 생성합니다.")
    with open("app/logger_config.py", "w", encoding="utf-8") as f:
        f.write(LOGGER_CONFIG)
    print("  ✅ app/logger_config.py 생성 완료")
else:
    print("✅ app/logger_config.py 있음")

# 5. app/middleware.py (Day 3 섹션 5)
if not os.path.exists("app/middleware.py"):
    print("⚠️ app/middleware.py 없음 → 생성합니다.")
    with open("app/middleware.py", "w", encoding="utf-8") as f:
        f.write(MIDDLEWARE)
    print("  ✅ app/middleware.py 생성 완료")
else:
    print("✅ app/middleware.py 있음")

# 6. 모델 파일 (Day 1 / 로컬이면 DP02 모델 재사용)
MODEL_PATH = Path("models/mnist_state_dict.pth")
DP02_MODEL = Path("../DP02/models/mnist_state_dict.pth")

if MODEL_PATH.exists():
    print("✅ models/mnist_state_dict.pth 있음")
elif DP02_MODEL.exists():
    shutil.copy2(DP02_MODEL, MODEL_PATH)
    print(f"✅ DP02 모델을 복사했습니다: {DP02_MODEL} → {MODEL_PATH}")
else:
    print("⚠️ 모델 파일 없음 → MNIST 모델을 학습하여 생성합니다. (약 1~2분 소요)")
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from app.model_utils import SimpleClassifier

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = DataLoader(
        datasets.MNIST("data", train=True, download=True, transform=transform),
        batch_size=64,
        shuffle=True,
    )

    model = SimpleClassifier(10)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(2):
        for batch_idx, (images, labels) in enumerate(train_data):
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 300 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_data)}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("  ✅ models/mnist_state_dict.pth 생성 완료")

print("\n🎉 모든 의존 파일이 준비되었습니다. 다음 셀로 진행하세요.")
