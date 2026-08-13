"""
모델 로드 및 추론 유틸리티
FastAPI 엔드포인트가 이 모듈을 import하여 사용합니다.
"""

import torch
import torch.nn as nn
from torchvision import transforms


# ===== 모델 정의 =====
class SimpleClassifier(nn.Module):
    """
    간단한 이미지 분류 모델
    - 입력: 1x28x28 (MNIST와 동일한 크기)
    - 출력: 10개 클래스에 대한 확률
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ===== 전처리 파이프라인 =====
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


# ===== 모델 로드 =====
def load_model(model_path: str = "models/mnist_state_dict.pth") -> nn.Module:
    """
    저장된 state_dict를 로드하여 추론 가능한 모델을 반환합니다.
    """
    model = SimpleClassifier(num_classes=10)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


# ===== 추론 함수 =====
def predict(model: nn.Module, input_tensor: torch.Tensor) -> dict:
    """
    모델에 입력 텐서를 전달하고 예측 결과를 반환합니다.

    Args:
        model: 로드된 PyTorch 모델
        input_tensor: 전처리된 입력 텐서 (1, 1, 28, 28)

    Returns:
        dict: {"label": int, "confidence": float, "probabilities": list}
    """
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    return {
        "label": predicted.item(),
        "confidence": round(confidence.item(), 4),
        "probabilities": probabilities[0].tolist(),
    }
