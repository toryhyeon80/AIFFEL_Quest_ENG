"""
모델 로드 및 추론 유틸리티
FastAPI 엔드포인트가 이 모듈을 import하여 사용합니다.
"""


import torch
import torch.nn as nn
from torchvision import transforms


# ===== 모델 정의 =====
class SimpleClassifier(nn.Module):
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
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


# ===== 모델 로드 =====
def load_model(model_path: str, num_classes: int = 10) -> nn.Module:
    """저장된 state_dict를 불러와서 추론 가능한 모델을 반환합니다."""
    model = SimpleClassifier(num_classes=num_classes)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


# ===== 추론 =====
# 클래스 이름 매핑
CLASS_NAMES = [str(i) for i in range(10)]   # MNIST: "0" ~ "9"


def predict(model: nn.Module, image_tensor: torch.Tensor) -> dict:
    """
    전처리된 이미지 텐서를 받아 추론 결과를 반환합니다.

    Args:
        model: 로드된 PyTorch 모델
        image_tensor: (1, 1, 28, 28) 형태의 텐서

    Returns:
        {
            "predicted_class": "7",
            "confidence": 0.98,
            "probabilities": {"0": 0.001, "1": 0.002, ...}
        }
    """
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

        predicted_idx = probabilities.argmax().item()
        confidence = probabilities[predicted_idx].item()

        prob_dict = {
            CLASS_NAMES[i]: round(probabilities[i].item(), 4)
            for i in range(len(CLASS_NAMES))
        }

    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": round(confidence, 4),
        "probabilities": prob_dict,
    }
