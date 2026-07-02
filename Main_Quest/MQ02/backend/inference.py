"""
backend/inference.py — YOLO 추론 엔진 (predict.py 로직 재사용)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from predict import load_class_names, load_yaml, resolve_weights, result_to_dict

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_API_CONFIG = ROOT / "configs" / "api.yaml"
DEFAULT_DATA_CONFIG = ROOT / "data" / "data.yaml"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def load_api_config(path: Path = DEFAULT_API_CONFIG) -> dict[str, Any]:
    return load_yaml(path.resolve())


def summarize_detections(detections: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {"dirt": 0, "damage": 0}
    for det in detections:
        name = str(det.get("class_name", "")).lower()
        if name in summary:
            summary[name] += 1
    return summary


class InferenceEngine:
    """best.pt를 1회 로드해 API 요청마다 재사용합니다."""

    def __init__(self) -> None:
        self.model: YOLO | None = None
        self.class_names: dict[int, str] = {}
        self.weights_path: Path | None = None
        self.device: str = "mps"
        self.conf: float = 0.25
        self.iou: float = 0.7
        self.imgsz: int = 640

    def load(self, api_cfg_path: Path = DEFAULT_API_CONFIG) -> None:
        api_cfg = load_api_config(api_cfg_path)
        predict_cfg = api_cfg.get("predict", {})

        self.weights_path = resolve_weights(Path(api_cfg.get("weights", "runs/detect/train/weights/best.pt")))
        self.device = str(api_cfg.get("device", "mps"))
        self.conf = float(predict_cfg.get("conf", 0.25))
        self.iou = float(predict_cfg.get("iou", 0.7))
        self.imgsz = int(predict_cfg.get("imgsz", 640))
        self.class_names = load_class_names(DEFAULT_DATA_CONFIG)
        self.model = YOLO(str(self.weights_path))

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_path(self, image_path: Path, source_label: str | None = None) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("InferenceEngine가 로드되지 않았습니다.")

        label = source_label or image_path.name
        results = self.model.predict(
            source=str(image_path),
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            save=False,
            verbose=False,
        )
        if not results:
            raise RuntimeError("추론 결과가 비어 있습니다.")

        payload = result_to_dict(results[0], self.class_names, label)
        payload["summary"] = summarize_detections(payload["detections"])
        return payload

    def model_info(self) -> dict[str, Any]:
        weights_ref = (
            str(self.weights_path.relative_to(ROOT))
            if self.weights_path and self.weights_path.is_relative_to(ROOT)
            else str(self.weights_path)
        )
        return {
            "weights": weights_ref,
            "device": self.device,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "classes": self.class_names,
        }


engine = InferenceEngine()
