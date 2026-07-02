"""
predict.py — YOLO11 단일·배치 추론 스크립트 (Phase 1 Test)
=========================================================

[역할]
  학습된 best.pt로 이미지(또는 폴더·영상)에 대해 Dirt/Damage 객체 탐지를 수행하고,
  BBox가 그려진 결과 이미지와 JSON을 저장합니다.

[파이프라인 위치]
  ① split_data.py  →  ② train.py  →  ③ val.py  →  ④ predict.py
  (Phase 2 FastAPI가 이 로직을 재사용합니다)

[설정 파일]
  - data/data.yaml         : 클래스 이름 (dirt, damage)
  - configs/predict.yaml   : 가중치, conf, 저장 경로 등

[주요 산출물]  runs/predict/<name>/
  - *.jpg                  : BBox 오버레이 결과 이미지
  - predictions.json       : 탐지 결과 (클래스·confidence·bbox)

[사용 예]
  python predict.py --source data/images/val/DJI_0004_02_07.png
  python predict.py --source data/images/val --name val_batch
  python predict.py --source path/to/video.mp4 --conf 0.3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICT_CONFIG = ROOT / "configs" / "predict.yaml"
DEFAULT_DATA_CONFIG = ROOT / "data" / "data.yaml"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_class_names(data_cfg_path: Path) -> dict[int, str]:
    data_cfg = load_yaml(data_cfg_path)
    names = data_cfg.get("names", {})
    return {int(k): str(v) for k, v in names.items()}


def resolve_weights(weights: Path) -> Path:
    path = weights if weights.is_absolute() else (ROOT / weights)
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"가중치 파일을 찾을 수 없습니다: {path}\n"
            "먼저 `python train.py`로 학습을 완료하거나 --weights 경로를 확인하세요."
        )
    return path


def resolve_source(source: Path) -> Path:
    path = source if source.is_absolute() else (ROOT / source)
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"입력 소스를 찾을 수 없습니다: {path}")
    return path


def build_predict_kwargs(predict_cfg: dict[str, Any]) -> dict[str, Any]:
    params = dict(predict_cfg.get("parameters", {}))
    if "project" in params:
        params["project"] = str((ROOT / params["project"]).resolve())
    kwargs: dict[str, Any] = dict(params)
    if "device" in predict_cfg:
        kwargs["device"] = predict_cfg["device"]
    return kwargs


def _default_run_name(source: Path) -> str:
    if source.is_file():
        return source.stem
    return f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def result_to_dict(
    result: Any,
    class_names: dict[int, str],
    source_label: str,
) -> dict[str, Any]:
    """Ultralytics Result 1건 → JSON 직렬화 dict."""
    detections: list[dict[str, Any]] = []
    boxes = result.boxes

    if boxes is not None and len(boxes):
        xyxy_list = boxes.xyxy.cpu().tolist()
        conf_list = boxes.conf.cpu().tolist()
        cls_list = boxes.cls.cpu().tolist()

        for xyxy, conf, cls_id in zip(xyxy_list, conf_list, cls_list):
            class_id = int(cls_id)
            x1, y1, x2, y2 = (round(v, 2) for v in xyxy)
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_names.get(class_id, str(class_id)),
                    "confidence": round(float(conf), 4),
                    "bbox_xyxy": [x1, y1, x2, y2],
                }
            )

    return {
        "source": source_label,
        "image_width": int(result.orig_shape[1]),
        "image_height": int(result.orig_shape[0]),
        "detection_count": len(detections),
        "detections": detections,
    }


def save_predictions_json(
    output_path: Path,
    payload: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO11 best.pt로 Dirt/Damage 객체 탐지 추론을 실행합니다."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="입력 경로 — 이미지 1장, 폴더, 영상 (.mp4 등)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PREDICT_CONFIG,
        help="추론 설정 YAML (기본: configs/predict.yaml)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="클래스 이름 참조용 data.yaml (기본: data/data.yaml)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="가중치 경로 (미지정 시 configs/predict.yaml 값 사용)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="confidence threshold 오버라이드",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="결과 저장 폴더 이름 (기본: 입력 파일 stem)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="추론 후 report.md 자동 갱신 건너뛰기",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predict_cfg = load_yaml(args.config.resolve())
    data_yaml = args.data.resolve()
    class_names = load_class_names(data_yaml)

    weights = resolve_weights(
        args.weights or Path(predict_cfg.get("weights", "runs/detect/train/weights/best.pt"))
    )
    source = resolve_source(args.source)

    predict_kwargs = build_predict_kwargs(predict_cfg)
    run_name = args.name or _default_run_name(source)
    predict_kwargs["name"] = run_name

    if args.conf is not None:
        predict_kwargs["conf"] = args.conf

    output_cfg = predict_cfg.get("output", {})
    save_json = bool(output_cfg.get("save_json", True))
    json_filename = str(output_cfg.get("json_filename", "predictions.json"))

    print(f"가중치: {weights}")
    print(f"입력: {source}")
    print(f"device: {predict_kwargs.get('device', 'mps')}")
    print(f"결과 저장(예정): {predict_kwargs['project']}/{run_name}/")

    model = YOLO(str(weights))
    results = model.predict(source=str(source), **predict_kwargs)

    per_image = [
        result_to_dict(r, class_names, Path(r.path).name if r.path else str(source))
        for r in results
    ]
    total_detections = sum(item["detection_count"] for item in per_image)

    save_dir = Path(results[0].save_dir) if results else Path(predict_kwargs["project"]) / run_name
    print(f"\n=== 추론 완료 ===")
    print(f"처리 파일 수: {len(per_image)}")
    print(f"총 탐지 BBox: {total_detections}")
    print(f"저장 위치: {save_dir}/")

    for item in per_image[:5]:
        if item["detection_count"]:
            preview = ", ".join(d["class_name"] for d in item["detections"][:3])
            suffix = "..." if item["detection_count"] > 3 else ""
            detail = f"({preview}{suffix})"
        else:
            detail = "(탐지 없음)"
        print(f"  - {item['source']}: {item['detection_count']}개 {detail}")
    if len(per_image) > 5:
        print(f"  ... 외 {len(per_image) - 5}개")

    if save_json and per_image:
        payload = {
            "weights": str(weights.relative_to(ROOT)) if weights.is_relative_to(ROOT) else str(weights),
            "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
            "device": predict_kwargs.get("device"),
            "conf": predict_kwargs.get("conf"),
            "iou": predict_kwargs.get("iou"),
            "predicted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": per_image,
        }
        json_path = save_dir / json_filename
        save_predictions_json(json_path, payload)
        print(f"JSON 저장: {json_path}")

    if not args.no_report and save_dir.exists():
        print("\nreport.md predict 추론 섹션 갱신 중 (update_report.py)...")
        subprocess.run(
            [sys.executable, str(ROOT / "update_report.py")],
            cwd=ROOT,
            check=False,
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        sys.exit(1)
