"""
train.py — YOLO11 모델 학습 스크립트
====================================

[역할]
  configs/train.yaml(또는 --config) 설정을 읽어 Ultralytics YOLO11 모델을 학습합니다.
  학습 중 매 epoch마다 val 세트로 mAP를 평가하고, best.pt 가중치를 저장합니다.

[파이프라인 위치]
  ① split_data.py  →  ② train.py  →  ③ val.py
  (이 파일이 Phase 1의 핵심 학습 단계)

[설정 파일]
  - data/data.yaml       : 데이터셋 경로·클래스(dirt, damage)
  - configs/train.yaml   : 본학습 (YOLO11s, 50 epoch, 도메인 증강)
  - configs/train_baseline.yaml : Baseline (YOLO11n, 20 epoch)

[주요 산출물]  runs/detect/<name>/
  - weights/best.pt      : val mAP 최고 epoch 가중치
  - results.csv          : epoch별 Loss·mAP 로그
  - results.png          : Loss/mAP 학습 곡선
  - confusion_matrix.png : 혼동 행렬 (학습 중 val 평가)
  - val_batch*_pred.jpg  : 검증 예측 시각화

[사용 예]
  python train.py                              # 본학습 (configs/train.yaml)
  python train.py --config configs/train_baseline.yaml --no-report  # Baseline
  python train.py --test                       # 1 Epoch 파이프라인 스모크 테스트
  python train.py --epochs 10                  # epoch 수 오버라이드
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

# 프로젝트 루트 — 상대 경로(project, data 등)를 절대 경로로 변환할 때 기준
ROOT = Path(__file__).resolve().parent

DEFAULT_TRAIN_CONFIG = ROOT / "configs" / "train.yaml"
DEFAULT_DATA_CONFIG = ROOT / "data" / "data.yaml"


# ---------------------------------------------------------------------------
# 설정 로드·검증
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict[str, Any]:
    """YAML 설정 파일을 읽어 dict로 반환."""
    if not path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_yaml(data_cfg_path: Path) -> Path:
    """
    data.yaml 유효성 검사.

    - train, val 경로 키 존재 여부
    - 실제 폴더(data/images/train, data/images/val) 존재 여부
    → split_data.py 미실행 시 명확한 오류 메시지 제공
    """
    data_cfg = load_yaml(data_cfg_path)
    dataset_root = (ROOT / data_cfg.get("path", "data")).resolve()

    for split in ("train", "val"):
        split_path = data_cfg.get(split)
        if not split_path:
            raise ValueError(f"data.yaml에 '{split}' 경로가 없습니다.")
        if not (dataset_root / split_path).exists():
            raise FileNotFoundError(
                f"데이터 분할 폴더가 없습니다: {dataset_root / split_path}\n"
                "먼저 `python split_data.py`를 실행해 주세요."
            )
    return data_cfg_path.resolve()


def build_train_kwargs(train_cfg: dict[str, Any], data_yaml: Path) -> dict[str, Any]:
    """
    train.yaml → Ultralytics model.train() 인자 dict 변환.

    hyperparameters: epochs, batch, patience, project, name 등
    augmentation:    flipud, mosaic, hsv 등 도메인 증강
    → 두 섹션을 병합하여 train()에 전달 (하드코딩 없음)
    """
    hyper = dict(train_cfg.get("hyperparameters", {}))
    aug = train_cfg.get("augmentation", {})

    # project를 절대 경로로 고정 — cwd에 따라 runs/가 다른 곳에 생기는 문제 방지
    if "project" in hyper:
        hyper["project"] = str((ROOT / hyper["project"]).resolve())

    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "device": train_cfg.get("device", "mps"),  # Apple M1 GPU 가속
        **hyper,
        **aug,
    }
    return kwargs


# ---------------------------------------------------------------------------
# OOM(메모리 부족) 대응 학습
# ---------------------------------------------------------------------------

def is_oom_error(exc: BaseException) -> bool:
    """MPS/CUDA 메모리 부족 예외인지 문자열로 판별."""
    message = str(exc).lower()
    return any(keyword in message for keyword in ("out of memory", "oom", "mps backend"))


def train_with_oom_fallback(
    model: YOLO,
    train_kwargs: dict[str, Any],
    batch_candidates: list[int],
) -> Any:
    """
    배치 크기를 순차적으로 줄이며 학습을 시도합니다.

    M1 16GB에서 batch 32 시 스왑 발생 → 8→4→2 순으로 재시도.
    oom_fallback_batches는 configs/train.yaml에서 설정.
    """
    last_error: BaseException | None = None

    for batch in batch_candidates:
        kwargs = {**train_kwargs, "batch": batch}
        print(f"\n[학습 시작] batch={batch}, device={kwargs.get('device')}")
        try:
            return model.train(**kwargs)
        except (RuntimeError, Exception) as exc:
            if is_oom_error(exc):
                print(f"[OOM] batch={batch} 실패 → 배치 크기를 줄여 재시도합니다.")
                last_error = exc
                continue
            raise  # OOM이 아닌 오류는 즉시 전파

    raise RuntimeError(
        "모든 배치 크기에서 학습에 실패했습니다. "
        "configs/train.yaml의 imgsz·batch를 낮추거나 data.yaml 경로를 확인하세요."
    ) from last_error


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """명령줄 인자 정의."""
    parser = argparse.ArgumentParser(description="YOLO11 풍력 터빈 파손 탐지 학습")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG,
        help="학습 설정 YAML (기본: configs/train.yaml)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="데이터셋 YAML (기본: data/data.yaml)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="1 Epoch만 실행하여 파이프라인 검증 (runs/detect/test_run)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="epoch 수 오버라이드 (미지정 시 configs/train.yaml 값 사용)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="학습 후 report.md 자동 갱신 건너뛰기 (baseline 학습 시 사용)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    """
    학습 실행 흐름:
      1. YAML 로드 및 데이터 경로 검증
      2. train kwargs 조립
      3. 테스트/epoch 오버라이드 적용
      4. YOLO 모델 로드 → OOM fallback 학습
      5. (선택) update_report.py로 report.md 자동 갱신
    """
    args = parse_args()
    train_cfg = load_yaml(args.config.resolve())
    data_yaml = resolve_data_yaml(args.data.resolve())
    train_kwargs = build_train_kwargs(train_cfg, data_yaml)

    # --test: 환경·데이터·MPS 동작 확인용 1 epoch 스모크 테스트
    if args.test:
        train_kwargs["epochs"] = 1
        train_kwargs["name"] = "test_run"
        print("[테스트 모드] 1 Epoch 파이프라인 검증을 실행합니다.")
    elif args.epochs is not None:
        train_kwargs["epochs"] = args.epochs

    model_name = train_cfg.get("model", "yolo11s.pt")
    print(f"모델: {model_name}")
    print(f"데이터: {data_yaml}")
    print(f"증강 flipud={train_kwargs.get('flipud')} (상하 반전 금지)")

    model = YOLO(model_name)

    # 설정 batch가 fallback 목록에 없으면 맨 앞에 추가
    batch_candidates = train_cfg.get("oom_fallback_batches", [train_kwargs.get("batch", 16)])
    if train_kwargs.get("batch") not in batch_candidates:
        batch_candidates = [train_kwargs["batch"], *batch_candidates]

    train_with_oom_fallback(model, train_kwargs, batch_candidates)
    print("\n학습이 완료되었습니다. 결과: runs/detect/ 폴더를 확인하세요.")

    # Baseline(--no-report)이나 테스트(--test) 모드에서는 리포트 갱신 생략
    if not args.test and not args.no_report:
        print("report.md 자동 갱신을 시도합니다...")
        subprocess.run([sys.executable, str(ROOT / "update_report.py")], check=False)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        sys.exit(1)
