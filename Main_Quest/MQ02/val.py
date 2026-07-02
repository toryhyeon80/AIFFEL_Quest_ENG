"""
val.py — YOLO11 모델 검증(Validation) 스크립트
==============================================

[역할]
  학습이 끝난 best.pt 가중치로 val 세트를 **재평가**하고,
  혼동 행렬·예측 이미지·메트릭 파일을 runs/detect/에 저장합니다.

[train.py와의 차이]
  - train.py: 학습 + 매 epoch val 평가 (가중치 업데이트)
  - val.py:   학습 완료 후 best.pt만 로드해 val 세트 **공식 재검증**
              → 리포트·발표용 최종 성적표·시각화 산출

[파이프라인 위치]
  ① split_data.py  →  ② train.py  →  ③ val.py
  (이 파일이 Phase 1의 최종 평가 단계)

[설정 파일]
  - data/data.yaml    : 데이터셋 경로·클래스
  - configs/val.yaml  : 가중치 경로, batch, 결과 폴더명(val_final) 등

[주요 산출물]  runs/detect/val_final/
  - val_metrics.yaml       : mAP·P/R 수치 (검증 메타정보)
  - results.csv            : train/results.csv와 동일 컬럼 (리포트 도구 호환)
  - confusion_matrix.png   : 혼동 행렬
  - BoxF1_curve.png        : F1 곡선
  - val_batch*_pred.jpg    : 검증 예측 BBox 시각화

[사용 예]
  python val.py
  python val.py --weights runs/detect/train/weights/best.pt
  python val.py --name val_final --no-report
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_VAL_CONFIG = ROOT / "configs" / "val.yaml"
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
    data.yaml 유효성 검사 (검증용 — val 경로만 확인).

    train.py와 달리 train 폴더는 검사하지 않음.
    """
    data_cfg = load_yaml(data_cfg_path)
    dataset_root = (ROOT / data_cfg.get("path", "data")).resolve()

    val_path = data_cfg.get("val")
    if not val_path:
        raise ValueError("data.yaml에 'val' 경로가 없습니다.")
    if not (dataset_root / val_path).exists():
        raise FileNotFoundError(
            f"검증 데이터 폴더가 없습니다: {dataset_root / val_path}\n"
            "먼저 `python split_data.py`를 실행해 주세요."
        )
    return data_cfg_path.resolve()


def resolve_weights(weights: Path) -> Path:
    """
    best.pt 등 가중치 파일 경로 확인.

    상대 경로는 프로젝트 루트 기준으로 해석.
    """
    path = weights if weights.is_absolute() else (ROOT / weights).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"가중치 파일을 찾을 수 없습니다: {path}\n"
            "먼저 `python train.py`로 학습을 완료해 주세요."
        )
    return path


def build_val_kwargs(val_cfg: dict[str, Any], data_yaml: Path) -> dict[str, Any]:
    """
    val.yaml → Ultralytics model.val() 인자 dict 변환.

    parameters: batch, imgsz, name(val_final), project, exist_ok 등
    """
    params = dict(val_cfg.get("parameters", {}))

    if "project" in params:
        params["project"] = str((ROOT / params["project"]).resolve())

    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "device": val_cfg.get("device", "mps"),
        **params,
    }
    return kwargs


# ---------------------------------------------------------------------------
# OOM(메모리 부족) 대응 검증
# ---------------------------------------------------------------------------

def is_oom_error(exc: BaseException) -> bool:
    """MPS/CUDA 메모리 부족 예외인지 문자열로 판별."""
    message = str(exc).lower()
    return any(keyword in message for keyword in ("out of memory", "oom", "mps backend"))


def val_with_oom_fallback(
    model: YOLO,
    val_kwargs: dict[str, Any],
    batch_candidates: list[int],
) -> Any:
    """
    배치 크기를 순차적으로 줄이며 val()을 시도합니다.

    train.py의 train_with_oom_fallback과 동일한 패턴.
    """
    last_error: BaseException | None = None

    for batch in batch_candidates:
        kwargs = {**val_kwargs, "batch": batch}
        print(f"\n[검증 시작] batch={batch}, device={kwargs.get('device')}")
        try:
            return model.val(**kwargs)
        except (RuntimeError, Exception) as exc:
            if is_oom_error(exc):
                print(f"[OOM] batch={batch} 실패 → 배치 크기를 줄여 재시도합니다.")
                last_error = exc
                continue
            raise

    raise RuntimeError(
        "모든 배치 크기에서 검증에 실패했습니다. "
        "configs/val.yaml의 batch·imgsz를 낮추거나 data.yaml 경로를 확인하세요."
    ) from last_error


# ---------------------------------------------------------------------------
# 검증 결과 저장 (Ultralytics val()은 results.csv를 자동 생성하지 않음)
# ---------------------------------------------------------------------------

def _train_best_epoch() -> int:
    """
    본학습 epoch 수를 읽어 results.csv의 epoch 컬럼에 기록.

    val() 자체는 epoch 개념이 없으므로, train/args.yaml의 epochs 값을 참조.
    """
    train_args = ROOT / "runs" / "detect" / "train" / "args.yaml"
    if not train_args.exists():
        return 1
    with train_args.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return int(data.get("epochs", 1))


def save_val_artifacts(output_dir: Path, weights_path: Path, metrics: Any) -> None:
    """
    검증 메트릭을 파일로 저장 — update_report.py / update_notion.py 호환용.

    [저장 파일]
      1. val_metrics.yaml — 검증 전용 메타 (weights 경로, source=val, mAP·P/R)
      2. results.csv      — train/results.csv와 동일 핵심 컬럼 1행
                            (리포트 스크립트가 train/val 구분 없이 읽을 수 있게)

    [이유]
      Ultralytics mode=val은 그래프·이미지는 저장하지만,
      train 형식의 results.csv는 만들지 않아서 수동 생성이 필요함.
    """
    try:
        weights_ref = str(weights_path.relative_to(ROOT))
    except ValueError:
        weights_ref = str(weights_path)

    # metrics.box: DetectionMetrics — 박스 단위 집계 지표
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    precision = float(getattr(metrics.box, "mp", 0.0))   # mean precision
    recall = float(getattr(metrics.box, "mr", 0.0))      # mean recall
    epoch = _train_best_epoch()

    class_names = list(getattr(metrics, "names", {}).values())
    per_class: dict[str, dict[str, float]] = {}
    if class_names:
        p_list = list(getattr(metrics.box, "p", []) or [])
        r_list = list(getattr(metrics.box, "r", []) or [])
        ap50_list = list(getattr(metrics.box, "ap50", []) or [])
        for idx, name in enumerate(class_names):
            per_class[name] = {
                "precision": float(p_list[idx]) if idx < len(p_list) else 0.0,
                "recall": float(r_list[idx]) if idx < len(r_list) else 0.0,
                "map50": float(ap50_list[idx]) if idx < len(ap50_list) else 0.0,
            }

    confusion_payload: dict[str, object] | None = None
    cm_obj = getattr(metrics, "confusion_matrix", None)
    if cm_obj is not None and getattr(cm_obj, "matrix", None) is not None:
        labels = class_names + ["background"]
        matrix = [[int(v) for v in row] for row in cm_obj.matrix]
        confusion_payload = {"labels": labels, "predicted_rows": matrix}

    # --- val_metrics.yaml: 사람·스크립트가 읽기 쉬운 검증 요약 ---
    val_metrics_path = output_dir / "val_metrics.yaml"
    payload: dict[str, object] = {
        "source": "val",
        "weights": weights_ref,
        "epoch": epoch,
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
    }
    if per_class:
        payload["classes"] = per_class
    if confusion_payload:
        payload["confusion_matrix"] = confusion_payload

    with val_metrics_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

    # --- results.csv: update_report·update_notion이 기대하는 컬럼 형식 ---
    results_row = {
        "epoch": epoch,
        "metrics/precision(B)": f"{precision:.5f}",
        "metrics/recall(B)": f"{recall:.5f}",
        "metrics/mAP50(B)": f"{map50:.5f}",
        "metrics/mAP50-95(B)": f"{map50_95:.5f}",
    }
    results_csv = output_dir / "results.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_row.keys()))
        writer.writeheader()
        writer.writerow(results_row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """명령줄 인자 정의."""
    parser = argparse.ArgumentParser(description="YOLO11 풍력 터빈 파손 탐지 검증")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_VAL_CONFIG,
        help="검증 설정 YAML (기본: configs/val.yaml)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="데이터셋 YAML (기본: data/data.yaml)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="가중치 경로 오버라이드 (미지정 시 configs/val.yaml 값 사용)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="결과 폴더 이름 오버라이드 (기본: val_final)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="검증 후 report.md 자동 갱신 건너뛰기",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    """
    검증 실행 흐름:
      1. YAML·가중치·데이터 경로 검증
      2. best.pt 로드 → val() 실행 (OOM fallback)
      3. val_metrics.yaml + results.csv 저장
      4. (선택) update_report.py로 report.md 갱신
    """
    args = parse_args()
    val_cfg = load_yaml(args.config.resolve())
    data_yaml = resolve_data_yaml(args.data.resolve())
    val_kwargs = build_val_kwargs(val_cfg, data_yaml)

    weights_path = resolve_weights(args.weights or Path(val_cfg.get("weights", "")))
    if args.name:
        val_kwargs["name"] = args.name

    run_name = val_kwargs.get("name", "val_final")
    expected_dir = Path(val_kwargs["project"]) / run_name

    print(f"가중치: {weights_path}")
    print(f"데이터: {data_yaml}")
    print(f"결과 저장(예정): {expected_dir}")

    model = YOLO(str(weights_path))

    batch_candidates = val_cfg.get("oom_fallback_batches", [val_kwargs.get("batch", 8)])
    if val_kwargs.get("batch") not in batch_candidates:
        batch_candidates = [val_kwargs["batch"], *batch_candidates]

    # model.val() → 혼동행렬, 예측 이미지, plots 등을 save_dir에 저장
    results = val_with_oom_fallback(model, val_kwargs, batch_candidates)
    output_dir = Path(results.save_dir)

    # Ultralytics가 만들지 않는 CSV/YAML을 수동 저장
    save_val_artifacts(output_dir, weights_path, results)
    print(f"\n검증이 완료되었습니다. 결과: {output_dir}/")
    print(f"mAP50={results.box.map50:.3f}, mAP50-95={results.box.map:.3f}")

    if not args.no_report and output_dir.exists():
        print("report.md 자동 갱신을 시도합니다...")
        subprocess.run(
            [sys.executable, str(ROOT / "update_report.py")],
            check=False,
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        sys.exit(1)
