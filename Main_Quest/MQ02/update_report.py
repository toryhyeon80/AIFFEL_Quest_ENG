"""
runs/detect/ 학습·검증 결과와 data/ 분할 통계를 스캔하여 report.md를 자동 갱신합니다.

갱신 항목:
  - 섹션 1: Train/Val 분할 표 (runs/split_summary.yaml 또는 data/ 폴더 집계)
  - 섹션 1: EDA 클래스 분포·인사이트·그래프 (runs/eda)
  - 섹션 2: Baseline / 최종 모델 성능 비교 표
  - 섹션 3: EXP별 성능 비교 표
  - 섹션 4: Loss/mAP 그래프(train) + 검증 시각화(val)
  - 섹션 4: predict.py Val 일괄 추론 (runs/predict/*/predictions.json)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from split_stats import DEFAULT_SPLIT_SUMMARY, load_split_summary

ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = ROOT / "report.md"
DEFAULT_RUNS_DIR = ROOT / "runs" / "detect"
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_TRAIN_CONFIG = ROOT / "configs" / "train.yaml"

SPLIT_START = "<!-- report:auto:split -->"
SPLIT_END = "<!-- /report:auto:split -->"

METRICS_START = "<!-- report:auto:metrics-visuals -->"
METRICS_END = "<!-- /report:auto:metrics-visuals -->"
PREDICTIONS_START = "<!-- report:auto:predictions -->"
PREDICTIONS_END = "<!-- /report:auto:predictions -->"
PREDICT_INFERENCE_START = "<!-- report:auto:predict-inference -->"
PREDICT_INFERENCE_END = "<!-- /report:auto:predict-inference -->"
RUN_SUMMARY_START = "<!-- report:auto:run-summary -->"
RUN_SUMMARY_END = "<!-- /report:auto:run-summary -->"
EXP_START = "<!-- report:auto:exp-comparison -->"
EXP_END = "<!-- /report:auto:exp-comparison -->"
HYPER_TUNING_START = "<!-- report:auto:hyper-tuning -->"
HYPER_TUNING_END = "<!-- /report:auto:hyper-tuning -->"
ERROR_ANALYSIS_START = "<!-- report:auto:error-analysis -->"
ERROR_ANALYSIS_END = "<!-- /report:auto:error-analysis -->"
EDA_START = "<!-- report:auto:eda -->"
EDA_END = "<!-- /report:auto:eda -->"

DEFAULT_EDA_DIR = ROOT / "runs" / "eda"
DEFAULT_PREDICT_DIR = ROOT / "runs" / "predict"
REPORT_ASSETS_DIR = ROOT / "report" / "assets"

FINAL_MODEL_ROW_PATTERN = re.compile(
    r"^\|\s*\*\*최종 모델\*\*\s*\|.*\|$",
    re.MULTILINE,
)
BASELINE_ROW_PATTERN = re.compile(
    r"^\|\s*\*\*Baseline\*\*\s*\|.*\|$",
    re.MULTILINE,
)
IMPROVEMENT_ROW_PATTERN = re.compile(
    r"^\|\s*\*\*성능 향상\*\*\s*\|.*\|$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    run_name: str
    model: str
    epochs_configured: int
    best_epoch: int
    map50: float
    map50_95: float
    updated_at: str
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 학습·분할 결과를 report.md에 반영합니다.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="(사용 안 함, 하위 호환용) 최종 모델은 항상 runs/detect/train 기준",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _pick_metric(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        if key in row and row[key] not in ("", None):
            return float(row[key])
    return None


def load_args_yaml(run_dir: Path) -> dict:
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        return {}
    with args_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_metrics_from_csv(results_csv: Path) -> tuple[int, float, float]:
    with results_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"results.csv가 비어 있습니다: {results_csv}")

    best = max(
        rows,
        key=lambda r: _pick_metric(r, "metrics/mAP50(B)", "mAP50", "map50") or 0.0,
    )
    map50 = _pick_metric(best, "metrics/mAP50(B)", "mAP50", "map50")
    map50_95 = _pick_metric(best, "metrics/mAP50-95(B)", "mAP50-95", "map50-95")
    if map50 is None or map50_95 is None:
        raise ValueError(f"mAP 컬럼을 찾을 수 없습니다: {results_csv}")

    epoch = int(float(best.get("epoch", len(rows))))
    return epoch, map50, map50_95


def format_model_name(model_path: str) -> str:
    name = Path(model_path).stem.lower()
    mapping = {
        "yolo11n": "YOLO11n (Nano)",
        "yolo11s": "YOLO11s (Small)",
        "yolov8n": "YOLOv8n (Nano)",
        "yolov8s": "YOLOv8s (Small)",
        "best": "YOLO11s (Small)",
    }
    return mapping.get(name, model_path)


def try_collect_run_result(run_dir: Path, note: str = "") -> RunResult | None:
    if not run_dir.exists():
        return None

    results_csv = run_dir / "results.csv"
    val_metrics_path = run_dir / "val_metrics.yaml"
    args = load_args_yaml(run_dir)

    if results_csv.exists():
        try:
            best_epoch, map50, map50_95 = load_metrics_from_csv(results_csv)
            model = str(args.get("model", ""))
            if not model and val_metrics_path.exists():
                with val_metrics_path.open(encoding="utf-8") as f:
                    model = str((yaml.safe_load(f) or {}).get("weights", "yolo11s.pt"))
            if not model:
                model = "yolo11s.pt"
            epochs_configured = int(args.get("epochs", best_epoch))
            return RunResult(
                run_dir=run_dir.resolve(),
                run_name=run_dir.name,
                model=model,
                epochs_configured=epochs_configured,
                best_epoch=best_epoch,
                map50=map50,
                map50_95=map50_95,
                updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                note=note,
            )
        except ValueError as exc:
            print(f"[경고] {run_dir.name} results.csv 파싱 실패: {exc}", file=sys.stderr)

    if val_metrics_path.exists():
        try:
            with val_metrics_path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return RunResult(
                run_dir=run_dir.resolve(),
                run_name=run_dir.name,
                model=str(data.get("weights", "yolo11s.pt")),
                epochs_configured=int(data.get("epoch", 50)),
                best_epoch=int(data.get("epoch", 50)),
                map50=float(data["map50"]),
                map50_95=float(data["map50_95"]),
                updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                note=note,
            )
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[경고] {run_dir.name} val_metrics.yaml 파싱 실패: {exc}", file=sys.stderr)

    return None


def find_val_run_dir(runs_dir: Path) -> Path | None:
    for name in ("val_final", "val_final-2"):
        path = runs_dir / name
        if path.exists():
            return path
    return None


def find_predict_run_dir(predict_root: Path = DEFAULT_PREDICT_DIR) -> Path | None:
    """runs/predict/ 아래 predictions.json이 있는 run 폴더 (val_batch 우선)."""
    if not predict_root.is_dir():
        return None

    preferred = predict_root / "val_batch"
    if (preferred / "predictions.json").is_file():
        return preferred

    candidates = [
        d for d in predict_root.iterdir()
        if d.is_dir() and (d / "predictions.json").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "predictions.json").stat().st_mtime)


def load_predict_payload(run_dir: Path) -> dict | None:
    path = run_dir / "predictions.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def summarize_predict_payload(payload: dict) -> dict:
    results = payload.get("results", [])
    by_class: dict[str, int] = {}
    for item in results:
        for det in item.get("detections", []):
            name = str(det.get("class_name", "?"))
            by_class[name] = by_class.get(name, 0) + 1

    with_detections = [r for r in results if int(r.get("detection_count", 0)) > 0]
    samples = sorted(
        with_detections,
        key=lambda r: int(r.get("detection_count", 0)),
        reverse=True,
    )[:3]

    total_images = len(results)
    images_with_det = len(with_detections)
    return {
        "total_images": total_images,
        "images_with_detections": images_with_det,
        "images_without_detections": total_images - images_with_det,
        "total_bboxes": sum(int(r.get("detection_count", 0)) for r in results),
        "detection_rate_pct": round(images_with_det / total_images * 100, 1) if total_images else 0.0,
        "by_class": by_class,
        "sample_sources": [str(r.get("source", "")) for r in samples],
    }


def resolve_predict_image_file(run_dir: Path, source_name: str) -> str | None:
    """predictions.json의 source 파일명 → runs/predict/ 실제 저장 파일 (jpg/png)."""
    direct = run_dir / source_name
    if direct.is_file():
        return source_name
    stem = Path(source_name).stem
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = run_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate.name
    return None


def update_predict_inference_section(
    content: str,
    run_dir: Path,
    payload: dict,
    summary: dict,
    updated_at: str,
) -> str:
    """predict.py Val 일괄 추론 집계·대표 이미지를 report.md에 반영합니다."""
    rel_run = to_repo_relative(run_dir)
    lines = [
        f"- **자동 반영:** {updated_at} (`predict.py` → `{run_dir.name}`)",
        f"- **가중치:** `{payload.get('weights', '—')}` | conf **{payload.get('conf', '—')}** | "
        f"device `{payload.get('device', '—')}`",
        f"- **입력:** `{payload.get('source', '—')}`",
        "",
        "> **Val 공식 평가(`val.py`)와 별도** — best.pt로 Val 전체에 추론만 수행한 Phase 1 Test 결과입니다.",
        "",
        "**Val 일괄 추론 집계**",
        "",
        "| 항목 | 값 |",
        "| :--- | ---: |",
        f"| **처리 이미지 수** | {summary['total_images']:,} |",
        f"| **탐지 있는 이미지** | {summary['images_with_detections']:,} ({summary['detection_rate_pct']}%) |",
        f"| **탐지 없음** | {summary['images_without_detections']:,} |",
        f"| **총 BBox** | {summary['total_bboxes']:,} |",
    ]

    if summary["by_class"]:
        lines.extend(["", "**클래스별 탐지 수**", "", "| 클래스 | BBox 수 |", "| :--- | ---: |"])
        for name, count in sorted(summary["by_class"].items()):
            display = name.capitalize() if name.islower() else name
            lines.append(f"| **{display}** | {count:,} |")

    sample_files: list[tuple[str, str]] = []
    for i, src in enumerate(summary["sample_sources"], start=1):
        filename = resolve_predict_image_file(run_dir, src)
        if not filename:
            continue
        det_count = next(
            (
                int(r.get("detection_count", 0))
                for r in payload.get("results", [])
                if r.get("source") == src
            ),
            0,
        )
        sample_files.append((filename, f"predict 추론 결과 {i} — {src} ({det_count} BBox)"))

    sample_images = build_image_lines(run_dir, sample_files, "predict")
    if sample_images:
        lines.extend(["", "**대표 추론 결과 (탐지 있음)**", ""])
        lines.extend(sample_images)

    lines.append("")
    lines.append(f"- **전체 결과:** `{rel_run}/predictions.json` · `{rel_run}/`")

    return replace_block(content, PREDICT_INFERENCE_START, PREDICT_INFERENCE_END, "\n".join(lines))


def to_repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def publish_report_asset(source: Path, asset_rel: str) -> str:
    """runs/ 산출물을 GitHub 공개용 report/assets/ 로 복사하고 상대 경로를 반환."""
    dest = REPORT_ASSETS_DIR / asset_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return to_repo_relative(dest)


def build_image_lines(
    run_dir: Path,
    files: list[tuple[str, str]],
    asset_namespace: str,
) -> list[str]:
    lines: list[str] = []
    for filename, caption in files:
        image_path = run_dir / filename
        if image_path.exists():
            rel = publish_report_asset(image_path, f"{asset_namespace}/{filename}")
            lines.append(f"![{caption}]({rel})")
    return lines


def replace_block(content: str, start: str, end: str, body: str) -> str:
    block = f"{start}\n{body}\n{end}"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(content):
        return pattern.sub(block, content, count=1)
    return content


def update_baseline_row(content: str, baseline: RunResult | None) -> str:
    if baseline is None:
        return content
    note = baseline.note or f"자동 반영 ({baseline.run_name})"
    new_row = (
        f"| **Baseline** | {format_model_name(baseline.model)} | {baseline.best_epoch} | "
        f"{baseline.map50:.3f} | {baseline.map50_95:.3f} | {note} |"
    )
    updated, count = BASELINE_ROW_PATTERN.subn(new_row, content, count=1)
    if count == 0:
        print("[경고] Baseline 표 행을 찾지 못했습니다.", file=sys.stderr)
    return updated


def update_final_model_row(
    content: str, final: RunResult, baseline: RunResult | None
) -> str:
    note = final.note or f"자동 반영 ({final.run_name}, epoch {final.best_epoch})"
    new_row = (
        f"| **최종 모델** | {format_model_name(final.model)} | {final.best_epoch} | "
        f"{final.map50:.3f} | {final.map50_95:.3f} | {note} |"
    )
    updated, count = FINAL_MODEL_ROW_PATTERN.subn(new_row, content, count=1)
    if count == 0:
        print("[경고] '최종 모델' 표 행을 찾지 못했습니다.", file=sys.stderr)
        return content

    if baseline:
        gain50 = (final.map50 - baseline.map50) * 100
        gain95 = (final.map50_95 - baseline.map50_95) * 100
        gain_row = (
            f"| **성능 향상** | - | - | **+ {gain50:.1f}%p** | **+ {gain95:.1f}%p** | "
            f"Baseline 대비 개선 |"
        )
    else:
        gain_row = (
            "| **성능 향상** | - | - | — | — | Baseline 학습 완료 후 자동 계산 |"
        )
    updated = IMPROVEMENT_ROW_PATTERN.subn(gain_row, updated, count=1)[0]
    return updated


def load_train_hyperparameters(train_dir: Path | None) -> dict[str, int | float]:
    """EXP 3 비고·리포트용 — train/args.yaml 우선, 없으면 configs/train.yaml."""
    if train_dir:
        args = load_args_yaml(train_dir)
        if args:
            return {
                "epochs": int(args.get("epochs", 50)),
                "batch": int(args.get("batch", 8)),
                "patience": int(args.get("patience", 10)),
            }
    if DEFAULT_TRAIN_CONFIG.is_file():
        with DEFAULT_TRAIN_CONFIG.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        hp = cfg.get("hyperparameters", {})
        return {
            "epochs": int(hp.get("epochs", 50)),
            "batch": int(hp.get("batch", 8)),
            "patience": int(hp.get("patience", 10)),
        }
    return {"epochs": 50, "batch": 8, "patience": 10}


def format_split_ratio_pct(ratio: float) -> str:
    return f"{ratio * 100:.1f}%"


def update_split_section(content: str, summary: dict, updated_at: str) -> str:
    """섹션 1 Train/Val 분할 표·구성 표 자동 갱신."""
    train = summary.get("train", {})
    val = summary.get("val", {})
    total = summary.get("total", {})
    total_images = int(total.get("images", 0))
    train_total = int(train.get("total", 0))
    val_total = int(val.get("total", 0))

    train_pct = format_split_ratio_pct(train_total / total_images) if total_images else "—"
    val_pct = format_split_ratio_pct(val_total / total_images) if total_images else "—"

    seed = summary.get("seed")
    seed_text = str(seed) if seed is not None else "—"
    source = summary.get("source", "unknown")

    lines = [
        f"- **자동 반영:** {updated_at} (`{source}`)",
        "",
        "| 항목 | 이미지 수 | 비율 |",
        "| :--------------- | ----------: | ----: |",
        f"| **전체 (Total)** | **{total_images:,}장** | 100% |",
        f"| **Train (학습)** | **{train_total:,}장** | {train_pct} |",
        f"| **Val (검증)** | **{val_total:,}장** | {val_pct} |",
        "",
        "**데이터 구성**",
        "",
        "| 구분 | Train | Val | 합계 |",
        "| :--- | ---: | ---: | ---: |",
        f"| **라벨 있음** (객체 BBox) | {int(train.get('labeled', 0)):,} | "
        f"{int(val.get('labeled', 0)):,} | {int(total.get('labeled', 0)):,} |",
        f"| **배경** (라벨 없음, negative) | {int(train.get('background', 0)):,} | "
        f"{int(val.get('background', 0)):,} | {int(total.get('background', 0)):,} |",
        "",
        f"- **분할 비율:** 약 **8 : 2** (Train : Val), seed={seed_text}",
        "- **분할 방식:** 랜덤 셔플 후 8:2 분할 (`split_data.py`)",
        "- **평가 세트:** Test 세트는 별도로 두지 않음 — **Val 세트**로 최종 성능 평가",
    ]
    orphan = summary.get("orphan_labels")
    if isinstance(orphan, int) and orphan > 0:
        lines.append(f"- **고아 라벨:** {orphan}개 (대응 이미지 없음 — 분할 제외)")

    return replace_block(content, SPLIT_START, SPLIT_END, "\n".join(lines))


def update_exp_comparison_table(
    content: str,
    baseline: RunResult | None,
    exp1: RunResult | None,
    final: RunResult,
    train_dir: Path | None,
) -> str:
    """실측 run 표시 — Baseline · EXP1(있으면) · 최종."""
    rows = [
        "| 실험 | 모델 | Epoch | mAP50 | mAP50-95 | 비고 |",
        "| :--- | :--- | ---: | ---: | ---: | :--- |",
    ]

    if baseline:
        rows.append(
            f"| **Baseline** | {format_model_name(baseline.model)} | {baseline.best_epoch} | "
            f"{baseline.map50:.3f} | {baseline.map50_95:.3f} | YOLO11n · 최소 증강 |"
        )
    else:
        rows.append("| **Baseline** | YOLO11n (Nano) | 20 | — | — | 학습 진행 중 |")

    if exp1:
        rows.append(
            f"| **EXP 1** | {format_model_name(exp1.model)} | {exp1.best_epoch} | "
            f"{exp1.map50:.3f} | {exp1.map50_95:.3f} | Small만 변경·20ep (`train_exp1_small_minaug.yaml`) |"
        )

    rows.append(
        f"| **최종 모델** | {format_model_name(final.model)} | {final.best_epoch} | "
        f"{final.map50:.3f} | {final.map50_95:.3f} | EXP 1~3 통합 (`configs/train.yaml`) |"
    )

    if baseline:
        delta = (final.map50 - baseline.map50) * 100
        rows.append(
            f"| **개선** | — | — | **+ {delta:.1f}%p** | "
            f"+ {(final.map50_95 - baseline.map50_95) * 100:.1f}%p | Baseline 대비 (최종) |"
        )

    body = "\n".join(rows)
    content = replace_block(content, EXP_START, EXP_END, body)
    return update_hyper_tuning_section(content, baseline, exp1, final, train_dir)


def update_hyper_tuning_section(
    content: str,
    baseline: RunResult | None,
    exp1: RunResult | None,
    final: RunResult,
    train_dir: Path | None,
) -> str:
    hp = load_train_hyperparameters(train_dir)
    if exp1 and baseline:
        exp1_note = (
            f"EXP1 단독(20ep·min aug) mAP50 **{exp1.map50:.3f}** — "
            f"Small만으로는 Baseline(**{baseline.map50:.3f}**) 미달 → EXP2·3 필요 |"
        )
    elif baseline:
        exp1_note = (
            f"Baseline {baseline.map50:.3f} → 최종 {final.map50:.3f} "
            f"(+{(final.map50 - baseline.map50) * 100:.1f}%p) |"
        )
    else:
        exp1_note = "미세 Damage 탐지 위해 Small 채택 |"

    lines = [
        "**EXP 1~3은 누적 설계 단계** — EXP1은 독립 ablation 완료, EXP2·3은 최종 모델에 통합 반영.",
        "",
        "| 단계 | 변경 | 선택 | 근거 |",
        "| :--- | :--- | :--- | :--- |",
        f"| **EXP 1** | 모델 크기 | Nano → **Small** | {exp1_note}",
        "| **EXP 2** | Data Augmentation | HSV·Mosaic·Mixup·Erasing | 도메인(안개·반사) · `flipud=0` |",
        f"| **EXP 3** | Epoch · Batch · Patience | **{hp['epochs']}ep · batch {hp['batch']} · patience {hp['patience']}** | "
        "M1 16GB OOM → batch 8 · Cosine LR |",
        "",
        "> **한계:** EXP2·3은 별도 독립 run 없이 최종 설정(`train.yaml`)에 누적 반영.",
    ]
    return replace_block(content, HYPER_TUNING_START, HYPER_TUNING_END, "\n".join(lines))


def _confusion_error_counts(confusion: dict) -> dict[str, int]:
    labels = confusion.get("labels", [])
    matrix = confusion.get("predicted_rows", [])
    if len(labels) < 3 or len(matrix) < 3:
        return {}

    idx = {name: i for i, name in enumerate(labels)}
    dirt, damage, bg = idx.get("dirt"), idx.get("damage"), idx.get("background")
    if None in (dirt, damage, bg):
        return {}

    return {
        "dirt_correct": int(matrix[dirt][dirt]),
        "damage_correct": int(matrix[damage][damage]),
        "dirt_to_damage": int(matrix[dirt][damage]),
        "damage_to_dirt": int(matrix[damage][dirt]),
        "dirt_to_background_fn": int(matrix[bg][dirt]),
        "damage_to_background_fn": int(matrix[bg][damage]),
        "background_to_dirt_fp": int(matrix[dirt][bg]),
        "background_to_damage_fp": int(matrix[damage][bg]),
    }


def _load_predict_error_examples(predict_run_dir: Path | None) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    if predict_run_dir is None:
        return [], []
    payload_path = predict_run_dir / "predictions.json"
    if not payload_path.exists():
        return [], []

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    labels_dir = DEFAULT_DATA_DIR / "labels" / "val"
    fn_cases: list[tuple[str, int]] = []
    fp_cases: list[tuple[str, int]] = []

    for item in payload.get("results", []):
        source = str(item.get("source", ""))
        stem = Path(source).stem
        label_path = labels_dir / f"{stem}.txt"
        has_label = label_path.exists() and label_path.read_text(encoding="utf-8").strip()
        det_count = int(item.get("detection_count", 0))
        if has_label and det_count == 0:
            gt_count = len(label_path.read_text(encoding="utf-8").strip().splitlines())
            fn_cases.append((source, gt_count))
        if not has_label and det_count >= 3:
            fp_cases.append((source, det_count))

    fn_cases.sort(key=lambda x: -x[1])
    fp_cases.sort(key=lambda x: -x[1])
    return fn_cases[:5], fp_cases[:5]


def update_error_analysis_section(
    content: str,
    val_dir: Path | None,
    predict_run_dir: Path | None,
    updated_at: str,
) -> str:
    val_metrics = load_val_metrics(val_dir) if val_dir else None
    if not val_metrics:
        return content

    lines = [f"- **자동 반영:** {updated_at} (`val_final` + `predict.py`)"]

    classes = val_metrics.get("classes") or {}
    if classes:
        lines.extend(["", "**클래스별 Val 지표**", ""])
        lines.extend(
            [
                "| 클래스 | Precision | Recall | mAP50 |",
                "| :--- | ---: | ---: | ---: |",
            ]
        )
        for name in ("dirt", "damage"):
            info = classes.get(name)
            if not info:
                continue
            label = "Dirt (0)" if name == "dirt" else "Damage (1)"
            lines.append(
                f"| **{label}** | {float(info['precision']):.3f} | "
                f"{float(info['recall']):.3f} | {float(info['map50']):.3f} |"
            )

    confusion = val_metrics.get("confusion_matrix") or {}
    errors = _confusion_error_counts(confusion)
    if errors:
        lines.extend(["", "**혼동행렬 기반 오류 패턴 (BBox 단위)**", ""])
        lines.extend(
            [
                "| 패턴 | 건수 | 해석 |",
                "| :--- | ---: | :--- |",
                f"| **Damage → Background (FN)** | **{errors['damage_to_background_fn']}** | Damage 미탐 (핵심 이슈) |",
                f"| Background → Damage (FP) | {errors['background_to_damage_fp']} | 배경 오탐 |",
                f"| Dirt → Background (FN) | {errors['dirt_to_background_fn']} | Dirt 미탐 |",
                f"| Background → Dirt (FP) | {errors['background_to_dirt_fp']} | Dirt 오탐 |",
                f"| Dirt ↔ Damage 혼동 | {errors['dirt_to_damage'] + errors['damage_to_dirt']} | 클래스 간 혼동 **낮음** |",
            ]
        )

    fn_cases, fp_cases = _load_predict_error_examples(predict_run_dir)
    if fn_cases or fp_cases:
        lines.extend(["", "**대표 오류 사례 (predict.py 스캔)**", ""])
        lines.extend(["| 유형 | 이미지 | GT/탐지 |", "| :--- | :--- | ---: |"])
        for source, gt in fn_cases[:3]:
            lines.append(f"| **FN (미탐)** | `{source}` | GT BBox **{gt}** · 탐지 0 |")
        for source, det in fp_cases[:2]:
            lines.append(f"| **FP (오탐)** | `{source}` | 배경 · 탐지 **{det}** |")

    body = "\n".join(lines)
    return replace_block(content, ERROR_ANALYSIS_START, ERROR_ANALYSIS_END, body)


def update_run_summary(
    content: str, final: RunResult, val_dir: Path | None
) -> str:
    val_note = f"`{val_dir.name}`" if val_dir else "—"
    lines = [
        f"- **최종 학습:** `{final.run_name}` | mAP50 **{final.map50:.3f}** | mAP50-95 **{final.map50_95:.3f}**",
        f"- **재검증:** {val_note}",
    ]
    if val_dir:
        val_metrics = load_val_metrics(val_dir)
        if val_metrics:
            lines.append(
                f"- **Val 메트릭:** mAP50 **{float(val_metrics['map50']):.3f}** | "
                f"Precision **{float(val_metrics['precision']):.3f}** | "
                f"Recall **{float(val_metrics['recall']):.3f}**"
            )
    lines.append(f"- **갱신 시각:** {final.updated_at}")
    return replace_block(content, RUN_SUMMARY_START, RUN_SUMMARY_END, "\n".join(lines))


def load_eda_summary(eda_dir: Path) -> dict | None:
    """runs/eda/eda_summary.yaml 을 읽습니다."""
    summary_path = eda_dir / "eda_summary.yaml"
    if not summary_path.exists():
        return None
    with summary_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or None


def update_eda_section(content: str, eda_dir: Path, updated_at: str) -> str:
    """
    EDA 섹션 자동 갱신 — 클래스 분포 표·인사이트·그래프 이미지 링크 삽입.
    Notion 업로드 시 update_notion.py가 이미지를 File Upload API로 변환합니다.
    """
    summary = load_eda_summary(eda_dir)
    if summary is None:
        return content

    lines: list[str] = [f"- **자동 반영:** {updated_at}"]
    lines.append(f"- **총 BBox:** {summary.get('total_bboxes', '—')}개")
    lines.append("")
    lines.append("**클래스별 BBox 분포**")
    lines.append("")
    lines.append("| 클래스 | BBox 수 | 비율 | 극소형 비율 (w,h < 0.2) |")
    lines.append("| :--- | ---: | ---: | ---: |")

    by_class = summary.get("by_class", {})
    for class_name, info in by_class.items():
        lines.append(
            f"| **{class_name}** | {info.get('count', '—')} | "
            f"{info.get('ratio_pct', '—')}% | {info.get('small_bbox_ratio_pct', '—')}% |"
        )

    insights = summary.get("insights", [])
    if insights:
        lines.append("")
        lines.append("**주요 인사이트**")
        lines.append("")
        for insight in insights:
            lines.append(f"- {insight}")

    eda_images = build_image_lines(
        eda_dir,
        [
            ("class_distribution.png", "Class Distribution: Dirt vs Damage"),
            ("bbox_size_distribution.png", "Bounding Box Size Distribution (Normalized)"),
            ("bbox_area_distribution.png", "Bounding Box Area Distribution"),
        ],
        "eda",
    )
    if eda_images:
        lines.append("")
        lines.append("**시각화**")
        lines.append("")
        lines.extend(eda_images)

    body = "\n".join(lines)
    return replace_block(content, EDA_START, EDA_END, body)


def load_val_metrics(val_dir: Path) -> dict | None:
    """val_final/val_metrics.yaml 에서 Precision·Recall 등을 읽습니다."""
    metrics_path = val_dir / "val_metrics.yaml"
    if not metrics_path.exists():
        return None
    with metrics_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or None


def update_visual_sections(
    content: str,
    train_dir: Path | None,
    val_dir: Path | None,
    updated_at: str,
) -> str:
    metrics_images: list[str] = []
    if train_dir:
        metrics_images += build_image_lines(
            train_dir,
            [("results.png", "Loss/mAP 학습 곡선 (Train)")],
            "metrics",
        )
    if val_dir:
        metrics_images += build_image_lines(
            val_dir,
            [
                ("confusion_matrix.png", "Confusion Matrix (Val)"),
                ("BoxF1_curve.png", "Box F1 Curve (Val)"),
            ],
            "metrics",
        )

    if metrics_images:
        metrics_lines = [f"- **자동 반영:** {updated_at}"]
        if val_dir:
            val_metrics = load_val_metrics(val_dir)
            if val_metrics:
                metrics_lines.append(
                    f"- **Val 재검증 (`{val_dir.name}`):** "
                    f"mAP50 **{float(val_metrics['map50']):.3f}** | "
                    f"mAP50-95 **{float(val_metrics['map50_95']):.3f}** | "
                    f"Precision **{float(val_metrics['precision']):.3f}** | "
                    f"Recall **{float(val_metrics['recall']):.3f}**"
                )
        body = "\n".join(metrics_lines) + "\n\n" + "\n\n".join(metrics_images)
        content = replace_block(content, METRICS_START, METRICS_END, body)

    if val_dir:
        preds = build_image_lines(
            val_dir,
            [
                ("val_batch0_pred.jpg", "검증 예측 결과 1"),
                ("val_batch1_pred.jpg", "검증 예측 결과 2"),
                ("val_batch2_pred.jpg", "검증 예측 결과 3"),
            ],
            "val",
        )
        if preds:
            body = f"- **Dirt / Damage 탐지 결과** — `{val_dir.name}`\n\n" + "\n\n".join(preds)
            content = replace_block(content, PREDICTIONS_START, PREDICTIONS_END, body)

    return content


def apply_report_updates(
    content: str,
    final: RunResult | None,
    baseline: RunResult | None,
    exp1: RunResult | None,
    train_dir: Path | None,
    val_dir: Path | None,
    eda_dir: Path | None = None,
    split_summary: dict | None = None,
    predict_run_dir: Path | None = None,
    updated_at: str | None = None,
) -> str:
    ts = updated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updated = content

    if split_summary:
        updated = update_split_section(updated, split_summary, ts)

    if predict_run_dir:
        payload = load_predict_payload(predict_run_dir)
        if payload:
            summary = summarize_predict_payload(payload)
            ts_predict = str(payload.get("predicted_at", ts))
            updated = update_predict_inference_section(
                updated, predict_run_dir, payload, summary, ts_predict
            )

    if final is None:
        if eda_dir and eda_dir.exists():
            updated = update_eda_section(updated, eda_dir, ts)
        updated = update_visual_sections(updated, train_dir, val_dir, ts)
        updated = update_error_analysis_section(updated, val_dir, predict_run_dir, ts)
        return updated

    updated = update_baseline_row(updated, baseline)
    updated = update_final_model_row(updated, final, baseline)
    updated = update_exp_comparison_table(updated, baseline, exp1, final, train_dir)
    updated = update_run_summary(updated, final, val_dir)
    if eda_dir and eda_dir.exists():
        updated = update_eda_section(updated, eda_dir, final.updated_at)
    updated = update_visual_sections(updated, train_dir, val_dir, final.updated_at)
    updated = update_error_analysis_section(
        updated, val_dir, predict_run_dir, final.updated_at
    )
    return updated


def main() -> None:
    args = parse_args()
    report_path = args.report.resolve()
    runs_dir = args.runs_dir.resolve()
    data_dir = args.data_dir.resolve()

    if not report_path.exists():
        raise SystemExit(f"리포트 파일을 찾을 수 없습니다: {report_path}")

    if args.run_dir is not None:
        print(
            "[참고] --run-dir는 더 이상 사용하지 않습니다. "
            "최종 모델 표는 runs/detect/train 기준으로 갱신합니다.",
            file=sys.stderr,
        )

    train_dir = runs_dir / "train"
    val_dir = find_val_run_dir(runs_dir)
    baseline_dir = runs_dir / "baseline"
    exp1_dir = runs_dir / "exp1_small_minaug"
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    split_summary = load_split_summary(ROOT / DEFAULT_SPLIT_SUMMARY, data_dir)
    final = try_collect_run_result(train_dir, note="본학습 best")
    baseline = try_collect_run_result(baseline_dir, note="Baseline Nano 학습")
    exp1 = try_collect_run_result(exp1_dir, note="EXP1 Small·min aug")
    eda_dir = DEFAULT_EDA_DIR if DEFAULT_EDA_DIR.exists() else None
    predict_run_dir = find_predict_run_dir()

    original = report_path.read_text(encoding="utf-8")
    updated = apply_report_updates(
        original,
        final,
        baseline,
        exp1,
        train_dir if train_dir.exists() else None,
        val_dir,
        eda_dir,
        split_summary,
        predict_run_dir,
        updated_at,
    )

    if split_summary:
        total = split_summary.get("total", {}).get("images", "—")
        print(f"데이터 분할: 총 {total}장 (report.md 섹션 1 반영)")
    else:
        print("데이터 분할: split_summary.yaml·data/ 폴더 없음 — 분할 표 미갱신")

    if final:
        print(f"최종 모델: {final.run_dir} (mAP50={final.map50:.3f})")
    else:
        print("최종 모델: 없음 (train/ results.csv 없음 — 학습 표·그래프 미갱신)")

    if baseline:
        print(f"Baseline: {baseline.run_dir} (mAP50={baseline.map50:.3f})")
    elif final:
        print("Baseline: 아직 없음 (python train.py --config configs/train_baseline.yaml --no-report)")

    if exp1:
        print(f"EXP1: {exp1.run_dir} (mAP50={exp1.map50:.3f}, best epoch {exp1.best_epoch})")

    if eda_dir:
        print(f"EDA: {eda_dir} (report.md 섹션 1 EDA 반영)")
    elif final or split_summary:
        print("EDA: 없음 (python eda.py 실행 후 update_report.py 재실행)")

    if predict_run_dir:
        payload = load_predict_payload(predict_run_dir)
        if payload:
            s = summarize_predict_payload(payload)
            print(
                f"predict 추론: {predict_run_dir.name} "
                f"({s['total_images']}장, BBox {s['total_bboxes']}개 — report.md 반영)"
            )
    else:
        print("predict 추론: 없음 (python predict.py --source data/images/val --name val_batch)")

    if updated == original:
        print("변경 사항이 없습니다.")
        return

    if args.dry_run:
        print("\n--- dry-run ---\n")
        print(updated)
        return

    report_path.write_text(updated, encoding="utf-8")
    print(f"report.md 갱신 완료: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        sys.exit(1)
