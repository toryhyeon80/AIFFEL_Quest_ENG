"""
split_data.py — 데이터 전처리: Train / Val 분할 스크립트
======================================================

[역할]
  풍력 터빈 블레이드 이미지·YOLO 라벨을 학습용(train)과 검증용(val)으로
  무작위 분할한 뒤, 하위 폴더로 **이동(move)** 합니다.

[파이프라인 위치]
  ① split_data.py  →  ② train.py  →  ③ val.py
  (이 파일이 Phase 1의 첫 단계)

[입력 구조]  (실행 전)
  data/images/   ← 루트에 이미지 파일 (.jpg, .png 등)
  data/labels/   ← 루트에 라벨 파일 (.txt, YOLO 형식)

[출력]
  data/images/train,  data/images/val
  data/labels/train,  data/labels/val
  runs/split_summary.yaml  ← report.md 분할 표 자동 갱신용

[핵심 규칙]
  - 이미지·라벨은 파일명 stem(확장자 제외) 기준 1:1 매칭
  - 라벨 없는 이미지 = 배경(negative) 샘플 → 이미지만 train/val에 포함
  - 이미지 없는 라벨(고아 라벨) → 경고 후 이동하지 않음

[사용 예]
  python split_data.py
  python split_data.py --train-ratio 0.8 --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from split_stats import (
    DEFAULT_SPLIT_SUMMARY,
    build_split_summary_from_samples,
    save_split_summary,
)

ROOT = Path(__file__).resolve().parent

# YOLO 학습에서 허용하는 이미지 확장자 목록
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

# 생성할 분할 폴더 이름 (test 세트는 현재 미사용)
SPLIT_NAMES = ("train", "val")


@dataclass(frozen=True)
class Sample:
    """
    분할의 최소 단위: 이미지 1장 + (선택) 라벨 1개.

    label_path가 None이면 배경 이미지(객체 라벨 없음)로 취급합니다.
    """

    stem: str              # 이미지·라벨 공통 파일명 (확장자 제외)
    image_path: Path       # 원본 이미지 경로 (data/images/ 루트)
    label_path: Path | None  # 대응 라벨 경로. 없으면 None


def parse_args() -> argparse.Namespace:
    """CLI 인자 파싱 — 분할 비율·시드·데이터 루트 경로."""
    parser = argparse.ArgumentParser(
        description="data/images, data/labels를 8:2 비율로 train/val로 분할합니다."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="데이터 루트 디렉터리 (기본: data)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="학습 데이터 비율 (기본: 0.8 → 8:2 분할)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 분할 시드 — 동일 시드면 항상 같은 분할 결과 (재현성)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="분할 후 report.md 자동 갱신 건너뛰기",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    """지원 확장자의 일반 이미지 파일인지 확인."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_samples(images_dir: Path, labels_dir: Path) -> tuple[list[Sample], list[Path]]:
    """
    루트 images/labels 폴더에서 분할 대상 샘플을 수집합니다.

    [동작]
      1. images/ 루트의 이미지 파일 목록 수집 (하위 train/val 폴더는 무시)
      2. labels/ 루트의 .txt 라벨 파일 목록 수집
      3. stem 기준으로 이미지–라벨 짝 생성
      4. 라벨만 있고 이미지가 없는 고아 라벨 목록 반환

    Returns:
        samples: 분할 대상 Sample 리스트 (이미지 기준)
        orphan_labels: 대응 이미지가 없는 라벨 경로 목록
    """
    # 루트 디렉터리 직속 파일만 대상 (이미 분할된 train/val 하위는 제외)
    image_files = [
        p for p in images_dir.iterdir() if is_image_file(p) and p.parent == images_dir
    ]
    label_files = [
        p for p in labels_dir.iterdir() if p.is_file() and p.suffix == ".txt" and p.parent == labels_dir
    ]

    # stem → Path 딕셔너리로 O(1) 매칭
    labels_by_stem = {p.stem: p for p in label_files}
    images_by_stem = {p.stem: p for p in image_files}

    samples: list[Sample] = []
    for stem, image_path in sorted(images_by_stem.items()):
        samples.append(
            Sample(
                stem=stem,
                image_path=image_path,
                # 라벨이 없으면 None → 배경 이미지로 train/val에 포함
                label_path=labels_by_stem.get(stem),
            )
        )

    # 이미지 없이 라벨만 있는 경우 — 분할·이동하지 않음
    orphan_labels = [
        path for stem, path in labels_by_stem.items() if stem not in images_by_stem
    ]
    return samples, orphan_labels


def split_samples(
    samples: list[Sample], train_ratio: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """
    전체 샘플을 train_ratio : (1 - train_ratio) 비율로 무작위 분할합니다.

    [예외 처리]
      - train이 0장이 되지 않도록 최소 1장 보장
      - val이 0장이 되지 않도록 train이 전체를 차지하면 1장을 val에 양보
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio는 0과 1 사이여야 합니다.")

    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)  # 고정 시드로 재현 가능한 셔플

    train_count = int(len(shuffled) * train_ratio)

    # 극단적 소량 데이터에서 빈 split 방지
    if train_count == 0 and shuffled:
        train_count = 1
    if train_count == len(shuffled) and len(shuffled) > 1:
        train_count = len(shuffled) - 1

    return shuffled[:train_count], shuffled[train_count:]


def ensure_split_dirs(images_dir: Path, labels_dir: Path) -> None:
    """train/val 하위 폴더가 없으면 생성 (images·labels 양쪽)."""
    for split in SPLIT_NAMES:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)


def move_sample(sample: Sample, split: str, images_dir: Path, labels_dir: Path) -> None:
    """
    Sample 1개를 지정 split(train 또는 val) 폴더로 이동합니다.

    - 이미지는 항상 이동
    - 라벨은 존재할 때만 이동 (배경 이미지는 labels 쪽 이동 없음)
    - shutil.move 사용 → 루트에서 파일이 사라지고 하위로 이동
    """
    dest_image = images_dir / split / sample.image_path.name
    shutil.move(str(sample.image_path), str(dest_image))

    if sample.label_path is not None:
        dest_label = labels_dir / split / sample.label_path.name
        shutil.move(str(sample.label_path), str(dest_label))


def print_summary(
    train_samples: list[Sample],
    val_samples: list[Sample],
    orphan_labels: list[Path],
) -> None:
    """분할 결과 요약 출력 — 라벨 있음/배경 이미지 수, 고아 라벨 경고."""
    def count_with_label(items: list[Sample]) -> int:
        return sum(1 for s in items if s.label_path is not None)

    def count_background(items: list[Sample]) -> int:
        return sum(1 for s in items if s.label_path is None)

    print("\n=== 분할 결과 ===")
    print(f"Train: {len(train_samples)}장 (라벨 있음 {count_with_label(train_samples)}, 배경 {count_background(train_samples)})")
    print(f"Val  : {len(val_samples)}장 (라벨 있음 {count_with_label(val_samples)}, 배경 {count_background(val_samples)})")

    if orphan_labels:
        print(f"\n[경고] 대응 이미지가 없는 라벨 {len(orphan_labels)}개 — 이동하지 않았습니다.")
        for path in orphan_labels[:5]:
            print(f"  - {path.name}")
        if len(orphan_labels) > 5:
            print(f"  ... 외 {len(orphan_labels) - 5}개")


def main() -> None:
    """
    메인 실행 흐름:
      1. 인자 파싱
      2. 샘플 수집 (이미지 기준 + 고아 라벨 검출)
      3. 8:2 무작위 분할
      4. train/val 폴더 생성
      5. 파일 이동
      6. 요약 출력
    """
    args = parse_args()
    images_dir = args.data_dir / "images"
    labels_dir = args.data_dir / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"이미지 폴더를 찾을 수 없습니다: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"라벨 폴더를 찾을 수 없습니다: {labels_dir}")

    samples, orphan_labels = collect_samples(images_dir, labels_dir)
    if not samples:
        raise RuntimeError("분할할 이미지가 없습니다.")

    train_samples, val_samples = split_samples(samples, args.train_ratio, args.seed)
    ensure_split_dirs(images_dir, labels_dir)

    for sample in train_samples:
        move_sample(sample, "train", images_dir, labels_dir)
    for sample in val_samples:
        move_sample(sample, "val", images_dir, labels_dir)

    print_summary(train_samples, val_samples, orphan_labels)

    summary = build_split_summary_from_samples(
        train_samples,
        val_samples,
        args.train_ratio,
        args.seed,
        orphan_labels,
        args.data_dir,
    )
    summary_path = save_split_summary(summary, ROOT / DEFAULT_SPLIT_SUMMARY)
    print(f"\n분할 요약 저장: {summary_path.relative_to(ROOT)}")

    if not args.no_report:
        print("report.md 자동 갱신을 시도합니다 (update_report.py)...")
        subprocess.run(
            [sys.executable, str(ROOT / "update_report.py")],
            cwd=ROOT,
            check=False,
        )


if __name__ == "__main__":
    main()
