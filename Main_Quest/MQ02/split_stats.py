"""
split_stats.py — Train/Val 분할 통계 (split_data.py · update_report.py 공용)

split_data.py 실행 시 runs/split_summary.yaml 저장.
update_report.py는 YAML 우선, 없으면 data/images·labels 폴더를 직접 집계합니다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "val")
DEFAULT_SPLIT_SUMMARY = Path("runs/split_summary.yaml")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _count_split(images_dir: Path, labels_dir: Path, split: str) -> dict[str, int]:
    split_images = images_dir / split
    split_labels = labels_dir / split
    if not split_images.is_dir():
        return {"total": 0, "labeled": 0, "background": 0}

    image_stems = {
        p.stem for p in split_images.iterdir() if is_image_file(p) and p.parent == split_images
    }
    label_stems = {
        p.stem for p in split_labels.glob("*.txt") if p.is_file() and p.parent == split_labels
    }
    labeled = len(image_stems & label_stems)
    background = len(image_stems - label_stems)
    return {"total": len(image_stems), "labeled": labeled, "background": background}


def collect_split_stats_from_dirs(data_dir: Path) -> dict | None:
    """data/images/{train,val} · data/labels/{train,val} 폴더에서 통계를 집계합니다."""
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return None

    train = _count_split(images_dir, labels_dir, "train")
    val = _count_split(images_dir, labels_dir, "val")
    if train["total"] == 0 and val["total"] == 0:
        return None

    total = train["total"] + val["total"]
    labeled_total = train["labeled"] + val["labeled"]
    background_total = train["background"] + val["background"]

    train_ratio = round(train["total"] / total, 4) if total else 0.0
    val_ratio = round(val["total"] / total, 4) if total else 0.0

    return {
        "source": "folder_scan",
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(data_dir),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "seed": None,
        "orphan_labels": None,
        "train": train,
        "val": val,
        "total": {
            "images": total,
            "labeled": labeled_total,
            "background": background_total,
        },
    }


def build_split_summary_from_samples(
    train_samples: list,
    val_samples: list,
    train_ratio: float,
    seed: int,
    orphan_labels: list[Path],
    data_dir: Path,
) -> dict:
    """split_data.py Sample 리스트로 split_summary.yaml 페이로드를 만듭니다."""

    def _counts(samples: list) -> dict[str, int]:
        labeled = sum(1 for s in samples if s.label_path is not None)
        background = len(samples) - labeled
        return {"total": len(samples), "labeled": labeled, "background": background}

    train = _counts(train_samples)
    val = _counts(val_samples)
    total_images = train["total"] + val["total"]

    return {
        "source": "split_data.py",
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(data_dir),
        "train_ratio": train_ratio,
        "val_ratio": round(1.0 - train_ratio, 4),
        "seed": seed,
        "orphan_labels": len(orphan_labels),
        "train": train,
        "val": val,
        "total": {
            "images": total_images,
            "labeled": train["labeled"] + val["labeled"],
            "background": train["background"] + val["background"],
        },
    }


def save_split_summary(summary: dict, path: Path = DEFAULT_SPLIT_SUMMARY) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, allow_unicode=True, sort_keys=False)
    return path


def load_split_summary(
    summary_path: Path = DEFAULT_SPLIT_SUMMARY,
    data_dir: Path = Path("data"),
) -> dict | None:
    """YAML 우선, 없으면 data/ 폴더 직접 집계."""
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if data.get("total", {}).get("images"):
            return data
    return collect_split_stats_from_dirs(data_dir.resolve())
