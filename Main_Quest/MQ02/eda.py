"""
eda.py — YOLO 라벨 EDA 및 시각화 스크립트
=========================================

[역할]
  학습 전(또는 학습 후) 데이터셋 특성을 파악하기 위한 탐색적 데이터 분석(EDA).
  YOLO 형식 라벨(.txt)을 읽어 클래스 분포·BBox 크기 분포 등을 시각화합니다.

[파이프라인 권장 위치]
  ① split_data.py  →  ② eda.py  →  ③ train.py  →  ④ val.py
  (학습 전 실행 권장. 학습 후에도 동일 데이터로 재실행 가능)

[입력]
  data/labels/train, data/labels/val  (또는 --labels-dir 로 지정)

[산출물]  runs/eda/
  - class_distribution.png      : 클래스별 BBox 개수 막대 그래프
  - bbox_size_distribution.png  : 정규화 Width×Height 산점도
  - bbox_area_distribution.png  : BBox 면적(w×h) 히스토그램
  - eda_summary.yaml            : 수치 요약 (리포트·발표 인용용)

[사용 예]
  python eda.py
  python eda.py --output runs/eda
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# GUI 없이 파일 저장만 — 서버·샌드박스 환경 호환
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_CONFIG = ROOT / "data" / "data.yaml"
DEFAULT_OUTPUT = ROOT / "runs" / "eda"

# YOLO 클래스 ID → 표시 이름 (첨부 EDA 이미지와 동일 형식)
CLASS_LABELS = {
    0: "Dirt (0)",
    1: "Damage (1)",
}
CLASS_COLORS = {
    0: "#f39c12",  # orange — Dirt
    1: "#e74c3c",  # red — Damage
}


@dataclass(frozen=True)
class BBoxRecord:
    """YOLO 라벨 1행 = BBox 1개."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    split: str       # train | val
    label_file: str  # 출처 라벨 파일명


def load_class_names(data_yaml: Path) -> dict[int, str]:
    """data.yaml의 names를 읽어 CLASS_LABELS 형식으로 변환."""
    if not data_yaml.exists():
        return CLASS_LABELS.copy()
    with data_yaml.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    names = cfg.get("names", {})
    if isinstance(names, dict):
        return {int(k): f"{v.capitalize()} ({k})" for k, v in names.items()}
    if isinstance(names, list):
        return {i: f"{n.capitalize()} ({i})" for i, n in enumerate(names)}
    return CLASS_LABELS.copy()


def collect_label_dirs(labels_root: Path) -> list[tuple[str, Path]]:
    """
    분석 대상 라벨 폴더 목록 반환.

    train/val 하위 폴더가 있으면 둘 다 사용, 없으면 루트만 사용.
    """
    dirs: list[tuple[str, Path]] = []
    for split in ("train", "val"):
        split_dir = labels_root / split
        if split_dir.is_dir():
            dirs.append((split, split_dir))
    if not dirs and labels_root.is_dir():
        dirs.append(("all", labels_root))
    return dirs


def parse_label_file(path: Path, split: str) -> list[BBoxRecord]:
    """
    YOLO 라벨 파일 파싱.

    형식: class_id x_center y_center width height  (모두 0~1 정규화)
    빈 줄·주석(#)·classes.txt/labels.txt 는 건너뜀.
    """
    records: list[BBoxRecord] = []
    skip_names = {"classes.txt", "labels.txt"}

    if path.name in skip_names or not path.suffix == ".txt":
        return records

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return records

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            print(f"[경고] 형식 오류 무시: {path} L{line_no}: {line}")
            continue
        try:
            class_id = int(float(parts[0]))
            x, y, w, h = (float(parts[i]) for i in range(1, 5))
        except ValueError:
            print(f"[경고] 숫자 변환 실패 무시: {path} L{line_no}: {line}")
            continue
        records.append(
            BBoxRecord(
                class_id=class_id,
                x_center=x,
                y_center=y,
                width=w,
                height=h,
                split=split,
                label_file=path.name,
            )
        )
    return records


def load_all_bboxes(labels_root: Path) -> list[BBoxRecord]:
    """train/val 라벨 폴더 전체에서 BBox 레코드 수집."""
    all_records: list[BBoxRecord] = []
    label_dirs = collect_label_dirs(labels_root)

    if not label_dirs:
        raise FileNotFoundError(f"라벨 폴더를 찾을 수 없습니다: {labels_root}")

    for split, folder in label_dirs:
        for path in sorted(folder.glob("*.txt")):
            all_records.extend(parse_label_file(path, split))

    if not all_records:
        raise RuntimeError(f"분석할 BBox가 없습니다. 라벨 경로를 확인하세요: {labels_root}")

    return all_records


def compute_summary(records: list[BBoxRecord], class_names: dict[int, str]) -> dict:
    """EDA 수치 요약 — eda_summary.yaml 및 콘솔 출력용."""
    by_class: dict[int, int] = {}
    by_split: dict[str, int] = {}
    small_threshold = 0.2  # 첨부 분석 기준: 0.2(20%) 미만 = 극소형 객체

    small_counts: dict[int, int] = {}
    areas: dict[int, list[float]] = {}

    for r in records:
        by_class[r.class_id] = by_class.get(r.class_id, 0) + 1
        by_split[r.split] = by_split.get(r.split, 0) + 1
        area = r.width * r.height
        areas.setdefault(r.class_id, []).append(area)

        is_small = r.width < small_threshold and r.height < small_threshold
        if is_small:
            small_counts[r.class_id] = small_counts.get(r.class_id, 0) + 1

    total = len(records)
    class_summary = {}
    for cid, count in sorted(by_class.items()):
        name = class_names.get(cid, f"class_{cid}")
        small = small_counts.get(cid, 0)
        class_areas = areas.get(cid, [0.0])
        class_summary[name] = {
            "count": count,
            "ratio_pct": round(count / total * 100, 2),
            "small_bbox_count": small,
            "small_bbox_ratio_pct": round(small / count * 100, 2) if count else 0.0,
            "mean_area": round(sum(class_areas) / len(class_areas), 6),
            "median_width": round(sorted([r.width for r in records if r.class_id == cid])[len(class_areas) // 2], 6),
            "median_height": round(sorted([r.height for r in records if r.class_id == cid])[len(class_areas) // 2], 6),
        }

    return {
        "total_bboxes": total,
        "total_label_files_with_bbox": len({r.label_file for r in records}),
        "by_split": by_split,
        "by_class": class_summary,
        "small_bbox_threshold": small_threshold,
        "insights": build_insights(by_class, class_names, small_counts, small_threshold),
    }


def build_insights(
    by_class: dict[int, int],
    class_names: dict[int, str],
    small_counts: dict[int, int],
    threshold: float,
) -> list[str]:
    """발표·리포트용 자동 인사이트 문장 생성."""
    insights: list[str] = []
    if len(by_class) >= 2:
        counts = sorted(by_class.items(), key=lambda x: x[1], reverse=True)
        top_id, top_n = counts[0]
        bottom_id, bottom_n = counts[-1]
        ratio = top_n / bottom_n if bottom_n else float("inf")
        insights.append(
            f"클래스 불균형: {class_names.get(top_id)} {top_n}개 vs "
            f"{class_names.get(bottom_id)} {bottom_n}개 (약 {ratio:.1f}배)."
        )

    for cid, count in by_class.items():
        small = small_counts.get(cid, 0)
        if count and small / count > 0.5:
            insights.append(
                f"{class_names.get(cid)} BBox의 {small / count * 100:.1f}%가 "
                f"width·height 모두 {threshold} 미만(극소형 객체)."
            )

    damage_id = 1
    if damage_id in by_class:
        small = small_counts.get(damage_id, 0)
        total = by_class[damage_id]
        if total and small / total > 0.6:
            insights.append(
                "Damage 클래스가 이미지 대비 매우 작은 BBox로 밀집 → "
                "imgsz 1024 이상 상향 또는 소형 객체 탐지 증강 검토 권장."
            )

    return insights


def plot_class_distribution(
    records: list[BBoxRecord],
    class_names: dict[int, str],
    output_path: Path,
) -> None:
    """클래스별 BBox 개수 막대 그래프 (첨부 이미지 2번 형식)."""
    counts: dict[str, int] = {}
    for r in records:
        label = class_names.get(r.class_id, f"Class ({r.class_id})")
        counts[label] = counts.get(label, 0) + 1

    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    colors = [CLASS_COLORS.get(int(k.split("(")[1].rstrip(")")), "#95a5a6") for k in labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=labels, y=values, hue=labels, palette=colors, legend=False, ax=ax)
    ax.set_title("Class Distribution: Dirt vs Damage", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Bounding Boxes")

    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, str(v), ha="center", va="bottom", fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bbox_size_distribution(
    records: list[BBoxRecord],
    class_names: dict[int, str],
    output_path: Path,
) -> None:
    """정규화 BBox Width×Height 산점도 (첨부 이미지 1번 형식)."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for class_id in sorted({r.class_id for r in records}):
        subset = [r for r in records if r.class_id == class_id]
        widths = [r.width for r in subset]
        heights = [r.height for r in subset]
        ax.scatter(
            widths,
            heights,
            alpha=0.35,
            s=18,
            c=CLASS_COLORS.get(class_id, "#95a5a6"),
            label=class_names.get(class_id, f"Class ({class_id})"),
            edgecolors="none",
        )

    ax.set_title("Bounding Box Size Distribution (Normalized)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Normalized Width (0~1)")
    ax.set_ylabel("Normalized Height (0~1)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.axvline(0.2, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Small (0.2)")
    ax.axhline(0.2, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bbox_area_distribution(
    records: list[BBoxRecord],
    class_names: dict[int, str],
    output_path: Path,
) -> None:
    """BBox 면적(w×h) 클래스별 히스토그램 — 소형 객체 밀도 보조 분석."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for class_id in sorted({r.class_id for r in records}):
        areas = [r.width * r.height for r in records if r.class_id == class_id]
        ax.hist(
            areas,
            bins=50,
            alpha=0.55,
            label=class_names.get(class_id, f"Class ({class_id})"),
            color=CLASS_COLORS.get(class_id, "#95a5a6"),
        )

    ax.set_title("Bounding Box Area Distribution (w × h)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Normalized Area (0~1)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_summary(summary: dict, output_path: Path) -> None:
    """수치 요약을 YAML로 저장."""
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, allow_unicode=True, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 라벨 EDA 및 시각화")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=ROOT / "data" / "labels",
        help="라벨 루트 디렉터리 (기본: data/labels)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="클래스명 참조용 data.yaml (기본: data/data.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="그래프·요약 저장 폴더 (기본: runs/eda)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="EDA 후 report.md 자동 갱신 건너뛰기",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_root = args.labels_dir.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(args.data.resolve())
    records = load_all_bboxes(labels_root)
    summary = compute_summary(records, class_names)

    plot_class_distribution(records, class_names, output_dir / "class_distribution.png")
    plot_bbox_size_distribution(records, class_names, output_dir / "bbox_size_distribution.png")
    plot_bbox_area_distribution(records, class_names, output_dir / "bbox_area_distribution.png")
    save_summary(summary, output_dir / "eda_summary.yaml")

    print("\n=== EDA 완료 ===")
    print(f"총 BBox: {summary['total_bboxes']}")
    for name, info in summary["by_class"].items():
        print(
            f"  {name}: {info['count']}개 "
            f"(극소형 {info['small_bbox_ratio_pct']}%)"
        )
    print("\n[인사이트]")
    for line in summary["insights"]:
        print(f"  - {line}")
    print(f"\n저장 위치: {output_dir}/")
    print("  - class_distribution.png")
    print("  - bbox_size_distribution.png")
    print("  - bbox_area_distribution.png")
    print("  - eda_summary.yaml")

    if not args.no_report:
        print("\nreport.md EDA 섹션 갱신 중 (update_report.py)...")
        subprocess.run(
            [sys.executable, str(ROOT / "update_report.py")],
            cwd=ROOT,
            check=False,
        )


if __name__ == "__main__":
    main()
