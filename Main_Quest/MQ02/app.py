"""
app.py — Phase 3 UI MVP (Streamlit 웹 데모)

풍력 터빈 블레이드 Dirt/Damage 탐지 — 학습된 best.pt 실시간 시연용.

실행:
  streamlit run app.py

설정:
  configs/predict.yaml (가중치·device·기본 conf)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import yaml
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
PREDICT_CONFIG = ROOT / "configs" / "predict.yaml"
DEFAULT_WEIGHTS = ROOT / "runs/detect/train/weights/best.pt"


def load_predict_config() -> dict[str, Any]:
    if not PREDICT_CONFIG.exists():
        return {}
    with PREDICT_CONFIG.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_weights(cfg: dict[str, Any]) -> Path:
    raw = Path(cfg.get("weights", DEFAULT_WEIGHTS))
    path = raw if raw.is_absolute() else (ROOT / raw)
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"가중치를 찾을 수 없습니다: {path}\n"
            "먼저 `python train.py`로 학습을 완료하세요."
        )
    return path


@st.cache_resource
def load_model(weights_str: str, device: str) -> YOLO:
    """모델은 서버 기동 시 1회만 로드 (@st.cache_resource)."""
    return YOLO(weights_str)


def count_by_class(results: Any) -> dict[str, int]:
    counts = {"dirt": 0, "damage": 0}
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return counts
    names = results[0].names
    for cls_id in boxes.cls.cpu().tolist():
        name = str(names.get(int(cls_id), cls_id)).lower()
        if name in counts:
            counts[name] += 1
    return counts


# --- 페이지 설정 ---
st.set_page_config(
    page_title="풍력 발전기 결함 탐지",
    page_icon="🌪️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

predict_cfg = load_predict_config()
params = predict_cfg.get("parameters", {})
try:
    weights_path = resolve_weights(predict_cfg)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

device = str(predict_cfg.get("device", "mps"))
default_conf = float(params.get("conf", 0.25))
default_iou = float(params.get("iou", 0.7))
default_imgsz = int(params.get("imgsz", 640))

st.title("🌪️ 풍력 발전기 표면 결함 탐지 AI")
st.caption(
    "드론 촬영 풍력 터빈 블레이드 이미지를 업로드하면 "
    "AI가 **오염(Dirt)** 과 **손상(Damage)** 을 자동 탐지합니다."
)

# --- 사이드바 ---
with st.sidebar:
    st.header("⚙️ 설정")
    conf_threshold = st.slider(
        "신뢰도 임계값 (Confidence)",
        min_value=0.1,
        max_value=1.0,
        value=default_conf,
        step=0.05,
        help="낮을수록 더 많이 탐지(Recall↑) · 오탐(FP)도 증가할 수 있습니다.",
    )
    st.info(
        "💡 **Recall–Precision Trade-off:** "
        "임계값을 낮추면 미세 Damage도 잡지만 오탐이 늘 수 있습니다."
    )
    st.divider()
    st.markdown("**모델 정보**")
    st.code(str(weights_path.relative_to(ROOT)), language=None)
    st.text(f"device: {device}")

model = load_model(str(weights_path), device)

# --- 업로드 ---
uploaded_file = st.file_uploader(
    "검사할 이미지를 업로드하세요 (jpg, png, jpeg)",
    type=["png", "jpg", "jpeg"],
)

# --- 탐지 실행 및 결과 ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 원본 이미지")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("🔍 AI 탐지 결과")

        if st.button("결함 탐지 실행", type="primary", use_container_width=True):
            with st.spinner("AI가 이미지를 분석 중입니다..."):
                img_array = np.array(image)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                results = model.predict(
                    source=img_bgr,
                    device=device,
                    conf=conf_threshold,
                    iou=default_iou,
                    imgsz=default_imgsz,
                    save=False,
                    verbose=False,
                )

                plotted = results[0].plot()
                result_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                counts = count_by_class(results)

            st.image(result_rgb, use_container_width=True)
            st.success("탐지 완료!")

            m1, m2, m3 = st.columns(3)
            total = counts["dirt"] + counts["damage"]
            m1.metric("총 BBox", total)
            m2.metric("Dirt (오염)", counts["dirt"])
            m3.metric("Damage (손상)", counts["damage"])

            if total == 0:
                st.warning(
                    "탐지된 객체가 없습니다. "
                    "신뢰도 임계값을 낮춰 보세요."
                )
else:
    st.info("👆 이미지를 업로드한 뒤 **결함 탐지 실행** 버튼을 눌러 주세요.")

st.divider()
st.caption(
    "YOLO11s · Apple M1 (MPS) · "
    "Phase 3 Streamlit MVP · "
    f"가중치: `{weights_path.name}`"
)
