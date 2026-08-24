#!/usr/bin/env python3
"""DP08 전체 데모 캡처 — baseline(8000/8501) + indoor(8001/8502)."""
from __future__ import annotations

import socket
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASELINE_ASSETS = ROOT / "assets"
INDOOR_ASSETS = ROOT / "indoor" / "assets"


def wait_port(port: int, timeout: float = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.3)
    return False


def capture_baseline(page, assets: Path) -> None:
    assets.mkdir(parents=True, exist_ok=True)

    # Swagger UI
    page.goto("http://127.0.0.1:8000/docs", wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(1500)
    page.screenshot(path=str(assets / "baseline_swagger.png"), full_page=True)

    # Streamlit — 정상 추론
    page.goto("http://127.0.0.1:8501", wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(2000)
    api = page.get_by_label("API Key")
    if api.count():
        api.fill("test-key-001")
    page.get_by_role("button", name="분석하기").click()
    page.get_by_text("score=", exact=False).wait_for(timeout=90000)
    page.wait_for_timeout(1000)
    page.screenshot(path=str(assets / "baseline_streamlit.png"), full_page=True)

    # 401 응답 캡처 (API 직접 호출 결과를 이미지로 저장)
    import json
    import requests
    from PIL import Image, ImageDraw, ImageFont

    r401 = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"text": "테스트"},
        timeout=10,
    )
    lines = [
        "POST /predict (API Key 없음)",
        f"HTTP {r401.status_code}",
        json.dumps(r401.json(), ensure_ascii=False, indent=2),
    ]
    img = Image.new("RGB", (900, 320), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = 20
    for line in lines:
        draw.text((20, y), line, fill=(0, 0, 0))
        y += 28
    img.save(assets / "baseline_auth_401.png")


def _indoor_chat(
    page,
    *,
    model_key: str = "3B",
    api_key: str = "test-key-001",
    prompt: str,
    wait_text: str,
    out_path: Path,
    timeout_ms: int = 300000,
) -> None:
    page.goto("http://127.0.0.1:8502", wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(2500)

    api = page.get_by_label("API Key")
    if api.count():
        api.click()
        page.keyboard.press("Meta+A")
        page.keyboard.type(api_key)

    # 모델 선택 (sidebar selectbox)
    sidebar = page.locator('[data-testid="stSidebar"]')
    selects = sidebar.locator('[data-baseweb="select"]')
    if selects.count():
        selects.first.click()
        page.get_by_role("option", name=model_key, exact=True).click()
        page.wait_for_timeout(800)

    chat = page.get_by_placeholder("실내 추천을 물어보세요...")
    chat.click()
    chat.fill(prompt)
    page.keyboard.press("Enter")
    page.get_by_text(wait_text, exact=False).wait_for(timeout=timeout_ms)
    page.wait_for_timeout(1500)
    page.screenshot(path=str(out_path), full_page=True)


def capture_indoor(page, assets: Path) -> None:
    assets.mkdir(parents=True, exist_ok=True)

    date_q = "주말에 서울에서 실내 데이트 코스 2개만 추천해줘"

    # 기본 실내 (3B)
    _indoor_chat(
        page,
        model_key="3B",
        prompt=date_q,
        wait_text="목록매칭",
        out_path=assets / "scenario_basic_indoor.png",
    )

    # 1.5B vs 3B
    _indoor_chat(
        page,
        model_key="1.5B",
        prompt=date_q,
        wait_text="model=1.5B",
        out_path=assets / "compare_1.5B_date.png",
    )
    _indoor_chat(
        page,
        model_key="3B",
        prompt=date_q,
        wait_text="model=3B",
        out_path=assets / "compare_3B_date.png",
    )

    # 야외 유도 → 실내
    _indoor_chat(
        page,
        model_key="3B",
        prompt="한강 산책하고 싶은데, 비슷한 분위기의 실내 대안 있어?",
        wait_text="목록매칭",
        out_path=assets / "scenario_outdoor_to_indoor.png",
    )

    # 인증 실패
    page.goto("http://127.0.0.1:8502", wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(2000)
    api = page.get_by_label("API Key")
    if api.count():
        api.click()
        page.keyboard.press("Meta+A")
        page.keyboard.type("wrong-key")
    chat = page.get_by_placeholder("실내 추천을 물어보세요...")
    chat.fill(date_q)
    page.keyboard.press("Enter")
    page.get_by_text("인증 실패", exact=False).wait_for(timeout=30000)
    page.wait_for_timeout(800)
    page.screenshot(path=str(assets / "scenario_auth_fail.png"), full_page=True)

    # RAG 데모
    _indoor_chat(
        page,
        model_key="3B",
        prompt=date_q,
        wait_text="RAG(embedding)",
        out_path=assets / "scenario_rag_demo.png",
    )


def main() -> None:
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    for port in (8000, 8501, 8001, 8502):
        if not wait_port(port, timeout=45):
            raise RuntimeError(f"port {port} not ready — 서버를 먼저 띄우세요.")

    BASELINE_ASSETS.mkdir(parents=True, exist_ok=True)
    INDOOR_ASSETS.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})

        print("=== baseline captures ===")
        if mode in ("all", "baseline"):
            capture_baseline(page, BASELINE_ASSETS)

        print("=== indoor captures (LLM — 수 분 소요) ===")
        if mode in ("all", "indoor"):
            capture_indoor(page, INDOOR_ASSETS)

        browser.close()

    print("done:")
    for d in (BASELINE_ASSETS, INDOOR_ASSETS):
        for f in sorted(d.glob("*.png")):
            print(f"  {f}")


if __name__ == "__main__":
    main()
