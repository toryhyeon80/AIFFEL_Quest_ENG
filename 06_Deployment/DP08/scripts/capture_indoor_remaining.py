#!/usr/bin/env python3
"""Indoor 남은 캡처만 실행."""
from pathlib import Path
import socket
import time

from playwright.sync_api import sync_playwright

ASSETS = Path(__file__).resolve().parents[1] / "indoor" / "assets"
DATE_Q = "주말에 서울에서 실내 데이트 코스 2개만 추천해줘"


def wait_port(port: int) -> None:
    for _ in range(60):
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise RuntimeError(f"port {port} down")


def chat(page, prompt, wait_for, out, model="3B", api_key="test-key-001"):
    page.goto("http://127.0.0.1:8502", wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(2500)
    page.get_by_label("API Key").fill(api_key)
    sidebar = page.locator('[data-testid="stSidebar"]')
    sel = sidebar.locator('[data-baseweb="select"]')
    if sel.count():
        sel.first.click()
        page.get_by_role("option", name=model, exact=True).click()
        page.wait_for_timeout(1000)
    box = page.get_by_placeholder("실내 추천을 물어보세요...")
    box.fill(prompt)
    page.keyboard.press("Enter")
    for text in (wait_for if isinstance(wait_for, list) else [wait_for]):
        try:
            page.get_by_text(text, exact=False).wait_for(timeout=300000)
            break
        except Exception:
            continue
    else:
        page.wait_for_timeout(5000)
    page.wait_for_timeout(1500)
    page.screenshot(path=str(out), full_page=True)
    print("saved", out)


def main():
    wait_port(8502)
    ASSETS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        page = p.chromium.launch(headless=True).new_page(viewport={"width": 1400, "height": 900})

        chat(page, DATE_Q, ["목록매칭", "model=3B"], ASSETS / "compare_3B_date.png", model="3B")
        chat(
            page,
            "한강 산책하고 싶은데, 비슷한 분위기의 실내 대안 있어?",
            ["가드레일", "목록매칭", "실내"],
            ASSETS / "scenario_outdoor_to_indoor.png",
            model="3B",
        )
        chat(page, DATE_Q, ["RAG(embedding)", "RAG", "목록매칭"], ASSETS / "scenario_rag_demo.png", model="3B")

        page.goto("http://127.0.0.1:8502", wait_until="domcontentloaded")
        page.wait_for_timeout(2000)
        page.get_by_label("API Key").fill("wrong-key")
        page.get_by_placeholder("실내 추천을 물어보세요...").fill(DATE_Q)
        page.keyboard.press("Enter")
        try:
            page.get_by_text("인증 실패", exact=False).wait_for(timeout=30000)
        except Exception:
            pass
        page.screenshot(path=str(ASSETS / "scenario_auth_fail.png"), full_page=True)
        print("saved", ASSETS / "scenario_auth_fail.png")


if __name__ == "__main__":
    main()
