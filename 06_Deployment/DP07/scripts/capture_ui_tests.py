#!/usr/bin/env python3
"""Automate DP07 Streamlit UI tests A–D and save screenshots."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
URL = "http://127.0.0.1:8501"


def log(msg: str) -> None:
    print(msg, flush=True)


def wait_healthy(page, timeout_ms: int = 60000) -> None:
    log(f"goto {URL}")
    page.goto(URL, wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_selector("text=한국어 챗봇", timeout=timeout_ms)
    page.wait_for_selector('input[type="password"]', timeout=timeout_ms)
    page.wait_for_selector('[data-testid="stChatInputTextArea"]', timeout=timeout_ms)
    log("page ready")


def set_api_key(page, key: str) -> None:
    box = page.locator('input[type="password"]').first
    box.click()
    box.fill(key)
    page.locator("text=한국어 챗봇").click()
    page.wait_for_timeout(800)
    log(f"api_key set ({key})")


def click_slider_ratio(page, slider_index: int, ratio: float) -> None:
    slider = page.locator('[data-testid="stSidebar"] [data-testid="stSlider"]').nth(slider_index)
    track = slider.bounding_box()
    if not track:
        log(f"slider {slider_index} missing")
        return
    ratio = max(0.02, min(0.98, ratio))
    page.mouse.click(track["x"] + track["width"] * ratio, track["y"] + track["height"] / 2)
    page.wait_for_timeout(700)


def chat_count(page) -> int:
    return page.locator('[data-testid="stChatMessage"]').count()


def wait_generation_done(page, min_messages: int, timeout_s: int = 180) -> None:
    """Wait until enough chat bubbles exist and spinner text is gone."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        n = chat_count(page)
        spinning = page.locator("text=생성 중").count() > 0
        auth_err = page.locator("text=인증 실패").count() > 0
        gen_fail = page.locator("text=응답을 생성하지 못했습니다").count() > 0
        if auth_err or gen_fail:
            log(f"done with error banner (messages={n})")
            return
        if n >= min_messages and not spinning:
            # ensure assistant text is not only spinner remnant
            last = page.locator('[data-testid="stChatMessage"]').last.inner_text().strip()
            if last and "생성 중" not in last:
                log(f"generation done (messages={n})")
                return
        time.sleep(0.5)
    raise TimeoutError(f"generation timeout (messages={chat_count(page)})")


def send_chat(page, text: str, expect_error: bool = False, timeout_s: int = 180) -> None:
    before = chat_count(page)
    chat = page.locator('[data-testid="stChatInputTextArea"]')
    chat.click()
    chat.fill(text)
    chat.press("Enter")
    log(f"sent: {text!r} (before={before})")
    if expect_error:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if page.locator("text=인증 실패").count() > 0:
                log("saw auth failure")
                page.wait_for_timeout(500)
                return
            time.sleep(0.4)
        raise TimeoutError("auth failure banner not found")
    wait_generation_done(page, min_messages=before + 2, timeout_s=timeout_s)
    page.wait_for_timeout(600)


def clear_chat(page) -> None:
    page.locator('[data-testid="stSidebar"] button', has_text="대화 초기화").click()
    page.wait_for_timeout(2000)
    log(f"cleared (messages={chat_count(page)})")


def shot(page, name: str) -> None:
    path = ASSETS / name
    page.screenshot(path=str(path), full_page=True)
    log(f"saved {path}")


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        wait_healthy(page)

        set_api_key(page, "test-key-001")
        click_slider_ratio(page, 0, 0.08)

        send_chat(page, "안녕하세요!")
        send_chat(page, "오늘 뭐 하면 좋을까?")
        shot(page, "ui_test_a_chat.png")

        clear_chat(page)
        click_slider_ratio(page, 1, 0.0)
        send_chat(page, "인공지능을 한 문장으로 설명해줘.")
        shot(page, "ui_test_b_temp_low.png")
        clear_chat(page)
        click_slider_ratio(page, 1, 0.78)
        send_chat(page, "인공지능을 한 문장으로 설명해줘.")
        shot(page, "ui_test_b_temp_high.png")

        send_chat(page, "이 메시지는 곧 지워질 거야.")
        shot(page, "ui_test_c_before_clear.png")
        clear_chat(page)
        shot(page, "ui_test_c_after_clear.png")

        set_api_key(page, "wrong-key")
        send_chat(page, "안녕하세요", expect_error=True)
        shot(page, "ui_test_d_auth_fail.png")

        browser.close()
    log("UI capture done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
