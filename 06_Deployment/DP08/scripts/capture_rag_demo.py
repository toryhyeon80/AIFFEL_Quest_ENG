"""Capture DP08 RAG demo screenshot via Playwright."""
from pathlib import Path
import time

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "assets" / "scenario_rag_demo.png"
URL = "http://127.0.0.1:8502"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2500)

        # ensure API key
        api = page.get_by_label("API Key")
        if api.count():
            api.fill("test-key-001")

        # RAG checkbox should be on by default; ensure checked
        rag = page.get_by_text("미니 RAG", exact=False)
        if rag.count():
            # click label area if needed - checkbox nearby
            pass

        # click first example prompt button
        btn = page.get_by_role("button", name="주말에 서울에서 실내 데이트 코스 2개만 추천해줘")
        btn.click()
        # wait for assistant response / RAG caption
        page.wait_for_timeout(1000)
        # spinner then response — allow long generation
        page.get_by_text("목록매칭", exact=False).wait_for(timeout=300000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT), full_page=True)
        browser.close()
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
