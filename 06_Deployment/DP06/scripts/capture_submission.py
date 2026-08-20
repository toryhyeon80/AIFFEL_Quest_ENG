#!/usr/bin/env python3
"""Capture DP06 section 6 submission screenshots."""

from __future__ import annotations

import asyncio
import io
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import contextlib
from pathlib import Path

import requests
import uvicorn
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

_SERVERS: dict = {}


def _port_open(host: str, port: int) -> bool:
    with contextlib.closing(socket.socket()) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _pids_on_port(port: int) -> list[int]:
    try:
        out = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []
    pids = []
    for line in out.split():
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)
    return pids


def stop_server(port: int = 8000) -> None:
    entry = _SERVERS.pop(port, None)
    if entry:
        server, thread = entry
        server.should_exit = True
        for _ in range(50):
            if not thread.is_alive():
                break
            time.sleep(0.1)
    for pid in _pids_on_port(port):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def serve_in_thread(app: str, host: str = "127.0.0.1", port: int = 8000):
    stop_server(port)
    for mod in list(sys.modules):
        if mod.startswith("app"):
            del sys.modules[mod]
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", loop="asyncio")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    _SERVERS[port] = (server, thread)
    for _ in range(40):
        if getattr(server, "started", False) and _port_open(host, port):
            return server
        if not thread.is_alive():
            break
        time.sleep(0.25)
    return server


def save_text_image(path: Path, title: str, lines: list[str], width: int = 920) -> None:
    font = ImageFont.load_default()
    line_h = 18
    pad = 16
    height = pad * 2 + line_h * (len(lines) + 1)
    img = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.text((pad, pad), title, fill=(0, 0, 0), font=font)
    y = pad + line_h + 4
    for line in lines:
        draw.text((pad, y), line, fill=(30, 30, 30), font=font)
        y += line_h
    img.save(path)


def capture_swagger(path: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto("http://127.0.0.1:8000/docs", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(path), full_page=False)
        browser.close()


def main() -> None:
    print("Starting server...")
    serve_in_thread("app.image_api:app", port=8000)
    assert _port_open("127.0.0.1", 8000), "Server failed to start"

    save_text_image(
        ASSETS / "section_6_server.png",
        "6.2 서버 실행",
        [
            "serve_in_thread('app.image_api:app', port=8000)",
            "서버 실행됨: http://127.0.0.1:8000",
            f"GET /health → {requests.get('http://127.0.0.1:8000/health').json()}",
        ],
    )

    r1 = requests.post(
        "http://127.0.0.1:8000/predict/image",
        files={"file": ("test.png", b"fake image data", "image/png")},
    )
    save_text_image(
        ASSETS / "section_6_test1_no_auth.png",
        "6.3 테스트 1 — 인증 없이 요청 → 401",
        [f"상태 코드: {r1.status_code}", f"응답: {r1.json()}"],
    )

    r2 = requests.post(
        "http://127.0.0.1:8000/predict/image",
        files={"file": ("test.png", b"fake image data", "image/png")},
        headers={"X-API-Key": "wrong-key"},
    )
    save_text_image(
        ASSETS / "section_6_test2_wrong_key.png",
        "6.4 테스트 2 — 잘못된 키 → 401",
        [f"상태 코드: {r2.status_code}", f"응답: {r2.json()}"],
    )

    dataset = datasets.MNIST(root="data", train=False, download=False)
    img, label = dataset[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    r3 = requests.post(
        "http://127.0.0.1:8000/predict/image",
        files={"file": ("digit.png", buf.getvalue(), "image/png")},
        headers={"X-API-Key": "test-key-001"},
    )
    save_text_image(
        ASSETS / "section_6_test3_success.png",
        "6.5 테스트 3 — 올바른 키 + MNIST 이미지 → 200",
        [
            f"테스트 이미지 정답: {label}",
            f"상태 코드: {r3.status_code}",
            f"예측 결과: {r3.json()}",
        ],
    )

    r4 = requests.post(
        "http://127.0.0.1:8000/predict/image",
        files={"file": ("test.txt", b"this is not an image", "text/plain")},
        headers={"X-API-Key": "test-key-001"},
    )
    save_text_image(
        ASSETS / "section_6_test4_bad_type.png",
        "6.6 테스트 4 — 잘못된 파일 형식 → 400",
        [f"상태 코드: {r4.status_code}", f"응답: {r4.json()}"],
    )

    batch_lines = ["=== 연속 추론 테스트 (5장) ===", ""]
    for i in range(5):
        img_i, truth = dataset[i]
        b = io.BytesIO()
        img_i.save(b, format="PNG")
        resp = requests.post(
            "http://127.0.0.1:8000/predict/image",
            files={"file": (f"digit_{i}.png", b.getvalue(), "image/png")},
            headers={"X-API-Key": "test-key-001"},
        )
        j = resp.json()
        pred = j.get("label", "?")
        conf = j.get("confidence", 0)
        mark = "✅" if str(truth) == str(pred) else "❌"
        batch_lines.append(
            f"  이미지 {i}: 정답={truth}, 예측={pred}, 확신도={conf:.4f} {mark}"
        )
    save_text_image(
        ASSETS / "section_6_test5_batch.png",
        "6.7 테스트 5 — 여러 이미지 연속 테스트",
        batch_lines,
    )

    swagger_path = ASSETS / "section_6_swagger.png"
    print("Capturing Swagger UI...")
    capture_swagger(swagger_path)

    stop_server(8000)
    print(f"Saved assets to {ASSETS}")
    for p in sorted(ASSETS.glob("section_6_*.png")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
