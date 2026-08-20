#!/usr/bin/env python3
"""Patch DP06.ipynb for local/Mac compatibility."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "DP06.ipynb"


def set_source(nb: dict, cell_idx: int, text: str) -> None:
    lines = [line + "\n" for line in text.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    nb["cells"][cell_idx]["source"] = lines


HELPER = r'''# 서버 실행 도우미 — 노트북 맨 처음에 한 번 실행하세요.
# 노트북 안에서 uvicorn 서버를 띄우고 멈추는 함수를 정의합니다.
# Colab의 nest_asyncio 방식 대신, macOS에서도 안정적인 표준 asyncio 루프를 사용합니다.
print("도우미 로딩 중...", flush=True)

import os, sys, asyncio, threading, time, socket, contextlib, subprocess, signal
from pathlib import Path

print("uvicorn import 중...", flush=True)
import uvicorn
print("uvicorn 준비 완료", flush=True)

cwd = Path.cwd()
if (cwd / "DP06.ipynb").exists() or (cwd / "app").exists():
    pass
elif (cwd / "DP06" / "DP06.ipynb").exists():
    os.chdir(cwd / "DP06")
elif cwd.name == "notebooks" and (cwd.parent / "app").exists():
    os.chdir(cwd.parent)

for _d in ("app", "models", "data", "frontend"):
    os.makedirs(_d, exist_ok=True)

_SERVERS = {}

def _port_open(host, port):
    with contextlib.closing(socket.socket()) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def _pids_on_port(port):
    """해당 포트를 LISTEN 중인 프로세스 PID 목록을 반환합니다."""
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

def stop_server(port=8000):
    """실행 중인 서버를 멈춥니다. 같은 포트의 다른 프로세스도 종료합니다."""
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
    for _ in range(50):
        if not _port_open("127.0.0.1", port):
            break
        time.sleep(0.1)

def serve_in_thread(app, host="127.0.0.1", port=8000, log_level="warning"):
    """백그라운드에서 uvicorn 서버를 띄웁니다.

    app: FastAPI 객체 또는 'app.main:app' 같은 import 경로.
    같은 포트에 서버가 이미 있으면 먼저 멈추고 새로 띄웁니다.
    """
    stop_server(port)
    if isinstance(app, str):
        sys.modules.pop(app.split(":")[0], None)
    for _ in range(50):
        if not _port_open(host, port):
            break
        time.sleep(0.1)
    if _port_open(host, port):
        print(f"포트 {port}가 아직 사용 중입니다. 다른 서버를 종료한 뒤 다시 실행하세요.")
        return None
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level, loop="asyncio")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    def _run():
        if sys.platform == "win32":
            loop = asyncio.SelectorEventLoop()
        else:
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    _SERVERS[port] = (server, thread)
    for _ in range(40):
        if getattr(server, "started", False) and _port_open(host, port):
            print(f"서버 실행됨: http://{host}:{port}")
            return server
        if not thread.is_alive():
            break
        time.sleep(0.25)
    print("서버가 시작되지 않았습니다. 위 로그를 확인하세요.")
    return server

print("서버 도우미 준비 완료 (serve_in_thread, stop_server)", flush=True)
print(f"작업 디렉터리: {Path.cwd()}", flush=True)
'''

SWAGGER_CELL = '''# Swagger UI 열기 — Colab은 iframe, 로컬(Mac)은 브라우저 탭
try:
    from google.colab import output as _colab_output
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

DOCS_URL = "http://localhost:8000/docs"

if IN_COLAB:
    from google.colab import output
    print("✅ Swagger UI를 아래에 띄웁니다 (Colab 프록시 → 포트 8000)")
    output.serve_kernel_port_as_iframe(8000, path="/docs", height="800")
else:
    import webbrowser
    print(f"✅ 브라우저에서 엽니다: {DOCS_URL}")
    print("   (Jupyter iframe은 localhost X-Frame 제한으로 비어 보일 수 있습니다)")
    webbrowser.open(DOCS_URL)
'''

MODEL_COPY_SNIPPET = '''
MODEL_PATH = "models/mnist_state_dict.pth"
CANDIDATES = [
    "../DP04/models/mnist_state_dict.pth",
    "../DP03/models/mnist_state_dict.pth",
    "../DP02/models/mnist_state_dict.pth",
]
if os.path.exists(MODEL_PATH):
    print(f"✅ {MODEL_PATH} 있음")
else:
    copied = False
    import shutil
    for src in CANDIDATES:
        if os.path.exists(src):
            shutil.copy2(src, MODEL_PATH)
            print(f"✅ 모델을 복사했습니다: {src} → {MODEL_PATH}")
            copied = True
            break
    if not copied:
'''


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    helper_idx = next(
        i for i, c in enumerate(nb["cells"])
        if c["cell_type"] == "code"
        and "serve_in_thread" in "".join(c.get("source") or [])
        and "서버 실행 도우미" in "".join(c.get("source") or [])
    )
    set_source(nb, helper_idx, HELPER)

    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source") or [])
        if c["cell_type"] == "code" and "Swagger UI를 노트북 안에서 열기" in src:
            set_source(nb, i, SWAGGER_CELL)
        if c["cell_type"] == "code" and "BACKEND_FILES" in src and "app/model_utils.py" in src:
            src = src.replace(
                "Day 6부터 바로 시작했거나, Colab 세션이 새로 뜬 경우에 필요합니다",
                "로컬에서 DP06만 연 경우에 필요합니다",
            )
            old = (
                'MODEL_PATH = "models/mnist_state_dict.pth"\n'
                "if os.path.exists(MODEL_PATH):\n"
                '    print(f"✅ {MODEL_PATH} 있음")\n'
                "else:\n"
            )
            if old in src:
                src = src.replace(old, MODEL_COPY_SNIPPET.lstrip("\n") + "        ")
            set_source(nb, i, src)
        if c["cell_type"] == "code" and "%%writefile app/image_api.py" in src:
            src = src.replace("asyncio.get_event_loop()", "asyncio.get_running_loop()")
            set_source(nb, i, src)
        if c["cell_type"] == "markdown" and "Day 6부터 바로 시작했거나, Colab 세션이 새로 뜬 경우" in src:
            src = src.replace(
                "Day 6부터 바로 시작했거나, Colab 세션이 새로 뜬 경우",
                "Day 6부터 바로 시작했거나, 로컬에서 DP06만 연 경우",
            )
            set_source(nb, i, src)
        if c["cell_type"] == "markdown" and "6.8 Swagger UI에서 테스트" in src:
            src = src.replace(
                "1. 아래 셀을 실행해 Swagger UI를 띄웁니다 (로컬이라면 http://localhost:8000/docs 도 됩니다)",
                "1. 아래 셀을 실행해 Swagger UI를 띄웁니다 (로컬 Mac은 브라우저에서 http://localhost:8000/docs 가 열립니다)",
            )
            src = src.replace(
                "- Colab은 런타임에 연결된 상태에서만 iframe을 그립니다. 셀을 다시 실행해보세요.",
                "- 로컬 Mac에서는 Jupyter iframe 대신 브라우저 탭을 사용하세요.",
            )
            set_source(nb, i, src)

    kernelspec = nb.setdefault("metadata", {}).setdefault("kernelspec", {})
    kernelspec.update(
        {
            "display_name": "DP06 .venv",
            "language": "python",
            "name": "dp06-venv",
        }
    )

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"patched {NB_PATH}")


if __name__ == "__main__":
    main()
