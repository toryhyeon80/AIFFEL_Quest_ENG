#!/usr/bin/env python3
"""Patch DP04.ipynb for local/Mac compatibility."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "DP04.ipynb"


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
if (cwd / "DP04.ipynb").exists() or (cwd / "app").exists():
    pass
elif (cwd / "DP04" / "DP04.ipynb").exists():
    os.chdir(cwd / "DP04")
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

SETUP_MD = """> **로컬(Mac) 환경 준비**
>
> Colab이 아니라면 아래를 한 번만 준비하면 됩니다.
>
> ```bash
> cd 06_Deployment/DP04
> python3.12 -m venv .venv
> source .venv/bin/activate
> pip install -r requirements.txt
> python -m ipykernel install --user --name dp04-venv --display-name "DP04 .venv"
> ```
>
> Cursor에서 커널을 **DP04 .venv** 로 선택한 뒤, 위 도우미 셀부터 순서대로 실행하세요.
> 포트 8000이 다른 앱에 점유되어 있으면 `serve_in_thread`가 자동으로 정리합니다.
> Streamlit은 노트북 iframe 대신 브라우저(`http://localhost:8501`)에서 여는 것이 안정적입니다.
> MNIST 가중치는 `../DP02/models/mnist_state_dict.pth` 가 있으면 학습 없이 복사합니다.
"""

PIP_CELL = '''# Colab이 아니라면 requirements.txt 설치분이 있으면 건너뜁니다.
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("streamlit") is None:
    print("Streamlit 설치 중...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "streamlit"])
else:
    import streamlit
    print(f"Streamlit 이미 설치됨: {streamlit.__version__}")
'''

STREAMLIT_INSTALL_CELL = '''import os
os.makedirs("frontend", exist_ok=True)

import importlib.util
import subprocess
import sys

if importlib.util.find_spec("streamlit") is None:
    print("Streamlit 미설치 → 지금 설치합니다.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "streamlit"])

import streamlit
print(f"✅ Streamlit: {streamlit.__version__}")
'''

STREAMLIT_CELL = '''# Streamlit은 .venv에 설치되어 있으면 pip를 건너뜁니다.
import importlib.util
import sys
import subprocess
import time
import socket
import contextlib
import tempfile
import os
import webbrowser

try:
    from google.colab import output as _colab_output
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if importlib.util.find_spec("streamlit") is None:
    print("Streamlit 미설치 → 지금 설치합니다.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "streamlit"])
else:
    print("Streamlit 이미 설치됨")

def run_streamlit(script, port=8501):
    """Streamlit을 백그라운드로 띄우고 포트가 열릴 때까지 확인합니다."""
    def port_open(p):
        with contextlib.closing(socket.socket()) as s:
            s.settimeout(0.5)
            return s.connect_ex(("127.0.0.1", p)) == 0

    if port_open(port):
        print(f"♻️  이미 실행 중 (포트 {port})")
        return None

    log_path = os.path.join(tempfile.gettempdir(), f"streamlit_{port}.log")
    log = open(log_path, "w", encoding="utf-8")

    extra = []
    if IN_COLAB:
        extra = [
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
        ]

    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", script,
         "--server.port", str(port),
         "--server.headless", "true",
         *extra],
        stdout=log, stderr=subprocess.STDOUT,
    )
    for _ in range(60):
        if proc.poll() is not None:
            log.close()
            print(f"❌ Streamlit이 종료됨 (code {proc.returncode}) — 로그:")
            print(open(log_path, encoding="utf-8").read()[-2000:])
            return proc
        if port_open(port):
            url = f"http://localhost:{port}"
            print(f"✅ 프론트엔드: {url}")
            print(f"   (로그: {log_path})")
            if not IN_COLAB:
                webbrowser.open(url)
            return proc
        time.sleep(0.25)
    proc.terminate(); log.close()
    print("❌ 15초 내에 포트가 열리지 않음 — 로그:")
    print(open(log_path, encoding="utf-8").read()[-2000:])
    return proc

proc = run_streamlit("frontend/app_dashboard.py", port=8501)
'''

DASHBOARD_CELL = '''def show_dashboard(port=8501, height=900):
    """실행 중인 프론트엔드를 엽니다. Colab은 iframe, 로컬은 브라우저."""
    def port_open(p):
        with contextlib.closing(socket.socket()) as s:
            s.settimeout(0.5)
            return s.connect_ex(("127.0.0.1", p)) == 0

    if not port_open(port):
        print(f"⚠️ 포트 {port}에 프론트엔드가 없습니다. Step 2 셀을 먼저 실행하세요.")
        return

    if IN_COLAB:
        from google.colab import output
        print(f"✅ 대시보드를 아래에 띄웁니다 (Colab 프록시 → 포트 {port})")
        output.serve_kernel_port_as_iframe(port, height=str(height))
    else:
        import webbrowser
        url = f"http://localhost:{port}"
        print(f"✅ 브라우저에서 엽니다: {url}")
        webbrowser.open(url)

show_dashboard(port=8501, height=900)
'''

MODEL_COPY_SNIPPET = '''
MODEL_PATH = "models/mnist_state_dict.pth"
CANDIDATES = [
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

    next_src = "".join(nb["cells"][helper_idx + 1].get("source") or [])
    if "로컬(Mac) 환경 준비" not in next_src:
        nb["cells"].insert(
            helper_idx + 1,
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in SETUP_MD.splitlines()],
            },
        )

    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source") or [])
        if c["cell_type"] == "code" and "!pip list | grep streamlit" in src:
            set_source(nb, i, PIP_CELL)
        if c["cell_type"] == "code" and 'os.makedirs("frontend"' in src and "import streamlit" in src:
            set_source(nb, i, STREAMLIT_INSTALL_CELL)
        if c["cell_type"] == "code" and "run_streamlit" in src and "frontend/app_dashboard.py" in src:
            set_source(nb, i, STREAMLIT_CELL)
        if c["cell_type"] == "code" and "def show_dashboard" in src:
            set_source(nb, i, DASHBOARD_CELL)
        if c["cell_type"] == "code" and "BACKEND_FILES" in src and "app/main_final.py" in src:
            src = src.replace("asyncio.get_event_loop()", "asyncio.get_running_loop()")
            src = src.replace("Colab 세션이 새로 뜬 경우에 필요합니다", "로컬에서 DP04만 연 경우에 필요합니다")
            old = (
                'MODEL_PATH = "models/mnist_state_dict.pth"\n'
                "if os.path.exists(MODEL_PATH):\n"
                '    print(f"✅ {MODEL_PATH} 있음")\n'
                "else:\n"
            )
            if old in src:
                src = src.replace(old, MODEL_COPY_SNIPPET.lstrip("\n") + "        ")
                # the training block needs extra indent because we nested under `if not copied`
                src = src.replace(
                    "        print(\"⚠️ 모델 파일 없음 → MNIST 모델을 학습합니다. (약 1~2분 소요)\")\n",
                    "        print(\"⚠️ 모델 파일 없음 → MNIST 모델을 학습합니다. (약 1~2분 소요)\")\n",
                )
            set_source(nb, i, src)
        if c["cell_type"] == "markdown" and "Colab 세션이 새로 뜬 경우" in src:
            src = src.replace(
                "Day 4부터 바로 시작했거나, Colab 세션이 새로 뜬 경우",
                "Day 4부터 바로 시작했거나, 로컬에서 DP04만 연 경우",
            )
            set_source(nb, i, src)
        if c["cell_type"] == "markdown" and "노트북 안에 iframe으로 바로 띄웁니다" in src:
            src = (
                "#### Step 3: 노트북에서 바로 확인\n"
                "\n"
                "Colab에서는 대시보드를 노트북 iframe으로 띄웁니다.  \n"
                "로컬(Mac)에서는 `http://localhost:8501` 을 브라우저에서 엽니다.\n"
                "\n"
                "> ⚠️ **화면이 비어 있거나 계속 로딩만 된다면**\n"
                "> - Step 2 셀이 `✅`를 출력했는지 먼저 확인하세요.\n"
                "> - 로컬에서는 Jupyter iframe 대신 브라우저 탭을 사용하세요.\n"
                "> - 대시보드 안의 서버 상태가 🔴이면 Step 1의 백엔드 셀을 다시 실행하세요.\n"
            )
            set_source(nb, i, src)
        if c["cell_type"] == "markdown" and "위 Step 3의 iframe에서 바로 조작합니다" in src:
            src = src.replace(
                "위 Step 3의 iframe에서 바로 조작합니다. (로컬 실행이라면 http://localhost:8501 도 됩니다)",
                "브라우저(http://localhost:8501) 또는 Colab iframe에서 바로 조작합니다.",
            )
            set_source(nb, i, src)
        if c["cell_type"] == "markdown" and "cd model-serving-course" in src:
            src = src.replace("cd model-serving-course", "cd 06_Deployment/DP04")
            set_source(nb, i, src)

    kernelspec = nb.setdefault("metadata", {}).setdefault("kernelspec", {})
    kernelspec.update(
        {
            "display_name": "DP04 .venv",
            "language": "python",
            "name": "dp04-venv",
        }
    )

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"patched {NB_PATH}")


if __name__ == "__main__":
    main()
