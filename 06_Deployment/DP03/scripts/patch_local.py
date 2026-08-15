#!/usr/bin/env python3
"""Patch DP03.ipynb for local/Mac compatibility."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "DP03.ipynb"


def set_source(nb: dict, cell_idx: int, text: str) -> None:
    lines = [line + "\n" for line in text.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    nb["cells"][cell_idx]["source"] = lines


HELPER = r'''# 서버 실행 도우미 — 노트북 맨 처음에 한 번 실행하세요.
# 노트북 안에서 uvicorn 서버를 띄우고 멈추는 함수를 정의합니다.
# Colab의 nest_asyncio 방식 대신, macOS에서도 안정적인 표준 asyncio 루프를 사용합니다.
import os, sys, asyncio, threading, time, socket, contextlib, subprocess, signal
from pathlib import Path
import uvicorn

# 작업 디렉터리를 DP03 루트(app/ 가 생길 위치)로 맞춥니다.
cwd = Path.cwd()
if (cwd / "DP03.ipynb").exists() or (cwd / "app").exists():
    pass
elif (cwd / "DP03" / "DP03.ipynb").exists():
    os.chdir(cwd / "DP03")
elif cwd.name == "notebooks" and (cwd.parent / "app").exists():
    os.chdir(cwd.parent)

for _d in ("app", "models", "data", "frontend"):
    os.makedirs(_d, exist_ok=True)

_SERVERS = {}  # port -> (server, thread)

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
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
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
        sys.modules.pop(app.split(":")[0], None)   # 파일을 다시 저장한 경우 최신 내용 반영
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
        # Windows는 SelectorEventLoop, 그 외(macOS 포함)는 기본 이벤트 루프를 사용합니다.
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

print("서버 도우미 준비 완료 (serve_in_thread, stop_server)")
print(f"작업 디렉터리: {Path.cwd()}")
'''

SETUP_MD_LINES = [
    "> **로컬(Mac) 환경 준비**\n",
    ">\n",
    "> Colab이 아니라면 아래를 한 번만 준비하면 됩니다.\n",
    ">\n",
    "> ```bash\n",
    "> cd 06_Deployment/DP03\n",
    "> python3 -m venv .venv\n",
    "> source .venv/bin/activate\n",
    "> pip install -r requirements.txt\n",
    "> ```\n",
    ">\n",
    "> Cursor/VS Code에서 커널을 `.venv (Python 3.x)` 로 선택한 뒤, 위 도우미 셀부터 순서대로 실행하세요.\n",
    "> 포트 8000이 다른 앱에 점유되어 있으면 `serve_in_thread`가 자동으로 정리합니다.\n",
]

DEPS_CELL = Path(__file__).with_name("dp03_deps_cell.py").read_text(encoding="utf-8")


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    set_source(nb, 1, HELPER)

    src2 = "".join(nb["cells"][2].get("source") or [])
    if "로컬(Mac) 환경 준비" not in src2:
        nb["cells"].insert(
            2,
            {"cell_type": "markdown", "metadata": {}, "source": SETUP_MD_LINES},
        )
        offset = 1
    else:
        offset = 0

    def idx(i: int) -> int:
        return i + offset if i >= 2 else i

    # markdown example
    c61 = idx(61)
    src61 = "".join(nb["cells"][c61]["source"])
    src61 = src61.replace(
        "loop = asyncio.get_event_loop()",
        "loop = asyncio.get_running_loop()",
    )
    set_source(nb, c61, src61)

    for orig in (63, 71, 97):
        i = idx(orig)
        src = "".join(nb["cells"][i]["source"])
        src = src.replace("asyncio.get_event_loop()", "asyncio.get_running_loop()")
        set_source(nb, i, src)

    set_source(nb, idx(95), DEPS_CELL)

    c94 = idx(94)
    src94 = "".join(nb["cells"][c94]["source"])
    if "DP02" not in src94:
        src94 = (
            src94.rstrip()
            + "\n\n> **로컬(Mac)**: `../DP02/models/mnist_state_dict.pth` 가 있으면 학습 없이 복사합니다.\n"
        )
        set_source(nb, c94, src94)

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    # verify
    nb2 = json.loads(NB_PATH.read_text(encoding="utf-8"))
    helper = "".join(nb2["cells"][1]["source"])
    assert "_pids_on_port" in helper
    deps = "".join(nb2["cells"][idx(95)]["source"])
    assert 'if not os.path.exists("app/model_utils.py")' in deps
    assert "DP02_MODEL" in deps
    assert "if True:" not in deps
    for orig in (63, 71, 97):
        s = "".join(nb2["cells"][idx(orig)]["source"])
        assert "get_event_loop" not in s
        assert "get_running_loop" in s
    print(f"patched {NB_PATH} cells={len(nb2['cells'])} offset={offset}")


if __name__ == "__main__":
    main()
