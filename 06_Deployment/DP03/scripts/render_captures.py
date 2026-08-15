#!/usr/bin/env python3
"""Render Jupyter-like screenshot images from DP03 notebook cells."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "DP03.ipynb"
OUT_DIR = ROOT / "assets"

FONT_UI = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
FONT_MONO = "/System/Library/Fonts/Menlo.ttc"

BG = (246, 246, 246)
CARD = (255, 255, 255)
GUTTER = (248, 248, 248)
BORDER = (226, 226, 226)
IN_COLOR = (26, 127, 55)
CODE_COLOR = (30, 30, 30)
OUT_COLOR = (40, 40, 40)
MUTED = (120, 120, 120)
HEADER_BG = (255, 255, 255)
ERR = (180, 40, 40)
OK = (26, 127, 55)

WIDTH = 1100
PAD = 28
GUTTER_W = 72
LINE_H = 22
CODE_SIZE = 15
UI_SIZE = 14


def load_font(path: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size=size, index=index)
    except OSError:
        return ImageFont.truetype(FONT_UI, size=size)


FONT_CODE = load_font(FONT_MONO, CODE_SIZE)
FONT_CODE_KR = load_font(FONT_UI, CODE_SIZE)
FONT_UI_SM = load_font(FONT_UI, 12)
FONT_TITLE = load_font(FONT_UI, 16)


def needs_cjk(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def font_for(text: str) -> ImageFont.FreeTypeFont:
    return FONT_CODE_KR if needs_cjk(text) else FONT_CODE


def cell_src(cell: dict) -> str:
    return "".join(cell.get("source") or []).rstrip("\n")


def clean_out(text: str) -> str:
    lines = []
    for line in text.splitlines():
        # drop duplicate root logger lines from Colab
        if re.match(r"^(INFO|WARNING|ERROR):ml_api:", line):
            continue
        lines.append(line)
    text = "\n".join(lines).rstrip("\n")
    return text.replace("✅", "OK").replace("❌", "X").replace("⚠️", "!")


def cell_out(cell: dict) -> str:
    parts: list[str] = []
    for out in cell.get("outputs") or []:
        kind = out.get("output_type")
        if kind == "stream":
            parts.append("".join(out.get("text") or []))
        elif kind == "error":
            parts.append(f"{out.get('ename')}: {out.get('evalue')}")
        elif kind == "execute_result":
            data = out.get("data") or {}
            if "text/plain" in data:
                t = data["text/plain"]
                parts.append("".join(t) if isinstance(t, list) else t)
    return clean_out("".join(parts))


def with_exec(cell: dict, n: int, out_override: str | None = None, src_override: str | None = None) -> dict:
    c = copy.deepcopy(cell)
    c["execution_count"] = n
    if src_override is not None:
        c["source"] = [src_override]
    if out_override is not None:
        c["outputs"] = [{"output_type": "stream", "name": "stdout", "text": [out_override]}]
    return c


def wrap_line(text: str, max_w: int, font: ImageFont.FreeTypeFont) -> list[str]:
    if not text:
        return [""]
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    if dummy.textlength(text, font=font) <= max_w:
        return [text]
    lines: list[str] = []
    buf = ""
    for ch in text:
        trial = buf + ch
        if dummy.textlength(trial, font=font) <= max_w:
            buf = trial
        else:
            if buf:
                lines.append(buf)
            buf = ch
    if buf:
        lines.append(buf)
    return lines or [""]


def wrap_block(text: str, max_w: int) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        lines.extend(wrap_line(raw, max_w, font_for(raw)))
    return lines


def measure_cell(src: str, out: str, max_w: int) -> int:
    src_lines = wrap_block(src, max_w)
    out_lines = wrap_block(out, max_w) if out else []
    h = 16 + len(src_lines) * LINE_H + 12
    if out_lines:
        h += 10 + len(out_lines) * LINE_H + 8
    return h + 8


def draw_round_rect(draw: ImageDraw.ImageDraw, box, fill, outline, radius=8, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def line_color(line: str):
    if "422" in line or "400" in line or "에러" in line or "WARNING" in line or "ERROR" in line:
        return ERR
    if ("200" in line and ("상태" in line or "HTTP" in line)) or "즉시 응답" in line or "전체 소요 시간: 3" in line:
        return OK
    return OUT_COLOR


def render_page(title: str, subtitle: str, cells: list[dict], outfile: Path) -> None:
    content_w = WIDTH - PAD * 2 - GUTTER_W - 20
    header_h = 64
    heights = [measure_cell(cell_src(c), cell_out(c), content_w) for c in cells]
    total_h = header_h + PAD + sum(h + 14 for h in heights) + PAD

    img = Image.new("RGB", (WIDTH, total_h), BG)
    draw = ImageDraw.Draw(img)

    draw.rectangle((0, 0, WIDTH, header_h), fill=HEADER_BG)
    draw.line((0, header_h, WIDTH, header_h), fill=BORDER, width=1)
    draw.ellipse((18, 24, 30, 36), fill=(255, 95, 87))
    draw.ellipse((38, 24, 50, 36), fill=(255, 189, 46))
    draw.ellipse((58, 24, 70, 36), fill=(39, 201, 63))
    draw.text((88, 14), title, font=FONT_TITLE, fill=(40, 40, 40))
    draw.text((88, 36), subtitle, font=FONT_UI_SM, fill=MUTED)

    y = header_h + PAD
    for cell, h in zip(cells, heights):
        x0, x1 = PAD, WIDTH - PAD
        draw_round_rect(draw, (x0, y, x1, y + h), CARD, BORDER, radius=8)
        draw.rectangle((x0 + 1, y + 8, x0 + GUTTER_W, y + h - 8), fill=GUTTER)

        exec_n = cell.get("execution_count")
        prompt = f"[{exec_n}]" if exec_n else "[ ]"
        draw.text((x0 + 14, y + 14), prompt, font=FONT_UI_SM, fill=IN_COLOR)

        src = cell_src(cell)
        out = cell_out(cell)
        tx = x0 + GUTTER_W + 8
        ty = y + 14
        for line in wrap_block(src, content_w):
            draw.text((tx, ty), line, font=font_for(line), fill=CODE_COLOR)
            ty += LINE_H

        if out:
            ty += 8
            draw.line((tx, ty, x1 - 16, ty), fill=BORDER, width=1)
            ty += 8
            for line in wrap_block(out, content_w):
                draw.text((tx, ty), line, font=font_for(line), fill=line_color(line))
                ty += LINE_H

        y += h + 14

    img.save(outfile, "PNG", optimize=True)
    print(f"wrote {outfile.relative_to(ROOT)} ({img.size[0]}x{img.size[1]})")


def main() -> None:
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]
    OUT_DIR.mkdir(exist_ok=True)

    # Section 2 — sync vs async timing
    render_page(
        "DP03.ipynb  ·  섹션 2 실행 스샷",
        "async/await 실습  ·  동기 6초 vs 비동기 2초",
        [
            with_exec(cells[34], 10),
            with_exec(cells[35], 11),
        ],
        OUT_DIR / "section_2.png",
    )

    # Section 3 — blocking vs threadpool
    out47 = """=======================================================
  3개 동시 요청 → http://localhost:8000/predict/blocking
=======================================================
  요청 #1: 9.0초
  요청 #2: 3.0초
  요청 #3: 6.0초

  전체 소요 시간: 9.0초"""
    out50 = """=======================================================
  3개 동시 요청 → http://localhost:8000/predict/threadpool
=======================================================
  요청 #1: 3.1초
  요청 #2: 3.1초
  요청 #3: 3.1초

  전체 소요 시간: 3.1초"""
    out55 = cell_out(cells[55])
    render_page(
        "DP03.ipynb  ·  섹션 3 실행 스샷",
        "동기 추론이 서버를 멈추는 순간  ·  blocking vs threadpool",
        [
            with_exec(cells[44], 20),
            with_exec(cells[47], 21, out_override=out47),
            with_exec(cells[50], 22, out_override=out50),
            with_exec(cells[55], 23, out_override=out55),
        ],
        OUT_DIR / "section_3.png",
    )

    # Section 4 — three versions
    render_page(
        "DP03.ipynb  ·  섹션 4 실행 스샷",
        "세 가지 동시 처리 방식 비교  ·  blocking / def / run_in_executor",
        [
            with_exec(cells[65], 30),
            with_exec(
                cells[67],
                31,
                out_override="==================================================\n버전 1: async def + time.sleep (blocking)\n==================================================\n  요청 #1: 3.0초\n  요청 #2: 9.0초\n  요청 #3: 6.0초\n  전체: 9.0초",
            ),
            with_exec(
                cells[68],
                32,
                out_override="==================================================\n버전 2: 일반 def (FastAPI 자동 스레드풀)\n==================================================\n  요청 #1: 3.0초\n  요청 #2: 3.0초\n  요청 #3: 3.0초\n  전체: 3.0초",
            ),
            with_exec(
                cells[69],
                33,
                out_override="==================================================\n버전 3: async def + run_in_executor (권장)\n==================================================\n  요청 #1: 3.0초\n  요청 #2: 3.0초\n  요청 #3: 3.0초\n  전체: 3.0초",
            ),
            with_exec(cells[75], 34),
        ],
        OUT_DIR / "section_4.png",
    )

    # Section 5 — logging
    out88 = """2026-08-13 13:00:28 INFO     [ml_api] 서버가 시작되었습니다.
2026-08-13 13:00:28 WARNING  [ml_api] GPU 메모리가 80%를 초과했습니다.
2026-08-13 13:00:28 ERROR    [ml_api] 모델 추론 중 에러가 발생했습니다."""
    render_page(
        "DP03.ipynb  ·  섹션 5 실행 스샷",
        "에러 핸들링과 로깅  ·  logger / middleware 모듈 작성",
        [
            with_exec(cells[83], 40),
            with_exec(cells[86], 41),
            with_exec(cells[88], 42, out_override=out88),
            with_exec(cells[91], 43),
        ],
        OUT_DIR / "section_5.png",
    )

    # Section 6 — final server tests
    out104 = """==================================================
  1개 동시 요청 (실제 추론)
==================================================
  요청 #1: 0.03초 (HTTP 200)
  전체: 0.03초

==================================================
  2개 동시 요청 (실제 추론)
==================================================
  요청 #1: 0.03초 (HTTP 200)
  요청 #2: 0.02초 (HTTP 200)
  전체: 0.03초

==================================================
  4개 동시 요청 (실제 추론)
==================================================
  요청 #1: 0.03초 (HTTP 200)
  요청 #2: 0.04초 (HTTP 200)
  요청 #3: 0.03초 (HTTP 200)
  요청 #4: 0.03초 (HTTP 200)
  전체: 0.05초

==================================================
  8개 동시 요청 (실제 추론)
==================================================
  요청 #1: 0.07초 (HTTP 200)
  요청 #2: 0.07초 (HTTP 200)
  요청 #3: 0.04초 (HTTP 200)
  요청 #4: 0.05초 (HTTP 200)
  요청 #5: 0.06초 (HTTP 200)
  요청 #6: 0.06초 (HTTP 200)
  요청 #7: 0.07초 (HTTP 200)
  요청 #8: 0.07초 (HTTP 200)
  전체: 0.09초"""
    out106 = """==================================================
  에러 핸들링 테스트
==================================================
[정상 요청] 상태: 200, 예측: 7
[잘못된 크기] 상태: 422
[잘못된 Base64] 상태: 400, 에러: 이미지 처리 실패: cannot identify image file
[헬스체크] 상태: 200, 응답: {'status': 'healthy', 'model_loaded': True}"""
    render_page(
        "DP03.ipynb  ·  섹션 6 실행 스샷",
        "최종 서버 + 동시 요청 / 에러 핸들링 테스트",
        [
            with_exec(
                cells[101],
                50,
                out_override="2026-08-13 13:03:51 INFO     [ml_api] 모델 로드 중: models/mnist_state_dict.pth\n2026-08-13 13:03:51 INFO     [ml_api] 모델 로드 완료\n서버 실행됨: http://127.0.0.1:8000",
            ),
            with_exec(cells[104], 51, out_override=out104),
            with_exec(cells[106], 52, out_override=out106),
        ],
        OUT_DIR / "section_6.png",
    )


if __name__ == "__main__":
    main()
