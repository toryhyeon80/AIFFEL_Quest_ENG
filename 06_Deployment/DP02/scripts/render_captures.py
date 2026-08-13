#!/usr/bin/env python3
"""Render Jupyter-like screenshot images from executed notebook cells."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "DP02.ipynb"
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
ACCENT = (0, 122, 204)
ERR = (180, 40, 40)

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
FONT_UI_REG = load_font(FONT_UI, UI_SIZE)
FONT_UI_SM = load_font(FONT_UI, 12)
FONT_TITLE = load_font(FONT_UI, 16)


def needs_cjk(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def font_for(text: str) -> ImageFont.FreeTypeFont:
    return FONT_CODE_KR if needs_cjk(text) else FONT_CODE


def cell_src(cell: dict) -> str:
    return "".join(cell.get("source") or []).rstrip("\n")


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
    text = "".join(parts).rstrip("\n")
    return text.replace("✅", "OK").replace("❌", "X")


def wrap_line(text: str, max_w: int, font: ImageFont.FreeTypeFont) -> list[str]:
    if not text:
        return [""]
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    if dummy.textlength(text, font=font) <= max_w:
        return [text]
    # wrap by characters for mixed CJK/ASCII
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


def measure_cell(src: str, out: str, exec_n, max_w: int) -> int:
    src_lines = wrap_block(src, max_w)
    out_lines = wrap_block(out, max_w) if out else []
    h = 16 + len(src_lines) * LINE_H + 12
    if out_lines:
        h += 10 + len(out_lines) * LINE_H + 8
    return h + 8


def draw_round_rect(draw: ImageDraw.ImageDraw, box, fill, outline, radius=8, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def render_page(title: str, subtitle: str, cells: list[dict], outfile: Path) -> None:
    content_w = WIDTH - PAD * 2 - GUTTER_W - 20
    header_h = 64
    heights = [measure_cell(cell_src(c), cell_out(c), c.get("execution_count"), content_w) for c in cells]
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
                color = ERR if ("422" in line and "상태" in line) or line.startswith("에러") else OUT_COLOR
                if "200" in line and "상태" in line:
                    color = (26, 127, 55)
                draw.text((tx, ty), line, font=font_for(line), fill=color)
                ty += LINE_H

        y += h + 14

    img.save(outfile, "PNG", optimize=True)
    print(f"wrote {outfile.relative_to(ROOT)} ({img.size[0]}x{img.size[1]})")


def main() -> None:
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]
    OUT_DIR.mkdir(exist_ok=True)

    render_page(
        "DP02.ipynb  ·  섹션 1.5 수행내역",
        "최소한의 FastAPI 서버 실행  ·  .venv (Python 3.12.13)",
        [cells[i] for i in (44, 45, 51, 52)],
        OUT_DIR / "section_1_5.png",
    )
    render_page(
        "DP02.ipynb  ·  섹션 5 수행내역",
        "MNIST 추론 API 서버 기동 및 헬스체크  ·  app.main:app",
        [cells[i] for i in (227, 230, 232)],
        OUT_DIR / "section_5_setup.png",
    )
    render_page(
        "DP02.ipynb  ·  섹션 5 수행내역",
        "테스트 2–3  ·  POST /predict",
        [cells[i] for i in (234, 237)],
        OUT_DIR / "section_5_predict.png",
    )
    render_page(
        "DP02.ipynb  ·  섹션 5 수행내역",
        "테스트 4  ·  MNIST 10장 연속 추론",
        [cells[239]],
        OUT_DIR / "section_5_batch.png",
    )
    render_page(
        "DP02.ipynb  ·  섹션 5 수행내역",
        "에러 상황 테스트  ·  Pydantic 422 검증",
        [cells[i] for i in (242, 245, 247, 249)],
        OUT_DIR / "section_5_errors.png",
    )


if __name__ == "__main__":
    main()
