#!/usr/bin/env python3
"""Stepwise Qwen2.5 Instruct size comparison for DP07 (Mac/Colab).

Runs the same prompts on multiple model sizes and writes a markdown table
under assets/model_compare_results.md (and prints to stdout).

Usage:
  cd 06_Deployment/DP07
  .venv/bin/python scripts/compare_models.py
  .venv/bin/python scripts/compare_models.py --models 0.5B 1.5B 3B
  .venv/bin/python scripts/compare_models.py --models 7B   # Colab GPU recommended
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "model_compare_results.md"

MODEL_MAP = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B": "Qwen/Qwen2.5-3B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
}

PROMPTS = [
    {
        "name": "인사",
        "messages": [{"role": "user", "content": "안녕하세요!"}],
    },
    {
        "name": "멀티턴",
        "messages": [
            {"role": "user", "content": "안녕하세요!"},
            {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"},
            {"role": "user", "content": "주말에 서울에서 할 만한 실내 활동을 두 가지만 짧게 추천해줘."},
        ],
    },
    {
        "name": "지시준수",
        "messages": [
            {
                "role": "user",
                "content": "사과를 영어로 한 단어만 답해. 다른 말은 하지 마.",
            }
        ],
    },
]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def input_device_of(model) -> torch.device:
    return next(model.parameters()).device


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 80) -> str:
    chat = [
        {
            "role": "system",
            "content": "너는 친절한 한국어 챗봇이야. 반드시 한국어로, 두세 문장으로 자연스럽게 대답해."
            if messages[-1]["content"] != "사과를 영어로 한 단어만 답해. 다른 말은 하지 마."
            else "지시사항을 정확히 따라.",
        }
    ]
    for msg in messages:
        role = "user" if msg["role"] == "user" else "assistant"
        chat.append({"role": role, "content": msg["content"]})

    encoded = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    )
    if not torch.is_tensor(encoded):
        encoded = encoded["input_ids"]
    input_ids = encoded.to(input_device_of(model))

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True).strip()
    return text or "(empty)"


def run_one(size: str, device: str) -> list[dict]:
    name = MODEL_MAP[size]
    print(f"\n=== Loading {name} on {device} ===", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(name)
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    # Colab/CUDA: device_map="auto"로 바로 GPU에 올려 CPU→GPU 이중 적재를 피합니다.
    # (기존 from_pretrained + .to(cuda)는 로드 중 OOM/강제 종료가 나기 쉽습니다.)
    load_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(name, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, **load_kwargs)
        model = model.to(device)
    model.eval()
    load_s = round(time.time() - t0, 1)
    print(f"loaded in {load_s}s (param device={input_device_of(model)})", flush=True)

    rows = []
    for p in PROMPTS:
        t1 = time.time()
        try:
            resp = generate(model, tokenizer, p["messages"])
            elapsed = round(time.time() - t1, 1)
            status = "ok"
        except Exception as e:  # noqa: BLE001
            resp = f"ERROR: {type(e).__name__}: {e}"
            elapsed = round(time.time() - t1, 1)
            status = "error"
        print(f"[{size}] {p['name']}: {elapsed}s | {resp[:120]}", flush=True)
        rows.append(
            {
                "size": size,
                "model": name,
                "prompt": p["name"],
                "seconds": elapsed,
                "load_seconds": load_s,
                "status": status,
                "response": resp,
            }
        )

    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return rows


def to_markdown(all_rows: list[dict], device: str) -> str:
    lines = [
        "# Qwen2.5 Instruct 크기 비교",
        "",
        f"- device: `{device}`",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 요약",
        "",
        "| 크기 | 모델 | 로드(초) | 인사(초) | 멀티턴(초) | 지시준수(초) |",
        "|------|------|----------|----------|------------|--------------|",
    ]
    by_size: dict[str, list[dict]] = {}
    for r in all_rows:
        by_size.setdefault(r["size"], []).append(r)
    for size, rows in by_size.items():
        load = rows[0]["load_seconds"]
        model = rows[0]["model"]
        sec = {r["prompt"]: r["seconds"] for r in rows}
        lines.append(
            f"| {size} | `{model}` | {load} | {sec.get('인사', '-')} | {sec.get('멀티턴', '-')} | {sec.get('지시준수', '-')} |"
        )

    lines += ["", "## 응답 전문", ""]
    for r in all_rows:
        lines += [
            f"### {r['size']} — {r['prompt']} ({r['seconds']}s, {r['status']})",
            "",
            "```",
            r["response"],
            "```",
            "",
        ]
    lines += [
        "## 체감 메모 (직접 작성)",
        "",
        "- 0.5B vs 1.5B:",
        "- 1.5B vs 3B:",
        "- 3B vs 7B:",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["0.5B", "1.5B"],
        choices=list(MODEL_MAP),
        help="Compare these sizes in order (default: 0.5B 1.5B)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output markdown path (default: assets/model_compare_results.md; "
        "7B-only runs write assets/model_compare_7B_colab.md)",
    )
    args = parser.parse_args()
    device = pick_device()
    print(f"device={device}, models={args.models}", flush=True)

    all_rows: list[dict] = []
    for size in args.models:
        try:
            all_rows.extend(run_one(size, device))
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {size}: {type(e).__name__}: {e}", flush=True)
            all_rows.append(
                {
                    "size": size,
                    "model": MODEL_MAP[size],
                    "prompt": "(load)",
                    "seconds": 0,
                    "load_seconds": 0,
                    "status": "error",
                    "response": f"LOAD ERROR: {type(e).__name__}: {e}",
                }
            )

    out = args.out
    if out is None:
        if args.models == ["7B"]:
            out = ROOT / "assets" / "model_compare_7B_colab.md"
        else:
            out = OUT
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(exist_ok=True)
    out.write_text(to_markdown(all_rows, device), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
