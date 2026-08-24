#!/usr/bin/env python3
"""DP08 서울 실내 추천 — Colab에서 큰 모델(기본 7B) 비교.

Mac MPS는 7B 생성에 부적합하니 Colab T4/GPU를 권장합니다.
DP08의 SYSTEM_PROMPT + 가드레일(야외/환각 1회 재생성)을 그대로 사용합니다.

Colab:
  %cd /content/AIFFEL_Quest_ENG
  !git pull origin main
  %cd 06_Deployment/DP08/indoor
  !pip install -q transformers accelerate torch sentencepiece sentence-transformers
  !python scripts/compare_indoor_colab.py --models 7B

Mac (3B까지 권장):
  cd 06_Deployment/DP08/indoor
  PYTHONPATH=. python scripts/compare_indoor_colab.py --models 3B
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.prompts import (
    SYSTEM_PROMPT,
    OUTDOOR_RETRY_INSTRUCTION,
    HALLUCINATION_RETRY_INSTRUCTION,
)
from app.guardrails import (
    looks_outdoor,
    looks_suspicious_place,
    outdoor_hits,
    suspicious_hits,
    known_place_hits,
)

MODEL_MAP = {
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B": "Qwen/Qwen2.5-3B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
}

SCENARIOS = [
    {
        "name": "기본_실내데이트",
        "messages": [
            {
                "role": "user",
                "content": "주말에 서울에서 실내 데이트 코스 2개만 추천해줘",
            }
        ],
    },
    {
        "name": "야외유도_한강",
        "messages": [
            {
                "role": "user",
                "content": "한강 산책하고 싶은데, 비슷한 분위기의 실내 대안 있어?",
            }
        ],
    },
    {
        "name": "다른거_재추천",
        "messages": [
            {
                "role": "user",
                "content": "주말에 서울에서 실내 데이트 코스 2개만 추천해줘",
            },
            {
                "role": "assistant",
                "content": "1) 국립중앙박물관 — 전시 관람\n2) 디뮤지엄 — 기획전",
            },
            {
                "role": "user",
                "content": "둘 다 빼고 다른 실내 데이트로 다시 추천해줘",
            },
        ],
    },
]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str, device: str):
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if device == "cuda":
        kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model = model.to(device)
    model.eval()
    print(f"loaded in {time.time() - t0:.1f}s (device={device})")
    return model, tokenizer


def _encode(tokenizer, model, chat, device: str):
    encoded = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    )
    if not torch.is_tensor(encoded):
        encoded = encoded["input_ids"]
    if device == "cuda":
        return encoded.to(next(model.parameters()).device)
    return encoded.to(device)


def generate_once(model, tokenizer, chat, device: str, max_new_tokens: int = 140) -> str:
    input_ids = _encode(tokenizer, model, chat, device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][input_ids.shape[1] :], skip_special_tokens=True)
    return text.strip() or "(empty)"


def generate_with_guardrails(
    model, tokenizer, messages: list[dict], device: str, max_new_tokens: int = 140
) -> dict:
    chat = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in messages:
        role = "user" if msg["role"] == "user" else "assistant"
        chat.append({"role": role, "content": msg["content"]})

    t0 = time.time()
    response = generate_once(model, tokenizer, chat, device, max_new_tokens)
    reasons: list[str] = []
    retried = False

    need_out = looks_outdoor(response)
    need_fake = looks_suspicious_place(response)
    if need_out or need_fake:
        if need_out:
            reasons.append("outdoor")
        if need_fake:
            reasons.append("hallucination")
        retry_msg = ""
        if need_out:
            retry_msg += OUTDOOR_RETRY_INSTRUCTION + " "
        if need_fake:
            retry_msg += HALLUCINATION_RETRY_INSTRUCTION
        retry_chat = chat + [
            {"role": "assistant", "content": response},
            {"role": "user", "content": retry_msg.strip()},
        ]
        response = generate_once(model, tokenizer, retry_chat, device, max_new_tokens)
        retried = True

    return {
        "response": response,
        "seconds": round(time.time() - t0, 1),
        "retried": retried,
        "retry_reasons": reasons,
        "outdoor_hits": outdoor_hits(response),
        "suspicious_hits": suspicious_hits(response),
        "place_hits": known_place_hits(response),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["7B"],
        choices=list(MODEL_MAP.keys()),
        help="비교할 모델 크기 (Colab은 7B 권장)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=140)
    args = parser.parse_args()

    device = pick_device()
    print(f"device={device}, models={args.models}")
    if device != "cuda" and "7B" in args.models:
        print("WARN: 7B는 Colab CUDA 권장. MPS/CPU는 매우 느리거나 실패할 수 있음.")

    lines: list[str] = [
        "# DP08 실내 추천 — Colab 모델 비교",
        "",
        f"- device: `{device}`",
        f"- generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- models: {', '.join(args.models)}",
        "- prompt: DP08 SYSTEM_PROMPT + outdoor/hallucination guardrails",
        "",
        "## 요약",
        "",
        "| 크기 | 시나리오 | 초 | 재생성 | 목록매칭 |",
        "|------|----------|----|--------|----------|",
    ]

    details: list[str] = ["", "## 응답 전문", ""]

    for key in args.models:
        name = MODEL_MAP[key]
        print(f"\n=== {key}: {name} ===")
        model, tokenizer = load_model(name, device)
        for sc in SCENARIOS:
            print(f"- {sc['name']} ...", flush=True)
            result = generate_with_guardrails(
                model, tokenizer, sc["messages"], device, args.max_new_tokens
            )
            places = ", ".join(p.split("(")[0].strip() for p in result["place_hits"][:3]) or "-"
            reasons = ",".join(result["retry_reasons"]) if result["retried"] else "-"
            lines.append(
                f"| {key} | {sc['name']} | {result['seconds']} | {reasons} | {places} |"
            )
            details.append(f"### {key} — {sc['name']} ({result['seconds']}s)")
            details.append("")
            details.append("```")
            details.append(result["response"])
            details.append("```")
            details.append("")
            details.append(
                f"- retried: {result['retried']} ({reasons})  \n"
                f"- place_hits: {result['place_hits']}  \n"
                f"- outdoor_hits: {result['outdoor_hits']}  \n"
                f"- suspicious_hits: {result['suspicious_hits']}"
            )
            details.append("")
            print(f"  {result['seconds']}s retried={result['retried']} places={places}")

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = ROOT / "assets" / "indoor_compare_colab.md"
    if args.models == ["7B"]:
        out = ROOT / "assets" / "indoor_compare_7B_colab.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines + details) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
