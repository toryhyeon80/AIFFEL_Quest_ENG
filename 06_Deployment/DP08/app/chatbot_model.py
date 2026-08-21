"""
서울 실내 추천 챗봇 모델.
특화 시스템 프롬프트 + (선택) 야외 응답 1회 재생성.
"""
from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.prompts import (
    SYSTEM_PROMPT,
    OUTDOOR_RETRY_INSTRUCTION,
    HALLUCINATION_RETRY_INSTRUCTION,
)
from app.guardrails import (
    looks_outdoor,
    outdoor_hits,
    looks_suspicious_place,
    suspicious_hits,
    known_place_hits,
    previously_recommended_places,
)


MODEL_CHOICES = {
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B": "Qwen/Qwen2.5-3B-Instruct",
}


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class IndoorChatbotModel:
    """인스트럭트 모델로 서울 실내 추천 응답을 생성합니다."""

    def __init__(self, model_key: str = "3B"):
        if model_key not in MODEL_CHOICES:
            raise ValueError(f"지원하지 않는 모델 키: {model_key}")

        self.model_key = model_key
        self.model_name = MODEL_CHOICES[model_key]
        self.device = resolve_device()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32

        load_kwargs: dict = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        # CUDA는 device_map으로 바로 GPU에 올려 CPU↔GPU 이중 적재를 피합니다.
        if self.device == "cuda":
            load_kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **load_kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **load_kwargs
            )
            self.model = self.model.to(self.device)

        self.model.eval()

    def _encode(self, chat_messages: list[dict]):
        encoded = self.tokenizer.apply_chat_template(
            chat_messages, add_generation_prompt=True, return_tensors="pt"
        )
        if not torch.is_tensor(encoded):
            encoded = encoded["input_ids"]
        return encoded.to(self.device)

    def _build_chat(self, messages: list[dict]) -> list[dict]:
        system_content = SYSTEM_PROMPT
        # 이전 턴에서 이미 추천한 장소는 시스템 프롬프트에 제외 목록으로 붙입니다.
        excluded = previously_recommended_places(messages)
        if excluded:
            lines = "\n".join(f"- {name}" for name in excluded)
            system_content += (
                "\n\n[이미 추천한 장소 — 이번 답변에서 제외]\n" + lines
            )

        chat = [{"role": "system", "content": system_content}]
        for msg in messages:
            role = "user" if msg["role"] == "user" else "assistant"
            chat.append({"role": role, "content": msg["content"]})
        return chat

    def _generate_once(
        self,
        chat: list[dict],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> str:
        input_ids = self._encode(chat)

        max_length = getattr(self.model.config, "max_position_embeddings", 2048)
        while input_ids.shape[1] > max_length - max_new_tokens and len(chat) > 2:
            chat.pop(1)
            input_ids = self._encode(chat)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.05),
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        return response if response else "(응답을 생성하지 못했습니다)"

    def generate_response(
        self,
        messages: list[dict],
        max_new_tokens: int = 140,
        temperature: float = 0.3,
        top_k: int = 50,
        top_p: float = 0.9,
        strict_indoor: bool = True,
    ) -> dict:
        """
        Returns:
            response, retried, retry_reasons, outdoor_hits, suspicious_hits,
            place_hits, excluded_places
        """
        excluded = previously_recommended_places(messages)
        chat = self._build_chat(messages)
        response = self._generate_once(
            chat, max_new_tokens, temperature, top_k, top_p
        )
        retried = False
        retry_reasons: list[str] = []

        need_outdoor = strict_indoor and looks_outdoor(response)
        need_place = looks_suspicious_place(response)

        if need_outdoor or need_place:
            if need_outdoor:
                retry_reasons.append("outdoor")
            if need_place:
                retry_reasons.append("hallucination")

            if need_outdoor and need_place:
                retry_msg = (
                    OUTDOOR_RETRY_INSTRUCTION + " " + HALLUCINATION_RETRY_INSTRUCTION
                )
            elif need_outdoor:
                retry_msg = OUTDOOR_RETRY_INSTRUCTION
            else:
                retry_msg = HALLUCINATION_RETRY_INSTRUCTION

            if excluded:
                short = ", ".join(p.split("(")[0].strip() for p in excluded[:5])
                retry_msg += f" 이미 추천한 장소({short})는 제외하세요."

            retry_chat = chat + [
                {"role": "assistant", "content": response},
                {"role": "user", "content": retry_msg},
            ]
            response = self._generate_once(
                retry_chat, max_new_tokens, temperature, top_k, top_p
            )
            retried = True

        return {
            "response": response,
            "retried": retried,
            "retry_reasons": retry_reasons,
            "outdoor_hits": outdoor_hits(response),
            "suspicious_hits": suspicious_hits(response),
            "place_hits": known_place_hits(response),
            "excluded_places": excluded,
        }
