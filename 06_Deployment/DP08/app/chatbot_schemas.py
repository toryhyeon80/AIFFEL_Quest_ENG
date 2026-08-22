"""서울 실내 추천 챗봇 API 스키마"""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class Message(BaseModel):
    role: str = Field(..., description="역할: 'user' 또는 'bot'")
    content: str = Field(..., min_length=1, description="메시지 내용")


class ChatRequest(BaseModel):
    messages: list[Message] = Field(
        ...,
        min_length=1,
        description="대화 기록. 마지막 메시지가 사용자의 현재 입력.",
    )
    max_new_tokens: int = Field(default=140, ge=10, le=500)
    temperature: float = Field(default=0.3, gt=0.0, le=2.0)
    model_key: Literal["1.5B", "3B"] = Field(
        default="3B",
        description="사용할 모델 크기. 서버에 다른 모델이 로드돼 있으면 교체 로드합니다.",
    )
    strict_indoor: bool = Field(
        default=True,
        description="야외 키워드 감지 시 1회 재생성",
    )
    use_rag: bool = Field(
        default=True,
        description="places.json 미니 RAG로 top-k 장소를 프롬프트에 주입",
    )
    rag_top_k: int = Field(default=5, ge=1, le=10)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "주말에 서울에서 실내 데이트 코스 2개만 추천해줘",
                        }
                    ],
                    "max_new_tokens": 140,
                    "temperature": 0.3,
                    "model_key": "3B",
                    "strict_indoor": True,
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    success: bool
    response: str
    model_name: str
    model_key: str
    user: Optional[str] = None
    retried: bool = False
    retry_reasons: list[str] = Field(default_factory=list)
    outdoor_hits: list[str] = Field(default_factory=list)
    suspicious_hits: list[str] = Field(default_factory=list)
    place_hits: list[str] = Field(default_factory=list)
    excluded_places: list[str] = Field(default_factory=list)
    rag_hits: list[str] = Field(default_factory=list)
    rag_backend: Optional[str] = Field(
        default=None,
        description="검색 백엔드: embedding 또는 keyword",
    )
