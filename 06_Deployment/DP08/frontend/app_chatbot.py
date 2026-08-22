"""
DP08 - 서울 실내 추천 챗봇 UI
"""
import streamlit as st
import requests


st.set_page_config(
    page_title="서울 실내 추천 봇",
    page_icon="🏛️",
    layout="wide",
)

# centered보다 넓게 — 예시 질문이 한 줄로 보이도록
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    div[data-testid="stExpander"] button p {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

API_BASE = "http://localhost:8000"

EXAMPLE_PROMPTS = [
    "주말에 서울에서 실내 데이트 코스 2개만 추천해줘",
    "비 오는 날 혼자 가기 좋은 실내 장소 알려줘",
    "아이랑 갈 수 있는 서울 실내 체험 추천해줘",
    "한강 산책하고 싶은데, 비슷한 분위기의 실내 대안 있어?",
]


def call_chat_api(
    messages,
    api_key,
    max_new_tokens=140,
    temperature=0.3,
    model_key="3B",
    strict_indoor=True,
    use_rag=True,
):
    try:
        resp = requests.post(
            f"{API_BASE}/chat",
            json={
                "messages": messages,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "model_key": model_key,
                "strict_indoor": strict_indoor,
                "use_rag": use_rag,
            },
            headers={"X-API-Key": api_key},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("🔌 **서버에 연결할 수 없습니다.** `uvicorn app.chatbot_api:app` 를 확인하세요.")
        return None
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            st.error("🔑 **인증 실패.** API Key를 확인하세요.")
        else:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            st.error(f"❌ **서버 에러** (HTTP {e.response.status_code}) {detail}")
        return None
    except Exception as e:
        st.error(f"❌ **오류:** {type(e).__name__}: {e}")
        return None


def format_meta_caption(meta: dict) -> str:
    bits = []
    reasons = meta.get("retry_reasons") or []
    if meta.get("retried"):
        if "hallucination" in reasons and "outdoor" in reasons:
            bits.append("가드레일 재생성(야외·환각)")
        elif "hallucination" in reasons:
            bits.append("가드레일 재생성(환각)")
        elif "outdoor" in reasons:
            bits.append("가드레일 재생성(야외)")
        else:
            bits.append("가드레일 재생성")
    if meta.get("outdoor_hits"):
        bits.append("잔여 야외: " + ", ".join(meta["outdoor_hits"]))
    if meta.get("suspicious_hits"):
        bits.append("잔여 의심명: " + ", ".join(meta["suspicious_hits"]))
    if meta.get("place_hits"):
        short = [p.split("(")[0].strip() for p in meta["place_hits"][:3]]
        bits.append("목록매칭: " + ", ".join(short))
    if meta.get("excluded_places"):
        short = [p.split("(")[0].strip() for p in meta["excluded_places"][:3]]
        bits.append("제외: " + ", ".join(short))
    if meta.get("rag_hits"):
        short = [p.split("(")[0].strip() for p in meta["rag_hits"][:3]]
        backend = meta.get("rag_backend") or "rag"
        bits.append(f"RAG({backend}): " + ", ".join(short))
    if meta.get("model_key"):
        bits.append(f"model={meta['model_key']}")
    return " · ".join(bits)


if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "server_model_key" not in st.session_state:
    st.session_state["server_model_key"] = None
if "server_model_name" not in st.session_state:
    st.session_state["server_model_name"] = None


with st.sidebar:
    st.header("설정")
    api_key = st.text_input("API Key", value="test-key-001", type="password")

    st.divider()
    model_key = st.selectbox(
        "모델",
        options=["1.5B", "3B"],
        index=1,
        help="기본은 3B(품질). 1.5B는 더 가볍고 빠릅니다. 전환 시 첫 요청에서 다시 로드합니다.",
    )
    profile = st.radio(
        "응답 스타일",
        options=["정확 (추천)", "창의"],
        index=0,
        help="정확=temperature 0.3, 창의=0.9",
    )
    temperature = 0.3 if profile.startswith("정확") else 0.9
    st.caption(f"현재 temperature = {temperature}")
    max_tokens = st.slider("최대 생성 토큰", 40, 300, 140, step=10)
    strict_indoor = st.checkbox("실내 가드레일 (야외 감지 시 1회 재생성)", value=True)
    use_rag = st.checkbox("미니 RAG (임베딩 top-k 주입)", value=True)

    st.divider()
    # /health로 연결 확인 + session에 로드 모델 동기화
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        if health.get("status") == "healthy":
            st.success("서버 연결됨")
            if health.get("model_key"):
                st.session_state["server_model_key"] = health.get("model_key")
            if health.get("model"):
                st.session_state["server_model_name"] = health.get("model")
        else:
            st.warning("모델 로딩 중...")
    except Exception:
        st.error("서버 연결 실패")

    loaded_key = st.session_state.get("server_model_key")
    loaded_name = st.session_state.get("server_model_name")
    if loaded_key:
        st.caption(f"로드됨: {loaded_key} · {loaded_name or 'N/A'}")
        if loaded_key != model_key:
            st.info(f"선택={model_key} → 다음 요청 시 서버 모델을 교체합니다.")
    elif loaded_name:
        st.caption(f"로드됨: {loaded_name}")

    st.divider()
    if st.button("대화 초기화"):
        st.session_state["chat_messages"] = []
        st.rerun()

    st.caption("Seoul Indoor Recommender · DP08")


st.title("서울 실내 추천 봇")
st.write(
    "서울에서 **실내 활동만** 추천합니다. "
    "구체 장소는 검증된 목록을 우선 사용하고, "
    "공원·한강 같은 야외 요청은 실내 대안으로 안내합니다."
)

with st.expander("예시 질문", expanded=not st.session_state["chat_messages"]):
    cols = st.columns(2)
    for i, prompt in enumerate(EXAMPLE_PROMPTS):
        if cols[i % 2].button(prompt, use_container_width=True, key=f"ex_{i}"):
            st.session_state["pending_input"] = prompt
            st.rerun()

for msg in st.session_state["chat_messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        meta = msg.get("meta")
        if meta:
            caption = format_meta_caption(meta)
            if caption:
                st.caption(caption)

pending = st.session_state.pop("pending_input", None)
user_input = st.chat_input("실내 추천을 물어보세요...")
if pending:
    user_input = pending

if user_input:
    st.session_state["chat_messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("실내 코스 생각 중... (모델 전환 시 첫 로드는 더 걸립니다)"):
            api_messages = []
            for msg in st.session_state["chat_messages"]:
                role = "user" if msg["role"] == "user" else "bot"
                api_messages.append({"role": role, "content": msg["content"]})

            result = call_chat_api(
                messages=api_messages,
                api_key=api_key,
                max_new_tokens=max_tokens,
                temperature=temperature,
                model_key=model_key,
                strict_indoor=strict_indoor,
                use_rag=use_rag,
            )

        if result and result.get("success"):
            bot_response = result["response"]
            meta = {
                "retried": result.get("retried", False),
                "retry_reasons": result.get("retry_reasons", []),
                "outdoor_hits": result.get("outdoor_hits", []),
                "suspicious_hits": result.get("suspicious_hits", []),
                "place_hits": result.get("place_hits", []),
                "excluded_places": result.get("excluded_places", []),
                "rag_hits": result.get("rag_hits", []),
                "rag_backend": result.get("rag_backend"),
                "model_key": result.get("model_key"),
            }
            # 채팅 직후 사이드바「로드됨」이 실제 사용 모델과 맞도록 동기화
            if result.get("model_key"):
                st.session_state["server_model_key"] = result["model_key"]
            if result.get("model_name"):
                st.session_state["server_model_name"] = result["model_name"]

            st.session_state["chat_messages"].append(
                {"role": "assistant", "content": bot_response, "meta": meta}
            )
            st.rerun()
        else:
            st.write("응답을 생성하지 못했습니다.")
