"""
DP08 — 한국어 감정 분석 Streamlit UI (노트북 baseline)
"""
import streamlit as st
import requests

st.set_page_config(page_title="감정 분석", page_icon="📈", layout="centered")

API_BASE = "http://localhost:8000"
DEFAULT_KEY = "test-key-001"

st.title("한국어 감정 분석")
st.write("금융 뉴스 문장의 감성을 **snunlp/KR-FinBert-SC** 로 분류합니다.")

with st.sidebar:
    st.header("설정")
    api_key = st.text_input("API Key", value=DEFAULT_KEY, type="password")
    st.caption("DP08 baseline · POST /predict")

    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        if health.get("status") == "healthy":
            st.success("서버 연결됨")
            st.caption(health.get("model", ""))
        else:
            st.warning("모델 로딩 중...")
    except Exception:
        st.error("서버 연결 실패 — `uvicorn app.main:app --port 8000`")

text = st.text_area(
    "분석할 문장",
    value="오늘 실적 발표가 기대 이상이라 주가가 크게 올랐다.",
    height=120,
)

if st.button("분석하기", type="primary"):
    if not text.strip():
        st.warning("문장을 입력하세요.")
    else:
        try:
            resp = requests.post(
                f"{API_BASE}/predict",
                json={"text": text.strip()},
                headers={"X-API-Key": api_key},
                timeout=60,
            )
            if resp.status_code == 401:
                st.error("인증 실패 — API Key를 확인하세요.")
            elif resp.status_code == 422:
                st.error("입력 검증 실패 — 문장 길이를 확인하세요.")
            else:
                resp.raise_for_status()
                data = resp.json()
                if data.get("success"):
                    st.success(f"**{data['label']}** (score={data['score']:.4f})")
                    if data.get("user"):
                        st.caption(f"user={data['user']}")
                else:
                    st.error("추론에 실패했습니다.")
        except requests.exceptions.ConnectionError:
            st.error("서버에 연결할 수 없습니다.")
        except Exception as e:
            st.error(f"오류: {e}")

with st.expander("예시 문장"):
    for sample in [
        "실적 쇼크로 주가가 급락했다.",
        "신제품 출시 소식에 투자 심리가 개선됐다.",
        "시장 전망은 중립적이다.",
    ]:
        st.code(sample)
