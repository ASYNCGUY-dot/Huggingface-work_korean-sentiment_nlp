"""
=============================================================================
STAGE 1 - BOSS BATTLE : 감정 분석 웹 대시보드 만들기
=============================================================================
목표 : 지금까지 배운 걸 합쳐 실제로 쓸 수 있는 웹 앱을 만든다.
       1일차 수업의 5.sentenceAnalysis.py 를 실무 수준으로 재구성했습니다.

★ 실행 방법이 다릅니다! python 이 아니라 streamlit 명령을 씁니다:

    streamlit run 05_streamlit_dashboard.py

브라우저가 자동으로 열립니다. 종료는 터미널에서 Ctrl+C.
=============================================================================
"""

import platform
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from transformers import pipeline


# =============================================================================
# [설정 1] 한글 폰트 — 이거 안 하면 그래프 글자가 □□□ 로 깨집니다
# =============================================================================
def set_korean_font():
    """운영체제별로 맞는 한글 폰트를 matplotlib에 설정합니다."""
    os_name = platform.system()
    if os_name == "Windows":
        plt.rc("font", family="Malgun Gothic")      # 맑은 고딕
    elif os_name == "Darwin":                        # macOS
        plt.rc("font", family="AppleGothic")
    else:                                            # Linux
        plt.rc("font", family="NanumGothic")
    # 마이너스(-) 기호가 깨지는 현상 방지
    plt.rcParams["axes.unicode_minus"] = False


set_korean_font()


# =============================================================================
# [설정 2] 페이지 설정 — 반드시 다른 st 명령보다 먼저 호출해야 합니다
# =============================================================================
st.set_page_config(page_title="감정 분석 대시보드", layout="wide")


# =============================================================================
# [핵심 1] 모델 캐싱 — Streamlit에서 가장 중요한 개념
# =============================================================================
# ★ Streamlit은 사용자가 버튼을 누를 때마다 이 파일 전체를 처음부터 다시 실행합니다.
#   캐싱이 없으면 클릭할 때마다 모델을 다시 로드해서 앱이 마비됩니다.
#
#   @st.cache_resource : 모델·DB연결 같은 "무거운 객체"용 (1회만 실행)
#   @st.cache_data     : 데이터프레임 같은 "값"용
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        framework="pt",
    )


# =============================================================================
# 헬퍼 함수
# =============================================================================
def parse_star(label: str) -> int:
    """'5 stars' → 5"""
    return int(label.split()[0])


def star_to_sentiment(star: int) -> str:
    if star >= 4:
        return "긍정 😃"
    elif star == 3:
        return "중립 😐"
    return "부정 😡"


# =============================================================================
# 메인 앱
# =============================================================================
def main():
    st.title("📊 한국어 감정 분석 대시보드")
    st.caption("모델: nlptown/bert-base-multilingual-uncased-sentiment (1~5 stars)")
    st.divider()

    # 모델 로드 (첫 실행 시에만 스피너가 돕니다)
    with st.spinner("AI 모델을 불러오는 중입니다..."):
        classifier = load_model()

    # -------------------------------------------------------------------------
    # [핵심 2] Session State — 화면이 새로고침돼도 데이터를 유지하는 저장소
    # -------------------------------------------------------------------------
    # 이게 없으면 새 문장을 입력할 때마다 이전 결과가 전부 사라집니다.
    if "records" not in st.session_state:
        st.session_state.records = []

    # -------------------------------------------------------------------------
    # 입력 영역
    # -------------------------------------------------------------------------
    col_input, col_reset = st.columns([4, 1])

    with col_input:
        st.subheader("✍️ 분석할 문장 입력")
        # st.form : 엔터를 쳐도 바로 실행되지 않고, 제출 버튼을 눌러야 실행됩니다.
        # clear_on_submit=True : 제출 후 입력창을 자동으로 비웁니다 (연속 입력 편의)
        with st.form("input_form", clear_on_submit=True):
            user_text = st.text_area(
                "여러 줄 입력 가능 (줄바꿈으로 구분)",
                placeholder="오늘 회의가 정말 유익했어요.\n배송이 너무 늦어서 실망했습니다.",
                height=120,
            )
            submitted = st.form_submit_button("분석하기", type="primary")

    with col_reset:
        st.subheader(" ")
        if st.button("🔄 전체 초기화", use_container_width=True):
            st.session_state.records = []
            st.rerun()   # 화면 즉시 새로고침

    # -------------------------------------------------------------------------
    # 분석 실행
    # -------------------------------------------------------------------------
    if submitted and user_text.strip():
        # 줄바꿈으로 나누고, 빈 줄은 제거
        lines = [line.strip() for line in user_text.split("\n") if line.strip()]

        with st.spinner(f"{len(lines)}개 문장을 분석 중..."):
            for line in lines:
                res = classifier(line)[0]
                star = parse_star(res["label"])
                st.session_state.records.append(
                    {
                        "문장": line,
                        "별점숫자": star,
                        "별점": "★" * star + "☆" * (5 - star),
                        "판정": star_to_sentiment(star),
                        "신뢰도": f"{res['score'] * 100:.1f}%",
                    }
                )
        st.success(f"{len(lines)}개 문장 분석 완료!")

    st.divider()

    # -------------------------------------------------------------------------
    # 결과가 없으면 안내만 보여주고 종료
    # -------------------------------------------------------------------------
    if not st.session_state.records:
        st.info("👆 위에 문장을 입력하고 '분석하기'를 눌러주세요.")
        return

    # -------------------------------------------------------------------------
    # 결과 표시 — 좌: 표 / 우: 그래프
    # -------------------------------------------------------------------------
    df = pd.DataFrame(st.session_state.records)
    stars = df["별점숫자"].tolist()

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("📋 분석 결과")

        # st.metric : 숫자를 크게 강조해서 보여주는 컴포넌트
        st.metric("누적 평균 별점", f"{sum(stars) / len(stars):.2f} / 5.00")

        # 화면에는 '별점숫자' 컬럼을 숨깁니다 (내부 계산용이라)
        st.dataframe(
            df[["문장", "판정", "별점", "신뢰도"]],
            use_container_width=True,
            height=320,
        )

        # CSV 다운로드
        # ★ encoding="utf-8-sig" 가 핵심입니다. 이게 없으면 엑셀에서 한글이 깨집니다.
        csv = df[["문장", "판정", "별점", "신뢰도"]].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 결과 CSV 다운로드",
            data=csv,
            file_name="sentiment_result.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with right:
        st.subheader("📈 별점 분포")

        counts = Counter(stars)
        x_labels = [f"{i}점" for i in range(1, 6)]
        y_values = [counts.get(i, 0) for i in range(1, 6)]

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
        bars = ax.bar(x_labels, y_values, color=colors, edgecolor="gray")

        ax.set_xlabel("감정 등급 (별점)")
        ax.set_ylabel("문장 개수")
        ax.set_title("입력 문장의 감정 분포")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        # y축을 정수 단위로 (0.5건 같은 눈금이 나오면 이상하니까)
        ax.set_yticks(range(0, max(y_values) + 2))

        # 막대 위에 개수 표시
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.05,
                    f"{int(h)}",
                    ha="center",
                    va="bottom",
                )

        st.pyplot(fig)

        # 요약 지표 3개를 가로로
        c1, c2, c3 = st.columns(3)
        c1.metric("긍정", f"{sum(1 for s in stars if s >= 4)}건")
        c2.metric("중립", f"{sum(1 for s in stars if s == 3)}건")
        c3.metric("부정", f"{sum(1 for s in stars if s <= 2)}건")


# =============================================================================
# 진입점
# =============================================================================
# ★ 이 패턴을 습관화하세요.
#   다른 파일에서 이 파일을 import 할 때 앱이 멋대로 실행되는 걸 막아줍니다.
if __name__ == "__main__":
    main()


# =============================================================================
# 🎯 보스전 미션
# =============================================================================
#  [난이도 ★]   문장을 10개 이상 입력해서 그래프와 CSV 다운로드를 확인하세요.
#
#  [난이도 ★★]  사이드바(st.sidebar)를 추가해서 모델을 고를 수 있게 만드세요.
#               힌트: st.sidebar.selectbox("모델 선택", ["nlptown...", "snunlp..."])
#               주의: 모델을 바꾸면 캐시도 바뀌어야 하니 load_model(model_id) 형태로
#                     인자를 받도록 수정해야 합니다.
#
#  [난이도 ★★★] CSV 파일 업로드 기능을 추가하세요.
#               힌트: uploaded = st.file_uploader("CSV 업로드", type=["csv"])
#                     df = pd.read_csv(uploaded)
#               → 이건 3일차 수업의 5-4.csvreadstreamlit.py 와 같은 기능입니다.
#
# 막히면 저(Claude)에게 물어보세요. 에러 메시지를 그대로 붙여넣으셔도 됩니다.
# =============================================================================
