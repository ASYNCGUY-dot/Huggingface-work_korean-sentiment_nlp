"""
=============================================================================
STAGE 5 - MISSION 1 : Gradio 기초 — Interface vs Blocks
=============================================================================
5일차 수업의 1·2번 실습(interface.py, block.py)을 하나로 합쳤습니다.

Gradio 는 파이썬 함수 하나를 웹 앱으로 만들어주는 도구입니다.
HTML/CSS/JS 를 몰라도 됩니다.

★ 실행 방법이 다릅니다
    python 01_gradio_basics.py
  → 브라우저가 자동으로 열립니다.
  → 종료는 터미널에서 Ctrl+C

실행 : python 01_gradio_basics.py
=============================================================================
"""

import gradio as gr

print(f"Gradio 버전: {gr.__version__}")


# =============================================================================
# [1] 실행할 함수 — 이게 앱의 '내용물'입니다
# =============================================================================
def greet(name, intensity):
    """Gradio 는 이 함수의 입력/출력에 맞춰 UI를 연결합니다."""
    if not name.strip():
        return "이름을 입력해주세요."
    return "안녕하세요, " + name + "님" + "!" * int(intensity)


def analyze_length(text):
    """글자 수를 세고 통계를 돌려줍니다. (출력이 2개인 함수 예시)"""
    if not text.strip():
        return "입력이 없습니다.", {}
    chars = len(text)
    words = len(text.split())
    summary = f"{chars}자 / {words}단어"
    # gr.Label 은 {이름: 확률} 형태의 딕셔너리를 막대그래프로 그려줍니다
    dist = {
        "짧음(50자 미만)": 1.0 if chars < 50 else 0.0,
        "보통(50~200자)": 1.0 if 50 <= chars < 200 else 0.0,
        "김(200자 이상)": 1.0 if chars >= 200 else 0.0,
    }
    return summary, dist


# =============================================================================
# [2] 방식 ① Interface — 배치를 Gradio가 알아서
# =============================================================================
# 장점 : 3줄이면 끝. 프로토타입에 최고.
# 단점 : 화면 구성을 내가 못 정합니다.
demo_interface = gr.Interface(
    fn=greet,
    inputs=[
        gr.Textbox(label="이름", placeholder="지수"),
        gr.Slider(1, 5, value=3, step=1, label="느낌표 개수"),
    ],
    outputs=gr.Textbox(label="인사말"),
    title="Interface 방식",
    description="입력/출력만 알려주면 Gradio가 화면을 자동으로 만듭니다.",
)


# =============================================================================
# [3] 방식 ② Blocks — 내가 직접 디자인
# =============================================================================
# with gr.Blocks() as demo:  안에서 컴포넌트를 '선언한 순서대로' 배치됩니다.
#   gr.Row()    : 가로로 나란히
#   gr.Column() : 세로로 쌓기
with gr.Blocks() as demo_blocks:
    gr.Markdown("## Blocks 방식")
    gr.Markdown("컴포넌트를 직접 배치합니다. 실제 서비스는 대부분 이쪽입니다.")

    with gr.Row():                          # 좌우로 나눔
        with gr.Column():                   # 왼쪽 열
            txt_in = gr.Textbox(label="분석할 문장", lines=5,
                                placeholder="여기에 아무 문장이나 써보세요")
            with gr.Row():                  # 버튼 두 개를 가로로
                btn = gr.Button("분석", variant="primary")
                clear = gr.ClearButton(value="지우기")

        with gr.Column():                   # 오른쪽 열
            txt_out = gr.Textbox(label="결과", interactive=False)
            label_out = gr.Label(label="길이 분포")

    # 이벤트 연결 : 버튼을 누르면 fn 실행, inputs 를 넣고 outputs 에 결과를 넣음
    btn.click(fn=analyze_length, inputs=txt_in, outputs=[txt_out, label_out])
    clear.add([txt_in, txt_out, label_out])

    # 예시 버튼 — 사용자가 뭘 넣어야 할지 모를 때 필수입니다
    gr.Examples(
        examples=["오늘 날씨가 좋네요.",
                  "이 영화 정말 최고예요. 꼭 보세요!",
                  "글쎄요, 기대했던 것과는 좀 다르네요."],
        inputs=txt_in,
    )


# =============================================================================
# [4] 탭으로 두 방식을 한 화면에
# =============================================================================
with gr.Blocks() as demo:
    gr.Markdown("# Gradio 기초 — 두 가지 방식 비교")
    with gr.Tabs():
        with gr.Tab("① Interface (자동 배치)"):
            demo_interface.render()
        with gr.Tab("② Blocks (직접 배치)"):
            demo_blocks.render()

    gr.Markdown("""
---
### 언제 무엇을 쓰나

| | Interface | Blocks |
|---|---|---|
| 코드량 | 3줄 | 20줄+ |
| 레이아웃 자유도 | 낮음 | 높음 |
| 용도 | 빠른 프로토타입 | **실제 서비스** |

### 자주 쓰는 컴포넌트

| 컴포넌트 | 용도 |
|---|---|
| `gr.Textbox` | 텍스트 입출력 |
| `gr.Slider` | 숫자 범위 (임계값 조절에 유용!) |
| `gr.Image` | 이미지 업로드 |
| `gr.Audio` | 음성 녹음/업로드 |
| `gr.Label` | 확률을 막대그래프로 |
| `gr.Dataframe` | 표 |
| `gr.File` | 파일 다운로드 |
""")


# =============================================================================
# [5] 실행
# =============================================================================
if __name__ == "__main__":
    # launch() 주요 옵션
    #   inbrowser=True : 브라우저 자동 실행
    #   share=True     : 72시간짜리 공개 링크 생성 (임시 데모용)
    #   server_port    : 포트 지정 (기본 7860)
    #
    # ★ Gradio 6.x 부터 theme 위치가 바뀌었습니다.
    #   구버전: gr.Blocks(theme=gr.themes.Soft())
    #   6.x   : demo.launch(theme=...)
    #   버전에 따라 에러가 나면 theme 인자를 빼고 실행하세요.
    try:
        demo.launch(inbrowser=True, theme=gr.themes.Soft())
    except TypeError:
        print("⚠️ 이 Gradio 버전은 launch(theme=) 를 지원하지 않습니다. theme 없이 실행합니다.")
        demo.launch(inbrowser=True)


# =============================================================================
# 🎯 미션
# =============================================================================
# 1. 두 탭을 번갈아 보며 차이를 느껴보세요. 어느 쪽이 서비스처럼 보이나요?
#
# 2. Blocks 탭에서 gr.Row() 를 gr.Column() 으로 바꿔보세요. 화면이 어떻게 변하나요?
#
# 3. ★ gr.Examples 를 지우고 실행해보세요.
#    처음 보는 사람이 뭘 입력해야 할지 알 수 있을까요?
#    → 예시 버튼은 '친절'이 아니라 '필수'입니다.
#
# 4. ★★ analyze_length 함수를 고쳐서 '한글 비율'도 계산해 반환해보세요.
#    출력 컴포넌트도 하나 더 추가해야 합니다.
# =============================================================================
