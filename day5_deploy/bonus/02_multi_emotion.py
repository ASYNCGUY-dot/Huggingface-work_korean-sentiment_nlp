"""
=============================================================================
BONUS 2 : 다중 감정 분석 + 음성 생성
=============================================================================
5일차 수업의 1.multi-text.py, 1-2.multi-textsound.py 를 합쳤습니다.

★ 지금까지와 다른 점 — 2지선다가 아닙니다

    STAGE 1~5 : 긍정 / 부정   (2개)
    이번       : anger, disgust, fear, joy, neutral, sadness, surprise  (7개)

    ★ neutral(중립)이 있습니다!
      STAGE 5 내내 "중립을 출력할 수 없다"가 한계였는데,
      클래스를 늘리면 그 문제가 해결됩니다.

⚠️ 이 모델은 영어 전용입니다. 한국어를 넣으면 STAGE 1의 그 사건이 재현됩니다.
   (일부러 확인해보시는 것도 좋습니다)

⏱️ 첫 실행 시 약 330MB 다운로드

실행 : python 02_multi_emotion.py
=============================================================================
"""

import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import gradio as gr

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)

_cache = {}


def get_classifier():
    if "clf" not in _cache:
        print("감정 분석 모델 로딩 중... (첫 실행은 330MB)")
        import torch
        from transformers import pipeline
        _cache["clf"] = pipeline(
            task="text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            framework="pt",
            device=0 if torch.cuda.is_available() else -1,
            # ★ 2026-08 수정: return_all_scores=True 는 폐기 예정 문법입니다.
            #   STAGE 1 시작할 때 제가 지적했던 그 항목입니다.
            #   top_k=None 이 현재 문법이고, 결과 구조는 동일합니다.
            top_k=None,
        )
        print("완료")
    return _cache["clf"]


# 감정별 한글 이름과 이모지
EMOTION_KO = {
    "anger": "분노 😠", "disgust": "혐오 🤢", "fear": "두려움 😨",
    "joy": "기쁨 😄", "neutral": "중립 😐", "sadness": "슬픔 😢",
    "surprise": "놀람 😲",
}


def analyze(text, make_voice):
    if not text or not text.strip():
        return "문장을 입력해주세요.", {}, None, ""

    text = text.strip()
    clf = get_classifier()

    # top_k=None 이면 [[{label, score}, ...]] 형태로 옵니다
    scores = clf(text)[0]
    scores = sorted(scores, key=lambda x: -x["score"])

    top = scores[0]
    label_ko = EMOTION_KO.get(top["label"], top["label"])
    summary = f"## {label_ko}\n\n확신도 **{top['score']:.1%}**"

    dist = {EMOTION_KO.get(s["label"], s["label"]): float(s["score"])
            for s in scores}

    notes = []
    # 7지선다이므로 무작위 기준선은 1/7 ≈ 0.143 입니다.
    # STAGE 1에서 배운 '배율' 개념을 그대로 적용합니다.
    baseline = 1 / len(scores)
    ratio = top["score"] / baseline
    if ratio < 2.0:
        notes.append(f"🤔 **확신도가 낮습니다.** 7지선다에서 찍으면 "
                     f"{baseline:.1%}인데 지금 {top['score']:.1%} "
                     f"(기준선의 {ratio:.1f}배)")

    # 1·2위가 비슷하면 애매한 문장입니다
    if len(scores) > 1 and scores[0]["score"] - scores[1]["score"] < 0.15:
        second = EMOTION_KO.get(scores[1]["label"], scores[1]["label"])
        notes.append(f"⚖️ **1·2위가 접전입니다.** {label_ko} vs {second} "
                     f"— 복합적인 감정일 수 있습니다.")

    # 한국어 감지 — 이 모델은 영어 전용입니다
    if any("가" <= ch <= "힣" for ch in text):
        notes.append("🚩 **한국어가 감지되었습니다.** 이 모델은 영어 전용입니다. "
                     "결과를 믿지 마세요. (STAGE 1의 그 사건이 재현됩니다)")

    # ---- 음성 생성 ------------------------------------------------------
    audio_path = None
    if make_voice:
        try:
            from gtts import gTTS
            # ★ 파일명을 uuid 로 만드는 이유:
            #   같은 이름을 쓰면 여러 사람이 동시에 쓸 때 서로 덮어씁니다.
            audio_path = str(OUT / f"tts_{uuid.uuid4().hex[:8]}.mp3")
            gTTS(text=text, lang="en").save(audio_path)
        except Exception as e:
            notes.append(f"🔇 음성 생성 실패: {type(e).__name__} "
                         f"(gTTS 는 인터넷 연결이 필요합니다)")
            audio_path = None

    return summary, dist, audio_path, "\n\n".join(notes)


# =============================================================================
# UI
# =============================================================================
with gr.Blocks(title="다중 감정 분석") as demo:
    gr.Markdown("""
# 📊 다중 감정 분석 + 음성 생성

`j-hartmann/emotion-english-distilroberta-base` — **7가지 감정**을 분류합니다.

분노 · 혐오 · 두려움 · 기쁨 · **중립** · 슬픔 · 놀람

> ★ 지금까지 만든 모델과 달리 **중립(neutral)이 있습니다.**
> STAGE 5 내내 "2지선다라 모르겠음을 출력할 수 없다"가 한계였죠.
> 클래스를 늘리면 그 문제가 풀립니다.
""")

    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label="영어 문장", lines=4,
                             placeholder="I feel amazing today!")
            voice = gr.Checkbox(value=False, label="🔊 음성 생성 (gTTS, 인터넷 필요)")
            btn = gr.Button("분석하기", variant="primary")
            gr.Examples(
                examples=[
                    ["I feel amazing today!"],
                    ["I can't believe you did that to me."],
                    ["The meeting is scheduled for 3 PM."],
                    ["I'm scared something bad will happen."],
                    ["Wait, what?! That came out of nowhere."],
                    ["오늘 정말 기분이 좋아요."],
                ],
                inputs=txt,
                label="예시 (마지막은 한국어 — 일부러 넣어보세요)",
            )

        with gr.Column():
            out_summary = gr.Markdown("### 결과가 여기에 표시됩니다")
            out_dist = gr.Label(label="7가지 감정 확률", num_top_classes=7)
            out_warn = gr.Markdown()
            out_audio = gr.Audio(label="생성된 음성", type="filepath")

    gr.Markdown("""
---
### 💡 관찰 포인트

1. **`"The meeting is scheduled for 3 PM."`** → 어떤 감정이 1위인가요?
   2지선다 모델은 이걸 억지로 긍정/부정 중 하나로 골랐습니다.
   여기서는 **neutral** 을 고를 수 있습니다.

2. **7지선다의 기준선은 14.3%** 입니다. (2지선다는 50%)
   그래서 40%도 꽤 확신하는 겁니다. STAGE 1에서 배운 **배율** 개념이죠.

3. **1·2위가 접전인 문장**을 찾아보세요. 사람도 헷갈리는 문장일 겁니다.

4. **한국어를 넣으면** 🚩 경고가 뜹니다. 그래도 결과는 나옵니다.
   STAGE 1 첫날의 그 문제, 다섯 번째 재회입니다.
""")

    btn.click(analyze, [txt, voice],
              [out_summary, out_dist, out_audio, out_warn])
    txt.submit(analyze, [txt, voice],
               [out_summary, out_dist, out_audio, out_warn])


if __name__ == "__main__":
    demo.launch(inbrowser=True)


# =============================================================================
# 🎯 미션
# =============================================================================
# 1. 예시 6개를 순서대로 눌러보세요. 마지막(한국어)에서 경고가 뜨나요?
#
# 2. ★ 감정이 섞인 문장을 만들어보세요.
#      "I'm happy for you but also a little jealous."
#    1·2위가 접전이라는 경고가 뜨나요?
#
# 3. ★★ 음성 생성을 켜고 한국어 문장을 넣으면?
#    gTTS 는 lang="en" 으로 고정돼 있어 한국어를 영어처럼 읽습니다.
#    → 코드에서 lang 을 자동 감지하도록 고쳐보세요.
#
# 4. ★★★ 이 7클래스 모델과 STAGE 3의 2클래스 모델을 나란히 놓고
#    같은 문장을 비교하는 앱을 만들어보세요.
#    "클래스를 늘리면 무엇을 얻고 무엇을 잃는가"가 보입니다.
# =============================================================================
