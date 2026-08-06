"""
=============================================================================
BONUS 3 : 음성 통역 파이프라인 (Whisper → 번역 → TTS)
=============================================================================
5일차 수업의 3.voicetest.py 입니다.

★ 모델 3개를 사슬처럼 연결합니다

    영어 음성 ──[Whisper]──▶ 영어 텍스트 ──[m2m100]──▶ 한국어 ──[gTTS]──▶ 한국어 음성
                 STT                        번역                  TTS

  ★ 이런 걸 '파이프라인 체이닝' 이라고 합니다.
    앞 모델의 출력이 뒤 모델의 입력이 되죠.
    ⚠️ 그래서 앞에서 생긴 오류가 뒤로 전파되어 증폭됩니다.

⏱️ 첫 실행 시 다운로드
    Whisper base   약  290MB
    m2m100_418M    약 1.9GB   ← 큽니다! 번역을 끄면 안 받습니다
    합계           약 2.2GB

실행 : python 03_voice_pipeline.py
=============================================================================
"""

import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import gradio as gr

BASE = Path(__file__).resolve().parent
SAMPLES = BASE / "samples" / "audio"

_cache = {}


def _device():
    import torch
    return 0 if torch.cuda.is_available() else -1


def get_stt():
    """음성 → 텍스트 (Whisper)"""
    if "stt" not in _cache:
        print("Whisper 로딩 중... (첫 실행은 290MB)")
        from transformers import pipeline
        _cache["stt"] = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=_device(),
        )
        print("완료")
    return _cache["stt"]


def get_translator():
    """영어 → 한국어 (m2m100). 1.9GB라 필요할 때만 부릅니다."""
    if "trans" not in _cache:
        print("번역 모델 로딩 중... (첫 실행은 1.9GB — 시간이 걸립니다)")
        from transformers import pipeline
        _cache["trans"] = pipeline(
            "translation",
            model="facebook/m2m100_418M",
            device=_device(),
        )
        print("완료")
    return _cache["trans"]


# =============================================================================
# 파이프라인
# =============================================================================
def process(audio_path, do_translate, do_tts):
    if not audio_path:
        return "", "", None, "음성 파일을 올리거나 녹음해주세요."

    notes = []

    # ── 1단계 : STT ──────────────────────────────────────────────────────
    try:
        stt = get_stt()
        result = stt(audio_path)
        text_en = (result.get("text") or "").strip()
    except Exception as e:
        return "", "", None, f"❌ 음성 인식 실패: {type(e).__name__}: {e}"

    if not text_en:
        return "", "", None, "🔇 음성에서 아무 말도 인식하지 못했습니다."

    # ★ Whisper 는 조용하거나 잡음뿐인 구간에서 엉뚱한 문장을 만들어냅니다.
    #   (환각, hallucination) 실무에서 유명한 문제입니다.
    if len(text_en) < 4:
        notes.append("🤔 인식된 텍스트가 매우 짧습니다. 음성이 잘 안 들렸을 수 있습니다.")

    # ── 2단계 : 번역 ─────────────────────────────────────────────────────
    text_ko = ""
    if do_translate:
        try:
            tr = get_translator()
            out = tr(text_en, src_lang="en", tgt_lang="ko")
            text_ko = out[0]["translation_text"]
        except Exception as e:
            notes.append(f"❌ 번역 실패: {type(e).__name__}: {e}")

    # ── 3단계 : TTS ──────────────────────────────────────────────────────
    audio_out = None
    if do_tts and text_ko:
        try:
            from gtts import gTTS
            f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            gTTS(text=text_ko, lang="ko").save(f.name)
            audio_out = f.name
        except Exception as e:
            notes.append(f"🔇 음성 생성 실패: {type(e).__name__} "
                         f"(gTTS 는 인터넷이 필요합니다)")

    # ── 오류 전파 경고 ───────────────────────────────────────────────────
    if do_translate and text_ko:
        notes.append(
            "⚠️ **오류가 전파됩니다.** Whisper 가 한 단어를 잘못 들으면 "
            "번역도 틀리고, 음성도 틀립니다. 1단계 텍스트를 먼저 확인하세요."
        )

    return text_en, text_ko, audio_out, "\n\n".join(notes)


# =============================================================================
# UI
# =============================================================================
examples = []
if SAMPLES.exists():
    for p in sorted(SAMPLES.glob("*.mp3"))[:5]:
        examples.append([str(p)])

with gr.Blocks(title="음성 통역 파이프라인") as demo:
    gr.Markdown("""
# 🎙️ 음성 통역 — 모델 3개를 사슬로 연결

```
영어 음성 ──[Whisper]──▶ 영어 텍스트 ──[m2m100]──▶ 한국어 ──[gTTS]──▶ 한국어 음성
             STT                       번역                  TTS
```

⚠️ **번역 모델이 1.9GB 입니다.** 첫 실행 시 다운로드에 시간이 걸립니다.
용량이 부담되면 아래에서 번역을 끄고 **음성 인식만** 써보세요.
""")

    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(label="영어 음성",
                                sources=["upload", "microphone"],
                                type="filepath")
            do_tr = gr.Checkbox(value=True, label="🌐 한국어 번역 (첫 실행 1.9GB)")
            do_tts = gr.Checkbox(value=True, label="🔊 한국어 음성 생성")
            btn = gr.Button("변환하기", variant="primary")
            if examples:
                gr.Examples(examples=examples, inputs=audio_in,
                            label="샘플 음성 (수업 자료)")

        with gr.Column():
            out_en = gr.Textbox(label="1단계 · 영어 텍스트 (Whisper)", lines=3)
            out_ko = gr.Textbox(label="2단계 · 한국어 번역 (m2m100)", lines=3)
            out_audio = gr.Audio(label="3단계 · 한국어 음성 (gTTS)", type="filepath")
            out_warn = gr.Markdown()

    gr.Markdown("""
---
### 💡 이 실습의 핵심 — 오류 전파(error propagation)

모델을 연결하면 **정확도가 곱해집니다.**

```
Whisper 90% × 번역 85% × TTS 100%  =  전체 76%
```

각 단계는 꽤 정확한데 전체는 확 떨어지죠.

**그래서 실무에서는 중간 결과를 반드시 보여줍니다.**
이 앱이 1·2·3단계를 따로 출력하는 이유입니다.
사용자가 "아, 1단계에서 잘못 들었구나"를 알 수 있어야 하니까요.

### 🚩 Whisper 의 유명한 문제 — 환각(hallucination)

조용한 구간이나 잡음만 있는 음성에서 **없는 말을 만들어냅니다.**
"Thank you for watching" 같은 문장이 자주 나와요. 유튜브 자막으로 학습해서입니다.

> 🔗 STAGE 1에서 `asdf qwer zxcv` 에 99.5% 확신을 준 그 성질입니다.
> **모델은 입력이 없어도 뭔가를 출력합니다.**
""")

    btn.click(process, [audio_in, do_tr, do_tts],
              [out_en, out_ko, out_audio, out_warn])


if __name__ == "__main__":
    demo.launch(inbrowser=True)


# =============================================================================
# 🎯 미션
# =============================================================================
# 1. 샘플 음성 5개를 순서대로 돌려보세요. 1단계 영어 텍스트가 정확한가요?
#
# 2. ★ 마이크로 직접 영어를 말해보세요. 발음이 어눌해도 알아듣나요?
#
# 3. ★★ 아무 말 없이 3초 녹음해서 넣어보세요.
#    Whisper 가 뭐라고 하나요? 환각이 나오나요?
#
# 4. ★★★ 번역을 끄고 STT 만 켜보세요. 훨씬 빠르죠?
#    → 실무에서 "모든 기능을 다 켜지 않는 것"도 설계입니다.
#      사용자가 필요한 단계만 쓰게 하면 비용과 지연이 줄어듭니다.
# =============================================================================
