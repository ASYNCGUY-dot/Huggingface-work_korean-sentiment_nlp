"""
=============================================================================
STAGE 5 - MISSION 2 : 내 모델을 웹 서비스로
=============================================================================
STAGE 3에서 직접 학습시킨 모델(검증 88%)을 Gradio 앱으로 감쌉니다.

★ 그냥 감싸기만 하는 게 아닙니다.
  STAGE 4에서 배운 안전장치를 전부 넣습니다.

    ① 임계값 슬라이더    — 사용자가 정밀도/재현율 균형을 직접 조절
    ② 확신도 표시        — 얼마나 믿을 만한지
    ③ 사람 검토 경고     — 임계값 근처는 자동 판정하지 않음
    ④ 도메인 밖 경고     — 학습해본 적 없는 종류면 경고
    ⑤ 빈 입력 방어       — 에러로 앱이 죽지 않도록

  ★ ①~④ 가 없는 AI 앱은 '데모'이고, 있으면 '서비스'입니다.

실행 : python 02_serve_my_model.py
=============================================================================
"""

import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# =============================================================================
# [0] 모델 로드 — 앱 시작 시 딱 한 번
# =============================================================================
BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE.parent / "day3_finetuning" / "data" / "colab_model"

if not MODEL_DIR.exists():
    raise SystemExit(
        f"❌ 모델이 없습니다: {MODEL_DIR}\n"
        "   STAGE 3의 Colab 노트북으로 학습한 모델을 이 경로에 풀어주세요."
    )

print("모델 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
print(f"완료: {model.config._name_or_path}")

MAX_LENGTH = 128

# =============================================================================
# 도메인 판정용 어휘
# =============================================================================
# ★ 2026-08-06 수정 — 설계 결함을 고쳤습니다.
#
#   처음엔 '토크나이저 사전'과 비교했습니다. 그런데 그건 사전학습 때 만들어진
#   32,000개 한국어 서브워드라, 한국어면 뭐든 다 들어 있습니다.
#   → "오늘 회의는 3시입니다"의 어휘 일치율이 높게 나와 경고가 안 떴습니다.
#   → 제 체크는 "영화 리뷰인가?"가 아니라 "한국어인가?"를 재고 있었습니다.
#
#   지금은 build_domain_vocab.py 가 만든 'NSMC 학습 데이터 어휘'와 비교합니다.
#   그게 진짜 도메인입니다.
import json

VOCAB_FILE = BASE / "domain_vocab.json"
DOMAIN_STEMS = set()

if VOCAB_FILE.exists():
    _d = json.loads(VOCAB_FILE.read_text(encoding="utf-8"))
    DOMAIN_STEMS = set(_d["stems"])
    print(f"도메인 어휘 로드: {len(DOMAIN_STEMS):,}개 어간 ({_d['source']})")
else:
    print("⚠️ domain_vocab.json 이 없습니다. 도메인 경고가 비활성화됩니다.")
    print("   활성화하려면: python build_domain_vocab.py")

DOMAIN_THRESHOLD = 0.50     # 어휘 일치율이 이 값 미만이면 '도메인 밖'

# ★ 확신도 기준선 — STAGE 3~4 실측 기반
#   영화 리뷰 문장:  87~99%
#   도메인 밖 문장:  65%
#   → 75% 미만이면 모델이 헤매고 있다는 신호입니다.
LOW_CONFIDENCE = 0.75


# =============================================================================
# [1] 핵심 예측 함수
# =============================================================================
def predict(text, threshold, review_band):
    """문장 하나를 분석해서 4가지 출력을 반환합니다.

    반환값의 개수와 순서는 아래 UI의 outputs= 리스트와 정확히 맞아야 합니다.
    """
    # ★ ⑤ 빈 입력 방어 — 이거 없으면 앱이 에러로 죽습니다
    if not text or not text.strip():
        return "문장을 입력해주세요.", {}, "", ""

    text = text.strip()

    # 추론
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0]
    p_pos = float(probs[1])

    # ★ ① 임계값 적용 — 0.5 고정이 아닙니다
    pred = 1 if p_pos >= threshold else 0
    label = "긍정 😊" if pred else "부정 😞"
    conf = p_pos if pred else 1 - p_pos

    verdict = f"## {label}\n\n확신도 **{conf:.1%}** · 긍정 확률 {p_pos:.1%}"
    dist = {"긍정": p_pos, "부정": 1 - p_pos}

    # ★ ③ 사람 검토 경고
    notes = []
    if abs(p_pos - threshold) < review_band:
        notes.append(
            f"⚠️ **판정 경계선 근처입니다** "
            f"(임계값 {threshold:.2f} ± {review_band:.2f})\n"
            f"자동 판정을 신뢰하지 말고 사람이 확인하세요."
        )

    # ★ ④-1 확신도가 낮으면 경고 — 가장 신뢰할 만한 신호
    #   모델 자신이 "잘 모르겠다"고 말하는 것과 같습니다.
    if conf < LOW_CONFIDENCE:
        notes.append(
            f"🤔 **확신도가 낮습니다** ({conf:.1%})\n"
            f"학습한 영화 리뷰 문장에서는 보통 87~99%가 나옵니다. "
            f"{LOW_CONFIDENCE:.0%} 미만이면 모델이 헤매고 있다는 뜻입니다."
        )

    # ★ ④-2 도메인 밖 경고 — 학습 데이터 어휘와 비교
    words = [w for w in re.findall(r"[가-힣]{2,}", text)]
    ratio = None
    if DOMAIN_STEMS and words:
        hits = sum(1 for w in words if w[:2] in DOMAIN_STEMS)
        ratio = hits / len(words)
        if ratio < DOMAIN_THRESHOLD:
            notes.append(
                f"🚩 **학습해본 적 없는 종류의 문장입니다** "
                f"(NSMC 영화 리뷰 어휘와 {ratio:.0%} 일치)\n"
                f"이 모델은 영화 리뷰 전용입니다. 다른 분야에서는 결과를 믿지 마세요."
            )

    # ★ ④-3 잘림 경고
    n_tokens = len(tokenizer.encode(text))
    if n_tokens > MAX_LENGTH:
        notes.append(
            f"✂️ **문장이 잘렸습니다** ({n_tokens}토큰 → {MAX_LENGTH}토큰)\n"
            f"한국어는 결론이 뒤에 오므로 판정이 틀릴 수 있습니다."
        )

    warning = "\n\n".join(notes)

    # 상세 정보에 판정 근거를 전부 노출합니다 (투명성)
    detail = f"토큰 {n_tokens}개 · 임계값 {threshold:.2f} · 긍정확률 {p_pos:.4f}"
    if ratio is not None:
        detail += f" · 도메인 일치 {ratio:.0%}"

    return verdict, dist, warning, detail


# =============================================================================
# [2] UI
# =============================================================================
with gr.Blocks(title="한국어 영화 리뷰 감성 분석") as demo:
    gr.Markdown("""
# 🎬 한국어 영화 리뷰 감성 분석기

`klue/roberta-base` 를 네이버 영화 리뷰(NSMC) 2만 건으로 파인튜닝한 모델입니다.
**검증 정확도 88%** · 학습에 쓰지 않은 데이터로 측정
""")

    with gr.Row():
        # ─────────────────────────── 왼쪽: 입력 ───────────────────────────
        with gr.Column(scale=1):
            text_in = gr.Textbox(
                label="분석할 문장",
                placeholder="예: 이건 진짜 내 인생 영화다.. 너무 감동이야.",
                lines=4,
            )

            with gr.Accordion("⚙️ 고급 설정", open=False):
                th_slider = gr.Slider(
                    0.05, 0.95, value=0.50, step=0.05,
                    label="판정 임계값",
                    info="높이면 '긍정' 판정에 신중해집니다 (정밀도↑ 재현율↓)",
                )
                band_slider = gr.Slider(
                    0.0, 0.30, value=0.10, step=0.01,
                    label="사람 검토 밴드",
                    info="임계값 ±이 범위 안이면 '사람이 확인하세요' 경고",
                )

            with gr.Row():
                btn = gr.Button("분석하기", variant="primary", scale=2)
                clear = gr.ClearButton(value="지우기", scale=1)

            gr.Examples(
                label="예시 (마지막 두 개는 모델이 약한 유형입니다)",
                examples=[
                    ["이건 진짜 내 인생 영화다.. 너무 감동이야."],
                    ["중간에 잠들 뻔했어요. 전개가 너무 느림."],
                    ["나쁘지 않은데 다시 볼 것 같진 않아요."],
                    ["기대했던 것과는 좀 다르네요."],
                    ["오늘 회의는 3시입니다. 자료 첨부드립니다."],
                ],
                inputs=text_in,
            )

        # ─────────────────────────── 오른쪽: 결과 ──────────────────────────
        with gr.Column(scale=1):
            verdict_out = gr.Markdown("### 결과가 여기에 표시됩니다")
            prob_out = gr.Label(label="확률 분포", num_top_classes=2)
            warn_out = gr.Markdown()
            detail_out = gr.Textbox(label="상세", interactive=False)

    gr.Markdown("""
---
### ⚠️ 이 모델의 알려진 한계

STAGE 1~4에서 **직접 실험해서 확인한** 것들입니다.

- **도메인** : 영화 리뷰로 학습했습니다. 상품평·뉴스·업무 메일은 보장 못 합니다.
- **우회 표현** : `"기대했던 것과는 좀 다르네요"` 처럼 부정 어휘가 없는 문장에 약합니다.
- **중립 없음** : 2지선다 모델이라 **"모르겠음"을 출력할 수 없습니다.** 애매해도 둘 중 하나를 고릅니다.
- **길이** : 128토큰을 넘으면 잘립니다. 한국어는 결론이 뒤에 와서 위험합니다.

### 💡 임계값 슬라이더를 움직여보세요

같은 문장인데 판정이 바뀌는 지점이 있습니다.
그게 **"모델이 확신하지 못하는 영역"** 입니다.
""")

    # 이벤트 연결
    inputs = [text_in, th_slider, band_slider]
    outputs = [verdict_out, prob_out, warn_out, detail_out]
    btn.click(fn=predict, inputs=inputs, outputs=outputs)
    text_in.submit(fn=predict, inputs=inputs, outputs=outputs)   # 엔터로도 실행
    clear.add([text_in, verdict_out, prob_out, warn_out, detail_out])


if __name__ == "__main__":
    # =========================================================================
    # ★ SHARE — 무료로 공개 링크 만들기
    # =========================================================================
    # 2026-08-06 확인: 허깅페이스 Spaces 에서 Gradio/Docker 는 PRO 유료 전용으로
    # 바뀌었습니다. 무료는 Static(HTML/JS)뿐이라 파이썬 앱을 못 올립니다.
    #
    # 대신 Gradio 가 원래부터 제공하는 기능을 씁니다.
    #   share=True  →  https://xxxxx.gradio.live 공개 주소 생성
    #
    #   ✅ 무료 / 30초 / 남에게 링크만 보내면 접속됨
    #   ⚠️ 72시간 유효 · 이 터미널이 켜져 있는 동안만 작동
    #
    # 링크가 필요 없을 땐 False 로 바꾸세요. (로컬 전용, 더 빠름)
    SHARE = True

    if SHARE:
        print("\n" + "=" * 70)
        print("공개 링크를 생성합니다.")
        print("  · 주소는 아래 'Running on public URL:' 줄에 나옵니다")
        print("  · 이 터미널을 끄면 링크도 끊깁니다")
        print("  · 최초 1회는 접속용 파일을 내려받느라 10~30초 더 걸립니다")
        print("=" * 70 + "\n")

    try:
        demo.launch(inbrowser=True, share=SHARE, theme=gr.themes.Soft())
    except TypeError:
        # Gradio 버전에 따라 theme 인자를 launch 에서 못 받는 경우
        demo.launch(inbrowser=True, share=SHARE)


# =============================================================================
# 🎯 미션
# =============================================================================
# 1. 예시 5개를 순서대로 눌러보세요. 마지막 두 개에서 경고가 뜨나요?
#
# 2. ★ 임계값 슬라이더를 0.5 → 0.9 로 올리고 같은 문장을 다시 분석하세요.
#    판정이 바뀌는 문장이 있나요? 그게 '경계선 문장'입니다.
#
# 3. ★★ 본인이 쓴 문장 10개를 넣어보세요.
#    틀린 것이 있다면, 그 문장에 경고가 떴나요?
#    (경고 없이 틀렸다면 그게 가장 위험한 케이스입니다)
#
# 4. ★★★ 이 앱에 '피드백 버튼'을 추가해보세요.
#    사용자가 "이거 틀렸어요"를 누르면 CSV에 기록하는 기능입니다.
#    → 실무 AI 서비스의 필수 요소입니다. 그 로그가 다음 학습 데이터가 됩니다.
# =============================================================================
