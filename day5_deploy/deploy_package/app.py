"""
한국어 영화 리뷰 감성 분석 — 허깅페이스 Spaces 배포용
klue/roberta-base 를 NSMC 2만건으로 파인튜닝
"""

import json
import re
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ★ 모델을 허브에서 불러옵니다. Space 저장소에 모델 파일이 없어도 됩니다.
MODEL_ID = "jisooC/nsmc-sentiment-klue"
MAX_LENGTH = 128
DOMAIN_THRESHOLD = 0.50     # 어휘 일치율이 이 값 미만이면 '도메인 밖'
LOW_CONFIDENCE = 0.75       # 확신도가 이 값 미만이면 '모델이 헤매는 중'

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

# ★ 도메인 판정용 어휘 — NSMC 학습 데이터에서 뽑은 실제 어간
#   토크나이저 사전이 아닙니다. 그걸 쓰면 "한국어인가?"만 재게 됩니다.
VOCAB_FILE = Path(__file__).parent / "domain_vocab.json"
DOMAIN_STEMS = set()
if VOCAB_FILE.exists():
    DOMAIN_STEMS = set(json.loads(VOCAB_FILE.read_text(encoding="utf-8"))["stems"])


def predict(text, threshold, review_band):
    if not text or not text.strip():
        return "문장을 입력해주세요.", {}, "", ""

    text = text.strip()
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0]
    p_pos = float(probs[1])

    pred = 1 if p_pos >= threshold else 0
    label = "긍정 😊" if pred else "부정 😞"
    conf = p_pos if pred else 1 - p_pos

    verdict = f"## {label}\n\n확신도 **{conf:.1%}** · 긍정 확률 {p_pos:.1%}"
    dist = {"긍정": p_pos, "부정": 1 - p_pos}

    notes = []
    if abs(p_pos - threshold) < review_band:
        notes.append("⚠️ **판정 경계선 근처입니다.** 사람이 확인하세요.")

    if conf < LOW_CONFIDENCE:
        notes.append(f"🤔 **확신도가 낮습니다** ({conf:.1%}). "
                     f"학습한 영화 리뷰에서는 보통 87~99%가 나옵니다.")

    words = re.findall(r"[가-힣]{2,}", text)
    ratio = None
    if DOMAIN_STEMS and words:
        hits = sum(1 for w in words if w[:2] in DOMAIN_STEMS)
        ratio = hits / len(words)
        if ratio < DOMAIN_THRESHOLD:
            notes.append(f"🚩 **학습해본 적 없는 종류의 문장입니다** "
                         f"(영화 리뷰 어휘와 {ratio:.0%} 일치). "
                         f"이 모델은 영화 리뷰 전용입니다.")

    n_tokens = len(tokenizer.encode(text))
    if n_tokens > MAX_LENGTH:
        notes.append(f"✂️ **문장이 잘렸습니다** ({n_tokens}→{MAX_LENGTH}토큰)")

    detail = f"토큰 {n_tokens}개 · 임계값 {threshold:.2f} · 긍정확률 {p_pos:.4f}"
    if ratio is not None:
        detail += f" · 도메인 일치 {ratio:.0%}"
    return verdict, dist, "\n\n".join(notes), detail


with gr.Blocks(title="한국어 영화 리뷰 감성 분석") as demo:
    gr.Markdown("""
# 🎬 한국어 영화 리뷰 감성 분석기

`klue/roberta-base` 를 네이버 영화 리뷰(NSMC) 2만 건으로 파인튜닝했습니다.
**검증 정확도 88%** (학습에 쓰지 않은 데이터 2,000건으로 측정)
""")

    with gr.Row():
        with gr.Column():
            text_in = gr.Textbox(label="분석할 문장", lines=4,
                                 placeholder="예: 이건 진짜 내 인생 영화다..")
            with gr.Accordion("⚙️ 고급 설정", open=False):
                th = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="판정 임계값")
                band = gr.Slider(0.0, 0.3, value=0.1, step=0.01, label="사람 검토 밴드")
            with gr.Row():
                btn = gr.Button("분석하기", variant="primary")
                clear = gr.ClearButton(value="지우기")
            gr.Examples(
                examples=[["이건 진짜 내 인생 영화다.. 너무 감동이야."],
                          ["중간에 잠들 뻔했어요. 전개가 너무 느림."],
                          ["나쁘지 않은데 다시 볼 것 같진 않아요."],
                          ["기대했던 것과는 좀 다르네요."]],
                inputs=text_in)

        with gr.Column():
            verdict = gr.Markdown("### 결과가 여기에 표시됩니다")
            prob = gr.Label(label="확률 분포", num_top_classes=2)
            warn = gr.Markdown()
            detail = gr.Textbox(label="상세", interactive=False)

    gr.Markdown("""
---
### ⚠️ 알려진 한계
- **도메인** : 영화 리뷰 전용입니다. 상품평·뉴스는 보장하지 않습니다.
- **우회 표현** : "기대했던 것과는 좀 다르네요" 처럼 부정 어휘가 없는 문장에 약합니다.
- **중립 없음** : 2지선다 모델이라 "모르겠음"을 출력할 수 없습니다.
- **길이** : 128토큰 초과 시 잘립니다.
""")

    ins = [text_in, th, band]
    outs = [verdict, prob, warn, detail]
    btn.click(predict, ins, outs)
    text_in.submit(predict, ins, outs)
    clear.add([text_in, verdict, prob, warn, detail])


if __name__ == "__main__":
    demo.launch()
