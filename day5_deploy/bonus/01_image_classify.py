"""
=============================================================================
BONUS 1 : 이미지 분류 + 설명 생성 (ViT + BLIP)
=============================================================================
5일차 수업의 2-1.googlevitmodel.py, 2-3.imagedescription.py 를 합쳤습니다.

★ 텍스트와 이미지는 놀랍도록 같은 방식으로 처리됩니다

    텍스트 : 문장 → 토큰으로 분할 → 숫자 ID → 텐서 → Transformer
    이미지 : 사진 → 패치(16×16)로 분할 → 벡터 → 텐서 → Transformer
                    ↑ 패치가 '단어' 역할을 합니다

    google/vit-base-patch16-224 이름의 뜻
      224×224 이미지를 16×16 패치로 자름 → 196개 패치 → 단어처럼 취급

⏱️ 첫 실행 시 모델 다운로드
    ViT  약 350MB
    BLIP 약 990MB   ← 설명 생성 기능을 켤 때만 받습니다

실행 : python 01_image_classify.py
=============================================================================
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import gradio as gr
import torch
from PIL import Image

BASE = Path(__file__).resolve().parent
SAMPLES = BASE / "samples" / "images"

# ⚙️ BLIP(설명 생성)은 990MB입니다. 용량이 부담되면 False 로 끄세요.
USE_BLIP = True


# =============================================================================
# ★ 지연 로딩(lazy loading) — 실무에서 중요한 패턴
# =============================================================================
# 모델을 파일 맨 위에서 다 불러오면, 앱 시작에 몇 분이 걸리고
# 쓰지도 않을 모델까지 메모리를 차지합니다.
# → '처음 필요할 때' 불러오고, 그다음부터는 재사용합니다.
_cache = {}


def get_vit():
    if "vit" not in _cache:
        print("ViT 모델 로딩 중... (첫 실행은 350MB 다운로드)")
        from transformers import ViTForImageClassification, ViTImageProcessor
        name = "google/vit-base-patch16-224"
        _cache["vit"] = (
            ViTImageProcessor.from_pretrained(name),
            ViTForImageClassification.from_pretrained(name).eval(),
        )
        print("완료")
    return _cache["vit"]


def get_blip():
    if "blip" not in _cache:
        print("BLIP 모델 로딩 중... (첫 실행은 990MB 다운로드)")
        from transformers import BlipForConditionalGeneration, BlipProcessor
        name = "Salesforce/blip-image-captioning-base"
        _cache["blip"] = (
            BlipProcessor.from_pretrained(name),
            BlipForConditionalGeneration.from_pretrained(name).eval(),
        )
        print("완료")
    return _cache["blip"]


# =============================================================================
# 분석 함수
# =============================================================================
def analyze(image, top_k, want_caption):
    if image is None:
        return {}, "이미지를 업로드해주세요.", ""

    # Gradio 가 numpy 로 주므로 PIL 로 변환합니다.
    # ★ 이미 PIL 인 경우도 있어서 검사가 필요합니다 (수업 노트의 그 에러)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")     # 흑백/투명 PNG 대비

    # ---- ViT 분류 -------------------------------------------------------
    processor, model = get_vit()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    k = int(top_k)
    top_p, top_i = torch.topk(probs, k)
    results = {model.config.id2label[i.item()]: float(p)
               for p, i in zip(top_p, top_i)}

    best_label = model.config.id2label[int(top_i[0])]
    best_prob = float(top_p[0])

    summary = f"## {best_label}\n\n확신도 **{best_prob:.1%}**"

    # ★ 이 모델은 ImageNet 1,000개 클래스만 압니다.
    #   그 밖의 것(한글 간판, 그래프, 문서)이 들어오면 억지로 1,000개 중에 고릅니다.
    #   STAGE 1~5 내내 나온 그 문제입니다.
    if best_prob < 0.30:
        summary += ("\n\n🚩 **확신도가 낮습니다.** 이 모델은 ImageNet 1,000개 "
                    "카테고리(동물·사물 위주)만 압니다. 그 밖의 이미지는 억지로 "
                    "가장 비슷한 것을 고릅니다.")

    # ---- BLIP 설명 ------------------------------------------------------
    caption = ""
    if want_caption and USE_BLIP:
        bp, bm = get_blip()
        binputs = bp(images=image, return_tensors="pt")
        with torch.no_grad():
            out = bm.generate(**binputs, max_new_tokens=30)
        caption = bp.decode(out[0], skip_special_tokens=True)
        caption = f"**BLIP 설명**: {caption}\n\n(영어로 생성됩니다)"

    return results, summary, caption


# =============================================================================
# UI
# =============================================================================
examples = []
if SAMPLES.exists():
    for name in ["cat.jpg", "dog.png", "car.jpg", "building.png",
                 "mangwon_bakery.jpg", "oecd_rnd_2022.png"]:
        p = SAMPLES / name
        if p.exists():
            examples.append([str(p)])

with gr.Blocks(title="이미지 분류 + 설명") as demo:
    gr.Markdown("""
# 🖼️ 이미지 분류 + 설명 생성

| 모델 | 역할 |
|---|---|
| `google/vit-base-patch16-224` | 이미지를 1,000개 카테고리로 분류 |
| `Salesforce/blip-image-captioning-base` | 이미지를 문장으로 설명 |
""")

    with gr.Row():
        with gr.Column():
            img = gr.Image(label="이미지", type="pil",
                           sources=["upload", "clipboard"])
            with gr.Accordion("⚙️ 설정", open=False):
                topk = gr.Slider(1, 10, value=5, step=1, label="상위 몇 개 표시")
                cap = gr.Checkbox(value=True, label="BLIP 설명 생성 (첫 실행 990MB)")
            btn = gr.Button("분석하기", variant="primary")
            if examples:
                gr.Examples(examples=examples, inputs=img,
                            label="샘플 (마지막 것은 그래프 — 모델이 약한 유형)")

        with gr.Column():
            summary_out = gr.Markdown("### 결과가 여기에 표시됩니다")
            label_out = gr.Label(label="분류 확률")
            cap_out = gr.Markdown()

    gr.Markdown("""
---
### 💡 관찰 포인트

1. **고양이·강아지·자동차** → 잘 맞힙니다. ImageNet 이 그런 걸로 학습됐거든요.
2. **그래프·문서·한글 간판** → 확신도가 뚝 떨어집니다.
   이 모델은 **1,000개 카테고리 밖의 개념이 존재하지 않습니다.**
   "모르겠다"를 출력할 수 없어 억지로 가장 비슷한 걸 고릅니다.
3. **BLIP 설명**은 영어로 나옵니다. 분류(1,000개 중 택1)와 달리
   **문장을 생성**하므로 훨씬 유연합니다.

> 🔗 STAGE 1에서 텍스트 모델이 `asdf qwer zxcv` 에 99.5% 확신을 준 것과
> 정확히 같은 구조입니다. **모달리티가 바뀌어도 성질은 같습니다.**
""")

    btn.click(analyze, [img, topk, cap], [label_out, summary_out, cap_out])


if __name__ == "__main__":
    demo.launch(inbrowser=True)


# =============================================================================
# 🎯 미션
# =============================================================================
# 1. cat.jpg, dog.png 를 넣어보세요. 품종까지 맞히나요?
#
# 2. ★ oecd_rnd_2022.png (그래프 이미지)를 넣어보세요.
#    확신도가 몇 %인가요? 🚩 경고가 뜨나요?
#
# 3. ★★ 본인 사진이나 한글 간판 사진을 넣어보세요.
#    ImageNet 1,000개 안에 없는 것을 어떻게 처리하나요?
#
# 4. ★★★ ViT 결과와 BLIP 설명을 비교하세요.
#    같은 이미지인데 왜 다른 이야기를 할까요?
#    (힌트: 하나는 '분류', 하나는 '생성' 입니다)
# =============================================================================
