"""
=============================================================================
STAGE 5 - MISSION 3 : 배포 패키지 만들기
=============================================================================
로컬에서 잘 도는 앱과 배포 가능한 앱은 다릅니다.

    로컬 : 내 컴퓨터 경로를 참조해도 됨 / 버전 고정이 안전 / 크기 무관
    배포 : 경로가 없음 / 버전 고정이 위험 / 크기 제한 있음

이 스크립트는 배포용 폴더를 자동으로 만들어줍니다.

    deploy_package/
      ├─ app.py            ← Spaces 가 찾는 이름. 반드시 이거여야 함
      ├─ requirements.txt  ← 배포용 (로컬과 다릅니다!)
      └─ README.md         ← Spaces 설정을 담는 특수 파일

★ 핵심 결정: 443MB 모델을 어떻게 배포할 것인가
   → 이 스크립트가 두 가지 방법을 안내합니다.

실행 : python 03_prepare_deploy.py
=============================================================================
"""

import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE.parent / "day3_finetuning" / "data" / "colab_model"
OUT = BASE / "deploy_package"

# ⚙️ 여기를 본인 것으로 바꾸세요
HF_USERNAME = "jisooC"                     # 허깅페이스 사용자명
MODEL_REPO = "nsmc-sentiment-klue"          # 모델 저장소 이름
SPACE_NAME = "korean-sentiment"             # Space 이름


# =============================================================================
# [1] 모델 크기 점검 — 배포 방식이 여기서 갈립니다
# =============================================================================
print("=" * 78)
print("[1] 모델 크기 점검")
print("=" * 78)

if not MODEL_DIR.exists():
    raise SystemExit(f"❌ 모델이 없습니다: {MODEL_DIR}")

files = sorted(MODEL_DIR.iterdir(), key=lambda p: -p.stat().st_size)
total = sum(f.stat().st_size for f in files) / 1e6

print()
for f in files:
    mb = f.stat().st_size / 1e6
    flag = "  🚨 100MB 초과" if mb > 100 else ""
    print(f"  {f.name:<28} {mb:>8.1f} MB{flag}")
print(f"  {'합계':<28} {total:>8.1f} MB")

print(f"""
  ⚠️ 크기 문제
     GitHub          : 파일 하나당 100MB 제한
     허깅페이스 Spaces : Git LFS 를 쓰면 올라가지만 느리고 관리가 번거로움

  ★ 그래서 실무에서는 이렇게 합니다.

     ❌ Space 저장소에 모델 파일을 통째로 넣기
     ✅ 모델은 Hugging Face '모델 허브'에 따로 올리고,
        app.py 에서는 이름으로 불러오기

            pipeline(model="{HF_USERNAME}/{MODEL_REPO}")

     장점 : Space 는 코드만 담아 가볍고, 모델은 여러 앱이 공유합니다.
            모델만 업데이트하고 앱은 그대로 둘 수도 있습니다.
""")


# =============================================================================
# [2] app.py 생성
# =============================================================================
print("=" * 78)
print("[2] 배포 패키지 생성")
print("=" * 78)

OUT.mkdir(exist_ok=True)

APP_PY = f'''"""
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
MODEL_ID = "{HF_USERNAME}/{MODEL_REPO}"
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
        return "문장을 입력해주세요.", {{}}, "", ""

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

    verdict = f"## {{label}}\\n\\n확신도 **{{conf:.1%}}** · 긍정 확률 {{p_pos:.1%}}"
    dist = {{"긍정": p_pos, "부정": 1 - p_pos}}

    notes = []
    if abs(p_pos - threshold) < review_band:
        notes.append("⚠️ **판정 경계선 근처입니다.** 사람이 확인하세요.")

    if conf < LOW_CONFIDENCE:
        notes.append(f"🤔 **확신도가 낮습니다** ({{conf:.1%}}). "
                     f"학습한 영화 리뷰에서는 보통 87~99%가 나옵니다.")

    words = re.findall(r"[가-힣]{{2,}}", text)
    ratio = None
    if DOMAIN_STEMS and words:
        hits = sum(1 for w in words if w[:2] in DOMAIN_STEMS)
        ratio = hits / len(words)
        if ratio < DOMAIN_THRESHOLD:
            notes.append(f"🚩 **학습해본 적 없는 종류의 문장입니다** "
                         f"(영화 리뷰 어휘와 {{ratio:.0%}} 일치). "
                         f"이 모델은 영화 리뷰 전용입니다.")

    n_tokens = len(tokenizer.encode(text))
    if n_tokens > MAX_LENGTH:
        notes.append(f"✂️ **문장이 잘렸습니다** ({{n_tokens}}→{{MAX_LENGTH}}토큰)")

    detail = f"토큰 {{n_tokens}}개 · 임계값 {{threshold:.2f}} · 긍정확률 {{p_pos:.4f}}"
    if ratio is not None:
        detail += f" · 도메인 일치 {{ratio:.0%}}"
    return verdict, dist, "\\n\\n".join(notes), detail


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
'''

(OUT / "app.py").write_text(APP_PY, encoding="utf-8")
print(f"\n  ✅ app.py")
print("     ★ 파일명이 반드시 app.py 여야 합니다. Spaces 가 이 이름을 찾습니다.")

# ★ 도메인 어휘 사전도 함께 복사 — 없으면 🚩 경고가 조용히 꺼집니다
VOCAB_SRC = BASE / "domain_vocab.json"
if VOCAB_SRC.exists():
    shutil.copy(VOCAB_SRC, OUT / "domain_vocab.json")
    kb = VOCAB_SRC.stat().st_size / 1024
    print(f"  ✅ domain_vocab.json ({kb:.0f} KB)")
    print("     도메인 밖 경고에 필요합니다. 안 넣으면 경고가 조용히 사라집니다.")
else:
    print("  ⚠️ domain_vocab.json 이 없습니다!")
    print("     → python build_domain_vocab.py 를 먼저 실행하세요.")
    print("     지금 배포하면 🚩 도메인 경고가 작동하지 않습니다.")


# =============================================================================
# [3] requirements.txt — 로컬과 정반대 전략
# =============================================================================
REQ = """gradio
transformers
torch
"""
(OUT / "requirements.txt").write_text(REQ, encoding="utf-8")

print(f"""
  ✅ requirements.txt
     내용: gradio / transformers / torch  (버전 없음)

     🤔 로컬에서는 transformers==4.44.2 로 고정했는데 왜 여기는 뺐을까요?

        로컬 : "어제 되던 게 오늘도 돼야" 한다 → 버전 고정 (재현성)
        배포 : "그 서버 환경에서 돌아가는 조합"을 찾아야 한다 → 느슨하게 (호환성)

     ★ 5일차 수업 노트에도 기록되어 있습니다.
       버전을 고정했다가 빌드가 계속 실패했고,
       "transformers" 처럼 버전명을 지우니 성공했습니다.
""")


# =============================================================================
# [4] README.md — Spaces 의 설정 파일
# =============================================================================
# ★ 2026-08-06 수정 — title 은 반드시 영숫자/-/_/. 만 쓰세요
#
#   `gradio deploy` 는 이 title 을 '저장소 ID' 로 그대로 씁니다.
#   한글이나 공백이 들어가면 이 에러가 납니다:
#     HFValidationError: Repo id must use alphanumeric chars, '-', '_' or '.'
#
#   ★ 한 필드가 두 용도(화면 제목 / 저장소 ID)로 쓰이므로,
#     제약이 더 강한 쪽(저장소 ID) 규칙을 따라야 합니다.
#
#   보이는 제목은 app.py 안의 gr.Markdown 이 담당하므로 손해가 없습니다.
README = f"""---
title: {SPACE_NAME}
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

# 한국어 영화 리뷰 감성 분석기

`klue/roberta-base` 를 네이버 영화 리뷰(NSMC) 20,000건으로 파인튜닝한 모델입니다.

## 성능

| 지표 | 값 |
|---|---|
| 정확도 | 88.0% (±1.4%) |
| 정밀도 | 0.870 |
| 재현율 | 0.884 |
| F1 | 0.877 |

NSMC test split 2,000건으로 측정했습니다. 학습에 사용되지 않은 데이터입니다.

## 한계

- 영화 리뷰 도메인 전용입니다.
- 부정 어휘가 없는 우회 표현에 약합니다.
- 2지선다 모델이라 "중립"을 출력할 수 없습니다.
- 128토큰을 초과하면 문장이 잘립니다.

## 모델

- 베이스: [klue/roberta-base](https://huggingface.co/klue/roberta-base)
- 파인튜닝 모델: [{HF_USERNAME}/{MODEL_REPO}](https://huggingface.co/{HF_USERNAME}/{MODEL_REPO})
- 데이터: [NSMC](https://huggingface.co/datasets/e9t/nsmc)
"""

(OUT / "README.md").write_text(README, encoding="utf-8")

print("""  ✅ README.md
     ★ 맨 위 --- 사이의 부분이 'YAML front matter' 입니다.
       Spaces 는 이걸 읽어서 SDK 종류, 실행할 파일, 썸네일 색을 정합니다.
       이게 없으면 Space 가 어떻게 실행할지 몰라 빌드가 실패합니다.

       app_file: app.py   ← 이 줄이 특히 중요합니다
""")


# =============================================================================
# [5] .gitignore
# =============================================================================
(OUT / ".gitignore").write_text(
    "__pycache__/\n*.pyc\n.env\n*.pkl\nflagged/\n", encoding="utf-8"
)
print("  ✅ .gitignore")


# =============================================================================
# [6] 완료 안내
# =============================================================================
print("\n" + "=" * 78)
print("배포 패키지 준비 완료")
print("=" * 78)

sizes = {p.name: p.stat().st_size for p in OUT.iterdir() if p.is_file()}
print()
for name, size in sorted(sizes.items()):
    print(f"  {name:<22} {size/1024:>7.1f} KB")
print(f"  {'합계':<22} {sum(sizes.values())/1024:>7.1f} KB")

print(f"""
  ★ 전체가 {sum(sizes.values())/1024:.0f}KB 입니다. 모델 443MB 는 포함되지 않았습니다.
    모델은 허브에 따로 올리기 때문입니다.

  📄 다음 단계는 04_DEPLOY_GUIDE.md 를 열어 따라 하세요.
     ① 모델을 허브에 업로드
     ② Space 생성
     ③ 이 폴더를 git push
""")


# =============================================================================
# 🎯 미션
# =============================================================================
# 1. deploy_package/app.py 를 열어보세요. 02번 파일과 뭐가 다른가요?
#    (힌트: 모델을 불러오는 줄)
#
# 2. README.md 의 YAML 부분에서 emoji 와 colorFrom 을 바꿔보세요.
#
# 3. ★ 로컬에서 배포판을 미리 테스트하려면:
#       cd deploy_package
#       python app.py
#    → 아직 허브에 모델이 없으니 에러가 납니다. 그게 정상입니다.
#      04번 가이드로 모델을 올린 뒤 다시 해보세요.
# =============================================================================
