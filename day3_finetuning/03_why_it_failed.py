"""
=============================================================================
STAGE 3 - MISSION 3 : 부검 — 왜 어떤 문장은 맞고 어떤 문장은 틀렸나
=============================================================================
02번에서 B조 성적이 A조보다 낮게 나왔습니다.
그런데 "그냥 못한다"가 아니라, 틀리는 데 규칙이 있습니다.

이번 미션은 그 규칙을 데이터로 증명합니다.

    가설 : "모델은 학습 데이터에서 본 단어가 있을 때만 제대로 판단한다"

⏱️ 추론만 하므로 빠릅니다 (10초 이내)

실행 : python 03_why_it_failed.py
=============================================================================
"""

import re

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from _common import MODEL_OUT, MOVIE_CSV, REAL_SENTENCES, ttr


# =============================================================================
# [0] 준비
# =============================================================================
for path, msg in [(MODEL_OUT, "01_train_first_model.py"),
                  (MOVIE_CSV, "STAGE 2의 05_full_pipeline.py")]:
    if not path.exists():
        print(f"❌ 없습니다: {path}\n   {msg} 를 먼저 실행하세요.")
        raise SystemExit(1)

tokenizer = AutoTokenizer.from_pretrained(MODEL_OUT)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_OUT)
model.eval()

train_df = pd.read_csv(MOVIE_CSV)


# =============================================================================
# [1] 학습 데이터의 어휘 목록 만들기
# =============================================================================
print("=" * 78)
print("[1] 모델이 본 적 있는 단어 목록")
print("=" * 78)

# 한글 2글자 이상 덩어리만 추출 (조사/어미 변화를 감안해 어간 위주로 비교)
train_words = set()
for s in train_df["content"]:
    train_words.update(re.findall(r"[가-힣]{2,}", str(s)))

ratio, uniq, total = ttr(train_df["content"])
print(f"\n  학습 문장 : {len(train_df)}개")
print(f"  고유 어절 : {uniq}개")
print(f"  어휘 다양성(TTR) : {ratio:.3f}")

# ★ 한국어에서 가장 기본적인 감정 단어들이 학습 데이터에 있는지 확인
BASIC_WORDS = ["좋", "싫", "눈물", "감동", "재밌", "행복", "슬프", "최고", "별로"]
joined = " ".join(str(s) for s in train_df["content"])

print(f"\n  📖 기본 감정 단어가 학습 데이터에 있는가?")
for w in BASIC_WORDS:
    present = w in joined
    print(f"    '{w}' {'✅ 있음' if present else '❌ 없음'}")


# =============================================================================
# [2] 예측 + 어휘 겹침 분석
# =============================================================================
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=32)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0]
    p = int(torch.argmax(probs))
    return p, float(probs[p])


def overlap(sent):
    """테스트 문장의 어절이 학습 어휘와 앞 2글자라도 겹치는지 찾습니다.

    왜 앞 2글자만 보나?
      한국어는 어미가 계속 바뀝니다. 감동적입니다 / 감동이야 / 감동했어요.
      어간('감동')만 같으면 모델 입장에서는 비슷한 신호입니다.
      완전 일치만 세면 실제 겹침을 과소평가하게 됩니다.
    """
    hits = []
    for w in re.findall(r"[가-힣]+", sent):
        if len(w) < 2:
            continue
        for t in train_words:
            if w[:2] == t[:2]:
                hits.append(f"{w}↔{t}")
                break
    return hits


print("\n\n" + "=" * 78)
print("[2] 문장별 부검")
print("=" * 78)

rows = []
for text, answer in REAL_SENTENCES:
    pred, conf = predict(text)
    hits = overlap(text)
    correct = pred == answer

    rows.append({
        "문장": text,
        "정답": "긍정" if answer else "부정",
        "예측": "긍정" if pred else "부정",
        "맞춤": correct,
        "확신": conf,
        "겹침수": len(hits),
    })

    print(f"\n  {'✅' if correct else '❌'} {text}")
    print(f"     정답 {'긍정' if answer else '부정'} / 예측 {'긍정' if pred else '부정'} ({conf:.1%})")
    print(f"     학습 어휘와 겹침 : {len(hits)}개  {hits if hits else '── 없음 ──'}")


# =============================================================================
# [3] ★ 결론 — 겹침 여부로 갈리는가
# =============================================================================
df = pd.DataFrame(rows)
has = df[df["겹침수"] > 0]
none = df[df["겹침수"] == 0]

print("\n\n" + "=" * 78)
print("[3] 가설 검증")
print("=" * 78)

print(f"\n{'구분':<28} {'문장수':>6} {'정답률':>8} {'평균확신':>9}")
print("-" * 78)
if len(has):
    print(f"{'학습 어휘와 겹침 있음':<24} {len(has):>6} {has['맞춤'].mean():>7.0%} {has['확신'].mean():>8.1%}")
if len(none):
    print(f"{'학습 어휘와 겹침 없음':<24} {len(none):>6} {none['맞춤'].mean():>7.0%} {none['확신'].mean():>8.1%}")

print(f"""
  📊 해석

    겹침이 있으면 → 모델이 아는 단어가 있으니 제대로 판단합니다.
    겹침이 없으면 → 아는 단어가 하나도 없어 사실상 찍습니다.
                    2지선다에서 찍으면 정답률 50% 근처가 나옵니다.

  ★ 그런데 확신도(확률)는 겹침 여부와 상관없이 높습니다.
    모델은 "나 이거 처음 봐서 모르겠는데"라고 말하지 못합니다.
    STAGE 1 첫날 배운 그 성질입니다. 여기서 또 만났네요.
""")

wrong = df[~df["맞춤"]]
if len(wrong):
    print("  🚨 틀린 문장들의 공통점:")
    for _, r in wrong.iterrows():
        print(f"     - {r['문장']}  (겹침 {r['겹침수']}개, 확신 {r['확신']:.0%})")
    print(f"\n     → 겹침 평균 {wrong['겹침수'].mean():.1f}개")


# =============================================================================
# [4] 결정적 실험 — 단어 하나만 바꿔보기
# =============================================================================
print("\n" + "=" * 78)
print("[4] 결정적 실험 : 뜻은 같은데 단어만 바꾸면?")
print("=" * 78)

PAIRS = [
    # (문장A, 문장B, 정답)  — 두 문장은 뜻이 같고 핵심 단어만 다릅니다
    ("이 영화 감동적입니다.",   "이 영화 뭉클했습니다.",     1),
    ("스토리 지루했어요.",       "스토리 따분했어요.",       0),
    ("연출 훌륭합니다.",         "연출 근사합니다.",         1),
    ("시간 아까워요.",           "시간 낭비였어요.",         0),
]

# ★ 학습 어휘 포함 여부를 손으로 적지 않고 '자동 판정'합니다.
#   (2026-08-05 수정: 처음엔 O/X를 하드코딩했는데, '낭비'가 실제로는
#    학습 데이터의 "시간 낭비입니다."에 들어 있어서 라벨이 틀렸습니다.
#    사람의 기억 대신 데이터에 물어보도록 바꿨습니다.)
def in_train_vocab(sent):
    hits = overlap(sent)
    return len(hits) > 0, hits

print(f"\n{'문장':<24} {'예측':>5} {'확신':>8} {'어휘':>6}  판정")
print("-" * 78)
pair_results = []
for a, b, ans in PAIRS:
    row = []
    for sent in (a, b):
        p, c = predict(sent)
        known, hits = in_train_vocab(sent)
        ok = p == ans
        print(f"{sent:<24} {'긍정' if p else '부정':>5} {c:>7.1%} "
              f"{'있음' if known else '없음':>6}  {'✅' if ok else '❌'}")
        row.append((sent, p, c, known, ok))
    # 두 문장의 예측이 뒤집혔는가?
    flipped = row[0][1] != row[1][1]
    print(f"{'  → 예측이 뒤집혔나?':<24} {'🔄 예 (같은 뜻인데 답이 다름)' if flipped else '유지'}")
    pair_results.append((row, flipped))
    print()

# 요약 — 학습 어휘가 '없는' 문장만 모아서 정답률 계산
unknown_rows = [r for row, _ in pair_results for r in row if not r[3]]
if unknown_rows:
    acc = sum(r[4] for r in unknown_rows) / len(unknown_rows)
    print(f"  📊 학습 어휘가 없는 문장 {len(unknown_rows)}개의 정답률 : {acc:.0%}")
    for sent, p, c, _, ok in unknown_rows:
        print(f"     {'✅' if ok else '❌'} {sent}  → {'긍정' if p else '부정'} ({c:.0%})")

print("""
  ★ 같은 뜻인데 단어만 바꿨더니 답이 뒤집혔다면,
    모델은 '의미'가 아니라 '글자'를 보고 있는 것입니다.

  💼 실무 결론
     이 모델은 서비스에 올리면 안 됩니다.
     사용자는 학습 템플릿대로 말해주지 않습니다.

     해결책 우선순위:
       1위. 데이터를 다양하게 (어휘 다양성 확보)   ← 압도적 1위
       2위. 데이터를 많이
       3위. 모델/하이퍼파라미터 조정               ← 효과가 가장 작음

     ★ 초보자는 3위부터 손댑니다. 실무자는 1위부터 봅니다.
""")

print("=" * 78)
print("→ 04_lightweight_wins.py : 딥러닝 말고 다른 길도 있습니다")
print("=" * 78)


# =============================================================================
# 🎯 미션
# =============================================================================
# 1. BASIC_WORDS 목록에서 '❌ 없음' 이 나온 단어들을 확인하세요.
#    한국어에서 가장 흔한 감정 단어가 학습 데이터에 없다면, 그 모델은
#    실제 사용자 문장의 절반도 이해하지 못합니다.
#
# 2. PAIRS 에 본인이 만든 쌍을 3개 추가해보세요.
#    (학습 어휘를 쓴 문장 / 같은 뜻인데 다른 단어를 쓴 문장)
#
# 3. ★ STAGE 2의 data/movie.csv 를 열어 실제로 '좋' 을 검색해보세요.
#    (VS Code에서 Ctrl+F) 몇 건이 나오나요?
# =============================================================================
