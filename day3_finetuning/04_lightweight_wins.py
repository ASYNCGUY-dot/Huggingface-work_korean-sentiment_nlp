"""
=============================================================================
STAGE 3 - MISSION 4 : 닭 잡는 데 소 잡는 칼 쓰지 마라
=============================================================================
방금 우리는 1.78억 파라미터 모델로 239초 학습해서 실전 정답률 80%를 얻었습니다.

같은 데이터를 scikit-learn 으로 처리하면 어떻게 될까요?

    BERT          : 1.78억 파라미터 / 239초 / 문맥 이해 시도
    TF-IDF + NB   : 수천 개 / 1초 미만 / 단어 빈도 통계

⏱️ 30초 이내 (학습이 1초도 안 걸립니다)

실행 : python 04_lightweight_wins.py
=============================================================================
"""

import time

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from _common import MODEL_OUT, MOVIE_CSV, REAL_SENTENCES, TEMPLATE_SENTENCES


# =============================================================================
# [0] 데이터 준비 — BERT와 완전히 동일한 데이터를 씁니다
# =============================================================================
if not MOVIE_CSV.exists():
    print(f"❌ 없습니다: {MOVIE_CSV}\n   STAGE 2의 05_full_pipeline.py 를 먼저 실행하세요.")
    raise SystemExit(1)

df = pd.read_csv(MOVIE_CSV)

# BERT와 같은 8:2 비율, 같은 seed → 공정한 비교
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

print("=" * 78)
print("공정한 비교를 위한 조건")
print("=" * 78)
print(f"  데이터   : {MOVIE_CSV.name} (BERT가 학습한 것과 동일)")
print(f"  분할     : 학습 {len(X_train)}건 / 검증 {len(X_test)}건 (8:2, seed=42)")


# =============================================================================
# [1] 경량 모델 2종 학습
# =============================================================================
print("\n" + "=" * 78)
print("[1] scikit-learn 모델 학습")
print("=" * 78)

MODELS = {
    # ── 단어 단위 (TfidfVectorizer 기본값) ──────────────────────────────
    # ★ 한국어에서는 이게 함정입니다. 이유는 아래 [4]번에서 확인합니다.
    "① 단어단위 + 나이브베이즈": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", MultinomialNB()),
    ]),
    "② 단어단위 + 로지스틱회귀": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]),
    # ── 글자 묶음(character n-gram) 단위 ────────────────────────────────
    # analyzer="char_wb" : 단어 경계 안에서 글자 n개씩 잘라 특징으로 씁니다.
    #   "전개가" → '전개', '개가', '전개가' ... 처럼 조각이 만들어지므로
    #   조사 '가'가 붙어도 '전개' 조각이 살아남습니다.
    #   ★ 한국어처럼 조사·어미가 계속 붙는 언어에서 아주 효과적인 기법입니다.
    "③ 글자묶음 + 나이브베이즈": Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
        ("clf", MultinomialNB()),
    ]),
    "④ 글자묶음 + 로지스틱회귀": Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
        ("clf", LogisticRegression(max_iter=1000)),
    ]),
}

trained = {}
for name, model in MODELS.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    val_acc = accuracy_score(y_test, model.predict(X_test))
    vocab = len(model.named_steps["tfidf"].vocabulary_)

    trained[name] = model
    print(f"\n  {name}")
    print(f"    학습 시간  : {elapsed:.3f}초")
    print(f"    사전 크기  : {vocab}개 단어")
    print(f"    검증 정확도: {val_acc:.1%}")


# =============================================================================
# [2] BERT 불러오기 (이미 학습된 것)
# =============================================================================
bert_ok = MODEL_OUT.exists()
if bert_ok:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_OUT)
    bert = AutoModelForSequenceClassification.from_pretrained(MODEL_OUT)
    bert.eval()

    def bert_predict(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           padding=True, max_length=32)
        with torch.no_grad():
            logits = bert(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        p = int(torch.argmax(probs))
        return p, float(probs[p])
else:
    print("\n⚠️ 학습된 BERT가 없어 경량 모델끼리만 비교합니다.")


# =============================================================================
# [3] 같은 시험지로 대결
# =============================================================================
def score_sklearn(model, cases):
    texts = [t for t, _ in cases]
    answers = [a for _, a in cases]
    preds = model.predict(texts)
    return sum(p == a for p, a in zip(preds, answers)) / len(cases)


def score_bert(cases):
    return sum(bert_predict(t)[0] == a for t, a in cases) / len(cases)


print("\n\n" + "=" * 78)
print("[2] 같은 시험지로 대결")
print("=" * 78)

results = []
for name, model in trained.items():
    results.append((name,
                    score_sklearn(model, TEMPLATE_SENTENCES),
                    score_sklearn(model, REAL_SENTENCES)))
if bert_ok:
    results.append(("BERT (1.78억 파라미터)",
                    score_bert(TEMPLATE_SENTENCES),
                    score_bert(REAL_SENTENCES)))

print(f"\n{'모델':<26} {'A조(템플릿)':>12} {'B조(실전)':>12}")
print("-" * 78)
for name, a, b in results:
    print(f"{name:<26} {a:>11.0%} {b:>11.0%}")


# =============================================================================
# [4] B조 문장별 상세 비교
# =============================================================================
print("\n" + "=" * 78)
print("[3] B조(실전 문장) 문장별 비교")
print("=" * 78)

best_sk_name = max(trained, key=lambda n: score_sklearn(trained[n], REAL_SENTENCES))
best_sk = trained[best_sk_name]

print(f"\n  비교 대상: {best_sk_name}  vs  BERT\n")
for text, ans in REAL_SENTENCES:
    sk_pred = int(best_sk.predict([text])[0])
    sk_ok = "✅" if sk_pred == ans else "❌"

    if bert_ok:
        bt_pred, _ = bert_predict(text)
        bt_ok = "✅" if bt_pred == ans else "❌"
        line = f"  경량 {sk_ok}  BERT {bt_ok}   {text}"
        # 둘의 답이 다른 경우 표시
        if sk_pred != bt_pred:
            line += "   ← 답이 갈림"
    else:
        line = f"  경량 {sk_ok}   {text}"
    print(line)


# =============================================================================
# [5] 결론
# =============================================================================
print("\n" + "=" * 78)
print("[4] ★ 한국어의 함정 — 조사 하나가 단어를 갈라놓는다")
print("=" * 78)

word_vec = trained["① 단어단위 + 나이브베이즈"].named_steps["tfidf"]
char_vec = trained["③ 글자묶음 + 나이브베이즈"].named_steps["tfidf"]

print(f"\n  단어 단위 사전 크기 : {len(word_vec.vocabulary_):>5}개")
print(f"  글자 묶음 사전 크기 : {len(char_vec.vocabulary_):>5}개")

print("\n  같은 뜻인데 사전에 따로 잡히는가?")
for a, b in [("전개", "전개가"), ("시간", "시간에"), ("실망", "실망도"), ("기대", "기대가")]:
    ina = a in word_vec.vocabulary_
    inb = b in word_vec.vocabulary_
    print(f"    '{a}' {'✅' if ina else '❌'}   '{b}' {'✅' if inb else '❌'}"
          f"   → {'같은 단어로 인식 못 함 🚨' if ina != inb else '동일 처리'}")

print("""
  ★ 이것이 한국어 NLP의 고전적 난제입니다.

    영어 : stock / stocks       → 두 형태뿐
    한국어 : 전개 / 전개가 / 전개는 / 전개를 / 전개도 / 전개까지 ...
             조사와 어미가 무한히 붙습니다(교착어).

    TfidfVectorizer 의 기본 토크나이저는 '띄어쓰기 + 완전 일치'입니다.
    → '전개'를 학습했어도 '전개가'는 처음 보는 단어가 됩니다.

  ✅ 해결책 3가지
     1) character n-gram  : analyzer="char_wb"  ← 오늘 ③④번이 쓴 방법. 설치 불필요
     2) 형태소 분석기      : KoNLPy, Mecab, Okt 등으로 어간만 추출
     3) 서브워드 토크나이저 : BERT 계열이 원래부터 하는 방식
""")


# =============================================================================
# [5] 결론 — 실측 결과에 근거해서
# =============================================================================
print("=" * 78)
print("결론")
print("=" * 78)

best_name, best_a, best_b = max(results, key=lambda r: r[2])
bert_row = next((r for r in results if r[0].startswith("BERT")), None)

print(f"\n  이번 실험의 B조(실전) 1위 : {best_name}  {best_b:.0%}")
if bert_row:
    sk_best = max((r for r in results if not r[0].startswith("BERT")), key=lambda r: r[2])
    print(f"  경량 모델 최고            : {sk_best[0]}  {sk_best[2]:.0%}")
    print(f"  BERT                     : {bert_row[2]:.0%}")

    if bert_row[2] > sk_best[2]:
        print("""
  ★ BERT가 이겼습니다. 이유는 두 가지입니다.

    ① 사전학습 지식 (전이학습의 본질)
       BERT는 이미 위키피디아로 한국어를 배운 상태로 시작했습니다.
       우리 데이터 600개에 없던 단어도 '어느 정도는' 압니다.
       경량 모델은 백지에서 시작하므로, 사전에 없는 단어 = 정보 0입니다.

    ② 서브워드 토크나이징
       '재밌더라구요'를 BERT는 '재밌'+'더라'+'구요' 로 쪼갭니다.
       학습 때 본 '재밌어요'와 '재밌' 조각을 공유하죠.
       단어 단위 TF-IDF에게 이 둘은 완전히 남남입니다.

    💡 즉, 데이터가 부실할수록 '사전학습된 지식'의 가치가 커집니다.
       이게 우리가 처음부터 모델을 만들지 않고 파인튜닝을 하는 이유입니다.
""")
    else:
        print("""
  ★ 경량 모델이 BERT를 따라잡았거나 이겼습니다.
    데이터에 문맥 정보가 부족하면 1.78억 파라미터가 놀게 됩니다.
""")

print("""
  💼 실무 판단표 (오늘 실측 결과 반영)

    상황                              추천
    ────────────────────────────────────────────────────────
    데이터 수백 건 + 영어             경량 모델로 baseline 먼저
    데이터 수백 건 + 한국어           경량 모델 쓰되 char n-gram 필수
    데이터 수천 건 이상               BERT 파인튜닝
    데이터가 부실하고 다양성이 낮음    ★ 사전학습 모델(BERT)이 유리
    설명 가능성이 중요                경량 모델 (어떤 단어가 기여했는지 볼 수 있음)
    속도·비용이 중요                  경량 모델 (학습 0.005초 vs 239초)

  ★ 그래도 원칙은 그대로입니다.
    경량 모델로 기준선을 먼저 만드세요. 5초면 됩니다.
    딥러닝이 그 기준선을 크게 못 이기면, 문제는 모델이 아니라 데이터입니다.
""")

print("=" * 78)
print("→ 05_boss_real_data.py : 진짜 사람이 쓴 데이터로 다시 도전합니다")
print("=" * 78)


# =============================================================================
# 🎯 미션
# =============================================================================
# 1. 두 경량 모델(NB vs 로지스틱회귀) 중 어느 쪽이 나은가요?
#
# 2. '답이 갈림' 표시가 붙은 문장을 보세요. 어느 쪽이 맞았나요?
#
# 3. ★ TfidfVectorizer 에 ngram_range=(1,2) 를 넣어보세요.
#       TfidfVectorizer(ngram_range=(1, 2))
#    단어 하나가 아니라 '두 단어 묶음'까지 봅니다.
#    "안 좋다" 같은 표현을 잡을 수 있게 되어 성능이 오를 수 있습니다.
#
# 4. ★★ 어느 단어가 판단에 크게 기여했는지 보기 (로지스틱회귀 전용)
#
#       import numpy as np
#       m = trained["④ 글자묶음 + 로지스틱회귀"]
#       names = m.named_steps["tfidf"].get_feature_names_out()
#       coef = m.named_steps["clf"].coef_[0]
#       top_pos = np.argsort(coef)[-10:][::-1]
#       top_neg = np.argsort(coef)[:10]
#       print("긍정 단어:", [names[i] for i in top_pos])
#       print("부정 단어:", [names[i] for i in top_neg])
#
#    → 경량 모델의 장점: 왜 그렇게 판단했는지 '설명'이 가능합니다.
#      BERT는 이게 거의 불가능합니다. 실무에서 중요한 차이입니다.
# =============================================================================
