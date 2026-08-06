"""
=============================================================================
STAGE 4 - MISSION 3 : 정확도 99%짜리 쓰레기 모델 만들기
=============================================================================
"우리 모델 정확도 99%입니다."

이 말을 들으면 실무자는 박수 대신 질문을 합니다.
    "데이터의 클래스 비율이 어떻게 되나요?"

이번 미션에서는 ★ 아무것도 학습하지 않고 정확도 99%를 받는 모델을
직접 만들어봅니다. 그리고 그게 왜 쓰레기인지 숫자로 봅니다.

⏱️ 학습 없음. 즉시 실행.

실행 : python 03_accuracy_trap.py
=============================================================================
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score,
                             precision_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from _common import DATA, print_confusion

import matplotlib.pyplot as plt

rng = np.random.default_rng(42)


# =============================================================================
# [1] 불균형 데이터 만들기 — 현실의 대부분이 이렇습니다
# =============================================================================
print("=" * 78)
print("[1] 희귀 질환 검진 데이터 (환자 1%)")
print("=" * 78)

N = 10000
POSITIVE_RATE = 0.01        # 실제 환자 비율 1%

y = (rng.random(N) < POSITIVE_RATE).astype(int)
n_pos, n_neg = y.sum(), N - y.sum()

print(f"""
  전체     {N:,}명
  환자     {n_pos:,}명 ({n_pos/N:.2%})
  정상     {n_neg:,}명 ({n_neg/N:.2%})

  ★ 이런 걸 '불균형 데이터(imbalanced data)'라고 합니다.
    현실에서 흔합니다: 사기 거래 0.1%, 불량품 2%, 이탈 고객 5%, 암 0.5%
""")


# =============================================================================
# [2] 🗑️ 아무것도 안 하는 모델
# =============================================================================
print("=" * 78)
print("[2] '무조건 정상'이라고만 답하는 모델")
print("=" * 78)

# DummyClassifier(strategy="most_frequent") : 가장 많은 클래스만 계속 답합니다.
#   학습이라 부를 것도 없습니다. 데이터를 세기만 합니다.
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(np.zeros((N, 1)), y)
y_dummy = dummy.predict(np.zeros((N, 1)))

acc = accuracy_score(y, y_dummy)
print(f"\n  🎉 정확도 : {acc:.2%}")
print("     이 모델은 아무것도 학습하지 않았습니다. 그냥 '정상'만 외칩니다.")

cm = confusion_matrix(y, y_dummy)
tn, fp, fn, tp = print_confusion(cm, labels=("정상", "환자"))

print(f"""  🚨 진실
     환자 {n_pos}명 중 찾아낸 사람 : {tp}명
     재현율 : {recall_score(y, y_dummy, zero_division=0):.4f}
     F1     : {f1_score(y, y_dummy, zero_division=0):.4f}

     ★ 정확도 {acc:.2%} / 재현율 0%
       환자를 한 명도 못 찾았는데 성적표는 만점에 가깝습니다.
       이 모델을 병원에 배포하면 사람이 죽습니다.
""")


# =============================================================================
# [3] 정확도가 왜 속이는지 — 산수로 확인
# =============================================================================
print("=" * 78)
print("[3] 왜 이런 일이 생기나")
print("=" * 78)

print(f"""
  정확도 = (TP + TN) / 전체 = ({tp} + {tn}) / {N} = {acc:.4f}

  분자에서 TN({tn:,})이 압도적입니다.
  '정상을 정상이라 맞춘 것'만으로 이미 {n_neg/N:.1%} 를 먹고 들어갑니다.

  ★ 정확도는 '다수 클래스'의 성적에 지배당합니다.
    소수 클래스(환자)를 얼마나 못 잡든 거의 영향이 없습니다.

  💡 기억할 규칙
     클래스 비율이 X : (1-X) 일 때,
     아무것도 안 하는 모델의 정확도는 max(X, 1-X) 입니다.
     → 이 값을 '기준선(baseline)'이라고 부릅니다.
     → 내 모델 정확도가 이 기준선보다 못하면 학습이 실패한 것입니다.

     이 데이터의 기준선 : {max(POSITIVE_RATE, 1-POSITIVE_RATE):.1%}
     ★ 정확도 {max(POSITIVE_RATE, 1-POSITIVE_RATE):.0%} 는 '성과'가 아니라 '출발점'입니다.
""")


# =============================================================================
# [4] 진짜 모델과 비교
# =============================================================================
print("=" * 78)
print("[4] 실제로 학습한 모델과 비교")
print("=" * 78)

# 가짜 의료 기록 생성
#
# ★ 2026-08-05 수정 — 현실성 보정
#   처음엔 정상군/환자군의 어휘를 완전히 분리해서 만들었습니다.
#   그랬더니 balanced 모델이 F1 = 1.0000, 즉 만점을 받았습니다.
#   ...STAGE 3에서 배운 그 함정입니다. 문제가 너무 쉬웠던 겁니다.
#
#   현실의 의료 기록은 이렇게 깔끔하지 않습니다.
#   → AMBIGUOUS(양쪽에 다 나타나는 애매한 소견)를 섞어 난이도를 올립니다.
NORMAL = ["정기 검진 결과 이상 없음", "혈압 정상 범위", "수치 안정적",
          "특이사항 없음", "경과 양호", "추적 관찰 불필요"]
PATIENT = ["종양 의심 소견", "세포 이상 증식 확인", "조직검사 권고",
           "수치 급격히 상승", "전이 가능성 있음", "즉시 정밀검사 필요"]
# 양쪽 그룹 모두에서 나타나는 소견 — 이것 때문에 완벽한 분리가 불가능해집니다
AMBIGUOUS = ["경계성 수치 관찰됨", "재검사 권고", "이전 대비 변화 있음",
             "추가 확인 필요", "판독 보류"]

AMBIGUOUS_RATE = 0.25      # 25%의 기록은 애매한 소견으로 대체

texts = []
for i, lab in enumerate(y):
    if rng.random() < AMBIGUOUS_RATE:
        base = rng.choice(AMBIGUOUS)      # 정상이든 환자든 같은 문구
    else:
        base = rng.choice(PATIENT if lab else NORMAL)
    texts.append(f"{base} 케이스{i}")

overlap = len(set(w for t in NORMAL for w in t.split())
              & set(w for t in PATIENT for w in t.split()))
print(f"""
  📋 데이터 난이도 설정
     정상 전용 소견 {len(NORMAL)}종 / 환자 전용 소견 {len(PATIENT)}종
     ★ 애매한 소견 {len(AMBIGUOUS)}종 — 정상·환자 양쪽에 모두 등장 ({AMBIGUOUS_RATE:.0%})

     이 {AMBIGUOUS_RATE:.0%}는 원리적으로 구분이 불가능합니다.
     → 어떤 모델도 100%를 받을 수 없습니다. 현실이 그렇습니다.
""")

Xtr, Xte, ytr, yte = train_test_split(
    texts, y, test_size=0.3, random_state=42, stratify=y
)

models = {
    "🗑️ 무조건 정상 (Dummy)": None,
    "📗 기본 로지스틱회귀": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]),
    # ★ class_weight="balanced" : 소수 클래스의 오답에 더 큰 벌점을 줍니다.
    #   불균형 데이터의 표준 처방입니다.
    "📘 가중치 조정 (balanced)": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]),
}

print(f"\n  {'모델':<26}{'정확도':>9}{'정밀도':>9}{'재현율':>9}{'F1':>9}")
print("  " + "-" * 64)

results = []
for name, model in models.items():
    if model is None:
        pred = np.zeros(len(yte), dtype=int)
    else:
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

    a = accuracy_score(yte, pred)
    p = precision_score(yte, pred, zero_division=0)
    r = recall_score(yte, pred, zero_division=0)
    f = f1_score(yte, pred, zero_division=0)
    results.append((name, a, p, r, f))
    print(f"  {name:<26}{a:>9.4f}{p:>9.4f}{r:>9.4f}{f:>9.4f}")

# ★ 만점은 축하가 아니라 의심 대상입니다 (STAGE 3에서 배운 것)
for name, a, p, r, f in results:
    if f >= 0.999 and "Dummy" not in name:
        print(f"\n  🚨 {name} 의 F1이 {f:.4f} 입니다.")
        print("     실제 문제에서 만점은 거의 나오지 않습니다.")
        print("     데이터가 너무 쉽거나, 정답이 새어 들어갔을 가능성을 의심하세요.")

# 놓친 환자 수를 사람 단위로 보여주기 — 지표보다 이게 와닿습니다
print(f"\n  🏥 검증셋 환자 {yte.sum()}명 기준, 각 모델이 놓친 환자 수")
for name, model in models.items():
    pred = np.zeros(len(yte), dtype=int) if model is None else model.predict(Xte)
    c = confusion_matrix(yte, pred)
    missed, false_alarm = c[1][0], c[0][1]
    print(f"    {name:<26} 놓침 {missed:>3}명 / 헛경보 {false_alarm:>3}명")

print("""
  📖 정확도 열만 보면 세 모델이 다 비슷해 보입니다.
     F1 열과 '놓친 환자 수'를 보세요. 완전히 다른 이야기가 나옵니다.

  ★ 불균형 데이터에서 정확도는 정보가 거의 없는 숫자입니다.
    F1, 재현율, 혼동행렬을 보세요.

  ★★ 그리고 보고할 때는 지표가 아니라 '사람 수'로 말하세요.
     "재현율 0.83" 보다  "환자 30명 중 5명을 놓칩니다" 가
     의사결정자에게 훨씬 정확하게 전달됩니다.
""")


# =============================================================================
# [5] 그래프
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 4.5))
names = [r[0] for r in results]
x = np.arange(len(names))
w = 0.2
for i, (label, idx, color) in enumerate([
    ("정확도", 1, "#aec7e8"), ("정밀도", 2, "#1f77b4"),
    ("재현율", 3, "#d62728"), ("F1", 4, "#2ca02c"),
]):
    ax.bar(x + (i - 1.5) * w, [r[idx] for r in results], w, label=label, color=color)

ax.set_xticks(x)
ax.set_xticklabels([n.split(" ", 1)[1] for n in names], fontsize=9)
ax.set_ylim(0, 1.08)
ax.set_ylabel("점수")
ax.set_title("불균형 데이터: 정확도만 보면 속는다", fontsize=13)
ax.legend(ncol=4, fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out = DATA / "accuracy_trap.png"
fig.savefig(out, dpi=120)
print(f"  🖼️ 그림 저장: {out}")

print("""
=============================================================================
💼 STAGE 4 핵심 정리

   1. 정확도를 보고할 때는 반드시 '기준선'을 함께 말하세요.
      "정확도 99%" (X)
      "정확도 99%, 기준선 99% — 즉 개선 없음" (O)

   2. 불균형 데이터의 기본 지표는 F1과 재현율입니다.

   3. 불균형 처방 3가지
      - class_weight="balanced"  : 소수 클래스 오답에 벌점 강화
      - 임계값 조정               : MISSION 2에서 배운 것
      - 데이터 리샘플링           : 소수 클래스 늘리기 / 다수 줄이기
=============================================================================
""")


# =============================================================================
# 🎯 미션
# =============================================================================
# 1. POSITIVE_RATE 를 0.01 → 0.3 으로 바꿔 다시 실행하세요.
#    Dummy 모델의 정확도가 어떻게 변하나요? 왜죠?
#
# 2. class_weight="balanced" 를 뺀 모델과 넣은 모델의 재현율을 비교하세요.
#    어느 쪽이 환자를 더 많이 찾아내나요?
#
# 3. ★ 실무 상황: 상사가 "정확도 95% 나왔으니 배포하자"고 합니다.
#    무엇을 먼저 확인해야 한다고 말씀드리시겠어요? 세 가지만 적어보세요.
# =============================================================================
