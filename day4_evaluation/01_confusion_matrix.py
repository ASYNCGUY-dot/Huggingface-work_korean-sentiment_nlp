"""
=============================================================================
STAGE 4 - MISSION 1 : 그 12%는 어디로 갔나
=============================================================================
STAGE 3에서 지수님이 만든 모델은 검증 정확도 88% 였습니다.
B조 실전 문장은 10/10 만점이었고요.

그런데 88% 라는 건 12% 를 틀렸다는 뜻입니다.

    ❓ 그 12%는 '어떻게' 틀렸을까요?
       긍정을 부정이라고 한 걸까요? 반대일까요?
       그리고 둘 중 어느 쪽이 더 아플까요?

정확도 숫자 하나로는 절대 알 수 없습니다. 혼동행렬이 필요합니다.

⏱️ 추론만 하므로 CPU에서도 2~4분

실행 : python 01_confusion_matrix.py
=============================================================================
"""

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from _common import COLAB_MODEL, DATA, explain_metrics, print_confusion

import matplotlib.pyplot as plt


N_TEST = 2000       # 평가에 쓸 문장 수 (늘리면 정확하지만 느립니다)
BATCH = 32
MAX_LENGTH = 128


# =============================================================================
# [0] 모델 불러오기
# =============================================================================
if not COLAB_MODEL.exists():
    print(f"❌ 모델이 없습니다: {COLAB_MODEL}")
    print("   STAGE 3의 Colab 노트북으로 학습한 모델을 이 경로에 풀어주세요.")
    raise SystemExit(1)

print("모델 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(COLAB_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(COLAB_MODEL)
model.eval()
print(f"  {model.config._name_or_path} / 라벨 {model.config.id2label}")


# =============================================================================
# [1] 평가용 데이터 — ★ 학습에 한 번도 안 쓴 test split
# =============================================================================
print("\n" + "=" * 78)
print("[1] 평가 데이터 준비")
print("=" * 78)

raw = load_dataset("e9t/nsmc", trust_remote_code=True)

# ★ 왜 test split 인가?
#   Colab에서는 train split 20,000건만 썼습니다.
#   test split 50,000건은 모델이 한 번도 본 적 없습니다.
#   진짜 실력을 재려면 반드시 '처음 보는' 데이터로 재야 합니다.
test = raw["test"].shuffle(seed=42).select(range(N_TEST))

texts = [str(t) for t in test["document"]]
y_true = np.array(test["label"])

print(f"\n  출처   : NSMC test split (학습에 사용된 적 없음)")
print(f"  건수   : {len(texts):,}건")
print(f"  라벨   : 긍정 {y_true.sum():,} / 부정 {len(y_true) - y_true.sum():,}")
print(f"  균형도 : {y_true.mean():.1%} : {1 - y_true.mean():.1%}")


# =============================================================================
# [2] 예측
# =============================================================================
print("\n" + "=" * 78)
print("[2] 예측 실행")
print("=" * 78)
print(f"\n  {len(texts):,}건을 {BATCH}개씩 묶어 처리합니다. 잠시 기다려주세요...\n")

y_pred, y_prob = [], []
for i in range(0, len(texts), BATCH):
    batch = texts[i:i + BATCH]
    inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                       padding=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    y_pred.extend(torch.argmax(probs, dim=-1).tolist())
    y_prob.extend(probs[:, 1].tolist())      # '긍정'일 확률만 저장

    done = min(i + BATCH, len(texts))
    if done % 320 == 0 or done == len(texts):
        print(f"    {done:>5,}/{len(texts):,} ({done/len(texts):.0%})")

y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# 나중 미션에서 재사용하려고 저장 (다시 추론하지 않아도 되도록)
np.savez(DATA / "predictions.npz", y_true=y_true, y_pred=y_pred, y_prob=y_prob)
print(f"\n  ✅ 예측 결과 저장: {DATA / 'predictions.npz'}")


# =============================================================================
# [3] ★ 혼동행렬 — 12%의 정체
# =============================================================================
print("\n" + "=" * 78)
print("[3] 혼동행렬 (Confusion Matrix)")
print("=" * 78)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = print_confusion(cm, labels=("부정", "긍정"))


# =============================================================================
# [4] 4대 지표 — 공식과 함께
# =============================================================================
print("=" * 78)
print("[4] 지표 계산 (공식 그대로)")
print("=" * 78)
print()
acc, prec, rec, f1 = explain_metrics(tn, fp, fn, tp, positive_name="긍정")


# =============================================================================
# [5] 어느 쪽으로 치우쳤나
# =============================================================================
print("=" * 78)
print("[5] 오답의 성향 진단")
print("=" * 78)

print(f"""
  FP {fp}건 : 부정 리뷰를 긍정이라고 오판
  FN {fn}건 : 긍정 리뷰를 부정이라고 놓침
""")

if fp > fn * 1.3:
    print("  🔵 이 모델은 '긍정 쪽으로 관대'합니다.")
    print("     의심스러우면 긍정이라고 답하는 경향이 있습니다.")
    print("     → 정밀도가 재현율보다 낮게 나옵니다.")
elif fn > fp * 1.3:
    print("  🔴 이 모델은 '부정 쪽으로 엄격'합니다.")
    print("     확실할 때만 긍정이라고 답합니다.")
    print("     → 재현율이 정밀도보다 낮게 나옵니다.")
else:
    print("  ⚖️ 두 방향의 오류가 비슷합니다. 균형 잡힌 모델입니다.")
    print("     학습 데이터의 라벨 비율이 균형이었기 때문입니다.")

print(f"\n  정밀도 {prec:.4f}  vs  재현율 {rec:.4f}  (차이 {abs(prec-rec):.4f})")


# -----------------------------------------------------------------------------
# ★ 그 차이, 진짜인가 우연인가 — 통계로 확인
# -----------------------------------------------------------------------------
# 눈으로 보면 FP가 FN보다 많아 보입니다. 그런데 그게 의미 있는 차이일까요?
# 동전을 240번 던져 앞면이 128번 나왔다고 "이 동전은 앞면이 잘 나온다"고
# 말할 수 있을까요? 이항검정으로 확인합니다.
import math

try:
    from scipy import stats
    pval = stats.binomtest(int(fp), int(fp + fn), 0.5).pvalue
    print(f"\n  🔬 이항검정 : 오답 {fp+fn}건 중 FP {fp}건")
    print(f"     p-value = {pval:.3f}")
    if pval > 0.05:
        print("     → 우연 범위입니다. 두 오류의 '방향성 차이'는 없다고 봐야 합니다.")
        print("       같은 실험을 다시 하면 FN이 더 많이 나올 수도 있습니다.")
    else:
        print("     → 통계적으로 유의미한 편향입니다.")
except ImportError:
    print("\n  (scipy 미설치 — 이항검정 생략. pip install scipy)")


# -----------------------------------------------------------------------------
# ★★ 정확도는 '점'이 아니라 '구간'입니다
# -----------------------------------------------------------------------------
se = math.sqrt(acc * (1 - acc) / len(y_true))
lo, hi = acc - 1.96 * se, acc + 1.96 * se

print(f"""
  📏 정확도의 95% 신뢰구간
     표본 {len(y_true):,}건 기준 : {acc:.1%} ± {1.96*se:.1%}  →  [{lo:.1%}, {hi:.1%}]

     ★ "정확도 88%" 는 정확히 88%라는 뜻이 아닙니다.
       "86.6%~89.4% 어딘가"라는 뜻입니다.

     💼 실무에서 중요한 이유
        모델 A 88.0% vs 모델 B 88.5% → "B가 낫다"고 말할 수 없습니다.
        신뢰구간이 겹치니까요. 구간이 안 겹칠 만큼 차이가 나야 의미가 있습니다.

     표본이 커야 구간이 좁아집니다:""")
for nn in (200, 2000, 20000):
    s = math.sqrt(acc * (1 - acc) / nn)
    print(f"        {nn:>6,}건 → ±{1.96*s:.1%}")


# =============================================================================
# [6] sklearn 공식 리포트
# =============================================================================
print("\n" + "=" * 78)
print("[6] classification_report (실무에서 가장 많이 쓰는 형태)")
print("=" * 78)
print()
print(classification_report(y_true, y_pred, target_names=["부정", "긍정"], digits=4))

print("""  📖 리포트 읽는 법
     support     : 해당 클래스의 실제 개수
     macro avg   : 두 클래스의 단순 평균 (클래스 크기 무시)
     weighted avg: 개수로 가중한 평균 (큰 클래스가 유리)

     ★ 불균형 데이터에서는 macro avg 를 보세요.
       weighted avg 는 다수 클래스에 끌려다닙니다.
""")


# =============================================================================
# [7] 그림으로 저장
# =============================================================================
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks([0, 1]); ax.set_xticklabels(["예측: 부정", "예측: 긍정"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["실제: 부정", "실제: 긍정"])
ax.set_title(f"혼동행렬 (정확도 {acc:.1%})", fontsize=13, pad=12)

names = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        color = "white" if cm[i][j] > cm.max() / 2 else "black"
        ax.text(j, i, f"{names[i][j]}\n{cm[i][j]:,}",
                ha="center", va="center", color=color, fontsize=13)

fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
out = DATA / "confusion_matrix.png"
fig.savefig(out, dpi=120)
print(f"  🖼️ 그림 저장: {out}")

print("\n" + "=" * 78)
print("→ 02_precision_recall.py : 정밀도와 재현율 중 무엇을 포기할 것인가")
print("=" * 78)


# =============================================================================
# 🎯 미션
# =============================================================================
# 1. FP와 FN 중 어느 쪽이 많나요? 그 의미를 [5]번 진단과 함께 이해하세요.
#
# 2. 정밀도와 재현율의 차이가 큰가요, 작은가요?
#    NSMC는 긍정:부정이 50:50인 균형 데이터입니다. 그게 결과에 어떤 영향을 줬을까요?
#
# 3. ★ N_TEST 를 500으로 줄여서 다시 돌려보세요.
#    정확도가 얼마나 달라지나요? 표본이 작으면 왜 못 믿나요?
#
# 4. ★★ 틀린 문장들을 직접 눈으로 보세요. 아래를 파일 끝에 추가:
#
#       wrong = [(t, yt, yp, pr) for t, yt, yp, pr
#                in zip(texts, y_true, y_pred, y_prob) if yt != yp]
#       wrong.sort(key=lambda x: abs(x[3] - 0.5))   # 애매했던 순
#       print("\n[모델이 가장 헷갈린 오답 10건]")
#       for t, yt, yp, pr in wrong[:10]:
#           print(f"  정답{'긍정' if yt else '부정'} 예측{'긍정' if yp else '부정'} "
#                 f"(긍정확률 {pr:.2f})  {t[:50]}")
#
#    → 사람이 봐도 애매한 문장인가요? 그렇다면 모델 탓이 아닙니다.
# =============================================================================
