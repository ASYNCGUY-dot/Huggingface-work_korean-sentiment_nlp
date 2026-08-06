"""
=============================================================================
STAGE 4 공용 도구
=============================================================================
직접 실행하는 파일이 아닙니다. 다른 파일들이 가져다 쓰는 부품 상자입니다.
=============================================================================
"""

import platform
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 경로
# -----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent          # day4_evaluation/
DATA = BASE / "data"
DATA.mkdir(parents=True, exist_ok=True)

DAY3 = BASE.parent / "day3_finetuning" / "data"
COLAB_MODEL = DAY3 / "colab_model"              # Colab GPU로 학습한 88% 모델
CPU_MODEL = DAY3 / "my_movie_model"             # 템플릿 데이터로 학습한 모델


# -----------------------------------------------------------------------------
# 한글 폰트 — 그래프 글자 깨짐 방지
# -----------------------------------------------------------------------------
def set_korean_font():
    os_name = platform.system()
    if os_name == "Windows":
        plt.rc("font", family="Malgun Gothic")
    elif os_name == "Darwin":
        plt.rc("font", family="AppleGothic")
    else:
        plt.rc("font", family="NanumGothic")
    plt.rcParams["axes.unicode_minus"] = False


set_korean_font()


# -----------------------------------------------------------------------------
# 혼동행렬을 텍스트로 예쁘게 출력
# -----------------------------------------------------------------------------
def print_confusion(cm, labels=("부정", "긍정")):
    """2x2 혼동행렬을 표로 출력하고 TP/FP/FN/TN 을 해설합니다.

    scikit-learn 의 confusion_matrix 는 [행=실제, 열=예측] 순서입니다.
        cm[0][0] = 실제0 예측0
        cm[0][1] = 실제0 예측1
        cm[1][0] = 실제1 예측0
        cm[1][1] = 실제1 예측1
    """
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]

    w = max(len(str(x)) for x in (tn, fp, fn, tp)) + 2
    print(f"\n{'':>12}{'예측: ' + labels[0]:>{w+6}}{'예측: ' + labels[1]:>{w+6}}")
    print(f"  {'실제: ' + labels[0]:<10}{tn:>{w+6}}{fp:>{w+6}}")
    print(f"  {'실제: ' + labels[1]:<10}{fn:>{w+6}}{tp:>{w+6}}")

    total = tn + fp + fn + tp
    print(f"""
  TN (True Negative)  {tn:>6}건  실제 {labels[0]} → {labels[0]}로 맞춤   ✅
  FP (False Positive) {fp:>6}건  실제 {labels[0]} → {labels[1]}로 오판   ❌ 헛것을 봄
  FN (False Negative) {fn:>6}건  실제 {labels[1]} → {labels[0]}로 놓침   ❌ 놓침
  TP (True Positive)  {tp:>6}건  실제 {labels[1]} → {labels[1]}로 맞춤   ✅
  ─────────────────────────────
  전체 {total}건 중 오답 {fp + fn}건 ({(fp + fn) / total:.1%})
""")
    return tn, fp, fn, tp


# -----------------------------------------------------------------------------
# 4대 지표를 손으로 계산해서 공식과 함께 보여주기
# -----------------------------------------------------------------------------
def explain_metrics(tn, fp, fn, tp, positive_name="긍정"):
    """공식을 숫자와 함께 출력합니다. 외우지 말고 유도 과정을 보세요."""
    total = tn + fp + fn + tp
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print(f"""  정확도 Accuracy  = (TP+TN)/전체 = ({tp}+{tn})/{total} = {acc:.4f}
     → 전체 중 몇 개나 맞췄나

  정밀도 Precision = TP/(TP+FP) = {tp}/({tp}+{fp}) = {prec:.4f}
     → '{positive_name}'이라고 말한 것 중 진짜 비율
     → 이 값이 낮으면: 아무거나 {positive_name}이라고 외치고 있다

  재현율 Recall    = TP/(TP+FN) = {tp}/({tp}+{fn}) = {rec:.4f}
     → 실제 '{positive_name}' 중 몇 개나 찾아냈나
     → 이 값이 낮으면: 있는 걸 놓치고 있다

  F1-score         = 2·P·R/(P+R) = {f1:.4f}
     → 정밀도와 재현율의 조화평균
     → ★ 왜 산술평균이 아닌가? 아래 [실험]에서 확인합니다
""")
    return acc, prec, rec, f1


# -----------------------------------------------------------------------------
# 산술평균 vs 조화평균 — F1이 조화평균인 이유
# -----------------------------------------------------------------------------
def check_separability(texts, labels, class_names=("클래스0", "클래스1")):
    """★ 이 데이터, 너무 쉬운 것 아닌가? — 학습 전에 반드시 돌려보세요.

    왜 만들었나
      STAGE 3~4에서 직접 만든 실습 데이터가 세 번 연속 만점(F1=1.0)을 냈습니다.
      매번 원인은 같았습니다. '두 클래스가 완전히 다른 단어를 쓴 것'입니다.

        정상: 결제, 명세서, 만기, 재발급 ...
        스팸: 대출, 당첨, 클릭, 리딩방 ...
        → 겹치는 단어가 없으면 단어 하나만 봐도 100% 분류됩니다.

      현실은 다릅니다. 피싱 문자는 정상 문자를 흉내냅니다.
      그래서 어렵고, 그래서 AI가 필요합니다.

    측정 방법
      두 클래스의 어휘 집합이 얼마나 겹치는지(자카드 유사도)를 봅니다.
        0.0  = 완전히 다른 단어만 씀 → 문제가 너무 쉬움 🚨
        0.3~0.6 = 현실적인 난이도 ✅
        1.0  = 똑같은 단어만 씀 → 구분 불가능
    """
    import re

    vocabs = []
    for target in (0, 1):
        joined = " ".join(str(t) for t, l in zip(texts, labels) if l == target)
        vocabs.append(set(re.findall(r"[가-힣a-zA-Z]{2,}", joined)))

    v0, v1 = vocabs
    inter = v0 & v1
    union = v0 | v1
    jaccard = len(inter) / len(union) if union else 0.0

    print(f"\n  🔍 데이터 난이도 진단 (어휘 겹침)")
    print(f"     {class_names[0]} 어휘 {len(v0):,}개 / {class_names[1]} 어휘 {len(v1):,}개")
    print(f"     겹치는 어휘 {len(inter):,}개")
    print(f"     자카드 유사도 {jaccard:.3f}", end="")

    if jaccard < 0.15:
        print("   🚨 너무 쉬움")
        print("     → 두 클래스가 거의 다른 단어를 씁니다.")
        print("       모델이 만점을 받아도 실력이 아닙니다. 데이터를 어렵게 만드세요.")
    elif jaccard < 0.3:
        print("   🟡 다소 쉬움")
    elif jaccard <= 0.7:
        print("   ✅ 현실적")
    else:
        print("   ⚠️ 너무 어려움 (구분 단서가 부족할 수 있음)")

    if inter and len(inter) <= 12:
        print(f"     겹친 단어: {sorted(inter)}")

    return jaccard


def mean_comparison(pairs):
    """(정밀도, 재현율) 쌍들에 대해 두 평균을 비교합니다."""
    print(f"\n  {'정밀도':>8}{'재현율':>8}{'산술평균':>10}{'F1(조화)':>10}   차이")
    print("  " + "-" * 52)
    for p, r in pairs:
        arith = (p + r) / 2
        harm = 2 * p * r / (p + r) if (p + r) else 0
        gap = arith - harm
        mark = "  🚨 한쪽이 극단적" if gap > 0.15 else ""
        print(f"  {p:>8.2f}{r:>8.2f}{arith:>10.2f}{harm:>10.2f}   {gap:+.2f}{mark}")
