"""
=============================================================================
STAGE 4 - MISSION 4 : 스팸 탐지 실전 (데이터 생성 → 학습 → 저장 → 배포)
=============================================================================
4일차 수업의 6·7·8번 실습(train_and_save / predict / newspam1000)을
하나로 합치고, 오늘 배운 지표와 임계값 조정을 붙였습니다.

★ 이번 미션의 실무 포인트
   지금까지는 "학습해서 정확도 확인"에서 끝났습니다.
   실무는 여기서부터입니다.
       모델을 파일로 저장 → 다른 프로그램에서 로드 → 새 데이터 예측
   이게 '서빙'의 가장 단순한 형태입니다. (STAGE 5의 예고편)

⏱️ 30초 이내

실행 : python 04_spam_pipeline.py
=============================================================================
"""

import itertools
import random

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from _common import DATA, check_separability, print_confusion

random.seed(42)
MODEL_PATH = DATA / "spam_model.pkl"


# =============================================================================
# [1] 스팸 데이터 생성 — ★ 일부러 불균형으로
# =============================================================================
print("=" * 78)
print("[1] 데이터 생성")
print("=" * 78)

# =============================================================================
# ★★ 2026-08-05 대폭 수정 — 난이도 현실화 (세 번의 실패 끝에)
# =============================================================================
# 처음 버전은 스팸과 정상이 완전히 다른 단어를 썼습니다.
#     정상: 결제, 명세서, 만기, 재발급 ...
#     스팸: 대출, 당첨, 클릭, 리딩방 ...
# 자카드 유사도 0.047. 단어 하나만 봐도 100% 분류되어 F1 = 1.0000 이 나왔습니다.
#
# 1차 수정: 피싱 문구를 섞어 어휘를 겹치게 함 → 자카드 0.61  ...그래도 F1 1.0000
#   왜? "본인 아니면 확인", "인증 필요" 같은 단서가 스팸에만 있었기 때문입니다.
#   ★ 어휘가 겹쳐도 '결정적 단서' 하나가 스팸에만 있으면 문제는 여전히 쉽습니다.
#     자카드 유사도만으로는 난이도를 보장할 수 없다는 걸 배웠습니다.
#
# 2차 수정(현재): ★ 은행도 실제로 보안 알림을 보냅니다.
#   "본인이 아니면 신고하세요"는 진짜 은행 문자에도 나오는 문구입니다.
#   → 같은 문구를 정상 메시지에도 넣었습니다.
#   → 이제 그 단서만으로는 판별이 불가능합니다. F1 약 0.87 수준의 현실적 난이도.
#
# 💡 여기서 얻은 진짜 교훈
#    합성 데이터로 현실적 난이도를 만드는 건 정말 어렵습니다.
#    그래서 NSMC 같은 '실제 사람이 만든 데이터'가 귀한 겁니다.
# =============================================================================

BANK = ["[신한카드]", "[KB국민은행]", "[우리은행]", "[하나카드]",
        "[토스]", "[카카오뱅크]", "[NH농협]", ""]

HAM_BODY = ["03/02 12:30 15,000원 결제 완료", "요청하신 통장 사본 발송 완료",
            "정기 예금 만기 안내", "비밀번호 3회 오류 영업점 방문 요망",
            "시스템 점검 안내", "체크카드 재발급 접수 완료",
            "이번 달 명세서가 발행되었습니다", "자동이체 정상 처리되었습니다",
            "출금 알림 32,000원", "적금 납입일이 도래했습니다",
            "공동인증서 갱신 기간입니다", "예약하신 상담 일정 안내",
            "해외 결제 승인 내역 안내", "카드 한도 상향 결과 안내",
            "계좌 잔액 조회 결과"]

HAM_TAIL = ["문의 1588-0000", "자세한 내용은 앱에서 확인", "감사합니다",
            "고객센터 문의 바랍니다", "영업시간 09-16시", "앱 알림 설정 변경 가능",
            "본 메시지는 안내용입니다", "수신 동의 고객 대상"]

# 노골적인 스팸 — 이건 쉽습니다
OBV_HEAD = ["[Web발신] (광고)", "(광고)", "[국제발신]", "★긴급★", "[Web발신]"]
OBV_BODY = ["정부지원 저금리 대출 안내", "최저 금리 2.5% 보장",
            "신용점수 무관 5천만원 승인", "투자 수익 300% 보장 리딩방",
            "당첨! 경품을 수령하세요", "최신 스마트폰 0원 기회",
            "비대면 특별 승인 대상자입니다", "무료 체험 이벤트 참여"]
OBV_TAIL = ["지금 바로 클릭", "상담 신청하세요", "선착순 마감",
            "즉시 입금 가능", "한도 조회하기", "무료 거부 080-123-4567"]

# ★ 핵심 — 이 문구들은 피싱도 쓰고 진짜 은행도 씁니다
CUE = ["본인 아니면 확인", "인증 필요", "취소하려면 클릭", "링크에서 조회", "즉시 신고"]


def _pool(*parts):
    return [" ".join(x for x in combo if x).strip()
            for combo in itertools.product(*parts)]


HAM_PLAIN = _pool(BANK, HAM_BODY, HAM_TAIL)              # 평범한 은행 알림
HAM_ALERT = _pool(BANK, HAM_BODY, CUE, HAM_TAIL)         # 은행의 진짜 보안 알림
OBVIOUS = _pool(OBV_HEAD, OBV_BODY, OBV_TAIL)            # 노골적 스팸


def take(pool, n, name):
    """중복 없이 n개를 뽑습니다. 못 채우면 조용히 넘어가지 않고 에러를 냅니다."""
    if n > len(pool):
        raise ValueError(
            f"[{name}] {n}개 요청 / 가능 조합 {len(pool)}개. 단어 목록을 늘리세요."
        )
    return random.sample(pool, n)


# ★ 실제 스팸 비율은 낮습니다. 현실을 반영해 8:2 로 만듭니다.
N_HAM, N_SPAM = 800, 200
HAM_ALERT_RATE = 0.12      # 정상 중 '은행 보안알림' 비율 ← 난이도 조절 다이얼
PHISHING_RATE = 0.70       # 스팸 중 '피싱' 비율 (나머지는 노골적 광고)

n_ham_alert = int(N_HAM * HAM_ALERT_RATE)
n_phish = int(N_SPAM * PHISHING_RATE)

print(f"  풀 크기 — 평범한알림 {len(HAM_PLAIN)} / 보안알림 {len(HAM_ALERT)} / 노골적스팸 {len(OBVIOUS)}")

# ★ 핵심 설계
#   '은행 보안알림'과 '피싱'을 같은 풀에서 뽑아 앞은 정상, 뒤는 스팸으로 라벨링합니다.
#   → 두 그룹의 문장 구조가 통계적으로 동일해집니다.
#   → 어떤 모델도 완벽 분리가 불가능합니다. 현실이 그렇습니다.
alert_pool = take(HAM_ALERT, n_ham_alert + n_phish, "보안알림/피싱")

texts = (take(HAM_PLAIN, N_HAM - n_ham_alert, "평범한알림")
         + alert_pool[:n_ham_alert]
         + take(OBVIOUS, N_SPAM - n_phish, "노골적스팸")
         + alert_pool[n_ham_alert:])
labels = [0] * N_HAM + [1] * N_SPAM      # 1 = 스팸

print(f"""
  구성
    정상 {N_HAM}건 = 평범한 알림 {N_HAM-n_ham_alert}건 + 🔶 은행 보안알림 {n_ham_alert}건
    스팸 {N_SPAM}건 = 노골적 광고 {N_SPAM-n_phish}건 + 🔶 피싱 {n_phish}건
    ★ 🔶 표시된 둘은 문장 구조가 같습니다. 이게 난이도의 핵심입니다.""")

assert len(texts) == len(labels), "문장 수와 라벨 수가 다릅니다"
assert len(set(texts)) == len(texts), "중복 문장이 있습니다"

print(f"\n  정상(ham)  {labels.count(0):>4}건")
print(f"  스팸(spam) {labels.count(1):>4}건   ({labels.count(1)/len(labels):.0%})")
print(f"  중복       {len(texts) - len(set(texts)):>4}건")
print(f"  → 불균형 데이터입니다. 기준선(무조건 정상) 정확도 = {labels.count(0)/len(labels):.0%}")

# ★ 학습 전 필수 점검 — 이 데이터가 너무 쉬운 건 아닌가?
check_separability(texts, labels, class_names=("정상", "스팸"))

X_tr, X_te, y_tr, y_te = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)
y_te = np.array(y_te)
print(f"  학습 {len(X_tr)}건 / 검증 {len(X_te)}건 (stratify 적용)")


# =============================================================================
# [2] 학습 — 한국어라 char n-gram (STAGE 3에서 배운 것)
# =============================================================================
print("\n" + "=" * 78)
print("[2] 학습")
print("=" * 78)

# -----------------------------------------------------------------------------
# ★ class_weight 를 두 가지로 비교합니다 — 만능이 아니라는 걸 보기 위해
# -----------------------------------------------------------------------------
# MISSION 3(의료 진단)에서는 class_weight="balanced" 가 놓친 환자를 26명 → 11명으로
# 줄여줬습니다. 재현율이 중요한 문제였으니까요.
#
# 그런데 스팸 필터는 '정밀도'가 중요합니다. 같은 처방이 통할까요?
def build(class_weight):
    return Pipeline([
        # analyzer="char_wb" : 한국어 조사·어미 문제를 우회 (STAGE 3 MISSION 4)
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
        ("clf", LogisticRegression(max_iter=1000, class_weight=class_weight)),
    ])


print(f"\n  {'class_weight':<16}{'정밀도':>9}{'재현율':>9}{'F1':>9}{'FP':>6}{'FN':>6}   성향")
print("  " + "-" * 68)

candidates = {}
for label, cw in (("balanced", "balanced"), ("None(기본)", None)):
    m = build(cw).fit(X_tr, y_tr)
    pr = m.predict_proba(X_te)[:, 1]
    pd_ = (pr >= 0.5).astype(int)
    c = confusion_matrix(y_te, pd_)
    tend = "스팸 쪽 과감" if c[0][1] > c[1][0] else "정상 쪽 신중"
    print(f"  {label:<16}{precision_score(y_te, pd_, zero_division=0):>9.4f}"
          f"{recall_score(y_te, pd_, zero_division=0):>9.4f}"
          f"{f1_score(y_te, pd_, zero_division=0):>9.4f}{c[0][1]:>6}{c[1][0]:>6}   {tend}")
    candidates[label] = (m, pr)

print("""
  🚨 같은 처방이 정반대로 작동했습니다.

     balanced   : 스팸을 하나도 안 놓침(FN 0) — 그런데 정상 메일 25건을 스팸함으로
     None(기본) : 정상 메일은 1건만 잃음 — 그런데 스팸 37건을 통과시킴

     ★ class_weight="balanced" 는 '재현율을 사는' 처방입니다.
       의료 진단(놓치면 안 됨)에는 좋고,
       스팸 필터(오판하면 안 됨)에는 역효과입니다.

     💼 처방은 문제에 맞춰야 합니다. 남이 쓴다고 따라 쓰면 안 됩니다.
""")

# 스팸 필터의 목적(정밀도 우선)에 맞는 쪽을 고릅니다.
model = build("balanced").fit(X_tr, y_tr)   # 아래 임계값 실험을 위해 balanced 사용

y_prob = model.predict_proba(X_te)[:, 1]     # 스팸일 확률
y_pred = (y_prob >= 0.5).astype(int)

print(f"\n  사전 크기 : {len(model.named_steps['tfidf'].vocabulary_):,}개")
print()
print(classification_report(y_te, y_pred, target_names=["정상", "스팸"], digits=4))

cm = confusion_matrix(y_te, y_pred)
tn, fp, fn, tp = print_confusion(cm, labels=("정상", "스팸"))


# =============================================================================
# [3] ★ 스팸 필터에서는 무엇이 더 아픈가
# =============================================================================
print("=" * 78)
print("[3] 임계값 결정 — 비즈니스 판단")
print("=" * 78)

print(f"""
  스팸 필터의 두 실수

    FP {fp}건 : 정상 메일을 스팸함으로 보냄
       → 계약서·면접 통보를 못 봅니다. ★ 사용자가 모르는 사이에 손해
    FN {fn}건 : 스팸을 받은편지함에 통과
       → 광고 하나 더 봅니다. 짜증나지만 되돌릴 수 있음

  ★ 스팸 필터는 '정밀도' 우선입니다.
    확실할 때만 스팸이라고 판정해야 합니다. → 임계값을 올립니다.
""")

print(f"  {'임계값':>6}{'정밀도':>9}{'재현율':>9}{'F1':>9}{'FP':>6}{'FN':>6}   판단")
print("  " + "-" * 62)

# 목표 정밀도 — "정상 메일 100건을 스팸함에 넣더라도 몇 건까지 감수하나"
# 0.95 = 스팸이라고 판정한 것 중 5%까지는 정상 메일이어도 괜찮다
target_precision = 0.95
chosen = None
for th in np.arange(0.3, 0.96, 0.05):
    pred = (y_prob >= th).astype(int)
    p = precision_score(y_te, pred, zero_division=0)
    r = recall_score(y_te, pred, zero_division=0)
    f = f1_score(y_te, pred, zero_division=0)
    c = confusion_matrix(y_te, pred)
    note = ""
    if p >= target_precision and chosen is None:
        chosen = float(th)
        note = f"← 정밀도 {target_precision:.0%} 최초 달성"
    print(f"  {th:>6.2f}{p:>9.4f}{r:>9.4f}{f:>9.4f}{c[0][1]:>6}{c[1][0]:>6}   {note}")

if chosen is None:
    chosen = 0.5
    print(f"""
  ⚠️ 어떤 임계값으로도 정밀도 {target_precision:.0%} 를 달성하지 못했습니다.
     임계값 조정만으로는 목표를 못 맞춘다는 뜻입니다.
     → 기본값 0.5 를 쓰되, 다음 중 하나가 필요합니다:
        · 더 좋은 모델 / 더 많은 데이터
        · 목표 정밀도를 현실적으로 낮추기
        · 애매한 건은 사람이 검토 (아래 [5]번의 Human-in-the-loop)""")
else:
    base = confusion_matrix(y_te, (y_prob >= 0.5).astype(int))
    sel = confusion_matrix(y_te, (y_prob >= chosen).astype(int))
    print(f"""
  ✅ 선택한 임계값 : {chosen:.2f}
     "정상 메일은 스팸함에 넣지 않는다"를 우선한 결정입니다.

     기본값 0.5 대비 변화
       FP(정상→스팸함) {base[0][1]:>3}건 → {sel[0][1]:>3}건   ({sel[0][1]-base[0][1]:+d}) ✅ 지킨 것
       FN(스팸 통과)   {base[1][0]:>3}건 → {sel[1][0]:>3}건   ({sel[1][0]-base[1][0]:+d}) ⬅ 치른 값

     ★ 이게 '의도적인 트레이드오프'의 실제 모습입니다.""")

    # -------------------------------------------------------------------------
    # ★★ 그런데 — 방금 우리가 한 게 결국 무엇이었나
    # -------------------------------------------------------------------------
    plain_m, plain_pr = candidates["None(기본)"]
    plain_pred = (plain_pr >= 0.5).astype(int)
    pc = precision_score(y_te, plain_pred, zero_division=0)
    rc = recall_score(y_te, plain_pred, zero_division=0)
    cc = confusion_matrix(y_te, plain_pred)

    sel_p = precision_score(y_te, (y_prob >= chosen).astype(int), zero_division=0)
    sel_r = recall_score(y_te, (y_prob >= chosen).astype(int), zero_division=0)

    print(f"""
  🔬 비교해보세요

     balanced + 임계값 {chosen:.2f}   → 정밀도 {sel_p:.4f} / 재현율 {sel_r:.4f} / FP {sel[0][1]} / FN {sel[1][0]}
     기본(None) + 임계값 0.50   → 정밀도 {pc:.4f} / 재현율 {rc:.4f} / FP {cc[0][1]} / FN {cc[1][0]}
""")
    if abs(sel_p - pc) < 0.02 and abs(sel_r - rc) < 0.02:
        print("""     🚨 거의 같습니다!

     ★ class_weight 와 임계값은 '같은 축'을 움직이는 다이얼입니다.
       둘 다 결정 경계를 좌우로 옮기는 일을 합니다.

       balanced 로 스팸 쪽으로 밀었다가 → 임계값으로 정상 쪽으로 되밀었으니
       결국 제자리로 돌아온 겁니다. 헛수고였습니다.

     💼 실무 규칙
        두 다이얼을 반대 방향으로 쓰지 마세요. 하나만 고르세요.
        · 임계값 조정  : 학습 끝난 모델에 사후 적용. 되돌리기 쉬움 ← 보통 이걸 씁니다
        · class_weight : 학습 자체를 바꿈. 재학습 필요""")

    # -------------------------------------------------------------------------
    # ★★★ 더 근본적인 질문 — 이 거래는 애초에 좋은 거래인가
    # -------------------------------------------------------------------------
    total_spam = int((y_te == 1).sum())
    leaked = int(sel[1][0])
    print(f"""
  ⚠️ 냉정하게 봅시다

     임계값 {chosen:.2f} 를 쓰면 스팸 {total_spam}건 중 {leaked}건({leaked/total_spam:.0%})이 통과합니다.
     스팸의 {leaked/total_spam:.0%}를 놓치는 필터는 사실상 켜놓은 의미가 없습니다.

     ★ 즉, 이 모델은 임계값 조정으로 해결되지 않습니다.
       정밀도와 재현율을 동시에 만족하는 지점이 존재하지 않습니다.

     → MISSION 2에서 배운 그 결론입니다:
       "둘 다 올리는 방법은 임계값 조정이 아니라 '더 좋은 모델'뿐이다."

     💼 이 상황에서 실무자가 할 말
        "임계값으로는 요구사항을 못 맞춥니다. 데이터가 더 필요합니다."
        이게 정직한 보고입니다. 억지로 숫자를 맞추는 게 아니라요.""")


# =============================================================================
# [4] 모델 저장 — joblib
# =============================================================================
print("=" * 78)
print("[4] 모델 저장 (배포 준비)")
print("=" * 78)

# ★ 모델만 저장하면 안 됩니다. 임계값도 함께 저장해야 합니다.
#   임계값은 '모델의 일부'가 아니라 '운영 설정'이지만,
#   같이 저장하지 않으면 배포 후 아무도 0.85를 기억하지 못합니다.
bundle = {
    "model": model,
    "threshold": float(chosen),
    "labels": {0: "정상", 1: "스팸"},
    "trained_on": f"ham {N_HAM} / spam {N_SPAM}",
    "note": "정밀도 우선. 정상 메일 오분류를 최소화하도록 임계값 상향.",
}
joblib.dump(bundle, MODEL_PATH)

size = MODEL_PATH.stat().st_size / 1024
print(f"\n  ✅ 저장 : {MODEL_PATH.name} ({size:.1f} KB)")
print(f"""
  💡 왜 이렇게 묶어서 저장하나?
     6개월 뒤 후임이 이 파일을 열었을 때
     "임계값이 왜 0.85지?" 를 알 수 있어야 합니다.
     모델 파일에 맥락을 함께 담는 게 실무 습관입니다.

  ★ BERT 모델은 save_pretrained(폴더), scikit-learn 모델은 joblib(.pkl).
    저장 방식이 다릅니다. 헷갈리지 마세요.
""")


# =============================================================================
# [5] 저장한 모델 다시 불러와서 쓰기 — ★ 이게 '서빙'입니다
# =============================================================================
print("=" * 78)
print("[5] 배포 시뮬레이션 — 다른 프로그램인 것처럼 로드")
print("=" * 78)

loaded = joblib.load(MODEL_PATH)
clf = loaded["model"]
th = loaded["threshold"]
names = loaded["labels"]

print(f"\n  로드 완료. 임계값 {th:.2f} / {loaded['note']}\n")

INCOMING = [
    "[Web발신] (광고) 서민금융진흥원 햇살론 승인 대상입니다 지금 바로 클릭",
    "[신한카드] 03/05 09:12 8,900원 결제 완료 문의 1588-0000",
    "지금 한도 조회하면 스타벅스 쿠폰 증정 즉시 입금 가능",
    "팀장님 내일 회의 자료 첨부드립니다 확인 부탁드려요",
    "★긴급★ 투자 수익 300% 보장 리딩방 선착순 마감",
]

# -----------------------------------------------------------------------------
# ★ 사람 검토 밴드를 '데이터로' 정합니다
# -----------------------------------------------------------------------------
# (2026-08-05 수정: 처음엔 ±0.15 로 고정했더니 5건 중 4건이 검토 대상이 됐습니다.
#  80%를 사람이 본다면 자동화가 아닙니다.
#  → 검증 데이터에서 "전체의 몇 %만 사람에게 넘길지"를 정하고,
#    그 비율에 맞는 밴드 폭을 역산합니다.)
REVIEW_BUDGET = 0.10        # 전체 물량의 10%까지만 사람이 검토
DOMAIN_THRESHOLD = 0.35     # 학습 어휘와 겹침이 이 비율 미만이면 '도메인 밖'으로 판정

dist = np.abs(y_prob - th)
band = float(np.quantile(dist, REVIEW_BUDGET))
print(f"  검토 밴드 계산: 전체의 {REVIEW_BUDGET:.0%}만 검토하려면 ±{band:.3f}")
print(f"  (고정값 ±0.15 를 쓰면 {np.mean(dist < 0.15):.0%}가 검토 대상이 됩니다)\n")

# 학습 데이터의 도메인 확인용 — 이 단어들이 하나도 없으면 '처음 보는 종류'입니다
train_vocab = set()
for t in X_tr:
    train_vocab.update(w for w in str(t).split() if len(w) >= 2)

for msg in INCOMING:
    prob = clf.predict_proba([msg])[0][1]
    pred = int(prob >= th)
    mark = "🚨" if pred else "✅"
    print(f"  {mark} [{names[pred]}] 스팸확률 {prob:.1%}")
    print(f"      {msg[:60]}")

    if abs(prob - th) < band:
        print(f"      ⚠️ 임계값({th:.2f}) ±{band:.3f} 안 → 사람 검토 대상")

    # ★ 도메인 밖 입력 경고 (STAGE 1에서 배운 그 문제)
    #
    # (2026-08-05 수정: 처음엔 "겹치는 단어가 0개"일 때만 경고했습니다.
    #  그랬더니 "팀장님 내일 회의 자료 첨부드립니다 확인 부탁드려요" 가
    #  '확인' 한 단어 때문에 통과했습니다. 7단어 중 1개(14%)면 사실상 도메인 밖인데요.
    #  → 개수가 아니라 '비율'로 판정하도록 바꿨습니다.)
    words = [w for w in msg.split() if len(w) >= 2]
    hits = [w for w in words if w in train_vocab]
    ratio = len(hits) / len(words) if words else 0.0

    if ratio < DOMAIN_THRESHOLD:
        print(f"      🚩 학습 어휘와 겹침 {len(hits)}/{len(words)} ({ratio:.0%}) "
              f"→ 학습해본 적 없는 종류입니다. 이 결과는 신뢰할 수 없습니다")
        print(f"         겹친 단어: {hits if hits else '없음'}")
    print()

print("""  🚩 표시가 붙은 메시지를 보세요.

     학습 데이터는 100% 은행 문자였습니다. 업무 메일은 한 건도 없습니다.
     그런 입력이 들어오면 모델은 '아는 단어'가 없어 사실상 찍습니다.
     그런데 확률은 그럴싸하게 나옵니다. STAGE 1 첫날 배운 그 문제입니다.

  💼 실무 설계
     서비스에는 '이건 내가 다룰 종류가 아니다'를 판정하는 관문이 따로 필요합니다.
     모델에게 모든 입력을 판단하게 하면 안 됩니다.
""")

print("=" * 78)
print("→ 05_boss_model_report.py : 배포 전 최종 성능 리포트를 만듭니다")
print("=" * 78)


# =============================================================================
# 🎯 미션
# =============================================================================
# 1. target_precision 을 0.99 → 0.90 으로 낮춰보세요.
#    임계값이 내려가나요? FN(놓친 스팸)은 몇 건이 되나요?
#
# 2. class_weight="balanced" 를 빼고 다시 실행하세요.
#    스팸의 재현율이 어떻게 변하나요?
#
# 3. INCOMING 에 본인이 받은 실제 문자를 넣어보세요. 잘 맞히나요?
#
# 4. ★ 임계값 근처 항목을 '사람 검토'로 빼는 이 패턴을
#    Human-in-the-loop 이라고 합니다. 실무 AI 시스템의 표준 설계입니다.
#    100% 자동화 대신, 애매한 것만 사람에게 넘기는 방식이죠.
#    → 이 아이디어를 STAGE 5의 Gradio 앱에 넣어보면 좋습니다.
# =============================================================================
