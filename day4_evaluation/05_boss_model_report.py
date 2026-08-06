"""
=============================================================================
STAGE 4 - BOSS BATTLE : 배포 전 최종 성능 리포트
=============================================================================
STAGE 5에서 이 모델을 세상에 공개합니다.
그 전에 반드시 통과해야 하는 관문이 있습니다.

    "이 모델을 배포해도 되는가?"

실무에서는 이 질문에 문서로 답합니다. 그게 '모델 카드(Model Card)'입니다.
오늘 배운 모든 지표를 모아 HTML 리포트를 자동 생성합니다.

★ 이 리포트가 STAGE 5의 첫 관문이 됩니다.

⏱️ 01번의 예측 결과를 재사용하므로 즉시 실행

실행 : python 05_boss_model_report.py
=============================================================================
"""

import math
from datetime import datetime

import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

from _common import COLAB_MODEL, DATA, set_korean_font

import matplotlib.pyplot as plt


# =============================================================================
# [0] 데이터 로드
# =============================================================================
pred_file = DATA / "predictions.npz"
if not pred_file.exists():
    print(f"❌ 없습니다: {pred_file}\n   01_confusion_matrix.py 를 먼저 실행하세요.")
    raise SystemExit(1)

d = np.load(pred_file)
y_true, y_pred, y_prob = d["y_true"], d["y_pred"], d["y_prob"]
n = len(y_true)

print("=" * 78)
print("배포 전 최종 검증")
print("=" * 78)
print(f"\n  모델    : {COLAB_MODEL.name}")
print(f"  평가셋  : NSMC test split {n:,}건 (학습에 사용 안 됨)")


# =============================================================================
# [1] 핵심 지표 집계
# =============================================================================
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

acc = (tp + tn) / n
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
auc = roc_auc_score(y_true, y_prob)

# 신뢰구간
se = math.sqrt(acc * (1 - acc) / n)
ci = 1.96 * se

# 기준선 (다수 클래스만 답하는 모델)
baseline = max(y_true.mean(), 1 - y_true.mean())

print(f"""
  정확도   {acc:.4f}  ±{ci:.4f}   (기준선 {baseline:.4f})
  정밀도   {prec:.4f}
  재현율   {rec:.4f}
  F1       {f1:.4f}
  ROC-AUC  {auc:.4f}
""")


# =============================================================================
# [2] ★ 배포 가능 여부 자동 판정 — 체크리스트
# =============================================================================
print("=" * 78)
print("[2] 배포 게이트 (하나라도 실패하면 배포 보류)")
print("=" * 78)

checks = []


def gate(name, passed, detail):
    checks.append((name, passed, detail))
    print(f"\n  {'✅ PASS' if passed else '🚨 FAIL'}  {name}")
    print(f"          {detail}")


# ① 기준선을 유의미하게 넘는가
gate("기준선 초과",
     acc - ci > baseline,
     f"정확도 하한 {acc-ci:.4f} vs 기준선 {baseline:.4f}  "
     f"→ {'신뢰구간 하한이 기준선보다 높음' if acc-ci > baseline else '오차범위 안 — 개선을 증명 못 함'}")

# ② 두 방향 오류가 심하게 치우치지 않았는가
skew = max(fp, fn) / max(min(fp, fn), 1)
gate("오류 균형",
     skew < 2.0,
     f"FP {fp} / FN {fn} (비율 {skew:.2f}배)  "
     f"→ {'균형적' if skew < 2.0 else '한쪽으로 치우침 — 임계값 재검토 필요'}")

# ③ 판별력이 있는가 (AUC)
gate("판별력(ROC-AUC)",
     auc >= 0.8,
     f"AUC {auc:.4f}  → 0.5=무작위 / 0.7=쓸만함 / 0.8=좋음 / 0.9=우수")

# ④ 만점이 아닌가 (데이터 누수 의심)
gate("데이터 누수 없음",
     f1 < 0.99,
     f"F1 {f1:.4f}  → {'정상 범위' if f1 < 0.99 else '만점에 가까움 — 정답 누수/데이터 중복 의심'}")

# ⑤ 표본이 충분한가
gate("표본 충분",
     n >= 1000,
     f"{n:,}건 (오차 ±{ci:.1%})  → 1,000건 미만이면 오차가 커서 판단 곤란")

# ⑥ 확신도가 캘리브레이션되어 있는가
#    맞을 때는 확신이 높고, 틀릴 때는 낮아야 정상입니다.
conf = np.maximum(y_prob, 1 - y_prob)
correct = y_true == y_pred
conf_ok = conf[correct].mean()
conf_bad = conf[~correct].mean()
gate("확신도 캘리브레이션",
     conf_ok > conf_bad + 0.05,
     f"맞을 때 {conf_ok:.3f} / 틀릴 때 {conf_bad:.3f} (차이 {conf_ok-conf_bad:+.3f})  "
     f"→ {'틀릴 때 확신이 낮음 = 정직한 모델' if conf_ok > conf_bad + 0.05 else '틀려도 확신함 — 위험'}")

passed = sum(1 for _, p, _ in checks if p)
print(f"\n  {'─' * 60}")
print(f"  통과 {passed}/{len(checks)}")
if passed == len(checks):
    print("  🎉 배포 가능 판정입니다.")
else:
    print("  ⚠️ 실패 항목을 해결하거나, 그 위험을 문서에 명시하고 승인받아야 합니다.")


# =============================================================================
# [3] 임계값별 운영 시나리오 표
# =============================================================================
print("\n" + "=" * 78)
print("[3] 운영 시나리오 — 서비스 성격에 따른 임계값 권고")
print("=" * 78)

scenarios = []
for th in np.arange(0.05, 1.0, 0.05):
    pr = (y_prob >= th).astype(int)
    c = confusion_matrix(y_true, pr)
    scenarios.append({
        "th": float(th),
        "prec": precision_score(y_true, pr, zero_division=0),
        "rec": recall_score(y_true, pr, zero_division=0),
        "f1": f1_score(y_true, pr, zero_division=0),
        "fp": int(c[0][1]), "fn": int(c[1][0]),
    })

# 목적별로 최적 임계값 추천
def pick(cond, key, reverse=True):
    cands = [s for s in scenarios if cond(s)]
    if not cands:
        return None
    return sorted(cands, key=lambda s: s[key], reverse=reverse)[0]

recos = [
    ("F1 최대 (균형)", pick(lambda s: True, "f1")),
    ("정밀도 95% 이상 유지", pick(lambda s: s["prec"] >= 0.95, "rec")),
    ("재현율 95% 이상 유지", pick(lambda s: s["rec"] >= 0.95, "prec")),
]

print(f"\n  {'목적':<22}{'임계값':>7}{'정밀도':>9}{'재현율':>9}{'F1':>9}{'FP':>6}{'FN':>6}")
print("  " + "-" * 70)
for label, s in recos:
    if s is None:
        print(f"  {label:<22}{'달성 불가':>7}")
    else:
        print(f"  {label:<22}{s['th']:>7.2f}{s['prec']:>9.4f}{s['rec']:>9.4f}"
              f"{s['f1']:>9.4f}{s['fp']:>6}{s['fn']:>6}")

print("""
  💼 이 표가 실무 산출물입니다.
     개발자는 임계값을 '정하지' 않습니다. 선택지를 '제시'합니다.
     결정은 서비스 책임자가 합니다.
""")


# =============================================================================
# [4] 그래프
# =============================================================================
set_korean_font()
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# ① 혼동행렬
im = axes[0].imshow(cm, cmap="Blues")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["예측 부정", "예측 긍정"])
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["실제 부정", "실제 긍정"])
axes[0].set_title(f"혼동행렬 (정확도 {acc:.1%})")
lbl = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, f"{lbl[i][j]}\n{cm[i][j]:,}", ha="center", va="center",
                     color="white" if cm[i][j] > cm.max()/2 else "black", fontsize=12)

# ② 임계값 곡선
ths = [s["th"] for s in scenarios]
axes[1].plot(ths, [s["prec"] for s in scenarios], "o-", ms=3, label="정밀도", color="#1f77b4")
axes[1].plot(ths, [s["rec"] for s in scenarios], "s-", ms=3, label="재현율", color="#d62728")
axes[1].plot(ths, [s["f1"] for s in scenarios], "^-", ms=3, label="F1", color="#2ca02c", lw=2)
axes[1].axvline(0.5, ls="--", c="gray", alpha=0.5)
axes[1].set_xlabel("임계값"); axes[1].set_ylabel("점수")
axes[1].set_title("임계값별 지표"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

# ③ 확신도 분포 (맞춤 vs 틀림)
axes[2].hist(conf[correct], bins=20, alpha=0.7, label=f"맞춤 (평균 {conf_ok:.2f})", color="#2ca02c")
axes[2].hist(conf[~correct], bins=20, alpha=0.7, label=f"틀림 (평균 {conf_bad:.2f})", color="#d62728")
axes[2].set_xlabel("확신도"); axes[2].set_ylabel("건수")
axes[2].set_title("확신도 분포 — 겹칠수록 위험")
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

fig.tight_layout()
png = DATA / "model_report.png"
fig.savefig(png, dpi=110)
print(f"  🖼️ 그림 저장: {png}")


# =============================================================================
# [5] HTML 모델 카드 생성
# =============================================================================
rows = "\n".join(
    f"<tr><td>{'✅' if p else '🚨'}</td><td>{name}</td><td>{detail}</td></tr>"
    for name, p, detail in checks
)
def reco_row(label, s):
    """추천 시나리오 한 줄을 HTML 표 행으로 만듭니다.

    f-string 안에 또 f-string 을 넣으면 따옴표가 충돌해 SyntaxError 가 납니다.
    이렇게 함수로 빼서 값을 먼저 문자열로 만들어두면 깔끔합니다.
    """
    if s is None:
        cells = ["달성 불가", "—", "—", "—", "—"]
    else:
        cells = [f"{s['th']:.2f}", f"{s['prec']:.4f}", f"{s['rec']:.4f}",
                 str(s["fp"]), str(s["fn"])]
    tds = "".join(f"<td>{c}</td>" for c in cells)
    return f"<tr><td>{label}</td>{tds}</tr>"


reco_rows = "\n".join(reco_row(label, s) for label, s in recos)

html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>모델 카드 — {COLAB_MODEL.name}</title>
<style>
 body{{font-family:'Malgun Gothic',sans-serif;max-width:900px;margin:40px auto;padding:0 20px;line-height:1.7;color:#222}}
 h1{{border-bottom:3px solid #333;padding-bottom:10px}}
 h2{{margin-top:36px;color:#1a4d80}}
 table{{border-collapse:collapse;width:100%;margin:16px 0}}
 th,td{{border:1px solid #ddd;padding:9px 12px;text-align:left;font-size:14px}}
 th{{background:#f4f6f8}}
 .metric{{display:inline-block;background:#f4f6f8;border-radius:8px;padding:14px 20px;margin:6px 8px 6px 0;min-width:130px}}
 .metric b{{display:block;font-size:26px;color:#1a4d80}}
 .warn{{background:#fff4e5;border-left:4px solid #ff9800;padding:12px 16px;margin:16px 0}}
 img{{max-width:100%;border:1px solid #eee;border-radius:6px}}
 code{{background:#f4f6f8;padding:2px 6px;border-radius:4px}}
</style></head><body>
<h1>모델 카드</h1>
<p><b>모델</b> {COLAB_MODEL.name} (klue/roberta-base 파인튜닝)<br>
<b>작업</b> 한국어 영화 리뷰 감성 분류 (긍정/부정)<br>
<b>평가 데이터</b> NSMC test split {n:,}건 — 학습에 사용되지 않음<br>
<b>작성</b> {datetime.now():%Y-%m-%d %H:%M}</p>

<h2>성능 지표</h2>
<div class="metric">정확도<b>{acc:.1%}</b>±{ci:.1%}</div>
<div class="metric">정밀도<b>{prec:.3f}</b></div>
<div class="metric">재현율<b>{rec:.3f}</b></div>
<div class="metric">F1<b>{f1:.3f}</b></div>
<div class="metric">ROC-AUC<b>{auc:.3f}</b></div>
<p>기준선(다수 클래스만 답하는 모델)은 <code>{baseline:.1%}</code> 입니다.
정확도는 점 추정치가 아니라 <b>{acc-ci:.1%} ~ {acc+ci:.1%}</b> 구간으로 읽어야 합니다.</p>

<h2>배포 게이트</h2>
<table><tr><th></th><th>항목</th><th>근거</th></tr>
{rows}
</table>
<p><b>통과 {passed}/{len(checks)}</b></p>

<h2>운영 시나리오</h2>
<table><tr><th>목적</th><th>임계값</th><th>정밀도</th><th>재현율</th><th>FP</th><th>FN</th></tr>
{reco_rows}
</table>
<div class="warn"><b>임계값은 개발자가 정하지 않습니다.</b>
위 선택지를 서비스 책임자에게 제시하고, 무엇을 포기할지 함께 결정합니다.</div>

<h2>알려진 한계</h2>
<ul>
<li>학습 데이터는 <b>영화 리뷰</b>입니다. 다른 도메인(상품평, 뉴스, 업무 메일)에서는 성능이 보장되지 않습니다.</li>
<li>부정 표현이 우회적인 문장("기대했던 것과는 좀 다르네요")에 약합니다.</li>
<li>학습 어휘에 없는 표현이 들어오면 확신도는 높지만 결과는 신뢰할 수 없습니다.</li>
<li>2지선다 모델이라 <b>"중립"이나 "모르겠음"을 출력할 수 없습니다.</b></li>
</ul>

<h2>그래프</h2>
<img src="{png.name}" alt="성능 그래프">

<hr><p style="color:#888;font-size:13px">
DONGA KDT 허깅페이스 과정 STAGE 4 산출물 · 자동 생성</p>
</body></html>"""

html_path = DATA / "model_card.html"
html_path.write_text(html, encoding="utf-8")

print(f"  📄 모델 카드 저장: {html_path}")
print(f"\n     브라우저로 열어보세요. STAGE 5 배포 심사의 첫 관문입니다.")

print("\n" + "=" * 78)
print("🏆 BOSS 클리어 — STAGE 4 완주")
print("=" * 78)


# =============================================================================
# 🎯 보스전 미션
# =============================================================================
#  [★]    model_card.html 을 브라우저로 열어보세요.
#         이 문서를 상사에게 제출한다고 상상하고, 빠진 정보가 없는지 보세요.
#
#  [★★]  '알려진 한계' 항목에 본인이 발견한 것을 추가해보세요.
#         (STAGE 1~4에서 겪은 것들이 많습니다)
#
#  [★★★] 배포 게이트에 항목을 하나 추가해보세요. 예시:
#         · 추론 속도 : 1건당 100ms 이내인가
#         · 모델 크기 : 배포 환경에 올라가는가 (지금 443MB)
#         · 라이선스 : 상업적 사용이 가능한가
#         → STAGE 5에서 실제로 이 세 가지에 부딪힙니다.
# =============================================================================
