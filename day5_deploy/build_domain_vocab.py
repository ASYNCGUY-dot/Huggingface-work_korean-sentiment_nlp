"""
=============================================================================
도메인 어휘 사전 만들기 (한 번만 실행)
=============================================================================
왜 필요한가

  02번 앱에서 "오늘 회의는 3시입니다"에 도메인 경고가 안 떴습니다.
  원인: 제가 '토크나이저 사전'과 비교하고 있었기 때문입니다.

      토크나이저 사전  = 사전학습(위키피디아 등) 때 만든 32,000개 한국어 서브워드
                        → 한국어면 뭐든 다 들어 있습니다
      학습 데이터 어휘 = NSMC 영화 리뷰에 실제로 나온 단어들
                        → 도메인 판정에 써야 하는 건 이쪽입니다

  ★ 제 체크는 "이게 영화 리뷰인가?"가 아니라 "이게 한국어인가?"를 재고 있었습니다.

이 스크립트는 NSMC 학습 데이터에서 실제 어휘를 뽑아 JSON으로 저장합니다.
앱은 그 파일만 읽으므로 배포해도 가볍고 빠릅니다.

⏱️ 1~2분 (NSMC 다운로드 포함, 캐시되어 있으면 수초)

실행 : python build_domain_vocab.py
=============================================================================
"""

import json
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "domain_vocab.json"

N_SAMPLES = 20000       # Colab에서 학습에 쓴 것과 같은 규모
MIN_COUNT = 2           # 2번 이상 나온 단어만 (오타·희귀어 제외)


def main():
    from datasets import load_dataset

    print("NSMC 로딩 중...")
    ds = load_dataset("e9t/nsmc", trust_remote_code=True)
    texts = ds["train"].shuffle(seed=42).select(range(N_SAMPLES))["document"]
    print(f"  {len(texts):,}건")

    # 한글 2글자 이상 덩어리만 추출
    counter = Counter()
    for t in texts:
        counter.update(re.findall(r"[가-힣]{2,}", str(t)))

    # 어간 비교를 위해 앞 2글자만 모읍니다.
    # (한국어는 조사·어미가 계속 붙으므로 앞 2글자가 실질적인 어간 역할을 합니다.
    #  STAGE 3에서 '재밌어요'와 '재밌더라구요'를 이렇게 맞춰봤죠.)
    stems = {w[:2] for w, c in counter.items() if c >= MIN_COUNT}

    data = {
        "source": "e9t/nsmc train split",
        "n_samples": len(texts),
        "min_count": MIN_COUNT,
        "n_words": len(counter),
        "stems": sorted(stems),
    }
    OUT.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    print(f"\n  전체 고유 어절 : {len(counter):,}개")
    print(f"  {MIN_COUNT}회 이상    : {sum(1 for c in counter.values() if c >= MIN_COUNT):,}개")
    print(f"  추출된 어간    : {len(stems):,}개")
    print(f"  저장           : {OUT.name} ({OUT.stat().st_size/1024:.0f} KB)")

    # 검증 — 영화 리뷰 문장 vs 도메인 밖 문장
    print("\n  검증")
    tests = [
        ("이건 진짜 내 인생 영화다.. 너무 감동이야.", "영화 리뷰"),
        ("중간에 잠들 뻔했어요. 전개가 너무 느림.", "영화 리뷰"),
        ("오늘 회의는 3시입니다. 자료 첨부드립니다.", "★ 도메인 밖"),
        ("계좌 잔액 조회 결과를 안내드립니다.", "★ 도메인 밖"),
        ("환자의 백혈구 수치가 상승했습니다.", "★ 도메인 밖"),
    ]
    print(f"\n  {'일치율':>7}  {'구분':<12} 문장")
    print("  " + "-" * 70)
    for text, kind in tests:
        words = [w for w in re.findall(r"[가-힣]{2,}", text)]
        hit = sum(1 for w in words if w[:2] in stems)
        ratio = hit / len(words) if words else 0
        mark = "🚩" if ratio < 0.5 else "  "
        print(f"  {ratio:>6.0%} {mark} {kind:<12} {text[:34]}")

    print("""
  → 영화 리뷰는 높게, 도메인 밖은 낮게 나와야 정상입니다.
    (완벽히 갈리진 않습니다. '오늘', '자료' 같은 흔한 단어는 영화 리뷰에도 나오니까요.)
""")


if __name__ == "__main__":
    main()
