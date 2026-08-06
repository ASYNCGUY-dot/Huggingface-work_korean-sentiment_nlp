"""
=============================================================================
STAGE 2 - MISSION 2 : 토크나이저를 직접 학습시키기 (BPE vs WordPiece)
=============================================================================
2일차 수업의 1·2·3번 실습을 하나로 합치고, 결정적인 실험을 하나 추가했습니다.

수업에서 이런 결과가 나왔었죠:
    WordPiece → ['[UNK]', '날', '##씨', '##가', '좋', '[UNK]', '[UNK]', '.']
    "왜 이렇게 [UNK] 투성이지?" → 강사님 설명: "데이터가 너무 적어서"

★ 오늘은 그 설명을 증명합니다.
  같은 코드를 문장 1개 / 문장 60개 두 가지 데이터로 각각 돌려서 비교합니다.

실행 : python 02_bpe_vs_wordpiece.py
=============================================================================
"""

import time
from pathlib import Path

from tokenizers import Tokenizer, models
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, WordPieceTrainer


# -----------------------------------------------------------------------------
# 경로 설정 — 어디서 실행해도 안 깨지는 방식
# -----------------------------------------------------------------------------
# Path(__file__) = 이 .py 파일 자신의 위치
# .resolve()     = 절대경로로 변환
# .parent        = 그 파일이 들어있는 폴더
# → 터미널 현재 위치와 무관하게 항상 같은 곳을 가리킵니다.
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
TEST_SENTENCE = "오늘은 날씨가 좋아서 놀러가기 좋은 날입니다."


# -----------------------------------------------------------------------------
# 토크나이저 학습 함수
# -----------------------------------------------------------------------------
def train_tokenizer(algorithm: str, corpus_path: Path):
    """algorithm: 'bpe' 또는 'wordpiece'. 학습된 토크나이저와 소요 시간을 반환."""

    if algorithm == "bpe":
        # BPE : 가장 자주 붙어 나오는 글자 쌍을 반복해서 합쳐 나감 (빈도 기반)
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            vocab_size=2000,        # 사전 최대 크기
            min_frequency=2,        # ★ 2번 이상 나온 조각만 사전에 등록
            special_tokens=SPECIAL_TOKENS,
        )
    else:
        # WordPiece : 언어모델 확률(likelihood)을 높이는 쌍을 합침. 단어 중간에 ## 표시
        tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        trainer = WordPieceTrainer(
            vocab_size=2000,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
        )

    # 공백(띄어쓰기)으로 먼저 단어를 나눈 뒤, 그 안에서 알고리즘이 조각을 만듭니다.
    tok.pre_tokenizer = Whitespace()

    start = time.time()
    tok.train(files=[str(corpus_path)], trainer=trainer)
    elapsed = time.time() - start

    return tok, elapsed


def char_loss(tok, text):
    """★ 진짜 정보 손실률 — 원문 글자 중 몇 %가 사라졌는가

    왜 이게 필요한가?
      [UNK] '개수'만 세면 속습니다.
      WordPiece는 단어 하나가 통째로 [UNK] 1개가 되고,
      BPE는 모르는 글자만 [UNK]로 남기고 나머지는 살립니다.
      → [UNK] 개수는 WordPiece가 적지만, 실제로 잃은 글자는 더 많을 수 있습니다.

    그래서 "복원했을 때 원문의 몇 글자가 살아남았나"를 직접 셉니다.
    """
    out = tok.encode(text)
    original = text.replace(" ", "")

    # [UNK]가 아닌 조각들에서 '##' 표시를 떼고 이어붙이면 살아남은 글자입니다.
    survived = "".join(
        t.replace("##", "") for t in out.tokens if t != "[UNK]"
    )
    lost = len(original) - len(survived)
    return lost, len(original), lost / len(original) * 100


def report(tok, elapsed, algorithm, corpus_name):
    """학습 결과를 보고합니다."""
    out = tok.encode(TEST_SENTENCE)
    unk = out.tokens.count("[UNK]")
    unk_ratio = unk / len(out.tokens) * 100 if out.tokens else 0
    lost, total, loss_ratio = char_loss(tok, TEST_SENTENCE)

    print(f"\n  [{algorithm.upper()}]  학습 데이터: {corpus_name}")
    print(f"    사전 크기 : {tok.get_vocab_size()}개")
    print(f"    학습 시간 : {elapsed:.4f}초")
    print(f"    조각 수   : {len(out.tokens)}개")
    print(f"    조각      : {out.tokens}")
    print(f"    [UNK] 개수: {unk}개 (전체 조각의 {unk_ratio:.0f}%)")
    print(f"    ★ 실제 손실: 원문 {total}자 중 {lost}자 증발 ({loss_ratio:.0f}%)", end="")
    if loss_ratio > 20:
        print("   🚨")
    elif loss_ratio > 0:
        print("   🟡")
    else:
        print("   ✅")
    return unk_ratio, loss_ratio, tok.get_vocab_size()


# =============================================================================
# 실험 : 작은 데이터 vs 큰 데이터
# =============================================================================
CORPORA = [
    ("문장 1개 (수업과 동일)", DATA / "sample_small.txt"),
    ("문장 60개", DATA / "sample_large.txt"),
]

print("=" * 78)
print(f"테스트 문장 : {TEST_SENTENCE}")
print("=" * 78)

summary = []

for corpus_name, corpus_path in CORPORA:
    if not corpus_path.exists():
        print(f"\n✖ 파일이 없습니다: {corpus_path}")
        continue

    line_count = sum(1 for _ in corpus_path.open(encoding="utf-8"))
    print(f"\n{'─' * 78}")
    print(f"■ 학습 데이터: {corpus_name}  ({line_count}줄)")
    print(f"{'─' * 78}")

    for algo in ("bpe", "wordpiece"):
        tok, elapsed = train_tokenizer(algo, corpus_path)
        unk_ratio, loss_ratio, vocab = report(tok, elapsed, algo, corpus_name)
        summary.append((corpus_name, algo, unk_ratio, loss_ratio, vocab))

        # 학습된 토크나이저를 파일로 저장 (재사용 가능)
        save_name = f"{algo}_{corpus_path.stem}.json"
        tok.save(str(DATA / save_name))


# =============================================================================
# 최종 비교표
# =============================================================================
print("\n\n" + "=" * 78)
print("최종 비교")
print("=" * 78)
print(f"\n{'데이터':<22} {'알고리즘':<12} {'사전크기':>8} {'[UNK]비율':>10} {'실제손실':>10}")
print("-" * 78)
for corpus_name, algo, unk_ratio, loss_ratio, vocab in summary:
    print(f"{corpus_name:<22} {algo:<12} {vocab:>8} {unk_ratio:>9.0f}% {loss_ratio:>9.0f}%")

print("""
  ★ 두 지표를 나란히 보세요. 순위가 뒤집히는 경우가 있습니다.
    [UNK] 비율은 WordPiece가 낮은데, 실제 손실은 BPE가 낮을 수 있습니다.

    이유: WordPiece는 단어 하나가 통째로 [UNK] 1개가 됩니다.
          '놀러가기' 4글자가 [UNK] 1개 → 개수는 1개지만 4글자를 잃음.
          BPE는 모르는 글자만 [UNK]로 남기고 '가','기'는 살려냅니다.
          → [UNK] 2개지만 잃은 건 2글자.

    💼 실무 교훈: 지표를 잘못 고르면 정반대 결론이 나옵니다.
       "무엇을 세고 있는가"를 항상 되물으세요.
       (STAGE 4의 '정확도의 함정'과 똑같은 구조입니다)
""")


# =============================================================================
# 🎓 결과를 보고 이해할 것
# =============================================================================
# 1. 데이터가 60배로 늘면 [UNK] 비율이 어떻게 변했나요?
#    → 수업에서 [UNK]가 쏟아졌던 건 알고리즘 탓이 아니라 데이터 탓이었습니다.
#
# 2. min_frequency=2 의 의미를 다시 보세요.
#    "2번 이상 나온 조각만 사전에 등록"
#    문장이 1개면? 대부분의 글자가 딱 1번만 나옵니다 → 전부 탈락 → [UNK]
#
# 3. BPE와 WordPiece의 조각 모양 차이를 보세요.
#    WordPiece 는 단어 중간 조각에 ## 이 붙습니다.
#    '날' + '##씨'  =  "이 조각은 앞 조각에 이어 붙은 것"이라는 표시입니다.
#    BPE 는 이 표시가 없어서 조각들이 평면적입니다.
#
# -----------------------------------------------------------------------------
# 💼 실무 관점 — 그래서 토크나이저를 직접 학습할 일이 있나?
# -----------------------------------------------------------------------------
#   거의 없습니다. 이유는 두 가지입니다.
#     ① 제대로 만들려면 수백만 문장이 필요합니다 (오늘 60문장으로도 부족했죠)
#     ② ★★ 결정적 이유: 토크나이저는 모델과 한 몸입니다.
#        모델은 "3번 ID = 안녕" 이라고 외운 상태입니다.
#        토크나이저만 새로 만들면 번호 체계가 어긋나 모델이 전부 오작동합니다.
#
#   그래서 실무에서는 AutoTokenizer.from_pretrained() 로 기성품을 씁니다.
#   직접 학습하는 경우는 "모델도 처음부터 학습할 때" 뿐입니다.
#
#   ※ 어제 배운 "tokenizer 인자를 다르게 주면 성능이 붕괴한다"의 진짜 이유가
#     바로 이 번호 체계 불일치입니다. 이제 연결되시죠?
#
# -----------------------------------------------------------------------------
# 🎯 미션
# -----------------------------------------------------------------------------
#  1. min_frequency 를 1로 바꾸고 sample_small.txt 로 다시 돌려보세요.
#     [UNK]가 사라지나요? 사전 크기는 어떻게 되나요?
#     → 그럼 항상 1로 두면 되지 않냐고요? 노이즈(오타·희귀문자)까지 전부
#       사전에 들어가 사전이 비대해집니다. 그래서 보통 2 이상을 씁니다.
#
#  2. data/sample_large.txt 를 열어 본인 문장 20개를 추가하고 다시 실행하세요.
#     [UNK]와 사전 크기가 어떻게 변하나요?
#
#  3. ★ 저장된 json 파일을 열어보세요 (data/bpe_sample_large.json).
#     "vocab" 항목에 조각과 번호가 어떻게 짝지어져 있는지 확인해보세요.
#     이게 바로 "토크나이저 사전"의 실체입니다.
# =============================================================================
