"""
=============================================================================
STAGE 2 - MISSION 3 : 특수 토큰과 패딩 — 모델에 들어가기 직전의 모습
=============================================================================
MISSION 1~2에서는 "조각내기"만 봤습니다.
하지만 모델에 실제로 들어가는 건 그게 아닙니다. 포장이 더 필요합니다.

    조각      →  [CLS] 조각들 [SEP]  →  길이 맞추기(PAD)  →  attention_mask
    (M1~M2)        (이번 미션)            (이번 미션)         (이번 미션)

실행 : python 03_special_tokens.py
=============================================================================
"""

from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL)


# =============================================================================
# [1] 특수 토큰 5인방 확인
# =============================================================================
print("=" * 78)
print("[1] 이 토크나이저의 특수 토큰")
print("=" * 78)

specials = [
    ("[PAD]", tokenizer.pad_token, tokenizer.pad_token_id, "길이 맞추기용 빈칸"),
    ("[UNK]", tokenizer.unk_token, tokenizer.unk_token_id, "사전에 없는 조각"),
    ("[CLS]", tokenizer.cls_token, tokenizer.cls_token_id, "문장 시작 + 전체 요약 자리"),
    ("[SEP]", tokenizer.sep_token, tokenizer.sep_token_id, "문장 끝 / 문장 경계"),
    ("[MASK]", tokenizer.mask_token, tokenizer.mask_token_id, "가리기 (학습용)"),
]

print(f"\n{'이름':<10} {'실제 토큰':<10} {'ID':>6}   역할")
print("-" * 78)
for name, token, tid, role in specials:
    print(f"{name:<10} {str(token):<10} {tid:>6}   {role}")

print(f"\n전체 사전 크기 : {tokenizer.vocab_size:,}개")


# =============================================================================
# [2] tokenize() vs encode() — 무엇이 다른가
# =============================================================================
print("\n\n" + "=" * 78)
print("[2] 조각내기만 vs 모델에 넣을 준비까지")
print("=" * 78)

text = "오늘 회의는 3시입니다."

only_tokens = tokenizer.tokenize(text)                 # 순수 조각만
full_ids = tokenizer.encode(text)                      # 특수토큰 포함 ID
back_to_tokens = tokenizer.convert_ids_to_tokens(full_ids)

print(f"\n원문            : {text}")
print(f"tokenize()      : {only_tokens}")
print(f"encode()        : {full_ids}")
print(f"  → 다시 조각으로 : {back_to_tokens}")
print("\n  ★ 앞뒤에 [CLS] 와 [SEP] 이 자동으로 붙었습니다.")
print("    [CLS] 자리의 출력값이 '문장 전체의 요약'이 되고,")
print("    분류 모델은 오직 그 자리 하나만 보고 판단합니다.")


# =============================================================================
# [3] ★ 패딩과 attention_mask — 여기가 이번 미션의 핵심
# =============================================================================
print("\n\n" + "=" * 78)
print("[3] 길이가 다른 문장들을 한 배치로 묶기")
print("=" * 78)

sentences = [
    "반갑습니다.",                              # 짧음
    "허깅페이스 데이터셋 전처리를 배우는 중입니다.",   # 김
]

# padding="max_length" : 짧은 문장을 [PAD]로 채워 길이를 max_length에 맞춤
# truncation=True      : max_length보다 길면 잘라냄
# max_length=12        : 기준 길이
encoded = tokenizer(
    sentences,
    padding="max_length",
    truncation=True,
    max_length=12,
)

for i, sent in enumerate(sentences):
    ids = encoded["input_ids"][i]
    mask = encoded["attention_mask"][i]
    toks = tokenizer.convert_ids_to_tokens(ids)

    print(f"\n  문장 {i + 1} : {sent}")
    print(f"    조각           : {toks}")
    print(f"    input_ids      : {ids}")
    print(f"    attention_mask : {mask}")

    real = sum(mask)
    pad = len(mask) - real
    print(f"    → 진짜 내용 {real}칸 + 빈칸 {pad}칸")

print("\n" + "-" * 78)
print("  🔑 attention_mask 를 이해하는 법")
print("-" * 78)
print("""
   1 = "여기 진짜 내용 있어. 읽어."
   0 = "여기는 자리 맞추려고 넣은 빈칸이야. 무시해."

   왜 필요한가?
     모델은 행렬 연산을 하려고 모든 문장을 같은 길이로 만듭니다.
     그런데 [PAD]도 엄연히 숫자(0번)입니다. 그냥 두면 모델이
     "빈칸 빈칸 빈칸"도 의미 있는 신호로 읽어버립니다.
     attention_mask 가 "저건 보지 마"라고 알려주는 역할입니다.

   ★ 실무에서 결과가 이상할 때 의심 순위 상위권입니다.
     직접 텐서를 만들다가 attention_mask 를 빼먹으면
     짧은 문장일수록 성능이 이상하게 떨어집니다.
""")


# =============================================================================
# [4] max_length 를 잘못 잡으면 생기는 일
# =============================================================================
print("\n" + "=" * 78)
print("[4] truncation — 길이를 너무 짧게 잡으면?")
print("=" * 78)

long_text = (
    "이 제품은 배송도 빠르고 포장도 깔끔했지만 "
    "정작 품질이 기대에 한참 못 미쳐서 결국 환불했습니다."
)

# 원문이 몇 토큰인지 먼저 확인
full_len = len(tokenizer.encode(long_text))
print(f"\n  원문 전체 : {long_text}")
print(f"  원문 토큰 수 : {full_len}개")

# 부정의 근거가 되는 핵심 단어들 — 잘렸을 때 살아남는지 추적합니다
KEY_NEGATIVE = ["못", "미쳐", "환불"]

for length in (10, 20, 40):
    ids = tokenizer.encode(long_text, truncation=True, max_length=length)
    restored = tokenizer.decode(ids, skip_special_tokens=True)

    survived = [w for w in KEY_NEGATIVE if w in restored]
    lost = [w for w in KEY_NEGATIVE if w not in restored]

    print(f"\n  max_length={length:>2} → {restored}")
    if not survived:
        print(f"     🚨 부정의 근거가 전부 소실 (잃은 단어: {', '.join(lost)})")
        print("        → 남은 문장만 읽으면 긍정으로 보입니다")
    elif lost:
        print(f"     🟡 일부 소실 (남음: {', '.join(survived)} / 잃음: {', '.join(lost)})")
    else:
        print("     ✅ 부정의 근거가 모두 보존되었습니다")

print("""
  ★ 왜 한국어에서 특히 위험한가

    영어 : "I do NOT like this product because ..."
           → 부정어(not)가 문장 앞쪽에 옵니다. 잘라도 살아남습니다.

    한국어 : "이 제품은 배송도 빠르고 포장도 깔끔했지만 ... 환불했습니다."
           → 결론(서술어)이 맨 끝에 옵니다. 뒤를 자르면 결론이 통째로 사라집니다.

    ★ 앞부분만 남으면 칭찬만 남습니다. 모델은 그걸 긍정으로 읽습니다.

  💡 max_length 정하는 법
     데이터의 토큰 길이 분포를 먼저 재고, 95%가 들어가는 값으로 잡으세요.
     무작정 512(기본값)로 하면 메모리와 시간을 몇 배로 낭비합니다.
     반대로 너무 짧게 잡으면 위에서 본 것처럼 결론이 잘려 나갑니다.
""")


# =============================================================================
# [5] ★ 조용한 손실 — 잘렸는지 알려주는 신호가 없다
# =============================================================================
print("=" * 78)
print("[5] truncation 은 경고를 주지 않는다")
print("=" * 78)

check_sentences = [
    "반갑습니다.",
    "허깅페이스 데이터셋 전처리를 배우는 중입니다.",
]
LIMIT = 12

print(f"\n  max_length={LIMIT} 로 처리할 때 각 문장의 운명:\n")
for s in check_sentences:
    need = len(tokenizer.encode(s))                      # 자르지 않았을 때 필요한 길이
    ids = tokenizer.encode(s, truncation=True, max_length=LIMIT)
    restored = tokenizer.decode(ids, skip_special_tokens=True)

    if need > LIMIT:
        print(f"  🚨 잘림 : {s}")
        print(f"       필요 {need}토큰 > 한도 {LIMIT}토큰  ({need - LIMIT}토큰 소실)")
        print(f"       남은 것 : {restored}")
    else:
        print(f"  ✅ 온전 : {s}  (필요 {need}토큰)")

print("""
  ★ 이게 무서운 이유
    truncation=True 는 조용히 자릅니다. 에러도, 경고도 없습니다.
    attention_mask 도 전부 1이라 "꽉 찼다"고만 보입니다.
    → 데이터 1만 건 중 3천 건이 잘려 나가도 아무도 모릅니다.

  💼 실무 습관 : 전처리 후 반드시 이 한 줄을 찍어보세요.

      over = sum(1 for t in texts if len(tokenizer.encode(t)) > MAX_LEN)
      print(f"잘린 문장: {over}건 / 전체 {len(texts)}건")
""")


# =============================================================================
# 🎯 미션
# =============================================================================
# 1. sentences 에 본인 문장 3개를 추가하고 max_length 를 조절해보세요.
#    attention_mask 의 1과 0이 어떻게 바뀌나요?
#
# 2. ★ 아래 코드로 내 데이터의 적정 max_length 를 재보세요. (실무 필수 기술)
#
#       lengths = [len(tokenizer.encode(s)) for s in sentences]
#       lengths.sort()
#       p95 = lengths[int(len(lengths) * 0.95) - 1]
#       print(f"최대 {max(lengths)} / 95% 지점 {p95}")
#
# 3. ★★ [MASK] 실험 — BERT가 어떻게 학습했는지 체험해보세요.
#
#       from transformers import pipeline
#       fill = pipeline("fill-mask", model="bert-base-multilingual-cased")
#       for r in fill("오늘 날씨가 정말 [MASK]."):
#           print(r["token_str"], round(r["score"], 4))
#
#    빈칸에 뭘 넣을지 모델이 맞힙니다. 이게 BERT의 원래 학습 방식입니다.
# =============================================================================
