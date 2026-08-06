"""
=============================================================================
STAGE 2 - BOSS BATTLE : CSV에서 텐서까지, 전처리 전 과정
=============================================================================
2일차 수업의 9번(데이터 생성) + 10번(전처리 파이프라인)을 합치고,
3일차 수업에서 강사님이 지적하신 데이터 품질 문제(중복·편향)를 미리 막았습니다.

    ① CSV 만들기 (중복X, 50:50 균형)
    ② pandas → Dataset 변환
    ③ 학습/검증 분리
    ④ 토크나이징 (batched=True)
    ⑤ 불필요 컬럼 제거
    ⑥ 텐서 변환
    ⑦ 디스크에 저장   ← 이 결과물이 STAGE 3의 입력이 됩니다

실행 : python 05_full_pipeline.py
=============================================================================
"""

import itertools
import random
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA / "movie.csv"
SAVE_PATH = DATA / "prepared_movie_dataset"
MODEL_NAME = "bert-base-multilingual-cased"
TOTAL = 600            # 긍정 300 + 부정 300

# ★ MAX_LENGTH 를 64 → 32 로 낮췄습니다 (2026-08-05 수정)
#   이 데이터의 실제 최대 길이는 25토큰입니다. 64는 과했습니다.
#   트랜스포머의 어텐션 연산량은 길이의 '제곱'에 비례합니다.
#       32 → 32² = 1,024
#       64 → 64² = 4,096   ← 4배 더 계산하고 결과는 똑같음
#   ※ 32는 25보다 여유가 있고 2의 거듭제곱이라 GPU 효율도 좋습니다.
MAX_LENGTH = 32
SEED = 42

random.seed(SEED)


# =============================================================================
# ① CSV 만들기 — 중복 없이, 정확히 50:50
# =============================================================================
POS = {
    "subjects": ["이 영화", "배우들의 연기", "스토리", "연출", "영상미",
                 "OST", "결말", "각본", "주인공", "캐릭터"],
    "adjectives": ["정말 최고예요", "완전 대박이네요", "감동적입니다", "기대 이상이에요",
                   "몰입감이 엄청나요", "여운이 남네요", "너무 재밌어요", "훌륭합니다"],
    "finishers": ["꼭 보세요!", "강력 추천합니다.", "인생 영화 등극!", "다시 보고 싶어요.",
                  "박수를 보냅니다.", "주변에 추천할게요.", "최고의 작품입니다."],
}

NEG = {
    "subjects": ["이 영화", "스토리", "전개", "연기력", "결말",
                 "내용", "연출", "주인공", "CG", "대사"],
    "adjectives": ["너무 뻔해서 지루했어요", "시간 아까워요", "실망스럽네요", "이해가 안 가요",
                   "집중이 안 돼요", "돈 아깝습니다", "어색하네요", "개연성이 없어요"],
    "finishers": ["절대 보지 마세요.", "비추천합니다.", "중도 하차할 뻔했네요.", "별로예요.",
                  "최악의 영화입니다.", "돈 돌려받고 싶네요.", "시간 낭비입니다."],
}


def make_sentences(templates, count):
    """★ 중복을 '원천 차단'하는 방법

    random.choice 를 반복하면 우연히 같은 조합이 또 나올 수 있습니다.
    itertools.product 로 가능한 모든 조합(10×8×7=560개)을 미리 만들어 두고
    거기서 필요한 개수만 random.sample 로 뽑으면 중복이 구조적으로 불가능합니다.
    """
    combos = itertools.product(
        templates["subjects"], templates["adjectives"], templates["finishers"]
    )
    sentences = [f"{s} {a}. {f}" for s, a, f in combos]

    if count > len(sentences):
        raise ValueError(
            f"요청 {count}개 > 생성 가능 최대 {len(sentences)}개. 단어 목록을 늘리세요."
        )
    return random.sample(sentences, count)


print("=" * 78)
print("① CSV 생성")
print("=" * 78)

half = TOTAL // 2
rows = [[s, 1] for s in make_sentences(POS, half)] + \
       [[s, 0] for s in make_sentences(NEG, half)]

df = pd.DataFrame(rows, columns=["content", "label"])

# ★ 셔플 — 긍정이 앞에 몰려 있으면 학습이 요동칩니다 (MISSION 4에서 배운 그 이유)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# encoding="utf-8-sig" : 엑셀에서 한글이 안 깨지게 (STAGE 1에서 확인했던 그것)
df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

print(f"\n  저장   : {CSV_PATH.name}")
print(f"  총 건수 : {len(df)}건")
print(f"  라벨    : 긍정 {(df.label == 1).sum()}건 / 부정 {(df.label == 0).sum()}건")
print(f"  중복    : {df.content.duplicated().sum()}건  ← 0이어야 정상")
print(f"\n  샘플 3건:")
for _, r in df.head(3).iterrows():
    print(f"    [{r.label}] {r.content}")

# -----------------------------------------------------------------------------
# ★ 데이터 품질 진단 — 문장은 안 겹치는데 '단어'는 겹친다
# -----------------------------------------------------------------------------
# 중복 0건이라고 안심하면 안 됩니다.
# 템플릿 조합으로 만든 데이터는 문장은 다 다르지만 부품이 몇 개 안 됩니다.
all_words = [w for s in df.content for w in s.split()]
unique_words = set(all_words)
ttr = len(unique_words) / len(all_words)      # Type-Token Ratio (어휘 다양성)

print(f"\n  📊 어휘 다양성 진단")
print(f"    전체 단어 {len(all_words):,}개 중 고유 단어 {len(unique_words)}개")
print(f"    TTR(어휘 다양성) = {ttr:.3f}")
if ttr < 0.1:
    print(f"    🚨 매우 낮습니다. 같은 단어가 평균 {len(all_words)/len(unique_words):.0f}번씩 반복됩니다.")
    print("       → 모델이 '감정'이 아니라 '이 문구가 나오면 0번'을 외울 위험이 큽니다.")
    print("       → STAGE 3에서 이 문제가 실제로 터지는 걸 보게 됩니다. 기억해두세요.")


# =============================================================================
# ② CSV 다시 읽기 → 허깅페이스 Dataset 으로 변환
# =============================================================================
print("\n\n" + "=" * 78)
print("② pandas → Dataset 변환")
print("=" * 78)

# 실무에서는 이 줄이 시작점입니다 (남이 준 CSV를 읽는 것)
df_loaded = pd.read_csv(CSV_PATH)
raw = Dataset.from_pandas(df_loaded)

print(f"\n{raw}")
print("\n  왜 변환하나? pandas 로는 map(batched=True), save_to_disk 같은")
print("  허깅페이스 전용 기능을 쓸 수 없기 때문입니다.")


# =============================================================================
# ③ 학습 / 검증 분리
# =============================================================================
print("\n\n" + "=" * 78)
print("③ 8:2 분리")
print("=" * 78)

# ★ stratify_by_column 으로 라벨 비율을 강제 유지 (MISSION 4에서 배운 그것)
#   label 이 CSV에서 온 int 컬럼이므로 먼저 ClassLabel 타입으로 변환해야 합니다.
raw = raw.class_encode_column("label")
split = raw.train_test_split(test_size=0.2, seed=SEED, stratify_by_column="label")

original_ratio = sum(raw["label"]) / len(raw)
print(f"\n  원본 긍정 비율 {original_ratio:.1%}  ← 이 값을 유지해야 정상")
print(f"  학습 {len(split['train'])}건 / 검증 {len(split['test'])}건")
for name in ("train", "test"):
    lb = split[name]["label"]
    ratio = sum(lb) / len(lb)
    flag = "✅" if abs(ratio - original_ratio) < 0.1 else "🚨"
    print(f"  {name:<6} 긍정 비율 {ratio:.1%}  {flag}")


# =============================================================================
# ④ 토크나이징
# =============================================================================
print("\n\n" + "=" * 78)
print("④ 토크나이징")
print("=" * 78)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 적정 max_length 를 먼저 측정 (MISSION 3에서 배운 방법)
lengths = sorted(len(tokenizer.encode(t)) for t in df_loaded["content"])
p95 = lengths[int(len(lengths) * 0.95) - 1]
print(f"\n  토큰 길이: 최소 {lengths[0]} / 95%지점 {p95} / 최대 {lengths[-1]}")
print(f"  → max_length={MAX_LENGTH} 로 설정 (95% 지점 {p95}보다 여유 있음)")


def preprocess(examples):
    """batched=True 이므로 examples['content'] 는 리스트로 들어옵니다."""
    return tokenizer(
        examples["content"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


tokenized = split.map(preprocess, batched=True)
print(f"\n  변환 후 컬럼 : {tokenized['train'].column_names}")
print("    input_ids      : 조각의 번호")
print("    token_type_ids : 문장 구분 (한 문장이면 전부 0)")
print("    attention_mask : 진짜 내용 1 / 빈칸 0")


# =============================================================================
# ⑤ 불필요 컬럼 제거
# =============================================================================
print("\n\n" + "=" * 78)
print("⑤ 원본 텍스트 컬럼 제거")
print("=" * 78)

tokenized = tokenized.remove_columns(["content"])
print(f"\n  남은 컬럼 : {tokenized['train'].column_names}")
print("""
  🚨 이 단계를 빼먹으면 STAGE 3에서 학습이 에러로 멈춥니다.
     모델은 숫자만 받습니다. 글자로 된 'content'는 모델 입장에서 이물질입니다.
""")


# =============================================================================
# ⑥ 텐서 변환
# =============================================================================
print("=" * 78)
print("⑥ PyTorch 텐서로 변환")
print("=" * 78)

tokenized.set_format("torch")

s = tokenized["train"][0]
print(f"\n  label 타입      : {type(s['label'])}")
print(f"  input_ids 타입  : {type(s['input_ids'])}")
print(f"  input_ids 크기  : {s['input_ids'].shape}   ← 토큰 {MAX_LENGTH}개")
print("""
  왜 텐서여야 하나? (4가지)
    ① 모델은 파이썬 리스트로 연산할 수 없습니다
    ② GPU 병렬 계산이 텐서에서만 가능합니다
    ③ 학습에 필요한 자동 미분(기울기 계산)이 텐서 전용입니다
    ④ 여러 문장을 한 번에 행렬로 묶어 처리해야 빠릅니다
""")


# =============================================================================
# ⑦ 저장 — STAGE 3의 입력물
# =============================================================================
print("=" * 78)
print("⑦ 디스크에 저장")
print("=" * 78)

tokenized.save_to_disk(str(SAVE_PATH))

print(f"\n  ✅ 저장 완료 : {SAVE_PATH}")
print("""
  이 폴더가 STAGE 3(파인튜닝)의 재료입니다.
  다시 불러올 때는:

      from datasets import load_from_disk
      ds = load_from_disk("data/prepared_movie_dataset")

  ★ 전처리를 한 번만 하고 저장해두는 이유:
    토크나이징은 데이터가 커지면 몇 분씩 걸립니다.
    학습 파라미터를 바꿔가며 10번 실험한다고 매번 다시 할 순 없죠.
    "전처리 한 번 → 저장 → 학습은 여러 번" 이 실무 표준 흐름입니다.
""")

print("=" * 78)
print("🏆 BOSS 클리어 — 데이터가 모델에 들어갈 준비를 마쳤습니다")
print("=" * 78)


# =============================================================================
# 🎯 보스전 미션
# =============================================================================
#  [★]    TOTAL 을 600 → 1000 으로 바꿔 실행하세요. 에러가 나나요?
#         (힌트: 생성 가능한 최대 조합은 10×8×7 = 560개. 한쪽당 500개면?)
#         에러가 난다면 단어 목록을 늘려서 해결해보세요.
#
#  [★★]  MAX_LENGTH 를 64 → 16 으로 줄이고 실행한 뒤,
#         저장된 데이터를 복원해서 문장이 잘렸는지 확인해보세요.
#
#             ids = tokenized["train"][0]["input_ids"]
#             print(tokenizer.decode(ids, skip_special_tokens=True))
#
#  [★★★] data/movie.csv 를 엑셀로 열어 본인 문장 20개를 직접 추가한 뒤,
#         ① 단계를 건너뛰고 ②단계부터 실행되도록 코드를 수정해보세요.
#         (힌트: CSV 생성 부분을 if CSV_PATH.exists(): 로 감싸기)
#         → 실무에서 "남이 준 CSV로 시작하는" 상황이 바로 이겁니다.
# =============================================================================
