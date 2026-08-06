"""
=============================================================================
STAGE 3 - BOSS BATTLE : 진짜 사람이 쓴 데이터로 다시 도전
=============================================================================
지금까지의 결론:
    문제는 모델이 아니라 데이터였다.
    템플릿 조합 데이터는 어휘가 좁아 사각지대를 만든다.

그럼 증명해봅시다.
    같은 모델 + 같은 코드 + 다른 데이터  →  결과가 달라지는가?

이번엔 사람이 실제로 쓴 리뷰로 학습합니다.

⏱️ 예상 소요 (CPU 기준)
    기본 설정(N_SAMPLES=1500, EPOCHS=2) → 약 8~15분
    시간이 부족하면 아래 N_SAMPLES 를 800으로 줄이세요.

실행 : python 05_boss_real_data.py
=============================================================================
"""

import re

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, Trainer,
)

from _common import (
    DATA, MODEL_NAME, REAL_SENTENCES,
    check_env, compute_metrics, make_training_args, ttr,
)


# =============================================================================
# ⚙️ 설정 — 프리셋 중 하나를 고르세요
# =============================================================================
# 2026-08-05 실측 (Windows CPU, batch=8, max_length=64) : 약 1.6초/스텝
#   총 스텝 = (문장수 × 0.8 ÷ batch) × epochs
#
#   ❗ 처음 설정(1500/2ep/64)으로 돌렸더니 검증 70%, 실전 50%가 나왔습니다.
#     원인은 '데이터가 나빠서'가 아니라 '학습이 부족해서'였습니다.
#     - loss 0.70 → 0.50 (무작위 ln2=0.693 에서 겨우 벗어남)
#     - grad_norm 이 3.6 → 16.0 으로 오히려 증가 (아직 배우는 중이라는 뜻)
#     - 어휘 8,172개를 1,200문장으로 배우려니 단어당 노출이 1.2회뿐
#
#     ★ 데이터를 어렵게 만들었으면 학습도 그만큼 늘려야 합니다.

PRESET = "균형"        # "빠름" / "균형" / "제대로" 중 선택

PRESETS = {
    #          문장수, 에폭, 최대길이, 학습률, 예상 소요(CPU)
    "빠름":   (1500,  2,  64, 2e-5, "약 8분   — 학습 부족. 흐름만 볼 때"),
    "균형":   (3000,  3,  96, 3e-5, "약 45분  — 의미 있는 성능이 나오기 시작"),
    "제대로": (8000,  3,  96, 3e-5, "약 2시간 — CPU로는 비추천. Colab GPU 권장"),
}
N_SAMPLES, EPOCHS, MAX_LENGTH, LR, _EST = PRESETS[PRESET]
BATCH = 8
WARMUP_RATIO = 0.1      # 초반에 학습률을 서서히 올려 학습 안정화
SEED = 42

print(f"\n⚙️  프리셋 '{PRESET}' : 문장 {N_SAMPLES} / {EPOCHS}에폭 / 길이 {MAX_LENGTH} / lr {LR}")
print(f"   예상 소요: {_EST}")
print("""
   ⏰ 너무 오래 걸리면 PRESET 을 "빠름" 으로 바꾸세요.
      단, "빠름"은 학습이 부족해 성능이 낮게 나옵니다. 그것도 하나의 교훈입니다.

   💡 강의노트에도 나온 이야기: "300→500→1000 ... 콜랩(GPU)을 사용"
      CPU 학습의 한계를 느끼셨다면, 그게 사람들이 GPU를 쓰는 이유입니다.
      Google Colab 무료 GPU면 이 학습이 1~2분에 끝납니다.
""")

BOSS_MODEL_OUT = DATA / "real_data_model"


# =============================================================================
# [1] 진짜 데이터 확보
# =============================================================================
print("=" * 78)
print("[1] 실제 사람이 쓴 리뷰 데이터 확보")
print("=" * 78)

# 여러 후보를 순서대로 시도합니다.
# NSMC(네이버 영화리뷰 20만건)가 1순위, 안 되면 rotten_tomatoes(영어)로 대체.
CANDIDATES = [
    ("e9t/nsmc", "document", "label", "한국어 · 네이버 영화 리뷰"),
    ("nsmc", "document", "label", "한국어 · 네이버 영화 리뷰"),
    ("rotten_tomatoes", "text", "label", "영어 · 로튼토마토 영화 리뷰"),
]

dataset = text_col = label_col = source_name = None
for name, tcol, lcol, desc in CANDIDATES:
    print(f"\n  시도 중: {name} ...")

    # ★ NSMC 같은 옛날 데이터셋은 '로딩 스크립트(nsmc.py)' 방식입니다.
    #   datasets 3.x 부터는 보안상 trust_remote_code=True 를 명시해야 실행됩니다.
    #   (남의 파이썬 코드를 내려받아 실행하는 것이므로 동의를 받는 절차입니다)
    #   → 먼저 기본 방식, 실패하면 trust_remote_code=True 로 재시도합니다.
    for kwargs in ({}, {"trust_remote_code": True}):
        try:
            ds = load_dataset(name, **kwargs)
            dataset, text_col, label_col = ds, tcol, lcol
            source_name = f"{name} ({desc})"
            extra = " (trust_remote_code=True 적용)" if kwargs else ""
            print(f"  ✅ 로드 성공{extra}")
            break
        except Exception as e:
            msg = str(e).split("\n")[0][:100]
            print(f"  ❌ 실패{' (trust_remote_code)' if kwargs else ''}: {type(e).__name__}: {msg}")
    if dataset is not None:
        break

if dataset is None:
    print("""
  ❌ 데이터셋을 하나도 불러오지 못했습니다.
     네트워크 또는 방화벽 문제일 수 있습니다.
     STAGE 2에서 rotten_tomatoes 를 받은 적이 있다면 캐시에 있을 수 있으니
     인터넷 연결을 확인하고 다시 시도해보세요.
""")
    raise SystemExit(1)

train_raw = dataset["train"]
print(f"\n  데이터 출처 : {source_name}")
print(f"  전체 건수   : {len(train_raw):,}건")

# ★ 반드시 섞고 나서 자른다 (STAGE 2에서 배운 그 교훈)
train_raw = train_raw.shuffle(seed=SEED).select(range(min(N_SAMPLES, len(train_raw))))


# =============================================================================
# [2] ★ 데이터 품질 비교 — 이게 이번 미션의 핵심
# =============================================================================
print("\n" + "=" * 78)
print("[2] 템플릿 데이터 vs 실제 데이터")
print("=" * 78)

texts = [str(t) for t in train_raw[text_col]]
real_ttr, real_uniq, real_total = ttr(texts)

print(f"""
  {'':<22} {'템플릿(STAGE 2)':>18} {'실제 데이터':>16}
  {'-' * 60}
  {'문장 수':<22} {'600':>18} {len(texts):>16,}
  {'고유 어절':<22} {'74':>18} {real_uniq:>16,}
  {'어휘 다양성(TTR)':<20} {'0.023':>18} {real_ttr:>16.3f}
""")

# 기본 감정 단어가 이번엔 있는가?
joined = " ".join(texts)
BASIC = ["좋", "싫", "눈물", "감동", "재밌", "행복", "슬프", "최고", "별로"] \
    if "nsmc" in source_name else ["good", "bad", "love", "boring", "perfect", "worst"]

print("  📖 기본 감정 단어 포함 여부")
for w in BASIC:
    n = joined.count(w)
    print(f"    '{w}' {'✅' if n else '❌'}  ({n}회 등장)")

print(f"""
  ★ TTR 이 {real_ttr / 0.023:.0f}배 높습니다.
    같은 문장 수라도 훨씬 다양한 표현이 들어 있다는 뜻입니다.
    STAGE 2의 템플릿 데이터는 부품 25개로 600문장을 만들었죠.
""")


# =============================================================================
# [3] 전처리
# =============================================================================
print("=" * 78)
print("[3] 전처리")
print("=" * 78)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 토큰 길이를 재서 max_length 근거 만들기 (STAGE 2에서 배운 습관)
lengths = sorted(len(tokenizer.encode(t)) for t in texts[:500])
p95 = lengths[int(len(lengths) * 0.95) - 1]
over = sum(1 for l in lengths if l > MAX_LENGTH)
print(f"\n  토큰 길이: 95%지점 {p95} / 최대 {lengths[-1]}")
print(f"  max_length={MAX_LENGTH} 로 자르면 표본 500건 중 {over}건({over/5:.0f}%)이 잘립니다")


def preprocess(examples):
    return tokenizer(
        [str(t) for t in examples[text_col]],
        truncation=True, padding="max_length", max_length=MAX_LENGTH,
    )


# 라벨 컬럼 이름을 'labels' 로 통일 (Trainer 규약)
ds = train_raw.map(preprocess, batched=True)
keep = ["input_ids", "attention_mask", "label"]
if "token_type_ids" in ds.column_names:
    keep.append("token_type_ids")
ds = ds.remove_columns([c for c in ds.column_names if c not in keep])

# ★ label 컬럼 타입을 확인하고 필요할 때만 변환합니다.
#   (2026-08-05 수정: 무조건 class_encode_column 을 부르면
#    rotten_tomatoes 처럼 이미 ClassLabel 인 데이터에서 ValueError 가 납니다.
#      "Class encoding is only supported for Value column,
#       and column label is ClassLabel.")
#
#   허깅페이스 허브 데이터셋 → 보통 이미 ClassLabel (변환 불필요)
#   내가 만든 CSV          → int64 (변환 필요)
from datasets import ClassLabel

label_feature = ds.features["label"]
if isinstance(label_feature, ClassLabel):
    print(f"  label 타입: 이미 ClassLabel → 변환 건너뜀")
else:
    print(f"  label 타입: {label_feature} → ClassLabel 로 변환")
    ds = ds.class_encode_column("label")

split = ds.train_test_split(test_size=0.2, seed=SEED, stratify_by_column="label")
split.set_format("torch")

print(f"  학습 {len(split['train'])}건 / 검증 {len(split['test'])}건")


# =============================================================================
# [4] 학습
# =============================================================================
print("\n" + "=" * 78)
print("[4] 학습")
print("=" * 78)
check_env()

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.config.id2label = {0: "부정", 1: "긍정"}
model.config.label2id = {"부정": 0, "긍정": 1}

steps = (len(split["train"]) // BATCH) * EPOCHS
print(f"\n  총 {steps} 스텝. CPU 기준 약 {steps * 2 / 60:.0f}분 예상됩니다.")
print("  ☕ 커피 한 잔 하고 오세요.\n")

args = make_training_args(
    output_dir=str(DATA / "results_real"),
    learning_rate=LR,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    # ★ warmup : 초반 몇 %는 학습률을 0에서 서서히 올립니다.
    #   첫 실행에서 grad_norm 이 3.6→16.0 으로 튀었던 건 학습이 불안정했다는 신호입니다.
    #   warmup 을 주면 초반 급격한 가중치 변동을 줄여 안정적으로 수렴합니다.
    warmup_ratio=WARMUP_RATIO,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    report_to=[],
    seed=SEED,
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=split["train"], eval_dataset=split["test"],
    compute_metrics=compute_metrics,
)
result = trainer.train()
metrics = trainer.evaluate()

print(f"\n  train_loss     : {result.training_loss:.4f}")
print(f"  검증 정확도     : {metrics['eval_accuracy']:.1%}")
print(f"  검증 F1        : {metrics['eval_f1']:.4f}")
print(f"  소요 시간       : {result.metrics['train_runtime']:.0f}초")

# -----------------------------------------------------------------------------
# ★ 학습이 충분했는지 자동 진단
# -----------------------------------------------------------------------------
import math
RANDOM_LOSS = math.log(2)      # 2지선다에서 완전 무작위일 때의 loss = 0.693

final_loss = result.training_loss
acc = metrics["eval_accuracy"]

print(f"\n  📉 진단 (무작위 상태의 loss = {RANDOM_LOSS:.3f})")
if final_loss > RANDOM_LOSS * 0.85:
    print(f"     🚨 학습 부족(underfitting) : loss {final_loss:.3f} 가 무작위({RANDOM_LOSS:.3f})에 너무 가깝습니다.")
    print("        → 데이터를 늘리거나(N_SAMPLES), 에폭을 늘리거나, 학습률을 올리세요.")
    print("        → 파일 상단 PRESET 을 한 단계 위로 바꿔보세요.")
elif final_loss < 0.05:
    print(f"     🚨 과적합 의심 : loss {final_loss:.3f} 는 거의 외운 수준입니다.")
    print("        → 데이터 다양성이 부족하거나 에폭이 과했을 수 있습니다.")
else:
    print(f"     ✅ 적절한 범위 : loss {final_loss:.3f}")

print(f"""
  ★ 정확도가 100% 가 아니라고 실망하지 마세요.
    실제 사람 문장은 애매한 것도, 비꼬는 것도, 오타도 있습니다.
    NSMC 같은 실제 데이터로 제대로 학습하면 보통 85~90% 대가 나옵니다.
    지금 {acc:.0%} 라면 아직 학습이 덜 된 것이지, 데이터가 나쁜 게 아닙니다.

    ★★ 템플릿 데이터의 100% 가 오히려 비정상이었습니다.
       쉬운 문제로 만점 받은 것과, 어려운 문제로 70점 받은 것 중
       실전에서 쓸 수 있는 쪽은 후자입니다. 단, 공부를 더 시켜야 합니다.
""")

model.save_pretrained(BOSS_MODEL_OUT)
tokenizer.save_pretrained(BOSS_MODEL_OUT)
print(f"  ✅ 저장 : {BOSS_MODEL_OUT}")


# =============================================================================
# [5] 최종 대결 — 실전 문장으로
# =============================================================================
print("\n" + "=" * 78)
print("[5] 최종 시험 — B조 실전 문장")
print("=" * 78)

model.eval()


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0]
    p = int(torch.argmax(probs))
    return p, float(probs[p])


is_korean = "nsmc" in source_name
if not is_korean:
    print("""
  ⚠️ 영어 데이터로 학습했으므로 한국어 테스트 문장은 의미가 없습니다.
     대신 영어 문장으로 시험합니다.
""")
    TEST = [
        ("I cried through the whole thing, it was beautiful.", 1),
        ("Best film I have seen all year.", 1),
        ("I almost fell asleep, the pacing was awful.", 0),
        ("Complete waste of money.", 0),
        ("Not bad, but I would not watch it again.", 0),
    ]
else:
    TEST = REAL_SENTENCES

correct = 0
for text, ans in TEST:
    p, c = predict(text)
    ok = p == ans
    correct += ok
    print(f"\n  {'✅' if ok else '❌'} {text}")
    print(f"     정답 {'긍정' if ans else '부정'} / 예측 {'긍정' if p else '부정'} ({c:.1%})")

acc = correct / len(TEST)

print("\n\n" + "=" * 78)
print("최종 성적표")
print("=" * 78)
print(f"""
  템플릿 데이터로 학습한 모델 (01번)  →  B조 정답률 80%
  실제 데이터로 학습한 모델 (지금)     →  B조 정답률 {acc:.0%}

  ★ 같은 모델(bert-base-multilingual-cased)
    같은 코드(Trainer)
    같은 하이퍼파라미터(lr 2e-5, batch 8)
    ─────────────────────────────────
    다른 것은 오직 '데이터' 하나뿐입니다.
""")

print("=" * 78)
print("🏆 BOSS 클리어 — STAGE 3 완주")
print("=" * 78)


# =============================================================================
# 🎯 보스전 미션
# =============================================================================
#  [★]    N_SAMPLES 를 800 → 1500 → 3000 으로 늘려가며 정답률을 기록해보세요.
#         데이터가 늘수록 성능이 어떻게 변하나요? 어디서 멈추나요?
#
#  [★★]  MISSION 3의 '결정적 실험' 문장 쌍을 이 모델에도 넣어보세요.
#           "이 영화 감동적입니다." vs "이 영화 뭉클했습니다."
#           "연출 훌륭합니다."     vs "연출 근사합니다."
#         이번엔 답이 뒤집히나요? 뒤집히지 않는다면 왜일까요?
#
#  [★★★] 이 모델과 01번 모델을 한 화면에서 비교하는 스크립트를 짜보세요.
#         → STAGE 5에서 Gradio 로 만들면 훨씬 보기 좋습니다. 미리 구상해두세요.
# =============================================================================
