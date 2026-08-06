"""
=============================================================================
STAGE 3 - MISSION 1 : 나만의 모델 학습시키기
=============================================================================
드디어 모델을 만듭니다.

    STAGE 1 : 남이 만든 모델을 쓰고 의심했다
    STAGE 2 : 모델에게 줄 밥을 준비했다
    STAGE 3 : 직접 훈련시킨다   ← 지금 여기

★ 무엇을 훈련시키나?
  베이스 모델은 "언어 이해(인코더)"만 갖고 있고 "판단(분류 헤드)"은 없습니다.
  STAGE 1에서 klue/roberta-base 가 정반대 문장에 점수차 0.0038 을 낸 이유죠.
  그 랜덤 주사위를 진짜 판단기로 만드는 과정이 파인튜닝입니다.

⏱️ 예상 소요: CPU 기준 3~6분 (GPU 있으면 30초 이내)

실행 : python 01_train_first_model.py
=============================================================================
"""

from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, Trainer

from _common import (
    MODEL_NAME, MODEL_OUT, PREPARED,
    check_env, compute_metrics, make_training_args,
)


# =============================================================================
# [0] 환경 점검 + 재료 확인
# =============================================================================
check_env()

if not PREPARED.exists():
    print(f"\n❌ 전처리 데이터가 없습니다: {PREPARED}")
    print("   STAGE 2의 05_full_pipeline.py 를 먼저 실행하세요.")
    raise SystemExit(1)

ds = load_from_disk(str(PREPARED))
print(f"\n재료 확인")
print(f"  학습 {len(ds['train'])}건 / 검증 {len(ds['test'])}건")
print(f"  컬럼 {ds['train'].column_names}")


# =============================================================================
# [1] 모델 불러오기
# =============================================================================
print("\n" + "=" * 78)
print("[1] 베이스 모델 로드")
print("=" * 78)

# ★ num_labels=2 : 분류할 클래스 개수 (긍정/부정)
#   이 숫자만큼의 출력 자리를 가진 '분류 헤드'가 새로 만들어집니다.
#   → 이 헤드는 랜덤 초기화 상태입니다. 지금부터 이걸 훈련시킵니다.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

# 라벨 이름 지정 — 이걸 해두면 나중에 LABEL_0 대신 '부정'이 출력됩니다.
# (STAGE 1에서 LABEL_0/LABEL_1 때문에 헷갈렸던 그 문제의 해결책)
model.config.id2label = {0: "부정", 1: "긍정"}
model.config.label2id = {"부정": 0, "긍정": 1}

total_params = sum(p.numel() for p in model.parameters())
print(f"\n  모델      : {MODEL_NAME}")
print(f"  파라미터  : {total_params:,}개")
print(f"  라벨      : {model.config.id2label}")
print("""
  ℹ️ 아래와 비슷한 경고가 떴다면 정상입니다:
     "Some weights ... newly initialized: ['classifier.weight', 'classifier.bias']
      You should probably TRAIN this model on a down-stream task"

     → "분류 헤드를 새로 만들었으니 학습시켜라"는 안내입니다.
       STAGE 1에서 이 경고를 무시하고 바로 썼다가 LABEL_0 사태가 났었죠.
       지금 우리가 하려는 게 바로 그 '학습'입니다.
""")


# =============================================================================
# [2] 학습 설정 (하이퍼파라미터)
# =============================================================================
print("=" * 78)
print("[2] 학습 설정")
print("=" * 78)

EPOCHS = 3
BATCH = 8
LR = 2e-5

args = make_training_args(
    output_dir=str(MODEL_OUT.parent / "results"),   # 중간 저장본 위치
    learning_rate=LR,              # 가중치를 한 번에 얼마나 크게 수정할지
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    num_train_epochs=EPOCHS,       # 전체 데이터를 몇 바퀴 돌 것인가
    weight_decay=0.01,             # 과적합 억제 규제
    eval_strategy="epoch",         # 매 에폭마다 검증 (버전에 따라 자동 변환됨)
    save_strategy="no",            # 실습이므로 중간 저장 생략 (디스크 절약)
    logging_steps=20,
    report_to=[],                  # wandb 등 외부 로깅 비활성화
    seed=42,
)

steps = (len(ds["train"]) // BATCH) * EPOCHS
print(f"""
  learning_rate = {LR}     한 걸음의 크기. BERT 파인튜닝은 2e-5~5e-5가 정석.
                           크면 발산하고, 작으면 학습이 안 됩니다.
  batch_size    = {BATCH}        한 번에 처리할 문장 수. CPU면 8, GPU면 16~32.
  epochs        = {EPOCHS}        전체 데이터를 {EPOCHS}바퀴 돕니다.
  weight_decay  = 0.01     특정 데이터에 집착하지 않도록 거는 브레이크.

  → 총 학습 스텝 : 약 {steps}회
""")


# =============================================================================
# [3] Trainer 조립 및 학습
# =============================================================================
print("=" * 78)
print("[3] 학습 시작")
print("=" * 78)
print("\n  ⏱️ CPU에서는 몇 분 걸립니다. 진행 바가 천천히 가도 정상입니다.\n")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    compute_metrics=compute_metrics,   # 정확도 + F1 계산
)

try:
    result = trainer.train()
except TypeError as e:
    print("\n" + "!" * 78)
    print("❌ TypeError 발생 — transformers 와 accelerate 버전 충돌입니다.")
    print("   수업 노트에도 기록된 그 에러입니다. 아래로 해결하세요:")
    print()
    print("     pip uninstall -y transformers accelerate")
    print("     pip install transformers==4.44.2 accelerate==0.34.0")
    print("!" * 78)
    raise


# =============================================================================
# [4] 결과 확인
# =============================================================================
print("\n" + "=" * 78)
print("[4] 학습 결과")
print("=" * 78)

metrics = trainer.evaluate()
train_loss = result.training_loss

print(f"\n  학습 손실(train_loss) : {train_loss:.4f}")
print(f"  검증 정확도(accuracy) : {metrics['eval_accuracy']:.1%}")
print(f"  검증 F1              : {metrics['eval_f1']:.4f}")
print(f"  소요 시간             : {result.metrics['train_runtime']:.0f}초")

print("""
  📖 loss(손실) 읽는 법
     정답과 예측의 차이입니다. 낮을수록 잘 맞힌 것.
     0.7 근처 → 아직 찍는 수준 (2지선다에서 무작위일 때의 값)
     0.3 이하 → 학습이 되고 있음
     0.05 이하 → 거의 외운 수준
""")

if train_loss < 0.1 and metrics["eval_accuracy"] > 0.95:
    print("  🎉 정확도가 매우 높습니다! ... 정말 좋은 걸까요?")
    print("     ⚠️ STAGE 2 마지막에 제가 심어둔 경고를 기억하세요.")
    print("        이 데이터의 어휘 다양성(TTR)은 0.026이었습니다.")
    print("        → 다음 미션(02번)에서 이 모델의 진짜 실력을 시험합니다.")


# =============================================================================
# [5] 저장
# =============================================================================
print("\n" + "=" * 78)
print("[5] 모델 저장")
print("=" * 78)

model.save_pretrained(MODEL_OUT)

# ★ 토크나이저도 반드시 함께 저장합니다.
#   모델만 저장하면 나중에 "어떤 토크나이저로 학습했더라?"가 됩니다.
#   STAGE 2에서 배웠듯 토크나이저와 모델은 한 몸입니다.
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(MODEL_OUT)

print(f"\n  ✅ 저장 완료 : {MODEL_OUT}")
print(f"  저장된 파일 : {sorted(p.name for p in MODEL_OUT.iterdir())}")
print("""
  💼 실무 습관: 모델과 토크나이저는 항상 같은 폴더에 세트로 저장하세요.
     그래야 pipeline(model=폴더경로) 한 줄로 바로 불러올 수 있습니다.
""")

print("=" * 78)
print("🎓 MISSION 1 완료 — 이제 02_the_trap.py 로 진짜 실력을 시험합니다")
print("=" * 78)


# =============================================================================
# 🎯 미션
# =============================================================================
# 1. train_loss 와 eval_accuracy 를 기록해두세요. 02번에서 비교합니다.
#
# 2. 로그의 loss 변화를 보세요. 처음 값과 마지막 값이 어떻게 다른가요?
#
# 3. ★ EPOCHS 를 1로 바꿔서 다시 돌려보세요. 정확도가 얼마나 떨어지나요?
#    (기존 결과를 덮어쓰니, 비교가 끝나면 3으로 되돌리고 재실행하세요)
# =============================================================================
