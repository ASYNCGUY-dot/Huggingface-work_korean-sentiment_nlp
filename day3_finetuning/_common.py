"""
=============================================================================
STAGE 3 공용 도구 모음
=============================================================================
여러 파일에서 반복해서 쓰는 것들을 모아놨습니다.
직접 실행하는 파일이 아니라, 다른 파일이 import 해서 쓰는 부품 상자입니다.
=============================================================================
"""

import inspect
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


# -----------------------------------------------------------------------------
# 경로 — 어디서 실행해도 안 깨지도록
# -----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent          # day3_finetuning/
DATA = BASE / "data"
DATA.mkdir(parents=True, exist_ok=True)

# STAGE 2에서 만든 전처리 결과물 (day2_tokenizer/data/prepared_movie_dataset)
DAY2_DATA = BASE.parent / "day2_tokenizer" / "data"
PREPARED = DAY2_DATA / "prepared_movie_dataset"
MOVIE_CSV = DAY2_DATA / "movie.csv"

MODEL_NAME = "bert-base-multilingual-cased"     # STAGE 2에서 토크나이징한 모델과 동일해야 함
MODEL_OUT = DATA / "my_movie_model"             # 01번이 학습 결과를 저장할 곳


# -----------------------------------------------------------------------------
# ★ 버전 방어 코드 — TrainingArguments 인자 이름이 버전마다 다릅니다
# -----------------------------------------------------------------------------
def make_training_args(**kwargs):
    """설치된 transformers 버전에 맞게 인자 이름을 자동으로 맞춰줍니다.

    왜 필요한가?
      transformers 4.41 부터 evaluation_strategy → eval_strategy 로 이름이 바뀌었습니다.
      4.46 에서는 옛 이름이 아예 삭제됩니다.
      수업 노트에도 이 경고가 찍혀 있었죠:
        "`evaluation_strategy` is deprecated ... Use `eval_strategy` instead"

      버전마다 코드를 고치는 대신, 실행 시점에 확인해서 맞추는 방식입니다.
      💼 실무에서 라이브러리 버전이 흔들릴 때 쓰는 방어 기법입니다.
    """
    from transformers import TrainingArguments

    params = inspect.signature(TrainingArguments.__init__).parameters

    # 신 → 구 이름으로 되돌려야 하는 경우
    if "eval_strategy" in kwargs and "eval_strategy" not in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    # 이 버전이 아예 모르는 인자는 조용히 버립니다 (에러 방지)
    unknown = [k for k in kwargs if k not in params]
    for k in unknown:
        kwargs.pop(k)
    if unknown:
        print(f"  ⚠️ 이 transformers 버전이 지원하지 않는 인자를 제외했습니다: {unknown}")

    return TrainingArguments(**kwargs)


# -----------------------------------------------------------------------------
# 환경 점검
# -----------------------------------------------------------------------------
def check_env():
    """학습 전에 환경을 확인하고, 예상 소요 시간을 알려줍니다."""
    import torch
    import transformers

    print("=" * 78)
    print("환경 점검")
    print("=" * 78)
    print(f"  transformers : {transformers.__version__}")
    print(f"  torch        : {torch.__version__}")

    try:
        import accelerate
        print(f"  accelerate   : {accelerate.__version__}")
    except ImportError:
        print("  accelerate   : ❌ 미설치 → pip install accelerate==0.34.0")

    gpu = torch.cuda.is_available()
    print(f"  GPU 사용 가능 : {'✅ ' + torch.cuda.get_device_name(0) if gpu else '❌ (CPU로 학습합니다)'}")
    if not gpu:
        print("\n  ℹ️ CPU 학습은 느립니다. 진행 바가 천천히 가도 정상입니다.")
        print("     'pin_memory' 경고가 떠도 무시하세요. GPU가 없어서 나는 안내입니다.")
    print("=" * 78)
    return gpu


# -----------------------------------------------------------------------------
# 평가 지표 — 정확도만 보면 속습니다 (STAGE 4의 예고편)
# -----------------------------------------------------------------------------
def compute_metrics(eval_pred):
    """Trainer 가 평가할 때 호출하는 함수. 정확도 + F1 을 함께 반환합니다."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)     # 가장 점수 높은 클래스 선택
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }


# -----------------------------------------------------------------------------
# 어휘 다양성 측정 (STAGE 2에서 배운 TTR)
# -----------------------------------------------------------------------------
def ttr(texts):
    """Type-Token Ratio : 고유 단어 수 ÷ 전체 단어 수. 1에 가까울수록 다양."""
    words = [w for t in texts for w in str(t).split()]
    if not words:
        return 0.0, 0, 0
    uniq = set(words)
    return len(uniq) / len(words), len(uniq), len(words)


# -----------------------------------------------------------------------------
# 실전 테스트 문장 — 템플릿에 없는 '진짜' 문장들
# -----------------------------------------------------------------------------
# 01번에서 학습한 모델을 02번에서 이 문장들로 시험합니다.
REAL_SENTENCES = [
    # (문장, 사람이 판단한 정답)  1=긍정, 0=부정
    ("이건 진짜 내 인생 영화다.. 너무 감동이야.", 1),
    ("보는 내내 눈물이 멈추질 않았어요.", 1),
    ("올해 본 것 중에 제일 좋았습니다.", 1),
    ("두 번 봐도 재밌더라구요.", 1),
    ("친구한테 벌써 세 번 얘기했어요.", 1),
    ("중간에 잠들 뻔했어요. 전개가 너무 느림.", 0),
    ("표값이 아까워서 화가 났습니다.", 0),
    ("도대체 뭘 말하려는 건지 모르겠네요.", 0),
    ("차라리 그 시간에 잠을 잘 걸 그랬어요.", 0),
    ("기대가 컸던 만큼 실망도 컸습니다.", 0),
]

# 학습 데이터의 템플릿을 그대로 흉내낸 문장 (모델이 외운 문구)
TEMPLATE_SENTENCES = [
    ("이 영화 정말 최고예요. 꼭 보세요!", 1),
    ("배우들의 연기 감동적입니다. 강력 추천합니다.", 1),
    ("스토리 너무 뻔해서 지루했어요. 비추천합니다.", 0),
    ("연기력 시간 아까워요. 최악의 영화입니다.", 0),
]
