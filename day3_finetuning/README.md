# 🎮 STAGE 3 — 파인튜닝과 Trainer

> 원본 자료: `3일차 수업내용-Transforms의 구조`

---

## 이 스테이지의 목표

**"직접 모델을 학습시키고, 그 모델이 왜 실패하는지 진단한다."**

STAGE 1은 남의 모델을 의심했고, STAGE 2는 데이터를 의심했습니다.
**STAGE 3는 내가 만든 모델을 의심합니다.** 가장 어려운 의심이에요.

---

## ⚠️ 시작 전 필수 확인

STAGE 2의 결과물이 있어야 합니다.

```
day2_tokenizer\data\prepared_movie_dataset\   ← 이 폴더가 있어야 함
```

없다면 STAGE 2의 `05_full_pipeline.py`를 먼저 실행하세요.

---

## 미션 순서

| 순서 | 파일 | 배우는 것 | 예상 시간 |
|---|---|---|---|
| 1 | `01_train_first_model.py` | Trainer로 파인튜닝, 모델 저장 | **3~6분** |
| 2 | `02_the_trap.py` | 💣 99% 모델의 진짜 실력 | 10초 |
| 3 | `03_why_it_failed.py` | 실패 원인 해부 | 10초 |
| 4 | `04_lightweight_wins.py` | 딥러닝 vs scikit-learn | 30초 |
| 5 | `05_boss_real_data.py` | 🔥 보스 — 좋은 데이터로 재학습 | 5~15분 |
| 6 | `QUIZ_day3.md` | 자가 진단 | — |

> `_common.py`는 실행하는 파일이 아닙니다. 다른 파일들이 가져다 쓰는 부품 상자예요.

---

## 실행 방법

```bat
cd "C:\Users\Donga\Desktop\DONGA AI LAB\DONGA KDT AI MODEL SERVING TRAIN HUGGING FACE"
.venv\Scripts\activate
cd day3_finetuning
python 01_train_first_model.py
```

---

## ⏱️ 이번 스테이지는 기다림이 있습니다

지금까지와 달리 **실제 학습**을 합니다. CPU에서는 몇 분씩 걸려요.

| 상황 | 정상입니다 |
|---|---|
| 진행 바가 천천히 감 | ✅ CPU 학습은 원래 느립니다 |
| `pin_memory` 경고 | ✅ GPU가 없어서 나는 안내 |
| `newly initialized` 경고 | ✅ 분류 헤드를 새로 만든다는 뜻 (오히려 정상) |
| CPU 팬이 돌아감 | ✅ 계산 중입니다 |

---

## 🚨 가장 흔한 사고 — 버전 충돌

```
TypeError: ... (Trainer 실행 중)
```

수업 노트에도 기록된 그 에러입니다. 해결책:

```bat
pip uninstall -y transformers accelerate
pip install transformers==4.44.2 accelerate==0.34.0
```

`01_train_first_model.py`가 이 에러를 잡아서 안내를 띄우도록 만들어뒀습니다.

> 💡 `evaluation_strategy` / `eval_strategy` 이름 변경 문제는 `_common.py`의 `make_training_args()`가 자동으로 처리합니다. 버전이 뭐든 알아서 맞춰줍니다.

---

## 핵심 용어 (미리 읽어두면 편합니다)

| 용어 | 뜻 | 실무 감각 |
|---|---|---|
| **epoch** | 전체 데이터를 1바퀴 | 데이터 많으면 2~3, 적으면 10~20 |
| **batch** | 한 번에 처리할 문장 수 | CPU 8, GPU 16~32 |
| **step** | 배치 1개 처리 1회 | 총 스텝 = 데이터÷배치×에폭 |
| **learning rate** | 한 걸음의 크기 | BERT 파인튜닝은 **2e-5 ~ 5e-5** |
| **loss** | 정답과 예측의 차이 | **낮을수록 좋음.** 0.7=찍는중, 0.05=외움 |
| **과적합** | 학습 데이터만 잘 맞힘 | STAGE 3의 최대 적 |

---

## 이 스테이지에서 가져갈 3가지

### 1️⃣ 파인튜닝 = 랜덤 주사위를 판단기로 바꾸는 것
STAGE 1에서 `klue/roberta-base`가 점수차 0.0038을 냈던 이유는 분류 헤드가 랜덤이었기 때문입니다. Trainer가 하는 일이 정확히 그 주사위를 훈련시키는 것입니다.

### 2️⃣ 검증 정확도 99%는 자랑이 아닐 수 있다
검증 데이터가 학습 데이터와 너무 닮았다는 신호일 수 있습니다.

### 3️⃣ 딥러닝이 항상 정답은 아니다
데이터가 수백 건이면 scikit-learn이 가성비·성능 모두 앞섭니다.
**"닭 잡는 데 소 잡는 칼 쓰지 마라"** — 수업 노트의 명언입니다.

---

## 클리어 조건

- [ ] 5개 파일 실행 완료
- [ ] `data/my_movie_model` 폴더 생성 확인
- [ ] 02번에서 A조/B조 성적 격차 직접 확인
- [ ] `QUIZ_day3.md` 70% 이상

완료되면 **"STAGE 3 클리어"** 라고 말씀해주세요.
