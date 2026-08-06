<div align="center">

# 🎬 한국어 영화 리뷰 감성 분석

**`klue/roberta-base` 파인튜닝부터 배포까지 — 5일간의 전 과정 기록**

[![Model](https://img.shields.io/badge/🤗_Model-jisooC%2Fnsmc--sentiment--klue-yellow)](https://huggingface.co/jisooC/nsmc-sentiment-klue)
[![Accuracy](https://img.shields.io/badge/Accuracy-88.0%25-brightgreen)]()
[![F1](https://img.shields.io/badge/F1-0.877-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.12-blue)]()

</div>

---

## 무엇을 만들었나

네이버 영화 리뷰(NSMC) 2만 건으로 `klue/roberta-base`를 파인튜닝해
한국어 감성 분류 모델을 만들고, 성능을 검증하고, 웹 서비스로 배포했습니다.

```python
from transformers import pipeline

clf = pipeline("text-classification", model="jisooC/nsmc-sentiment-klue")
clf("이건 진짜 내 인생 영화다.. 너무 감동이야.")
# [{'label': '긍정', 'score': 0.992}]
```

---

## 성능

**NSMC test split 2,000건** — 학습에 한 번도 사용하지 않은 데이터

| 지표 | 값 |
|---|---|
| 정확도 | **88.0%** (95% CI 86.6 ~ 89.4%) |
| 정밀도 | 0.8698 |
| 재현율 | 0.8842 |
| F1 | 0.8769 |

|  | 예측 부정 | 예측 긍정 |
|:---|---:|---:|
| **실제 부정** | 905 | 128 |
| **실제 긍정** | 112 | 855 |

> **일반화 검증** — 학습 시 검증 정확도 88%, 별도 test split에서도 **88.00%**.
> 격차 0%p로 과적합이 없음을 확인했습니다.

---

## 🔬 핵심 실험 — 데이터가 성능을 좌우한다

**같은 모델 · 같은 코드 · 같은 하이퍼파라미터**로 데이터만 바꿔 세 번 학습했습니다.

| 학습 데이터 | 고유 어휘 | 검증 정확도 | **실전 문장 정답률** |
|---|---:|---:|---:|
| 템플릿 생성 600건 | 74 | **100%** | 80% |
| NSMC 1,500건 | 8,172 | 70% | **50%** |
| **NSMC 20,000건** | **70,541** | 88% | **100%** |

### 발견

**1. 검증 정확도 100%는 과적합의 신호였습니다.**
템플릿 데이터의 어휘 다양성(TTR)이 0.023에 불과해, 모델이 감정이 아니라 문구를 암기했습니다.

**2. 학습 어휘와 겹치는 문장은 100%, 겹치지 않으면 50%(무작위)였습니다.**
`좋다`라는 기본 단어가 학습 데이터 600문장 전체에 **0회** 등장 → `"올해 본 것 중 제일 좋았습니다"`를 부정으로 오판.

**3. 다양성만 높이면 오히려 성능이 떨어졌습니다.**
어휘 8,172개를 1,500문장으로 배우려니 단어당 노출이 1.2회뿐. 데이터를 어렵게 만들었으면 **학습량도 함께 늘려야** 합니다.

---

## 🛡️ 서비스 설계 — 데모가 아닌 서비스로

모델을 감싸는 데 그치지 않고, **실험으로 확인한 한계를 안전장치로 구현**했습니다.

| 장치 | 근거 |
|---|---|
| 임계값 슬라이더 | 정밀도/재현율 트레이드오프를 사용자가 선택 |
| 확신도 경고 | 75% 미만 시 경고 — 도메인 밖 문장이 65%로 측정됨 |
| 도메인 밖 감지 | NSMC 학습 어휘와 50% 미만 일치 시 경고 |
| 사람 검토 밴드 | 임계값 근처는 자동 판정하지 않음 (Human-in-the-loop) |
| 길이 초과 경고 | 한국어는 결론이 뒤에 와서 잘림이 치명적 |
| 한계 명시 | UI에 실측 기반 한계를 공개 |

### 도메인 밖 감지 실측

| 입력 | 어휘 일치 | 확신도 | 경고 |
|---|---:|---:|:---:|
| "이건 진짜 내 인생 영화다.." | 100% | 99.2% | — |
| "중간에 잠들 뻔했어요" | 83% | 99.1% | — |
| "오늘 회의는 3시입니다" | **40%** | **65.1%** | 🚩🤔 |

두 개의 **독립적인 신호**(모델 내부 확신도 / 외부 어휘 대조)가 같은 문장을 지목합니다.

---

## 📁 저장소 구조

```
├── day1_pipeline/        파이프라인 · 모델 선택의 함정
├── day2_tokenizer/       토크나이저 · BPE/WordPiece · 데이터셋
├── day3_finetuning/      파인튜닝 · Trainer · Colab GPU 노트북
├── day4_evaluation/      혼동행렬 · 정밀도/재현율 · 임계값 · 모델 카드
├── day5_deploy/          Gradio 서빙 · 배포 · 멀티모달 보너스
│
├── REPORT.md             📄 상세 기술 보고서
├── MASTER_GUIDE.md       📘 전체 개념 정리 + 용어사전
├── PROGRESS.md           📍 실측 기록
└── requirements.txt
```

> ⚠️ 학습된 모델 파일(총 1.8GB)은 저장소에 포함되지 않습니다.
> [Hugging Face Model Hub](https://huggingface.co/jisooC/nsmc-sentiment-klue)에서 받으세요.

---

## 🚀 실행 방법

```bash
git clone https://github.com/ASYNCGUY-dot/korean-sentiment-nlp.git
cd korean-sentiment-nlp

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 웹 앱 실행

```bash
cd day5_deploy
python build_domain_vocab.py     # 도메인 어휘 사전 생성 (1회)
python 02_serve_my_model.py
```

> 모델은 Hugging Face Hub에서 자동으로 내려받습니다 (423MB).

### 직접 학습해보기

`day3_finetuning/colab_train_nsmc.ipynb`를 Google Colab에 올리고
런타임을 **T4 GPU**로 바꾼 뒤 실행하세요. **약 4분**이면 끝납니다.

---

## ⚠️ 알려진 한계

- **도메인** — 영화 리뷰 전용. 상품평·뉴스·업무 문서는 성능을 보장하지 않습니다.
- **우회 표현** — `"기대했던 것과는 좀 다르네요"`처럼 부정 어휘가 없는 문장에 약합니다.
- **중립 불가** — 2지선다 모델이라 "모르겠음"을 출력할 수 없습니다.
- **길이** — 128토큰 초과 시 절단됩니다.

---

## 🛠 기술 스택

```
모델    transformers 4.44.2 · PyTorch · klue/roberta-base
데이터  datasets · NSMC (e9t/nsmc)
학습    Trainer API · Google Colab T4 GPU · fp16
평가    scikit-learn · matplotlib
서빙    Gradio 6.x
배포    Hugging Face Model Hub
```

---

## 📄 더 읽을거리

| 문서 | 내용 |
|---|---|
| [REPORT.md](REPORT.md) | 상세 기술 보고서 — 실험 설계, 발견, 해결한 문제 12건 |
| [MASTER_GUIDE.md](MASTER_GUIDE.md) | 개념 정리 · 용어사전 · 에러 대응 사전 |
| [PROGRESS.md](PROGRESS.md) | 단계별 실측 기록 |

---

<div align="center">

**DONGA KDT AI 과정** · 2026.08

</div>
