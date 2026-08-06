# 📘 허깅페이스 5일 마스터 가이드

> 첨부하신 1~5일차 수업 자료 전량을 읽고 재구성한 학습 지도
> 작성일: 2026-08-05

---

## 목차

- [Part 0. AI 모델 개발의 전체 흐름](#part-0-ai-모델-개발의-전체-흐름)
- [Part 1. STAGE 1 — 파이프라인](#part-1-stage-1--파이프라인)
- [Part 2. STAGE 2 — 토크나이저와 데이터셋](#part-2-stage-2--토크나이저와-데이터셋)
- [Part 3. STAGE 3 — 파인튜닝과 Trainer](#part-3-stage-3--파인튜닝과-trainer)
- [Part 4. STAGE 4 — 경량 모델과 성능 평가](#part-4-stage-4--경량-모델과-성능-평가)
- [Part 5. STAGE 5 — 서빙과 배포](#part-5-stage-5--서빙과-배포)
- [부록 A. 용어사전](#부록-a-용어사전)
- [부록 B. 에러 대응 사전](#부록-b-에러-대응-사전)
- [부록 C. 실무 체크리스트](#부록-c-실무-체크리스트)

---

## Part 0. AI 모델 개발의 전체 흐름

수업 1일차 첫 자료가 "AI 모델 개발 전체 과정"이었습니다. 이게 5일 내내 반복되는 뼈대입니다.

```
① 데이터 수집  →  ② 전처리  →  ③ 모델 설계/선택  →  ④ 학습  →  ⑤ 평가  →  ⑥ 서빙/배포
```

이걸 5일차 커리큘럼에 매핑하면:

| 단계 | 무엇을 하나 | 몇 일차 | 대표 도구 |
|---|---|---|---|
| ① 수집 | CSV 만들기, 허브에서 데이터셋 받기 | 2일차 | `load_dataset`, pandas |
| ② 전처리 | 텍스트 → 토큰 → 텐서 | 2일차 | `AutoTokenizer`, `.map()` |
| ③ 설계 | 어떤 사전학습 모델을 쓸지 결정 | 1·3일차 | `AutoModelFor...` |
| ④ 학습 | 파인튜닝 | 3일차 | `Trainer`, `TrainingArguments` |
| ⑤ 평가 | 혼동행렬, F1 | 3·4일차 | scikit-learn |
| ⑥ 배포 | 웹 UI + 클라우드 | 4·5일차 | Gradio, HF Spaces |

### 🔑 지금 당장 이해해야 할 개념 하나

> **AI 모델이란 "입력값을 받아 확률을 계산해주는 함수"다.**

수업 1일차 노트 첫 줄에 이렇게 적혀 있습니다. 이게 정말 핵심입니다.

`classifier("이 영화 최고!")` 를 호출하면 결과가 `{'label': 'POSITIVE', 'score': 0.9971}` 로 나옵니다.
모델은 "긍정이야!"라고 확신하는 게 아니라, **"긍정일 확률이 99.71%"** 라고 계산해준 것뿐입니다.

그래서 score가 0.51이면? 모델이 사실상 동전을 던진 겁니다. **점수가 낮으면 답을 믿지 마세요.** 이건 3일차 실습에서 뼈저리게 배우게 됩니다.

---

## Part 1. STAGE 1 — 파이프라인

### 1-1. pipeline이 뭘 대신 해주나

수업 1일차의 첫 코드는 딱 3줄입니다.

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="...", framework="pt")
result = classifier("I love this!")
```

이 3줄 뒤에서 실제로 벌어지는 일:

```
입력 문장
  ↓ ① 토크나이저 자동 로드 & 문장을 숫자 ID로 변환
  ↓ ② 모델 가중치 자동 다운로드 (처음 1회, 캐시에 저장)
  ↓ ③ 숫자를 텐서로 변환 → 모델에 투입
  ↓ ④ 모델이 raw 점수(logits) 출력
  ↓ ⑤ softmax로 확률 변환 + 라벨 이름 매핑
결과 딕셔너리
```

**pipeline은 ①~⑤를 전부 자동화한 "고수준 API"입니다.** 편한 대신 세밀한 제어가 안 됩니다. 3일차에서 이 단계를 하나씩 손으로 분해하게 됩니다.

### 1-2. `framework="pt"` 의 정체

| 값 | 의미 |
|---|---|
| `"pt"` | PyTorch로 실행 |
| `"tf"` | TensorFlow로 실행 |
| 생략 | 설치된 걸 자동 탐색 (둘 다 있으면 느려지거나 경고 발생) |

**실무 권장: 항상 `framework="pt"` 명시.**
이유 — 허깅페이스 모델의 절대다수가 PyTorch 기준이고, 자동 탐색 과정에서 TensorFlow를 불필요하게 로딩하면 시작이 느려집니다.

> 💡 참고: transformers **v5부터는 TensorFlow 백엔드가 아예 제거**됐습니다. 즉 미래에는 `framework` 인자 자체가 의미가 없어집니다. 지금 4.x를 쓰는 동안만 신경 쓰면 됩니다.

### 1-3. ⭐ 1일차 최대의 교훈 — 모델을 지정하지 않으면 생기는 참사

수업 노트에 이 실험이 그대로 기록돼 있습니다.

```python
# 모델을 지정 안 한 경우
classifier = pipeline("sentiment-analysis")
classifier("오늘 주식시장은 투자자들의 불안으로 큰 폭 하락했습니다.")
# → POSITIVE, 0.8022  ← ???
```

왜 이런 일이? 모델을 생략하면 허깅페이스가 기본 모델을 자동 선택하는데, 그게 `distilbert-base-uncased-finetuned-sst-2-english` — **영어 영화 리뷰 전용 모델**입니다. 한국어를 학습한 적이 없습니다.

한국어 문장을 넣으면 모델은 "모르는 글자 덩어리"를 보고 아무 확률이나 뱉습니다.

#### 목적별 한국어 모델 선택표 (수업 노트 기준)

| 상황 | 추천 모델 | 특징 |
|---|---|---|
| 금융/주식 뉴스 | `snunlp/KR-FinBert-SC` | 금융 도메인 특화. 단 "감정"이 아니라 **주가 영향 방향성**을 예측 |
| 일반 뉴스 이해 | `klue/roberta-base` | 한국어 이해 최강. 단 **분류 헤드가 없어서 그냥 쓰면 LABEL_0/1만 나옴** |
| 구어체/댓글 | `beomi/KcBERT-base` | 인터넷 댓글 학습 |
| 빠른 처리 | `KoELECTRA` 계열 | 가볍고 빠름 |
| 다국어 별점 | `nlptown/bert-base-multilingual-uncased-sentiment` | 1~5 stars 반환. 한국어도 처리 가능 |

### 1-4. ⭐⭐ 가장 헷갈리는 함정 — "베이스 모델"과 "파인튜닝된 모델"

수업 노트에서 `klue/roberta-base`를 썼더니 `LABEL_0`, `LABEL_1`이 나오고 확률이 0.57밖에 안 나온 사건이 있었죠. 이게 왜 그런지 정확히 알아야 합니다.

```
[베이스 모델]  klue/roberta-base
    = 한국어를 "이해"하는 능력만 학습된 상태
    = 감정이 뭔지, 스팸이 뭔지는 전혀 모름
    = 분류용으로 쓰면 분류 레이어가 랜덤 초기화됨 → 결과가 무작위

[파인튜닝된 모델]  daekeun-ml/koelectra-small-v3-nsmc
    = 베이스 모델 + 감정 데이터로 추가 학습 완료
    = 바로 감정 분류에 쓸 수 있음
```

**비유로:**
베이스 모델 = 한국어를 유창하게 하는 외국인.
파인튜닝된 모델 = 한국어도 하고 영화 평론 훈련도 받은 사람.
전자에게 "이 리뷰 좋아? 나빠?"라고 물으면 말은 알아듣지만 판단 기준이 없습니다.

> **판별법:** 허깅페이스 모델 페이지에서 결과 라벨이 `LABEL_0`, `LABEL_1`처럼 나오면 → 파인튜닝 안 된 베이스 모델입니다.
> `POSITIVE`, `5 stars`, `negative` 처럼 의미 있는 이름이 나오면 → 파인튜닝된 모델입니다.

### 1-5. tokenizer 인자, 써야 하나 말아야 하나

수업 노트의 결론이 정확합니다.

| 방식 | 결과 |
|---|---|
| 모델과 같은 tokenizer 명시 | ✅ 정상 |
| **다른** tokenizer 지정 | ❌ 성능 붕괴 (토큰 분해 방식이 달라 의미가 왜곡) |
| **생략** | ✅ 자동으로 같은 걸 로드 — **가장 안전, 추천** |

**실무 팁:** 그냥 생략하세요. 명시할 이유는 커스텀 토크나이저를 쓸 때뿐입니다.

### 1-6. STAGE 1 실무 팁 모음

- **모델 캐시 위치**: `C:\Users\Donga\.cache\huggingface\hub`. 모델이 GB 단위라 디스크가 금방 찹니다. 안 쓰는 모델은 폴더째 지우면 됩니다.
- **Windows symlink 경고**: 수업 로그에도 나온 그 경고. 무시해도 되지만 없애려면 Windows 개발자 모드를 켜세요.
- **모델 고르는 순서**: ① 태스크로 필터 → ② 언어로 필터 → ③ 다운로드 수 정렬 → ④ 모델 카드의 학습 데이터 확인. 다운로드 수는 "검증된 정도"의 대리 지표입니다.
- **`if __name__ == "__main__":`** — Streamlit/Gradio 파일에는 꼭 넣으세요. 다른 파일에서 import할 때 서버가 멋대로 뜨는 걸 막아줍니다.

---

## Part 2. STAGE 2 — 토크나이저와 데이터셋

### 2-1. 왜 토크나이저가 필요한가

컴퓨터는 글자를 모릅니다. 숫자만 압니다.

```
"안녕하세요"  →  ['안녕', '하세요']  →  [9321, 118610]  →  tensor([9321, 118610])
   원문           토큰(조각)              토큰 ID            텐서
```

이 변환기가 토크나이저입니다.

### 2-2. BPE vs WordPiece — 실전 비교

수업 3번째 실습에서 같은 문장을 두 방식으로 돌린 실제 결과입니다.

```
입력: "오늘은 날씨가 좋아서 놀러가기 좋은 날입니다."

BPE       → ['오','늘','날','씨','가','좋','아','서','가','기','좋','날','니','다','.']
WordPiece → ['[UNK]','날','##씨','##가','좋','##아','##서','[UNK]','[UNK]','[UNK]','.']
```

| 항목 | BPE | WordPiece |
|---|---|---|
| 병합 기준 | **빈도수** — 자주 붙어 나오는 쌍을 합침 | **확률(likelihood)** — 언어모델 성능을 높이는 쌍을 합침 |
| 표시 방식 | 평면적 | 단어 중간 조각에 `##` 접두어 |
| 모르는 단어 | 글자 단위로 쪼갬 | `[UNK]`로 처리 |
| 대표 모델 | GPT 계열, RoBERTa | BERT 계열 |
| 주 용도 | **텍스트 생성** (챗봇, 번역, 요약) | **텍스트 분석** (분류, 검색, 감정분석) |

> ⚠️ 위 실습에서 `[UNK]`가 많이 나온 건 WordPiece가 나빠서가 아닙니다.
> **학습 데이터가 sample.txt 한 문장뿐이었기 때문**입니다. `min_frequency=2`(2번 이상 나온 조각만 사전 등록) 조건을 대부분의 단어가 못 넘긴 거죠.
> **교훈: 토크나이저를 직접 학습시키려면 최소 수만 문장이 필요합니다.** 실무에서 토크나이저를 직접 학습하는 일은 드물고, 보통은 `AutoTokenizer.from_pretrained()`로 기성품을 씁니다.

### 2-3. 특수 토큰 5종 — 외워두면 평생 씁니다

| 토큰 | 이름 | 역할 |
|---|---|---|
| `[PAD]` | Padding | 문장 길이를 맞추려고 빈칸 채우기. 배치 처리에 필수 |
| `[UNK]` | Unknown | 사전에 없는 단어 |
| `[CLS]` | Classification | 문장 맨 앞. **이 자리의 출력값이 문장 전체 요약** → 분류에 사용 |
| `[SEP]` | Separator | 문장과 문장의 경계 |
| `[MASK]` | Mask | 학습 시 단어를 가리고 맞히게 함 (BERT 학습법) |

`[CLS]`가 특히 중요합니다. 분류 모델이 문장 전체를 판단할 때 **`[CLS]` 자리의 벡터 하나만 보고 결정**합니다.

### 2-4. DatasetDict 구조

```python
dataset = load_dataset("rotten_tomatoes")
```
```
DatasetDict({
    train:      8530개   ← 모델이 공부할 문제집
    validation: 1066개   ← 공부 중 실력 확인용 모의고사
    test:       1066개   ← 최종 시험 (한 번도 안 본 문제)
})
```

**왜 3개로 나누나?** 학습 데이터로 평가하면 무조건 만점이 나옵니다. 답을 외운 거니까요. 이걸 **과적합(overfitting)** 이라고 합니다.

주요 메서드:
```python
ds.filter(lambda x: "perfect" in x["text"])   # 조건 필터링
ds.shuffle(seed=42)                            # 섞기 (편향 방지, 필수!)
ds.sort("text")                                # 정렬
ds.map(fn, batched=True)                       # 일괄 변환 (batched=True가 속도 핵심)
ds.train_test_split(test_size=0.2)             # 8:2 분할
```

### 2-5. 텐서 변환 — 왜 필요한가

```python
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
```

수업 3일차 자료에 이유가 잘 정리돼 있습니다.

1. **모델은 숫자+텐서만 이해** — 파이썬 리스트로는 연산 불가
2. **GPU 가속** — 텐서여야 GPU에서 병렬 계산 가능
3. **자동 미분** — 학습하려면 기울기 계산이 필요한데 텐서만 지원
4. **배치 처리** — 여러 문장을 한 번에 행렬 연산

```
torch.Size([128])       ← 문장 1개, 토큰 128개
torch.Size([8, 128])    ← 문장 8개(배치), 각 128 토큰
```

### 2-6. 전처리 3종 세트

```python
tokenizer(texts, padding="max_length", truncation=True, max_length=128)
```

| 인자 | 의미 | 안 쓰면? |
|---|---|---|
| `padding="max_length"` | 짧은 문장을 `[PAD]`로 채워 길이 통일 | 길이가 제각각이라 배치 처리 불가 |
| `truncation=True` | 긴 문장을 잘라냄 | 모델 최대 길이 초과 시 에러 |
| `max_length=128` | 기준 길이 | 기본값(512)이라 메모리 낭비 |

**`max_length` 정하는 법:** 데이터의 문장 길이 분포를 보고 **95%가 들어가는 값**으로 잡으세요. 무작정 512로 하면 메모리와 시간을 4배 낭비합니다.

### 2-7. 데이터 파이프라인 흐름 (수업 노트의 비유가 좋습니다)

```
Pandas          → 데이터 읽기/수정        (요리 재료 손질)
Dataset         → 허깅페이스 규격화        (그릇에 담기)
Tokenizer       → 숫자 변환              (잘게 썰기)
remove_columns  → 원본 텍스트 제거        (이물질 제거)
set_format      → 텐서 변환              (손님상에 올리기)
```

> ⚠️ `remove_columns(["content"])`를 빼먹으면 학습 중 에러가 납니다. 모델 입장에서 글자 컬럼은 이물질입니다.

---

## Part 3. STAGE 3 — 파인튜닝과 Trainer

### 3-1. 전이학습(Transfer Learning)이란

```
[사전학습]  위키피디아 수십 GB로 "언어 자체"를 학습     ← 구글/네이버가 해줌 (돈 수억)
     ↓
[파인튜닝]  내 데이터 1000개로 "특정 업무"만 추가 학습   ← 우리가 함 (노트북으로 6분)
```

이게 허깅페이스가 혁명적인 이유입니다. **바닥부터 만들 필요가 없습니다.**

### 3-2. Trainer 3종 세트

```python
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(model=model, args=training_args,
                  train_dataset=ds["train"], eval_dataset=ds["test"])
trainer.train()
```

#### 하이퍼파라미터 의미

| 이름 | 뜻 | 실무 감각 |
|---|---|---|
| `learning_rate` | 한 번에 얼마나 크게 수정할지 | BERT 파인튜닝은 **2e-5 ~ 5e-5**가 정석. 크면 발산, 작으면 안 배움 |
| `batch_size` | 한 번에 처리할 문장 수 | 메모리에 맞춰서. CPU면 8, GPU면 16~32 |
| `num_train_epochs` | 전체 데이터 반복 횟수 | 데이터 많으면 2~3, 적으면 10~20 |
| `weight_decay` | 과적합 억제 규제 | 0.01이 관례 |

> **학습률 비유:** 산에서 가장 낮은 지점을 찾아 내려가는데, 보폭이 학습률입니다. 너무 크면 골짜기를 건너뛰고, 너무 작으면 해 지기 전에 못 내려옵니다.

### 3-3. ⭐⭐⭐ 3일차 최대의 교훈 — 데이터가 적으면 반드시 실패한다

수업 노트에 정직하게 기록된 실패 사례입니다.

```
학습 데이터 5개로 학습한 스팸 분류기 결과:
  "축하합니다! 1억 원 경품에 당첨되셨습니다!"  →  정상(HAM) 73.61%   ← 완전 오답
```

원인 3가지 (수업 노트 그대로):

1. **데이터 기아** — 5개는 수능을 기출 5문제 풀고 보는 것과 같음
2. **사전학습 지식과 충돌** — 베이스 모델은 위키피디아를 학습했고, 위키피디아에는 "경품 당첨" 같은 문구가 드묾. 그래서 "평범한 문장(정상)" 쪽으로 편향됨
3. **학습 부족** — epoch 3회로는 모델의 뇌 구조가 안 바뀜

**해결책 우선순위:**
1. **데이터 늘리기** (압도적 1순위) — 최소 수백, 이상적으로 수천 개
2. epoch 늘리기 (10~20)
3. 클래스 비율 맞추기 (50:50)

수업 자료의 `9-3.moviecsvmake3.py`가 이걸 잘 다룹니다.
- `itertools.product`로 모든 조합을 미리 생성 → **중복 원천 차단**
- `random.sample`로 필요한 개수만 추출
- `df.sample(frac=1)`로 셔플 → **편향 방지**

### 3-4. ⭐⭐ "딥러닝이 항상 정답은 아니다"

3일차 후반의 반전입니다. 같은 스팸 문제를 scikit-learn으로 풀었더니:

| 방식 | 데이터 5개 | 데이터 20개 |
|---|---|---|
| BERT (딥러닝) | 3문제 중 0개 정답 | — |
| TF-IDF + 나이브베이즈 | 3문제 중 1개 정답 | **3문제 중 3개 정답** |

**왜?**
BERT는 "문맥을 이해"하려 하는데, 이해하려면 데이터가 많이 필요합니다.
TF-IDF는 그냥 **"스팸엔 '당첨', '금리' 같은 단어가 자주 나오네"** 라는 단어 통계만 셉니다. 데이터가 적을 땐 이 단순함이 더 강력합니다.

> 💬 수업 노트의 명언: **"닭 잡는 데 소 잡는 칼 쓰지 마라."**
> 데이터가 수백 건 수준이면 scikit-learn이 가성비·성능 모두 압승입니다.

#### 언제 무엇을 쓸까

| 데이터 규모 | 추천 | 이유 |
|---|---|---|
| ~수백 건 | TF-IDF + 나이브베이즈 / 로지스틱회귀 | 빠르고 안정적, 학습 몇 초 |
| 수천 ~ 수만 건 | BERT 파인튜닝 | 문맥 이해 효과가 나타남 |
| 수십만 건+ | 대형 모델 파인튜닝 | 본격적인 성능 |

### 3-5. TF-IDF 개념

**TF (Term Frequency)** — 이 문서에 이 단어가 몇 번 나왔나
**IDF (Inverse Document Frequency)** — 이 단어가 전체 문서에서 얼마나 희귀한가

```
TF-IDF 점수 = TF × IDF
```

"은/는/이/가"는 모든 문서에 나오니 IDF가 낮아 점수가 0에 가깝습니다.
"비트코인"은 특정 문서에만 나오니 IDF가 높아 점수가 큽니다.
**즉, 그 문서를 특징짓는 단어에 높은 점수를 줍니다.**

#### TF-IDF vs 토크나이저 — 헷갈리기 쉬운 차이

| | TF-IDF | 토크나이저(BERT) |
|---|---|---|
| 출력 | 단어 중요도 점수 | 토큰 ID |
| 순서 정보 | ❌ 없음 (bag of words) | ✅ 있음 |
| 문맥 이해 | ❌ "안 좋다"와 "좋다"를 구분 못 함 | ✅ 구분함 |
| 속도 | 매우 빠름 | 느림 |

---

## Part 4. STAGE 4 — 경량 모델과 성능 평가

### 4-1. 혼동행렬(Confusion Matrix) — 반드시 외우세요

|  | 예측: 정상 | 예측: 스팸 |
|---|---|---|
| **실제: 정상** | TN (True Negative) | FP (False Positive) ⚠️ |
| **실제: 스팸** | FN (False Negative) ⚠️ | TP (True Positive) |

- **FP (거짓 양성)** = 정상 메일을 스팸으로 잘못 분류 → **중요 메일이 스팸함으로 사라짐**
- **FN (거짓 음성)** = 스팸을 정상으로 놓침 → 광고가 받은편지함에 들어옴

### 4-2. 4대 지표

| 지표 | 공식 | 질문 | 언제 중요한가 |
|---|---|---|---|
| **정확도** Accuracy | (TP+TN)/전체 | 전체 중 몇 개 맞췄나 | 클래스 비율이 균형일 때만 의미 있음 |
| **정밀도** Precision | TP/(TP+FP) | 스팸이라 한 것 중 진짜 스팸 비율 | **FP가 치명적일 때** (스팸필터, 형사재판) |
| **재현율** Recall | TP/(TP+FN) | 실제 스팸 중 몇 개나 잡았나 | **FN이 치명적일 때** (암 진단, 화재감지) |
| **F1-score** | 정밀도·재현율의 조화평균 | 둘의 균형 | **불균형 데이터의 기본 지표** |

### 4-3. ⭐ 정확도의 함정

암 환자가 1%인 데이터에서, **"전부 정상"이라고만 답하는 모델**의 정확도는 **99%**입니다.
하지만 암 환자를 한 명도 못 찾았으니 재현율은 **0%**. 쓸모없는 모델입니다.

> **불균형 데이터에서 정확도만 보는 건 자기기만입니다. F1과 혼동행렬을 같이 보세요.**

수업 4일차 LLM 활용 노트에도 이 문제의식이 그대로 나옵니다 — "500:500 데이터로 학습했는데 스팸 확률이 10%밖에 안 나온다, 어디가 문제인가?"

### 4-4. 정밀도 vs 재현율 트레이드오프

둘은 시소 관계입니다. 임계값(threshold)을 조정하면 한쪽이 오르고 한쪽이 내려갑니다.

```
임계값 0.9로 올림 → 확실할 때만 스팸 판정 → 정밀도 ↑, 재현율 ↓
임계값 0.3으로 낮춤 → 의심되면 스팸 판정 → 재현율 ↑, 정밀도 ↓
```

**실무에서는 비즈니스 요구가 임계값을 결정합니다.** 은행 사기탐지는 재현율을, 스팸필터는 정밀도를 우선합니다.

### 4-5. 모델 저장/로드 — 두 가지 방식

```python
# ① scikit-learn → joblib (.pkl)
import joblib
joblib.dump(model, "finance_spam_model.pkl")
model = joblib.load("finance_spam_model.pkl")

# ② 허깅페이스 → save_pretrained (폴더)
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")   # ★ 토크나이저도 반드시 같이!
```

> ⚠️ **가장 흔한 실수: 토크나이저를 안 저장하는 것.**
> 모델만 저장하면 나중에 "그때 어떤 토크나이저 썼더라?"가 됩니다. 세트로 저장하세요.

### 4-6. 경로 처리 — 실무 필수 습관

수업 자료에서 반복적으로 나오는 패턴입니다. 꼭 익히세요.

```python
from pathlib import Path

current_dir = Path(__file__).resolve().parent   # 이 .py 파일이 있는 폴더
base_path = current_dir / "data"                # 그 아래 data 폴더
base_path.mkdir(parents=True, exist_ok=True)    # 없으면 만들고, 있으면 통과
file_path = base_path / "movie.csv"
```

**왜 이렇게?** `"data/movie.csv"` 같은 상대경로는 **터미널의 현재 위치**를 기준으로 잡힙니다. VS Code에서 실행 버튼을 누를 때와 터미널에서 직접 실행할 때 위치가 달라져서 "파일이 없다" 에러가 납니다. `Path(__file__)`은 **파일 위치 기준**이라 어디서 실행해도 동일합니다.

### 4-7. KNN — 4일차에 나온 개념

**K-Nearest Neighbors**: 새 데이터가 들어오면 **가장 가까운 K개 이웃의 다수결**로 분류.

- 학습 과정이 없음 (데이터를 그냥 저장만 함) → "게으른 학습기"
- K가 작으면 노이즈에 민감, 크면 경계가 뭉개짐. 보통 **홀수**로 (동점 방지)
- 거리 계산은 주로 **유클리드 거리**: √((x₁-x₂)² + (y₁-y₂)²)
- 특징 스케일이 다르면 망가짐 → **정규화 필수**

---

## Part 5. STAGE 5 — 서빙과 배포

### 5-1. "서빙"이란

```
[학습 완료된 모델 파일]  ← 이대로는 나 혼자만 씀
        ↓ 서빙
[누구나 접속 가능한 웹 서비스]
```

**서빙(Serving)** = 모델을 요청받아 응답하는 상태로 만드는 것
**배포(Deployment)** = 그걸 실제 서버에 올려 운영하는 것

### 5-2. Gradio — Interface vs Blocks

```python
# ① Interface — 화면 배치를 Gradio가 알아서
demo = gr.Interface(fn=hello, inputs=gr.Textbox(), outputs=gr.Textbox())

# ② Blocks — 내가 직접 디자인
with gr.Blocks() as demo:
    gr.Markdown("# 제목")
    with gr.Row():           # 가로 배치
        with gr.Column():    # 세로 배치
            txt = gr.Textbox(label="입력")
            btn = gr.Button("분석", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="결과", interactive=False)
    btn.click(fn=predict, inputs=txt, outputs=out)
```

| | Interface | Blocks |
|---|---|---|
| 코드량 | 3줄 | 20줄+ |
| 레이아웃 자유도 | 낮음 | 높음 |
| 언제 | 빠른 프로토타입 | **실제 서비스** |

**주요 컴포넌트:** `Textbox` `Image` `Audio` `Label`(확률 막대그래프) `Button` `ClearButton` `File` `Slider` `Dropdown`

**launch 옵션:**
```python
demo.launch(
    inbrowser=True,          # 브라우저 자동 실행
    share=True,              # 72시간 공개 링크 생성 (임시 데모용)
    server_port=7860,
)
```

### 5-3. Streamlit vs Gradio — 언제 뭘 쓰나

| | Streamlit | Gradio |
|---|---|---|
| 강점 | 데이터 대시보드, 표/차트 | AI 모델 데모 |
| 상태 관리 | `st.session_state` 필요 | 이벤트 기반이라 덜 필요 |
| HF Spaces 배포 | 가능 | **가장 잘 지원됨** |
| 이 과정에서 | 1~3일차 분석 대시보드 | 4~5일차 모델 서빙 |

> **Streamlit 핵심 함정:** 버튼 하나만 눌러도 **파일 전체가 처음부터 다시 실행**됩니다.
> 그래서 두 가지가 필수입니다.
> - `@st.cache_resource` — 모델 로딩을 1회만 (없으면 클릭할 때마다 모델 재다운로드)
> - `st.session_state` — 데이터 유지 (없으면 이전 결과가 다 날아감)

### 5-4. 5일차 멀티모달 실습 정리

수업 5일차 소스코드 4개의 구조입니다.

| 파일 | 하는 일 | 사용 모델 |
|---|---|---|
| `1.multi-text.py` | 다중 감정 분석 (7가지 감정 확률) | `j-hartmann/emotion-english-distilroberta-base` |
| `1-2.multi-textsound.py` | 감정 분석 + TTS 음성 생성 | 위 + `gTTS` |
| `2-1.googlevitmodel.py` | 이미지 분류 | `google/vit-base-patch16-224` |
| `2-3.imagedescription.py` | 이미지 분류 + 설명 생성 | 위 + `Salesforce/blip-image-captioning-base` |
| `3.voicetest.py` | 음성→영어텍스트→한국어→음성 | `whisper-base` + `m2m100_418M` + `gTTS` |

> 🔧 **수정 필요 사항 (2026-08 검증):**
> `1.multi-text.py`와 `1-2.multi-textsound.py`의 `return_all_scores=True`는 **deprecated**입니다.
> → `top_k=None` 으로 바꾸세요. 결과 구조는 동일합니다.

#### ViT — 이미지를 어떻게 처리하나

텍스트와 놀랍도록 똑같습니다.

```
텍스트:  문장 → 토큰으로 분할 → 숫자 ID → 텐서 → Transformer
이미지:  사진 → 패치(16×16)로 분할 → 벡터 → 텐서 → Transformer
```

`vit-base-patch16-224` 이름의 뜻: 224×224 이미지를 16×16 패치로 자름 → 총 196개 패치 → 이걸 "단어"처럼 취급.

### 5-5. ⭐ 허깅페이스 Spaces 배포 절차 (수업 5일차 순서 그대로)

```
① 허깅페이스 계정 생성 + 이메일 인증
     ↳ 인증 안 하면 Space 생성이 막힙니다

② Access Token 발급
     Settings → Access Tokens → New token
     ↳ ★ 권한을 Repository - Write 로 줘야 push됨

③ New Space 생성
     SDK: Gradio 선택 / 이름 지정 / Public

④ 로컬에 클론
     git clone https://huggingface.co/spaces/{사용자명}/{스페이스명}
     cd {스페이스명}

⑤ 파일 2개 배치
     app.py             ← 반드시 이 이름! (Spaces가 이 파일을 찾습니다)
     requirements.txt   ← 필요한 패키지만 최소로

⑥ push
     git add .
     git commit -m "deploy"
     git push origin main
     ↳ 사용자명 / 비밀번호 자리에 Access Token 입력

⑦ 빌드 로그 확인
     https://huggingface.co/spaces/{사용자명}/{스페이스명}?logs=build
```

### 5-6. 배포 실패 3대 원인과 해결

수업 노트에 실제로 겪은 사례가 기록돼 있습니다.

| 증상 | 원인 | 해결 |
|---|---|---|
| 빌드가 계속 실패 | requirements.txt 버전 고정이 너무 빡빡 | **버전 번호를 빼고** `transformers` 만 적어 호환 버전이 자동 선택되게 함 |
| `theme` 관련 에러 | Gradio 6.x에서 위치 변경 | `gr.Blocks(theme=...)` → `demo.launch(theme=...)` |
| GPU 사용 불가 | ZeroGPU 설정 누락 | `import spaces` + 함수 위에 `@spaces.GPU` 데코레이터 |

**배포용 requirements.txt (수업에서 성공한 조합):**
```
gradio>=6.5.1
transformers
Pillow>=11.0.0
spaces
torchvision
```

> 💡 **핵심 인사이트:** 로컬은 버전을 **고정**하고(재현성), 배포는 버전을 **느슨하게**(호환성) — 정반대입니다.
> 이유: 로컬은 "내가 어제 되던 게 오늘도 돼야" 하고, 배포 서버는 "그 서버 환경에서 되는 조합"을 찾아야 하기 때문입니다.

### 5-7. 배포 전 체크리스트

- [ ] 파일명이 정확히 `app.py` 인가
- [ ] `.env`, API 키, 토큰이 코드에 하드코딩돼 있지 않은가 (**Spaces는 Settings → Secrets 사용**)
- [ ] `requirements.txt`에 불필요한 무거운 패키지가 없는가
- [ ] 모델 크기가 적절한가 (무료 Space는 CPU 2코어 / 16GB RAM)
- [ ] 입력이 없을 때(None) 처리를 했는가
- [ ] `try/except`로 에러 시 앱이 죽지 않게 했는가
- [ ] `.gitignore`에 `.venv`, `.env`, `__pycache__` 를 넣었는가

---

## 부록 A. 용어사전

### 모델 관련
| 용어 | 뜻 |
|---|---|
| **사전학습 (Pre-training)** | 대량의 일반 데이터로 언어/이미지 자체를 학습한 상태 |
| **파인튜닝 (Fine-tuning)** | 사전학습 모델을 내 데이터로 추가 학습 |
| **전이학습 (Transfer Learning)** | 학습된 지식을 다른 문제에 재사용하는 개념 (파인튜닝은 그 방법 중 하나) |
| **체크포인트** | 학습 중간 저장본 |
| **가중치 (Weights)** | 모델이 학습한 숫자들. 모델의 "기억" 그 자체 |
| **logits** | 모델의 raw 출력값. 확률 아님. softmax를 거쳐야 확률 |
| **softmax** | logits를 합이 1인 확률로 변환하는 함수 |
| **num_labels** | 분류할 클래스 개수 (긍정/부정이면 2) |

### 학습 관련
| 용어 | 뜻 |
|---|---|
| **epoch** | 전체 데이터를 1번 다 훑는 것 |
| **batch** | 한 번에 처리하는 데이터 묶음 |
| **step** | 배치 1개를 처리하는 1회 (전체 step = 데이터수 ÷ batch × epoch) |
| **learning rate** | 가중치를 얼마나 크게 수정할지 |
| **loss** | 정답과 예측의 차이. **낮을수록 좋음** |
| **과적합 (Overfitting)** | 학습 데이터만 잘 맞히고 새 데이터는 못 맞힘 |
| **과소적합 (Underfitting)** | 학습 데이터조차 못 맞힘 |

### 데이터 관련
| 용어 | 뜻 |
|---|---|
| **라벨 (Label)** | 정답. 지도학습에 필수 |
| **지도학습** | 정답이 있는 데이터로 학습 (분류, 회귀) |
| **비지도학습** | 정답 없이 패턴 발견 (군집화) |
| **데이터 증강** | 데이터를 늘리는 기법 |
| **편향 (Bias)** | 데이터가 한쪽으로 치우친 상태 |
| **셔플** | 데이터 순서를 섞는 것. 편향 방지에 필수 |

### 서빙 관련
| 용어 | 뜻 |
|---|---|
| **추론 (Inference)** | 학습된 모델로 예측하는 것 |
| **STT** | Speech To Text — 음성→텍스트 (Whisper) |
| **TTS** | Text To Speech — 텍스트→음성 (gTTS) |
| **멀티모달** | 텍스트/이미지/음성을 함께 다루는 것 |
| **ZeroGPU** | HF Spaces의 무료 GPU 공유 시스템 |

---

## 부록 B. 에러 대응 사전

수업 자료에서 실제로 발생한 에러들입니다.

### `TypeError` (Trainer 실행 시)
```
원인: transformers와 accelerate 버전 불일치
해결: pip uninstall -y transformers accelerate
      pip install transformers==4.44.2 accelerate==0.34.0
```

### `evaluation_strategy is deprecated`
```
원인: transformers 4.46+에서 이름 변경
해결: evaluation_strategy=  →  eval_strategy=
```

### `return_all_scores is deprecated`
```
원인: text-classification pipeline 인자 변경
해결: return_all_scores=True  →  top_k=None
```

### numpy 관련 런타임 에러
```
원인: numpy 2.x와 transformers 4.44 계열 비호환
해결: pip install "numpy<2"
```

### `pin_memory` 경고
```
'pin_memory' argument is set as true but no accelerator is found
→ GPU가 없어서 나는 정보성 경고. 무시해도 됩니다.
```

### Windows symlink 경고
```
huggingface_hub cache-system uses symlinks by default...
→ 무시 가능. 없애려면 Windows 개발자 모드 켜기.
```

### 한글 그래프 깨짐 (□□□)
```python
import platform
import matplotlib.pyplot as plt
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지
```

### CSV 한글 깨짐 (엑셀에서)
```python
df.to_csv(path, index=False, encoding="utf-8-sig")   # ★ sig가 핵심
```

### 모델 예측이 엉망일 때 (진단 순서)
```
1. 모델이 해당 언어를 학습했는가?          → 모델 카드 확인
2. 파인튜닝된 모델인가? (LABEL_0이 나오면 X) → 다른 모델 찾기
3. 학습 데이터가 충분한가?                 → 최소 수백 개
4. 클래스 비율이 균형인가?                 → 50:50 확인
5. 데이터에 중복/편향이 있는가?            → shuffle + 중복 제거
```

---

## 부록 C. 실무 체크리스트

### 새 AI 프로젝트를 시작할 때
- [ ] 이게 정말 AI로 풀어야 할 문제인가? (규칙 기반으로 되면 그게 낫습니다)
- [ ] 데이터가 있는가? 몇 건인가? 출처가 신뢰할 만한가?
- [ ] 라벨이 있는가? 없으면 누가 붙이나?
- [ ] 성공 기준은? (정확도 몇 %면 쓸 만한가)
- [ ] FP와 FN 중 뭐가 더 치명적인가? → 정밀도/재현율 우선순위 결정

### 모델 선택할 때
- [ ] 태스크가 맞는가 (분류/생성/번역...)
- [ ] 언어가 맞는가
- [ ] 파인튜닝된 모델인가, 베이스 모델인가
- [ ] 라이선스가 상업적 이용 가능한가
- [ ] 모델 크기가 내 인프라에 맞는가
- [ ] 다운로드 수 / 최근 업데이트일

### 데이터 준비할 때
- [ ] 중복 제거
- [ ] 클래스 비율 확인 (심하게 불균형이면 조정)
- [ ] 셔플
- [ ] train/validation/test 분리 (보통 8:1:1 또는 8:2)
- [ ] 개인정보 제거

### 배포 전
- [ ] API 키/토큰이 코드에 없는가
- [ ] 입력이 None일 때 처리했는가
- [ ] try/except로 앱이 안 죽는가
- [ ] 응답 시간이 허용 범위인가
- [ ] 로그를 남기고 있는가

---

## 📚 참고 자료 (실제 확인한 출처)

버전 정보는 2026년 8월 5일 기준으로 실제 검색해 확인했습니다.

| 자료 | URL | 확인 내용 |
|---|---|---|
| Transformers 릴리스 | https://github.com/huggingface/transformers/releases | 최신 5.14.1 (2026-07-16 기준) |
| Transformers PyPI | https://pypi.org/project/transformers/ | 버전 이력 |
| Transformers v5 마이그레이션 가이드 | https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md | TF/Flax 제거, dtype auto, safetensors 전용, Python 3.10+ |
| Gradio PyPI | https://pypi.org/project/gradio/ | 6.21.0 (2026-07-29 릴리스) |
| Gradio Changelog | https://gradio.app/changelog | 버전별 변경사항 |
| Gradio Releases | https://github.com/gradio-app/gradio/releases | 릴리스 노트 |
| text_classification.py 소스 | https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/text_classification.py | `return_all_scores` deprecated → `top_k` |
| Pipelines 공식 문서 | https://huggingface.co/docs/transformers/en/main_classes/pipelines | pipeline API 레퍼런스 |
| Transformers 설치 가이드 | https://huggingface.co/docs/transformers/installation | Python 3.10+, PyTorch 2.5+ |
| PyTorch 설치 | https://pytorch.org | 환경별 설치 명령어 |

> ⚠️ 위 표의 날짜는 검색 결과에서 확인된 내용입니다. 각 페이지를 직접 열어 재확인하시는 걸 권합니다.
> 그 외 이 문서의 모든 실습 내용·에러 로그·모델명은 **첨부하신 수업 자료에서 직접 읽은 것**입니다.

---

*이 문서는 첨부하신 5개 zip(1~5일차 강의노트 txt 5개, 실습 .py 12개, requirements6.txt)을 전부 읽고 작성했습니다.*
