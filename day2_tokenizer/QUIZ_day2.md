# 🧩 STAGE 2 자가 진단 퀴즈

> 15문제. 11개(70%) 이상이면 STAGE 3으로 진행합니다.
> **먼저 풀고** 맨 아래 정답을 보세요.

---

## Part A. 객관식 (10문제)

**Q1.** 토크나이저의 역할은?
1. 모델의 정확도를 높인다
2. 글자를 모델이 계산할 수 있는 숫자로 바꾼다
3. 문장의 감정을 판단한다
4. 학습 속도를 조절한다

---

**Q2.** BPE와 WordPiece의 병합 기준으로 옳게 짝지은 것은?
1. BPE=확률, WordPiece=빈도
2. BPE=빈도, WordPiece=확률(likelihood)
3. 둘 다 빈도
4. 둘 다 확률

---

**Q3.** WordPiece 결과에서 `##씨`의 `##`가 뜻하는 것은?
1. 특수 문자
2. 이 조각은 앞 조각에 이어 붙은 것
3. 모르는 단어
4. 문장의 끝

---

**Q4.** 수업 실습에서 `[UNK]`가 대량으로 나온 근본 원인은?
1. WordPiece 알고리즘의 결함
2. 한국어를 지원하지 않아서
3. 학습 데이터가 1문장뿐이라 `min_frequency=2`를 넘긴 조각이 거의 없어서
4. `vocab_size`가 너무 커서

---

**Q5.** 분류 모델이 문장 전체를 판단할 때 실제로 보는 위치는?
1. 첫 단어
2. 마지막 단어
3. `[CLS]` 토큰 자리
4. 모든 토큰의 평균

---

**Q6.** `attention_mask`의 `0`이 의미하는 것은?
1. 부정적인 단어
2. 패딩된 빈칸이므로 무시하라
3. 모르는 단어
4. 문장의 경계

---

**Q7.** `padding="max_length"`가 필요한 이유는?
1. 정확도를 높이려고
2. 길이가 제각각이면 배치로 묶어 행렬 연산을 할 수 없어서
3. 한글 처리를 위해
4. 저장 용량을 줄이려고

---

**Q8.** DatasetDict를 train/validation/test로 나누는 이유는?
1. 저장 공간 절약
2. 학습 속도 향상
3. 학습에 쓴 데이터로 평가하면 과적합을 발견할 수 없어서
4. 허깅페이스 규격이라서

---

**Q9.** `shuffle(seed=42)`에서 `seed`의 역할은?
1. 섞는 강도 조절
2. 재현 가능한 무작위 — 같은 seed면 항상 같은 순서
3. 데이터 개수 제한
4. 라벨 비율 유지

---

**Q10.** `map(fn, batched=True)`의 `batched=True` 효과는?
1. 결과가 더 정확해진다
2. 데이터를 묶음으로 넘겨 처리 속도가 크게 빨라진다
3. 메모리를 적게 쓴다
4. 라벨이 자동 생성된다

---

## Part B. 주관식 (5문제)

**Q11.** 전처리 파이프라인 6단계를 순서대로 쓰세요.

```
pandas → ① ______ → ② ______ → ③ ______ → ④ ______ → ⑤ ______ → 저장
```

---

**Q12.** 모델과 토크나이저를 반드시 짝으로 써야 하는 이유를 설명하세요. (STAGE 1에서 배운 "다른 tokenizer 쓰면 성능 붕괴"와 연결해서)

---

**Q13.** 특수 토큰 5개의 이름과 역할을 각각 쓰세요.

---

**Q14.** `remove_columns(["content"])`를 빼먹으면 STAGE 3에서 어떤 문제가 생기나요? 왜죠?

---

**Q15.** 아래 코드에 문제가 3개 있습니다. 찾아서 고치세요.

```python
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd

df = pd.read_csv("movie.csv")
ds = Dataset.from_pandas(df)
split = ds.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def preprocess(examples):
    return tokenizer(examples["content"], max_length=512)

tokenized = split.map(preprocess)
tokenized.set_format("torch")
```

---
---
---

# ✅ 정답 및 해설

<details>
<summary>클릭해서 펼치기</summary>

## Part A

**Q1. → 2번**
토크나이저는 변환기입니다. 판단은 모델이 합니다.

**Q2. → 2번**
BPE는 "자주 붙어 나오는 쌍"을 합칩니다. WordPiece는 "언어모델 확률을 가장 높이는 쌍"을 합칩니다.

**Q3. → 2번**
`날` + `##씨` = 원래 한 단어였다는 표시. 나중에 원문 복원이 가능해집니다.

**Q4. → 3번**
`min_frequency=2`는 "2번 이상 나온 조각만 사전에 등록"입니다. 문장 1개에서는 대부분의 글자가 1번만 나오니 전부 탈락합니다. MISSION 2에서 데이터를 60배로 늘려 직접 증명했습니다.

**Q5. → 3번**
`[CLS]` 위치의 출력 벡터가 문장 전체의 요약 역할을 합니다. 분류 헤드는 그 벡터 하나만 입력으로 받습니다.

**Q6. → 2번**
`1` = 진짜 내용, `0` = 무시할 패딩.

**Q7. → 2번**
행렬 연산은 모양이 맞아야 합니다. 길이가 제각각이면 배치를 만들 수 없습니다.

**Q8. → 3번**
학습 데이터로 평가하면 "답을 외운" 상태라 무조건 만점입니다. 과적합을 못 잡습니다.

**Q9. → 2번**
같은 seed면 몇 번을 돌려도 같은 순서. 실험 재현성의 핵심입니다.

**Q10. → 2번**
1건씩 함수 호출 대신 1000건씩 묶어 넘깁니다. 토크나이징처럼 무거운 작업에서 차이가 큽니다.

---

## Part B

**Q11.**
```
pandas → ① Dataset 변환 → ② train_test_split → ③ 토크나이징(map, batched=True)
       → ④ remove_columns → ⑤ set_format("torch") → 저장(save_to_disk)
```

**Q12. 예시 답안**
> 토크나이저는 "글자 ↔ 번호" 사전을 갖고 있고, 모델은 그 번호 체계를 전제로 학습되어 있다. 모델 입장에서 "3번 = 안녕"이라고 외운 상태다.
>
> 다른 토크나이저를 쓰면 같은 글자에 다른 번호가 붙는다. 모델은 그 번호를 자기가 아는 의미로 해석하므로, 입력 전체가 다른 뜻으로 왜곡된다. 에러는 나지 않고 결과만 조용히 망가진다.
>
> 그래서 `tokenizer` 인자는 생략하는 것이 가장 안전하다. 생략하면 모델과 짝인 것이 자동 로드된다.

**Q13.**
| 토큰 | 역할 |
|---|---|
| `[PAD]` | 길이를 맞추기 위한 빈칸 |
| `[UNK]` | 사전에 없는 조각 |
| `[CLS]` | 문장 시작 + 전체 요약 자리 (분류에 사용) |
| `[SEP]` | 문장 끝 / 문장 간 경계 |
| `[MASK]` | 학습 시 단어를 가리는 용도 |

**Q14. 예시 답안**
> 학습 시 에러가 난다. 모델은 숫자 텐서만 입력으로 받는데, `content`는 문자열 컬럼이라 텐서로 변환할 수 없다. Trainer가 배치를 만들 때 문자열을 텐서로 쌓으려다 실패한다. 전처리가 끝나면 원본 텍스트는 모델 입장에서 이물질이다.

**Q15. 문제 3개와 수정**

```python
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd

df = pd.read_csv("movie.csv")
ds = Dataset.from_pandas(df)

# 문제 ①: seed 미지정 → 실행할 때마다 다르게 나뉘어 재현 불가
split = ds.train_test_split(test_size=0.2, seed=42)

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def preprocess(examples):
    # 문제 ②: truncation / padding 누락 → 길이가 제각각이라 배치 불가
    #          max_length=512 도 과함. 데이터 길이를 재고 정할 것
    return tokenizer(
        examples["content"],
        truncation=True,
        padding="max_length",
        max_length=64,
    )

# 문제 ③: batched=True 누락 → 1건씩 처리해서 매우 느림
tokenized = split.map(preprocess, batched=True)

# (추가) remove_columns 도 빠져 있습니다. 있으면 더 좋습니다.
tokenized = tokenized.remove_columns(["content"])
tokenized.set_format("torch")
```

</details>

---

| 점수 | 다음 |
|---|---|
| 11개 이상 | 🎉 "STAGE 2 클리어" 라고 알려주세요 |
| 8~10개 | 틀린 문항의 해설을 읽고 해당 실습 재실행 |
| 7개 이하 | `MASTER_GUIDE.md` Part 2 다시 읽고 재도전 |
