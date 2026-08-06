# 🎮 STAGE 2 — 토크나이저와 데이터셋

> 원본 자료: `2일차 수업내용-tokenizer`

---

## 이 스테이지의 목표

**"어제 남긴 미스터리를 전부 해결하고, 모델에 넣을 데이터를 직접 만든다."**

STAGE 1에서 우리는 계속 이런 말을 했습니다.

> "한국어가 의미 없는 조각으로 부서집니다"
> "`좋지 않아`에서 `좋`이 긍정 신호를 들고 갑니다"
> "모델과 토크나이저가 다르면 성능이 붕괴합니다"

**전부 제 추측이었습니다.** 오늘은 그걸 눈으로 확인합니다.

---

## 미션 순서

| 순서 | 파일 | 배우는 것 | 실행 |
|---|---|---|---|
| 1 | `01_see_the_breakage.py` | 🔍 어제 사건의 현장 검증 | `python 01_see_the_breakage.py` |
| 2 | `02_bpe_vs_wordpiece.py` | BPE vs WordPiece, 데이터 크기의 영향 | `python 02_bpe_vs_wordpiece.py` |
| 3 | `03_special_tokens.py` | 특수토큰, 패딩, attention_mask | `python 03_special_tokens.py` |
| 4 | `04_dataset_explore.py` | DatasetDict, filter/shuffle/split/map | `python 04_dataset_explore.py` |
| 5 | `05_full_pipeline.py` | 🔥 보스 — CSV에서 텐서까지 | `python 05_full_pipeline.py` |
| 6 | `QUIZ_day2.md` | 자가 진단 | 파일 열기 |

---

## 폴더 구조

```
day2_tokenizer/
├─ data/
│  ├─ sample_small.txt   ← 문장 1개 (수업과 동일 조건)
│  ├─ sample_large.txt   ← 문장 60개 (비교 실험용)
│  └─ (실행하면 여기에 결과물이 쌓입니다)
├─ 01 ~ 05 실습 파일
└─ QUIZ_day2.md
```

---

## 실행 전

```bat
cd "C:\Users\Donga\Desktop\DONGA AI LAB\DONGA KDT AI MODEL SERVING TRAIN HUGGING FACE"
.venv\Scripts\activate
cd day2_tokenizer
```

> ⚠️ **STAGE 1의 그 실수를 반복하지 마세요.** `cd day2_tokenizer` 를 빼먹으면 파일을 못 찾습니다.

---

## ⏱️ 다운로드 부담이 적습니다

STAGE 1과 달리 이번엔 **토크나이저 파일(수 MB)** 만 받습니다. 모델 가중치(수백 MB)는 안 받아요.
예외는 MISSION 4의 데이터셋(약 900KB)뿐입니다. 전체적으로 빠릅니다.

---

## 반드시 가져가야 할 4가지

### 1️⃣ 토크나이저와 모델은 한 몸이다
모델은 "3번 ID = 안녕"이라고 **외운 상태**입니다. 토크나이저를 바꾸면 번호 체계가 어긋나서 모델이 전부 오작동합니다. 어제 배운 "tokenizer 인자는 생략이 안전하다"의 진짜 이유입니다.

### 2️⃣ `[CLS]` 자리가 문장 전체의 요약이다
분류 모델은 문장 전체를 보는 게 아니라 **`[CLS]` 위치의 벡터 하나만** 보고 판단합니다.

### 3️⃣ `attention_mask`가 없으면 모델이 빈칸을 읽는다
`1` = 진짜 내용 / `0` = 무시할 빈칸. 빼먹으면 짧은 문장일수록 성능이 이상해집니다.

### 4️⃣ 전처리는 한 번 하고 저장한다
`save_to_disk()` → `load_from_disk()`. 학습 실험을 10번 돌릴 때 토크나이징을 10번 하면 안 되니까요.

---

## 자주 만나는 에러

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: tokenizers` | 패키지 누락 | `pip install tokenizers` |
| `ModuleNotFoundError: datasets` | 패키지 누락 | `pip install datasets==3.2.0` |
| `FileNotFoundError: sample_small.txt` | 폴더 위치 | `cd day2_tokenizer` 확인 |
| `numpy` 관련 런타임 에러 | numpy 2.x 충돌 | `pip install "numpy<2"` |
| 데이터셋 다운로드 실패 | 네트워크 | 잠시 후 재시도 |

---

## 클리어 조건

- [ ] 5개 파일 실행 완료
- [ ] `data/prepared_movie_dataset` 폴더 생성 확인 (STAGE 3 재료)
- [ ] `QUIZ_day2.md` 70% 이상
- [ ] 각 파일의 🎯 미션 최소 1개씩

완료되면 **"STAGE 2 클리어"** 라고 말씀해주세요.
