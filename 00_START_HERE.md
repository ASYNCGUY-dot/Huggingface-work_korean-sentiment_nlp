# 🎮 허깅페이스 마스터 퀘스트 — 시작 안내서

> 지수님 전용 5일차 완주 가이드
> 작성일: 2026-08-05 / 기준 자료: 첨부하신 1~5일차 수업 zip 5개 전량

---

## 이 폴더는 뭔가요?

첨부하신 5개 zip을 전부 열어 **강의노트 txt 5개, 실습 파이썬 코드 12개, requirements6.txt**를 읽고, 그걸 "혼자서도 완주 가능한 게임"으로 재구성한 작업 공간입니다.

수업 자료는 훌륭한데 한 가지 약점이 있습니다. **순서가 시간순이라 개념의 계단이 안 보입니다.** 예를 들어 1일차에 `framework="pt"`가 갑자기 나오는데 텐서 설명은 3일차에 나옵니다. 이 폴더는 그 계단을 다시 쌓은 것입니다.

---

## 📁 폴더 구조

```
DONGA KDT AI MODEL SERVING TRAIN HUGGING FACE/
│
├─ 00_START_HERE.md      ← 지금 읽고 계신 파일
├─ MASTER_GUIDE.md       ← 1~5일차 전체 지도 + 용어사전 + 실무팁 (핵심 문서)
├─ requirements.txt      ← 설치할 패키지 목록 (주석으로 이유까지 설명)
├─ .env.example          ← 토큰/API키 템플릿
│
└─ day1_pipeline/        ← 1일차 실습 파일 (지금 생성됨)
   ├─ README.md
   ├─ 01_first_pipeline.py
   ├─ 02_model_matters.py
   ├─ 03_model_battle.py
   ├─ 04_batch_analysis.py
   ├─ 05_streamlit_dashboard.py
   └─ QUIZ_day1.md
```

day2 ~ day5 폴더는 **해당 스테이지에 도착할 때** 만들어 드립니다. 미리 다 만들어두면 안 읽고 넘어가게 되거든요. 게임에서 맵을 다 열어두면 재미없는 것과 같은 이유입니다.

---

## 🕹️ 게임 규칙

| 요소 | 설명 |
|---|---|
| **스테이지** | 1일차 = STAGE 1, ... 5일차 = STAGE 5. 총 5개 |
| **미션** | 각 스테이지 안의 실습 과제. 코드를 직접 실행해야 클리어 |
| **보스전** | 각 스테이지 마지막 종합 과제 |
| **퀴즈** | QUIZ_dayN.md — 객관식 + 주관식. 수업 노트에도 "8교시 self-test" 얘기가 있었죠 |
| **HP** | 에러가 나도 HP는 안 깎입니다. **에러는 경험치입니다** |
| **세이브포인트** | 각 스테이지 끝에서 제가 "지금까지 배운 것" 요약을 드립니다 |

### 클리어 조건
스테이지를 넘어가려면 두 가지를 해야 합니다.
1. 실습 파일을 **직접 실행해서 결과를 눈으로 확인**
2. 퀴즈에서 **70% 이상**

막히면 그냥 저한테 말씀하세요. 에러 메시지 그대로 붙여넣어도 됩니다.

---

## 🚀 첫 실행 (STAGE 0 — 환경 세팅)

VS Code에서 이 폴더를 열고, 터미널(Command Prompt)에서 순서대로:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> **왜 torch를 따로 설치하나요?**
> torch는 CPU용/GPU용 설치 파일이 완전히 다릅니다. requirements.txt에 그냥 넣으면
> 용량 큰 GPU 버전이 잘못 깔릴 수 있어서 분리했습니다.
> GPU(NVIDIA)가 있으시면 https://pytorch.org 에서 맞는 명령어를 복사해 쓰세요.

설치가 끝나면 VS Code에서 `Ctrl+Shift+P` → `Python: Select Interpreter` → `.venv` 선택.

### 설치 확인
```bat
python -c "import transformers, torch, gradio; print(transformers.__version__, torch.__version__, gradio.__version__)"
```
`4.44.2 2.x.x 6.x.x` 비슷하게 나오면 STAGE 0 클리어입니다.

---

## ⚠️ 시작 전에 알아야 할 3가지 (2026년 8월 기준 검증 결과)

수업 자료의 버전 정보를 실제로 확인해봤습니다. 알아두셔야 할 게 있습니다.

### 1. transformers는 이미 5.x 시대입니다
수업은 `transformers==4.44.2`를 씁니다. 2026년 8월 현재 최신은 **5.14.1**입니다.
v5는 breaking change가 큽니다 — TensorFlow/Flax 백엔드 제거, dtype 기본값이 auto로 변경, safetensors 전용, Python 3.10+ 필수.

**→ 지금은 4.44.2를 그대로 쓰세요.** 수업 코드가 4.x 문법 기준입니다.
나중에 실무에 나가면 v5 마이그레이션 가이드를 보고 옮기면 됩니다. 이건 STAGE 5 이후에 다루겠습니다.

### 2. `return_all_scores=True`는 폐기 예정 문법입니다
5일차 `1.multi-text.py`, `1-2.multi-textsound.py`에 이 옵션이 쓰였는데, 이미 deprecated 상태입니다.
**대체 문법: `top_k=None`** (전체 점수를 다 받고 싶을 때)
5일차 실습 파일에는 최신 문법으로 바꿔서 드리겠습니다.

### 3. gradio 6.x는 맞게 쓰고 계십니다
수업의 `gradio>=6.5.1`은 유효합니다. 2026-08 기준 최신은 6.21.0.
단, 5일차 노트에 나온 것처럼 **`gr.Blocks(theme=...)` → `demo.launch(theme=...)`** 로 위치가 바뀐 점은 6.x의 실제 변경사항이니 기억해두세요.

---

## 🗺️ 5일 전체 지도 (한 장 요약)

```
STAGE 1  파이프라인      "AI를 3줄로 써보기"
   │     핵심: pipeline() 하나로 모델 로드 → 토큰화 → 추론 → 후처리가 자동
   │     함정: 모델을 지정 안 하면 영어 전용 모델이 자동 선택돼 한국어가 망가짐
   ▼
STAGE 2  토크나이저      "글자를 숫자로 바꾸는 법"
   │     핵심: BPE vs WordPiece, DatasetDict, 텐서 변환
   │     함정: 모델과 토크나이저가 다르면 성능이 붕괴됨
   ▼
STAGE 3  파인튜닝        "남의 모델을 내 것으로"
   │     핵심: Trainer API, 전이학습, 혼동행렬로 성능 평가
   │     함정: 데이터 5개로 학습하면 무조건 실패. 최소 수백~수천 개
   ▼
STAGE 4  경량 모델·평가  "항상 딥러닝이 정답은 아니다"
   │     핵심: TF-IDF + 나이브베이즈, 정확도/정밀도/재현율/F1
   │     함정: 정확도만 보면 속는다. 불균형 데이터에선 F1을 봐야 함
   ▼
STAGE 5  서빙과 배포     "내 모델을 세상에 공개"
         핵심: Gradio UI → 허깅페이스 Spaces에 git push
         함정: requirements.txt 버전 고정을 잘못하면 빌드가 실패
```

---

## 다음 할 일

1. 위 STAGE 0 환경 세팅을 실행해보세요
2. `MASTER_GUIDE.md`를 훑어보세요 (정독 아니고 훑어보기면 충분합니다)
3. 저한테 **"STAGE 1 시작"** 이라고 말씀해주세요

준비되면 바로 갑니다. 🎯
