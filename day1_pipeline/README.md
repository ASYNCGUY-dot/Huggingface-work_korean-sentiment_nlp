# 🎮 STAGE 1 — 허깅페이스 파이프라인

> 원본 자료: `1일차 수업내용-허깅페이스 생태계 입문 및 Pipeline 활용(AI 모델개발의 흐름)`

---

## 이 스테이지의 목표

**"AI 모델을 3줄로 쓸 수 있다. 그리고 왜 결과가 이상한지 진단할 수 있다."**

두 번째가 진짜 목표입니다. 3줄로 돌리는 건 누구나 합니다. 결과가 틀렸을 때 원인을 짚는 게 실력입니다.

---

## 미션 순서

| 순서 | 파일 | 배우는 것 | 실행 명령 |
|---|---|---|---|
| 1 | `01_first_pipeline.py` | pipeline 기본, 신뢰도 읽는 법 | `python 01_first_pipeline.py` |
| 2 | `02_model_matters.py` | 모델 선택이 결과를 좌우한다 | `python 02_model_matters.py` |
| 3 | `03_model_battle.py` | 베이스 모델 vs 파인튜닝 모델 | `python 03_model_battle.py` |
| 4 | `04_batch_analysis.py` | 일괄 분석 + 통계 집계 | `python 04_batch_analysis.py` |
| 5 | `05_streamlit_dashboard.py` | 🔥 보스전 — 웹 대시보드 | `streamlit run 05_streamlit_dashboard.py` |
| 6 | `QUIZ_day1.md` | 자가 진단 | 파일 열어서 풀기 |

**반드시 순서대로 하세요.** 뒤 파일이 앞 파일의 결과를 전제로 설명합니다.

---

## 실행 전 확인

터미널 프롬프트 앞에 `(.venv)` 가 붙어 있어야 합니다.

```bat
cd "C:\Users\Donga\Desktop\DONGA AI LAB\DONGA KDT AI MODEL SERVING TRAIN HUGGING FACE"
.venv\Scripts\activate
cd day1_pipeline
```

---

## ⏱️ 첫 실행이 느린 이유

각 파일을 처음 돌리면 모델을 다운로드합니다. 파일당 100MB~700MB.
`C:\Users\Donga\.cache\huggingface\hub` 에 저장되고, **2회차부터는 몇 초**로 끝납니다.

진행 바가 멈춘 것처럼 보여도 기다려주세요. 네트워크 문제면 에러가 뜹니다.

---

## 이 스테이지에서 반드시 가져가야 할 3가지

### 1️⃣ pipeline은 5단계를 자동화한 것
토크나이저 로드 → 숫자 변환 → 텐서 변환 → 모델 추론 → 확률 변환 & 라벨 매핑

### 2️⃣ 모델을 지정하지 않으면 영어 전용 모델이 붙는다
```python
pipeline("sentiment-analysis")                    # ❌ 한국어 망가짐
pipeline("sentiment-analysis", model="...")       # ✅
```

### 3️⃣ 라벨이 `LABEL_0` 이면 파인튜닝 안 된 모델이다
그런 모델은 분류에 쓰면 안 됩니다. 점수가 0.5~0.7 사이에서 놀면 사실상 찍은 겁니다.

---

## 자주 만나는 에러

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: transformers` | 가상환경 활성화 안 됨 | `.venv\Scripts\activate` |
| `symlinks ... Caching files will still work` | Windows 심볼릭 링크 경고 | 무시 가능 |
| `Some weights ... newly initialized` | 베이스 모델을 분류에 씀 | 정상 경고. MISSION 3 참고 |
| 다운로드가 멈춤 | 네트워크 / 회사 방화벽 | 잠시 후 재시도 |
| 그래프 글자가 □□□ | 한글 폰트 미설정 | `plt.rc('font', family='Malgun Gothic')` |

---

## 클리어 조건

- [ ] 5개 파일 모두 실행해서 결과를 눈으로 확인
- [ ] `QUIZ_day1.md` 에서 70% 이상
- [ ] 각 파일 하단의 🎯 미션 중 최소 1개씩 수행

전부 됐으면 저한테 **"STAGE 1 클리어"** 라고 말씀해주세요. STAGE 2 열어드립니다.
