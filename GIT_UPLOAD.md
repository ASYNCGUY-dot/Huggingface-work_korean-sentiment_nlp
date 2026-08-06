# 📤 GitHub 업로드 명령어

> GitHub: **ASYNCGUY-dot**
> 저장소 이름 제안: **`korean-sentiment-nlp`** (바꾸고 싶으면 아래 전부 같이 수정)

---

## ⚠️ 시작 전 반드시 확인

이 폴더에는 **1.8GB의 모델 파일**이 있습니다.

| 파일 | 크기 |
|---|---|
| `colab_model/model.safetensors` | 423 MB |
| `my_movie_model/model.safetensors` | 679 MB |
| `real_data_model/model.safetensors` | 679 MB |

**GitHub는 파일 1개당 100MB가 한계입니다.** `.gitignore`로 전부 막아뒀지만,
아래 **1단계에서 반드시 확인**하고 넘어가세요.

---

## 1️⃣ 무엇이 올라갈지 먼저 확인 (가장 중요)

```bat
cd "C:\Users\Donga\Desktop\DONGA AI LAB\DONGA KDT AI MODEL SERVING TRAIN HUGGING FACE"
git init
git add .
git status --short
```

### ✅ 이런 게 보여야 정상

```
A  README.md
A  REPORT.md
A  requirements.txt
A  day1_pipeline/01_first_pipeline.py
A  day3_finetuning/colab_train_nsmc.ipynb
...
```

### 🚨 이런 게 보이면 즉시 중단

```
A  day3_finetuning/data/colab_model/model.safetensors     ← 절대 안 됨
A  .env                                                     ← 절대 안 됨
A  .venv/...                                                ← 절대 안 됨
```

### 용량 확인

```bat
git count-objects -vH
```

`size-pack` 이 **10MB 이하**면 정상입니다. 수백 MB면 뭔가 잘못 들어간 겁니다.

> 문제가 있으면 `git rm -r --cached .` 로 되돌리고 `.gitignore`를 고친 뒤 다시 `git add .`

---

## 2️⃣ 커밋

```bat
git config user.name "jisoo"
git config user.email "choijisu05160@gmail.com"

git commit -m "한국어 영화 리뷰 감성 분석 모델 개발 및 배포"
```

---

## 3️⃣ GitHub 저장소 생성

https://github.com/new

| 항목 | 값 |
|---|---|
| Repository name | `korean-sentiment-nlp` |
| Description | `klue/roberta-base 파인튜닝부터 배포까지 · 정확도 88%` |
| Public / Private | **Public** (포트폴리오용) |
| Add README | ❌ **체크 해제** (이미 있음) |
| Add .gitignore | ❌ **체크 해제** (이미 있음) |
| License | MIT 또는 Apache 2.0 |

**Create repository** 클릭.

---

## 4️⃣ 연결 후 push

```bat
git branch -M main
git remote add origin https://github.com/ASYNCGUY-dot/korean-sentiment-nlp.git
git push -u origin main
```

### 인증을 물어보면

| 항목 | 입력 |
|---|---|
| Username | `ASYNCGUY-dot` |
| Password | **Personal Access Token** ← 계정 비밀번호 아님 |

토큰 발급: https://github.com/settings/tokens → **Generate new token (classic)**
→ 권한 **`repo`** 체크 → 생성 → 복사

> 💡 허깅페이스와 똑같은 방식입니다. GitHub도 2021년부터 비밀번호 인증을 막았습니다.

---

## 5️⃣ 업로드 후 확인

```
https://github.com/ASYNCGUY-dot/korean-sentiment-nlp
```

- [ ] README가 첫 화면에 예쁘게 렌더링되는가
- [ ] 배지(Model / Accuracy / F1)가 보이는가
- [ ] 저장소 용량이 10MB 이하인가
- [ ] **`.env` 파일이 없는가** ← 🚨 제일 중요
- [ ] `model.safetensors` 가 없는가
- [ ] `day4_evaluation/data/*.png` 그래프는 들어갔는가

### 🚨 `.env` 가 올라갔다면

1. https://huggingface.co/settings/tokens 에서 **토큰 즉시 삭제**
2. https://github.com/settings/tokens 에서 GitHub 토큰도 삭제
3. ```bat
   git rm --cached .env
   git commit -m "remove .env"
   git push
   ```
4. **주의**: 커밋 히스토리에는 남습니다. 확실히 지우려면 저장소를 삭제하고 다시 만드는 게 안전합니다.

---

## 6️⃣ 마무리 — 저장소 꾸미기

GitHub 저장소 페이지 우측 **⚙️ 톱니바퀴**(About) 클릭:

**Description**
```
klue/roberta-base 를 NSMC 2만건으로 파인튜닝한 한국어 감성 분석 모델 · 정확도 88%
```

**Website**
```
https://huggingface.co/jisooC/nsmc-sentiment-klue
```

**Topics** (검색 노출에 도움됩니다)
```
nlp  korean  huggingface  transformers  sentiment-analysis
bert  roberta  fine-tuning  gradio  pytorch  nsmc
```

---

## 📌 다음에 코드를 고쳤을 때

```bat
git add .
git commit -m "무엇을 바꿨는지 한 줄"
git push
```

세 줄이면 됩니다.
