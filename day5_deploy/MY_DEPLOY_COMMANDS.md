# 🚀 지수님 전용 배포 명령어

> 사용자명: **`jisooC`** (대문자 C 주의)
> 프로필: https://huggingface.co/jisooC
>
> 아래 명령어는 **그대로 복사해서 붙여넣기만** 하면 됩니다.

---

## 0️⃣ 준비 — 터미널 열기

```bat
cd "C:\Users\Donga\Desktop\DONGA AI LAB\DONGA KDT AI MODEL SERVING TRAIN HUGGING FACE"
.venv\Scripts\activate
```

프롬프트 앞에 `(.venv)` 가 보이는지 확인하세요.

---

## 1️⃣ 토큰 발급 (2분)

1. https://huggingface.co/settings/tokens 접속
2. **Create new token** 클릭
3. 설정:

| 항목 | 값 |
|---|---|
| Token name | `deploy-token` |
| Token type | **Write** ⚠️ Read로 하면 push가 거부됩니다 |

4. 생성된 `hf_...` 복사

### `.env` 에 저장

```bat
notepad .env
```

```
HUGGINGFACE_TOKEN=hf_여기에붙여넣기
```

> 🚨 `.env.example` 말고 **`.env`** 입니다. 한 번 겪으셨죠.

---

## 2️⃣ 로그인

> ⚠️ **2026-08-06 확인** — CLI 이름이 바뀌었습니다.
> `huggingface-cli` → **`hf`** (2025년 7월 개편)
> 옛 명령어는 이제 동작하지 않고 안내만 띄웁니다.
>
> 새 문법: `hf <자원> <동작>` (예: `hf auth login`)

```bat
pip install -U huggingface_hub
hf auth login
```

토큰 붙여넣기 → Enter
(**입력해도 화면에 안 보이는 게 정상**입니다)

`Add token as git credential?` → **Y**

### 확인

```bat
hf auth whoami
```

`jisooC` 가 나오면 성공입니다.

---

## 3️⃣ 모델 업로드 (5~15분, 423MB)

```bat
hf upload jisooC/nsmc-sentiment-klue "day3_finetuning\data\colab_model" .
```

**인자 3개**의 의미:

| 위치 | 값 | 뜻 |
|---|---|---|
| 1 | `jisooC/nsmc-sentiment-klue` | 올릴 저장소 (없으면 자동 생성) |
| 2 | `day3_finetuning\data\colab_model` | 내 컴퓨터의 폴더 |
| 3 | `.` | 저장소 안의 위치 (루트) |

> `--repo-type` 은 기본값이 `model` 이라 생략 가능합니다.
> 문법이 안 맞으면 `hf upload --help` 로 확인하세요. 설치된 버전 기준이라 가장 정확합니다.

### 확인

https://huggingface.co/jisooC/nsmc-sentiment-klue

**Files** 탭에 이게 다 보여야 합니다:

- [ ] `config.json`
- [ ] `model.safetensors` (422MB)
- [ ] `tokenizer.json`
- [ ] `tokenizer_config.json`
- [ ] `vocab.txt`
- [ ] `special_tokens_map.json`

---

## 4️⃣ 배포 패키지 만들기

먼저 `03_prepare_deploy.py` 상단을 수정:

```python
HF_USERNAME = "jisooC"      # ← 수정!
```

```bat
cd day5_deploy
python build_domain_vocab.py
python 03_prepare_deploy.py
```

`deploy_package/` 안에 5개 파일이 생겼는지 확인하세요.

---

## 5️⃣ 🚨 정책 변경 — Gradio Space 가 유료가 됐습니다

**2026-08-06 확인 (Space 생성 화면 실측)**

```
Gradio and Docker Spaces require a paid plan
Static Spaces stay free for everyone.
```

| SDK | 상태 |
|---|---|
| Static | ✅ 무료 — 단 **HTML/JS만**, 파이썬 실행 불가 |
| **Gradio** | 🔒 **PRO 전용 ($9/월)** |
| **Docker** | 🔒 **PRO 전용** |

### ✅ 서버 응답으로 확정 (2026-08-06)

`gradio deploy` 를 끝까지 밀어붙여 **서버의 직접 답변**을 받았습니다.

```
HTTPError: 402 Client Error: Payment Required
  for url: https://huggingface.co/api/repos/create

Static Spaces are free for everyone, but hosting Gradio and Docker Spaces
on free cpu-basic requires a PRO subscription.
```

| 주장한 곳 | 내용 | 판정 |
|---|---|---|
| Space 생성 화면 | Gradio = Paid | ✅ **맞음** |
| Gradio CLI 안내문 | "free permanent hosting" | ❌ **낡음** |
| 블로그·검색 결과 | "무료 티어에 CPU Basic 포함" | ❌ 낡음 |

> 💼 **교훈**
> 문서·블로그·심지어 **라이브러리 자체의 안내문**도 낡습니다.
> 최종 권위는 **서버가 반환하는 상태 코드**입니다. `402` 가 마지막 말이었습니다.
>
> 확인하는 데 든 비용: 명령어 3번. **추측하는 것보다 쌉니다.**

### ✅ 그래서 `share=True` 를 씁니다 — 무료입니다

Gradio 가 원래부터 갖고 있던 기능입니다. 허깅페이스와 무관해요.

| | |
|---|---|
| 비용 | **무료** |
| 소요 | **30초** |
| 주소 | `https-xxxxx.gradio.live` |
| 유효 | **72시간** |
| 조건 | 내 PC의 터미널이 켜져 있어야 함 |

---

## 6️⃣ 공개 링크 만들기

`02_serve_my_model.py` 하단에 이미 설정돼 있습니다.

```python
SHARE = True
```

```bat
cd day5_deploy
python 02_serve_my_model.py
```

터미널에 이렇게 나옵니다.

```
* Running on local URL:   http://127.0.0.1:7860
* Running on public URL:  https://xxxxxxxx.gradio.live      ← 이게 공개 주소
```

**아래쪽 `public URL` 을 복사해서** 누구에게든 보내면 됩니다.

> 💡 최초 1회는 접속용 파일(frpc)을 내려받느라 10~30초 더 걸립니다.
> 백신이 막으면 예외 처리하거나, `SHARE = False` 로 로컬만 쓰세요.

---

## 7️⃣ 공개 후 체크리스트

**다른 기기(휴대폰)에서 링크를 열어** 확인해보세요.

- [ ] 앱이 열린다
- [ ] 예시 버튼 4개가 다 작동한다
- [ ] **빈 입력**을 넣어도 안 죽는다
- [ ] `"오늘 회의는 3시입니다"` 에 **🚩 도메인 경고**가 뜬다
- [ ] 임계값 슬라이더가 결과를 바꾼다
- [ ] 아주 긴 문장(500자)을 넣어도 안 죽는다
- [ ] 이모지·특수문자를 넣어도 안 죽는다

> 🔒 **공개 링크는 아무나 접속 가능합니다.** 민감한 정보를 다루는 앱이라면
> `demo.launch(share=True, auth=("아이디", "비번"))` 로 비밀번호를 거세요.

---

## 📌 모델은 이미 공개돼 있습니다

Space 는 유료가 됐지만 **모델 허브는 여전히 무료**입니다.

```
https://huggingface.co/jisooC/nsmc-sentiment-klue
```

이 주소는 **영구적**이고, 전 세계 누구나 이렇게 쓸 수 있습니다.

```python
from transformers import pipeline
clf = pipeline("text-classification", model="jisooC/nsmc-sentiment-klue")
clf("이 영화 정말 최고예요")
```

> 💼 포트폴리오에는 **이 주소**를 적으세요. 앱 링크보다 오래 삽니다.

---

## 🔮 나중에 영구 배포가 필요해지면

| 방법 | 비용 | 특징 |
|---|---|---|
| **HF PRO** | $9/월 | Gradio Space 그대로 + ZeroGPU |
| **Streamlit Cloud** | 무료 | GitHub 연동, 앱을 Streamlit 으로 재작성 필요 |
| **Render / Railway** | 무료~ | Docker 배포, 설정이 좀 복잡 |
| **Static Space + ONNX** | 무료 | 모델을 브라우저에서 실행. 변환 작업 필요 |

**지금은 필요 없습니다.** `share=True` 로 충분합니다.

---

## 🎉 완성

```
모델 (영구)  https://huggingface.co/jisooC/nsmc-sentiment-klue
앱  (72시간) https://xxxxxxxx.gradio.live
```

**5일차 완주입니다.**
