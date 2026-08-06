# 🚀 허깅페이스 배포 실전 가이드

> STAGE 5 - MISSION 4
> 5일차 수업의 배포 절차를 실제로 수행합니다.

---

## 전체 그림

```
① 허깅페이스 계정 + 이메일 인증
② Access Token 발급 (Write 권한)
③ 모델을 '모델 허브'에 업로드     ← 443MB, 한 번만
④ Space 생성
⑤ 코드를 Space에 git push        ← 6KB, 앞으로 자주
⑥ 빌드 로그 확인
```

> 💡 **왜 ③과 ⑤를 나누나?**
> 모델은 무겁고 잘 안 바뀝니다. 코드는 가볍고 자주 바뀝니다.
> 분리해두면 UI를 고칠 때마다 443MB를 다시 올리지 않아도 됩니다.

---

## ① 계정 준비

1. https://huggingface.co/join 가입
2. **이메일 인증 필수** — 안 하면 Space 생성이 막힙니다
   - 수업 노트에도 기록된 함정입니다
   - Settings → 이메일 확인 → 인증 메일 발송

---

## ② Access Token 발급

https://huggingface.co/settings/tokens → **New token**

| 항목 | 값 |
|---|---|
| Name | `deploy-token` (아무거나) |
| Type | **Write** ← ⚠️ Read로 하면 push가 거부됩니다 |

발급된 `hf_...` 를 복사해서 **`.env` 파일에** 넣으세요.

```bat
cd "C:\Users\Donga\Desktop\DONGA AI LAB\DONGA KDT AI MODEL SERVING TRAIN HUGGING FACE"
notepad .env
```

```
HUGGINGFACE_TOKEN=hf_여기에붙여넣기
```

> 🚨 `.env.example` 이 아니라 **`.env`** 입니다. 이미 한 번 겪으셨죠.

---

## ③ 모델을 허브에 업로드

### 터미널에서 로그인

```bat
.venv\Scripts\activate
pip install huggingface_hub
huggingface-cli login
```

토큰을 붙여넣습니다. (입력해도 화면에 안 보이는 게 정상입니다)

### 업로드

```bat
huggingface-cli upload yeansu/nsmc-sentiment-klue "day3_finetuning\data\colab_model" . --repo-type=model
```

> `yeansu` 자리에 **본인 사용자명**을 넣으세요.
> 저장소가 없으면 자동으로 만들어집니다.
> 443MB라 몇 분 걸립니다.

### 확인

`https://huggingface.co/본인이름/nsmc-sentiment-klue` 접속

Files 탭에 `model.safetensors`, `config.json`, `tokenizer.json` 등이 보이면 성공입니다.

### 파이썬으로 하고 싶다면

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("nsmc-sentiment-klue", repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="day3_finetuning/data/colab_model",
    repo_id="yeansu/nsmc-sentiment-klue",
    repo_type="model",
)
```

---

## ④ Space 생성

https://huggingface.co/new-space

| 항목 | 값 |
|---|---|
| Space name | `korean-sentiment` |
| License | Apache 2.0 |
| SDK | **Gradio** |
| Hardware | CPU basic (무료) |
| Visibility | Public |

---

## ⑤ 코드 push

```bat
cd day5_deploy

git clone https://huggingface.co/spaces/yeansu/korean-sentiment
cd korean-sentiment

REM 배포 패키지 내용을 복사
copy ..\deploy_package\* .

git add .
git commit -m "deploy: 한국어 감성 분석기"
git push origin main
```

**사용자명**과 **비밀번호**를 물어보면:
- 사용자명: 허깅페이스 아이디
- 비밀번호: **토큰(`hf_...`)** ← 계정 비밀번호가 아닙니다

---

## ⑥ 빌드 확인

```
https://huggingface.co/spaces/yeansu/korean-sentiment?logs=build
```

| 상태 | 의미 |
|---|---|
| `Building` | 패키지 설치 중 (3~10분) |
| `Running` | 🎉 성공 |
| `Build error` | 로그를 읽어야 합니다 ↓ |

---

## 🚨 빌드 실패 대응표

수업 노트와 실무에서 자주 나오는 것들입니다.

| 에러 메시지 | 원인 | 해결 |
|---|---|---|
| `No such file: app.py` | 파일명이 다름 | 반드시 `app.py` |
| `ERROR: Could not find a version` | 버전 고정이 빡빡함 | requirements에서 `==버전` 제거 |
| `theme` 관련 `TypeError` | Gradio 6.x 변경 | `gr.Blocks(theme=)` → `demo.launch(theme=)` |
| `Repository not found` | 모델 이름 오타 / private | 이름 확인, Public으로 변경 |
| `CUDA out of memory` | 무료 CPU인데 GPU 코드 | `device` 지정 제거 |
| 빌드는 됐는데 화면이 안 뜸 | `demo.launch()` 누락 | `if __name__ == "__main__":` 안에 있는지 확인 |

### 로그 읽는 순서

1. **맨 아래부터** 위로 읽으세요. 마지막 에러가 진짜 원인입니다
2. `Traceback` 이 보이면 그 아래 마지막 줄이 핵심
3. `pip` 관련이면 requirements 문제
4. `import` 관련이면 패키지 누락

---

## ⚡ (선택) ZeroGPU 사용

무료 GPU를 쓰고 싶다면:

```python
import spaces

@spaces.GPU
def predict(text, threshold, review_band):
    ...
```

`requirements.txt` 에 `spaces` 추가.

> ⚠️ Space 설정에서 Hardware를 **ZeroGPU** 로 바꿔야 합니다.
> 우리 모델은 CPU로도 충분히 빠르니 지금은 필요 없습니다.

---

## ✅ 배포 후 체크리스트

- [ ] 앱이 실제로 열리는가
- [ ] 예시 버튼이 작동하는가
- [ ] **빈 입력**을 넣어도 안 죽는가
- [ ] 아주 긴 문장을 넣어도 안 죽는가
- [ ] 이모지·특수문자를 넣어도 안 죽는가
- [ ] 임계값 슬라이더가 결과를 바꾸는가
- [ ] `.env` 나 토큰이 저장소에 안 올라갔는가 ← **꼭 확인**
- [ ] README의 한계 설명이 정확한가

### 토큰 유출 확인

```
https://huggingface.co/spaces/본인이름/korean-sentiment/tree/main
```

파일 목록에 `.env` 가 있으면 **즉시 토큰을 폐기**하고 다시 발급하세요.

---

## 🎯 미션

| 난이도 | 과제 |
|---|---|
| ★ | 배포를 완료하고 URL을 공유해보세요 |
| ★★ | README의 '한계' 항목에 본인이 발견한 것을 추가 |
| ★★★ | 앱에 피드백 버튼을 달고 재배포 (`git push` 한 번이면 됩니다) |

---

## 💡 배포 후에 알게 되는 것들

배포는 끝이 아니라 시작입니다.

- **사람들은 예상 못 한 입력을 넣습니다.** 영화 리뷰 앱에 일기를 씁니다
- **모델은 시간이 지나면 낡습니다.** 신조어, 새 영화 제목을 모릅니다
- **로그가 다음 학습 데이터입니다.** 사용자가 뭘 넣는지가 가장 좋은 데이터셋입니다

> 이걸 **MLOps** 라고 합니다. 배포 → 모니터링 → 재학습 → 배포의 순환이죠.
> 오늘 만든 건 그 첫 바퀴입니다.
