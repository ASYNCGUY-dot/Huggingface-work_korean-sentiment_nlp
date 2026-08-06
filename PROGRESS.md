# ✅ 완주 기록

> 최종 갱신 2026-08-06
> **아래 숫자는 전부 지수님이 직접 실행해서 나온 실측값입니다.**

---

## 완주 현황

```
STAGE 1  파이프라인          ✅
STAGE 2  토크나이저          ✅
STAGE 3  파인튜닝            ✅   모델 완성 (검증 88%)
STAGE 4  성능 평가           ✅   배포 게이트 통과 · 모델 카드 생성
STAGE 5  서빙과 배포         ✅   모델 허브 공개 + 앱 공개
보너스   멀티모달            🔶   3개 중 1개 완료
```

---

## 🔗 공개된 결과물

| | 주소 | 유효 |
|---|---|---|
| **모델** | https://huggingface.co/jisooC/nsmc-sentiment-klue | **영구** |
| 앱 | `https://8eff409fffbfac937e.gradio.live` | 최대 1주일 |

```python
from transformers import pipeline
clf = pipeline("text-classification", model="jisooC/nsmc-sentiment-klue")
clf("이 영화 정말 최고예요")
```

---

## 📊 최종 성능

NSMC test split 2,000건 (학습에 사용되지 않은 데이터)

| 지표 | 값 |
|---|---|
| 정확도 | **88.0%** (95% CI 86.6~89.4%) |
| 정밀도 | 0.8698 |
| 재현율 | 0.8842 |
| F1 | 0.8769 |

|  | 예측 부정 | 예측 긍정 |
|---|---|---|
| 실제 부정 | 905 | 128 |
| 실제 긍정 | 112 | 855 |

**일반화 검증**: Colab 검증 88% = test split 88.00%. 격차 0%p.

---

## 🎯 데이터가 성능을 좌우함을 실험으로 확인

같은 모델·코드·하이퍼파라미터, **데이터만 변경**

| 학습 데이터 | 고유 어휘 | 검증 | 실전 |
|---|---|---|---|
| 템플릿 600건 | 74 | 100% | 80% |
| NSMC 1,500건 | 8,172 | 70% | 50% |
| **NSMC 20,000건** | **70,541** | 88% | **100%** |

---

## 📁 산출물

| 파일 | 위치 |
|---|---|
| 학습된 모델 (423MB) | `day3_finetuning/data/colab_model/` |
| Colab 노트북 | `day3_finetuning/colab_train_nsmc.ipynb` |
| 모델 카드 HTML | `day4_evaluation/data/model_card.html` |
| 혼동행렬·PR곡선 그림 | `day4_evaluation/data/*.png` |
| 스팸 모델 | `day4_evaluation/data/spam_model.pkl` |
| 배포 패키지 | `day5_deploy/deploy_package/` |
| 도메인 어휘 사전 | `day5_deploy/domain_vocab.json` |
| **포트폴리오 문서** | `PORTFOLIO.md` |

---

## 🔶 남은 선택 항목

```bat
cd day5_deploy\bonus
python 01_image_classify.py    # ViT 이미지 분류 + BLIP 설명 (350MB)
python 03_voice_pipeline.py    # Whisper → 번역 → TTS (290MB~)
```

퀴즈 3세트(`QUIZ_day1~3.md`)도 그대로 있습니다.

---

## 🔧 진행 중 해결한 문제 12건

| # | 문제 | 해결 |
|---|---|---|
| 1 | `.env.example`에 실제 토큰 | 폐기 + `.gitignore` 강화 |
| 2 | 2지선다 기준을 5지선다에 적용 | 배율(ratio) 방식 |
| 3 | 정렬 데이터 앞에서 자름 → 100:0 | shuffle + stratify |
| 4 | `max_length=64` (9% 잘림) | 프리셋 + 96 |
| 5 | 합성 데이터 3회 연속 F1=1.0 | `check_separability()` 신설 |
| 6 | 도메인 판정 "겹침 0개" | 비율 35% 기준 |
| 7 | 검토 밴드 고정 ±0.15 (80% 검토) | 예산 역산 (±0.075) |
| 8 | 도메인 체크가 토크나이저 사전 참조 | NSMC 학습 어휘로 교체 |
| 9 | `huggingface-cli` 폐기 | `hf` CLI 전환 |
| 10 | `huggingface_hub` 1.x 충돌 | 버전 상한 `<1.0` |
| 11 | `gradio deploy` KeyError | 토큰 로그인으로 전환 |
| 12 | YAML `title`에 한글 | ASCII로 변경 |

---

## 📌 확정된 사실 (2026-08-06 서버 응답)

```
HTTPError: 402 Payment Required
Static Spaces are free for everyone, but hosting Gradio and Docker Spaces
on free cpu-basic requires a PRO subscription.
```

허깅페이스 **Gradio Space는 PRO 전용**입니다. 문서·블로그·Gradio CLI 안내문은 아직 낡은 정보를 표시합니다.

**무료 대안**: `demo.launch(share=True)` — 최대 1주일 공개 링크.

---

## 🎬 5일의 궤적

| Day | 질문 | 답 |
|---|---|---|
| 1 | 왜 이 모델은 바보인가 | 분류 헤드가 랜덤 (점수차 0.0038) |
| 2 | 왜 한국어가 부서지나 | `uncased` → 자모 분해 |
| 3 | 왜 100%인데 실전 80%인가 | 어휘 사각지대 (`좋` 0회) |
| 4 | 12%는 어디서 틀렸나 | FP 128 / FN 112 |
| 5 | 세상에 내놓기 | 모델 + 앱 공개 |
| 보너스 | 왜 중립을 못 냈나 | **출력 칸이 없었다** |

**Day 1의 미스터리가 마지막 날 풀렸습니다.**

```
"The meeting is scheduled for 3 PM."
  Day 1 (2클래스) → POSITIVE 96.7%
  Day 5 (7클래스) → 중립 91.2%  ·  기쁨 1%
```
