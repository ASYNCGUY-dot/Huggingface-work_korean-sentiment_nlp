---
title: korean-sentiment
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

# 한국어 영화 리뷰 감성 분석기

`klue/roberta-base` 를 네이버 영화 리뷰(NSMC) 20,000건으로 파인튜닝한 모델입니다.

## 성능

| 지표 | 값 |
|---|---|
| 정확도 | 88.0% (±1.4%) |
| 정밀도 | 0.870 |
| 재현율 | 0.884 |
| F1 | 0.877 |

NSMC test split 2,000건으로 측정했습니다. 학습에 사용되지 않은 데이터입니다.

## 한계

- 영화 리뷰 도메인 전용입니다.
- 부정 어휘가 없는 우회 표현에 약합니다.
- 2지선다 모델이라 "중립"을 출력할 수 없습니다.
- 128토큰을 초과하면 문장이 잘립니다.

## 모델

- 베이스: [klue/roberta-base](https://huggingface.co/klue/roberta-base)
- 파인튜닝 모델: [jisooC/nsmc-sentiment-klue](https://huggingface.co/jisooC/nsmc-sentiment-klue)
- 데이터: [NSMC](https://huggingface.co/datasets/e9t/nsmc)
