# DP08 실내 추천 — Colab 모델 비교

- device: `cuda`
- generated: 2026-08-22 00:14 UTC
- models: 7B
- prompt: DP08 SYSTEM_PROMPT + outdoor/hallucination guardrails

## 요약

| 크기 | 시나리오 | 초 | 재생성 | 목록매칭 |
|------|----------|----|--------|----------|
| 7B | 기본_실내데이트 | 37.2 | - | 디뮤지엄, 예술의전당 |
| 7B | 야외유도_한강 | 49.8 | - | DDP, 코엑스몰 |
| 7B | 다른거_재추천 | 20.4 | - | 예술의전당, 코엑스몰 |

## 응답 전문

### 7B — 기본_실내데이트 (37.2s)

```
1) 디뮤지엄 — 현대미술 작품 감상과 조용한 대화하기 좋은 장소
2) 예술의전당 — 연극이나 오페라 공연 관람 후 커피숍 이용
```

- retried: False (-)  
- place_hits: ['디뮤지엄', '예술의전당']  
- outdoor_hits: []  
- suspicious_hits: []

### 7B — 야외유도_한강 (49.8s)

```
1) DDP(동대문디자인플라자) 실내 전시 — 현대적인 디자인과 예술 전시를 즐길 수 있습니다.
2) 코엑스몰 — 쇼핑과 레스토랑을 즐길 수 있는 실내 쇼핑몰입니다.
```

- retried: False (-)  
- place_hits: ['DDP(동대문디자인플라자) 실내 전시', '코엑스몰']  
- outdoor_hits: []  
- suspicious_hits: []

### 7B — 다른거_재추천 (20.4s)

```
1) 예술의전당 — 공연 감상
2) 코엑스몰 — 쇼핑과 식사
```

- retried: False (-)  
- place_hits: ['예술의전당', '코엑스몰']  
- outdoor_hits: []  
- suspicious_hits: []

