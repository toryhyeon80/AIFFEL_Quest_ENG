---
omd: 0.1
brand: LogPick Core
reference: oh-my-design (MIT) — OmD v0.1 + Google Stitch DESIGN.md
inspiration: Linear (admin density), Stripe (B2B clarity), KR B2B SaaS tone
stack: Next.js + Tailwind (frontend-user), React/Vite + Tailwind (frontend-admin)
---

# DESIGN.md — LogPick Core

> **에이전트 지침**: UI·컴포넌트·마이크로카피·레이아웃을 생성하기 전에 본 문서를 읽고 적용하십시오.  
> 토큰만으로는 부족합니다. §10–15(브랜드 철학)을 제약 조건으로 취급하십시오.  
> 스펙 기반: [oh-my-design OmD v0.1](https://github.com/kwakseongjae/oh-my-design/blob/main/spec/omd-v0.1.md) (MIT)

### Design Hierarchy

1. **항상** 루트 `DESIGN.md`(본 문서)를 먼저 적용합니다.
2. 작업 중인 기능 폴더에 `DESIGN.md`가 있으면 **함께** 읽고, 화면 배치·기능 전용 UI는 기능 문서를 따릅니다.
3. **충돌 시**: 색상·폰트·Voice·공통 States → 루트 우선 / 화면 레이아웃·기능 컴포넌트 → 기능 `DESIGN.md` 우선.
4. 기능별 문서 위치·명명: `ARCHITECTURE.md` §3~§4, `docs/architecture/WEB_CORE.md` §3 참조.

---

## 1. Visual Theme & Atmosphere

LogPick Core는 **모듈형 SaaS 백오피스의 중심축**입니다. 화면은 두 가지 모드로 나뉩니다.

| Surface | 분위기 | 목표 |
|---------|--------|------|
| **User Web** (`frontend-user`) | 밝고 개방적, 신뢰감 있는 기업 소개 | 방문자가 3초 안에 "무엇을 하는 회사인지" 이해 |
| **Admin** (`frontend-admin`) | 차분하고 밀도 있는 작업 공간 | 운영자가 콘텐츠를 빠르게 수정·저장 |

전체 무드: **정돈된 전문성**. 화려한 그라데이션, 장식용 일러스트, 과도한 그림자는 금지합니다.  
흰 배경 위에 **LogPick Navy**와 **LogPick Teal**이 구조와 상호작용을 나눕니다.

---

## 2. Color Palette & Roles

색상은 Tailwind 확장 또는 CSS 변수로 매핑합니다. 이름과 hex를 항상 함께 사용하십시오.

### Brand

| Token | Hex | Role |
|-------|-----|------|
| **LogPick Navy** | `#1E3A5F` | 헤더, 주요 제목, Admin 사이드바 배경 |
| **LogPick Teal** | `#0D9488` | Primary CTA, 링크, 포커스 링, 활성 탭 |
| **LogPick Teal Hover** | `#0F766E` | Primary 버튼 hover |
| **LogPick Teal Soft** | `#CCFBF1` | 선택·성공 하이라이트 배경 |

### Neutrals

| Token | Hex | Role |
|-------|-----|------|
| **Surface White** | `#FFFFFF` | 카드, 폼, Admin 메인 캔버스 |
| **Surface Mist** | `#F8FAFC` | 페이지 배경 (User Web) |
| **Surface Slate** | `#F1F5F9` | Admin 페이지 배경, zebra row |
| **Border Line** | `#E2E8F0` | 구분선, 입력 필드 border |
| **Text Primary** | `#0F172A` | 본문, 제목 |
| **Text Secondary** | `#64748B` | 보조 설명, 메타 정보 |
| **Text Muted** | `#94A3B8` | placeholder, 비활성 |

### Semantic

| Token | Hex | Role |
|-------|-----|------|
| **Signal Red** | `#DC2626` | 오류, 삭제 확인 |
| **Signal Amber** | `#D97706` | 경고, 미저장 변경 |
| **Signal Green** | `#059669` | 저장 완료, 검증 통과 |

**Why:** Navy는 레거시 ERP·관리자 도구의 무게감을, Teal은 "확장 가능한 모듈"이라는 성장 이미지를 전달합니다. 보라색·그라데이션 Hero는 LogPick가 거부하는 시각 언어입니다.

---

## 3. Typography Rules

### Font Stack

```css
/* Korean + Latin co-equal */
font-family: "Pretendard Variable", Pretendard, -apple-system,
  BlinkMacSystemFont, "Segoe UI", sans-serif;
```

- **User Web**: 제목 `font-semibold`~`font-bold`, 본문 `text-base` (16px), line-height 1.6
- **Admin**: 본문 `text-sm` (14px), 테이블·폼은 `text-sm` 고정, line-height 1.5

### Scale

| Level | User Web | Admin | Use |
|-------|----------|-------|-----|
| Display | 36px / 2.25rem | — | 랜딩 Hero H1 |
| H1 | 30px | 24px | 페이지 제목 |
| H2 | 24px | 20px | 섹션 제목 |
| H3 | 20px | 16px | 카드 제목 |
| Body | 16px | 14px | 본문 |
| Caption | 14px | 12px | 라벨, 힌트 |

- 금액·수치·ID: `font-variant-numeric: tabular-nums`
- 코드·API 경로: `font-mono` (Admin 한정)

---

## 4. Component Stylings

### Buttons

| Variant | Style |
|---------|-------|
| **Primary** | `bg-[#0D9488]` text-white, `rounded-lg` (8px), px-4 py-2, hover `#0F766E` |
| **Secondary** | white bg, `border border-[#E2E8F0]`, text `#0F172A` |
| **Ghost** | transparent, text `#0D9488`, hover `bg-[#CCFBF1]` |
| **Danger** | `bg-[#DC2626]` text-white |

- Primary는 화면당 **하나**. Admin 폼 하단 우측 정렬.
- 라벨: 동사형 한국어 (`저장하기`, `로그인`, `문의하기`). `Submit`, `Get Started` 금지.

### Inputs

- Height 40px (Admin), 44px (User Web 터치 친화)
- Border `#E2E8F0`, focus ring 2px `#0D9488` at 40% opacity
- Label: `text-sm font-medium text-[#0F172A]`, input 위 6px 간격
- Error: border `#DC2626`, helper text 12px `#DC2626`

### Cards

- `bg-white rounded-xl` (12px), border `1px solid #E2E8F0`
- Shadow: `0 1px 3px rgba(15, 23, 42, 0.06)` — 단일 레이어, 검정 기반
- Padding: 20px (User), 16px (Admin)

### Navigation

**User Web**: 상단 고정 navbar, 높이 64px, white bg, bottom border. 로고 좌측, 메뉴 우측.  
**Admin**: 좌측 사이드바 240px, `bg-[#1E3A5F]`, 메뉴 텍스트 white/80%, active `bg-white/10` + Teal 좌측 3px 바.

### Tables (Admin)

- Header: `bg-[#F1F5F9] text-xs uppercase tracking-wide text-[#64748B]`
- Row hover: `bg-[#F8FAFC]`
- Cell padding: 12px 16px

---

## 5. Layout Principles

### User Web (Next.js)

- Max width `max-w-6xl` (1152px), 좌우 `px-4 md:px-6`
- 섹션 간 vertical rhythm: 64px (mobile 48px)
- Hero: 좌측 텍스트 + 우측 이미지/일러스트 (이미지 없으면 텍스트만 — placeholder 박스 금지)

### Admin (React + Vite)

- Full viewport: sidebar + `main` flex
- Content area max-width 없음 — 폼·테이블은 가로 전체 활용
- Page header: 제목 + breadcrumb + primary action 한 줄

### Spacing Scale (Tailwind)

4, 6, 8, 12, 16, 24, 32, 48, 64 — 4의 배수만 사용.

---

## 6. Depth & Elevation

| Level | Shadow | Use |
|-------|--------|-----|
| **Flat** | none | Admin 테이블, 인라인 폼 |
| **Raised** | `0 1px 3px rgba(15,23,42,0.06)` | 카드, dropdown |
| **Overlay** | `0 8px 24px rgba(15,23,42,0.12)` | 모달, 사이드 패널 |

- Colored shadow 금지. Multi-layer stack 금지.
- Admin은 대부분 Flat — 밀도 우선.

---

## 7. Do's and Don'ts

### Do

- 메뉴·네비게이션 데이터는 API/상수 객체로 분리 (하드코딩 배열 지양 — `docs/architecture/WEB_CORE.md` §10)
- 한국어 UI copy 기본, 영문은 브랜드명·기술 용어만
- Empty/Loading/Error는 §14 테이블 준수
- `prefers-reduced-motion` 존중

### Don't

- Inter/Roboto를 "안전한 기본값"으로 선택하지 말 것 — Pretendard 사용
- 그라데이션 Hero, 보라 Primary, 장식 이모지
- Admin에 마케팅용 대형 배너·일러스트
- `rounded-xl`만 쓰고 px 값 없이 추상적으로 기술
- 동일 화면 Primary 버튼 2개 이상

---

## 8. Responsive Behavior

| Breakpoint | User Web | Admin |
|------------|----------|-------|
| `< md` (768px) | 햄버거 메뉴, 단일 컬럼, Hero 스택 | 사이드바 접기 → 아이콘 레일 64px |
| `md+` | 2컬럼 Hero 가능 | 사이드바 240px 고정 |
| `lg+` | max-w-6xl 중앙 정렬 | 동일 |

- 테이블: mobile에서 카드 리스트로 변환 또는 horizontal scroll (카드 리스트 우선)
- Touch target 최소 44×44px (User Web)

---

## 9. Agent Prompt Guide

에이전트가 코드를 생성할 때:

1. **스택 매핑**
   - `frontend-user`: Next.js App Router + Tailwind. Server Component 기본, 인터랙션만 `"use client"`.
   - `frontend-admin`: React + Vite + TypeScript + Tailwind. 모든 props에 interface 정의.

2. **Tailwind 토큰 예시** (`tailwind.config` extend 권장)

```js
colors: {
  logpick: {
    navy: "#1E3A5F",
    teal: "#0D9488",
    "teal-hover": "#0F766E",
    "teal-soft": "#CCFBF1",
  },
},
borderRadius: { lg: "8px", xl: "12px" },
```

3. **파일 배치**
   - 공통 UI primitives: 각 frontend의 `components/ui/`
   - 페이지별 레이아웃: `components/layout/`
   - 디자인 토큰 상수: `lib/design-tokens.ts` (Admin), `lib/design-tokens.ts` (User)

4. **검증**: User → `npm run build`, Admin → `npm run lint` + `tsc --noEmit` + `npm run build`

5. **본 문서에 없는 색·폰트·motion 값은 사용 금지** — 모호하면 §12 Principles로 판단.

---

## 10. Voice & Tone

**Voice** — LogPick Core는 **차분한 B2B 파트너**처럼 말합니다.  
과장 없이, 기술 용어는 필요할 때만, 사용자를 존중하는 평서문. 이모지 없음.

| Context | Tone | 예시 |
|---------|------|------|
| CTA | 짧은 동사, 2~4음절 | `문의하기`, `저장하기`, `로그인` |
| Admin 라벨 | 명사형, 간결 | `회사 소개`, `연혁 관리`, `로고 이미지` |
| Error | 구체적, 비난 없음, 다음 행동 제시 | `저장에 실패했습니다. 네트워크 연결을 확인한 뒤 다시 시도해 주세요.` |
| Success | 과거형 한 문장 | `변경 사항이 저장되었습니다.` |
| Empty | 이유 + 권장 행동 | `등록된 연혁이 없습니다. 첫 항목을 추가해 보세요.` |
| User 소개 copy | 신뢰·명확, 2인칭 최소 | `모듈형 커머스의 핵심 인프라를 제공합니다.` |

**Forbidden:** `Oops`, `Please note that`, `Get Started`, `Submit`, `Click here`, `안녕하세요!`, 불필요한 `~해요` 남발.

---

## 11. Brand Narrative

LogPick Core는 **여러 이커머스·ERP·AI 플러그인이 꽂히는 중심 엔진**입니다.  
1단계 MVP는 회사 소개(User)와 콘텐츠 관리(Admin)이지만, 디자인은 "임시 랜딩"이 아니라 **장기 SaaS 플랫폼의 첫 블록**으로 보여야 합니다.

거부하는 것: 쇼핑몰 템플릿의 화려함, 레거시 ERP의 회색 밀도 과다, AI 슬롭 UI(그라데이션·보라 CTA).  
지향하는 것: Linear급 Admin 효율 + Stripe급 B2B 신뢰 + 한국어 서비스의 명료함.

Navy는 "운영의 무게", Teal은 "연결과 확장"을 상징합니다.

---

## 12. Principles

1. **한 화면, 한 주요 행동.** Primary CTA는 하나. Admin 저장은 페이지 하단 고정.
2. **User는 넓게, Admin은 촘촘히.** 같은 컴포넌트라도 surface에 따라 spacing·font-size를 조정.
3. **데이터는 API-ready.** 메뉴·연혁·소개 텍스트는 하드코딩이 아닌 fetch 가능 구조.
4. **Teal은 상호작용에만.** 장식 배경·구분선에 Teal 사용 금지.
5. **신뢰는 절제에서.** 그림자·애니메이션·카피 모두 최소한으로 — 금융급은 아니어도 B2B급 신뢰.
6. **한글·영문 동등.** Pretendard로 혼용 렌더링 전제. 영문만 있는 버튼 라벨 금지.
7. **플러그인 확장을 시각적으로 암시.** "모듈", "코어", "연동" 어휘는 과하지 않게 — UI 구조(사이드바 섹션, 카드 그리드)로 표현.

---

## 13. Personas

**민수 (32, 스타트업 대표)**  
LogPick Core로 회사 소개 사이트를 빠르게 띄우려 함. 모바일에서도 로고·소개글이 깨지지 않기를 기대.

**지은 (28, 운영 매니저)**  
Admin에서 회사 소개·연혁·로고를 직접 수정. "저장" 버튼 위치가 매번 같아야 하고, 실수로 덮어쓰기 전 확인을 원함.

**현우 (35, 개발 파트너)**  
향후 쇼핑몰 플러그인을 붙일 예정. Admin/User 컴포넌트가 Tailwind 토큰으로 일관되길 원해 커스텀 CSS 난립을 싫어함.

---

## 14. States

| State | Treatment |
|-------|-----------|
| **Empty** | `Text Secondary` 한 줄 설명 + Secondary 버튼(권장 행동). 일러스트 없음. |
| **Loading** | 레이아웃과 동일한 skeleton, `#F1F5F9` 블록, 1.2s shimmer. 전체 화면 스피너는 초기 진입 1회만. |
| **Error (inline)** | `Signal Red` border + 필드 아래 12px 메시지 |
| **Error (page)** | Admin: 상단 배너 `bg-red-50 border-red-200 text-red-800` |
| **Success** | Admin: 토스트 3초 — `bg-[#1E3A5F]` text-white, 아이콘 없음 |
| **Unsaved** | Admin 헤더에 `Signal Amber` 점 + `저장되지 않은 변경` 캡션 |
| **Skeleton** | 금융 수치 해당 없음 — 일반 텍스트·이미지 영역에만 적용 |

---

## 15. Motion & Easing

### Durations

| Token | Value | Use |
|-------|-------|-----|
| `motion-instant` | 0ms | 토글, checkbox |
| `motion-fast` | 150ms | hover, focus |
| `motion-standard` | 250ms | dropdown, sidebar toggle |
| `motion-slow` | 400ms | 모달 enter |

### Easings

| Token | Curve |
|-------|-------|
| `ease-enter` | `cubic-bezier(0, 0, 0.2, 1)` |
| `ease-exit` | `cubic-bezier(0.4, 0, 1, 1)` |
| `ease-standard` | `cubic-bezier(0.4, 0, 0.2, 1)` |

### Signature

1. **Sidebar (Admin):** width transition `motion-standard / ease-standard`
2. **Toast:** slide from top 8px + fade, `motion-fast / ease-enter`, dismiss `ease-exit`
3. **`prefers-reduced-motion: reduce`:** 모든 duration → `motion-instant`

---

## Appendix — oh-my-design 워크플로우 연동

위키독스 교재 워크플로우대로 스킬·레퍼런스를 설치하려면:

```bash
npx oh-my-design-cli install-skills --agent cursor
```

설치 후 에이전트에게 자연어로 요청:

> "Linear + B2B SaaS 톤으로 LogPick Core DESIGN.md를 검토하고 토큰을 맞춰줘."

221개 레퍼런스 중 변경을 원하면 [oh-my-design Builder](https://oh-my-design.kr/)에서 브랜드를 고른 뒤 본 파일과 병합하십시오.

---

*Spec: OmD v0.1 · [oh-my-design](https://github.com/kwakseongjae/oh-my-design) (MIT)*
