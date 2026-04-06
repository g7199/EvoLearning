---
name: no-mock-experiments
description: Never run mock experiments as if they are real results. Always use actual LLM for paper experiments.
type: feedback
---

mock ordering으로 실험한 결과를 논문 결과처럼 보고하지 말 것. 사용자가 매우 화남.

**Why:** mock은 개발/디버깅용이지 논문 결과가 아님. LLM ordering이 SimPath의 핵심 컴포넌트인데 mock으로 대체하면 의미 없음. 사용자가 API 키를 줬는데도 mock으로 실험한 것은 지시 무시.

**How to apply:**
- 논문 실험은 반드시 실제 LLM (Anthropic/OpenAI) 사용
- mock은 코드 디버깅/파이프라인 검증에만 한정
- mock 결과를 논문 테이블에 절대 넣지 말 것
