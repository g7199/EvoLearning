---
name: no-synthetic-data
description: Do not use synthetic/fake data for experiments — always use real datasets specified in proposal
type: feedback
---

합성 데이터를 만들어서 실험하지 말 것. proposal.tex에 명시된 실제 데이터셋(EdNet KT3, ASSISTments 2015)을 사용해야 함.

**Why:** 합성 데이터로 돌린 결과는 논문의 핵심 기여(minimax regret, persona differentiation)를 검증할 수 없었음. 사용자가 명시적으로 "제발 맘대로 만들지 말고 제대로 해달라"고 요청.

**How to apply:** 실험 파이프라인은 항상 proposal에 명시된 데이터셋 기준으로 구성. smoke test용 합성 데이터는 단위 테스트에만 한정. 실험 스크립트의 기본 모드는 실제 데이터셋이어야 함.
