"""Prompt templates for LLM-based path generation."""

# Scoring prompt: LLM scores each question, we construct the path
SCORING_PROMPT = """You are an expert educational path advisor. Score each question's educational value (1-10) for this student.

## Student Knowledge State (lower = weaker, needs more practice)
{mastery_state}

## Key Learning Science Facts
- P(correct) ≈ sigmoid(3 × (mastery - difficulty))
- When student answers correctly: mastery improves by ~15% of remaining gap
- 2+ consecutive wrong answers → high dropout risk
- Therefore: start with high P(correct) questions, then gradually increase difficulty
- Questions slightly BELOW the student's mastery give ~70% correct rate = ideal learning zone

## Questions to Score
{question_pool}

## Scoring Criteria
- 10: Perfect ZPD match for a weak KC, high P(correct), builds confidence
- 7-9: Good ZPD match, targets weak area, reasonable difficulty
- 4-6: Moderate value, somewhat relevant to weak areas
- 1-3: Poor match, too hard or targets already-strong KC

## Output
Return ONLY a JSON object mapping question_id to score:
{{"q123": 9, "q456": 7, ...}}
Score ALL questions listed above."""

# Ordering prompt (kept for comparison/fallback)
ORDERING_PROMPT = """Select exactly {path_length} question IDs from the pool to create an optimal learning path.

Student mastery:
{mastery_state}

Key: P(correct) ≈ sigmoid(3 × (mastery - difficulty)). Start easy (high P(correct)), ramp up gradually. 2+ consecutive wrong → dropout risk.

Question pool (ONLY use these IDs):
{question_pool}

Strategy: {strategy_hint}

{previous_paths}

Return ONLY a JSON array, nothing else:
["q123", "q456", ...]"""

STRATEGY_HINTS = {
    1: "Weakest-first: target lowest mastery KCs, START with easiest questions (difficulty < mastery), then gradually increase.",
    2: "Confidence-build: first 2 questions MUST have difficulty < 0.15, then tackle weak areas.",
    3: "Interleave: alternate between weak and moderate KCs, keep difficulty within ZPD (mastery ± 0.15).",
    4: "Intensive: focus 4 questions on the single weakest KC (easiest first), rest on 2nd weakest.",
    5: "Breadth-first: cover as many different weak KCs as possible, all within ZPD.",
}
