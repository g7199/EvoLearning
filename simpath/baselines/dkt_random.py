"""B1: DKT + Random baseline."""

import random
from typing import List, Dict, Optional


def recommend_random(
    mastery: Dict[str, float],
    skill_to_questions: Dict[str, List[dict]],
    L: int = 8,
    mastery_threshold: float = 0.6,
    pool: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Select L random questions from weak KCs.
    If pool is provided, select from that pool (same as SimPath for fair comparison).
    """
    if pool is not None:
        if len(pool) <= L:
            return list(pool)
        return random.sample(pool, L)

    # Fallback: select from skill_to_questions
    weak_kcs = [kc for kc, m in mastery.items() if m < mastery_threshold]
    if not weak_kcs:
        weak_kcs = list(mastery.keys())

    candidates = []
    seen = set()
    for kc in weak_kcs:
        for q in skill_to_questions.get(kc, []):
            if q["question_id"] not in seen:
                seen.add(q["question_id"])
                candidates.append(q)

    if len(candidates) <= L:
        return candidates
    return random.sample(candidates, L)
