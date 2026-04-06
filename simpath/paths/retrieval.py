"""Step A: Deterministic retrieval of candidate question pool (Section 4.1).

Improved: larger pool with difficulty balancing to prevent dropout-heavy paths.
"""

from typing import List, Dict, Optional
import random as _random


def retrieve_candidate_pool(
    mastery: Dict[str, float],
    question_bank: List[dict],
    skill_to_questions: Dict[str, List[dict]],
    last_practice_days: Dict[str, float] = None,
    L: int = 8,
    mastery_threshold: float = 0.6,
    zpd_low_offset: float = 0.15,
    zpd_high_offset: float = 0.35,
    per_kc_cap: int = 15,
    target_pool_size: int = 60,
) -> List[dict]:
    """
    Pure rule-based filtering. No LLM involved.
    Returns a filtered question pool (target 40-80 questions).

    Key improvement: includes confidence-building easy questions alongside
    ZPD questions to balance mastery gain with survival rate.
    """
    pool = {}

    if last_practice_days is None:
        last_practice_days = {}

    # Sort KCs by mastery (weakest first)
    sorted_kcs = sorted(mastery.items(), key=lambda x: x[1])

    # 1. Weak KC questions (mastery < threshold) — ZPD filtered
    weak_kcs = [(kc, m) for kc, m in sorted_kcs if m < mastery_threshold]
    for kc, m in weak_kcs:
        zpd_low = max(0.0, m - zpd_low_offset)
        zpd_high = min(1.0, m + zpd_high_offset)

        candidates = [
            q for q in skill_to_questions.get(kc, [])
            if zpd_low <= q["difficulty"] <= zpd_high
        ]
        # Sort by proximity to sweet spot (slightly above mastery)
        sweet_spot = min(m + 0.1, 1.0)
        candidates.sort(key=lambda q: abs(q["difficulty"] - sweet_spot))
        for q in candidates[:per_kc_cap]:
            pool[q["question_id"]] = q

    # 2. Confidence builders — easy questions from weak KCs (difficulty < mastery)
    # These reduce dropout risk and build momentum
    for kc, m in weak_kcs[:10]:  # top 10 weakest
        easy = [
            q for q in skill_to_questions.get(kc, [])
            if q["difficulty"] < m and q["question_id"] not in pool
        ]
        easy.sort(key=lambda q: q["difficulty"])
        for q in easy[:3]:
            pool[q["question_id"]] = q

    # 3. Moderate KC questions (0.4 <= mastery < 0.7) — review/reinforcement
    moderate_kcs = [(kc, m) for kc, m in sorted_kcs
                    if 0.4 <= m < 0.7 and kc not in dict(weak_kcs)]
    for kc, m in moderate_kcs[:5]:
        review = [
            q for q in skill_to_questions.get(kc, [])
            if q["difficulty"] <= m + 0.1 and q["question_id"] not in pool
        ]
        for q in review[:3]:
            pool[q["question_id"]] = q

    result = list(pool.values())

    # 4. If pool still too small, expand
    if len(result) < max(L * 3, 24):
        result = _expand_pool(mastery, skill_to_questions, pool,
                              target_size=max(target_pool_size, L * 5))

    return result


def _expand_pool(mastery, skill_to_questions, existing_pool, target_size):
    """Expand pool with wider difficulty range."""
    pool = dict(existing_pool)
    for kc, m in sorted(mastery.items(), key=lambda x: x[1]):
        zpd_low = max(0.0, m - 0.3)
        zpd_high = min(1.0, m + 0.5)
        for q in skill_to_questions.get(kc, []):
            if q["question_id"] not in pool and zpd_low <= q["difficulty"] <= zpd_high:
                pool[q["question_id"]] = q
            if len(pool) >= target_size:
                return list(pool.values())
    return list(pool.values())
