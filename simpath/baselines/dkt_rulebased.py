"""B2: DKT + Rule-based baseline."""

from typing import List, Dict, Optional


def recommend_rulebased(
    mastery: Dict[str, float],
    skill_to_questions: Dict[str, List[dict]],
    L: int = 8,
    mastery_threshold: float = 0.6,
    diff_range: float = 0.2,
    pool: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Sort KCs by mastery ascending. Select questions with difficulty in
    [mastery, mastery+0.2]. Enforce interleaving.
    If pool is provided, select from that pool (same as SimPath for fair comparison).
    """
    source = pool if pool is not None else _gather_from_s2q(mastery, skill_to_questions, mastery_threshold)

    # Sort by: weakest KC first, then difficulty ascending
    def sort_key(q):
        avg_m = sum(mastery.get(kc, 0.5) for kc in q["kc_ids"]) / max(len(q["kc_ids"]), 1)
        return (avg_m, q["difficulty"])

    sorted_qs = sorted(source, key=sort_key)

    path = []
    seen = set()
    last_kc = None

    for q in sorted_qs:
        if len(path) >= L:
            break
        if q["question_id"] in seen:
            continue

        # Interleaving check
        primary_kc = q["kc_ids"][0] if q["kc_ids"] else None
        if primary_kc == last_kc and len(path) >= 2:
            prev_kc = path[-1]["kc_ids"][0] if path[-1]["kc_ids"] else None
            if prev_kc == primary_kc:
                continue

        path.append(q)
        seen.add(q["question_id"])
        last_kc = primary_kc

    # Fill if short
    if len(path) < L:
        for q in sorted_qs:
            if q["question_id"] not in seen:
                path.append(q)
                seen.add(q["question_id"])
                if len(path) >= L:
                    break

    return path[:L]


def _gather_from_s2q(mastery, skill_to_questions, threshold):
    candidates = []
    seen = set()
    weak_kcs = sorted(
        [(kc, m) for kc, m in mastery.items() if m < threshold],
        key=lambda x: x[1],
    )
    if not weak_kcs:
        weak_kcs = sorted(mastery.items(), key=lambda x: x[1])

    for kc, m in weak_kcs:
        for q in skill_to_questions.get(kc, []):
            if q["question_id"] not in seen:
                seen.add(q["question_id"])
                candidates.append(q)
    return candidates
