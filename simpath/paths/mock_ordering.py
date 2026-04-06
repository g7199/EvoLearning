"""Mock LLM ordering: deterministic strategies that mirror the 5 strategy hints (Section 4.3)."""

from typing import List, Dict
import random


STRATEGIES = {
    1: "weakest_first",
    2: "confidence_build",
    3: "interleave",
    4: "intensive",
    5: "breadth_first",
}


def mock_generate_paths(
    mastery: Dict[str, float],
    pool: List[dict],
    K: int = 5,
    L: int = 8,
) -> List[List[dict]]:
    """Generate K candidate paths from pool using deterministic strategies."""
    paths = []
    strategies = [weakest_first, confidence_build, interleave, intensive, breadth_first]

    for i in range(K):
        strategy_fn = strategies[i % len(strategies)]
        path = strategy_fn(mastery, pool, L)
        paths.append(path)

    return paths


def weakest_first(mastery: Dict[str, float], pool: List[dict], L: int) -> List[dict]:
    """Strategy 1: Prioritize weakest KCs, gradually increasing difficulty."""
    def sort_key(q):
        avg_m = _avg_mastery(mastery, q["kc_ids"])
        return (avg_m, q["difficulty"])
    sorted_pool = sorted(pool, key=sort_key)
    return _deduplicate(sorted_pool)[:L]


def confidence_build(mastery: Dict[str, float], pool: List[dict], L: int) -> List[dict]:
    """Strategy 2: Start with easy confidence builder, then tackle weak areas."""
    sorted_by_diff = sorted(pool, key=lambda q: q["difficulty"])
    easy = sorted_by_diff[:1]
    rest = sorted(pool[1:], key=lambda q: _avg_mastery(mastery, q["kc_ids"]))
    combined = easy + rest
    return _deduplicate(combined)[:L]


def interleave(mastery: Dict[str, float], pool: List[dict], L: int) -> List[dict]:
    """Strategy 3: Interleave weak KCs with stale/review KCs."""
    weak = [q for q in pool if _avg_mastery(mastery, q["kc_ids"]) < 0.4]
    other = [q for q in pool if q not in weak]
    weak.sort(key=lambda q: _avg_mastery(mastery, q["kc_ids"]))
    other.sort(key=lambda q: q["difficulty"])

    result = []
    wi, oi = 0, 0
    for i in range(L):
        if i % 2 == 0 and wi < len(weak):
            result.append(weak[wi]); wi += 1
        elif oi < len(other):
            result.append(other[oi]); oi += 1
        elif wi < len(weak):
            result.append(weak[wi]); wi += 1

    return _deduplicate(result)[:L]


def intensive(mastery: Dict[str, float], pool: List[dict], L: int) -> List[dict]:
    """Strategy 4: Focus on single weakest KC with intensive practice."""
    if not pool:
        return []
    # Find weakest KC
    kc_mastery = {}
    for q in pool:
        for kc in q["kc_ids"]:
            if kc not in kc_mastery:
                kc_mastery[kc] = mastery.get(kc, 0.5)
    if not kc_mastery:
        return pool[:L]

    weakest_kc = min(kc_mastery, key=kc_mastery.get)
    focused = [q for q in pool if weakest_kc in q["kc_ids"]]
    focused.sort(key=lambda q: q["difficulty"])
    others = [q for q in pool if q not in focused]
    others.sort(key=lambda q: _avg_mastery(mastery, q["kc_ids"]))

    # 4 from weakest KC, rest from others
    result = focused[:4] + others[:(L - min(4, len(focused)))]
    return _deduplicate(result)[:L]


def breadth_first(mastery: Dict[str, float], pool: List[dict], L: int) -> List[dict]:
    """Strategy 5: Spread across as many different weak KCs as possible."""
    kc_queues = {}
    for q in pool:
        for kc in q["kc_ids"]:
            if mastery.get(kc, 0.5) < 0.6:
                kc_queues.setdefault(kc, []).append(q)

    for kc in kc_queues:
        kc_queues[kc].sort(key=lambda q: q["difficulty"])

    result = []
    seen = set()
    kc_order = sorted(kc_queues.keys(), key=lambda kc: mastery.get(kc, 0.5))

    while len(result) < L and kc_order:
        for kc in list(kc_order):
            if len(result) >= L:
                break
            queue = kc_queues[kc]
            while queue and queue[0]["question_id"] in seen:
                queue.pop(0)
            if queue:
                q = queue.pop(0)
                seen.add(q["question_id"])
                result.append(q)
            else:
                kc_order.remove(kc)

    return result[:L]


def _avg_mastery(mastery: Dict[str, float], kc_ids: List[str]) -> float:
    if not kc_ids:
        return 0.5
    return sum(mastery.get(kc, 0.5) for kc in kc_ids) / len(kc_ids)


def _deduplicate(questions: List[dict]) -> List[dict]:
    seen = set()
    result = []
    for q in questions:
        if q["question_id"] not in seen:
            seen.add(q["question_id"])
            result.append(q)
    return result
