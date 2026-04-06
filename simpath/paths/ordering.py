"""Step B: LLM-based path generation — Hybrid Scoring + Heuristic Construction."""

import json
import re
import random as _random
from typing import List, Dict, Tuple
from simpath.llm.client import LLMClient
from simpath.paths.prompts import SCORING_PROMPT, ORDERING_PROMPT, STRATEGY_HINTS


def generate_candidate_paths_llm(
    mastery: Dict[str, float],
    pool: List[dict],
    llm_client: LLMClient,
    K: int = 5,
    L: int = 8,
    seed_base: int = 42,
) -> Tuple[List[List[dict]], dict]:
    """
    Generate K candidate paths using LLM scoring + heuristic construction.

    Phase 1: Pre-filter pool to top candidates (KPS + easy question guarantee)
    Phase 2: LLM scores each candidate question (pointwise, not listwise)
    Phase 3: Construct K diverse paths from scores using heuristic rules
    """
    pool_map = {q["question_id"]: q for q in pool}
    stats = {"total_ids_requested": 0, "valid_ids": 0, "fills": 0,
             "constraint_violations": 0, "llm_calls": 0}

    # Phase 1: Pre-filter to ~20 best candidates with easy question guarantee
    candidates = _prefilter_pool(mastery, pool, max_items=20)

    # Phase 2: LLM scores each candidate
    scores = _llm_score_questions(mastery, candidates, llm_client)
    stats["llm_calls"] = 1

    # Phase 3: Construct K diverse paths from scores
    paths = _construct_diverse_paths(mastery, candidates, scores, K, L)

    # Compute stats
    for path in paths:
        stats["valid_ids"] += len(path)
        stats["total_ids_requested"] += L
        if _has_constraint_violation(path):
            stats["constraint_violations"] += 1
        if len(path) < L:
            stats["fills"] += L - len(path)

    total_req = max(stats["total_ids_requested"], 1)
    stats["pool_hit_rate"] = stats["valid_ids"] / total_req
    stats["fill_rate"] = stats["fills"] / (K * L)
    stats["constraint_violation_rate"] = stats["constraint_violations"] / K

    return paths, stats


def _prefilter_pool(mastery: dict, pool: list, max_items: int = 20) -> list:
    """Pre-filter pool using KPS scoring, ensuring easy questions are included."""
    weak_kcs = {kc for kc, m in mastery.items() if m < 0.6}

    # Score each question by KPS
    scored = []
    for q in pool:
        kps = 0.0
        for kc in q["kc_ids"]:
            if kc in weak_kcs:
                m = mastery.get(kc, 0.5)
                gap = 1.0 - m
                diff = q["difficulty"]
                zpd = max(0, 1.0 - abs(diff - m + 0.05) * 3)
                kps += gap * zpd
        scored.append((kps, q))

    scored.sort(key=lambda x: -x[0])
    top = [q for _, q in scored[:max_items]]
    top_ids = {q["question_id"] for q in top}

    # Guarantee: include easiest question per weak KC (confidence builders)
    kc_to_easiest = {}
    for q in pool:
        for kc in q["kc_ids"]:
            if kc in weak_kcs:
                if kc not in kc_to_easiest or q["difficulty"] < kc_to_easiest[kc]["difficulty"]:
                    kc_to_easiest[kc] = q

    for kc, q in kc_to_easiest.items():
        if q["question_id"] not in top_ids and len(top) < max_items + 5:
            top.append(q)
            top_ids.add(q["question_id"])

    return top


def _llm_score_questions(mastery: dict, candidates: list, llm_client: LLMClient) -> dict:
    """Ask LLM to score each candidate question (pointwise scoring)."""
    mastery_desc = _format_mastery_compact(mastery)
    pool_desc = "\n".join([
        f"{q['question_id']} kc={','.join(q['kc_ids'])} d={q['difficulty']:.2f}"
        for q in candidates
    ])

    prompt = SCORING_PROMPT.format(
        mastery_state=mastery_desc,
        question_pool=pool_desc,
    )

    response = llm_client.generate(prompt)
    scores = _parse_scores(response, candidates)
    return scores


def _parse_scores(text: str, candidates: list) -> dict:
    """Parse LLM scoring response into {question_id: score}."""
    # Try JSON parse
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            return {str(k): float(v) for k, v in obj.items()}
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON object in text
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return {str(k): float(v) for k, v in obj.items()}
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: assign scores based on KPS
    scores = {}
    for q in candidates:
        scores[q["question_id"]] = 5.0  # neutral score
    return scores


def _construct_diverse_paths(mastery: dict, candidates: list, scores: dict,
                             K: int, L: int) -> list:
    """Construct K diverse paths from LLM scores + heuristic rules."""
    weak_kcs = sorted(
        [(kc, m) for kc, m in mastery.items() if m < 0.6],
        key=lambda x: x[1]
    )

    paths = []
    for k in range(K):
        if k == 0:
            # Path 1: LLM top scores, weakest KC first, difficulty ascending
            path = _build_path_scored(candidates, scores, mastery, weak_kcs, L,
                                       focus_kc=None, start_easy=True)
        elif k == 1:
            # Path 2: Focus on 2nd weakest KC
            focus = weak_kcs[1][0] if len(weak_kcs) > 1 else weak_kcs[0][0]
            path = _build_path_scored(candidates, scores, mastery, weak_kcs, L,
                                       focus_kc=focus, start_easy=True)
        elif k == 2:
            # Path 3: Breadth — maximize KC coverage
            path = _build_path_breadth(candidates, scores, mastery, weak_kcs, L)
        elif k == 3:
            # Path 4: Intensive — 4 questions on weakest KC
            path = _build_path_intensive(candidates, scores, mastery, weak_kcs, L)
        else:
            # Path 5: Score-only (pure LLM ranking, no heuristic override)
            path = _build_path_pure_score(candidates, scores, L)

        paths.append(path)

    return paths


def _build_path_scored(candidates, scores, mastery, weak_kcs, L,
                        focus_kc=None, start_easy=True):
    """Build path: LLM scores weighted, start with easy confidence builder."""
    # Sort by: score (descending), then difficulty (ascending for position 1-2)
    pool = list(candidates)
    path = []
    used = set()

    # Step 1: Start with easiest question from weakest (or focus) KC
    target_kc = focus_kc or (weak_kcs[0][0] if weak_kcs else None)
    if target_kc and start_easy:
        kc_qs = [q for q in pool if target_kc in q["kc_ids"]]
        kc_qs.sort(key=lambda q: q["difficulty"])
        if kc_qs:
            path.append(kc_qs[0])
            used.add(kc_qs[0]["question_id"])

    # Step 2: Fill rest by score, with difficulty progression
    remaining = [(scores.get(q["question_id"], 5), q) for q in pool
                 if q["question_id"] not in used]
    remaining.sort(key=lambda x: (-x[0], x[1]["difficulty"]))

    last_kc = path[-1]["kc_ids"][0] if path and path[-1]["kc_ids"] else None
    for score, q in remaining:
        if len(path) >= L:
            break
        primary_kc = q["kc_ids"][0] if q["kc_ids"] else None
        # Interleaving: avoid 3 consecutive same KC
        if primary_kc == last_kc and len(path) >= 2:
            prev_kc = path[-2]["kc_ids"][0] if path[-2]["kc_ids"] else None
            if prev_kc == primary_kc:
                continue
        path.append(q)
        used.add(q["question_id"])
        last_kc = primary_kc

    return path[:L]


def _build_path_breadth(candidates, scores, mastery, weak_kcs, L):
    """Maximize KC coverage: one question per KC, round-robin."""
    kc_queues = {}
    for q in candidates:
        for kc in q["kc_ids"]:
            if mastery.get(kc, 0.5) < 0.6:
                kc_queues.setdefault(kc, []).append(q)

    for kc in kc_queues:
        kc_queues[kc].sort(key=lambda q: (-scores.get(q["question_id"], 5), q["difficulty"]))

    path = []
    used = set()
    kc_order = [kc for kc, _ in weak_kcs if kc in kc_queues]

    while len(path) < L and kc_order:
        for kc in list(kc_order):
            if len(path) >= L:
                break
            queue = kc_queues.get(kc, [])
            while queue and queue[0]["question_id"] in used:
                queue.pop(0)
            if queue:
                path.append(queue.pop(0))
                used.add(path[-1]["question_id"])
            else:
                kc_order.remove(kc)

    return path[:L]


def _build_path_intensive(candidates, scores, mastery, weak_kcs, L):
    """Focus 4 questions on weakest KC, rest on 2nd weakest."""
    path = []
    used = set()

    for focus_idx, count in [(0, 4), (1, L - 4)]:
        if focus_idx >= len(weak_kcs):
            continue
        kc = weak_kcs[focus_idx][0]
        kc_qs = [q for q in candidates if kc in q["kc_ids"] and q["question_id"] not in used]
        kc_qs.sort(key=lambda q: q["difficulty"])  # easiest first
        for q in kc_qs[:count]:
            path.append(q)
            used.add(q["question_id"])

    # Fill remaining
    rest = [(scores.get(q["question_id"], 5), q) for q in candidates if q["question_id"] not in used]
    rest.sort(key=lambda x: -x[0])
    for _, q in rest:
        if len(path) >= L:
            break
        path.append(q)

    return path[:L]


def _build_path_pure_score(candidates, scores, L):
    """Pure LLM score ranking, no heuristic override."""
    scored = [(scores.get(q["question_id"], 5), q) for q in candidates]
    scored.sort(key=lambda x: -x[0])
    return [q for _, q in scored[:L]]


def _format_mastery_compact(mastery: dict) -> str:
    weak = [(kc, m) for kc, m in sorted(mastery.items(), key=lambda x: x[1]) if m < 0.6]
    lines = [f"  {kc}: {m:.2f} (WEAK)" for kc, m in weak[:10]]
    strong_count = sum(1 for m in mastery.values() if m >= 0.6)
    if strong_count:
        lines.append(f"  ... {strong_count} strong KCs (mastery ≥ 0.6) omitted")
    return "\n".join(lines)


def _has_constraint_violation(path: list) -> bool:
    if len(path) < 3:
        return False
    for i in range(len(path) - 2):
        kcs_i = set(path[i].get("kc_ids", []))
        kcs_j = set(path[i+1].get("kc_ids", []))
        kcs_k = set(path[i+2].get("kc_ids", []))
        if kcs_i & kcs_j & kcs_k:
            return True
    return False
