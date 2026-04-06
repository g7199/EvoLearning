"""
B4: LPReKL-style baseline — KT diagnosis + LLM generate + retrieve.
Reference: LPReKL (Electronics 2025)

Pipeline:
1. KT diagnoses weak KCs
2. LLM generates "ideal exercise description" for weak areas
3. Retrieve most similar exercises from pool by matching
4. Score by knowledge promotion score (KPS)
"""

import json
import re
from typing import List, Dict, Optional


def recommend_lprekl(
    mastery: Dict[str, float],
    pool: List[dict],
    kc_list: List[str],
    L: int = 8,
    llm_client=None,
) -> List[dict]:
    """
    LPReKL-style: KT diagnosis → LLM reference generation → retrieve & rank.
    If llm_client is provided, uses LLM for reference exercise generation.
    Otherwise falls back to KPS-only heuristic.
    """
    # Step 1: KT diagnosis — identify weak KCs
    weak_kcs = sorted(mastery.items(), key=lambda x: x[1])
    weak_kcs = [(kc, m) for kc, m in weak_kcs if m < 0.6][:10]
    if not weak_kcs:
        weak_kcs = sorted(mastery.items(), key=lambda x: x[1])[:5]

    # Step 2: LLM generates reference exercise descriptions
    llm_ranked_ids = []
    if llm_client is not None:
        llm_ranked_ids = _llm_generate_and_retrieve(mastery, pool, weak_kcs, llm_client, L)

    # Step 3: Score all pool questions by KPS
    scored = _score_by_kps(mastery, pool, weak_kcs)

    # Step 4: Merge LLM recommendations with KPS ranking
    path = _merge_rankings(llm_ranked_ids, scored, pool, L)

    return path


def _llm_generate_and_retrieve(mastery, pool, weak_kcs, llm_client, L):
    """Use LLM to generate ideal exercise selection."""
    weak_desc = "\n".join([f"  {kc}: mastery={m:.2f}" for kc, m in weak_kcs])

    pool_desc = "\n".join([
        f"{q['question_id']} kc={','.join(q['kc_ids'])} d={q['difficulty']:.2f}"
        for q in pool[:40]
    ])

    prompt = f"""You are a KT-based learning path advisor. A student has these weak knowledge areas:
{weak_desc}

Select {L} exercises from this pool that would best improve the student's weak areas.
Prioritize exercises that:
1. Target the weakest KCs
2. Have difficulty slightly above current mastery (ZPD)
3. Build progressively from easier to harder

Pool:
{pool_desc}

Return ONLY a JSON array of question IDs:
["q123", "q456", ...]"""

    try:
        response = llm_client.generate(prompt)
        # Parse response
        ids = _parse_ids(response)
        return ids
    except Exception as e:
        return []


def _parse_ids(text):
    """Extract question IDs from LLM response."""
    try:
        arr = json.loads(text.strip())
        if isinstance(arr, list):
            return [str(x) for x in arr]
    except json.JSONDecodeError:
        pass

    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except json.JSONDecodeError:
            pass

    return re.findall(r'q_?\d+', text)


def _score_by_kps(mastery, pool, weak_kcs):
    """Knowledge Promotion Score — how much a question helps weak areas."""
    weak_set = {kc for kc, _ in weak_kcs}
    scored = []
    for q in pool:
        kps = 0.0
        for kc in q["kc_ids"]:
            if kc in weak_set:
                m = mastery.get(kc, 0.5)
                gap = 1.0 - m
                diff = q["difficulty"]
                # ZPD peak: best when difficulty is slightly above mastery
                zpd = max(0, 1.0 - abs(diff - m - 0.1) * 3)
                kps += gap * zpd
        scored.append((kps, q))
    scored.sort(key=lambda x: -x[0])
    return scored


def _merge_rankings(llm_ids, scored, pool, L):
    """Merge LLM selection with KPS ranking. LLM gets priority for top slots."""
    pool_map = {q["question_id"]: q for q in pool}
    path = []
    seen = set()

    # LLM selections first (if available)
    for qid in llm_ids:
        if len(path) >= L:
            break
        if qid in pool_map and qid not in seen:
            path.append(pool_map[qid])
            seen.add(qid)

    # Fill remaining with KPS ranking
    for _, q in scored:
        if len(path) >= L:
            break
        if q["question_id"] not in seen:
            path.append(q)
            seen.add(q["question_id"])

    return path[:L]
