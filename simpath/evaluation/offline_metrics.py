"""
Offline evaluation metrics following standard LPR protocol.
EP = (Ee - Es) / (Esup - Es) — normalized learning gain (SRC, CSEAL, IB-GRPO).
"""

import numpy as np
from typing import Dict, List, Tuple
from simpath.kt.simulate import simulate_kt, SimulationResult
from simpath.personas.definitions import PersonaParams, THETA_GRIT, THETA_FRAGILE
from simpath.utils.seeds import simulation_seed, set_global_seed


def compute_EP(m_before: Dict[str, float], m_after: Dict[str, float],
               target_kcs: List[str]) -> float:
    """
    Standard normalized learning gain (SRC/CSEAL/IB-GRPO).
    EP = mean((Ee_c - Es_c) / (1.0 - Es_c)) for c in target_kcs
    where Es_c = m_before[c], Ee_c = m_after[c], Esup = 1.0
    """
    if not target_kcs:
        return 0.0
    eps = []
    for c in target_kcs:
        es = m_before.get(c, 0.5)
        ee = m_after.get(c, es)
        denom = 1.0 - es
        if denom < 1e-6:
            # Already mastered — no room to improve
            eps.append(0.0)
        else:
            eps.append((ee - es) / denom)
    return float(np.mean(eps))


def evaluate_path(
    mastery: Dict[str, float],
    path: List[dict],
    theta_real: PersonaParams,
    target_kcs: List[str],
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    N_sim: int = 10,
    student_id: str = "default",
    path_idx: int = 0,
    predict_fn=None,
) -> Dict[str, float]:
    """
    Evaluate a single recommended path across all personas.
    Uses FIXED target_kcs (same for all methods) — no path-specific KC filtering.
    Returns: {EP, SR, CE, RI}
    """
    personas = [THETA_GRIT, THETA_FRAGILE, theta_real]
    persona_ep, persona_sr, persona_ce = [], [], []
    w1, w2, w3 = weights

    for j, persona in enumerate(personas):
        ep_runs, sr_runs, ce_runs = [], [], []
        for sim in range(N_sim):
            seed = simulation_seed(student_id, path_idx, j, sim)
            set_global_seed(seed)
            result = simulate_kt(mastery, path, persona, predict_fn)

            ep_runs.append(compute_EP(mastery, result.final_mastery, target_kcs))
            sr_runs.append(1.0 - float(result.dropout))
            ce_runs.append(result.steps_completed / max(result.steps_total, 1))

        persona_ep.append(np.mean(ep_runs))
        persona_sr.append(np.mean(sr_runs))
        persona_ce.append(np.mean(ce_runs))

    # Composite scores per persona
    phi = [w1 * ep + w2 * sr + w3 * ce
           for ep, sr, ce in zip(persona_ep, persona_sr, persona_ce)]

    # Robustness Index
    phi_max = max(max(phi), 1e-8)
    phi_std = float(np.std(phi))
    ri = 1.0 - (phi_std / phi_max)

    return {
        "EP": float(np.mean(persona_ep)),
        "SR": float(np.mean(persona_sr)),
        "CE": float(np.mean(persona_ce)),
        "RI": float(ri),
    }


def evaluate_method(
    test_students: List[dict],
    recommend_fn,
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    N_sim: int = 10,
    predict_fn=None,
) -> Dict[str, Tuple[float, float]]:
    """Evaluate a recommendation method across all test students."""
    all_results = {"EP": [], "SR": [], "CE": [], "RI": []}

    for student in test_students:
        path = recommend_fn(student)
        metrics = evaluate_path(
            mastery=student["mastery"],
            path=path,
            theta_real=student["theta_real"],
            target_kcs=student["target_kcs"],
            weights=weights,
            N_sim=N_sim,
            student_id=student["student_id"],
            predict_fn=predict_fn,
        )
        for k, v in metrics.items():
            all_results[k].append(v)

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in all_results.items()}
