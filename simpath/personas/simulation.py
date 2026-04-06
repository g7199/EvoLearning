"""Full persona simulation matrix (Section 5.4)."""

import numpy as np
from typing import List, Dict

from simpath.personas.definitions import PersonaParams, THETA_GRIT, THETA_FRAGILE
from simpath.kt.simulate import simulate_kt, SimulationResult
from simpath.evaluation.offline_metrics import compute_EP
from simpath.utils.seeds import simulation_seed, set_global_seed


def simulate_all_personas(
    mastery: Dict[str, float],
    candidate_paths: List[List[dict]],
    theta_real: PersonaParams,
    student_id: str = "default",
    N_sim: int = 10,
    target_kcs: List[str] = None,
    predict_fn=None,
) -> np.ndarray:
    """
    Run simulation matrix.
    Uses FIXED target_kcs for ALL paths (no path-specific KC filtering).

    Returns:
        S: np.ndarray of shape [K, 3, 3]
           Axes: [path_index, persona_index, metric_index]
           Metrics: [EP, survival_rate, completion_rate]
    """
    personas = [THETA_GRIT, THETA_FRAGILE, theta_real]
    K = len(candidate_paths)
    S = np.zeros((K, 3, 3))

    if target_kcs is None:
        target_kcs = [kc for kc, m in mastery.items() if m < 0.6]
        if not target_kcs:
            target_kcs = sorted(mastery.keys(), key=lambda k: mastery[k])[:8]

    for i in range(K):
        for j, persona in enumerate(personas):
            results: List[SimulationResult] = []
            for sim in range(N_sim):
                seed = simulation_seed(student_id, i, j, sim)
                set_global_seed(seed)

                result = simulate_kt(
                    mastery=mastery,
                    path=candidate_paths[i],
                    persona=persona,
                    predict_fn=predict_fn,
                )
                results.append(result)

            # Use FIXED target_kcs for all paths (fair comparison)
            S[i, j, 0] = np.mean([compute_EP(mastery, r.final_mastery, target_kcs) for r in results])
            S[i, j, 1] = 1.0 - np.mean([float(r.dropout) for r in results])
            S[i, j, 2] = np.mean([r.steps_completed / max(r.steps_total, 1) for r in results])

    return S
