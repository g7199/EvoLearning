"""Robust path selection via minimax regret (Section 6.3-6.4)."""

import numpy as np
from typing import Tuple, Dict, List


def compute_weights(
    mastery: Dict[str, float],
    recent_dropout_rate: float = 0.0,
    w_default: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    dropout_risk_threshold: float = 0.3,
    low_mastery_threshold: float = 0.3,
    weak_kcs: List[str] = None,
) -> Tuple[float, float, float]:
    """Dynamic weight computation (Section 6.2)."""
    w1, w2, w3 = w_default

    if recent_dropout_rate > dropout_risk_threshold:
        w2 += 0.15; w1 -= 0.10; w3 -= 0.05

    if weak_kcs:
        mean_weak = np.mean([mastery.get(kc, 0.5) for kc in weak_kcs])
        if mean_weak < low_mastery_threshold:
            w1 += 0.10; w2 -= 0.05; w3 -= 0.05

    total = w1 + w2 + w3
    return (w1 / total, w2 / total, w3 / total)


def composite_scores(S: np.ndarray, weights: Tuple[float, float, float]) -> np.ndarray:
    """
    Compute composite score matrix phi[K, J].
    S: [K, J, 3] simulation matrix
    """
    w = np.array(weights)
    return np.einsum('ijk,k->ij', S, w)


def minimax_regret_select(phi: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Minimax regret path selection.

    Args:
        phi: [K, J] composite scores

    Returns:
        best_idx: index of selected path
        regret: [K, J] regret matrix
        max_regret: [K] maximum regret per path
    """
    K, J = phi.shape
    phi_star = phi.max(axis=0)  # [J] best achievable per persona
    regret = phi_star[None, :] - phi  # [K, J]
    max_regret = regret.max(axis=1)  # [K]
    best_idx = int(np.argmin(max_regret))
    return best_idx, regret, max_regret


def average_select(phi: np.ndarray) -> int:
    return int(np.argmax(phi.mean(axis=1)))


def maximin_select(phi: np.ndarray) -> int:
    return int(np.argmax(phi.min(axis=1)))


def hurwicz_select(phi: np.ndarray, alpha: float = 0.3) -> int:
    scores = alpha * phi.max(axis=1) + (1 - alpha) * phi.min(axis=1)
    return int(np.argmax(scores))


def weighted_minimax_select(phi: np.ndarray, persona_weights: np.ndarray = None) -> int:
    """Minimax regret with persona weighting (Realistic gets 1.5x)."""
    if persona_weights is None:
        persona_weights = np.array([1.0, 1.0, 1.5])
    phi_star = phi.max(axis=0)
    regret = (phi_star[None, :] - phi) * persona_weights[None, :]
    max_regret = regret.max(axis=1)
    return int(np.argmin(max_regret))
