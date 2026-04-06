"""Core simulation function: simulate a student walking through a learning path under a persona."""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from simpath.personas.definitions import PersonaParams


@dataclass
class StepResult:
    question_id: str
    kc_ids: List[str]
    result: str  # "correct", "incorrect", "skip"
    mastery_snapshot: Dict[str, float]


@dataclass
class SimulationResult:
    final_mastery: Dict[str, float]
    trajectory: List[StepResult]
    dropout: bool
    steps_completed: int
    steps_total: int


def simulate_kt(
    mastery: Dict[str, float],
    path: List[dict],  # each: {"question_id": str, "kc_ids": [str], "difficulty": float}
    persona: PersonaParams,
    predict_fn=None,  # (mastery, question) -> p_correct; if None, uses difficulty-based
) -> SimulationResult:
    """
    Simulate a student walking through a path under a given persona.

    Args:
        mastery: current knowledge state {kc_id: mastery_level}
        path: ordered list of questions
        persona: persona parameters
        predict_fn: optional function to predict correctness probability
    """
    m = dict(mastery)  # copy
    trajectory = []
    dropout = False
    consecutive_wrong = 0

    for t, question in enumerate(path):
        q_id = question["question_id"]
        kc_ids = question["kc_ids"]
        diff = question["difficulty"]

        # --- Skip check ---
        if persona.skip_threshold is not None and diff > persona.skip_threshold:
            if random.random() < 0.5:
                trajectory.append(StepResult(
                    question_id=q_id, kc_ids=kc_ids,
                    result="skip", mastery_snapshot=dict(m),
                ))
                consecutive_wrong += 1
                # Dropout check after skip
                if _check_dropout(consecutive_wrong, diff, persona):
                    dropout = True
                    break
                continue

        # --- Predict correctness ---
        if predict_fn is not None:
            p_correct = predict_fn(m, question)
        else:
            # IRT-style: P(correct) = sigmoid(mastery - difficulty)
            # More realistic than linear combination
            avg_mastery = np.mean([m.get(kc, 0.5) for kc in kc_ids]) if kc_ids else 0.5
            logit = 3.0 * (avg_mastery - diff)  # scale factor for sharper curve
            p_correct = 1.0 / (1.0 + np.exp(-logit))

        p_adjusted = max(0.01, min(0.99, p_correct + persona.p_correct_boost))
        correct = random.random() < p_adjusted

        # --- Update mastery ---
        for kc in kc_ids:
            cur = m.get(kc, 0.5)
            if correct:
                m[kc] = cur + persona.learn_rate * (1 - cur)
            else:
                m[kc] = max(0.0, cur - persona.forget_penalty * cur)

        # Track consecutive wrong at question level (not KC level)
        if correct:
            consecutive_wrong = 0
        else:
            consecutive_wrong += 1

        result_str = "correct" if correct else "incorrect"
        trajectory.append(StepResult(
            question_id=q_id, kc_ids=kc_ids,
            result=result_str, mastery_snapshot=dict(m),
        ))

        # --- Dropout check ---
        if _check_dropout(consecutive_wrong, diff, persona):
            dropout = True
            break

    return SimulationResult(
        final_mastery=m,
        trajectory=trajectory,
        dropout=dropout,
        steps_completed=len(trajectory),
        steps_total=len(path),
    )


def _check_dropout(consecutive_wrong: int, difficulty: float, persona: PersonaParams) -> bool:
    if consecutive_wrong == 0:
        return False
    p_dropout = min(1.0, persona.dropout_base_fn(consecutive_wrong) *
                    (1 + persona.difficulty_sensitivity * difficulty))
    return random.random() < p_dropout


def delta_mastery(m_before: Dict[str, float], m_after: Dict[str, float],
                  target_kcs: List[str]) -> float:
    """Compute mean positive mastery gain over targeted KCs."""
    if not target_kcs:
        return 0.0
    gains = [max(0.0, m_after.get(kc, 0.0) - m_before.get(kc, 0.0)) for kc in target_kcs]
    return float(np.mean(gains))
