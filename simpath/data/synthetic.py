"""Generate synthetic dataset for smoke testing the pipeline."""

import random
import numpy as np
from typing import List, Dict, Tuple


def generate_synthetic_dataset(
    n_students: int = 200,
    n_questions: int = 100,
    n_kcs: int = 10,
    min_interactions: int = 50,
    max_interactions: int = 200,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], Dict[str, List[dict]]]:
    """
    Generate synthetic interaction data.

    Returns:
        students: list of student dicts with interaction histories
        questions: list of question metadata
        skill_to_questions: mapping from KC id to questions
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # Generate questions with KC assignments and difficulties
    questions = []
    skill_to_questions: Dict[str, List[dict]] = {f"kc_{i}": [] for i in range(n_kcs)}

    for q_idx in range(n_questions):
        n_skills = rng.choice([1, 2], p=[0.7, 0.3])
        kc_ids = [f"kc_{k}" for k in rng.choice(n_kcs, size=n_skills, replace=False)]
        difficulty = float(rng.beta(2, 2))  # centered around 0.5

        q = {
            "question_id": f"q_{q_idx}",
            "kc_ids": kc_ids,
            "difficulty": round(difficulty, 3),
        }
        questions.append(q)
        for kc in kc_ids:
            skill_to_questions[kc].append(q)

    # Generate students
    students = []
    for s_idx in range(n_students):
        n_interactions = rng.integers(min_interactions, max_interactions + 1)

        # Student's true mastery (latent)
        true_mastery = {f"kc_{k}": float(rng.beta(2, 3)) for k in range(n_kcs)}

        # Student behavior type
        behavior = rng.choice(["gritty", "fragile", "normal"], p=[0.25, 0.25, 0.50])
        if behavior == "gritty":
            dropout_tendency = 0.02
            learn_speed = 0.15
        elif behavior == "fragile":
            dropout_tendency = 0.15
            learn_speed = 0.08
        else:
            dropout_tendency = 0.05
            learn_speed = 0.10

        # Generate interaction history
        history = []
        current_mastery = dict(true_mastery)
        base_time = 1_600_000_000  # some base unix timestamp

        for t in range(n_interactions):
            q = questions[rng.integers(n_questions)]
            avg_m = np.mean([current_mastery.get(kc, 0.3) for kc in q["kc_ids"]])
            p_correct = max(0.05, min(0.95, avg_m + (0.5 - q["difficulty"]) * 0.4))
            correct = int(rng.random() < p_correct)

            elapsed = max(3, rng.normal(30, 15))
            if behavior == "fragile" and q["difficulty"] > 0.7:
                elapsed = max(2, elapsed * 0.5)

            interaction = {
                "question_id": q["question_id"],
                "kc_ids": q["kc_ids"],
                "difficulty": q["difficulty"],
                "correct": correct,
                "elapsed_time": round(float(elapsed), 1),
                "timestamp": base_time + t * rng.integers(60, 600),
            }
            history.append(interaction)

            # Update mastery
            for kc in q["kc_ids"]:
                if correct:
                    current_mastery[kc] += learn_speed * (1 - current_mastery[kc])
                else:
                    current_mastery[kc] = max(0.0, current_mastery[kc] - 0.02)

        # Split: 70/10/10/10
        n = len(history)
        splits = [int(n * 0.7), int(n * 0.8), int(n * 0.9)]
        train = history[:splits[0]]
        val = history[splits[0]:splits[1]]
        rec_input = history[splits[1]:splits[2]]
        held_out = history[splits[2]:]

        # Compute mastery from rec_input (recent history)
        mastery_estimate = {}
        for kc_id in [f"kc_{k}" for k in range(n_kcs)]:
            kc_interactions = [h for h in rec_input if kc_id in h["kc_ids"]]
            if kc_interactions:
                mastery_estimate[kc_id] = np.mean([h["correct"] for h in kc_interactions])
            else:
                # Use training data as fallback
                train_kc = [h for h in train if kc_id in h["kc_ids"]]
                mastery_estimate[kc_id] = np.mean([h["correct"] for h in train_kc]) if train_kc else 0.5

        # Compute behavioral features
        sessions = _split_sessions(history, gap_seconds=1800)
        avg_session_len = np.mean([len(s) for s in sessions]) if sessions else 10

        consecutive_wrong_drops = _compute_dropout_rates(sessions)

        fast_answers = [h for h in history if h["elapsed_time"] < 5]
        skip_rate = len(fast_answers) / max(len(history), 1)

        students.append({
            "student_id": f"s_{s_idx}",
            "behavior_type": behavior,
            "train": train,
            "val": val,
            "rec_input": rec_input,
            "held_out": held_out,
            "mastery": mastery_estimate,
            "features": {
                "avg_session_length": float(avg_session_len),
                "dropout_rates": consecutive_wrong_drops,
                "avg_elapsed_time": float(np.mean([h["elapsed_time"] for h in history])),
                "skip_rate": float(skip_rate),
                "overall_accuracy": float(np.mean([h["correct"] for h in history])),
            },
        })

    return students, questions, skill_to_questions


def _split_sessions(history: List[dict], gap_seconds: int = 1800) -> List[List[dict]]:
    if not history:
        return []
    sessions = [[history[0]]]
    for h in history[1:]:
        if h["timestamp"] - sessions[-1][-1]["timestamp"] > gap_seconds:
            sessions.append([])
        sessions[-1].append(h)
    return sessions


def _compute_dropout_rates(sessions: List[List[dict]]) -> Dict[int, float]:
    counts = {k: {"end": 0, "cont": 0} for k in range(1, 6)}
    for session in sessions:
        consec = 0
        for i, h in enumerate(session):
            if h["correct"] == 0:
                consec += 1
                if consec in counts:
                    if i == len(session) - 1:
                        counts[consec]["end"] += 1
                    else:
                        counts[consec]["cont"] += 1
            else:
                consec = 0

    rates = {}
    for k in range(1, 6):
        total = counts[k]["end"] + counts[k]["cont"]
        rates[k] = counts[k]["end"] / max(total, 1)
    return rates
