"""Extract realistic persona parameters from student data (Section 5.2)."""

import numpy as np
from typing import List, Dict, Optional
from simpath.personas.definitions import PersonaParams, make_realistic_persona


def extract_realistic_params(features: dict) -> PersonaParams:
    """
    Build a realistic persona from extracted behavioral features.

    Args:
        features: dict with keys from synthetic data or preprocessing:
            avg_session_length, dropout_rates, avg_elapsed_time,
            skip_rate, overall_accuracy
    """
    # Learn rate: proxy from accuracy
    accuracy = features.get("overall_accuracy", 0.5)
    learn_rate = float(np.clip(accuracy * 0.2, 0.01, 0.30))

    # Forget penalty: inversely related to session engagement
    avg_session = features.get("avg_session_length", 10)
    forget_penalty = float(np.clip(0.15 - avg_session * 0.005, 0.01, 0.20))

    # Dropout rates (empirical)
    dropout_rates = features.get("dropout_rates", {k: 0.05 * k for k in range(1, 6)})

    # Difficulty sensitivity
    difficulty_sensitivity = 0.3  # default

    # Skip threshold
    skip_rate = features.get("skip_rate", 0.0)
    skip_threshold = None
    if skip_rate > 0.1:
        skip_threshold = max(0.3, 1.0 - skip_rate * 2)

    # Time multiplier
    avg_elapsed = features.get("avg_elapsed_time", 30)
    global_mean = 30.0  # approximate
    time_multiplier = float(avg_elapsed / global_mean)

    return make_realistic_persona(
        learn_rate=learn_rate,
        forget_penalty=forget_penalty,
        dropout_rates=dropout_rates,
        difficulty_sensitivity=difficulty_sensitivity,
        skip_threshold=skip_threshold,
        time_multiplier=time_multiplier,
    )
