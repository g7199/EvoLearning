from dataclasses import dataclass, field
from typing import Optional, Callable
import math


@dataclass
class PersonaParams:
    name: str
    learn_rate: float
    forget_penalty: float
    dropout_base_fn: Callable[[int], float]  # k (consecutive wrong) -> base dropout prob
    difficulty_sensitivity: float
    skip_threshold: Optional[float]  # None = never skip
    p_correct_boost: float
    time_multiplier: float


def _make_dropout_fn(coeff: float, exp: float) -> Callable[[int], float]:
    def fn(k: int) -> float:
        return coeff * (k ** exp)
    return fn


THETA_GRIT = PersonaParams(
    name="grit",
    learn_rate=0.15,
    forget_penalty=0.05,
    dropout_base_fn=_make_dropout_fn(0.02, 1.0),  # 0.02 * k
    difficulty_sensitivity=0.2,
    skip_threshold=None,
    p_correct_boost=0.05,
    time_multiplier=1.3,
)

THETA_FRAGILE = PersonaParams(
    name="fragile",
    learn_rate=0.10,
    forget_penalty=0.10,
    dropout_base_fn=_make_dropout_fn(0.15, 1.5),  # 0.15 * k^1.5
    difficulty_sensitivity=0.5,
    skip_threshold=0.7,
    p_correct_boost=-0.08,
    time_multiplier=0.7,
)


def make_realistic_persona(
    learn_rate: float,
    forget_penalty: float,
    dropout_rates: dict,  # {k: probability}
    difficulty_sensitivity: float = 0.3,
    skip_threshold: Optional[float] = None,
    time_multiplier: float = 1.0,
) -> PersonaParams:
    def dropout_fn(k: int) -> float:
        return dropout_rates.get(k, dropout_rates.get(max(dropout_rates.keys()), 0.5))

    return PersonaParams(
        name="realistic",
        learn_rate=learn_rate,
        forget_penalty=forget_penalty,
        dropout_base_fn=dropout_fn,
        difficulty_sensitivity=difficulty_sensitivity,
        skip_threshold=skip_threshold,
        p_correct_boost=0.0,
        time_multiplier=time_multiplier,
    )
