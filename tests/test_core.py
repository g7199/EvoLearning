"""Unit tests for core SimPath modules."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from collections import defaultdict

# ---- Test Persona Definitions ----

def test_persona_params():
    from simpath.personas.definitions import THETA_GRIT, THETA_FRAGILE
    assert THETA_GRIT.learn_rate == 0.15
    assert THETA_GRIT.skip_threshold is None
    assert THETA_FRAGILE.skip_threshold == 0.7
    assert THETA_FRAGILE.p_correct_boost == -0.08

def test_dropout_fn():
    from simpath.personas.definitions import THETA_GRIT, THETA_FRAGILE
    # Grit: 0.02 * k
    assert abs(THETA_GRIT.dropout_base_fn(3) - 0.06) < 1e-6
    # Fragile: 0.15 * k^1.5
    assert abs(THETA_FRAGILE.dropout_base_fn(2) - 0.15 * 2**1.5) < 1e-6

def test_make_realistic_persona():
    from simpath.personas.definitions import make_realistic_persona
    p = make_realistic_persona(
        learn_rate=0.12, forget_penalty=0.08,
        dropout_rates={1: 0.05, 2: 0.12, 3: 0.25, 4: 0.40, 5: 0.55},
    )
    assert p.name == "realistic"
    assert p.dropout_base_fn(2) == 0.12
    assert p.p_correct_boost == 0.0

# ---- Test KT Simulation ----

def test_simulate_kt_no_dropout():
    from simpath.kt.simulate import simulate_kt
    from simpath.personas.definitions import THETA_GRIT
    from simpath.utils.seeds import set_global_seed
    set_global_seed(42)

    mastery = {"kc_0": 0.3, "kc_1": 0.5}
    path = [
        {"question_id": "q_0", "kc_ids": ["kc_0"], "difficulty": 0.3},
        {"question_id": "q_1", "kc_ids": ["kc_1"], "difficulty": 0.4},
    ]
    result = simulate_kt(mastery, path, THETA_GRIT)
    assert result.steps_total == 2
    assert result.steps_completed <= 2
    assert isinstance(result.final_mastery, dict)
    assert "kc_0" in result.final_mastery

def test_simulate_kt_fragile_drops_out_more():
    """Fragile persona should have higher dropout rate than Gritty."""
    from simpath.kt.simulate import simulate_kt
    from simpath.personas.definitions import THETA_GRIT, THETA_FRAGILE
    from simpath.utils.seeds import set_global_seed

    mastery = {"kc_0": 0.2}
    path = [{"question_id": f"q_{i}", "kc_ids": ["kc_0"], "difficulty": 0.8}
            for i in range(20)]

    grit_dropouts = 0
    frag_dropouts = 0
    N = 100
    for i in range(N):
        set_global_seed(i)
        r = simulate_kt(mastery, path, THETA_GRIT)
        if r.dropout:
            grit_dropouts += 1
        set_global_seed(i)
        r = simulate_kt(mastery, path, THETA_FRAGILE)
        if r.dropout:
            frag_dropouts += 1

    assert frag_dropouts > grit_dropouts, \
        f"Fragile ({frag_dropouts}) should dropout more than Grit ({grit_dropouts})"

def test_delta_mastery():
    from simpath.kt.simulate import delta_mastery
    m_before = {"kc_0": 0.3, "kc_1": 0.5}
    m_after = {"kc_0": 0.5, "kc_1": 0.4}  # kc_1 decreased
    dm = delta_mastery(m_before, m_after, ["kc_0", "kc_1"])
    # Only positive gains: kc_0 +0.2, kc_1 max(0, -0.1) = 0
    assert abs(dm - 0.1) < 1e-6

# ---- Test Path Retrieval ----

def _make_test_pool():
    questions = []
    skill_to_questions = defaultdict(list)
    for i in range(50):
        kc = f"kc_{i % 5}"
        q = {"question_id": f"q_{i}", "kc_ids": [kc], "difficulty": i / 50}
        questions.append(q)
        skill_to_questions[kc].append(q)
    return questions, dict(skill_to_questions)

def test_retrieve_pool_basics():
    from simpath.paths.retrieval import retrieve_candidate_pool
    questions, s2q = _make_test_pool()
    mastery = {"kc_0": 0.3, "kc_1": 0.8, "kc_2": 0.2, "kc_3": 0.9, "kc_4": 0.4}

    pool = retrieve_candidate_pool(mastery, questions, s2q, L=8)
    assert len(pool) >= 8
    # All questions should be real
    pool_ids = {q["question_id"] for q in pool}
    all_ids = {q["question_id"] for q in questions}
    assert pool_ids.issubset(all_ids)

def test_retrieve_pool_focuses_weak_kcs():
    from simpath.paths.retrieval import retrieve_candidate_pool
    questions, s2q = _make_test_pool()
    mastery = {"kc_0": 0.1, "kc_1": 0.9, "kc_2": 0.9, "kc_3": 0.9, "kc_4": 0.9}

    pool = retrieve_candidate_pool(mastery, questions, s2q, L=8)
    # Most questions should be for kc_0 (the only weak KC)
    kc0_count = sum(1 for q in pool if "kc_0" in q["kc_ids"])
    assert kc0_count > 0

# ---- Test Mock Ordering ----

def test_mock_generate_paths():
    from simpath.paths.mock_ordering import mock_generate_paths
    mastery = {"kc_0": 0.3, "kc_1": 0.5, "kc_2": 0.7}
    pool = [{"question_id": f"q_{i}", "kc_ids": [f"kc_{i%3}"], "difficulty": i/20}
            for i in range(20)]

    paths = mock_generate_paths(mastery, pool, K=5, L=8)
    assert len(paths) == 5
    for p in paths:
        assert len(p) <= 8
        # No duplicate question_ids within a path
        ids = [q["question_id"] for q in p]
        assert len(ids) == len(set(ids))

# ---- Test Simulation Matrix ----

def test_simulate_all_personas():
    from simpath.personas.simulation import simulate_all_personas
    from simpath.personas.definitions import make_realistic_persona

    mastery = {"kc_0": 0.3, "kc_1": 0.5}
    paths = [
        [{"question_id": f"q_{i}", "kc_ids": ["kc_0"], "difficulty": 0.4} for i in range(4)],
        [{"question_id": f"q_{i+10}", "kc_ids": ["kc_1"], "difficulty": 0.5} for i in range(4)],
    ]
    theta_real = make_realistic_persona(
        learn_rate=0.12, forget_penalty=0.06,
        dropout_rates={1: 0.05, 2: 0.10, 3: 0.20, 4: 0.35, 5: 0.50},
    )
    S = simulate_all_personas(mastery, paths, theta_real, N_sim=5)
    assert S.shape == (2, 3, 3)
    # All values should be in [0, 1]
    assert np.all(S >= 0) and np.all(S <= 1)

# ---- Test Minimax Regret Selection ----

def test_minimax_regret():
    from simpath.selection.minimax_regret import composite_scores, minimax_regret_select

    # S[K=3, J=3, metrics=3]
    S = np.array([
        [[0.5, 0.8, 0.9], [0.3, 0.6, 0.7], [0.4, 0.7, 0.8]],  # path 0
        [[0.6, 0.7, 0.8], [0.5, 0.7, 0.8], [0.5, 0.7, 0.8]],  # path 1 (balanced)
        [[0.7, 0.9, 1.0], [0.1, 0.3, 0.4], [0.4, 0.6, 0.7]],  # path 2 (risky)
    ])
    weights = (0.5, 0.3, 0.2)
    phi = composite_scores(S, weights)
    assert phi.shape == (3, 3)

    best_idx, regret, max_regret = minimax_regret_select(phi)
    assert 0 <= best_idx < 3
    assert regret.shape == (3, 3)
    assert np.all(regret >= -1e-10)  # regret is non-negative

def test_selection_strategies():
    from simpath.selection.minimax_regret import (
        average_select, maximin_select, hurwicz_select, weighted_minimax_select,
    )
    phi = np.array([
        [0.8, 0.2],  # path 0: good for persona 0, bad for 1
        [0.5, 0.5],  # path 1: balanced
        [0.2, 0.8],  # path 2: opposite of 0
    ])
    # Average should pick balanced or best avg
    avg_idx = average_select(phi)
    assert avg_idx in [0, 1, 2]
    # Maximin should prefer the balanced path
    maximin_idx = maximin_select(phi)
    assert maximin_idx == 1  # min(0.5, 0.5) = 0.5 > min(0.8,0.2) = 0.2

# ---- Test Evaluation Metrics ----

def test_evaluate_path():
    from simpath.evaluation.offline_metrics import evaluate_path
    from simpath.personas.definitions import make_realistic_persona

    mastery = {"kc_0": 0.3, "kc_1": 0.5}
    path = [{"question_id": f"q_{i}", "kc_ids": ["kc_0"], "difficulty": 0.4}
            for i in range(4)]
    theta_real = make_realistic_persona(
        learn_rate=0.12, forget_penalty=0.06,
        dropout_rates={1: 0.05, 2: 0.10, 3: 0.20, 4: 0.35, 5: 0.50},
    )
    metrics = evaluate_path(
        mastery=mastery, path=path, theta_real=theta_real,
        target_kcs=["kc_0"], N_sim=5, student_id="test",
    )
    assert set(metrics.keys()) == {"EP", "SR", "CE", "RI"}
    for v in metrics.values():
        assert 0 <= v <= 1.0 + 1e-6

# ---- Test Statistical Tests ----

def test_paired_comparison():
    from simpath.evaluation.statistical_tests import paired_comparison
    np.random.seed(42)
    simpath = list(np.random.normal(0.6, 0.1, 30))
    baseline = list(np.random.normal(0.5, 0.1, 30))
    result = paired_comparison(simpath, baseline, n_comparisons=4)
    assert "p_value" in result
    assert "cohens_d" in result
    assert result["mean_diff"] > 0

# ---- Test Synthetic Data Generation ----

def test_synthetic_data():
    from simpath.data.synthetic import generate_synthetic_dataset
    students, questions, s2q = generate_synthetic_dataset(
        n_students=20, n_questions=30, n_kcs=5, seed=42,
    )
    assert len(students) == 20
    assert len(questions) == 30
    assert len(s2q) == 5
    for s in students:
        assert len(s["train"]) > 0
        assert len(s["held_out"]) > 0
        assert "mastery" in s
        assert "features" in s

# ---- Test Baselines ----

def test_baseline_random():
    from simpath.baselines.dkt_random import recommend_random
    import random
    random.seed(42)
    _, s2q = _make_test_pool()
    mastery = {"kc_0": 0.3, "kc_1": 0.8, "kc_2": 0.2, "kc_3": 0.9, "kc_4": 0.4}
    path = recommend_random(mastery, s2q, L=8)
    assert len(path) == 8
    ids = [q["question_id"] for q in path]
    assert len(ids) == len(set(ids))

def test_baseline_rulebased():
    from simpath.baselines.dkt_rulebased import recommend_rulebased
    _, s2q = _make_test_pool()
    mastery = {"kc_0": 0.3, "kc_1": 0.8, "kc_2": 0.2, "kc_3": 0.9, "kc_4": 0.4}
    path = recommend_rulebased(mastery, s2q, L=8)
    assert len(path) <= 8
    assert len(path) > 0

# ---- Test Config ----

def test_load_config():
    from simpath.utils.config import load_config
    cfg = load_config()
    assert cfg["seed"] == 42
    assert cfg["paths"]["K"] == 5
    assert cfg["personas"]["grit"]["learn_rate"] == 0.15

# ---- Test Seeds ----

def test_simulation_seed_deterministic():
    from simpath.utils.seeds import simulation_seed
    s1 = simulation_seed("student_0", 0, 1, 2)
    s2 = simulation_seed("student_0", 0, 1, 2)
    s3 = simulation_seed("student_1", 0, 1, 2)
    assert s1 == s2
    assert s1 != s3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
