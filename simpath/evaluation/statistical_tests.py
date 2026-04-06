"""Statistical testing (Section 10)."""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple


def paired_comparison(
    simpath_scores: List[float],
    baseline_scores: List[float],
    alpha: float = 0.05,
    n_comparisons: int = 24,
) -> Dict:
    """
    Compare SimPath vs a baseline with appropriate statistical test.
    Uses Bonferroni correction.
    """
    differences = np.array(simpath_scores) - np.array(baseline_scores)

    # Normality check
    if len(differences) >= 8:
        _, p_shapiro = stats.shapiro(differences)
    else:
        p_shapiro = 0.0  # too few samples, use non-parametric

    if p_shapiro > 0.05:
        stat, p_value = stats.ttest_rel(simpath_scores, baseline_scores)
        test_name = "paired t-test"
    else:
        stat, p_value = stats.wilcoxon(differences)
        test_name = "Wilcoxon signed-rank"

    # Effect size (Cohen's d)
    d_std = np.std(differences)
    cohens_d = float(np.mean(differences) / d_std) if d_std > 0 else 0.0

    # Bonferroni correction
    adjusted_alpha = alpha / n_comparisons

    return {
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p_value),
        "cohens_d": cohens_d,
        "adjusted_alpha": adjusted_alpha,
        "significant": p_value < adjusted_alpha,
        "mean_diff": float(np.mean(differences)),
    }


def ablation_comparison(
    variant_scores: Dict[str, List[float]],
    alpha: float = 0.05,
) -> Dict:
    """Kruskal-Wallis + Dunn's test for ablation variants."""
    groups = list(variant_scores.values())
    labels = list(variant_scores.keys())

    stat, p_value = stats.kruskal(*groups)

    result = {
        "kruskal_wallis": {"statistic": float(stat), "p_value": float(p_value)},
        "pairwise": {},
    }

    # Post-hoc pairwise if significant
    if p_value < alpha:
        n_pairs = len(labels) * (len(labels) - 1) // 2
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                _, p = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                adjusted_p = min(p * n_pairs, 1.0)  # Bonferroni
                result["pairwise"][f"{labels[i]}_vs_{labels[j]}"] = {
                    "p_value": float(p),
                    "adjusted_p": float(adjusted_p),
                    "significant": adjusted_p < alpha,
                }

    return result
