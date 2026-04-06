"""B6: SimPath-NoRobust baseline (average instead of minimax regret)."""

from simpath.selection.minimax_regret import composite_scores, average_select


def select_norobust(S, weights):
    """Select path by average composite score instead of minimax regret."""
    phi = composite_scores(S, weights)
    return average_select(phi)
