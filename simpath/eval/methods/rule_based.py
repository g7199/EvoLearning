"""Rule-based baseline: graph topological sort, weakest-first."""
import numpy as np
from simpath.eval.methods import register_method
from simpath.eval.methods.base import BaseMethod


@register_method
class RuleBasedMethod(BaseMethod):
    name = "Rule-based"
    needs_graph = True
    needs_training = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = None

    def train(self, train_data, val_data, kes, graph, experts, out_dir=None, **kwargs):
        self.graph = graph

    def predict(self, mastery, targets, kes=None, hc=None, hr=None):
        order = self.graph.find_learning_order(targets, mastery, mastery_threshold=0.5)
        path = order[:self.L]
        if len(path) < self.L:
            used = set(path)
            zpd = sorted(range(self.num_c), key=lambda c: abs(mastery[c] - 0.5))
            for c in zpd:
                if c not in used:
                    path.append(c)
                    used.add(c)
                    if len(path) >= self.L:
                        break
        return path[:self.L]
