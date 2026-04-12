"""Random baseline."""
import numpy as np
from simpath.eval.methods import register_method
from simpath.eval.methods.base import BaseMethod


@register_method
class RandomMethod(BaseMethod):
    name = "Random"
    needs_training = False

    def train(self, train_data, val_data, kes, graph, experts, out_dir=None, **kwargs):
        pass

    def predict(self, mastery, targets):
        pool = list(range(self.num_c))
        np.random.shuffle(pool)
        return pool[:self.L]
