"""Target-repeat baseline."""
import numpy as np
from simpath.eval.methods import register_method
from simpath.eval.methods.base import BaseMethod


@register_method
class TargetRepeatMethod(BaseMethod):
    name = "Target-repeat"
    needs_training = False

    def train(self, train_data, val_data, kes, graph, experts, out_dir=None, **kwargs):
        pass

    def predict(self, mastery, targets):
        return (targets * ((self.L // len(targets)) + 1))[:self.L]
