"""
KnowLP (Cheng et al., 2025) — DLELP + EDU-GraphRAG (TextGrad + 3-vote).
Same RL algorithm as DLELP, but graph built via LLM with TextGrad refinement.
"""
from simpath.eval.methods import register_method
from simpath.eval.methods.dlelp import DLELPMethod


@register_method
class KnowLPMethod(DLELPMethod):
    name = "KnowLP"
    graph_type = 'knowlp'  # uses EDU-GraphRAG graph (not plain LLM or DKT)
