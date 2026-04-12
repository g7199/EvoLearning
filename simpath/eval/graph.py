"""Graph loading and A* candidate generation."""
import heapq
import numpy as np
import pickle
from typing import List, Set
from simpath.graph.concept_graph import ConceptGraph


def load_graph(dataset_config: dict, graph_type: str = 'dkt') -> ConceptGraph:
    """Load concept graph. graph_type: 'dkt', 'llm', or 'knowlp'."""
    num_c = dataset_config['num_c']

    if graph_type == 'knowlp':
        # EDU-GraphRAG: TextGrad + 3-vote self-consistency
        from simpath.eval.knowlp_graph import build_edu_graphrag
        return build_edu_graphrag(dataset_config, provider='openai')

    if graph_type == 'llm':
        path = dataset_config.get('graph_llm_path')
        if path is None:
            raise ValueError(f"No LLM graph available for this dataset")
        with open(path, 'rb') as f:
            dd = pickle.load(f)
        g = ConceptGraph(num_c)
        g.prereq = dd['prereq']
        g.sim = dd['sim']
        g.names = dd.get('names', {})
        return g

    # DKT influence graph
    with open(dataset_config['graph_dkt_path'], 'rb') as f:
        dd = pickle.load(f)
    cross = dd['influence'].copy()
    np.fill_diagonal(cross, 0)
    thr = dataset_config['graph_thr']
    g = ConceptGraph(num_c)
    g.prereq = (cross > thr).astype(np.float32)
    for a in range(num_c):
        for b in range(a + 1, num_c):
            if cross[a, b] > thr * 0.5 and cross[b, a] > thr * 0.5:
                g.sim[a, b] = 1.0
                g.sim[b, a] = 1.0
    g.names = dd.get('names', {})
    return g


def get_graph_candidates(targets: List[int], mastery: np.ndarray,
                         graph: ConceptGraph, num_c: int, cap: int = 30) -> List[int]:
    """Graph-guided candidate selection: targets + prereqs + similar + ZPD fill."""
    r = set()
    for t in targets:
        r.add(t)
        for p in graph.get_prerequisites(t):
            r.add(p)
        for s in graph.get_similar(t, top_k=5):
            r.add(s)
    if len(r) < cap:
        for c in sorted(range(num_c), key=lambda c: abs(mastery[c] - 0.5)):
            r.add(c)
            if len(r) >= cap:
                break
    return list(r)[:cap]


def astar_candidates(targets: List[int], mastery: np.ndarray,
                     graph: ConceptGraph, num_c: int, used: Set[int]) -> Set[int]:
    """
    A* backward tracing from targets through prerequisite graph.
    Paper: "A* algorithm for dynamic candidate action space determination."
    Cost g(n) = accumulated (1 - mastery) along path.
    Heuristic h(n) = 0 (Dijkstra-like, guarantees optimality).
    Returns candidate set of concepts on prerequisite paths to targets.
    """
    candidates = set()
    for target in targets:
        if target in used:
            continue
        candidates.add(target)
        visited = set()
        # (cost, concept)
        open_set = [(1.0 - mastery[target], target)]
        while open_set:
            cost, node = heapq.heappop(open_set)
            if node in visited:
                continue
            visited.add(node)
            candidates.add(node)
            for prereq in graph.get_prerequisites(node):
                if prereq not in visited and prereq not in used:
                    if mastery[prereq] < 0.7:
                        heapq.heappush(open_set, (cost + (1.0 - mastery[prereq]), prereq))
    candidates -= used
    # Add similar concepts if too few
    if len(candidates) < 5:
        for t in targets:
            for s in graph.get_similar(t, top_k=5):
                if s not in used:
                    candidates.add(s)
    # ZPD fill
    if len(candidates) < 5:
        zpd = sorted(range(num_c), key=lambda c: abs(mastery[c] - 0.5))
        for c in zpd:
            if c not in used and c not in candidates:
                candidates.add(c)
                if len(candidates) >= 15:
                    break
    return candidates


def get_goal_candidates(goal: int, mastery: np.ndarray,
                        graph: ConceptGraph, num_c: int,
                        used: Set[int], cap: int = 20) -> List[int]:
    """GEHRL: graph-filtered candidates for a specific target goal."""
    r = set()
    r.add(goal)
    for p in graph.get_prerequisites(goal):
        r.add(p)
    for s in graph.get_similar(goal, top_k=5):
        r.add(s)
    # 2-hop expansion
    for p in list(r):
        for pp in graph.get_prerequisites(p):
            r.add(pp)
    if len(r) < cap:
        for c in sorted(range(num_c), key=lambda c: abs(mastery[c] - 0.5)):
            r.add(c)
            if len(r) >= cap:
                break
    return [c for c in list(r)[:cap] if c not in used]
