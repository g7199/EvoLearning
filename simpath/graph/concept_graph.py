"""
EDU-Graph RAG — Build prerequisite & similarity graphs for KCs.
Adapted from DLELP (2025).

1. LLM generates KC descriptions (TextGrad-style iterative refinement)
2. LLM extracts prerequisite + similarity relationships
3. Build adjacency matrices for both graphs
"""
import json
import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional


class ConceptGraph:
    """Prerequisite and similarity graphs over knowledge concepts."""

    def __init__(self, n_concepts: int):
        self.n_concepts = n_concepts
        # prereq[i][j] = 1 means concept i is prerequisite of j
        self.prereq = np.zeros((n_concepts, n_concepts), dtype=np.float32)
        # sim[i][j] = similarity score between i and j
        self.sim = np.zeros((n_concepts, n_concepts), dtype=np.float32)
        # concept descriptions
        self.descriptions: Dict[int, str] = {}
        # concept names (from dataset)
        self.names: Dict[int, str] = {}

    def get_prerequisites(self, concept: int) -> List[int]:
        """Get all prerequisite concepts of `concept`."""
        return list(np.where(self.prereq[:, concept] > 0.5)[0])

    def get_dependents(self, concept: int) -> List[int]:
        """Get concepts that depend on `concept`."""
        return list(np.where(self.prereq[concept, :] > 0.5)[0])

    def get_similar(self, concept: int, top_k: int = 5) -> List[int]:
        """Get top-k most similar concepts."""
        scores = self.sim[concept]
        scores[concept] = -1  # exclude self
        idx = np.argsort(scores)[::-1][:top_k]
        return [int(i) for i in idx if scores[i] > 0]

    def find_learning_order(self, targets: List[int], mastery: np.ndarray,
                            mastery_threshold: float = 0.5) -> List[int]:
        """
        P-Agent style: trace back prerequisites from targets,
        find unmastered prerequisites, return topological order.
        """
        to_learn = set()
        visited = set()

        def trace_back(c):
            if c in visited:
                return
            visited.add(c)
            if mastery[c] < mastery_threshold:
                to_learn.add(c)
            for prereq in self.get_prerequisites(c):
                if mastery[prereq] < mastery_threshold:
                    trace_back(prereq)

        for t in targets:
            trace_back(t)

        # Topological sort (Kahn's algorithm)
        # Only over concepts in to_learn
        in_degree = {c: 0 for c in to_learn}
        for c in to_learn:
            for p in self.get_prerequisites(c):
                if p in to_learn:
                    in_degree[c] += 1

        queue = [c for c in to_learn if in_degree[c] == 0]
        queue.sort(key=lambda c: mastery[c])  # weakest first among no-prereqs
        order = []
        while queue:
            c = queue.pop(0)
            order.append(c)
            for dep in self.get_dependents(c):
                if dep in in_degree:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        queue.append(dep)
                        queue.sort(key=lambda x: mastery[x])

        # Add remaining (cycles or isolated) by mastery
        remaining = [c for c in to_learn if c not in order]
        remaining.sort(key=lambda c: mastery[c])
        order.extend(remaining)

        return order

    def save(self, path: str):
        data = {
            'n_concepts': self.n_concepts,
            'prereq': self.prereq,
            'sim': self.sim,
            'descriptions': self.descriptions,
            'names': self.names,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'ConceptGraph':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        g = cls(data['n_concepts'])
        g.prereq = data['prereq']
        g.sim = data['sim']
        g.descriptions = data['descriptions']
        g.names = data['names']
        return g


def build_graph_with_llm(concept_names: Dict[int, str], llm_client,
                         batch_size: int = 20) -> ConceptGraph:
    """
    Build concept graph using LLM (EDU-Graph RAG style).
    1. Generate descriptions for each concept
    2. Extract prerequisite relationships
    3. Extract similarity relationships
    """
    n = max(concept_names.keys()) + 1
    graph = ConceptGraph(n)
    graph.names = concept_names

    all_concepts = sorted(concept_names.keys())
    name_list = "\n".join([f"  {c}: {concept_names[c]}" for c in all_concepts])

    # Step 1: Generate descriptions (batch)
    print(f"[Graph] Generating descriptions for {len(all_concepts)} concepts...")
    for i in range(0, len(all_concepts), batch_size):
        batch = all_concepts[i:i+batch_size]
        batch_names = "\n".join([f"  {c}: {concept_names[c]}" for c in batch])
        resp = llm_client.generate(
            f"For each math/science concept below, write a brief (1-2 sentence) "
            f"educational description explaining what it covers.\n\n"
            f"Concepts:\n{batch_names}\n\n"
            f"Return JSON: {{\"concept_id\": \"description\", ...}}"
        )
        try:
            desc = json.loads(resp.strip().strip('```json').strip('```'))
            for k, v in desc.items():
                graph.descriptions[int(k)] = v
        except json.JSONDecodeError:
            for c in batch:
                graph.descriptions[c] = concept_names[c]

    # Step 2: Extract prerequisite relationships (batch)
    print(f"[Graph] Extracting prerequisite relationships...")
    for i in range(0, len(all_concepts), batch_size):
        batch = all_concepts[i:i+batch_size]
        batch_info = "\n".join([
            f"  {c}: {concept_names[c]} — {graph.descriptions.get(c, '')}"
            for c in batch
        ])
        resp = llm_client.generate(
            f"Given these math/science concepts and ALL concepts in the curriculum:\n\n"
            f"Current batch:\n{batch_info}\n\n"
            f"All concepts:\n{name_list}\n\n"
            f"For each concept in the batch, identify which OTHER concepts are "
            f"direct prerequisites (must be learned before this concept).\n\n"
            f"Return JSON: {{\"concept_id\": [prereq_id1, prereq_id2, ...], ...}}\n"
            f"Use concept IDs (numbers). Empty list if no prerequisites."
        )
        try:
            prereqs = json.loads(resp.strip().strip('```json').strip('```'))
            for k, v in prereqs.items():
                c = int(k)
                for p in v:
                    p = int(p)
                    if 0 <= p < n and p != c:
                        graph.prereq[p, c] = 1.0
        except (json.JSONDecodeError, ValueError):
            pass

    # Step 3: Extract similarity relationships (batch)
    print(f"[Graph] Extracting similarity relationships...")
    for i in range(0, len(all_concepts), batch_size):
        batch = all_concepts[i:i+batch_size]
        batch_info = "\n".join([
            f"  {c}: {concept_names[c]}"
            for c in batch
        ])
        resp = llm_client.generate(
            f"Given these math/science concepts:\n\n"
            f"Current batch:\n{batch_info}\n\n"
            f"All concepts:\n{name_list}\n\n"
            f"For each concept in the batch, identify the top 3-5 most similar or "
            f"closely related concepts (confusable or complementary).\n\n"
            f"Return JSON: {{\"concept_id\": [similar_id1, similar_id2, ...], ...}}"
        )
        try:
            sims = json.loads(resp.strip().strip('```json').strip('```'))
            for k, v in sims.items():
                c = int(k)
                for s in v:
                    s = int(s)
                    if 0 <= s < n and s != c:
                        graph.sim[c, s] = 1.0
                        graph.sim[s, c] = 1.0
        except (json.JSONDecodeError, ValueError):
            pass

    # Stats
    n_prereq = int(graph.prereq.sum())
    n_sim = int(graph.sim.sum()) // 2
    print(f"[Graph] Built: {n_prereq} prerequisite edges, {n_sim} similarity edges")

    return graph
