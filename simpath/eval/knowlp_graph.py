"""
EDU-GraphRAG — KnowLP's graph generation pipeline (arXiv 2506.22303).
Exact reproduction: TextGrad iterative refinement + 3-vote self-consistency.

Usage:
    from simpath.eval.knowlp_graph import build_edu_graphrag
    graph = build_edu_graphrag(dataset_config, provider='openai')
"""
import os, json, pickle, time, re
import numpy as np
from typing import Dict, List, Tuple
from simpath.graph.concept_graph import ConceptGraph


# ═══ Step 1: Initial KC Description Generation ═══

def generate_initial_descriptions(concept_names: Dict[int, str], llm) -> Dict[int, str]:
    """Generate educational descriptions for each KC via LLM."""
    descriptions = {}
    all_names = "\n".join([f"  {c}: {concept_names[c]}" for c in sorted(concept_names.keys())])

    for cid in sorted(concept_names.keys()):
        prompt = (
            f"You are an expert math/science educator.\n\n"
            f"Concept: {concept_names[cid]}\n\n"
            f"Write a precise 2-3 sentence educational description of this concept. "
            f"Include: (1) what it covers, (2) key skills involved, "
            f"(3) where it fits in the curriculum.\n\n"
            f"Description:"
        )
        resp = llm.generate(prompt)
        descriptions[cid] = resp.strip()
        if (cid + 1) % 20 == 0:
            print(f"    Descriptions: {cid+1}/{len(concept_names)}", flush=True)

    return descriptions


# ═══ Step 2: TextGrad Iterative Refinement ═══

def textgrad_refine(descriptions: Dict[int, str], concept_names: Dict[int, str],
                    llm, n_iter: int = 3) -> Dict[int, str]:
    """
    TextGrad-style iterative refinement of KC descriptions.
    For each iteration:
      1. Use current descriptions to extract sample relationships
      2. Evaluate quality (loss = inconsistency count)
      3. LLM generates "gradient" (improvement suggestions)
      4. Update descriptions based on gradient
    """
    all_cids = sorted(concept_names.keys())

    for iteration in range(n_iter):
        print(f"    TextGrad iteration {iteration+1}/{n_iter}...", flush=True)

        # Sample a subset of concept pairs for evaluation
        np.random.seed(iteration)
        sample_pairs = []
        for _ in range(min(30, len(all_cids))):
            a, b = np.random.choice(all_cids, 2, replace=False)
            sample_pairs.append((int(a), int(b)))

        # Extract relationships using current descriptions
        pair_results = []
        for a, b in sample_pairs:
            prompt = (
                f"Given these two math/science concepts with their descriptions:\n\n"
                f"Concept A ({concept_names[a]}): {descriptions[a]}\n\n"
                f"Concept B ({concept_names[b]}): {descriptions[b]}\n\n"
                f"Determine:\n"
                f"1. Is A a prerequisite of B? (yes/no)\n"
                f"2. Is B a prerequisite of A? (yes/no)\n"
                f"3. Are they similar/related? (yes/no)\n\n"
                f"Think step by step, then answer in JSON: "
                f'{{\"a_prereq_b\": true/false, \"b_prereq_a\": true/false, \"similar\": true/false}}'
            )
            resp = llm.generate(prompt)
            try:
                # Extract JSON from response
                match = re.search(r'\{[^}]+\}', resp)
                if match:
                    result = json.loads(match.group())
                    pair_results.append((a, b, result))
            except (json.JSONDecodeError, AttributeError):
                pass

        # Check for inconsistencies (loss signal)
        inconsistencies = []
        for a, b, r in pair_results:
            # Mutual prerequisite is inconsistent
            if r.get('a_prereq_b') and r.get('b_prereq_a'):
                inconsistencies.append(
                    f"Concepts {concept_names[a]} and {concept_names[b]} "
                    f"are marked as mutual prerequisites, which is contradictory."
                )
            # Prerequisite + similar might indicate unclear descriptions
            if (r.get('a_prereq_b') or r.get('b_prereq_a')) and r.get('similar'):
                inconsistencies.append(
                    f"Concepts {concept_names[a]} and {concept_names[b]} "
                    f"are both prerequisite-related and similar — descriptions may be ambiguous."
                )

        if not inconsistencies:
            print(f"      No inconsistencies found, descriptions converged.", flush=True)
            break

        # Generate "gradient" — improvement suggestions for problematic descriptions
        problem_concepts = set()
        for inc in inconsistencies:
            for cid in all_cids:
                if concept_names[cid] in inc:
                    problem_concepts.add(cid)

        for cid in list(problem_concepts)[:20]:  # limit updates per iteration
            gradient_prompt = (
                f"You are refining educational concept descriptions to reduce ambiguity.\n\n"
                f"Current description for '{concept_names[cid]}':\n"
                f"{descriptions[cid]}\n\n"
                f"Issues found:\n"
                + "\n".join([inc for inc in inconsistencies if concept_names[cid] in inc][:3])
                + f"\n\nRewrite the description to be more precise and clearly distinguish "
                f"this concept from related ones. Focus on unique aspects.\n\n"
                f"Improved description:"
            )
            resp = llm.generate(gradient_prompt)
            descriptions[cid] = resp.strip()

        print(f"      Updated {len(problem_concepts)} descriptions "
              f"({len(inconsistencies)} inconsistencies)", flush=True)

    return descriptions


# ═══ Step 3: Relationship Extraction with 3-Vote Self-Consistency ═══

def extract_relationships(descriptions: Dict[int, str], concept_names: Dict[int, str],
                          llm, n_votes: int = 3, batch_size: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract prerequisite and similarity relationships with CoT + self-consistency.
    Each query is repeated n_votes times; majority vote determines the relationship.
    """
    num_c = max(concept_names.keys()) + 1
    prereq_votes = np.zeros((num_c, num_c, n_votes), dtype=np.int32)
    sim_votes = np.zeros((num_c, num_c, n_votes), dtype=np.int32)

    all_cids = sorted(concept_names.keys())
    all_info = "\n".join([
        f"  {c}: {concept_names[c]} — {descriptions.get(c, '')[:100]}"
        for c in all_cids
    ])

    for vote in range(n_votes):
        print(f"    Vote {vote+1}/{n_votes}...", flush=True)

        # Prerequisite extraction (batched)
        for i in range(0, len(all_cids), batch_size):
            batch = all_cids[i:i + batch_size]
            batch_info = "\n".join([
                f"  {c}: {concept_names[c]} — {descriptions.get(c, '')[:150]}"
                for c in batch
            ])
            prompt = (
                f"You are an expert educator analyzing prerequisite relationships.\n\n"
                f"Current batch of concepts:\n{batch_info}\n\n"
                f"All concepts in the curriculum:\n{all_info}\n\n"
                f"For each concept in the batch, identify which OTHER concepts are "
                f"DIRECT prerequisites (must be learned BEFORE this concept).\n\n"
                f"Think carefully about learning order. A is prerequisite of B means "
                f"understanding A is necessary to learn B.\n\n"
                f"Return JSON: {{\"concept_id\": [prereq_id1, prereq_id2, ...], ...}}\n"
                f"Use concept IDs (numbers). Empty list if no prerequisites."
            )
            resp = llm.generate(prompt)  # LLM instance temperature used
            try:
                match = re.search(r'\{[\s\S]*\}', resp)
                if match:
                    prereqs = json.loads(match.group())
                    for k, v in prereqs.items():
                        c = int(k)
                        for p in v:
                            p = int(p)
                            if 0 <= p < num_c and p != c:
                                prereq_votes[p, c, vote] = 1
            except (json.JSONDecodeError, ValueError):
                pass

        # Similarity extraction (batched)
        for i in range(0, len(all_cids), batch_size):
            batch = all_cids[i:i + batch_size]
            batch_info = "\n".join([
                f"  {c}: {concept_names[c]}" for c in batch
            ])
            prompt = (
                f"You are an expert educator analyzing concept similarity.\n\n"
                f"Current batch:\n{batch_info}\n\n"
                f"All concepts:\n{all_info}\n\n"
                f"For each concept in the batch, identify the top 3-5 most similar or "
                f"closely related concepts (share common skills, often confused, or complementary).\n\n"
                f"Return JSON: {{\"concept_id\": [similar_id1, similar_id2, ...], ...}}"
            )
            resp = llm.generate(prompt)
            try:
                match = re.search(r'\{[\s\S]*\}', resp)
                if match:
                    sims = json.loads(match.group())
                    for k, v in sims.items():
                        c = int(k)
                        for s in v:
                            s = int(s)
                            if 0 <= s < num_c and s != c:
                                sim_votes[c, s, vote] = 1
            except (json.JSONDecodeError, ValueError):
                pass

    # Majority vote
    prereq = (prereq_votes.sum(axis=2) >= (n_votes / 2 + 0.5)).astype(np.float32)
    sim_raw = (sim_votes.sum(axis=2) >= (n_votes / 2 + 0.5)).astype(np.float32)
    # Make similarity symmetric
    sim = np.maximum(sim_raw, sim_raw.T)

    return prereq, sim


# ═══ Full Pipeline ═══

def build_edu_graphrag(dataset_config: dict, provider: str = 'openai',
                       cache_path: str = None) -> ConceptGraph:
    """
    Build KnowLP's EDU-GraphRAG graph. Full pipeline with caching.
    """
    import torch
    num_c = dataset_config['num_c']

    # Check cache
    if cache_path is None:
        dataset_name = os.path.basename(dataset_config['data_path']).split('_')[0]
        cache_path = f'outputs/knowlp_graph_{dataset_name}.pkl'

    if os.path.exists(cache_path):
        print(f"  [KnowLP] Loading cached EDU-GraphRAG graph from {cache_path}")
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        g = ConceptGraph(num_c)
        g.prereq = data['prereq']
        g.sim = data['sim']
        g.names = data.get('names', {})
        g.descriptions = data.get('descriptions', {})
        return g

    # Get concept names
    ckpt = torch.load(dataset_config['dkt_path'], weights_only=False, map_location='cpu')
    skill_map = ckpt['skill_map']
    inv_map = {v: k for k, v in skill_map.items()}

    with open(dataset_config['data_path'], 'rb') as f:
        data = pickle.load(f)

    # Build concept names from dataset
    concept_names = {}
    kc_list = data.get('kc_list', [])

    # Try Junyi-style: kc names are descriptive strings (e.g. 'algebra-functions')
    if kc_list and isinstance(kc_list[0], str):
        for kc in kc_list:
            if kc in skill_map:
                concept_names[skill_map[kc]] = kc.replace('-', ' ').title()

    # If Junyi-style didn't work (ASSIST09: kc_list is numeric strings like '1','10'),
    # load skill names from raw CSV
    if len(concept_names) < num_c // 2:
        concept_names = {}
        try:
            import pandas as pd
            df = pd.read_csv('data/raw/assist09/skill_builder_data_corrected.csv',
                             encoding='latin-1', low_memory=False)
            df = df.dropna(subset=['skill_id'])
            df['skill_id'] = df['skill_id'].astype(int)
            skills = df[['skill_id', 'skill_name']].drop_duplicates()
            raw_names = dict(zip(skills['skill_id'], skills['skill_name']))
            for cidx in range(num_c):
                orig_id = inv_map.get(cidx)
                if orig_id is not None and orig_id in raw_names:
                    concept_names[cidx] = str(raw_names[orig_id])
                else:
                    concept_names[cidx] = f'Concept_{cidx}'
        except Exception:
            for cidx in range(num_c):
                concept_names[cidx] = f'Concept_{cidx}'

    print(f"  [KnowLP] EDU-GraphRAG: {len(concept_names)} concepts, provider={provider}")

    # Initialize LLM
    from dotenv import load_dotenv
    load_dotenv()
    from simpath.llm.client import LLMClient
    llm = LLMClient(provider=provider, temperature=0.6, max_tokens=2048)

    # Check intermediate cache (descriptions from Step 1+2)
    desc_cache = cache_path.replace('.pkl', '_descriptions.pkl')
    if os.path.exists(desc_cache):
        print(f"  [KnowLP] Loading cached descriptions from {desc_cache}")
        with open(desc_cache, 'rb') as f:
            descriptions = pickle.load(f)
    else:
        # Step 1: Generate descriptions
        print(f"  [KnowLP] Step 1: Generating descriptions...", flush=True)
        descriptions = generate_initial_descriptions(concept_names, llm)

        # Step 2: TextGrad refinement
        print(f"  [KnowLP] Step 2: TextGrad refinement...", flush=True)
        descriptions = textgrad_refine(descriptions, concept_names, llm, n_iter=3)

        # Save intermediate
        with open(desc_cache, 'wb') as f:
            pickle.dump(descriptions, f)
        print(f"  [KnowLP] Descriptions cached: {desc_cache}")

    # Step 3: Extract relationships with 3-vote self-consistency
    print(f"  [KnowLP] Step 3: Relationship extraction (3-vote)...", flush=True)
    prereq, sim = extract_relationships(descriptions, concept_names, llm, n_votes=3)

    # Build graph
    g = ConceptGraph(num_c)
    g.prereq = prereq
    g.sim = sim
    g.names = concept_names
    g.descriptions = descriptions

    n_prereq = int(prereq.sum())
    n_sim = int(sim.sum()) // 2
    print(f"  [KnowLP] Graph: {n_prereq} prereq, {n_sim} sim edges")

    # Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'prereq': prereq, 'sim': sim,
            'names': concept_names, 'descriptions': descriptions,
            'n_concepts': num_c,
            'method': 'EDU-GraphRAG (TextGrad + 3-vote self-consistency)',
            'provider': provider,
        }, f)
    print(f"  [KnowLP] Cached: {cache_path}")

    return g
