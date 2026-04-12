#!/usr/bin/env python3
"""
EvoLearning — Full Setup Pipeline.
Run this ONCE before experiments to prepare everything from scratch.

Steps:
  1. Preprocess datasets (ASSIST09, Junyi)
  2. Train DKT models (KES)
  3. Build DKT influence graphs
  4. Generate Evo expert trajectories (DPP-5)
  5. Build KnowLP EDU-GraphRAG graph (requires OpenAI API key)

Usage:
  python scripts/setup_pipeline.py --dataset assist09 --gpu 0
  python scripts/setup_pipeline.py --dataset junyi --gpu 1
  python scripts/setup_pipeline.py --dataset all --gpu 0
"""
import sys, os, argparse, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch


def setup_assist09(gpu):
    """Full pipeline for ASSIST09 dataset."""
    device = gpu if gpu == 'cpu' else f'cuda:{gpu}'
    print(f"\n{'='*60}")
    print(f"  ASSIST09 Setup Pipeline (GPU:{gpu})")
    print(f"{'='*60}")

    # Step 1: Check raw data
    raw_path = 'data/raw/assist09/skill_builder_data_corrected.csv'
    if not os.path.exists(raw_path):
        print(f"\n  ERROR: Raw data not found at {raw_path}")
        print(f"  Download from: https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data")
        print(f"  Place skill_builder_data_corrected.csv in data/raw/assist09/")
        return False

    # Step 2: Preprocess
    proc_path = 'data/processed/assist09/assist09_processed.pkl'
    if os.path.exists(proc_path):
        print(f"\n  [1/5] Preprocessed data exists: {proc_path}")
    else:
        print(f"\n  [1/5] Preprocessing ASSIST09...")
        from simpath.data.preprocess import preprocess_assist09
        preprocess_assist09(raw_path, out_dir='data/processed/assist09')

    # Step 3: Train DKT
    dkt_path = 'outputs/checkpoints/pykt_dkt_best_assist09.pt'
    if os.path.exists(dkt_path):
        print(f"  [2/5] DKT model exists: {dkt_path}")
    else:
        print(f"  [2/5] Training DKT on ASSIST09...")
        print(f"         This requires pykt. Run: python scripts/train_dkt_assist09.py --device {device}")
        return False

    # Step 4: Build DKT influence graph
    graph_path = 'outputs/concept_graph_dkt_assist09.pkl'
    if os.path.exists(graph_path):
        print(f"  [3/5] DKT graph exists: {graph_path}")
    else:
        print(f"  [3/5] Building DKT influence graph...")
        _build_dkt_graph('assist09', device)

    # Step 5: Generate Evo experts (DPP-5)
    evo_path = 'outputs/evo_dpk5_dktgraph.pkl'
    if os.path.exists(evo_path):
        print(f"  [4/5] Evo experts exist: {evo_path}")
    else:
        print(f"  [4/5] Generating Evo experts (this takes ~1 hour)...")
        _generate_evo_experts('assist09', device)

    # Step 6: KnowLP graph (optional, needs API key)
    knowlp_path = 'outputs/knowlp_graph_assist09.pkl'
    if os.path.exists(knowlp_path):
        print(f"  [5/5] KnowLP graph exists: {knowlp_path}")
    else:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            if os.environ.get('OPENAI_API_KEY'):
                print(f"  [5/5] Building KnowLP EDU-GraphRAG graph (~15 min, API calls)...")
                from simpath.eval.knowlp_graph import build_edu_graphrag
                from simpath.eval.config import get_dataset_config
                build_edu_graphrag(get_dataset_config('assist09'), provider='openai')
            else:
                print(f"  [5/5] SKIP: KnowLP graph (no OPENAI_API_KEY in .env)")
        except Exception as e:
            print(f"  [5/5] SKIP: KnowLP graph ({e})")

    print(f"\n  ASSIST09 setup complete!")
    return True


def setup_junyi(gpu):
    """Full pipeline for Junyi dataset."""
    device = gpu if gpu == 'cpu' else f'cuda:{gpu}'
    print(f"\n{'='*60}")
    print(f"  Junyi Setup Pipeline (GPU:{gpu})")
    print(f"{'='*60}")

    # Step 1: Check raw data
    raw_path = 'data/raw/junyi/junyi_extracted/junyi_ProblemLog_original.csv'
    if not os.path.exists(raw_path):
        print(f"\n  ERROR: Raw data not found at {raw_path}")
        print(f"  Download from: https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198")
        return False

    # Step 2: Preprocess
    proc_path = 'data/processed/junyi/junyi_processed.pkl'
    if os.path.exists(proc_path):
        print(f"\n  [1/5] Preprocessed data exists: {proc_path}")
    else:
        print(f"\n  [1/5] Preprocessing Junyi...")
        os.system(f'{sys.executable} scripts/run_junyi_pipeline.py preprocess')

    # Step 3: Train DKT
    dkt_path = 'outputs/checkpoints/pykt_dkt_best_junyi.pt'
    if os.path.exists(dkt_path):
        print(f"  [2/5] DKT model exists: {dkt_path}")
    else:
        print(f"  [2/5] Training DKT on Junyi...")
        os.system(f'{sys.executable} scripts/train_dkt_junyi.py --device {device}')

    # Step 4: Build DKT influence graph
    graph_path = 'outputs/concept_graph_dkt_junyi.pkl'
    if os.path.exists(graph_path):
        print(f"  [3/5] DKT graph exists: {graph_path}")
    else:
        print(f"  [3/5] Building DKT influence graph...")
        _build_dkt_graph('junyi', device)

    # Step 5: Generate Evo experts
    evo_path = 'outputs/evo_dpk5_junyi.pkl'
    if os.path.exists(evo_path):
        print(f"  [4/5] Evo experts exist: {evo_path}")
    else:
        print(f"  [4/5] Generating Evo experts (this takes ~1 hour)...")
        _generate_evo_experts('junyi', device)

    # Step 6: KnowLP graph
    knowlp_path = 'outputs/knowlp_graph_junyi.pkl'
    if os.path.exists(knowlp_path):
        print(f"  [5/5] KnowLP graph exists: {knowlp_path}")
    else:
        print(f"  [5/5] SKIP: KnowLP graph for Junyi (not yet supported)")

    print(f"\n  Junyi setup complete!")
    return True


def _build_dkt_graph(dataset, device):
    """Build DKT influence graph from trained DKT model."""
    from simpath.eval.config import get_dataset_config
    from simpath.eval.kes import load_dkt
    ds = get_dataset_config(dataset)
    dkt, skill_map = load_dkt(ds, device)
    num_c = ds['num_c']

    from simpath.eval.data import load_data
    train_data, _, _ = load_data(ds)

    print(f"    Computing influence matrix ({num_c}x{num_c})...")
    influence = np.zeros((num_c, num_c), dtype=np.float32)
    n_sample = min(200, len(train_data))

    for ci in range(num_c):
        if (ci + 1) % 10 == 0:
            print(f"      Concept {ci+1}/{num_c}...", flush=True)
        for si in range(n_sample):
            hc, hr, _ = train_data[si]
            if not hc:
                continue
            q = torch.tensor([hc[-200:]], dtype=torch.long, device=device)
            r = torch.tensor([hr[-200:]], dtype=torch.long, device=device)
            with torch.no_grad():
                m_before = dkt(q, r)[0, -1, :].cpu().numpy()
            q2 = torch.tensor([hc[-200:] + [ci]], dtype=torch.long, device=device)
            r2 = torch.tensor([hr[-200:] + [1]], dtype=torch.long, device=device)
            with torch.no_grad():
                m_after = dkt(q2, r2)[0, -1, :].cpu().numpy()
            influence[ci] += (m_after - m_before)
        influence[ci] /= n_sample

    # Get concept names
    inv_map = {v: k for k, v in skill_map.items()}
    names = {}
    try:
        if dataset == 'assist09':
            import pandas as pd
            df = pd.read_csv('data/raw/assist09/skill_builder_data_corrected.csv',
                             encoding='latin-1', low_memory=False)
            df = df.dropna(subset=['skill_id']); df['skill_id'] = df['skill_id'].astype(int)
            raw = dict(zip(df['skill_id'], df['skill_name']))
            for cidx in range(num_c):
                oid = inv_map.get(cidx)
                names[cidx] = str(raw.get(oid, f'Concept_{cidx}')) if oid else f'Concept_{cidx}'
        else:
            with open(ds['data_path'], 'rb') as f:
                data = pickle.load(f)
            for kc in data['kc_list']:
                if kc in skill_map:
                    names[skill_map[kc]] = kc
    except Exception:
        pass

    os.makedirs('outputs', exist_ok=True)
    with open(ds['graph_dkt_path'], 'wb') as f:
        pickle.dump({'influence': influence, 'names': names,
                     'n_concepts': num_c, 'dataset': dataset}, f)
    print(f"    Saved: {ds['graph_dkt_path']}")


def _generate_evo_experts(dataset, device):
    """Generate DPP-5 diverse Evo expert trajectories."""
    import heapq
    from simpath.eval.config import get_dataset_config
    from simpath.eval.kes import KES, load_dkt
    from simpath.eval.data import load_data
    from simpath.eval.graph import load_graph

    ds = get_dataset_config(dataset)
    dkt, _ = load_dkt(ds, device)
    kes = KES(dkt, ds['num_c'], device)
    train_data, _, _ = load_data(ds)
    graph = load_graph(ds, 'dkt')

    num_c = ds['num_c']; L = 5; POP = 60; GEN = 50; CAP = 30

    def get_candidates(targets, mastery):
        r = set()
        for t in targets:
            r.add(t)
            for p in graph.get_prerequisites(t): r.add(p)
            for s in graph.get_similar(t, top_k=5): r.add(s)
        if len(r) < CAP:
            for c in sorted(range(num_c), key=lambda c: abs(mastery[c] - 0.5)):
                r.add(c);
                if len(r) >= CAP: break
        return list(r)[:CAP]

    def evo_single(hc, hr, targets):
        m0 = kes.mastery(hc, hr)
        es = m0[targets]
        cands = get_candidates(targets, m0)
        bl = min(len(hc), 200)

        # Pre-allocate for batched eval
        q_base = torch.zeros(1, bl, dtype=torch.long, device=device)
        r_base = torch.zeros(1, bl, dtype=torch.long, device=device)
        if bl > 0:
            q_base[0, :bl] = torch.tensor(hc[-200:], dtype=torch.long)
            r_base[0, :bl] = torch.tensor(hr[-200:], dtype=torch.long)

        pops = np.zeros((POP, L), dtype=np.int64)
        for p in range(POP):
            if len(cands) >= L:
                pops[p] = np.random.choice(cands, L, replace=False)
            else:
                pops[p, :len(cands)] = cands

        total_len = bl + L
        hist_q = torch.zeros(POP, total_len, dtype=torch.long, device=device)
        hist_r = torch.zeros(POP, total_len, dtype=torch.long, device=device)
        if bl > 0:
            hist_q[:, :bl] = q_base.expand(POP, -1)
            hist_r[:, :bl] = r_base.expand(POP, -1)
        tgt_t = torch.tensor(targets, dtype=torch.long, device=device)

        def eval_fit(pops):
            hist_q[:, bl:] = 0; hist_r[:, bl:] = 0
            pops_t = torch.tensor(pops, dtype=torch.long, device=device)
            for step in range(L):
                cur = bl + step
                with torch.no_grad():
                    out = dkt(hist_q[:, :cur], hist_r[:, :cur]) if cur > 0 else None
                if out is not None:
                    pc = out[torch.arange(POP, device=device), cur-1, pops_t[:, step]]
                else:
                    pc = torch.full((POP,), 0.5, device=device)
                hist_q[:, cur] = pops_t[:, step]
                hist_r[:, cur] = (pc > 0.5).long()
            with torch.no_grad():
                out_f = dkt(hist_q[:, :total_len], hist_r[:, :total_len])
            ee = out_f[:, total_len-1, tgt_t]
            es_t = torch.tensor(es, dtype=torch.float32, device=device).unsqueeze(0)
            denom = (1.0 - es_t).clamp(min=0.01)
            ep = ((ee - es_t) / denom).mean(1).cpu().numpy()
            return ep

        fit = eval_fit(pops)
        for gen in range(GEN):
            new = np.zeros_like(pops)
            i1 = np.random.randint(POP, size=POP); i2 = np.random.randint(POP, size=POP)
            new = pops[np.where(fit[i1] > fit[i2], i1, i2)].copy()
            for i in range(0, POP-1, 2):
                if np.random.random() < 0.8:
                    cx = np.random.randint(1, L)
                    ch = list(new[i, :cx])
                    for c in new[i+1]:
                        if c not in ch and len(ch) < L: ch.append(c)
                    while len(ch) < L:
                        c = np.random.choice(cands)
                        if c not in ch: ch.append(c)
                    new[i] = ch[:L]
            for i in range(POP):
                if np.random.random() < 0.3:
                    pos = np.random.randint(L); nc = np.random.choice(cands)
                    if nc not in new[i]: new[i, pos] = nc
            new[0] = pops[np.argmax(fit)]
            pops = new; fit = eval_fit(pops)

        # DPP-5 selection
        pl = [pops[p].tolist() for p in range(POP)]
        sel = []; rem = list(range(POP))
        for _ in range(5):
            bi, bs = -1, -float('inf')
            for idx in rem:
                s = fit[idx] if not sel else fit[idx] - 0.5 * max(
                    len(set(pl[idx]) & set(pl[s])) / L for s in sel)
                if s > bs: bs = s; bi = idx
            if bi >= 0: sel.append(bi); rem.remove(bi)
        return [(m0, targets, pl[i], fit[i], hc, hr) for i in sel]

    t0 = time.time(); all_experts = []
    for si, (hc, hr, tgts) in enumerate(train_data):
        all_experts.extend(evo_single(hc, hr, tgts))
        if (si+1) % 50 == 0:
            ep = np.mean([e[3] for e in all_experts])
            print(f"    {si+1}/{len(train_data)} | EP={ep:+.4f} | "
                  f"{(si+1)/(time.time()-t0):.2f} st/s", flush=True)

    os.makedirs('outputs', exist_ok=True)
    with open(ds['evo_path'], 'wb') as f:
        pickle.dump(all_experts, f)
    print(f"    Saved {len(all_experts)} experts to {ds['evo_path']}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="EvoLearning Setup Pipeline")
    p.add_argument('--dataset', default='all', choices=['assist09', 'junyi', 'all'])
    p.add_argument('--gpu', default='0')
    args = p.parse_args()

    if args.dataset in ('assist09', 'all'):
        setup_assist09(args.gpu)
    if args.dataset in ('junyi', 'all'):
        setup_junyi(args.gpu)

    print(f"\n{'='*60}")
    print(f"  Setup complete! Run experiments with:")
    print(f"  python scripts/run_experiment.py --dataset assist09 --method all --L 5 --seed 42 --gpu 0,1")
    print(f"{'='*60}")
