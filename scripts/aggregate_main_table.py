#!/usr/bin/env python3
"""
Aggregate per-cell experiment reports into the main results table
(Table 1: tab:main in the paper).

Input layout (produced by scripts/run_experiment.py):

    outputs/experiments/DATASET_Lk_seedS/METHOD/report.json

where DATASET is junyi/assist15/ednet/assist09, k is the path length, S the
seed, and METHOD one of the eleven columns below. Each report.json contains
best_test_ep (mean EP on the 729-learner test split). This script walks
those files for every (method, dataset, L, seed) combination, then:

1.  Overrides DLELP / KnowLP with the graph-aware re-evaluation stored at
    outputs/dlelp_knowlp_graph_aware_results.json. The original report.json
    files for these two methods were generated before the graph-aware
    candidate-masking patch in
    simpath/eval/methods/dlelp.py:build_graph_candidate_mask and would
    otherwise report identical numbers for DLELP and KnowLP.

2.  Aggregates seeds 42, 123, 7 into mean +/- std per (method, dataset, L).

3.  Writes three artefacts next to the experiments folder:

       outputs/main_table.txt   fixed-width ASCII table (used in the paper)
       outputs/main_table.csv   machine-readable, one row per cell
       outputs/main_table.tex   LaTeX rows with best (bold) / 2nd (underline)

Usage:

    python scripts/aggregate_main_table.py                  (junyi/assist15/ednet)
    python scripts/aggregate_main_table.py --datasets all   (include assist09)
    python scripts/aggregate_main_table.py --metric latest  (use latest_test_ep)
    python scripts/aggregate_main_table.py --out_dir DIR    (custom output dir)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = REPO_ROOT / 'outputs' / 'experiments'
OVERRIDE_PATH = REPO_ROOT / 'outputs' / 'dlelp_knowlp_graph_aware_results.json'

# Paper layout
DATASETS_MAIN = ['junyi', 'assist15', 'ednet']
DATASETS_ALL = ['junyi', 'assist15', 'ednet', 'assist09']

DATASET_LABEL = dict([
    ('junyi',    'Junyi (39 KCs)'),
    ('assist15', 'ASSIST15 (100 KCs)'),
    ('ednet',    'EdNet (189 KCs)'),
    ('assist09', 'ASSIST09 (110 KCs)'),
])
LENGTHS = [5, 10, 20]
SEEDS = [42, 123, 7]

# Column order matches Table 1 of the paper
METHODS = [
    'Target-repeat',
    'GRU4Rec', 'SASRec',
    'PPO-vanilla',
    'CSEAL', 'GEHRL', 'DLELP', 'KnowLP',
    'EvoLearning-BC', 'EvoLearning-AWR', 'EvoLearning-DAPG',
]
DISPLAY = dict([
    ('Target-repeat',    'Target'),
    ('PPO-vanilla',      'PPO'),
    ('EvoLearning-BC',   'EVOL-BC'),
    ('EvoLearning-AWR',  'EVOL-AWR'),
    ('EvoLearning-DAPG', 'EVOL-DAPG'),
])
OVERRIDE_METHODS = set(['DLELP', 'KnowLP'])


def load_override(path):
    """Parse the flat override file into a dict keyed by tuple.

    Keys are METHOD_DATASET_Lk_sS. Dataset and method names never contain an
    underscore in this project, so a 4-way split is unambiguous.
    """
    if not path.exists():
        return dict()
    with open(path) as f:
        flat = json.load(f)
    out = dict()
    for k, v in flat.items():
        try:
            method, dataset, L_tok, s_tok = k.split('_')
            out[(method, dataset, int(L_tok[1:]), int(s_tok[1:]))] = float(v)
        except (ValueError, IndexError):
            print('  WARN: skipping malformed override key: ' + k, file=sys.stderr)
    return out


def load_cell(method, dataset, L, seed, metric):
    """Return the requested EP from a single cell, or None if missing."""
    p = EXP_ROOT / ('%s_L%d_seed%d' % (dataset, L, seed)) / method / 'report.json'
    if not p.exists():
        return None
    with open(p) as f:
        r = json.load(f)
    key = metric + '_test_ep'
    return float(r[key]) if key in r else None


def aggregate(method, dataset, L, override, metric):
    """Return (mean, std, n_seeds, used_seeds) for one cell, or None."""
    vals, seeds_used = [], []
    for s in SEEDS:
        v = override.get((method, dataset, L, s)) if method in OVERRIDE_METHODS else None
        if v is None:
            v = load_cell(method, dataset, L, s, metric)
        if v is not None:
            vals.append(v)
            seeds_used.append(s)
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals)), len(vals), seeds_used


def build_table(datasets, override, metric):
    table = dict()
    for ds in datasets:
        for L in LENGTHS:
            for m in METHODS:
                a = aggregate(m, ds, L, override, metric)
                if a is not None:
                    table[(m, ds, L)] = a
    return table


def _fmt_cell(mu, sd):
    sign = '+' if mu == abs(mu) else '-'
    return '%s%.3f+/-0.%s' % (sign, abs(mu), ('%.6f' % sd).split('.')[1][:3])


def render_txt(table, datasets):
    """Fixed-width ASCII table; one block per dataset."""
    NL = chr(10)
    cell_w = max(14, max(len(DISPLAY.get(m, m)) for m in METHODS) + 1)
    header = ('Dataset'.ljust(20) + ' ' + 'L'.rjust(3) + ' '
              + ' '.join(DISPLAY.get(m, m).rjust(cell_w) for m in METHODS))
    bar = ''.ljust(len(header), '-')
    lines = [bar, header, bar]
    for ds in datasets:
        for L in LENGTHS:
            label = DATASET_LABEL[ds] if L == LENGTHS[0] else ''
            row = [label.ljust(20), str(L).rjust(3)]
            for m in METHODS:
                a = table.get((m, ds, L))
                cell = _fmt_cell(a[0], a[1]) if a else '--'
                row.append(cell.rjust(cell_w))
            lines.append(' '.join(row))
        lines.append('')
    lines.append(bar)
    lines.append('Methods: %d   Cells filled: %d   Seeds: %s'
                 % (len(METHODS), len(table), SEEDS))
    return NL.join(lines) + NL


def render_csv(table, datasets):
    NL = chr(10)
    rows = ['dataset,L,method,mean,std,n_seeds']
    for ds in datasets:
        for L in LENGTHS:
            for m in METHODS:
                a = table.get((m, ds, L))
                if a is None:
                    continue
                mu, sd, n, _ = a
                rows.append('%s,%d,%s,%.6f,%.6f,%d' % (ds, L, m, mu, sd, n))
    return NL.join(rows) + NL


def render_latex(table, datasets):
    """LaTeX rows matching tab:main; best in bold, 2nd best underlined."""
    DOLLAR = chr(36)
    STAR = chr(42)
    AMP = chr(38)
    BSL = chr(92)
    NL = chr(10)
    pm_open = '{' + BSL + 'scriptsize' + DOLLAR + BSL + 'pm' + DOLLAR + '.'
    neg = DOLLAR + '-' + DOLLAR
    sep = ' ' + AMP + ' '
    eol = ' ' + BSL + BSL
    multirow = BSL + 'multirow{3}{' + STAR + '}{'
    midrule = BSL + 'midrule'

    def fmt(mu, sd):
        num = ('%+.3f' % mu).replace('-', neg)
        sd_str = ('%.6f' % sd).split('.')[1][:3]
        return num + pm_open + sd_str + '}'

    rows = []
    for ds in datasets:
        for i, L in enumerate(LENGTHS):
            mus = dict()
            for m in METHODS:
                key = (m, ds, L)
                mus[m] = table[key][0] if key in table else -1e9
            ranked = sorted(mus.items(), key=lambda x: x[1], reverse=True)
            best_m, second_m = ranked[0][0], ranked[1][0]
            cells = []
            for m in METHODS:
                a = table.get((m, ds, L))
                if a is None:
                    cells.append('--')
                    continue
                cell = fmt(a[0], a[1])
                if m == best_m:
                    cell = BSL + 'textbf{' + cell + '}'
                elif m == second_m:
                    cell = BSL + 'underline{' + cell + '}'
                cells.append(cell)
            prefix = (multirow + DATASET_LABEL[ds] + '}') if i == 0 else ''
            rows.append(prefix + sep + str(L) + sep + sep.join(cells) + eol)
        rows.append(midrule)
    if rows and rows[-1] == midrule:
        rows.pop()
    return NL.join(rows) + NL


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', default='paper', choices=['paper', 'all'],
                    help='paper = junyi, assist15, ednet (default). '
                         'all also includes assist09.')
    ap.add_argument('--metric', default='best', choices=['best', 'latest'],
                    help='Which checkpoint to read from report.json.')
    ap.add_argument('--out_dir', default=str(REPO_ROOT / 'outputs'),
                    help='Directory to write the three main_table files.')
    args = ap.parse_args()

    datasets = DATASETS_MAIN if args.datasets == 'paper' else DATASETS_ALL
    override = load_override(OVERRIDE_PATH)
    print('  Override entries loaded: %d (%s)'
          % (len(override), OVERRIDE_PATH.name))

    table = build_table(datasets, override, args.metric)
    missing = [(m, ds, L) for ds in datasets for L in LENGTHS for m in METHODS
               if (m, ds, L) not in table]
    if missing:
        print('  WARN: %d cells missing. first 5: %s'
              % (len(missing), missing[:5]), file=sys.stderr)

    txt = render_txt(table, datasets)
    csv_str = render_csv(table, datasets)
    tex = render_latex(table, datasets)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'main_table.txt').write_text(txt)
    (out_dir / 'main_table.csv').write_text(csv_str)
    (out_dir / 'main_table.tex').write_text(tex)

    print()
    print(txt)
    print('Wrote: ' + str(out_dir / 'main_table.txt'))
    print('       ' + str(out_dir / 'main_table.csv'))
    print('       ' + str(out_dir / 'main_table.tex'))


if __name__ == '__main__':
    main()
