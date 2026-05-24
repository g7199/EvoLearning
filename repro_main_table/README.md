# Main Results Table -- Self-Contained Reproduction

This folder reproduces Table 1 (tab:main) of the paper and is fully
self-contained: the script reads only files inside this folder and writes
only inside this folder. It never touches the rest of the repository or
any external path, and it has no third-party dependencies (pure Python 3).

## Contents

- aggregate_main_table.py    table generator (pure Python, stdlib only)
- reports/                   bundled per-run scalars: one report.json per
                             reports/DATASET_Lk_seedS/METHOD/ holding
                             best_test_ep and latest_test_ep. 243 files
                             for the three paper datasets x 3 path lengths
                             x 3 seeds x 9 methods (plus partial ASSIST09).
- dlelp_knowlp_graph_aware_results.json   per-seed EP for DLELP and KnowLP
                             from the graph-aware re-evaluation (override).
- main_table.txt             generated fixed-width ASCII table
- main_table.csv             generated, one row per (dataset, L, method)
- main_table.tex             generated LaTeX rows (best bold, 2nd under).

## How to run

No installation is required (pure standard library). From the repo root:

    python3 repro_main_table/aggregate_main_table.py

Optionally, run it inside an isolated virtual environment -- there is
still nothing to install:

    python3 -m venv .venv
    .venv/bin/python repro_main_table/aggregate_main_table.py

Options:

    --metric best      use best-checkpoint EP (default)
    --metric latest    use latest-checkpoint EP
    --out_dir DIR      where to write the outputs (default: this folder)

## What it does

1. Walks reports/DATASET_Lk_seedS/METHOD/report.json and reads
   best_test_ep for every (method, dataset, L, seed).
2. Overrides DLELP and KnowLP with dlelp_knowlp_graph_aware_results.json.
   The original runs for these two predate the graph-aware candidate
   masking in simpath/eval/methods/dlelp.py and would otherwise be equal.
3. Aggregates seeds 42, 123, 7: the mean is rounded and the population
   std is truncated to three decimals, matching the published table.

## Reproduction accuracy

All 99 cell means reproduce the paper exactly. Standard deviations match
for 88 of 99 cells; the 11 DLELP/KnowLP cells differ by at most 0.001 EP
because the override file stores per-seed EP at four decimals.

## Relation to the live pipeline

scripts/aggregate_main_table.py is the same generator wired to the live
outputs/experiments tree (gitignored, produced by run_experiment.py) and
uses numpy. This folder is a frozen, committed snapshot of those inputs so
the table regenerates from a clean checkout with no setup.
