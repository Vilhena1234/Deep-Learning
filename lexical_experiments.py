"""
Lexical baseline experiments (Person 2 - Soni). Produces the tables and the plot in results/lexical/.

Run from the repo root, after `python data.py`:
    python scripts/lexical_experiments.py --dataset squad
    python scripts/lexical_experiments.py --dataset fiqa

What it runs (plan section 7, owner 2):
    1. Grid search over k1 and b for BM25 on the TUNING split           -> bm25_grid_<dataset>.csv + .png
    2. TF-IDF vs BM25 (default) vs BM25 (tuned) vs BM25 with b = 0       -> main_<dataset>.csv
       on the REPORT split
    3. Preprocessing ablation: stopwords / stemming on and off           -> preprocessing_<dataset>.csv

Splits (never tune on the numbers we report):
    FiQA : tune on dev, report on test      (corpus = all 57,638 passages)
    SQuAD: tune on 2,000 train questions (train corpus), report on dev (2,067 dev paragraphs)

The metrics below are a small local copy so this script runs on its own.
Once src/evaluation/metrics.py (Person 1) has Precision@k, Recall@k and MRR, switch to that.
"""

import argparse
import csv
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from src.retrievers.bm25 import BM25  # noqa: E402
from src.retrievers.tf_idf import TFIDF  # noqa: E402

KS = (1, 5, 10, 100)
SPLITS = {'fiqa': ('dev', 'test'), 'squad': ('train', 'dev')}  # (tune, report)
K1_GRID = (0.25, 0.5, 0.9, 1.2, 1.5, 2.0, 3.0)
B_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT = dict(k1=1.2, b=0.75)


def load_split(name: str, split: str, max_queries: int | None = None, seed: int = 0):
    """Reads data/processed/<name>_<split>.json written by data.py (same as data.load, full corpus)."""
    path = os.path.join('data', 'processed', f'{name}_{split}.json')
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    corpus, queries, qrels = d['corpus'], d['queries'], d['qrels']
    if max_queries is not None and len(queries) > max_queries:
        qids = random.Random(seed).sample(sorted(queries), max_queries)
        queries = {q: queries[q] for q in qids}
    return corpus, queries, qrels


def evaluate(retriever, queries: dict, qrels: dict) -> dict:
    """Precision@k, Recall@k (k in KS) and MRR, averaged over queries. MRR is cut off at rank max(KS)."""
    documents = retriever.index.GetDocuments()
    qids = list(queries)
    hits = retriever.TopK([queries[q] for q in qids], k=max(KS))
    totals = {f'P@{k}': 0.0 for k in KS} | {f'R@{k}': 0.0 for k in KS} | {'MRR': 0.0}
    for qid, ranked in zip(qids, hits):
        relevant = {doc for doc, rel in qrels[qid].items() if rel > 0}
        is_rel = [documents[j].GetId() in relevant for j, _ in ranked]
        for k in KS:
            found = sum(is_rel[:k])
            totals[f'P@{k}'] += found / k
            totals[f'R@{k}'] += found / len(relevant)
        totals['MRR'] += next((1 / (rank + 1) for rank, r in enumerate(is_rel) if r), 0.0)
    return {m: round(v / len(qids), 4) for m, v in totals.items()}


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'  wrote {path}')


def plot_grid(path: str, grid_rows: list[dict], metric: str, dataset: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for b in B_GRID:
        ys = [r[metric] for r in grid_rows if r['b'] == b]
        ax.plot(K1_GRID, ys, marker='o', label=f'b = {b}')
    ax.set_xlabel('k1 (saturation)')
    ax.set_ylabel(metric)
    ax.set_title(f'BM25 sensitivity, {dataset} (tuning split)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'  wrote {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['fiqa', 'squad'], required=True)
    parser.add_argument('--metric', default='R@10', help='metric used to pick the best k1 and b')
    parser.add_argument('--max-tune-queries', type=int, default=2000)
    parser.add_argument('--out', default=os.path.join('results', 'lexical'))
    args = parser.parse_args()

    tune_split, report_split = SPLITS[args.dataset]
    tune_corpus, tune_q, tune_qrels = load_split(args.dataset, tune_split, args.max_tune_queries)
    corpus, queries, qrels = load_split(args.dataset, report_split)
    print(f'{args.dataset}: tune on {tune_split} ({len(tune_q)} queries, {len(tune_corpus)} docs), '
          f'report on {report_split} ({len(queries)} queries, {len(corpus)} docs)')

    # 1. k1 / b grid on the tuning split
    print('\n[1] BM25 k1 / b grid')
    bm25 = BM25.from_corpus_dict(tune_corpus, **DEFAULT)
    grid = []
    for k1 in K1_GRID:
        for b in B_GRID:
            bm25.SetParameters(k1, b)
            grid.append({'k1': k1, 'b': b} | evaluate(bm25, tune_q, tune_qrels))
    best = max(grid, key=lambda r: r[args.metric])
    best_b0 = max((r for r in grid if r['b'] == 0.0), key=lambda r: r[args.metric])
    print(f'  best {args.metric}: k1={best["k1"]}, b={best["b"]} ({best[args.metric]}); '
          f'best with b=0: k1={best_b0["k1"]}')
    write_csv(os.path.join(args.out, f'bm25_grid_{args.dataset}.csv'), grid)
    plot_grid(os.path.join(args.out, f'bm25_grid_{args.dataset}.png'), grid, args.metric, args.dataset)

    # 2. Main comparison on the report split
    print('\n[2] TF-IDF vs BM25')
    rows = []
    t = time.time()
    rows.append({'method': 'TF-IDF', 'k1': '', 'b': ''} | evaluate(TFIDF.from_corpus_dict(corpus), queries, qrels))
    bm25 = BM25.from_corpus_dict(corpus, **DEFAULT)
    for label, k1, b in [('BM25 default', DEFAULT['k1'], DEFAULT['b']),
                         ('BM25 tuned', best['k1'], best['b']),
                         ('BM25 b=0 (no length norm.)', best_b0['k1'], 0.0)]:
        bm25.SetParameters(k1, b)
        rows.append({'method': label, 'k1': k1, 'b': b} | evaluate(bm25, queries, qrels))
    for r in rows:
        print(f'  {r["method"]:28s} R@10={r["R@10"]:.3f}  MRR={r["MRR"]:.3f}')
    print(f'  ({time.time() - t:.0f} s)')
    write_csv(os.path.join(args.out, f'main_{args.dataset}.csv'), rows)

    # 3. Preprocessing ablation on the report split (tuned BM25 and TF-IDF)
    print('\n[3] Preprocessing ablation')
    rows = []
    for stop in (False, True):
        for stem in (False, True):
            opts = dict(remove_stopwords=stop, stem=stem)
            rows.append({'method': 'TF-IDF', 'stopwords removed': stop, 'stemming': stem}
                        | evaluate(TFIDF.from_corpus_dict(corpus, **opts), queries, qrels))
            rows.append({'method': 'BM25 tuned', 'stopwords removed': stop, 'stemming': stem}
                        | evaluate(BM25.from_corpus_dict(corpus, k1=best['k1'], b=best['b'], **opts), queries, qrels))
            print(f'  stopwords={stop!s:5} stem={stem!s:5}  '
                  f'TF-IDF R@10={rows[-2]["R@10"]:.3f}  BM25 R@10={rows[-1]["R@10"]:.3f}')
    write_csv(os.path.join(args.out, f'preprocessing_{args.dataset}.csv'), rows)


if __name__ == '__main__':
    main()
