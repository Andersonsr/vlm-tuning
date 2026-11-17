"""
Examples of text-to-text retrieval for a qualitative review: the --n best and --n worst queries of one encoder on one
description type (campo), each with its --top_k first retrieved descriptions, saved to a json for examples_docx.py.

Same protocol as text_retrieval.py: each distinct (description, labels) pair of the campo is a query against all the
others. Queries are ranked by average precision (positives: exactly the same labels), ties by nDCG@10 (relevance:
Jaccard overlap of the labels); only queries with a positive are picked, and at most one per label combination, so the
examples are not variants of the same case.

Usage (from vlm-tuning):
    python retrieval/retrieval_examples.py --input conv_test.json --campo composition --model hf:e5-large
    python retrieval/examples_docx.py conv_test_composition_e5-large_examples.json composition_examples.docx
"""
import json
import os
import re
import sys
from argparse import ArgumentParser

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_retrieval as T  # noqa: E402


def encoder_description(spec):
    kind, _, name = spec.partition(':')
    if kind == 'hf':
        model_id, pooling, prefix = T.PRESETS.get(name, (name, 'mean', ''))
        return f'{model_id} ({pooling} pooling' + (f', "{prefix.strip()}" prefix)' if prefix else ')')
    return spec


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', required=True, help='split json with conversations and labels (e.g. conv_test.json)')
    parser.add_argument('--campo', default='composition', help='description type')
    parser.add_argument('--model', default='hf:e5-large', help='encoder, as in text_retrieval.py --models')
    parser.add_argument('--n', default=5, type=int, help='number of best and of worst queries')
    parser.add_argument('--top_k', default=5, type=int, help='retrieved descriptions listed per query')
    parser.add_argument('--relevant', default=0, type=int, choices=range(6), metavar='{0..5}',
                        help='also list up to this many descriptions with the same labels as the query that are not '
                             'among the top_k, with the rank the encoder gave them')
    parser.add_argument('--out', default=None, help='json (default: <input>_<campo>_<model>_examples.json)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_length', default=None, type=int, help='token limit of hf encoders (default: the model\'s)')
    parser.add_argument('--checkpoint', default='best', help='vlm: runs: best, last, none or a checkpoint path')
    parser.add_argument('--bpe', default=None, help='BPE vocabulary for dinotxt (default: downloaded)')
    parser.add_argument('--hub_repo', default='facebookresearch/dinov3')
    parser.add_argument('--hub_source', default='github', choices=['github', 'local'])
    args = parser.parse_args()

    items = T.load_items(args.input, [args.campo]).get(args.campo)
    if not items:
        raise ValueError(f'no descriptions of campo {args.campo} with labels in {args.input}')
    texts = [t for t, _, _ in items]
    keys = np.array([k for _, k, _ in items])
    jac = T.jaccard_matrix([s for _, _, s in items])

    e = T.build_encoder(args.model, args).encode(texts)
    agg = T.retrieval_metrics(e, list(keys), jac)
    print('aggregate (same as text_retrieval.py):', {k: round(v, 3) for k, v in agg.items()})

    x = e / np.linalg.norm(e, axis=1, keepdims=True).clip(min=1e-12)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)[:, :-1]  # the query itself (-inf) is last
    disc = 1 / np.log2(np.arange(2, 12))
    per = []
    for q in range(len(items)):
        ranked = order[q]
        rel = keys[ranked] == keys[q]
        if not rel.any():
            continue
        gains = jac[q, ranked]
        ideal = (np.sort(gains)[::-1][:10] * disc).sum()
        ndcg = float((gains[:10] * disc).sum() / ideal) if ideal > 0 else 0.0
        ap = float(((np.cumsum(rel) / np.arange(1, len(rel) + 1)) * rel).sum() / rel.sum())
        per.append({'q': q, 'ap': ap, 'ndcg': ndcg, 'first_exact_rank': int(np.argmax(rel)) + 1,
                    'n_exact': int(rel.sum())})

    def pick(sorted_rows):
        chosen, combos = [], set()
        for r in sorted_rows:
            if keys[r['q']] in combos:
                continue
            chosen.append(r)
            combos.add(keys[r['q']])
            if len(chosen) == args.n:
                break
        return chosen

    best = pick(sorted(per, key=lambda r: (-r['ap'], -r['ndcg'], -r['n_exact'])))
    worst = pick(sorted(per, key=lambda r: (r['ap'], r['ndcg'])))

    def entry(r, group):
        q = r['q']
        # relevant: descriptions with the same labels outside the top_k listed, in the order the encoder ranked them
        relevant = [(i + 1, j) for i, j in enumerate(order[q]) if keys[j] == keys[q] and i >= args.top_k]
        return {'group': group, 'query': texts[q], 'labels': json.loads(keys[q]), 'ap': r['ap'], 'ndcg10': r['ndcg'],
                'first_exact_rank': r['first_exact_rank'], 'n_exact': r['n_exact'],
                'retrieved': [{'rank': i + 1, 'text': texts[j], 'labels': json.loads(keys[j]),
                               'cosine': float(sim[q, j]), 'jaccard': float(jac[q, j]), 'exact': bool(keys[j] == keys[q])}
                              for i, j in enumerate(order[q][:args.top_k])],
                'relevant': [{'rank': rank, 'text': texts[j], 'cosine': float(sim[q, j])}
                             for rank, j in relevant[:args.relevant]]}

    res = {'model': args.model, 'encoder': encoder_description(args.model), 'split': os.path.basename(args.input),
           'campo': args.campo, 'n_items': len(items), 'n_queries_with_match': len(per), 'aggregate': agg,
           'samples': [entry(r, 'best') for r in best] + [entry(r, 'worst') for r in worst]}

    tag = re.sub(r'[^\w.-]+', '_', args.model.split(':', 1)[-1].replace('/config.yaml', ''))[-60:]
    out = args.out or f'{os.path.splitext(args.input)[0]}_{args.campo}_{tag}_examples.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    for s in res['samples']:
        print(s['group'], f"AP {s['ap']:.3f} nDCG {s['ndcg10']:.3f} first exact @{s['first_exact_rank']} "
              f"({s['n_exact']} matches) |", s['query'][:90].replace('\n', ' '))
    print(f'saved to {out}')
