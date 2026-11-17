"""
Distribution of the pairwise Jaccard overlap of the label sets of one campo, over the test and val splits merged: the
image pairs image_retrieval.py ranks (distinct (image, labels) pairs, images smaller than --size skipped, pairs of the
same image left out), i.e. the graded relevance of its nDCG@10.

Usage (from vlm-tuning):
    python retrieval/plot_jaccard.py --out jaccard_distribution.png
"""
import os
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import jaccard_matrix  # noqa: E402
from image_retrieval import exclusion_mask, load_items  # noqa: E402

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--splits', nargs='+', default=[os.path.join(HERE, 'conv_test.json'), os.path.join(HERE, 'conv_val.json')])
    parser.add_argument('--campo', default='composition')
    parser.add_argument('--size', type=int, default=256, help='as image_retrieval.py: smaller images skipped')
    parser.add_argument('--out', default='jaccard_distribution.png')
    args = parser.parse_args()

    items = []
    for split in args.splits:
        items += load_items(split, '', [args.campo], min_size=args.size)[0][args.campo]
    jac = jaccard_matrix([it['labels'] for it in items])
    values = jac[np.triu(~exclusion_mask(items, False), k=1)]
    print(f'{args.campo}: images {len(items)}  pairs {len(values)}  mean {values.mean():.3f}  '
          f'J=0 {np.mean(values == 0):.1%}  J=1 {np.mean(values == 1):.1%}')

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(values, bins=np.linspace(0, 1, 21), weights=np.full(len(values), 1 / len(values)))
    ax.set_title(f'{args.campo} (n={len(items)} images, mean {values.mean():.2f})')
    ax.set_xlabel('Jaccard similarity')
    ax.set_ylabel('share of image pairs')
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'saved {args.out}')
