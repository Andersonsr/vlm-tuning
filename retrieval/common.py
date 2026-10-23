"""
Shared protocol of the retrieval benchmarks (text_retrieval.py, image_retrieval.py, crossmodal_retrieval.py): the label
classes of the samples and the ranking metrics.

Every distinct combination of labels of a description type (campo) is a class:
    exact     the positives are the items with exactly the same labels; R@K is the share of queries with a positive in
              the top K, mAP the mean average precision. Only queries with a positive count.
    graded    nDCG@10 with relevance = Jaccard overlap of the two label sets (as (category, value) pairs), so items
              sharing most labels count too; every item is a query.
"""
import os

import numpy as np

# the dinov3 repository (parent of vlm-tuning): geo, its split jsons and the torch hub entry points
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DEFAULT_INPUT = os.path.join(REPO, 'geo', 'eval', 'conv_test.json')
KS = (1, 5, 10)
METRICS = ('R@1', 'R@5', 'R@10', 'mAP', 'nDCG@10')


def normalize_labels(labels):
    """Stripped, deduplicated and sorted values; categories without values dropped (as geo/eval/judge_reliability)."""
    labels = {c: sorted({v.strip() for v in vs if v.strip()}) for c, vs in labels.items()}
    return {c: vs for c, vs in labels.items() if vs}


def image_path(root, sample):
    # as geo dataset.ConversationDataset
    if 'cd_guid' in sample:
        return os.path.join(root, '{}.png'.format(sample['cd_guid']))
    if 'image_name' in sample:
        return os.path.join(root, sample['image_name'].replace('\\', '/'))
    if 'image_id' in sample:
        return os.path.join(root, '{}.png'.format(sample['image_id']))
    raise ValueError('sample without image: {}'.format(sorted(sample)))


def jaccard_matrix(label_sets, others=None):
    """Pairwise Jaccard overlap of the label sets, [n, n], or against others, [n, m] (e.g. images vs descriptions)."""
    others = label_sets if others is None else others
    vocab = {pair: i for i, pair in enumerate(sorted(set().union(*label_sets, *others)))}

    def onehot(sets):
        out = np.zeros((len(sets), len(vocab)), dtype=np.float32)
        for row, labels in enumerate(sets):
            out[row, [vocab[p] for p in labels]] = 1
        return out

    a, b = onehot(label_sets), onehot(others)
    inter = a @ b.T
    return inter / (a.sum(1)[:, None] + b.sum(1)[None, :] - inter)


def ranking_metrics(sim, query_keys, gallery_keys, jaccard, excluded=None):
    """Exact-class R@K and mAP, and Jaccard-graded nDCG@10, of each query's ranking of the gallery by sim [nq, ng]
    (same or another modality). jaccard: [nq, ng] label overlap; excluded: optional [nq, ng] boolean mask of pairs left
    out of the rankings. R@K and mAP count only the queries with a positive (same labels) in their gallery."""
    sim = np.array(sim, copy=True)  # the caller's matrix is left as it is
    if excluded is not None:
        sim[excluded] = -np.inf
    order = np.argsort(-sim, axis=1)
    query_keys, gallery_keys = np.array(query_keys), np.array(gallery_keys)

    hits = {k: [] for k in KS}
    aps, ndcgs = [], []
    for q in range(len(query_keys)):
        ranked = order[q] if excluded is None else order[q][~excluded[q, order[q]]]
        relevant = gallery_keys[ranked] == query_keys[q]
        if relevant.any():
            for k in KS:
                hits[k].append(relevant[:k].any())
            precision = np.cumsum(relevant) / np.arange(1, len(relevant) + 1)
            aps.append(float((precision * relevant).sum() / relevant.sum()))
        gains = jaccard[q, ranked]
        top = min(10, len(gains))
        discounts = 1 / np.log2(np.arange(2, top + 2))
        ideal = (np.sort(gains)[::-1][:top] * discounts).sum()
        ndcgs.append(float((gains[:top] * discounts).sum() / ideal) if ideal > 0 else 0.0)
    out = {f'R@{k}': float(np.mean(v)) if v else float('nan') for k, v in hits.items()}
    out.update({'mAP': float(np.mean(aps)) if aps else float('nan'), 'nDCG@10': float(np.mean(ndcgs)),
                'queries': len(aps), 'items': len(query_keys)})
    return out


def retrieval_metrics(embeddings, keys, jaccard, exclude=None):
    """ranking_metrics of every item against all the others (same modality). exclude: optional [n, n] boolean mask of
    further pairs left out of each query's ranking (e.g. the same image under other labels, or images of the same
    slide); the query itself is always left out."""
    x = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    excluded = np.eye(len(keys), dtype=bool) if exclude is None else (exclude | np.eye(len(keys), dtype=bool))
    return ranking_metrics(x @ x.T, keys, keys, jaccard, excluded)
