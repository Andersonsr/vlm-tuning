"""
Cross modal retrieval (I2T, optionally T2I) of trained models, with the protocol of the single modality tests
(text_retrieval.py and image_retrieval.py, shared protocol in common.py), on the same split json, so the numbers are
comparable.

Every distinct combination of labels of a description type (campo) is a class. Each image of a campo is a query
against every distinct description of the same campo (i2t), or each description against every image (t2i):
    exact     the positives are the items of the other modality with exactly the same labels; R@K is the share of
              queries with a positive in the top K, mAP the mean average precision. Only queries with a positive count.
    graded    nDCG@10 with relevance = Jaccard overlap of the two label sets (as (category, value) pairs), so items
              sharing most labels count too; every item is a query.
Each (image, labels) and (description, labels) pair is used once per campo. --exclude_same_slide leaves out of an
image's ranking the descriptions written only for its slide (its own description among them), so the positives come
from other slides.

Models: any run of trainLight.py (--configs, the config.yaml saved in each run folder), built and loaded as in
training (vlm_run.py; --checkpoint best by default). Images are center crops of --size pixels at their native
resolution (crop_transform, as in validation); images whose shorter side is smaller are skipped.

Usage (from vlm-tuning; --input defaults to geo/eval/conv_test.json of the dinov3 repository):
    python retrieval/crossmodal_retrieval.py --configs <run>/config.yaml
    python retrieval/crossmodal_retrieval.py --configs <run1>/config.yaml <run2>/config.yaml --directions i2t t2i \\
        --exclude_same_slide --random
"""
import json
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
from lightning.pytorch import seed_everything

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEFAULT_INPUT, METRICS, image_path, jaccard_matrix, normalize_labels, ranking_metrics  # noqa: E402
from vlm_run import encode_images, encode_texts, load_run, pairwise_similarity, run_name  # noqa: E402
from dataset.datasets import GEO_INDICES  # noqa: E402


def load_items(path, root, campos, groups=None, min_size=None):
    """
    Distinct (image, labels) and (description, labels) pairs per campo, as in the single modality tests.

    :return: images {campo: [dict(image, slide, key, labels)]}, texts {campo: [dict(text, slides, key, labels)]} and
        the number of images skipped for being smaller than min_size
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    images, texts, small = defaultdict(dict), defaultdict(dict), set()
    for sample in data:
        campo = sample.get('campo')
        if campo not in campos or (groups and sample.get('user_group') not in groups) or not sample.get('labels'):
            continue
        labels = normalize_labels(sample['labels'])
        if not labels:
            continue
        key = json.dumps(labels, ensure_ascii=False, sort_keys=True)
        label_set = {(c, v) for c, vs in labels.items() for v in vs}
        slide = sample.get('slide_id')

        text = sample['conversation'][1]['content']
        if isinstance(text, str) and text.strip():
            # a description may be shared by several slides, its slides tell which images it was written for
            item = texts[campo].setdefault((text.strip(), key), {'text': text.strip(), 'slides': set(), 'key': key,
                                                                   'labels': label_set})
            item['slides'].add(None if slide is None else str(slide))

        image = image_path(root, sample)
        if min_size and sample.get('size') is not None and sample['size'] < min_size:
            small.add(image)
            continue
        images[campo].setdefault((image, key), {'image': image, 'slide': None if slide is None else str(slide),
                                                'key': key, 'labels': label_set})

    images = {c: list(v.values()) for c, v in images.items()}
    texts = {c: list(v.values()) for c, v in texts.items()}
    return images, texts, len(small)


def same_slide_mask(images, texts):
    """[n images, n texts] pairs where the description was written only for the image's slide."""
    return np.array([[im['slide'] is not None and t['slides'] == {im['slide']} for t in texts] for im in images])


def report(results, campos, directions):
    lines = ['| model | direction | campo | R@1 | R@5 | R@10 | mAP | nDCG@10 (Jaccard) | queries / items / gallery |',
             '|---' * 9 + '|']
    for name, per_direction in results.items():
        for direction in directions:
            for campo in campos + ['mean']:
                m = per_direction[direction].get(campo)
                count = '{} / {} / {}'.format(m['queries'], m['items'], m['gallery']) if 'queries' in m else ''
                lines.append('| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} |'.format(
                    name, direction, campo, m['R@1'], m['R@5'], m['R@10'], m['mAP'], m['nDCG@10'], count))
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__.split('\n\n')[1])
    parser.add_argument('--configs', nargs='+', required=True, help='config.yaml of trainLight.py runs (saved in their folders)')
    parser.add_argument('--names', nargs='+', default=None, help='model names in the results (default: run folder names)')
    parser.add_argument('--set', nargs='*', default=[], help='config overrides applied to every run, KEY=VALUE')
    parser.add_argument('--checkpoint', default='best', help='best, last, none (the starting model) or a checkpoint path, see vlm_run.py')
    parser.add_argument('--input', default=DEFAULT_INPUT, help='split json with the samples and labels')
    parser.add_argument('--root', default=None, help='image folder (default: dataset.root of the first run)')
    parser.add_argument('--campos', nargs='+', default=None, help='default: the geo_index_val of the first run')
    parser.add_argument('--groups', nargs='+', type=int, default=None, help='user groups kept (default: all, as the single modality tests)')
    parser.add_argument('--size', type=int, default=None, help='center crop side in pixels, smaller images skipped (default: dataset.resolutions[-1] of the first run)')
    parser.add_argument('--directions', nargs='+', choices=['i2t', 't2i'], default=['i2t'])
    parser.add_argument('--exclude_same_slide', action='store_true', help='leave out the descriptions written only for the query image slide')
    parser.add_argument('--random', action='store_true', help='add a random similarity baseline')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--bf16', action='store_true', help='autocast the models to bfloat16')
    parser.add_argument('--out', default=None, help='json with the results (default: <output_dir>/crossmodal_<input>.json with one run, else <input>_crossmodal.json)')
    args = parser.parse_args()

    seed_everything(777, workers=True)
    broken = [s for s in args.set if '=' not in s]
    if broken:
        raise ValueError('--set entries must be KEY=VALUE, got {}'.format(broken))
    if args.names is not None and (len(args.names) != len(args.configs) or len(set(args.names)) != len(args.names)):
        raise ValueError('--names: one distinct name per config')

    first = OmegaConf.merge(OmegaConf.load(args.configs[0]), OmegaConf.from_dotlist(args.set))
    root = args.root or first.dataset.root
    size = args.size or first.dataset.resolutions[-1]
    campos = args.campos or [GEO_INDICES[i] for i in first.dataset.geo_index_val]

    images, texts, small = load_items(args.input, root, campos, args.groups, size)
    campos = [c for c in campos if c in images and c in texts]
    print('images from {}; {} images smaller than {} px skipped'.format(root, small, size))
    jaccard, exclude = {}, {}
    for campo in campos:
        jaccard[campo] = jaccard_matrix([it['labels'] for it in images[campo]], [t['labels'] for t in texts[campo]])
        exclude[campo] = same_slide_mask(images[campo], texts[campo]) if args.exclude_same_slide else None
        print('{}: {} images, {} descriptions, {} label combinations'.format(
            campo, len(images[campo]), len(texts[campo]), len({it['key'] for it in images[campo] + texts[campo]})))
    paths = sorted({it['image'] for campo in campos for it in images[campo]})
    index = {p: i for i, p in enumerate(paths)}
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError('{} images not found, e.g. {}'.format(len(missing), missing[:3]))

    def evaluate(similarity):
        """similarity(campo) -> [n images, n texts]; ranking metrics of each direction, per campo and their mean."""
        out = {d: {} for d in args.directions}
        for campo in campos:
            sim = similarity(campo)
            image_keys, text_keys = [it['key'] for it in images[campo]], [t['key'] for t in texts[campo]]
            for direction in args.directions:
                if direction == 'i2t':
                    m = ranking_metrics(sim, image_keys, text_keys, jaccard[campo], exclude[campo])
                else:
                    m = ranking_metrics(sim.T, text_keys, image_keys, jaccard[campo].T,
                                        None if exclude[campo] is None else exclude[campo].T)
                m['gallery'] = sim.shape[1] if direction == 'i2t' else sim.shape[0]
                out[direction][campo] = m
                print('  {} {:12s} R@1 {:.3f} R@5 {:.3f} R@10 {:.3f} mAP {:.3f} nDCG@10 {:.3f} ({} queries / {} items / {} gallery)'.format(
                    direction, campo, m['R@1'], m['R@5'], m['R@10'], m['mAP'], m['nDCG@10'], m['queries'], m['items'], m['gallery']))
        for direction in args.directions:
            out[direction]['mean'] = {k: float(np.mean([out[direction][c][k] for c in campos])) for k in METRICS}
        return out

    results, weights, confs = {}, {}, []
    if args.random:
        print('\n=== random')
        rng = np.random.default_rng(0)
        results['random'] = evaluate(lambda campo: rng.standard_normal((len(images[campo]), len(texts[campo]))).astype(np.float32))

    for i, config in enumerate(args.configs):
        model, conf = load_run(config, args.set, args.checkpoint, args.device)
        name = args.names[i] if args.names else run_name(conf)
        confs.append(conf)
        weights[name] = conf.model.load_weights
        print('\n=== {} ({})'.format(name, config))
        image_features = encode_images(model, paths, size, args.device, args.batch_size, args.workers, args.bf16)
        print('  image features {}'.format(tuple(image_features.shape)))

        def model_similarity(campo):
            rows = torch.tensor([index[it['image']] for it in images[campo]])
            text_features = encode_texts(model, [t['text'] for t in texts[campo]], args.device, args.batch_size, args.bf16)
            return pairwise_similarity(image_features[rows], text_features).numpy()

        results[name] = evaluate(model_similarity)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print('\n' + report(results, campos, args.directions))
    suffix = '_noslide' if args.exclude_same_slide else ''
    stem = os.path.splitext(os.path.basename(args.input))[0]
    out = args.out or (os.path.join(confs[0].output_dir, 'crossmodal_{}{}.json'.format(stem, suffix)) if len(confs) == 1
                       else os.path.splitext(args.input)[0] + '_crossmodal{}.json'.format(suffix))
    settings = {'configs': args.configs, 'set': args.set, 'checkpoint': args.checkpoint, 'weights': weights,
                'input': args.input, 'root': root, 'campos': campos, 'groups': args.groups, 'size': size,
                'exclude_same_slide': args.exclude_same_slide}
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'settings': settings, 'results': results}, f, indent=2)
    print('\nsaved to {}'.format(out))
