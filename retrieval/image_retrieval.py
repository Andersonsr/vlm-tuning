"""
Image-to-image retrieval with vision encoders, with the metrics of text_retrieval.py, to compare the encoders on the
thin-section images.

Every distinct combination of labels of a description type (campo) is a class. Each image is a query against every
other image of the same campo:
    exact     the positives are the images with exactly the same labels; R@K is the share of queries with a positive
              in the top K, mAP the mean average precision. Only queries with a positive count.
    graded    nDCG@10 with relevance = Jaccard overlap of the two label sets (as (category, value) pairs), so images
              sharing most labels count too; every image is a query.
Each (image, labels) pair is used once per campo; the same image under other labels (another annotation) is left out
of its rankings, since its embedding is identical. --exclude_same_slide also leaves out the images of the query's slide
(neighbouring crops of a slide look alike and mostly share the labels).

Encoders (--configs), two kinds of config files, mixed freely:
    trainLight.py runs     the config.yaml saved in a run folder: any model trained in vlm-tuning, built and loaded as
                           in training (vlm_run.py; --checkpoint best by default); the embedding is model.encode_image,
                           the image side of the aligned space; images as in validation (crop_transform); --pooling
                           global only. Runs trained with DualEncoder 'max' pooling keep their k query outputs per
                           image, and the similarity of two images is their best query-to-query match (the highest
                           cosine over the k x k query pairs), as each text is matched by its closest query in training
    encoder configs        (encoder section, baselines.py): off-the-shelf encoders, DINOv3 (HuggingFace), Fossil (DINOv3
                           SSL checkpoint of this project), DINOtxt and CLIP (examples in encoders/)
Embeddings of encoder configs:
    --pooling global       the encoder's own image vector (return_grid=False): CLS for DINOv3/Fossil, CLS or CLS + mean
                           patch for DINOtxt (encoder.average_patch), the aligned head output for DualEncoder
    --pooling mean         mean of the grid tokens (return_grid=True: patch tokens, or the Q-Former query outputs)
    --pooling global+mean  both, concatenated (each normalised first)

Images are center crops of --size pixels at their native resolution (the standardized px per mm is kept; as
crop_transform, the order used in training), the same crop for every encoder; images whose shorter side is smaller
are skipped. CLIP uses its own preprocessing (224 px).

Usage (from vlm-tuning; --input: split json with the samples and their labels, e.g. conv_test.json):
    python retrieval/image_retrieval.py --input conv_test.json \
        --configs retrieval/encoders/dinov3.yaml retrieval/encoders/fossil.yaml <run>/config.yaml
    python retrieval/image_retrieval.py --input conv_test.json --configs <run1>/config.yaml <run2>/config.yaml --checkpoint last
"""
import json
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageFile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import (METRICS, image_path, jaccard_matrix, normalize_labels, retrieval_metrics,  # noqa: E402
                    similarity_metrics)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ------------------------------------------------------------------------------------------------------------ data

def load_items(path, root, campos=None, groups=None, min_size=None):
    """Distinct (image, labels) pairs per campo: {campo: [dict(image, slide, key, labels)]}, and the number of images
    skipped for being smaller than min_size."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    seen, items, small = set(), defaultdict(list), set()
    for sample in data:
        campo = sample.get('campo')
        if campos and campo not in campos:
            continue
        if groups and sample.get('user_group') not in groups:
            continue
        if not sample.get('labels'):
            continue
        labels = normalize_labels(sample['labels'])
        if not labels:
            continue
        image = image_path(root, sample)
        if min_size and sample.get('size') is not None and sample['size'] < min_size:
            small.add(image)
            continue
        key = json.dumps(labels, ensure_ascii=False, sort_keys=True)
        if (campo, image, key) in seen:
            continue
        seen.add((campo, image, key))
        items[campo].append({'image': image, 'slide': sample.get('slide_id'), 'key': key,
                             'labels': {(c, v) for c, vs in labels.items() for v in vs}})
    return dict(items), len(small)


def exclusion_mask(items, same_slide):
    """[n, n] pairs left out of the rankings: the same image (other labels), and with same_slide the same slide."""
    images = np.array([it['image'] for it in items])
    mask = images[:, None] == images[None, :]
    if same_slide:
        slides = np.array([str(it['slide']) for it in items])
        known = np.array([it['slide'] is not None for it in items])
        mask |= (slides[:, None] == slides[None, :]) & known[:, None] & known[None, :]
    return mask


class Images(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            return self.transform(image.convert('RGB'))


class Crop:
    """Center crop of size pixels at native resolution, normalised with the ImageNet statistics (crop_transform); an
    image whose shorter side is smaller is enlarged to it first."""

    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        import torchvision.transforms.functional as TF
        if min(image.size) < self.size:
            image = TF.resize(image, self.size)
        w, h = image.size
        left, top = w // 2 - self.size // 2, h // 2 - self.size // 2
        x = TF.to_tensor(image.crop((left, top, left + self.size, top + self.size)))
        return TF.normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


# -------------------------------------------------------------------------------------------------------- encoders

class RunEncoder(torch.nn.Module):
    """Image side of a trainLight.py run with the interface of the baselines (forward(x, return_grid))."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, return_grid):
        if return_grid:
            raise ValueError('trainLight.py runs embed images with model.encode_image only, use --pooling global')
        return self.model.encode_image(x)  # [n, k, d] with DualEncoder 'max' pooling: every query output


@torch.no_grad()
def query_match_similarity(features, device, chunk=16):
    """[n, n] best query-to-query match of normalised query outputs [n, k, d]: the highest cosine over the k x k
    query pairs of each two images. chunk images at a time ([chunk * k, n * k] similarities in memory)."""
    n, k, _ = features.shape
    gallery = features.reshape(n * k, -1).to(device)
    sim = torch.empty(n, n)
    for start in range(0, n, chunk):
        block = features[start:start + chunk].to(device)
        b = block.shape[0]
        pairs = (block.reshape(b * k, -1) @ gallery.T).reshape(b, k, n, k)
        sim[start:start + b] = pairs.amax(dim=(1, 3)).cpu()
    return sim.numpy()


def is_run(path):
    from vlm_run import is_run_config
    return is_run_config(OmegaConf.load(path))


def build_encoder(path, overrides, device, checkpoint='best'):
    if is_run(path):
        from vlm_run import load_run
        model, conf = load_run(path, overrides, checkpoint, device)
        return RunEncoder(model), conf

    import vlm_run  # noqa: F401  vlm-tuning's packages on the path
    from baselines import build_baseline
    conf = OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(overrides))
    return build_baseline(conf).to(device).eval(), conf


def model_dtype(encoder):
    return next(p for p in encoder.parameters() if p.is_floating_point()).dtype


@torch.no_grad()
def encode(encoder, paths, size, pooling, device, batch_size, workers, bf16):
    is_clip = type(encoder).__name__ == 'CLIP'
    if is_clip and pooling != 'global':
        raise ValueError('CLIP has no grid output, use --pooling global')
    if isinstance(encoder, RunEncoder):
        from vlm_run import RunImages
        dataset = RunImages(paths, size)  # crop_transform, as in the run's validation
    else:
        dataset = Images(paths, encoder._preprocess if is_clip else Crop(size))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers)
    dtype = model_dtype(encoder)
    out = []
    for i, x in enumerate(loader):
        x = x.to(device, dtype=dtype)
        with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16, enabled=bf16):
            parts = []
            if pooling in ('global', 'global+mean'):
                parts.append(encoder(x, return_grid=False))
            if pooling in ('mean', 'global+mean'):
                parts.append(encoder(x, return_grid=True).mean(1))
        parts = [torch.nn.functional.normalize(p.float(), dim=-1) for p in parts]
        out.append(torch.cat(parts, -1).cpu())
        if i % 20 == 0:
            print(f'  {min((i + 1) * batch_size, len(paths))}/{len(paths)} images')
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------------------------------------- report

def load_runs(path):
    """Earlier executions saved in the results file: {'runs': [{time, settings, results}]}; a file of the former
    single-execution format ({settings, results}) becomes the first run."""
    if not os.path.isfile(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'runs' in data:
        return data['runs']
    return [{'time': None, 'settings': data['settings'], 'results': data['results']}]


def report(results, campos):
    lines = ['| encoder | campo | R@1 | R@5 | R@10 | mAP | nDCG@10 (Jaccard) | queries / images |', '|---' * 8 + '|']
    for name, per_campo in results.items():
        for campo in campos + ['mean']:
            m = per_campo.get(campo)
            if m:
                count = f"{m['queries']} / {m['items']}" if 'queries' in m else ''
                lines.append(f"| {name} | {campo} | {m['R@1']:.3f} | {m['R@5']:.3f} | {m['R@10']:.3f} | {m['mAP']:.3f} "
                             f"| {m['nDCG@10']:.3f} | {count} |")
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--input', required=True, help='split json with the samples and labels (e.g. conv_test.json)')
    parser.add_argument('--root', default=None, help='image folder (default: data.root of the first config)')
    parser.add_argument('--configs', nargs='+', required=True, help='trainLight.py run configs and/or encoder configs (encoders/), one per encoder')
    parser.add_argument('--names', nargs='+', default=None, help='encoder names in the results (default: run folder or config names)')
    parser.add_argument('--set', nargs='*', default=[], help='config overrides applied to every encoder, KEY=VALUE')
    parser.add_argument('--checkpoint', default='best', help='runs: best, last, none (the starting model) or a checkpoint path, see vlm_run.py')
    parser.add_argument('--campos', nargs='+', default=['composition', 'texture', 'porosity', 'diagenesis'])
    parser.add_argument('--groups', nargs='+', type=int, default=None, help='user groups kept (default: all)')
    parser.add_argument('--size', type=int, default=256, help='center crop side in pixels (smaller images skipped)')
    parser.add_argument('--pooling', choices=['global', 'mean', 'global+mean'], default='global')
    parser.add_argument('--exclude_same_slide', action='store_true', help='leave out images of the query slide')
    parser.add_argument('--random', action='store_true', help='add a random-embedding baseline')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--bf16', action='store_true', help='autocast the encoders to bfloat16')
    parser.add_argument('--out', default=None, help='json the results are appended to, one entry per execution with its settings (default: <input>_image_retrieval.json)')
    args = parser.parse_args()

    def default_name(config):
        conf = OmegaConf.load(config)
        if is_run(config):
            return os.path.basename(os.path.normpath(conf.output_dir))
        return os.path.splitext(os.path.basename(config))[0]

    def config_root(config):
        conf = OmegaConf.load(config)
        return conf.dataset.get('root') if is_run(config) else conf.get('data', {}).get('root')

    names = args.names or [default_name(c) for c in args.configs]
    if len(names) != len(args.configs) or len(set(names)) != len(names):
        raise ValueError('--names: one distinct name per config')
    broken = [s for s in args.set if '=' not in s]
    if broken:
        raise ValueError(f'--set entries must be KEY=VALUE, got {broken}')
    root = args.root or config_root(args.configs[0])
    if not root:
        raise ValueError('--root: image folder (the first config has no data.root)')

    items, small = load_items(args.input, root, args.campos, args.groups, args.size)
    campos = [c for c in args.campos if c in items]
    print(f'images from {root}; {small} images smaller than {args.size} px skipped')
    jaccard, exclude = {}, {}
    for campo in campos:
        keys = [it['key'] for it in items[campo]]
        jaccard[campo] = jaccard_matrix([it['labels'] for it in items[campo]])
        exclude[campo] = exclusion_mask(items[campo], args.exclude_same_slide)
        print(f'{campo}: {len(keys)} images, {len(set(keys))} label combinations')
    paths = sorted({it['image'] for campo in campos for it in items[campo]})
    index = {p: i for i, p in enumerate(paths)}
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f'{len(missing)} images not found, e.g. {missing[:3]}')

    def evaluate(embeddings):
        """embeddings [n, d] (cosine), or [n, k, d] query outputs of 'max' pooling runs (best query-to-query match)."""
        out = {}
        for campo in campos:
            rows = [index[it['image']] for it in items[campo]]
            keys = [it['key'] for it in items[campo]]
            if embeddings.ndim == 3:
                sim = query_match_similarity(torch.from_numpy(embeddings[rows]), args.device)
                m = similarity_metrics(sim, keys, jaccard[campo], exclude[campo])
            else:
                m = retrieval_metrics(embeddings[rows], keys, jaccard[campo], exclude[campo])
            out[campo] = m
            print(f"  {campo:12s} R@1 {m['R@1']:.3f} R@5 {m['R@5']:.3f} R@10 {m['R@10']:.3f} mAP {m['mAP']:.3f} "
                  f"nDCG@10 {m['nDCG@10']:.3f} ({m['queries']} queries / {m['items']} images)")
        out['mean'] = {k: float(np.mean([out[c][k] for c in campos])) for k in METRICS}
        return out

    results, weights, similarity = {}, {}, {}
    if args.random:
        print('\n=== random')
        results['random'] = evaluate(np.random.default_rng(0).standard_normal((len(paths), 256)).astype(np.float32))

    for name, config in zip(names, args.configs):
        print(f'\n=== {name} ({config})')
        encoder, conf = build_encoder(config, args.set, args.device, args.checkpoint)
        if isinstance(encoder, RunEncoder):
            weights[name] = conf.model.load_weights
        embeddings = encode(encoder, paths, args.size, args.pooling, args.device, args.batch_size, args.workers,
                            args.bf16)
        if embeddings.ndim == 3:
            similarity[name] = 'best query-to-query match'
            print(f'  {embeddings.shape[1]} query outputs of {embeddings.shape[2]}-d per image, best query-to-query match')
        else:
            similarity[name] = 'cosine'
            print(f'  {embeddings.shape[1]}-d embeddings')
        results[name] = evaluate(embeddings)
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print('\n' + report(results, campos))
    out = args.out or os.path.splitext(args.input)[0] + '_image_retrieval.json'
    settings = {k: getattr(args, k) for k in ('input', 'root', 'configs', 'set', 'checkpoint', 'groups', 'size',
                                              'pooling', 'exclude_same_slide')}
    settings['weights'] = weights  # checkpoint loaded for each run
    settings['similarity'] = similarity
    runs = load_runs(out)
    runs.append({'time': datetime.now().isoformat(timespec='seconds'), 'settings': settings, 'results': results})
    tmp = out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'runs': runs}, f, indent=2)
    os.replace(tmp, out)  # the earlier runs are not lost if writing fails
    print(f'\nappended to {out} ({len(runs)} runs)')
