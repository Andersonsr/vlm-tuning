"""
Image-to-image retrieval with off-the-shelf variable-resolution vision encoders (DINOv3, SigLIP 2 NaFlex, Perception
Encoder Core, C-RADIOv3), on the samples and the two resolutions of trainLight.py, with the metrics of
text_retrieval.py / image_retrieval.py.

Data, as trainLight.py's geo datasets (GeoDataset, paths and settings from a training config: dataset.root,
dataset.<split>_annotation, dataset.geo_group, dataset.resolutions): one sample per slide (its first image, as in
validation), the slides with images of at least resolutions[-1] pixels center-cropped to resolutions[-1] and the others
to resolutions[0]. As the training batches never mix the two, each resolution is ranked on its own. Each encoder gets
the crop in [0, 1] and applies its own preprocessing (normalization; PE: resized to a multiple of its 14 px patch).

Every distinct combination of labels of a field is a class; each sample is a query against the other samples of the
same field and resolution:
    exact     the positives are the samples with exactly the same labels: R@1/5/10 and mAP (queries with a positive)
    graded    nDCG@10 with relevance = Jaccard overlap of the two label sets; every sample is a query
Samples without labels of the field ('no info') are left out.

Encoders (--models), the image embedding of each:
    dinov3    facebook/dinov3-vitl16-pretrain-lvd1689m (transformers), CLS token
    siglip2   google/siglip2-so400m-patch16-naflex (transformers), pooled output, max_num_patches = the crop's patches
    pe        PE-Core-L14-336 (perception_models: --pe_repo, a clone of github.com/facebookresearch/perception_models),
              attention-pooled projected embedding
    radio     nvidia/C-RADIOv3-L (transformers, trust_remote_code), summary
Weights are downloaded from the HuggingFace hub on the first run (HF_HOME for the cache).

Usage (from vlm-tuning):
    python retrieval/vision_encoders.py --config model/configs/geo_default.yaml --pe_repo ~/perception_models --random
    python retrieval/vision_encoders.py --config model/configs/geo_default.yaml --models dinov3 radio --split val
"""
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # vlm-tuning: dataset, model
sys.path.insert(0, HERE)
from common import METRICS, jaccard_matrix, normalize_labels, retrieval_metrics  # noqa: E402
from image_retrieval import load_runs, report  # noqa: E402
from dataset.datasets import GEO_INDICES, GeoDataset  # noqa: E402
from model.encoders import IMAGENET_MEAN, IMAGENET_STD, crop_transform  # noqa: E402

DINOV3_ID = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
SIGLIP2_ID = 'google/siglip2-so400m-patch16-naflex'
PE_CONFIG = 'PE-Core-L14-336'
RADIO_ID = 'nvidia/C-RADIOv3-L'


# ------------------------------------------------------------------------------------------------------------ data

def load_samples(conf, split, fields):
    """{field: {crop: [dict(image, key, labels)]}}: the samples of trainLight.py's geo datasets, one per slide."""
    annotation = conf.dataset[f'{split}_annotation']
    low, high = conf.dataset.resolutions[0], conf.dataset.resolutions[-1]
    samples = {}
    for idx in fields:
        field = GEO_INDICES[idx]
        samples[field] = {}
        for crop, larger in [(high, True), (low, False)]:
            dataset = GeoDataset(conf.dataset.root, annotation, None, None, conf.dataset.geo_group, idx, size=high,
                                 randomImage=False, larger=larger)
            items = []
            for images, labels in zip(dataset.images, dataset.labels):
                labels = normalize_labels({c: [str(v) for v in (vs if isinstance(vs, (tuple, list)) else [vs])]
                                           for c, vs in labels if c != 'no info'})
                if not labels:
                    continue
                items.append({'image': os.path.join(conf.dataset.root, f'{images[0]}.png'),
                              'key': json.dumps(labels, ensure_ascii=False, sort_keys=True),
                              'labels': {(c, v) for c, vs in labels.items() for v in vs}})
            samples[field][crop] = items
    return samples


class Crops(torch.utils.data.Dataset):
    """trainLight.py's validation crop (crop_transform, center crop), back in [0, 1] for each encoder's own
    normalization."""

    def __init__(self, paths, size):
        self.paths, self.size = paths, size
        self.mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
        self.std = torch.tensor(IMAGENET_STD)[:, None, None]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return crop_transform([self.paths[i]], self.size, 16) * self.std + self.mean


# -------------------------------------------------------------------------------------------------------- encoders
# forward(x): x [n, 3, s, s] in [0, 1] -> image embeddings [n, d]

class Dinov3(nn.Module):
    def __init__(self, args):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(DINOV3_ID)

    def forward(self, x):
        return self.model(pixel_values=TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)).last_hidden_state[:, 0]


class Siglip2(nn.Module):
    def __init__(self, args):
        super().__init__()
        from transformers import AutoModel, AutoProcessor
        self.model = AutoModel.from_pretrained(SIGLIP2_ID)
        self.processor = AutoProcessor.from_pretrained(SIGLIP2_ID).image_processor

    def forward(self, x):
        patch = self.processor.patch_size
        # the crop at its own resolution: as many patches as it has (no resizing), normalized by the processor
        inputs = self.processor(images=list(x.float().cpu()), do_rescale=False, return_tensors='pt',
                                max_num_patches=(x.shape[-2] // patch) * (x.shape[-1] // patch))
        inputs = {k: v.to(x.device) for k, v in inputs.items()}
        out = self.model.get_image_features(**inputs)
        return out if torch.is_tensor(out) else out.pooler_output


class PECore(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.pe_repo:
            sys.path.insert(0, os.path.expanduser(args.pe_repo))
        import core.vision_encoder.pe as pe
        self.model = pe.CLIP.from_config(PE_CONFIG, pretrained=True).visual  # the text tower is not needed

    def forward(self, x):
        # 14 px patches: the nearest multiple of 14 (256 -> 252, 512 -> 518); posemb interpolated, RoPE to the grid
        side = round(x.shape[-1] / self.model.patch_size) * self.model.patch_size
        x = F.interpolate(x, size=(side, side), mode='bicubic', align_corners=False, antialias=True)
        return self.model(TF.normalize(x, [0.5] * 3, [0.5] * 3))


class Radio(nn.Module):
    def __init__(self, args):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(RADIO_ID, trust_remote_code=True)

    def forward(self, x):
        summary, _ = self.model(x)  # inputs in [0, 1], normalized inside the model
        return summary


ENCODERS = {'dinov3': Dinov3, 'siglip2': Siglip2, 'pe': PECore, 'radio': Radio}


@torch.no_grad()
def embed(encoder, paths, crop, args):
    loader = torch.utils.data.DataLoader(Crops(paths, crop), batch_size=args.batch_size, num_workers=args.workers)
    out = []
    for i, x in enumerate(loader):
        x = x.to(args.device)
        with torch.autocast(device_type=args.device.split(':')[0], dtype=torch.bfloat16, enabled=args.bf16):
            out.append(F.normalize(encoder(x).float(), dim=-1).cpu())
        if i % 20 == 0:
            print(f'  {crop}px: {min((i + 1) * args.batch_size, len(paths))}/{len(paths)} images')
    return torch.cat(out).numpy()


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', required=True, help='trainLight.py config with the dataset settings (e.g. model/configs/geo_default.yaml)')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--fields', nargs='+', type=int, default=[1, 2, 3, 4], help=f'geo indices {GEO_INDICES}')
    parser.add_argument('--models', nargs='+', default=list(ENCODERS), choices=list(ENCODERS))
    parser.add_argument('--pe_repo', default=None, help='clone of perception_models (if not installed)')
    parser.add_argument('--random', action='store_true', help='add a random-embedding baseline')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--bf16', action='store_true', help='autocast the encoders to bfloat16')
    parser.add_argument('--out', default=None, help='json the results are appended to (default: vision_encoders_<split>.json)')
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    samples = load_samples(conf, args.split, args.fields)
    fields = [GEO_INDICES[i] for i in args.fields]
    crops = list(samples[fields[0]])
    jaccard, paths = {}, {}
    for crop in crops:
        for field in fields:
            items = samples[field][crop]
            jaccard[field, crop] = jaccard_matrix([it['labels'] for it in items])
            print(f'{field} {crop}px: {len(items)} slides, {len({it["key"] for it in items})} label combinations')
        paths[crop] = sorted({it['image'] for field in fields for it in samples[field][crop]})
    missing = [p for crop in crops for p in paths[crop] if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f'{len(missing)} images not found, e.g. {missing[:3]}')

    def evaluate(embeddings):
        """embeddings: {crop: [n, d] in the order of paths[crop]} -> {crop: {field: metrics, 'mean': metrics}}"""
        out = {}
        for crop in crops:
            index = {p: i for i, p in enumerate(paths[crop])}
            out[str(crop)] = {}
            for field in fields:
                items = samples[field][crop]
                rows = [index[it['image']] for it in items]
                m = retrieval_metrics(embeddings[crop][rows], [it['key'] for it in items], jaccard[field, crop])
                out[str(crop)][field] = m
                print(f"  {crop}px {field:12s} R@1 {m['R@1']:.3f} R@5 {m['R@5']:.3f} R@10 {m['R@10']:.3f} "
                      f"mAP {m['mAP']:.3f} nDCG@10 {m['nDCG@10']:.3f} ({m['queries']} queries / {m['items']} slides)")
            out[str(crop)]['mean'] = {k: float(np.mean([out[str(crop)][f][k] for f in fields])) for k in METRICS}
        return out

    results = {}
    if args.random:
        print('\n=== random')
        rng = np.random.default_rng(0)
        results['random'] = evaluate({c: rng.standard_normal((len(paths[c]), 256)).astype(np.float32) for c in crops})

    for name in args.models:
        print(f'\n=== {name}')
        encoder = ENCODERS[name](args).to(args.device).eval()
        results[name] = evaluate({crop: embed(encoder, paths[crop], crop, args) for crop in crops})
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for crop in crops:
        print(f'\n{crop}px crops\n' + report({name: r[str(crop)] for name, r in results.items()}, fields))

    out = args.out or f'vision_encoders_{args.split}.json'
    settings = {'config': args.config, 'split': args.split, 'annotation': conf.dataset[f'{args.split}_annotation'],
                'geo_group': list(conf.dataset.geo_group), 'resolutions': list(conf.dataset.resolutions),
                'models': {'dinov3': DINOV3_ID, 'siglip2': SIGLIP2_ID, 'pe': PE_CONFIG, 'radio': RADIO_ID},
                'bf16': args.bf16}
    runs = load_runs(out)
    runs.append({'time': datetime.now().isoformat(timespec='seconds'), 'settings': settings, 'results': results})
    tmp = out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'runs': runs}, f, indent=2)
    os.replace(tmp, out)
    print(f'\nappended to {out} ({len(runs)} runs)')
