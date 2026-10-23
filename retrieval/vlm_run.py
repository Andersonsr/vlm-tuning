"""
Models trained with trainLight.py, built as training builds them, so the benchmarks can evaluate any run:
    the run's saved config.yaml merged over trainLight.DEFAULTS (as a new run is; old configs get the entries added
    since), model.createModel (every family: CLIP, LongCLIP, RemoteCLIP, GeoRSCLIP, DINOtxt, DualEncoder with SSL or
    DINO-TXT vision, LoRA, residual adapters, heads), and images as in validation (crop_transform center crops).

Checkpoints (checkpoint):
    best   the run's best checkpoint-*.ckpt (ModelCheckpoint on train.monitor); if it is a DeepSpeed folder, the
           consolidated pytorch_model/pytorch_model.bin (consolidate.py --best)
    last   pytorch_model/pytorch_model.bin, else last.ckpt (model.createModel.find_weights, as evaluateModel.py)
    none   no run weights: the model the run started from (train.init_weights, or the pretrained towers)
    <path> that checkpoint
"""
import os
import sys
from glob import glob

import torch
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
VLM = os.path.dirname(HERE)
# vlm-tuning's packages (dataset, model) and model/ itself, which its modules import from directly, before any other
# folder with the same module names (geo's dataset.py, model/ and model/loratorch_utils.py)
for path in (os.path.join(VLM, 'model'), VLM):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(1, path)

from trainLight import DEFAULTS  # noqa: E402
from model.createModel import createModel, find_weights  # noqa: E402
from model.encoders import crop_transform  # noqa: E402
from model.dual_encoder import pairwise_similarity, pool_queries  # noqa: E402,F401

CHECKPOINTS = ('best', 'last', 'none')


def is_run_config(conf):
    """A trainLight.py run config (model and dataset sections), not a geo encoder config (encoder section)."""
    return 'model' in conf and 'dataset' in conf and 'encoder' not in conf


def run_checkpoint(conf, checkpoint='best'):
    """Weights of the run in conf.output_dir (see the module docstring), or False for checkpoint 'none'."""
    if checkpoint == 'none':
        return conf.train.get('init_weights') or False
    if checkpoint not in CHECKPOINTS:
        return checkpoint
    run = conf.output_dir
    consolidated = os.path.join(run, 'pytorch_model', 'pytorch_model.bin')
    if checkpoint == 'best':
        best = sorted(glob(os.path.join(run, 'checkpoint-*.ckpt')), key=os.path.getmtime)
        if best and os.path.isfile(best[-1]):
            return best[-1]
        if best and os.path.isfile(consolidated):
            # DeepSpeed saves folders: the consolidated weights, assumed of the best one (consolidate.py --best)
            print('WARNING: {} is a DeepSpeed folder, using {}'.format(best[-1], consolidated))
            return consolidated
        if best:
            raise FileNotFoundError('{} is a DeepSpeed folder: consolidate it (consolidate.py --dir {} --best) or use '
                                    'checkpoint last'.format(best[-1], run))
        print('WARNING: no checkpoint-*.ckpt in {}, using the last weights'.format(run))
    return find_weights(OmegaConf.merge(conf, {'model': {'load_weights': True}}))


def load_run(path, overrides=(), checkpoint='best', device='cpu'):
    """
    :param path: config.yaml saved in the run folder by trainLight.py
    :param overrides: KEY=VALUE config entries applied on top
    :return: the model in eval mode on device and its config
    """
    conf = OmegaConf.merge(DEFAULTS, OmegaConf.load(path), OmegaConf.from_dotlist(list(overrides)))
    conf.model.load_weights = run_checkpoint(conf, checkpoint)
    print('{}: weights {}'.format(conf.name, conf.model.load_weights or 'none (the starting model)'))
    model = createModel(conf).to(device)
    model.eval()
    return model, conf


def run_name(conf):
    return os.path.basename(os.path.normpath(conf.output_dir))


class RunImages(torch.utils.data.Dataset):
    """Center crops of size pixels at native resolution, as the validation of trainLight.py (geo datasets)."""

    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return crop_transform([self.paths[i]], self.size, 16)


@torch.no_grad()
def encode_images(model, paths, size, device, batch_size, workers, bf16=False):
    """Normalised model.encode_image features, [n, d], or [n, k, d] with DualEncoder 'max' pooling."""
    device = torch.device(device)
    loader = torch.utils.data.DataLoader(RunImages(paths, size), batch_size=batch_size, num_workers=workers)
    out = []
    for i, x in enumerate(loader):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            features = model.encode_image(x.to(device))
        out.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu())
        if i % 20 == 0:
            print('  {}/{} images'.format(min((i + 1) * batch_size, len(paths)), len(paths)))
    return torch.cat(out)


@torch.no_grad()
def encode_texts(model, texts, device, batch_size, bf16=False):
    """Normalised model.encode_text features of the texts tokenized by the run's tokenizer, [n, d]."""
    device = torch.device(device)
    out = []
    for start in range(0, len(texts), batch_size):
        tokens = model.tokenize(list(texts[start:start + batch_size])).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            features = model.encode_text(tokens)
        out.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu())
    return torch.cat(out)


def truncated(model, texts):
    """Share of texts longer than the run's text context (HF tokenizers of DualEncoder runs; nan otherwise)."""
    tokenizer = model.tokenize
    if not hasattr(tokenizer, 'tokenizer'):
        return float('nan')
    lengths = [len(ids) for ids in tokenizer.tokenizer([tokenizer.prefix + t for t in texts])['input_ids']]
    return sum(n > tokenizer.max_length for n in lengths) / max(len(lengths), 1)
