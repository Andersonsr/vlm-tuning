"""CC3M as a map-style dataset, read straight from the WebDataset shards written by dataset/download_cc3m.py.

The shards are indexed once (byte offset and size of every image and caption inside the tars) and the index is cached
in <root>/cc3m_<split>_index.npz, so samples are read with a seek, without extracting the tars. Being map-style, it has
a length and works with DistributedSingleDatasetBatchSampler like the other datasets of trainLight.py.

Build the indexes before a multi-gpu run (otherwise every rank builds them):
    python -m dataset.cc3m --root D:\\datasets\\cc3m
"""
import argparse
import io
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

IMAGE_EXTENSIONS = ('jpg', 'jpeg', 'png', 'webp')


def find_shards(root, split):
    """Tars of a split: <root>/cc3m-<split>-NNNN.tar (hugging face mirror) or <root>/<split>/NNNNN.tar (img2dataset)."""
    shards = sorted(glob(os.path.join(root, f'*{split}*.tar')) + glob(os.path.join(root, split, '*.tar')))
    if not shards:
        raise FileNotFoundError(f'no {split} shards in {root}, download them with dataset/download_cc3m.py')
    return [os.path.relpath(path, root) for path in shards]


def index_shard(path):
    """(key, image offset, image size, caption offset, caption size) of every complete sample of a tar."""
    files = {}
    with tarfile.open(path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            # webdataset convention: the key is the file name up to its first dot, the rest the field
            key, _, ext = os.path.basename(member.name).partition('.')
            files.setdefault(key, {})[ext.lower()] = (member.offset_data, member.size)

    samples = []
    for key, fields in files.items():
        image = next((fields[ext] for ext in IMAGE_EXTENSIONS if ext in fields), None)
        if image is not None and 'txt' in fields:
            samples.append((key, *image, *fields['txt']))
    return samples


def build_index(root, split, workers=8):
    shards = find_shards(root, split)
    path = os.path.join(root, f'cc3m_{split}_index.npz')
    if os.path.isfile(path):
        with np.load(path) as index:
            if index['shards'].tolist() == shards:
                return path
        print(f'{path} is outdated (the shards changed), rebuilding it')

    keys, shard_ids, entries = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(index_shard, [os.path.join(root, s) for s in shards])
        for i, samples in enumerate(tqdm(results, total=len(shards), desc=f'indexing cc3m {split}')):
            for key, *offsets in samples:
                keys.append(key)
                shard_ids.append(i)
                entries.append(offsets)

    tmp = f'{path}.{os.getpid()}.tmp.npz'  # written whole, other ranks may be building it too
    np.savez(tmp, shards=np.array(shards), keys=np.array(keys, dtype='S'),
             shard_ids=np.array(shard_ids, dtype=np.int32), entries=np.array(entries, dtype=np.int64).reshape(-1, 4))
    os.replace(tmp, path)
    return path


class CC3MDataset(Dataset):
    def __init__(self, rootDir, split, preprocess, tokenizer, workers=8):
        """
        :param rootDir: folder given to download_cc3m.py --out
        :param split: 'train' or 'validation'
        :param preprocess: list of PIL images -> [n, C, H, W] tensor (model.prepareImages)
        :param tokenizer: model.tokenize
        """
        self.root = rootDir
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        # numpy arrays rather than python lists: no copy-on-read of 3M objects in every dataloader worker
        with np.load(build_index(rootDir, split, workers)) as index:
            self.shards = index['shards'].tolist()
            self.keys = index['keys']
            self.shard_ids = index['shard_ids']
            self.entries = index['entries']

        self.files = {}  # open tars of this process

    def __getstate__(self):
        # dataloader workers open their own file handles
        state = self.__dict__.copy()
        state['files'] = {}
        return state

    def read(self, shard_id, offset, size):
        if shard_id not in self.files:
            self.files[shard_id] = open(os.path.join(self.root, self.shards[shard_id]), 'rb')
        f = self.files[shard_id]
        f.seek(offset)
        return f.read(size)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        shard_id = int(self.shard_ids[index])
        image_offset, image_size, text_offset, text_size = self.entries[index].tolist()

        image = Image.open(io.BytesIO(self.read(shard_id, image_offset, image_size))).convert('RGB')
        text = self.read(shard_id, text_offset, text_size).decode('utf-8').strip()

        return {
            'image': self.preprocess([image]).squeeze(0),
            'tokens': self.tokenizer([text]).squeeze(0),
            'text': text,
            'image_name': '{}/{}'.format(self.shards[shard_id], self.keys[index].decode()),
            'class': index,  # every pair is its own class: multi_positive and siglip losses reduce to single positive
        }

    def collate(self, batch):
        data = {}
        for item in batch:
            for key, val in item.items():
                if key not in data.keys():
                    data[key] = []

                data[key].append(val)

        data['image'] = torch.stack(data['image'])
        data['tokens'] = torch.stack(data['tokens'])
        data['class'] = torch.tensor(data['class'])
        return data

    def get_loader(self, batchSize, shuffle):
        return DataLoader(self, batch_size=batchSize, shuffle=shuffle, collate_fn=self.collate, num_workers=15, pin_memory=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Index the CC3M shards (cached next to them) before training')
    parser.add_argument('--root', required=True, help='folder given to download_cc3m.py --out')
    parser.add_argument('--splits', nargs='+', choices=['train', 'validation'], default=['train', 'validation'])
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    for split in args.splits:
        path = build_index(args.root, split, args.workers)
        with np.load(path) as index:
            print(f'{split}: {len(index["entries"])} samples in {len(index["shards"])} shards, index {path}')
