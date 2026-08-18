import io
import json
import re
from PIL import Image
import os
from random import randint
import torch
from model.createModel import createModel
from omegaconf import OmegaConf 
import torch.distributed as dist
import torch
from torch.utils.data import BatchSampler, Dataset, DataLoader
import pandas as pd
import ast
from typing import Literal
import torch
import math
import random
Image.MAX_IMAGE_PIXELS = 500000000


GEO_INDICES = {0: 'classification', 1: 'composition', 2: 'texture', 3: 'porosity', 4:'diagenesis'}
LABEL_MAP = {1: ['Constituintes Principais', 'Constituintes Secundários', 'Gênese', 'Tamanho do Elemento', 'Comp. Atual do Elemento', 'Acessórios', 'Núcleo do Esferulito'],
             0: ['litologia_microscopica'],
             2: ['Estrutura/Textura', 'Granulação <2 mm', 'Granulação modal principal (mm)', 'Granulação secundária (mm)', 'Seleção', 'Empacotamento', 'Arranjo', 'Matriz', 'Tipo de Matriz', 'Matriz (Dunham 1962)', 'Tipo de Laminação', 'Laminação Caracterizada Por', 'Proporção Cascalho/Areia/Lama', 'Tipo Contato entre Partic.', 'Tamanho do Cristal', 'Integridade das Conchas', 'Orientação das Conchas'],
             3: ['Tipo(s) de Poro(s)', 'Estimativa Visual', 'Tam. Modal do(s) Poro(s)'],
             4: ['Eventos Diagenéticos', 'Cimento', 'Espaço Interconstituintes']}

def normalizeObj(obj):
    return tuple(
        (key, tuple(sorted(value)) if isinstance(value, list) else value)
        for key, value in sorted(obj.items())
    )


class DistributedSingleDatasetBatchSampler(BatchSampler):
    """
    BatchSampler for a ConcatDataset that:
      - never mixes datasets within a batch
      - supports DDP
      - works with PyTorch Lightning (use_distributed_sampler=False)
    """

    def __init__(
        self,
        dataset_lengths,
        batch_size,
        shuffle=True,
        drop_last=True,
        seed=0  ,
    ):
        self.dataset_lengths = list(dataset_lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        self.epoch = 0

        self.dataset_ranges = []

        start = 0
        for length in self.dataset_lengths:
            self.dataset_ranges.append((start, start + length))
            start += length

    def set_epoch(self, epoch):
        """Called by Lightning if available, otherwise can be called manually."""
        self.epoch = epoch

    def __iter__(self):

        # DDP information (available once Lightning has initialized)
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        all_batches = []

        # Build homogeneous batches
        for start, end in self.dataset_ranges:

            length = end - start

            if self.shuffle:
                indices = (
                    torch.randperm(length, generator=g) + start
                ).tolist()
            else:
                indices = list(range(start, end))

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]

                if len(batch) == self.batch_size:
                    all_batches.append(batch)
                elif not self.drop_last:
                    all_batches.append(batch)

        # Shuffle batch order
        if self.shuffle:
            perm = torch.randperm(
                len(all_batches),
                generator=g,
            ).tolist()

            all_batches = [all_batches[i] for i in perm]

        # ------------------------------------------------------------------
        # Make the number of batches divisible by world size
        # ------------------------------------------------------------------

        if world_size > 1:

            remainder = len(all_batches) % world_size

            if remainder != 0:

                if self.drop_last:
                    all_batches = all_batches[: len(all_batches) - remainder]
                else:
                    extra = world_size - remainder

                    # repeat first batches
                    all_batches.extend(all_batches[:extra])

        # Split batches across GPUs
        my_batches = all_batches[rank::world_size]

        yield from my_batches

    def __len__(self):

        total_batches = 0

        for length in self.dataset_lengths:

            if self.drop_last:
                total_batches += length // self.batch_size
            else:
                total_batches += math.ceil(length / self.batch_size)

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1

        if self.drop_last:
            total_batches = (total_batches // world_size)
        else:
            total_batches = math.ceil(total_batches / world_size)

        return total_batches

class GeoDataset(Dataset):
    def __init__(self, 
                 rootDir: str, 
                 annotationFile: str, 
                 preprocess: callable, 
                 tokenizer: callable, 
                 group: list, 
                 geo_idx: int=1, 
                 randomImage=False,
                 return_labels=True,
                 size=256,
                 larger=True,
                 ):
        
        data = pd.read_csv(os.path.join(annotationFile))
  
        self.return_labels = return_labels   
        self.label_map = LABEL_MAP[geo_idx]
        self.images = []
        self.texts = []
        self.labels = []
        self.categories = []
        
        data = data.dropna(subset=[GEO_INDICES[geo_idx]])
        data = data[data['user_group'].isin(group)]
        if larger:
            data = data[data['size'] >= size]
        else:    
            data = data[data['size'] < size]

        groups = data.groupby('slide_id')
        for group, values in groups:
            self.images.append(values['image_id'].to_list())
            self.texts.append(values[GEO_INDICES[geo_idx]].to_list()[0])

        if return_labels:
            labels = []
            for e in data['labels'].to_list():
                e = e.replace('"', "cramunhao").replace("'", '"').replace("cramunhao", "'")
                labels.append(json.loads(e))

            data['labels'] = labels
            data['labels'] = data['labels'].map(normalizeObj)
            self.labels = data['labels'].to_list()

            _, self.categories  =  pd.factorize(data['labels'] )

        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.random = randomImage
        self.root = rootDir

    def __len__(self):
            return len(self.images)
    
    def __getitem__(self, index):
        k = randint(0, len(self.images[index])-1) if self.random else 0 # random image
        name = os.path.join(self.root, '{}.png'.format(self.images[index][k])) 
        text = self.texts[index] # index could be 0 instead of random

        payload =  {
            'image': self.preprocess([name]).squeeze(0),
            'tokens': self.tokenizer(text, ).squeeze(0),
            'text': text, 
            'image_name': name, 
        }
        
        if self.return_labels:
            payload['labels'] = self.labels[index]
            payload['class'] = self.categories.get_loc(self.labels[index])
            
        return payload

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

        
class CaptionDataset(Dataset):
    def __init__(self, rootDir, annotationFile, dataset, preprocess, tokenizer, random=False, all_texts=False):
        datasets = {
            'coco': os.path.join(rootDir, '{}2017'.format(annotationFile.split('.')[0])),
            'nwpu': os.path.join(rootDir, 'images'),
            'rsicd': os.path.join(rootDir, 'images'),
        }

        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.random = random
        self.all_texts = all_texts
        try:
            self.root = datasets[dataset]

        except ValueError:
            raise ValueError("Invalid dataset value, supported datasets are: " + " ".join(datasets.keys()))

        print(self.root)
        self.labels = json.load(open('/nethome/recpinfo/users/fibz/data/dataset/nwpu/labels.json', 'r'))

        filepath = os.path.join(rootDir, annotationFile)
        print(f"loading file at: {filepath}")
        if os.path.splitext(filepath)[-1] == '.json':
            self.data = json.load(open(filepath, 'r'))
                
        else:
            raise ValueError("Invalid annotation file format, supported formats are: .json")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        name = sample['image_name'].replace('\\', '/')
        image = self.preprocess([os.path.join(self.root, name)]).squeeze(0)
        
        text = sample['captions']
        if self.random:
            k = randint(0, len(text)-1)
        
        else:
            k = 0

        text = text if self.all_texts else [text[k]]      
        tokens = self.tokenizer(text).squeeze(0)            
        # print('get item', image.shape, tokens.shape, text)
        payload = {
            'image': image,
            'tokens': tokens,
            'text': text,
            'image_name': name, 
        }

        if 'class' in sample.keys():
            payload['labels'] = self.labels[sample['class']]

        return payload
    
    def collate(self, batch):
        data = {}
        for e in batch:
            for key in e.keys():
                if key not in data.keys():
                    data[key] = [e[key]]
                else: 
                    data[key] += [e[key]]

           
        data['image'] = torch.stack(data['image'])
        data['tokens'] = torch.stack(data['tokens'])
        # print('image shape', data['image'].shape)
        # print('text shape', data['tokens'].shape)
        # print(data['text'])
        return data

    def get_loader(self, batchSize, shuffle):
        if self.all_texts:
            return DataLoader(self, batch_size=batchSize, shuffle=shuffle, num_workers=15, pin_memory=True, collate_fn=self.collate)

        return DataLoader(self, batch_size=batchSize, shuffle=shuffle, num_workers=15, pin_memory=True)
