import json
from PIL import Image
import os
from random import randint
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
from typing import Literal
Image.MAX_IMAGE_PIXELS = 500000000


GEO_INDICES = {0: 'classification', 1: 'composition', 2: 'texture', 3: 'porosity', 4:'diagenesis'}
LABEL_MAP = {1: ['Constituintes Principais', 'Constituintes Secundários', 'Gênese', 'Tamanho do Elemento', 'Comp. Atual do Elemento', 'Acessórios', 'Núcleo do Esferulito'],
             0: ['litologia_microscopica'],
             2: ['Estrutura/Textura', 'Granulação <2 mm', 'Granulação modal principal (mm)', 'Granulação secundária (mm)', 'Seleção', 'Empacotamento', 'Arranjo', 'Matriz', 'Tipo de Matriz', 'Matriz (Dunham 1962)', 'Tipo de Laminação', 'Laminação Caracterizada Por', 'Proporção Cascalho/Areia/Lama', 'Tipo Contato entre Partic.', 'Tamanho do Cristal', 'Integridade das Conchas', 'Orientação das Conchas'],
             3: ['Tipo(s) de Poro(s)', 'Estimativa Visual', 'Tam. Modal do(s) Poro(s)'],
             4: ['Eventos Diagenéticos', 'Cimento', 'Espaço Interconstituintes']}

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
                 ):
        
        data = pd.read_csv(os.path.join(rootDir, annotationFile))
  
        self.return_labels = return_labels   
        self.label_map = LABEL_MAP[geo_idx]
        self.images = []
        self.texts = []
        self.labels = []

        data = data.dropna(subset=[GEO_INDICES[geo_idx]])
        data = data[data['user_group'].isin(group)]    
    
        groups = data.groupby('slide_id')
        for group, values in groups:
            self.images.append(values['image_id'].to_list())
            self.texts.append(values[GEO_INDICES[geo_idx]].to_list()[0])

            if return_labels:
                labels = {}
                for idx, ob in values['labels'].items():
                    # ob = ob.replace("'", '"')
                    for key, val in ast.literal_eval(ob).items() :
                        if key in LABEL_MAP[geo_idx]:
                            labels[key] = sorted(val)
                self.labels.append(labels)      

        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.random = randomImage
        self.root = rootDir

    def __len__(self):
            return len(self.images)
    
    def __getitem__(self, index):
        k = randint(0, len(self.images[index])-1) if self.random else 0 # random image
        name = os.path.join(self.root, 'images', '{}.png'.format(self.images[index][k])) 
        text = self.texts[index] # index could be 0 instead of random

        payload =  {
            'image': self.preprocess([name]).squeeze(0),
            'tokens': self.tokenizer(text, truncate=True).squeeze(0),
            'text': text, 
            'image_name': name, 
        }
        
        if self.return_labels:
            payload['labels'] = self.labels[index]
            
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
        return data

    def get_loader(self, batchSize, shuffle):
        return DataLoader(self, batch_size=batchSize, shuffle=shuffle, collate_fn=self.collate, num_workers=15, pin_memory=True)

        
class CaptionDataset(Dataset):
    def __init__(self, rootDir, annotationFile, dataset, preprocess, tokenizer, random=False, all_texts=False):
        datasets = {
            'coco': os.path.join(rootDir, '{}2017'.format(annotationFile.split('.')[0])),
            'nwpu': os.path.join(rootDir, 'images')
        }
        self.data = json.load(open(os.path.join(rootDir, annotationFile), 'r'))
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.random = random
        self.all_texts = all_texts
        try:
            self.root = datasets[dataset]

        except ValueError:
            raise ValueError("Invalid dataset value, supported datasets are: " + " ".join(datasets.keys()))
        
        if dataset == 'nwpu':
            self.labels = json.load(open(os.path.join(rootDir, 'labels.json'), 'r'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.random:
            k = randint(0, len(sample['captions'])-1)
        
        else:
            k = 0
        
        name = sample['image_name'].replace('\\', '/')
        texts = sample['captions'] if self.all_texts else sample['captions'][k]
        
        sample = {
            'image': self.preprocess([os.path.join(self.root, name)]).squeeze(0),
            'tokens': self.tokenizer(texts).squeeze(0),
            'text': texts,
            'image_name': name, 
        }
        if hasattr(self, 'labels'):
            sample['labels'] = self.labels[name.split('/')[0]]
        
        return sample

    def get_loader(self, batchSize, shuffle):
        return DataLoader(self, batch_size=batchSize, shuffle=shuffle, num_workers=15, pin_memory=True)
