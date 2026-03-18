import json
from PIL import Image
import os
from random import randint
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Literal
Image.MAX_IMAGE_PIXELS = 500000000


GEO_INDICES = {0: 'classification', 1: 'composition', 2: 'texture', 3: 'porosity', 4:'diagenesis'}

class GeoDataset(Dataset):
    def __init__(self, 
                 rootDir: str, 
                 annotationFile: str, 
                 preprocess: callable, 
                 tokenizer: callable, 
                 group: Literal['1', '2', '3', '2+3'], 
                 geo_idx: list=[0], 
                 randomImage=False):
        
        data = pd.read_csv(os.path.join(rootDir, annotationFile))
        self.images = []
        self.texts = []
        data = data.dropna(subset=[GEO_INDICES[i] for i in geo_idx])
        
        if group == '2+3':
            data = data[data['user_group'].isin([2, 3])]
        
        else:
            data = data[data['user_group'] == int(group)]

        groups = data.groupby('slide_id')
        for group, values in groups:
            self.images.append(values['image_id'].to_list())
            text = []
            for idx in geo_idx:
                text.append(values[GEO_INDICES[idx]].to_list()[0])  # all texts are equal 
            
            self.texts.append(text)

        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.random = randomImage
        self.root = rootDir

    def __len__(self):
            return len(self.images)
    
    def __getitem__(self, index):
        k = randint(0, len(self.images[index])-1) if self.random else 0 # random image
        name = os.path.join(self.root, 'images', '{}.png'.format(self.images[index][k])) 
        text = self.texts[index][randint(0, len(self.texts[index])-1)]

        return {
            'images': self.preprocess([name]).squeeze(0),
            'captions': self.tokenizer(text, truncate=True).squeeze(0),
        }
    
    def get_loader(self, batchSize, shuffle):
        return DataLoader(self, batch_size=batchSize, shuffle=shuffle, num_workers=15, pin_memory=True)

        
class CaptionDataset(Dataset):
    def __init__(self, rootDir, annotationFile, dataset, preprocess, tokenizer, random=False):
        datasets = {
            'coco': os.path.join(rootDir, '{}2017'.format(annotationFile.split('.')[0])),
            'nwpu': os.path.join(rootDir, 'images')
        }
        self.data = json.load(open(os.path.join(rootDir, annotationFile), 'r'))
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.random = random
        try:
            self.root = datasets[dataset]

        except ValueError:
            raise ValueError("Invalid dataset value, supported datasets are: " + " ".join(datasets.keys()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.random:
            k = randint(0, len(sample['captions'])-1)
        
        else:
            k = 0
        
        name = sample['image_name'].replace('\\', '/')
        return {
            'images': self.preprocess([os.path.join(self.root, name)]).squeeze(0),
            'captions': self.tokenizer(sample['captions'][k]).squeeze(0),
        }

    def get_loader(self, batchSize, shuffle):
        return DataLoader(self, batch_size=batchSize, shuffle=shuffle, num_workers=15, pin_memory=True)
