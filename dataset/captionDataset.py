import json
import os.path
from random import randint, sample
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler


#TODO: move preprocess function here
class CaptionDataset(Dataset):
    def __init__(self, rootDir, annotationFile, dataset, preprocess, tokenizer):
        datasets = {
            'coco': os.path.join(rootDir, '{}2017'.format(annotationFile.split('.')[0])),
            'nwpu': os.path.join(rootDir, 'images')
        }
        self.data = json.load(open(os.path.join(rootDir, annotationFile), 'r'))
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        try:
            self.root = datasets[dataset]
        except ValueError:
            raise ValueError("Invalid dataset value, supported datasets are: " + " ".join(datasets.keys()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        k = randint(0, 4)
        # print('SAMPLE', sample)
        return {
            'images': self.preprocess([os.path.join(self.root, sample['image_name'])]).squeeze(0),
            'captions': self.tokenizer(sample['captions'][k]).squeeze(0),
        }

    def get_loader(self, batchSize):
        sampler = RandomSampler(self)
        return DataLoader(self, batch_size=batchSize, sampler=sampler, )

