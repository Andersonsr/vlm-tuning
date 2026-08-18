from omegaconf import OmegaConf
import argparse
import os
from dataset.datasets import CaptionDataset, GeoDataset, GEO_INDICES, DistributedSingleDatasetBatchSampler
from model.encoders import resize_transform, crop_transform
import lightning as L
from lightning.pytorch import seed_everything
import torch
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, ConcatDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from glob import glob
from model.createModel import createModel
import pandas as pd    
import torch.distributed as dist



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='model/configs/CLIP_default.yaml')
    parser.add_argument('--multiresolution', action='store_true', default=False, help='use to enable multiresolution training')
    parser.add_argument('--batch_size', default=None, type=int)

    args = parser.parse_args()
    conf = OmegaConf.load(args.config)

    seed_everything(777, workers=True)
    if args.batch_size is not None:
        conf.train.batch_size = args.batch_size

    model = createModel(conf)
    model.learnable_parameters()

    if conf.dataset.name != 'geo':
        train_dataset = CaptionDataset(
            conf.dataset.root, 
            conf.dataset.train_annotation, 
            conf.dataset.name, 
            model.prepareImages, 
            model.tokenize, 
            random=conf.dataset.random
            )
        
        train_loader = train_dataset.get_loader(conf.train.batch_size, True)
        
    else:
        train_datasets = []
        for idx in conf.dataset.geo_index:
            # smaller dimension
            # larger dimension
            train_dataset = GeoDataset(
                conf.dataset.root, 
                conf.dataset.val_annotation, 
                lambda x: crop_transform(x,  conf.dataset.resolutions[-1], 16), #2nd dim is the largest dim
                model.tokenize, 
                conf.dataset.geo_group,
                idx,
                size= conf.dataset.resolutions[-1], # larger than 2nd dim
                randomImage=True,
                larger=True,
                )
            train_datasets.append(train_dataset)

            if args.multiresolution:
                train_dataset = GeoDataset(
                    conf.dataset.root, 
                    conf.dataset.val_annotation, 
                    lambda x: crop_transform(x,  conf.dataset.resolutions[0], 16), 
                    model.tokenize, 
                    conf.dataset.geo_group,
                    idx,
                    size= conf.dataset.resolutions[-1], 
                    randomImage=True,
                    larger=False    
                    )
                train_datasets.append(train_dataset)
            
            
        if len(train_datasets) > 1:
            combined_dataset = ConcatDataset(train_datasets)
            sampler = DistributedSingleDatasetBatchSampler(
                dataset_lengths=[len(d) for d in train_datasets],
                batch_size=conf.train.batch_size,
                shuffle=True,
                drop_last=True,
            )
            
            train_loader = DataLoader(
                combined_dataset,
                batch_sampler=sampler,
                collate_fn=train_datasets[0].collate,
                num_workers=8,
            )

        elif len(train_datasets) == 1:
            train_loader = train_datasets[0].get_loader(conf.train.batch_size, True)

    for batch in train_loader:
        query_labels = batch['class']
        print(batch['class'])
        positive_mask = (
            query_labels[:, None] == query_labels[None, :]
        ).float()

        target = positive_mask / positive_mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1.0)

        print(target.shape)

        plt.figure(figsize=(6, 5))
        plt.imshow(target, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Scale Bar')
        plt.title(f"Multipositive targets")
        plt.savefig(f"target GEO.png")
        break