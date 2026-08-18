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
    parser.add_argument('--dataset_name', required=True, type=str, help='used only in the plot title')
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
        
        val_dataset = CaptionDataset(conf.dataset.root, conf.dataset.val_annotation, conf.dataset.name, model.prepareImages, model.tokenize, random=False)
        print('train dataset size: {} val dataset size {}'.format(len(train_dataset), len(val_dataset)))
        train_loader = train_dataset.get_loader(conf.train.batch_size, True)
        val_loader = val_dataset.get_loader(conf.train.batch_size, False)

    else:
        train_datasets = []
        for idx in conf.dataset.geo_index:
            # smaller dimension
            # larger dimension
            train_dataset = GeoDataset(
                conf.dataset.root, 
                conf.dataset.train_annotation, 
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
                    conf.dataset.train_annotation, 
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

    copies = []
    has_copies = 0
    logit_scale = 100

    for i, batch in enumerate(train_loader):
        df = pd.DataFrame({'text': batch['text']})
        copies.append(len(df) - len(df['text'].drop_duplicates()))      
        if len(df) - len(df['text'].drop_duplicates()) > 0:
            has_copies += 1

        if i == 0:
            with torch.no_grad():
                image_features = model.encode_image(batch['image'])
                text_features = model.encode_text(batch['tokens'])
        
                # normalize before gathering
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)            

                logits_per_image = logit_scale * image_features @ text_features.T

                # PLOT logits
                plt.figure(figsize=(6, 5))
                plt.imshow(logits_per_image, cmap='viridis', interpolation='nearest')
                plt.colorbar(label='Scale Bar')
                plt.title(f"Scaled logits Heatmap {args.dataset_name}")
                plt.savefig(f"heatmap {args.dataset_name}.png")
                plt.clf()

                # PLOT softmax
                plt.figure(figsize=(6, 5))
                plt.imshow(logits_per_image.softmax(dim=-1), cmap='viridis', interpolation='nearest')
                plt.colorbar(label='Scale Bar')
                plt.title(f"Softmax Heatmap {args.dataset_name}")
                plt.savefig(f"softmax {args.dataset_name}.png")
                
                break 

    print(copies)
    print('total number of batches:', len(copies))
    print('batches with any number of copies:', has_copies)
    print('mean copies:', sum(copies)/len(copies))
    print('copies ratio', has_copies / len(copies))
    print('batch size:', conf.train.batch_size)
