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
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='model/configs/CLIP_default.yaml')
    parser.add_argument('--multiresolution', action='store_true', default=False, help='use to enable multiresolution training')
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--split', default='val', type=str, choices=['train', 'val'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    experiment_name = args.config.split("/")[-2]

    seed_everything(777, workers=True)
    if args.batch_size is not None:
        conf.train.batch_size = args.batch_size

    model = createModel(conf)
    model.learnable_parameters()

    if conf.dataset.name != 'geo':
        train_dataset = CaptionDataset(
            conf.dataset.root, 
            conf.dataset.train_annotation if args.split == 'train' else conf.dataset.val_annotation, 
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
                conf.dataset.train_annotation if args.split == 'train' else conf.dataset.val_annotation, 
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
                    conf.dataset.train_annotation if args.split == 'train' else conf.dataset.val_annotation, 
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
        # # batch positives
        df = pd.DataFrame({'class': batch['class'].cpu().numpy(), 'labels': batch['labels']}).drop_duplicates(subset=['class'])
        print(df[['labels', 'class']])

        query_labels = batch['class']
        positive_mask = (
            query_labels[:, None] == query_labels[None, :]
        ).float()
        
        plt.figure(figsize=(6, 5))
        plt.imshow(positive_mask, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Scale Bar')
        plt.title(f"Multipositive {experiment_name} {args.split}")
        plt.savefig(f"images/targets {experiment_name} {args.split}.png")
        plt.clf()
        
        model = model.to(device)
        # print('Emebeddings visualization')

        with torch.no_grad():
            image_features = model.model.encode_image(batch['image'].to(device))
            text_features = model.model.encode_text(batch['tokens'].to(device))
            features = torch.cat((image_features, text_features))
            features = features.cpu().numpy()

        logits = 100 * image_features @ text_features.t()
    
        plt.figure(figsize=(6, 5))
        plt.imshow(logits.softmax(dim=-1).cpu().numpy(), cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Scale Bar')
        plt.title(f"softmax {experiment_name} {args.split}")
        plt.savefig(f"images/softmax {experiment_name} {args.split}.png")
        plt.clf()
        
        tsne = TSNE(n_components=2, learning_rate='auto', metric='cosine', method='exact', random_state=42)
        x = tsne.fit_transform(features)

        df = pd.DataFrame(x, columns=['t-SNE 1', 't-SNE 2'])
        df['Class'] = torch.cat((batch['class'], batch['class']))

        n = batch['class'].shape[0]

        df['modality'] = ['vision' if i < n else 'text' for i in range(n*2)]
        k = len(df['Class'].drop_duplicates())

        plt.figure(figsize=(8, 6))
        ax = sns.scatterplot(
            data=df, 
            x='t-SNE 1', 
            y='t-SNE 2', 
            # s=25,
            style='modality',
            hue='Class',       # Color by this column
            palette='Dark2', # Choose a color palette
            alpha=0.5,
            legend=True,
        )

        plt.title(f'TSNE {experiment_name} {args.split} classes={k}')
        plt.savefig(f'images/TSNE {experiment_name} {args.split}.png')
        plt.clf()
        break