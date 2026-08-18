from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset.datasets import CaptionDataset, GeoDataset
from sympy import Si
import pickle
from torchmetrics.retrieval import RetrievalPrecision, RetrievalRecall
import os
from model.encoders import resize_transform, crop_transform
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from model.createModel import createModel
import lightning as l 
from omegaconf import OmegaConf
import argparse
import pandas as pd
from model.encoders import CLIP
from torch.utils.data import DataLoader, ConcatDataset, Subset


def normalizeObj(obj):
    return tuple(
        (key, tuple(sorted(value)) if isinstance(value, list) else value)
        for key, value in sorted(obj.items())
    )


def gap_distance(images, texts):
    images /= images.norm(dim=-1, keepdim=True)
    texts /= texts.norm(dim=-1, keepdim=True)

    image_centroid = images.mean(dim=0)
    texts_centroid = texts.mean(dim=0)

    centroid_distance = torch.linalg.norm(image_centroid - texts_centroid)
    pairwise_distance = torch.diagonal(torch.cdist(images, texts, p=2)).mean()
    return centroid_distance, pairwise_distance


def retrieval(images, texts, name, labels=None):
    ks = [1, 5, 10]
    images /= images.norm(dim=-1, keepdim=True)
    texts /= texts.norm(dim=-1, keepdim=True)
    similarities = texts @ images.T
    targets = torch.eye(similarities.shape[0])
    indexes = torch.arange(targets.shape[0])
    indexes = indexes.repeat(targets.shape[0], 1).T

    t2i = []
    i2t = []
    for k in ks:
        rk = RetrievalRecall(top_k=k)
        t2i.append(rk(similarities, targets, indexes))
        i2t.append(rk(similarities.T, targets, indexes))


    data = {}
    for i in range(len(ks)):
        data[('i2t', f'r@{ks[i]}')] = [f'{i2t[i].cpu().item():.3f}']

    for i in range(len(ks)):
        data[('t2i', f'r@{ks[i]}')] = [f'{t2i[i].cpu().item():.3f}']

    print(pd.DataFrame(data))

    plt.clf()
    plt.plot(ks, t2i, label='text to image')
    plt.plot(ks, i2t, label='image to text')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('r@k')
    plt.title(f'recall {name}')
    plt.savefig(f'plots/retrieval_{name}.png')


def similarity(images, texts):
    images /= images.norm(dim=-1, keepdim=True)
    texts /= texts.norm(dim=-1, keepdim=True)
    similarities = texts @ images.T
    mean = similarities.mean()
    positive_mean = torch.diagonal(similarities).mean()

    # negative mean
    off_diagonal = similarities * (1 - torch.eye(similarities.shape[0]))
    n = similarities.shape[0]
    negative_mean = off_diagonal.sum() / (n**2 - n)

    return mean, positive_mean, negative_mean
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--conf', type=str, help='configuration file path', required=True)
    parser.add_argument('--split', choices=['train', 'val'], required=True)
    parser.add_argument('--all_texts', action='store_true', default=False)
    parser.add_argument('--batch', type=int, default=5000) # composition 1411
    parser.add_argument('--annotation', type=str, default=None)
    parser.add_argument('--root', type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    conf = OmegaConf.load(args.conf)
    model = createModel(conf)
    model = model.to(device)
    model.eval()
    # print(model)
    if args.annotation is not None:
        annotation = args.annotation

    else:
        annotation = conf.dataset.train_annotation if args.split == 'train' else conf.dataset.val_annotation
    
    # print('batch size', args.batch)
    if conf.dataset.name != 'geo':
        dataset = CaptionDataset(
            conf.dataset.root if args.root is None else args.root, 
            annotation, 
            conf.dataset.name, 
            model.prepareImages, 
            model.tokenize, 
            random=False,
            all_texts=args.all_texts
            )

        loader = dataset.get_loader(args.batch, False)

    else:
        for idx in conf.dataset.geo_index_val:
            print(f'Loading geo index {idx}')
            dataset = GeoDataset(
                conf.dataset.root if args.root is None else args.root, 
                annotation, 
                lambda x: crop_transform(x,  conf.dataset.resolutions[-1], 16),  
                model.tokenize, 
                conf.dataset.geo_group,
                idx,
                randomImage=False,
                return_labels=True,
                larger=True,
                size= conf.dataset.resolutions[-1] # 0=min, 1=max
                )
           
            loader = dataset.get_loader(args.batch, False)

    results = {'t2i': [], 'i2t': [], 'k': []}
    
    for batch in loader:
        print(batch['tokens'].shape)
        print(batch['image'].shape)

        with torch.no_grad():
            context_len = batch['tokens'].shape[-1] # context length
            bs = batch['tokens'].shape[0]
            # print(context_len, bs)
            ncaptions = 1

            if len(batch['tokens'].shape) > 2:
                text_features = model.model.encode_text(batch['tokens'].view(-1, context_len).to(device))
                ncaptions = batch['tokens'].shape[1]

            else:
                text_features = model.model.encode_text(batch['tokens'].to(device))
                            
            image_features = model.model.encode_image(batch['image'].to(device))
            
            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = model.model.logit_scale.exp()
            logits_per_image = logit_scale.to(image_features.device) * (image_features @ text_features.t())
            logits_per_text = logits_per_image.t()

            # print('Image logits shape', logits_per_image.shape)

            if conf.dataset.name == 'geo':
                labels = batch['class']
                targets = []
                
                for label in labels:
                    equal = []
                    for other_label in labels:
                        equal.append(int(label == other_label))

                    targets.append(equal)

                # torch.eye()
                targets = torch.Tensor(targets).to(logits_per_image.device)
                indexes = torch.arange(targets.shape[0]).to(logits_per_image.device)
                indexes = indexes.repeat(targets.shape[1], 1).T
                
                sums = targets.sum(dim=0)
                average = sums.mean()
                print('average number of positive values', average)
                
                for k in [1, 5, 10, 20, 50, 100]:
                    rk = RetrievalRecall(top_k=k)
                    results['i2t'].append(rk(logits_per_image, targets, indexes).detach().item())
                    results['t2i'].append(rk(logits_per_text, targets, indexes).detach().item())
                    results['k'].append(k)
            
            else:
                # image and text shapes can be different NxN*5 
                # retrieval i2t
                targets_i = torch.zeros(logits_per_image.shape).to(logits_per_image.device)
                for i in range(targets_i.shape[0]):
                    targets_i[i, int(i*ncaptions): int((i+1)*ncaptions)] = 1
                
                indexes_i = torch.arange(targets_i.shape[0])
                indexes_i = indexes_i.repeat(targets_i.shape[1], 1).T
                
                # retrieval t21
                targets_t = torch.zeros(logits_per_text.shape).to(logits_per_image.device)
                for i in range(targets_t.shape[1]):
                    targets_t[int(i*ncaptions): int((i+1)*ncaptions), i] = 1
                
                indexes_t = torch.arange(targets_t.shape[0])
                indexes_t = indexes_t.repeat(targets_t.shape[1], 1).T

                for k in [1, 5, 10, 20, 50, 100]:
                    rk = RetrievalRecall(top_k=k)
                    results['i2t'].append(rk(logits_per_image, targets_i, indexes_i).detach().item())
                    results['t2i'].append(rk(logits_per_text, targets_t, indexes_t).detach().item())
                    results['k'].append(k)
    
        break

    name = 'all_texts_' if args.all_texts else ''
    name += f'{args.split}_'
    save_path = os.path.join(os.path.dirname(args.conf), f'{name}retrieval_results.csv')
    pd.DataFrame.from_dict(results).to_csv(save_path)
    print(results)
    print(f'saving to: {save_path}')

    with open(os.path.join(os.path.dirname(args.conf), f'{name}logits.pkl'), 'wb') as file:
        pickle.dump(logits_per_image.detach().cpu(), file)