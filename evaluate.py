from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset.datasets import CaptionDataset, GeoDataset
from sympy import Si
import pickle
from torchmetrics.retrieval import RetrievalPrecision, RetrievalRecall
import os
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from model.createModel import createModel
import lightning as l 
from omegaconf import OmegaConf
import argparse
import pandas as pd
from model.encoders import CLIP
from torch.utils.data import DataLoader, ConcatDataset


def gap_distance(images, texts):
    images /= images.norm(dim=-1, keepdim=True)
    texts /= texts.norm(dim=-1, keepdim=True)

    image_centroid = images.mean(dim=0)
    texts_centroid = texts.mean(dim=0)

    # print(coco_image_centroid.shape)
    # print(coco_texts_centroid.shape)

    centroid_distance = torch.linalg.norm(image_centroid - texts_centroid)
    pairwise_distance = torch.diagonal(torch.cdist(images, texts, p=2)).mean()
    return centroid_distance, pairwise_distance


def SVD(images, texts, name, threshold=0.99):
    cov_i = torch.cov(images.T)
    cov_t = torch.cov(texts.T)

    _, Si, _ = torch.linalg.svd(cov_i)
    _, St, _ = torch.linalg.svd(cov_t)

    image_cumsum = torch.cumsum(Si, dim=0) / torch.sum(Si, dim=0)
    image_cone_end = torch.argmax((image_cumsum > threshold).int()).detach().numpy()
    texts_cumsum = torch.cumsum(St, dim=0) / torch.sum(St, dim=0)
    texts_cone_end = torch.argmax((texts_cumsum > threshold).int()).detach().numpy()

    plt.clf()
    plt.plot(range(Si.shape[0]), torch.log(Si).detach().numpy(), label=f'images')
    plt.plot(range(St.shape[0]), torch.log(St).detach().numpy(), label=f'texts')

    plt.axvline(x=image_cone_end, label=f'image effective dim. {image_cone_end}', color='purple', linestyle='--')
    plt.axvline(x=texts_cone_end, label=f'text effective dim. {texts_cone_end}', color='red', linestyle='--')
    plt.legend()
    plt.xlabel('index (i)')
    plt.ylabel('log (Si)')
    plt.title(f'Effective dimension, threshold = {threshold}')
    plt.savefig(f'plots/distance_{name}.png')

    # mean_i = torch.mean(images, dim=0)
    # mean_t = torch.mean(texts, dim=0)
    # max_i, _ = torch.max(images, dim=0)
    # max_t, _ = torch.max(texts, dim=0)
    # min_i, _ = torch.min(images, dim=0)
    # min_t, _ = torch.min(texts, dim=0)

    # plt.clf()
    # plt.plot(range(len(cov_i)), np.sort(cov_i.diagonal().cpu().numpy())[::-1], label='var image')
    # plt.plot(range(len(cov_t)), np.sort(cov_t.diagonal().cpu().numpy())[::-1], label='var text')
    # plt.plot(range(len(mean_i)), mean_i.cpu().numpy(), label=f'mean image')
    # plt.plot(range(len(mean_t)), mean_t.cpu().numpy(), label=f'mean text')
    # plt.plot(range(len(max_i)), max_i.cpu().numpy(), label=f'max image')
    # plt.plot(range(len(max_t)), max_t.cpu().numpy(), label=f'max text')
    # plt.plot(range(len(min_i)), min_i.cpu().numpy(), label=f'min image')
    # plt.plot(range(len(min_t)), min_t.cpu().numpy(), label=f'min text')
    # plt.ylim(-1.0, 1.0)
    # plt.legend()
    # plt.xlabel('index')
    # # plt.ylabel('Variance')
    # plt.title(f'stats {name}')
    # plt.savefig(f'plots/stats_{name}.png')


def truncate(images, texts, threshold):
    # identify cone end
    cov_i = torch.cov(images.T)
    cov_t = torch.cov(texts.T)
    Ui, Si, Vi = torch.linalg.svd(cov_i)
    Ut, St, Vt = torch.linalg.svd(cov_t)

    image_cumsum = torch.cumsum(Si, dim=0) / torch.sum(Si, dim=0)
    image_cone_end = torch.argmax((image_cumsum> threshold).int()).detach().numpy()
    texts_cumsum = torch.cumsum(St, dim=0) / torch.sum(St, dim=0)
    texts_cone_end = torch.argmax((texts_cumsum > threshold).int()).detach().numpy()
    cone_end = max(image_cone_end, texts_cone_end)

    # image_svd = TruncatedSVD(n_components=cone_end.item())
    # text_svd = TruncatedSVD(n_components=cone_end.item())
    # images_t = image_svd.fit_transform(images.detach().numpy())
    # texts_t = text_svd.fit_transform(texts.detach().numpy())

    Ui, Si, Vi = torch.linalg.svd(images.T)
    Ut, St, Vt = torch.linalg.svd(texts.T)

    Ui = Ui[:, :cone_end]
    Vi = Vi[:cone_end, :]
    Si = Si[:cone_end]

    Ut = Ut[:, :cone_end]
    Vt = Vt[:cone_end, :]
    St = St[:cone_end]

    return images @ Ui @ torch.diag(Si) @ Vi, texts @ Ut @ torch.diag(St) @ Vt


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
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    conf = OmegaConf.load(args.conf)
    model = createModel(conf)
    model = model.to(device)
    model.eval()

    annotation = conf.dataset.train_annotation if args.split == 'train' else conf.dataset.val_annotation
        
    if conf.dataset.name != 'geo':
        dataset = CaptionDataset(
            conf.dataset.root, 
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
                conf.dataset.root, 
                annotation, 
                model.prepareImages, 
                model.tokenize, 
                conf.dataset.geo_group,
                idx,
                randomImage=False,
                return_labels=True
                )
            loader = dataset.get_loader(args.batch, False)

    results = {'t2i': [], 'i2t': [], 'k': []}
    
    for batch in loader:
        print(batch['tokens'].shape)
        print(batch['image'].shape)

        with torch.no_grad():
            context_len = batch['tokens'].shape[-1] # context length
            bs = batch['tokens'].shape[0]
            print(context_len, bs)
            ncaptions = 1

            labels = batch['labels']

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

            print('Image logits shape', logits_per_image.shape)

            if conf.dataset.name == 'geo':
                labels = batch['labels']
                targets = []
                
                for label in labels:
                    equal = []
                    for other_label in labels:
                        equal.append(int(label == other_label))
                    targets.append(equal)
                
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

    print(results)
    name = 'all_texts_' if args.all_texts else ''
    name += f'{args.split}_'
    save_path = os.path.join(os.path.dirname(args.conf), f'{name}retrieval_results.csv')
    pd.DataFrame.from_dict(results).to_csv(save_path)
    
    with open(os.path.join(os.path.dirname(args.conf), f'{name}logits.pkl'), 'wb') as file:
        pickle.dump(logits_per_image.detach().cpu(), file)