from omegaconf import OmegaConf
import argparse
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchmetrics.functional import auroc
from lightning.pytorch import seed_everything
from dataset.datasets import CaptionDataset, GeoDataset, GEO_INDICES
from model.encoders import crop_transform
from model.createModel import createModel


KS = [1, 5, 10, 20, 50, 100]


def build_loaders(conf, model, split, batch_size, multi_positive):
    """
    :return: list of (dataset name, loader), one for each geo index in geo_index_val or a single one for caption datasets
    """
    annotation = conf.dataset.train_annotation if split == 'train' else conf.dataset.val_annotation
    loaders = []
    if conf.dataset.name != 'geo':
        dataset = CaptionDataset(conf.dataset.root, annotation, conf.dataset.name, model.prepareImages, model.tokenize, random=False)
        loaders.append((conf.dataset.name, dataset.get_loader(batch_size, True)))

    else:
        for idx in conf.dataset.geo_index_val:
            dataset = GeoDataset(
                conf.dataset.root,
                annotation,
                lambda x: crop_transform(x, conf.dataset.resolutions[-1], 16),
                model.tokenize,
                conf.dataset.geo_group,
                idx,
                size=conf.dataset.resolutions[-1],
                randomImage=False,
                return_labels=True,
                larger=True,
                )
            loaders.append((GEO_INDICES[idx], dataset.get_loader(batch_size, True)))

    return loaders


def collect_features(model, loader, device, max_batches=None):
    """
    Encodes the split once, features are kept in the loader order so the batches can be recovered by slicing.

    :return: normalized image features, normalized text features and class labels (or None)
    """
    images, texts, labels = [], [], []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        with torch.no_grad():
            image_features = model.encode_image(batch['image'].to(device))
            text_features = model.encode_text(batch['tokens'].to(device))

        images.append((image_features / image_features.norm(dim=-1, keepdim=True)).float())
        texts.append((text_features / text_features.norm(dim=-1, keepdim=True)).float())
        if 'class' in batch:
            # geo dataset
            labels.append(batch['class'])
        elif 'labels' in batch and torch.is_tensor(batch['labels']):
            # caption datasets with classes (nwpu), index from labels.json
            labels.append(batch['labels'])
        print('batch {}: {} samples'.format(i, image_features.shape[0]))

    labels = torch.cat(labels).to(device) if len(labels) > 0 else None
    return torch.cat(images), torch.cat(texts), labels


def get_class_names(dataset):
    """
    :return: class names ordered by class index, or None when the dataset has no classes
    """
    categories = getattr(dataset, 'categories', None)
    if categories is not None and len(categories) > 0:
        # geo dataset, pandas index from factorize
        return list(categories)

    labels = getattr(dataset, 'labels', None)
    if isinstance(labels, dict):
        # caption datasets, labels.json maps name -> index
        return sorted(labels, key=labels.get)

    return None


def class_colors(n_classes):
    # enough distinct colors for nwpu (45 classes), cycles after 60
    if n_classes <= 10:
        return list(plt.get_cmap('tab10').colors)
    if n_classes <= 20:
        return list(plt.get_cmap('tab20').colors)
    return list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('tab20b').colors) + list(plt.get_cmap('tab20c').colors)


def batch_similarities(images, texts, pair_labels, batch_size):
    """
    Image-text similarities inside each batch, as seen by the training loss, split into positive and negative pairs.

    :return: (positive cosine, negative cosine), flattened over all batches
    """
    positives, negatives = [], []

    for start in range(0, images.shape[0], batch_size):
        cosine = images[start:start + batch_size] @ texts[start:start + batch_size].T
        labels = pair_labels[start:start + batch_size]
        positive_mask = labels[:, None] == labels[None, :]
        positives.append(cosine[positive_mask].cpu())
        negatives.append(cosine[~positive_mask].cpu())

    return torch.cat(positives), torch.cat(negatives)


def retrieval_at_k(queries, keys, query_labels, key_labels, ks, chunk_size=1024):
    """
    Recall@k and precision@k as in torchmetrics RetrievalRecall and RetrievalPrecision, averaged over queries:
        recall: positives in the top k / positives of the query
        precision: positives in the top k / k

    Pairs with the same label are positives, the similarity matrix is computed in chunks of queries to save memory.

    :return: (recall, precision), dicts of k -> value
    """
    max_k = min(max(ks), keys.shape[0])
    hits = {k: 0. for k in ks}
    precision = {k: 0. for k in ks}

    for start in range(0, queries.shape[0], chunk_size):
        q = queries[start:start + chunk_size]
        positive = query_labels[start:start + chunk_size, None] == key_labels[None, :]
        top = (q @ keys.T).topk(max_k, dim=-1).indices
        retrieved = positive.gather(1, top).float().cumsum(dim=-1)
        n_positives = positive.sum(dim=-1).clamp(min=1)

        for k in ks:
            hits[k] += (retrieved[:, min(k, max_k) - 1] / n_positives).sum().item()
            precision[k] += (retrieved[:, min(k, max_k) - 1] / k).sum().item()

    n = queries.shape[0]
    return {k: hits[k] / n for k in ks}, {k: precision[k] / n for k in ks}


def plot_distributions(positives, negatives, logit_scale, logit_bias, siglip, title, path, bins):
    # logits as used by the training loss, the bias is only part of the sigmoid loss
    pos_logits = logit_scale * positives + (logit_bias if siglip else 0.)
    neg_logits = logit_scale * negatives + (logit_bias if siglip else 0.)

    scores = torch.cat([positives, negatives])
    targets = torch.cat([torch.ones_like(positives), torch.zeros_like(negatives)]).long()
    auc = auroc(scores, targets, task='binary').item()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    panels = [
        (axes[0], positives, negatives, 'cosine similarity'),
        (axes[1], pos_logits, neg_logits, 'logit (scale {:.2f}{})'.format(logit_scale, ', bias {:.2f}'.format(logit_bias) if siglip else '')),
    ]

    for ax, pos, neg, xlabel in panels:
        # same bin edges for both, so the densities are comparable
        edges = torch.linspace(min(pos.min(), neg.min()).item(), max(pos.max(), neg.max()).item(), bins + 1).numpy()
        ax.hist(neg.numpy(), bins=edges, density=True, alpha=0.6, label='negatives (n={})'.format(len(neg)), color='tab:red')
        ax.hist(pos.numpy(), bins=edges, density=True, alpha=0.6, label='positives (n={})'.format(len(pos)), color='tab:blue')
        ax.axvline(pos.mean().item(), color='tab:blue', linestyle='--', linewidth=1)
        ax.axvline(neg.mean().item(), color='tab:red', linestyle='--', linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('density')
        ax.legend()

    if siglip:
        # sigmoid(logit) = 0.5, pairs on the right are classified as positives
        axes[1].axvline(0., color='black', linewidth=1, label='decision boundary')
        axes[1].legend()

    fig.suptitle('{} (AUC {:.4f})'.format(title, auc))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print('  positives cosine mean {:.4f} std {:.4f}'.format(positives.mean().item(), positives.std().item()))
    print('  negatives cosine mean {:.4f} std {:.4f}'.format(negatives.mean().item(), negatives.std().item()))
    print('  AUC {:.4f}'.format(auc))
    print('  saved at', path)


def plot_embeddings(images, texts, labels, class_names, method, n_pairs, title, path):
    """
    Projects both modalities into the same 2D space, markers identify the modality and lines connect each image to its
    paired text. When class labels are available the points are colored by class.

    :param method: 'pca' (fitted on all samples, then applied to the plotted ones) or 'tsne' (only the plotted ones)
    :param n_pairs: number of image-text pairs plotted, sampled at random
    """
    generator = torch.Generator().manual_seed(777)
    idx = torch.randperm(images.shape[0], generator=generator)[:n_pairs].to(images.device)
    n = idx.shape[0]
    joint = torch.cat([images[idx], texts[idx]]).cpu().numpy()

    if method == 'pca':
        pca = PCA(n_components=2).fit(torch.cat([images, texts]).cpu().numpy())
        points = pca.transform(joint)
        xlabel = 'PC1 ({:.1%} var)'.format(pca.explained_variance_ratio_[0])
        ylabel = 'PC2 ({:.1%} var)'.format(pca.explained_variance_ratio_[1])
    else:
        points = TSNE(n_components=2, init='pca', perplexity=min(30., (2 * n - 1) / 3), random_state=777).fit_transform(joint)
        xlabel, ylabel = 't-SNE 1', 't-SNE 2'

    image_points, text_points = points[:n], points[n:]

    # wider figure for the two column legend of datasets with many classes
    fig, ax = plt.subplots(figsize=(11 if labels is not None and int(labels.max()) >= 22 else 8, 7))
    # positive pairs
    ax.add_collection(LineCollection(list(zip(image_points, text_points)), colors='gray', linewidths=0.5, alpha=0.4, zorder=1))

    if labels is not None:
        classes = labels[idx].cpu()
        palette = class_colors(len(class_names) if class_names is not None else int(labels.max()) + 1)
        colors = [palette[c % len(palette)] for c in classes.tolist()]
        image_colors, text_colors = colors, colors
    else:
        image_colors, text_colors = 'tab:blue', 'tab:orange'

    ax.scatter(image_points[:, 0], image_points[:, 1], c=image_colors, marker='o', s=25, edgecolors='black', linewidths=0.3, zorder=2)
    ax.scatter(text_points[:, 0], text_points[:, 1], c=text_colors, marker='^', s=35, edgecolors='black', linewidths=0.3, zorder=2)

    handles = [
        Line2D([], [], marker='o', linestyle='', color='tab:blue' if labels is None else 'lightgray', markeredgecolor='black', label='image'),
        Line2D([], [], marker='^', linestyle='', color='tab:orange' if labels is None else 'lightgray', markeredgecolor='black', label='text'),
        Line2D([], [], color='gray', alpha=0.6, label='positive pair'),
    ]
    if labels is not None:
        for c in sorted(set(classes.tolist())):
            name = class_names[c] if class_names is not None else 'class {}'.format(c)
            handles.append(Line2D([], [], marker='s', linestyle='', color=palette[c % len(palette)], label=name))

    # long legends (nwpu has 45 classes) are split in columns
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=7 if len(handles) > 25 else 8, ncol=1 if len(handles) <= 25 else 2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('{} ({} pairs, {})'.format(title, n, method.upper() if method == 'pca' else 't-SNE'))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('  saved at', path)


def plot_retrieval(df, metric, title, path):
    """
    :param metric: 'recall' or 'precision', column of df
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for direction, label in [('i2t', 'image to text'), ('t2i', 'text to image')]:
        rows = df[df['direction'] == direction]
        ax.plot(rows['k'], rows[metric], marker='o', label=label)

    ax.set_xscale('log')
    ax.set_xticks(KS)
    ax.set_xticklabels(KS)
    if metric == 'recall':
        ax.set_ylim(0, 1)
    else:
        # with a single positive per query P@k <= 1/k, a fixed 0-1 range would flatten the curves
        ax.set_ylim(bottom=0)
    ax.set_xlabel('k')
    ax.set_ylabel('R@k' if metric == 'recall' else 'P@k')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('  saved at', path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots the logits distributions of positive and negative pairs and computes cross modal retrieval R@k of a trained model')
    parser.add_argument('--config', type=str, required=True, help='config.yaml saved in the experiment dir by trainLight.py')
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--batch_size', type=int, default=None, help='logits distributions use the pairs inside each batch, defaults to the training batch size')
    parser.add_argument('--max_batches', type=int, default=None, help='limit the number of batches used, the retrieval gallery has max_batches * batch_size samples')
    parser.add_argument('--multi_positive', action='store_true', default=None, help='pairs with the same class are positives, defaults to the training setting')
    parser.add_argument('--bins', type=int, default=100, help='histogram bins of the logits distributions')
    parser.add_argument('--reduction', choices=['pca', 'tsne'], default='pca', help='dimension reduction used to plot both modalities together')
    parser.add_argument('--n_pairs', type=int, default=300, help='number of image-text pairs in the embeddings plot')
    parser.add_argument('--title', type=str, default=None, help='plot title, defaults to the experiment dir name and split')
    parser.add_argument('--output', type=str, default=None, help='output dir, defaults to the experiment dir')
    args = parser.parse_args()

    seed_everything(777, workers=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    conf = OmegaConf.load(args.config)
    conf.model.load_weights = True
    batch_size = args.batch_size if args.batch_size is not None else conf.train.batch_size
    multi_positive = args.multi_positive if args.multi_positive is not None else conf.train.get('multi_positive', False)
    siglip = conf.train.get('loss', 'contrastive') == 'siglip'
    # logits and retrieval results depend on how positives are defined, keep both versions side by side
    suffix = '_multipositive' if multi_positive else ''
    output = args.output if args.output is not None else conf.output_dir
    experiment = os.path.basename(os.path.normpath(conf.output_dir))
    os.makedirs(output, exist_ok=True)

    model = createModel(conf).to(device)
    model.eval()

    logit_scale = model.model.logit_scale.exp().item()
    logit_bias = model.logit_bias.item()
    loaders = build_loaders(conf, model, args.split, batch_size, multi_positive)
    base_title = args.title if args.title is not None else '{} {}'.format(experiment, args.split)

    results = []
    for name, loader in loaders:
        images, texts, labels = collect_features(model, loader, device, args.max_batches)
        if multi_positive and labels is None:
            raise ValueError('multi positive needs class labels, {} has none'.format(name))
        # without multi positive only the paired sample is a positive
        pair_labels = labels if multi_positive else torch.arange(images.shape[0], device=device)
        # with several geo indices, the dataset name tells the plots apart
        title = base_title if len(loaders) == 1 else '{} {}'.format(base_title, name)
        # the embeddings plot does not depend on how positives are defined, so only the other plots are marked
        metric_title = '{} multipositive'.format(title) if multi_positive else title
        print(metric_title)

        # logits distributions, pairs inside each batch
        positives, negatives = batch_similarities(images, texts, pair_labels, batch_size)
        plot_distributions(positives, negatives, logit_scale, logit_bias, siglip, metric_title,
                           os.path.join(output, 'logits_{}_{}{}.png'.format(name, args.split, suffix)), args.bins)

        # retrieval, every sample of the split is in the gallery
        metrics = {
            'i2t': retrieval_at_k(images, texts, pair_labels, pair_labels, KS),
            't2i': retrieval_at_k(texts, images, pair_labels, pair_labels, KS),
        }

        df = pd.DataFrame([
            {'experiment': experiment, 'split': args.split, 'dataset': name, 'multi_positive': multi_positive,
             'gallery_size': images.shape[0], 'direction': direction, 'k': k, 'recall': recall[k], 'precision': precision[k]}
            for direction, (recall, precision) in metrics.items() for k in KS
        ])
        results.append(df)
        print(df.pivot(index='k', columns='direction', values=['recall', 'precision']).to_string(float_format='{:.4f}'.format))
        plot_retrieval(df, 'recall', metric_title, os.path.join(output, 'retrieval_{}_{}{}.png'.format(name, args.split, suffix)))
        plot_retrieval(df, 'precision', metric_title, os.path.join(output, 'precision_{}_{}{}.png'.format(name, args.split, suffix)))

        # both modalities in the same 2D space
        class_names = get_class_names(loader.dataset)
        plot_embeddings(images, texts, labels, class_names, args.reduction, args.n_pairs, title,
                        os.path.join(output, 'embeddings_{}_{}_{}.png'.format(args.reduction, name, args.split)))

    csv_path = os.path.join(output, 'retrieval_{}{}.csv'.format(args.split, suffix))
    pd.concat(results).to_csv(csv_path, index=False)
    print('metrics saved at', csv_path)

    # python evaluateModel.py --config /nethome/recpinfo/users/fibz/data/checkpoint/vlm-finetuning/<run>/config.yaml --split val
