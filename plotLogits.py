from omegaconf import OmegaConf
import argparse
import os
import torch
import matplotlib.pyplot as plt
from torchmetrics.functional import auroc
from lightning.pytorch import seed_everything
from dataset.datasets import CaptionDataset, GeoDataset, GEO_INDICES
from model.encoders import crop_transform
from model.createModel import createModel


def collect_logits(model, loader, device, multi_positive, max_batches=None):
    """
    Computes the image-text similarities inside each batch and splits them into positive and negative pairs.

    :return: (positive cosine, negative cosine), flattened over all batches
    """
    positives, negatives = [], []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        with torch.no_grad():
            image_features = model.encode_image(batch['image'].to(device))
            text_features = model.encode_text(batch['tokens'].to(device))

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            cosine = (image_features @ text_features.T).float().cpu()

        if multi_positive:
            # every pair with the same class is a positive
            labels = batch['class']
            positive_mask = labels[:, None] == labels[None, :]
        else:
            positive_mask = torch.eye(cosine.shape[0], dtype=torch.bool)

        positives.append(cosine[positive_mask])
        negatives.append(cosine[~positive_mask])
        print('batch {}: {} positives, {} negatives'.format(i, positive_mask.sum().item(), (~positive_mask).sum().item()))

    return torch.cat(positives), torch.cat(negatives)


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

    print(title)
    print('  positives cosine mean {:.4f} std {:.4f}'.format(positives.mean().item(), positives.std().item()))
    print('  negatives cosine mean {:.4f} std {:.4f}'.format(negatives.mean().item(), negatives.std().item()))
    print('  AUC {:.4f}'.format(auc))
    print('  saved at', path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots the distributions of the logits of positive and negative image-text pairs of a trained model')
    parser.add_argument('--config', type=str, required=True, help='config.yaml saved in the experiment dir by trainLight.py')
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--batch_size', type=int, default=None, help='pairs are compared inside each batch, defaults to the training batch size')
    parser.add_argument('--max_batches', type=int, default=None, help='limit the number of batches used')
    parser.add_argument('--multi_positive', action='store_true', default=None, help='pairs with the same class are positives, defaults to the training setting')
    parser.add_argument('--bins', type=int, default=100)
    parser.add_argument('--output', type=str, default=None, help='output dir, defaults to the experiment dir')
    args = parser.parse_args()

    seed_everything(777, workers=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    conf = OmegaConf.load(args.config)
    conf.model.load_weights = True
    batch_size = args.batch_size if args.batch_size is not None else conf.train.batch_size
    multi_positive = args.multi_positive if args.multi_positive is not None else conf.train.get('multi_positive', False)
    siglip = conf.train.get('loss', 'contrastive') == 'siglip'
    output = args.output if args.output is not None else conf.output_dir
    os.makedirs(output, exist_ok=True)

    model = createModel(conf).to(device)
    model.eval()

    logit_scale = model.model.logit_scale.exp().item()
    logit_bias = model.logit_bias.item()
    annotation = conf.dataset.train_annotation if args.split == 'train' else conf.dataset.val_annotation

    # (title, loader) for each evaluated dataset
    loaders = []
    if conf.dataset.name != 'geo':
        if multi_positive:
            raise ValueError('multi positive needs class labels, only available for the geo dataset')

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

    for name, loader in loaders:
        positives, negatives = collect_logits(model, loader, device, multi_positive, args.max_batches)
        title = '{} {} batch {}{}'.format(name, args.split, batch_size, ' multi positive' if multi_positive else '')
        path = os.path.join(output, 'logits_{}_{}.png'.format(name, args.split))
        plot_distributions(positives, negatives, logit_scale, logit_bias, siglip, title, path, args.bins)

    # python plotLogits.py --config /nethome/recpinfo/users/fibz/data/checkpoint/vlm-finetuning/<run>/config.yaml --split val
