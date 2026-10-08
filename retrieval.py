from omegaconf import OmegaConf
import argparse
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from lightning.pytorch import seed_everything
from model.createModel import createModel
from plotLogits import build_loaders


KS = [1, 5, 10, 20, 50, 100]


def collect_features(model, loader, device, max_batches=None):
    """
    Encodes the whole split, the retrieval gallery is made of every sample, not only the ones in the same batch.

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
            labels.append(batch['class'])
        print('batch {}: {} samples'.format(i, image_features.shape[0]))

    labels = torch.cat(labels).to(device) if len(labels) > 0 else None
    return torch.cat(images), torch.cat(texts), labels


def recall_at_k(queries, keys, query_labels, key_labels, ks, chunk_size=1024):
    """
    Recall@k as in torchmetrics RetrievalRecall: positives in the top k / positives of the query, averaged over queries.

    Pairs with the same label are positives, the similarity matrix is computed in chunks of queries to save memory.
    """
    max_k = min(max(ks), keys.shape[0])
    hits = {k: 0. for k in ks}

    for start in range(0, queries.shape[0], chunk_size):
        q = queries[start:start + chunk_size]
        positive = query_labels[start:start + chunk_size, None] == key_labels[None, :]
        top = (q @ keys.T).topk(max_k, dim=-1).indices
        retrieved = positive.gather(1, top).float().cumsum(dim=-1)
        n_positives = positive.sum(dim=-1).clamp(min=1)

        for k in ks:
            hits[k] += (retrieved[:, min(k, max_k) - 1] / n_positives).sum().item()

    return {k: hits[k] / queries.shape[0] for k in ks}


def plot_recall(df, title, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for direction, label in [('i2t', 'image to text'), ('t2i', 'text to image')]:
        rows = df[df['direction'] == direction]
        ax.plot(rows['k'], rows['recall'], marker='o', label=label)

    ax.set_xscale('log')
    ax.set_xticks(KS)
    ax.set_xticklabels(KS)
    ax.set_ylim(0, 1)
    ax.set_xlabel('k')
    ax.set_ylabel('R@k')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('  saved at', path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computes cross modal retrieval R@k of a trained model')
    parser.add_argument('--config', type=str, required=True, help='config.yaml saved in the experiment dir by trainLight.py')
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size used to encode the samples, defaults to the training batch size')
    parser.add_argument('--max_batches', type=int, default=None, help='limit the number of batches used, the gallery has max_batches * batch_size samples')
    parser.add_argument('--multi_positive', action='store_true', default=None, help='pairs with the same class are positives, defaults to the training setting')
    parser.add_argument('--title', type=str, default=None, help='plot title, defaults to the experiment dir name and split')
    parser.add_argument('--output', type=str, default=None, help='output dir, defaults to the experiment dir')
    args = parser.parse_args()

    seed_everything(777, workers=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    conf = OmegaConf.load(args.config)
    conf.model.load_weights = True
    batch_size = args.batch_size if args.batch_size is not None else conf.train.batch_size
    multi_positive = args.multi_positive if args.multi_positive is not None else conf.train.get('multi_positive', False)
    output = args.output if args.output is not None else conf.output_dir
    experiment = os.path.basename(os.path.normpath(conf.output_dir))
    os.makedirs(output, exist_ok=True)

    model = createModel(conf).to(device)
    model.eval()

    loaders = build_loaders(conf, model, args.split, batch_size, multi_positive)
    base_title = args.title if args.title is not None else '{} {}'.format(experiment, args.split)

    results = []
    for name, loader in loaders:
        images, texts, labels = collect_features(model, loader, device, args.max_batches)
        # without multi positive only the paired sample is a positive
        pair_labels = labels if multi_positive else torch.arange(images.shape[0], device=device)

        recalls = {
            'i2t': recall_at_k(images, texts, pair_labels, pair_labels, KS),
            't2i': recall_at_k(texts, images, pair_labels, pair_labels, KS),
        }

        rows = [
            {'experiment': experiment, 'split': args.split, 'dataset': name, 'multi_positive': multi_positive,
             'gallery_size': images.shape[0], 'direction': direction, 'k': k, 'recall': recall}
            for direction, values in recalls.items() for k, recall in values.items()
        ]
        df = pd.DataFrame(rows)
        results.append(df)
        print(name)
        print(df.pivot(index='k', columns='direction', values='recall').to_string(float_format='{:.4f}'.format))

        # with several geo indices, the dataset name tells the plots apart
        title = base_title if len(loaders) == 1 else '{} {}'.format(base_title, name)
        plot_recall(df, title, os.path.join(output, 'retrieval_{}_{}.png'.format(name, args.split)))

    csv_path = os.path.join(output, 'retrieval_{}.csv'.format(args.split))
    pd.concat(results).to_csv(csv_path, index=False)
    print('metrics saved at', csv_path)

    # python retrieval.py --config /nethome/recpinfo/users/fibz/data/checkpoint/vlm-finetuning/<run>/config.yaml --split val
