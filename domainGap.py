import pickle
import torch
import matplotlib.pyplot as plt
from sympy import Si
from torchmetrics.retrieval import RetrievalPrecision, RetrievalRecall
import pandas as pd
from sklearn.decomposition import TruncatedSVD


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
    data = pickle.load(open("D:\\checkpoints\\NWPU_CLIPb32_lora_inf\\nwpu_test.pkl", 'rb'))
    # data = pickle.load(open("D:\\CLIP_embeddings\\nwpu_test.pkl", 'rb'))

    # labels from name
    labels_discrete = {}
    labels = []
    k = 0
    for image in data['image']:
        label = '_'.join(image.split('_')[:-1])
        if label not in labels_discrete.keys():
            labels_discrete[label] = k
            k += 1

        labels.append(labels_discrete[label])

    print(labels_discrete)

    images = data['image_embeddings']
    texts = data['text_embeddings']
    print(images.shape, texts.shape)

    centroid_distance, pairwise_distance = gap_distance(images, texts)
    _, mean_positive_similarity, mean_negative_similarity = similarity(images, texts)
    print(mean_positive_similarity.detach().cpu().item(),
          mean_negative_similarity.detach().cpu().item(),
          centroid_distance.detach().cpu().item(),
          pairwise_distance.detach().cpu().item())

    SVD(images, texts, 'NWPU LoRA 200 epochs')
    retrieval(images, texts, 'NWPU LoRA 200 epochs')
