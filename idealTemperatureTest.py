import math
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import torch


if __name__ == '__main__':
    data = pickle.load(open("D:\\CLIP_embeddings\\nwpu_test.pkl", 'rb'))

    images = data['image_embeddings']
    texts = data['text_embeddings']

    images = images / images.norm(dim=-1, keepdim=True)
    texts = texts / texts.norm(dim=-1, keepdim=True)

    logits = images @ texts.T
    logits = torch.cat([logits, logits.T])
    print(logits.shape)

    # max_logits = np.max(logits.cpu().numpy(), axis=-1)

    mean_logits = stats.trim_mean(logits, 0.2, axis=-1)
    std_logits = stats.median_abs_deviation(logits.cpu().detach().numpy(), scale='normal', axis=-1)
    max_logits = mean_logits + std_logits * 5.325
    print(np.mean(max_logits), np.mean(mean_logits), np.mean(std_logits))

    t1 = 100
    n1 = 32768
    n2 = 64

    dk = max_logits - mean_logits

    a = std_logits**2/2
    b = -dk
    c = dk*t1 - (std_logits*t1)**2/2 + math.log((n2-1)/(n1-1))

    # kmads = dk / std_logits
    # print('dists', kmads.mean())

    delta = np.sqrt(b ** 2 - 4 * a * c)
    T21 = (-b + delta) / 2 / a
    T22 = (-b - delta) / 2 / a
    print('T2', np.mean(T22))
    print('T1', np.mean(T21))
