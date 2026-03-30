import seaborn
import matplotlib.pyplot as plt
import os
import pandas as pd


if __name__ == '__main__':
    root = '/nethome/recpinfo/users/fibz/data/checkpoint/vlm-finetuning/'
    experiments = ['NWPU-single-CLIP-base32-LoRA-batch1024-original', 'NWPU-single-CLIP-base32-LoRA-batch1024-originalb', 'NWPU-single-CLIP-base32-LoRA-batch1024-t50']
    names = ['t=100 run 1', 't=100 run 2', 't=50']
    suffix = 'val_' # 'all_texts_' or 'train_' or 'val_'

    seaborn.set_theme() 
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    for i, experiment in enumerate(experiments):
        name = names[i]
        path = os.path.join(root, experiment, f'{suffix}retrieval_results.csv')
        data = pd.read_csv(path)
        print(name, data)

        ax1.plot(data['k'][:4], data['i2t'][:4], label=name)
        ax2.plot(data['k'][:4], data['t2i'][:4], label=name)

    ax1.set_xlabel('k')
    ax1.set_ylabel('r@k')
    ax1.set_title(f'I2T')

    ax2.set_xlabel('k')
    ax2.set_ylabel('r@k')
    ax2.set_title(f'T2I')

    plt.suptitle('CLIP-B/32 BS=1024')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('retrieval.png')