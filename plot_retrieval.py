import seaborn
import matplotlib.pyplot as plt
import os
import pandas as pd


if __name__ == '__main__':
    root = '/nethome/recpinfo/users/fibz/data/checkpoint/vlm-finetuning/'
    experiments = [
        'GEO-CLIP-large14-LoRA-batch512-original-composition', 
        'GEO-LongCLIP-large14-LoRA-batch512-original-composition',
    ]
    names = ['experimento 1', 'experimento 2', ]
    suffix = 'val_' # 'all_texts_' or 'train_' or 'val_'

    seaborn.set_theme() 
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    for i, experiment in enumerate(experiments):
        name = names[i]
        path = os.path.join(root, experiment, f'{suffix}retrieval_results.csv')
        data = pd.read_csv(path)
        print(name, data)

        ax1.plot(data['k'], data['i2t'], label=name)
        ax2.plot(data['k'], data['t2i'], label=name)

    ax1.set_xlabel('k')
    ax1.set_ylabel('r@k')
    ax1.set_title(f'I2T')

    ax2.set_xlabel('k')
    ax2.set_ylabel('r@k')
    ax2.set_title(f'T2I')

    plt.suptitle('Composition BS=512')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('retrieval.png')