import seaborn
import matplotlib.pyplot as plt
import os
import pandas as pd


if __name__ == '__main__':
    root = '/nethome/recpinfo/users/fibz/data/checkpoint/vlm-finetuning/'
    experiments = [
        "GEO-DINO-48-t100-r12-composition",
        "GEO-DINO-48-t25-r12-composition", 
    ]
    # names = ['experimento 1', 'experimento 2', ]
    names = ['t100', 't25',]
    suffix = 'train_' # 'all_texts_' or 'train_' or 'val_'
    batch = ''
    n=476

    palette = ["#4C72B0",  "#55A868", '#884066']
    
    seaborn.set_theme() 
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    for i, experiment in enumerate(experiments):
        for j, suffix in enumerate([ 'val_', 'train_']):
            name = suffix + names[i] 
            path = os.path.join(root, experiment, f'{suffix}{batch}retrieval_results.csv')
            data = pd.read_csv(path)
            print(name, experiment)
            print(data)

            ax1.plot(
                data['k'][:-1], 
                data['i2t'][:-1], 
                label=names[i], 
                marker='o' if suffix == 'train_' else 'd', 
                color=palette[i],
                markersize=5,
                # markerfacecolor='white',
                markeredgewidth=1.2,
            )
            
            ax2.plot(
                data['k'][:-1], 
                data['t2i'][:-1], 
                label=name, 
                marker='o' if suffix == 'train_' else 'd', 
                color=palette[i],
                markersize=5,
                # markerfacecolor='white',
                markeredgewidth=1.2,
            )

    ax1.set_xlabel('k')
    ax1.set_ylabel('r@k')
    ax1.set_title(f'I2T')
    ax1.set_xticks([1, 5, 10, 20, 50])
   
    ax2.set_xlabel('k')
    ax2.set_ylabel('r@k')
    ax2.set_title(f'T2I')
    ax2.set_xticks([1, 5, 10, 20, 50])
    
    plt.suptitle(f'Composition N={n}, DINO LoRA, batch 48, resolution=512')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('retrieval.png')