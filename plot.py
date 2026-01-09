import matplotlib.pyplot as plt
import numpy as np
import os
from math import ceil
import seaborn as sns
import pandas as pd
sns.set_theme()


def plot_metrics_single_iteration(file):
    df = pd.read_excel(file)
    # print(df[df['batch_size'] == 32])
    df = pd.crosstab(index=df['temperature'], columns=df['batch_size'], values=df['loss'], aggfunc='sum')
    sns.lineplot(data=df)
    plt.ylabel('Mean confidence')
    plt.title('Image positive pair prediction')
    plt.show()


def plot_metrics(file):
    df = pd.read_excel(file)
    tab = pd.crosstab(index=df['temperature'], columns=df['batch_size'], values=df['mean model confidence'], aggfunc='mean')
    print(tab)
    sns.lineplot(data=tab)
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.ylabel('Model confidence')
    plt.title('Mean image positive pair prediction 50 batches NWPU')
    plt.tight_layout()
    plt.ylim(0, 0.8)
    plt.show()


def plot_train(log, name):
    f = open(log, 'r')
    data = {}
    for line in f:
        split = line.split(':')
        if split[0] not in data:
            data[split[0]] = [float(split[1])]
        else:
            data[split[0]].append(float(split[1]))

    val_loss = data['validation loss']
    train_loss = data['training loss']
    print('min temperature: ', min(data['temperature']))

    val_interval = len(train_loss) / len(val_loss)

    # plt.plot([(i+1)*val_interval for i in range(len(val_loss))], val_loss, label='validation loss')
    # plt.plot(range(1, len(train_loss)+1), train_loss, label='training loss')
    plt.plot(range(len(data['accuracy'])), np.array(data['accuracy']) * 100, label='accuracy * 100')
    plt.plot(range(len(data['confidence'])), np.array(data['confidence']) * 100, label='confidence * 100')
    plt.plot(range(len(data['temperature'])), data['temperature'], label='temperature')

    plt.legend()
    plt.title(f'training log: {name}')
    plt.xlabel('step')
    plt.savefig(os.path.join('plots', 'training_log ' + name + '.png'))


if __name__ == '__main__':
    # plot_metrics('results_nwpu.xlsx')
    plot_train('D:\\checkpoints\\NWPU_CLIPb32_lora2_temp\\training.log', 'NWPU LoRA2+learnable temperature')
    