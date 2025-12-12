import argparse
import pickle
import torch
import pandas as pd
from tqdm import tqdm
from torchmetrics.classification import BinaryCalibrationError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=str, required=True)
    args = parser.parse_args()

    data = pickle.load(open(args.embeddings, 'rb'))
    images_ = data['image_embeddings']
    texts_ = data['text_embeddings']
    texts_ = texts_ / texts_.norm(dim=-1, keepdim=True)
    images_ = images_ / images_.norm(dim=-1, keepdim=True)

    batch_sizes = [32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32]
    temperatures = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 1]
    n = 50

    results = []
    pbar = tqdm(total=len(batch_sizes)*len(temperatures)*n)
    ECE = BinaryCalibrationError(n_bins=10, norm='l1')  # L1 norm for ECE
    CE = torch.nn.CrossEntropyLoss()

    for n in range(n):
        for i, batch_size in enumerate(batch_sizes):
            if i == 0:
                indices = torch.randperm(images_.shape[0])[:batch_size]
            else:
                # sub sample last batch
                indices = torch.randperm(batch_images.shape[0])[:batch_size]

            batch_images = images_[indices].to('cuda')
            batch_texts = texts_[indices].to('cuda')
            logits = batch_images @ batch_texts.T

            for t in temperatures:
                scaled_logits = t * logits
                probs = scaled_logits.softmax(dim=-1)
                targets = torch.arange(scaled_logits.shape[0]).to('cuda')

                loss_i = CE(scaled_logits, targets)
                loss_t = CE(scaled_logits.T, targets)
                loss = (loss_t + loss_i) / 2

                # image -> text
                confidence, pred = probs.max(dim=-1)
                acc_i = torch.sum((pred == targets).int()) / len(pred)
                ece_i = ECE(confidence, (pred == targets).int())

                # text -> image
                # pred = scaled_logits.T.softmax(dim=-1).argmax(dim=-1)
                # acc_t = torch.sum((pred == targets).int()) / len(pred)
                # ece_t = ECE(confidence, (pred == targets).int())

                output = {
                    'batch_size': scaled_logits.shape[0],
                    'temperature': t,
                    'loss': loss.detach().cpu().item(),
                    'accuracy': acc_i.detach().cpu().item(),
                    'ece': ece_i.detach().cpu().item(),
                    'mean model confidence': torch.mean(confidence).detach().cpu().item(),
                    'diagonal confidence': torch.diagonal(probs).mean().detach().cpu().item(),
                    'iteration': n,
                }
                results.append(output)
                pbar.update(1)

        df = pd.DataFrame(results)
        df.to_excel('results.xlsx', index=False)

