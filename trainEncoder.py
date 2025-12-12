import sys
from loratorch import register_model_param_after_backward, lora_state_dict
from omegaconf import OmegaConf
import argparse
import os
from dataset.captionDataset import CaptionDataset
import torch
from math import ceil
from model.createModel import createModel
from torch.optim import AdamW
from tqdm import tqdm

PBAR = None


def trainEpoch(loader, model, optim, conf, device):
    step_loss = []
    step_accuracy = []
    step_confidence = []

    for i, batch in enumerate(loader):
        # print((batch['images']))
        images = model.prepareImages(batch['images']).to(device)
        texts = model.tokenize(batch['captions']).to(device)

        loss, scaled_logits = model(images, texts)

        optim.zero_grad()
        loss.backward()
        optim.step()
        register_model_param_after_backward(model)

        probs = scaled_logits.softmax(dim=-1)
        confidence, pred = probs.max(dim=-1)
        targets = torch.arange(probs.shape[0]).to(probs.device)
        accuracy = torch.sum((pred == targets).int()) / probs.shape[0]

        step_loss.append(loss.detach().cpu().item())
        step_accuracy.append(accuracy.detach().cpu().item())
        step_confidence.append(torch.mean(confidence).detach().cpu().item())
        PBAR.update(1)

        if (i + 1) % conf.log_interval == 0 or i+1 == len(loader):
            mean_loss = sum(step_loss) / len(step_loss)
            mean_accuracy = sum(step_accuracy) / len(step_accuracy)
            mean_confidence = sum(step_confidence) / len(step_confidence)

            file = open(os.path.join(conf.output_dir, 'training.log'), 'a')
            file.write('training loss:'+str(mean_loss) + '\n')
            file.write('temperature:' + str(torch.exp(model.model.logit_scale).cpu().item()) + '\n')
            file.write('confidence:' + str(mean_confidence) + '\n')
            file.write('accuracy:' + str(mean_accuracy) + '\n')
            file.close()

            step_loss = []
            step_accuracy = []
            step_confidence = []


def validationEpoch(loader, model, conf, device):
    step_loss = []

    for i, batch in enumerate(loader):
        images = model.prepareImages(batch['images']).to(device)
        texts = model.tokenize(batch['captions']).to(device)
        with torch.no_grad():
            loss, _ = model(images, texts)
            step_loss.append(loss.detach().cpu().item())

        PBAR.update(1)

    mean_loss = sum(step_loss) / len(step_loss)
    file = open(os.path.join(conf.output_dir, 'training.log'), 'a')
    file.write('validation loss:'+str(mean_loss) + '\n')
    return mean_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='model/configs/CLIP_default.yaml')
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(conf.output_dir, exist_ok=True)
    file = open(os.path.join(conf.output_dir, 'training.log'), 'w')
    file.close()

    model = createModel(conf.model).to(device)
    model.learnable_parameters()
    # model.temperature = model.temperature.to(device)

    train_dataset = CaptionDataset(conf.dataset.root, conf.dataset.train_annotation, conf.dataset.name)
    val_dataset = CaptionDataset(conf.dataset.root, conf.dataset.val_annotation, conf.dataset.name)
    print('train dataset size: {} val dataset size {}'.format(len(train_dataset), len(val_dataset)))

    optimizer = AdamW(model.parameters(), lr=conf.train.learning_rate)
    os.makedirs(conf.output_dir, exist_ok=True)
    conf.model.load_weights = os.path.join(conf.output_dir, 'checkpoint.pt')
    OmegaConf.save(config=conf, f=os.path.join(conf.output_dir, 'config.yaml'))

    train_steps = len(train_dataset) / conf.train.batch_size
    val_steps = len(val_dataset) / conf.train.batch_size
    PBAR = tqdm(total=(ceil(train_steps) + ceil(val_steps)) * conf.train.epochs)
    best_val_loss = float('inf')

    for epoch in range(conf.train.epochs):
        train_loader = train_dataset.get_loader(conf.train.batch_size)
        trainEpoch(train_loader, model, optimizer, conf, device)

        val_loader = val_dataset.get_loader(conf.train.batch_size)
        loss = validationEpoch(val_loader, model, conf, device)

        if conf.model.apply_lora:
            torch.save(lora_state_dict(model), os.path.join(conf.output_dir, 'lora.pt'))

        torch.save(model.state_dict(), os.path.join(conf.output_dir, 'checkpoint.pt'))
