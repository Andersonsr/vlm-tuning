from omegaconf import OmegaConf
import argparse
import os
from dataset.captionDataset import CaptionDataset
import torch
from model.lora_utils import save_lora, get_lora_parameters, get_list_lora_layers
from model.createModel import createModel
import lightning as L
import pytorch_lightning as pl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='model/configs/CLIP_default.yaml')
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)

    os.makedirs(conf.output_dir, exist_ok=True)
    file = open(os.path.join(conf.output_dir, 'training.log'), 'w')
    file.close()

    conf.model.lora.backbone = conf.model.name.split(':')[-1]
    conf.model.lora.save_path = conf.output_dir
    OmegaConf.save(config=conf, f=os.path.join(conf.output_dir, 'config.yaml'))

    model = createModel(conf)
    model.learnable_parameters()

    train_dataset = CaptionDataset(conf.dataset.root, conf.dataset.train_annotation, conf.dataset.name, model.prepareImages, model.tokenize)
    val_dataset = CaptionDataset(conf.dataset.root, conf.dataset.val_annotation, conf.dataset.name, model.prepareImages, model.tokenize)
    print('train dataset size: {} val dataset size {}'.format(len(train_dataset), len(val_dataset)))

    os.makedirs(conf.output_dir, exist_ok=True)

    # train
    train_loader = train_dataset.get_loader(conf.train.batch_size)
    val_loader = val_dataset.get_loader(conf.train.batch_size)

    trainer = L.Trainer(
        max_epochs=conf.train.epochs,
        devices='auto',
        accelerator=conf.train.accelerator,
        num_nodes=conf.train.nodes,
        )
    trainer.fit(model, train_loader, val_loader)

    if conf.model.lora.apply:
        save_lora(conf.model.lora, get_list_lora_layers(conf.model.lora, model.model))

    torch.save(model.state_dict(), os.path.join(conf.output_dir, 'checkpoint.pt'))
