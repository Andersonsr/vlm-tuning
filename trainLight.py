from omegaconf import OmegaConf
import argparse
import os

from torch.optim import Optimizer

from dataset.captionDataset import CaptionDataset
import torch
from model.lora_utils import save_lora, get_lora_parameters, get_list_lora_layers
from model.createModel import createModel
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, BaseFinetuning
from model.lora_utils import mark_only_lora_as_trainable


class LoraFinetuning(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module):
        mark_only_lora_as_trainable(pl_module)

    def finetune_function(self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer) -> None:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='model/configs/CLIP_default.yaml')
    parser.add_argument('--nnodes', type=int, default=1, help='Number of nodes available to the job')
    parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu', 'cpu', 'auto'], help='accelerator used to run the job')
    parser.add_argument('--name', type=str, default='test', help='run name')
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)

    os.makedirs(conf.output_dir, exist_ok=True)

    conf.model.lora.backbone = conf.model.name.split(':')[-1]
    conf.model.lora.save_path = conf.output_dir
    OmegaConf.save(config=conf, f=os.path.join(conf.output_dir, 'config.yaml'))

    model = createModel(conf)
    model.learnable_parameters()

    train_dataset = CaptionDataset(conf.dataset.root, conf.dataset.train_annotation, conf.dataset.name, model.prepareImages, model.tokenize)
    val_dataset = CaptionDataset(conf.dataset.root, conf.dataset.val_annotation, conf.dataset.name, model.prepareImages, model.tokenize)
    print('train dataset size: {} val dataset size {}'.format(len(train_dataset), len(val_dataset)))

    # train
    train_loader = train_dataset.get_loader(conf.train.batch_size, True)
    val_loader = val_dataset.get_loader(conf.train.batch_size, False)

    wandb_logger = WandbLogger(project="VLM-finetuning", name=args.name)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Quantity to monitor (e.g., "val_loss", "val_acc")
        dirpath=conf.output_dir,  # Directory to save the checkpoints
        filename="checkpoint-{epoch:02d}",  # Checkpoint file name with dynamic metrics
        save_top_k=3,  # Save the top 3 best models
        mode="min",  # "min" for loss, "max" for accuracy
        save_last=True  # Save the last checkpoint with a "last.ckpt" file name
    )

    callbacks = [checkpoint_callback]
    # if conf.model.lora.apply:
    #     callbacks.append(LoraFinetuning())

    trainer = L.Trainer(
        max_epochs=conf.train.epochs,
        devices='auto',
        accelerator=args.accelerator,
        num_nodes=args.nnodes,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=100
    )

    trainer.fit(model, train_loader, val_loader, )
