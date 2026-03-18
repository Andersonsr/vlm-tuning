from omegaconf import OmegaConf
import argparse
import os
from dataset.datasets import CaptionDataset, GeoDataset, GEO_INDICES
from model.createModel import createModel
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from clip.model import ResidualAttentionBlock

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='model/configs/CLIP_default.yaml')
    parser.add_argument('--nnodes', type=int, default=1, help='Number of nodes available to the job')
    parser.add_argument('--gpus', type=int, default=1, help='gpus per node')
    parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu', 'cpu', 'auto'], help='accelerator used to run the job')
    parser.add_argument('--name', type=str, default='test', help='run name')
    parser.add_argument('--strategy', type=str, default='auto', choices=['fsdp', 'deepspeed_stage_2',]) 
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)

    os.makedirs(os.path.join(conf.output_dir, args.name),  exist_ok=True)    
    conf.model.lora.backbone = conf.model.name.split(':')[-1]
    
    model = createModel(conf)
    model.learnable_parameters()
    if conf.dataset.name != 'geo':
        train_dataset = CaptionDataset(
            conf.dataset.root, 
            conf.dataset.train_annotation, 
            conf.dataset.name, 
            model.prepareImages, 
            model.tokenize, 
            random=conf.dataset.random
            )
        
        val_dataset = CaptionDataset(conf.dataset.root, conf.dataset.val_annotation, conf.dataset.name, model.prepareImages, model.tokenize, random=False)
        print('train dataset size: {} val dataset size {}'.format(len(train_dataset), len(val_dataset)))
        train_loader = train_dataset.get_loader(conf.train.batch_size, True)
        val_loader = val_dataset.get_loader(3150, False)
        # val_loader = val_dataset.get_loader(conf.train.batch_size, False)

    else:
        # TODO: load data with multiple validation datasets
        train_dataset = GeoDataset(
            conf.dataset.root, 
            conf.dataset.train_annotation, 
            model.prepareImages, 
            model.tokenize, 
            conf.dataset.geo_group,
            conf.dataset.geo_index,
            randomImage=True,
            )
        train_loader = train_dataset.get_loader(conf.train.batch_size, True)
        
        val_loader = []
        for key in conf.dataset.geo_index_val:
            dataset = GeoDataset(
                conf.dataset.root, 
                conf.dataset.val_annotation, 
                model.prepareImages, 
                model.tokenize, 
                conf.dataset.geo_group,
                [key],
                randomImage=False,
                )
            loader = dataset.get_loader(3150, False)
            val_loader.append(loader)
        
    if conf.train.cooling.iterations <= 1:
        # training lenght ratio 
        training_len = (len(train_dataset) // (conf.train.batch_size * args.gpus)) * conf.train.epochs
        model.cooling_steps = int(conf.train.cooling.iterations * training_len)
        conf.train.cooling.iterations = int(conf.train.cooling.iterations * training_len)

    conf.model.lora.save_path = conf.output_dir
    OmegaConf.save(config=conf, f=os.path.join(conf.output_dir, args.name, 'config.yaml'))
    
    # train
    print('model path ', os.path.join(conf.output_dir, args.name))

    wandb_logger = WandbLogger(project="VLM-finetuning", name=args.name)
    checkpoint_callback = ModelCheckpoint(
        monitor=conf.train.monitor,  # Quantity to monitor (e.g., "val_loss", "val_acc")
        dirpath=os.path.join(conf.output_dir, args.name),  # Directory to save the checkpoints
        filename="checkpoint-{epoch:02d}",  # Checkpoint file name with dynamic metrics
        save_top_k=3,  # Save the top 3 best models
        mode="min",  # "min" for loss, "max" for accuracy
        save_last=True,  # Save the last checkpoint with a "last.ckpt" file name
    )

    callbacks = [checkpoint_callback]
    
    trainer = L.Trainer(
        max_epochs=conf.train.epochs,
        devices=args.gpus,
        accelerator=args.accelerator,
        num_nodes=args.nnodes,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=conf.log_interval,
        strategy=args.strategy,
    )
    trainer.fit(model, train_loader, val_loader, )
    