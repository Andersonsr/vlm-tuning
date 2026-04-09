from omegaconf import OmegaConf
import argparse
import os
from dataset.datasets import CaptionDataset, GeoDataset, GEO_INDICES
from model.createModel import createModel
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, ConcatDataset
from glob import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='model/configs/CLIP_default.yaml')
    parser.add_argument('--nnodes', type=int, default=1, help='Number of nodes available to the job')
    parser.add_argument('--gpus', type=int, default=1, help='gpus per node')
    parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu', 'cpu', 'auto'], help='accelerator used to run the job')
    parser.add_argument('--name', type=str, default='test', help='run name')
    parser.add_argument('--strategy', type=str, default='auto', choices=['fsdp', 'deepspeed_stage_2',])
    parser.add_argument('--temp', type=float, default=None, help='used to override conf model temperature') 
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)

    seed_everything(777, workers=True)

    os.makedirs(os.path.join(conf.output_dir, args.name),  exist_ok=True)    
    conf.model.lora.backbone = conf.model.name.split(':')[-1]
    if args.temp is not None:
        conf.model.temperature = args.temp

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
        val_loader = val_dataset.get_loader(5000, False)

    else:
        datasets = []
        for idx in conf.dataset.geo_index:
            train_dataset = GeoDataset(
                conf.dataset.root, 
                conf.dataset.train_annotation, 
                model.prepareImages, 
                model.tokenize, 
                conf.dataset.geo_group,
                idx,
                randomImage=True,
                )
            
            datasets.append(train_dataset)
        
        if len(datasets) > 1:
            print('multi train datasets')
            train_dataset = ConcatDataset(datasets)
            train_loader  = DataLoader(train_dataset, batch_size=conf.train.batch_size, shuffle=True) 

        elif len(datasets) == 1:
            print('single train dataset')
            train_loader = datasets[0].get_loader(conf.train.batch_size, True)
            

        val_loader = []
        for idx in conf.dataset.geo_index_val:
            dataset = GeoDataset(
                conf.dataset.root, 
                conf.dataset.val_annotation, 
                model.prepareImages, 
                model.tokenize, 
                conf.dataset.geo_group,
                idx,
                randomImage=False,
                )
            
            loader = dataset.get_loader(1411, False)
            val_loader.append(loader)
        print(f'number of validation datasets {len(val_loader)}')
    
    if conf.train.cooling.iterations <= 1:
        # training lenght ratio 
        training_len = (len(train_dataset) // (conf.train.batch_size * args.gpus)) * conf.train.epochs
        model.cooling_steps = int(conf.train.cooling.iterations * training_len)
        conf.train.cooling.iterations = int(conf.train.cooling.iterations * training_len)

    conf.model.load_weights = True
    conf.output_dir = os.path.join(conf.output_dir, args.name)
    OmegaConf.save(config=conf, f=os.path.join(conf.output_dir, 'config.yaml'))
    
    # train
    print('model path ', os.path.join(conf.output_dir))

    wandb_logger = WandbLogger(project="VLM-finetuning", name=args.name)
    checkpoint_callback = ModelCheckpoint(
        monitor=conf.train.monitor,  # Quantity to monitor (e.g., "val_loss", "val_acc")
        dirpath=conf.output_dir,  # Directory to save the checkpoints
        filename="checkpoint-{epoch:02d}",  # Checkpoint file name with dynamic metrics
        save_top_k=1,  # Save the top k best models
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
    
    trainer.fit(model, train_loader, val_loader,)
    trainer.save_checkpoint(os.path.join(conf.output_dir, 'manual_save.ckpt'))