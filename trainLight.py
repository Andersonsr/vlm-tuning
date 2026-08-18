from omegaconf import OmegaConf
import argparse
import os
from dataset.datasets import CaptionDataset, GeoDataset, GEO_INDICES, DistributedSingleDatasetBatchSampler
from model.encoders import resize_transform, crop_transform
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, ConcatDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from glob import glob
from model.createModel import createModel
import torch.distributed as dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='model/configs/CLIP_default.yaml')
    parser.add_argument('--nnodes', type=int, default=1, help='Number of nodes available to the job')
    parser.add_argument('--gpus', type=int, default=1, help='gpus per node')
    parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu', 'cpu', 'auto'], help='accelerator used to run the job')
    parser.add_argument('--name', type=str, default='test', help='run name')
    parser.add_argument('--lora_rank', type=int, default=None)
    parser.add_argument('--lora_alpha', type=int, default=None)
    parser.add_argument('--multipositive', action='store_true', default=None)
    parser.add_argument('--strategy', type=str, default='auto', choices=['fsdp', 'deepspeed_stage_2',])
    parser.add_argument('--temp', type=float, default=None, help='used to overwrite config temperature')
    parser.add_argument('--batch_size', type=int, default=None, help='use to overwrite config batch size')
    parser.add_argument('--multiresolution', action='store_true', default=False, help='use to enable multiresolution training')
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)

    seed_everything(777, workers=True)

    os.makedirs(os.path.join(conf.output_dir, args.name),  exist_ok=True)    
    conf.model.lora.backbone = conf.model.name.split(':')[-1]
    if args.multipositive is not None:
        conf.train.multi_positive = args.multipositive    

    if args.temp is not None:
        conf.model.temperature = args.temp

    if args.batch_size is not None:
        conf.train.batch_size = args.batch_size

    if args.lora_rank is not None: 
        conf.model.lora.r = args.lora_rank

    if args.lora_alpha is not None:
        conf.model.lora.alpha = args.lora_alpha
        
    global_bs = conf.train.batch_size if "WORLD_SIZE" not in os.environ.keys() else int(os.environ["WORLD_SIZE"]) * conf.train.batch_size 
    
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
        val_loader = val_dataset.get_loader(conf.train.batch_size, False)

    else:
        train_datasets = []
        for idx in conf.dataset.geo_index:

            train_dataset = GeoDataset(
                conf.dataset.root, 
                conf.dataset.train_annotation, 
                lambda x: crop_transform(x,  conf.dataset.resolutions[-1], 16), #2nd dim is the largest dim
                model.tokenize, 
                conf.dataset.geo_group,
                idx,
                size= conf.dataset.resolutions[-1], # larger than 2nd dim
                randomImage=True,
                larger=True,
                )
            train_datasets.append(train_dataset)

            if args.multiresolution:
                train_dataset = GeoDataset(
                    conf.dataset.root, 
                    conf.dataset.train_annotation, 
                    lambda x: crop_transform(x,  conf.dataset.resolutions[0], 16), 
                    model.tokenize, 
                    conf.dataset.geo_group,
                    idx,
                    size= conf.dataset.resolutions[-1], 
                    randomImage=True,
                    larger=False    
                    )
                train_datasets.append(train_dataset)
            
            
        if len(train_datasets) > 1:
            combined_dataset = ConcatDataset(train_datasets)
            sampler = DistributedSingleDatasetBatchSampler(
                dataset_lengths=[len(d) for d in train_datasets],
                batch_size=conf.train.batch_size,
                shuffle=True,
                drop_last=True,
            )
            
            train_loader = DataLoader(
                combined_dataset,
                batch_sampler=sampler,
                collate_fn=train_datasets[0].collate,
                num_workers=8,
            )

        elif len(train_datasets) == 1:
            train_loader = train_datasets[0].get_loader(conf.train.batch_size, True)
            

        val_datasets = []
        for idx in conf.dataset.geo_index_val:
            dataset = GeoDataset(
                conf.dataset.root, 
                conf.dataset.val_annotation, 
                lambda x: crop_transform(x,  conf.dataset.resolutions[-1], 16), 
                model.tokenize, 
                conf.dataset.geo_group,
                idx,
                size= conf.dataset.resolutions[-1],
                randomImage=False,
                larger=True,
                )
            
            val_datasets.append(dataset)

            if args.multiresolution:    
                dataset = GeoDataset(
                    conf.dataset.root, 
                    conf.dataset.val_annotation, 
                    lambda x: crop_transform(x,  conf.dataset.resolutions[0], 16), 
                    model.tokenize, 
                    conf.dataset.geo_group,
                    idx,
                    size= conf.dataset.resolutions[-1],
                    randomImage=True,
                    larger=False    
                    )
                val_datasets.append(dataset)
        
        
        if len(val_datasets) > 1:
            combined_dataset = ConcatDataset(val_datasets)
            lengths = [len(dataset) for dataset in val_datasets]
            sampler = DistributedSingleDatasetBatchSampler(
                dataset_lengths=[len(d) for d in val_datasets],
                batch_size=conf.train.batch_size,
                shuffle=False,
                drop_last=True,
            )
            
            val_loader = DataLoader(
                combined_dataset,
                batch_sampler=sampler,
                collate_fn=train_datasets[0].collate,
                num_workers=8,
            )
                    
        else:
            val_loader = val_datasets[0].get_loader(conf.train.batch_size, False)

    if conf.train.cooling.iterations <= 1:
        # training lenght ratio 
        training_len = (len(train_loader) //  args.gpus) * conf.train.epochs
        print(training_len, 'training length')
        model.cooling_steps = int(conf.train.cooling.iterations * training_len)
        conf.train.cooling.iterations = int(conf.train.cooling.iterations * training_len)

    conf.model.load_weights = True
    conf.output_dir = os.path.join(conf.output_dir, args.name)
    OmegaConf.save(config=conf, f=os.path.join(conf.output_dir, 'config.yaml'))
    
    # train
    print('model path ', os.path.join(conf.output_dir))
    print(model)
    
    wandb_logger = WandbLogger(project="VLM-finetuning", name=args.name)
    print(f'monitoring metric: {conf.train.monitor}')
    checkpoint_callback = ModelCheckpoint(
        monitor=conf.train.monitor,  # Quantity to monitor (e.g., "val_loss", "val_acc")
        dirpath=conf.output_dir,  # Directory to save the checkpoints
        filename="checkpoint-{epoch:02d}",  # Checkpoint file name with dynamic metrics
        save_top_k=1,  # Save the top k best models
        mode="min" if 'loss' in conf.train.monitor else "max",  # "min" for loss, "max" for accuracy
        save_last=True,  # Save the last checkpoint with a "last.ckpt" file name
    )

    callbacks = [checkpoint_callback]
    trainer = L.Trainer(
        max_epochs=conf.train.epochs,
        devices=args.gpus,
        accelerator=args.accelerator,
        num_nodes=args.nnodes,
        logger=wandb_logger,
        use_distributed_sampler=False,
        callbacks=callbacks,
        log_every_n_steps=conf.log_interval,
        strategy=args.strategy,
    )

    print('starting training')
    trainer.fit(model, train_loader, val_loader,)
    trainer.save_checkpoint(os.path.join(conf.output_dir, 'manual_save.ckpt'))