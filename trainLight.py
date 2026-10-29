from omegaconf import OmegaConf
import argparse
import inspect
import os
import uuid
from collections import defaultdict
from dataset.datasets import CaptionDataset, GeoDataset, GEO_INDICES, DistributedSingleDatasetBatchSampler
from dataset.cc3m import CC3MDataset
from model.encoders import resize_transform, crop_transform
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, ConcatDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DeepSpeedStrategy
from glob import glob
from model.createModel import createModel
from model.encoders import OPTIMIZER_DEFAULTS
import torch.distributed as dist


def newest_checkpoint(folder, prune=True):
    """Most recent of the checkpoints a run leaves in its folder: hpc_ckpt_<n>.ckpt (saved by lightning when slurm
    sends the requeue signal) and last.ckpt (end of every epoch); None if there is none. On rank 0, older hpc
    checkpoints are removed, they are superseded."""
    candidates = []
    for path in glob(os.path.join(folder, 'hpc_ckpt_*.ckpt')) + [os.path.join(folder, 'last.ckpt')]:
        try:
            candidates.append((os.path.getmtime(path), path))
        except OSError:  # missing (or just removed by rank 0)
            pass
    if not candidates:
        return None
    newest = max(candidates)[1]
    if prune and is_rank_zero():
        for _, path in candidates:
            if path != newest and os.path.basename(path).startswith('hpc_ckpt_') and os.path.isfile(path):
                os.remove(path)
    return newest


# entries every run config has; a config file may leave them out (its value is then the default here)
DEFAULTS = {
    'name': 'test',
    'wandb_id': None,
    'model': {'vision_head_only': False, 'average_local': False},
    'train': {'nnodes': 1, 'gpus': 1, 'accelerator': 'gpu', 'strategy': 'auto', 'init_weights': None,
              'optimizer': OPTIMIZER_DEFAULTS},
    'dataset': {'multiresolution': False, 'geometric_aug': False, 'spectral_aug': False, 'p_spectral': 0.8,
                'nlm_aug': False, 'num_workers': 8},
}

# command line option -> config entry it sets (the options are shortcuts for --set)
OPTION_KEYS = {
    'name': 'name',
    'nnodes': 'train.nnodes',
    'gpus': 'train.gpus',
    'accelerator': 'train.accelerator',
    'strategy': 'train.strategy',
    'lora_rank': 'model.lora.r',
    'lora_alpha': 'model.lora.alpha',
    'multipositive': 'train.multi_positive',
    'lr': 'train.learning_rate',
    'temp': 'model.temperature',
    'loss': 'train.loss',
    'sigmoid_bias': 'model.sigmoid_bias',
    'batch_size': 'train.batch_size',
    'vision_head': 'model.vision_head_only',
    'average_local': 'model.average_local',
    'backbone_weights': 'model.backbone_weights',
    'backbone_branch': 'model.backbone_branch',
    'multiresolution': 'dataset.multiresolution',
    'geometric_aug': 'dataset.geometric_aug',
    'spectral_aug': 'dataset.spectral_aug',
    'p_spectral': 'dataset.p_spectral',
    'nlm_aug': 'dataset.nlm_aug',
}


def is_rank_zero():
    # before the trainer exists: slurm task id, torchrun rank, or the local rank of lightning's spawned processes
    return int(os.environ.get('SLURM_PROCID', os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))) == 0


def overrides(args):
    """Config entries set on the command line: the options of OPTION_KEYS that were given, then --set."""
    entries = [f'{key}={getattr(args, option)}' for option, key in OPTION_KEYS.items() if getattr(args, option) is not None]
    broken = [item for item in args.set if '=' not in item]
    if broken:
        # e.g. a value with a space split by the shell: --set train.monitor=a b -> 'train.monitor=a', 'b'
        raise ValueError(f'--set entries must be KEY=VALUE, got {broken} (a value with spaces split by the shell? quote it)')
    return OmegaConf.from_dotlist(entries + list(args.set))


def resume_checkpoint(args):
    """The checkpoint given to --resume, None for a run folder or --resume alone (lightning and deepspeed checkpoints
    end in .ckpt, deepspeed's are folders)."""
    if args.resume in (None, 'auto'):
        return None
    path = os.path.normpath(args.resume)
    return None if os.path.isdir(path) and not path.endswith('.ckpt') else path


def run_folder(args):
    """Folder of the run to resume: --resume <run folder | checkpoint>, else <output_dir of --config>/<name>."""
    if resume_checkpoint(args):
        return os.path.dirname(resume_checkpoint(args))
    if args.resume not in (None, 'auto'):
        return args.resume
    conf = OmegaConf.merge(DEFAULTS, OmegaConf.load(args.config), overrides(args))
    return os.path.join(conf.output_dir, conf.name)


def flatten(tree, prefix=''):
    """(dotted key, value) of every leaf of a nested dict."""
    for k, v in tree.items():
        if isinstance(v, dict):
            yield from flatten(v, f'{prefix}{k}.')
        else:
            yield f'{prefix}{k}', v


def save_config(conf, path):
    tmp = path + '.tmp'  # written whole, the other ranks may be reading it
    OmegaConf.save(config=conf, f=tmp)
    os.replace(tmp, path)


def run_config(args):
    """(config, run folder, resume?): a new run's config is --config with the options on top; a resumed one
    (--resume, or --requeue when the run folder has a saved config) is its saved config.yaml."""
    resume = args.resume is not None
    run_dir = run_folder(args) if resume or args.requeue else None
    saved = os.path.join(run_dir, 'config.yaml') if run_dir else None
    if resume and not os.path.isfile(saved):
        raise FileNotFoundError(f'--resume: no saved config {saved}')
    resume = resume or (args.requeue and os.path.isfile(saved))

    if resume:
        # the run as it was saved; options given now are applied on top, so they change the run: warn
        conf = OmegaConf.load(saved)
        changes = overrides(args)
        if args.name is not None and args.name != conf.name:
            raise ValueError(f'--name {args.name} does not match the run being resumed ({conf.name})')
        # name and output_dir only locate the run (the saved output_dir is the run folder itself)
        flat = [(k, v) for k, v in flatten(OmegaConf.to_container(changes)) if k not in ('name', 'output_dir')]
        changed = {k: v for k, v in flat if OmegaConf.select(conf, k) != v}
        if changed:
            print('WARNING: resuming with config changes:', changed)
            conf = OmegaConf.merge(conf, OmegaConf.from_dotlist([f'{k}={v}' for k, v in changed.items()]))
        # weights come from the lightning checkpoint, on top of what the run started from
        conf.model.load_weights = conf.train.init_weights or False
    else:
        conf = OmegaConf.merge(DEFAULTS, OmegaConf.load(args.config), overrides(args))
        run_dir = os.path.join(conf.output_dir, conf.name)
        if newest_checkpoint(run_dir, prune=False) is not None:
            raise FileExistsError(f'{run_dir} already has checkpoints: use --resume (or --requeue) to continue that run, '
                                  'or another --name')
        conf.train.init_weights = conf.model.load_weights or None
        conf.model.lora.backbone = conf.model.name.split(':')[-1]
        if conf.wandb_id is None:
            conf.wandb_id = uuid.uuid4().hex[:12]  # saved with the config, so a resumed run logs to the same wandb run

    return conf, run_dir, resume


def build_parser():
    parser = argparse.ArgumentParser(description='Contrastive training. Every option below (except --config, --resume, '
                                     '--requeue) sets a config entry and is saved with the run in <output_dir>/<name>/config.yaml; '
                                     '--resume and --requeue rebuild the run from that file.')
    parser.add_argument('--config', type=str, default='model/configs/CLIP_default.yaml', help='config of a new run')
    parser.add_argument('--resume', nargs='?', const='auto', default=None, metavar='RUN',
                        help='resume a run from its saved config.yaml and newest checkpoint: --resume <run folder>, '
                             '--resume <checkpoint>, or --resume alone for <output_dir of --config>/<--name>')
    parser.add_argument('--requeue', action='store_true', default=False,
                        help='for (re)queued slurm jobs: resume <output_dir>/<name> as --resume does if it has a saved '
                             'config, else start it from scratch')
    parser.add_argument('--set', nargs='+', default=[], metavar='KEY=VALUE',
                        help='overwrite any config entry, e.g. --set model.aligner.vision=mlp model.lora.apply=true')
    parser.add_argument('--name', type=str, default=None, help='run name, folder under output_dir (name)')
    parser.add_argument('--nnodes', type=int, default=None, help='number of nodes (train.nnodes)')
    parser.add_argument('--gpus', type=int, default=None, help='gpus per node (train.gpus)')
    parser.add_argument('--accelerator', type=str, default=None, choices=['gpu', 'cpu', 'auto'], help='train.accelerator')
    parser.add_argument('--strategy', type=str, default=None, choices=['auto', 'fsdp', 'deepspeed_stage_2'], help='train.strategy')
    parser.add_argument('--lora_rank', type=int, default=None, help='model.lora.r')
    parser.add_argument('--lora_alpha', type=int, default=None, help='model.lora.alpha')
    parser.add_argument('--multipositive', action='store_true', default=None, help='train.multi_positive')
    parser.add_argument('--lr', default=None, type=float, help='train.learning_rate')
    parser.add_argument('--temp', type=float, default=None, help='initial temperature (model.temperature)')
    parser.add_argument('--loss', type=str, default=None, choices=['contrastive', 'siglip'], help='train.loss')
    parser.add_argument('--sigmoid_bias', type=float, default=None, help='initial sigmoid bias of the siglip loss (model.sigmoid_bias)')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size per gpu (train.batch_size)')
    parser.add_argument('--vision_head', default=None, action='store_true', help='DINOtxt: train the vision head only (model.vision_head_only)')
    parser.add_argument('--average_local', default=None, action='store_true', help='DINOtxt: average of the patch embeddings with the cls embedding (model.average_local)')
    parser.add_argument('--backbone_weights', type=str, default=None, help="DINOtxt only: DINOv3 SSL checkpoint whose backbone replaces DINO-TXT's (model.backbone_weights)")
    parser.add_argument('--backbone_branch', type=str, default=None, choices=['teacher', 'student'], help='branch of --backbone_weights to load (model.backbone_branch)')
    parser.add_argument('--multiresolution', action='store_true', default=None, help='train and validate on both dataset.resolutions (dataset.multiresolution)')
    parser.add_argument('--geometric_aug', action='store_true', default=None, help='random crop shift, horizontal flip and 90 degree rotations in training (dataset.geometric_aug)')
    parser.add_argument('--spectral_aug', action='store_true', default=None, help='equalization, contrast, sharpen, dichromacy, color cast, vignette, brightness in training (dataset.spectral_aug)')
    parser.add_argument('--p_spectral', type=float, default=None, help='probability of a spectral augmentation (dataset.p_spectral)')
    parser.add_argument('--nlm_aug', action='store_true', default=None, help='NLM denoising augmentation in training, slow, needs opencv (dataset.nlm_aug)')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()

    conf, run_dir, resume = run_config(args)
    print(f"run {conf.name}: {'resumed from ' + os.path.join(run_dir, 'config.yaml') if resume else 'new, config ' + args.config}")
    seed_everything(777, workers=True)
    os.makedirs(run_dir, exist_ok=True)

    global_bs = conf.train.batch_size if "WORLD_SIZE" not in os.environ.keys() else int(os.environ["WORLD_SIZE"]) * conf.train.batch_size 
    
    model = createModel(conf)
    model.learnable_parameters()

    groups = defaultdict(int)

    for name, param in model.named_parameters():
        if param.requires_grad:
            # adjust this depending on your module hierarchy
            key = ".".join(name.replace('visual.transformer', 'visual').split(".")[:3])
            groups[key] += param.numel()

    for name, n in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"{n/1e6:8.3f}M  {name}")

    if conf.dataset.name == 'cc3m':
        train_dataset = CC3MDataset(conf.dataset.root, 'train', model.prepareImages, model.tokenize)
        val_dataset = CC3MDataset(conf.dataset.root, 'validation', model.prepareImages, model.tokenize)
        print('train dataset size: {} val dataset size {}'.format(len(train_dataset), len(val_dataset)))

        # the sampler shards the batches over the gpus (use_distributed_sampler=False)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=DistributedSingleDatasetBatchSampler(
                dataset_lengths=[len(train_dataset)],
                batch_size=conf.train.batch_size,
                shuffle=True,
                drop_last=True,
            ),
            collate_fn=train_dataset.collate,
            num_workers=conf.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=DistributedSingleDatasetBatchSampler(
                dataset_lengths=[len(val_dataset)],
                batch_size=conf.train.batch_size,
                shuffle=False,
                drop_last=False,  # every validation sample
            ),
            collate_fn=val_dataset.collate,
            num_workers=conf.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    elif conf.dataset.name != 'geo':
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
        geometric_aug = getattr(conf.dataset, 'geometric_aug', False)
        spectral_aug = getattr(conf.dataset, 'spectral_aug', False)
        nlm_aug = getattr(conf.dataset, 'nlm_aug', False)
        p_spectral = getattr(conf.dataset, 'p_spectral', 0.8)
        for idx in conf.dataset.geo_index:

            train_dataset = GeoDataset(
                conf.dataset.root, 
                conf.dataset.train_annotation, 
                lambda x: crop_transform(x,  conf.dataset.resolutions[-1], 16, geometric=geometric_aug, spectral=spectral_aug, nlm=nlm_aug, p_spectral=p_spectral), # 2nd dim is the largest dim
                model.tokenize, 
                conf.dataset.geo_group,
                idx,
                size= conf.dataset.resolutions[-1], # larger than 2nd dim
                randomImage=True,
                larger=True,
                )
            train_datasets.append(train_dataset)

            if conf.dataset.multiresolution:
                train_dataset = GeoDataset(
                    conf.dataset.root, 
                    conf.dataset.train_annotation, 
                    lambda x: crop_transform(x,  conf.dataset.resolutions[0], 16, geometric=geometric_aug, spectral=spectral_aug, nlm=nlm_aug, p_spectral=p_spectral), 
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
            

        # one validation loader per field (description type), with both resolutions when multiresolution is on: the
        # metrics are logged per field (<field>_val_loss, '<field>_t2i r@5', ...) and averaged over the fields (val_loss,
        # 'mean_t2i r@5', ...). Batches never mix resolutions (the sampler draws each from a single dataset)
        val_loader = []
        for idx in conf.dataset.geo_index_val:
            sizes = [(conf.dataset.resolutions[-1], True)]
            if conf.dataset.multiresolution:
                sizes.append((conf.dataset.resolutions[0], False))
            field_datasets = [
                GeoDataset(
                    conf.dataset.root,
                    conf.dataset.val_annotation,
                    lambda x, crop=crop: crop_transform(x, crop, 16),
                    model.tokenize,
                    conf.dataset.geo_group,
                    idx,
                    size=conf.dataset.resolutions[-1],
                    randomImage=False,
                    larger=larger,
                    )
                for crop, larger in sizes
            ]
            print('validation {}: {} samples'.format(GEO_INDICES[idx], [len(d) for d in field_datasets]))
            val_loader.append(DataLoader(
                ConcatDataset(field_datasets),
                batch_sampler=DistributedSingleDatasetBatchSampler(
                    dataset_lengths=[len(d) for d in field_datasets],
                    batch_size=conf.train.batch_size,
                    shuffle=False,
                    drop_last=False,  # every validation sample
                ),
                collate_fn=field_datasets[0].collate,
                num_workers=8,
            ))

    if conf.train.cooling.iterations <= 1:
        # training lenght ratio 
        training_len = (len(train_loader) //  conf.train.gpus) * conf.train.epochs
        print(training_len, 'training length')
        model.cooling_steps = int(conf.train.cooling.iterations * training_len)
        conf.train.cooling.iterations = int(conf.train.cooling.iterations * training_len)

    # the saved config describes the run folder (output_dir) and loads its weights, as the evaluation scripts read it;
    # --resume rebuilds the run from it
    conf.model.load_weights = True
    conf.output_dir = run_dir
    if is_rank_zero():
        save_config(conf, os.path.join(run_dir, 'config.yaml'))
    
    # train
    print('model path ', os.path.join(conf.output_dir))
    print(model)
    
    # the id is saved in the config: a resumed or requeued run continues its wandb run
    wandb_logger = WandbLogger(project="VLM-finetuning", name=conf.name, id=conf.wandb_id, resume='allow')
    print(f'monitoring metric: {conf.train.monitor}')
    checkpoint_callback = ModelCheckpoint(
        monitor=conf.train.monitor,  # Quantity to monitor (e.g., "val_loss", "val_acc")
        dirpath=conf.output_dir,  # Directory to save the checkpoints
        filename="checkpoint-{epoch:02d}",  # Checkpoint file name with dynamic metrics
        save_top_k=1,  # Save the top k best models
        mode="min" if 'loss' in conf.train.monitor else "max",  # "min" for loss, "max" for accuracy
        save_last=True,  # Save the last checkpoint with a "last.ckpt" file name
    )

    callbacks = [checkpoint_callback]  # the learning rate is logged by the model (learning_rate, head_learning_rate)

    strategy = conf.train.strategy
    if strategy == 'deepspeed_stage_2' and (conf.model.lora.apply or conf.model.name.split(':')[0] == 'DualEncoder'):
        # deepspeed writes its own shards, so frozen backbone weights must be excluded here
        strategy = DeepSpeedStrategy(stage=2, exclude_frozen_parameters=True)

    trainer = L.Trainer(
        max_epochs=conf.train.epochs,
        devices=conf.train.gpus,
        accelerator=conf.train.accelerator,
        num_nodes=conf.train.nnodes,
        logger=wandb_logger,
        use_distributed_sampler=False,
        callbacks=callbacks,
        log_every_n_steps=conf.log_interval,
        gradient_clip_val=conf.train.optimizer.grad_clip,
        strategy=strategy,
        default_root_dir=conf.output_dir,  # where the slurm requeue signal handler saves hpc_ckpt_*.ckpt
    )

    print('starting training')
    ckpt_path = None
    if resume:
        ckpt_path = resume_checkpoint(args) or newest_checkpoint(run_dir)
        if ckpt_path is None:
            print('no checkpoint in', run_dir, 'yet: training from the start with the saved config')
        else:
            print('resuming from', ckpt_path)
    # the checkpoints hold the omegaconf config (hyperparameters), which torch's weights_only loading rejects
    fit_kwargs = {'weights_only': False} if ckpt_path and 'weights_only' in inspect.signature(trainer.fit).parameters else {}
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path, **fit_kwargs)
    trainer.save_checkpoint(os.path.join(conf.output_dir, 'manual_save.ckpt'))