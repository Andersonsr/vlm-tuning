import pickle
from omegaconf import OmegaConf
import argparse
import os
from dataset.datasets import CaptionDataset
import torch
from model.createModel import createModel
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from dataset.datasets import CaptionDataset, GeoDataset, GEO_INDICES, DistributedSingleDatasetBatchSampler
from model.encoders import resize_transform, crop_transform

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--dataset_root', type=str, default='D:\\datasets\\coco_2017\\')
    parser.add_argument('--annotations', type=str, default='train.json')
    parser.add_argument('--save_path', type=str, default='D:\\embeddings\\default_path.pkl')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_config', type=str, default='model/configs/CLIP_default.yaml')
    parser.add_argument('--all_texts', action='store_true', default=False)

    args = parser.parse_args()

    conf = OmegaConf.load(args.model_config)
    model = createModel(conf).to(device)
    model.eval()

    root = args.dataset_root
    annotations = args.annotations
    
    if args.dataset != 'geo':
        dataset = CaptionDataset(
            root, 
            annotations, 
            args.dataset,
            model.prepareImages, 
            model.tokenize, 
            random=False,
            all_texts=args.all_texts,
            )
       
        loader = dataset.get_loader(args.batch_size, False)

    else:
        datasets = []
        for idx in conf.dataset.geo_index_val:
            dataset = GeoDataset(
                root, 
                annotations, 
                lambda x: crop_transform(x,  conf.dataset.resolutions[-1], 16), 
                model.tokenize, 
                conf.dataset.geo_group,
                idx,
                size= conf.dataset.resolutions[-1],
                randomImage=False,
                larger=True,
                )
            
            datasets.append(dataset)

            dataset = GeoDataset(
                root, 
                annotations, 
                lambda x: crop_transform(x,  conf.dataset.resolutions[0], 16), 
                model.tokenize, 
                conf.dataset.geo_group,
                idx,
                size= conf.dataset.resolutions[-1],
                randomImage=True,
                larger=False    
                )
            datasets.append(dataset)
    
        if len(datasets) > 1:
            combined_dataset = ConcatDataset(datasets)
            lengths = [len(dataset) for dataset in datasets]
            sampler = DistributedSingleDatasetBatchSampler(
                dataset_lengths=[len(d) for d in datasets],
                batch_size=conf.train.batch_size,
                shuffle=True,
                drop_last=True,
            )
            
            loader = DataLoader(
                combined_dataset,
                batch_sampler=sampler,
                collate_fn=datasets[0].collate,
                num_workers=8,
            )
                    
        else:
            loader = datasets[0].get_loader(conf.train.batch_size, False)


    images_emb = None
    texts_emb = None
    captions = []
    images = []
    images_names = []
    labels = []

    for batch in tqdm(loader):
        with torch.no_grad():
            if args.all_texts:
                bs, caps, dim = batch['tokens'].shape
                tokens = batch['tokens'].view(-1, dim)
                txt_embeds = model.model.encode_text(tokens.to(device))
                txt_embeds = txt_embeds.view(bs, caps, -1)       
                
            else:
                txt_embeds = model.model.encode_text(batch['tokens'].to(device))
            
            im_embeds = model.model.encode_image(batch['image'].to(device))

            if images_emb is None:
                images_emb = im_embeds.detach().cpu()
                texts_emb = txt_embeds.detach().cpu()

            else:
                images_emb = torch.concat((images_emb, im_embeds.detach().cpu()), dim=0)
                texts_emb = torch.concat((texts_emb, txt_embeds.detach().cpu()), dim=0)
            
            captions += batch['text']
            images_names += batch['image_name']
            if 'labels' in batch.keys():
                labels += batch['labels']
    
    print('features shape', images_emb.shape, texts_emb.shape)
    # print(captions[:10])
    # print(images_names[:10])
    
    data = {'captions': captions, 'image_embeddings': images_emb, 'text_embeddings': texts_emb, 'image_name': images_names,}
    if len(labels) > 0:
        data['labels'] = labels
    
    pickle.dump(data, open(args.save_path, 'wb'))
    print('saved embeddings at', args.save_path)
