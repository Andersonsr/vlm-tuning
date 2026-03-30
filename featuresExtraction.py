import pickle

from omegaconf import OmegaConf
import argparse
import os
from dataset.datasets import CaptionDataset
import torch
from model.createModel import createModel
from torch.optim import AdamW
from tqdm import tqdm


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--dataset_root', type=str, default='D:\\datasets\\coco_2017\\')
    parser.add_argument('--annotations', type=str, default='train.json')
    parser.add_argument('--save_path', type=str, default='D:\\embeddings\\default_path.pkl')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_config', type=str, default='model/configs/CLIP_default.yaml')

    args = parser.parse_args()

    conf = OmegaConf.load(args.model_config)
    model = createModel(conf).to(device)
    model.eval()

    root = args.dataset_root
    annotations = args.annotations

    dataset = CaptionDataset(
        root, 
        annotations, 
        args.dataset,
        model.prepareImages, 
        model.tokenize, 
        random=False,
        all_texts=False,
        )
       
    loader = dataset.get_loader(args.batch_size, False)

    images_emb = None
    texts_emb = None
    captions = []
    images = []

    for batch in tqdm(loader):
        with torch.no_grad():
            txt_embeds = model.model.encode_text(batch['captions'].to(device))
            im_embeds = model.model.encode_image(batch['images'].to(device))

            if images_emb is None:
                images_emb = im_embeds.detach().cpu()
                texts_emb = txt_embeds.detach().cpu()

            else:
                images_emb = torch.concat((images_emb, im_embeds.detach().cpu()), dim=0)
                texts_emb = torch.concat((texts_emb, txt_embeds.detach().cpu()), dim=0)
            
            captions += batch['captions']

    data = {'captions': captions, 'image_embeddings': images_emb, 'text_embeddings': texts_emb}
    pickle.dump(data, open(args.save_path, 'wb'))

