from omegaconf import OmegaConf
from dataset.datasets import CaptionDataset
import torch
from model.createModel import createModel
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import pickle
import json
import os

if __name__ == '__main__':
    path='/nethome/recpinfo/users/fibz/data/embeddings/nwpu/'
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = json.load(open('/nethome/recpinfo/users/fibz/data/dataset/nwpu/labels.json', 'r'))
    texts = [f'a photo of a {e}' for e in labels.keys()]

    for t in ['t25', 't50', 't75', 't100']:
        if t == 'CLIP':
            model_config = '/nethome/recpinfo/users/fibz/data/checkpoint/vlm-finetuning/CLIP/config.yaml'
        else:
            model_config = f'/nethome/recpinfo/users/fibz/data/checkpoint/vlm-finetuning/NWPU-CLIP-base32-LoRA-batch64-{t}/config.yaml'
        
        conf = OmegaConf.load(model_config)
        model = createModel(conf).to(device)
        model.eval()

        dataset = CaptionDataset(
            '/nethome/recpinfo/users/fibz/data/dataset/nwpu/', 
            'val.json', 
            'nwpu',
            model.prepareImages, 
            model.tokenize, 
            random=False,
            all_texts=False,
        )

        loader = dataset.get_loader(5000, False)
        images = None
        y_test = None

        for batch in loader:
            images = batch['image']
            y_test = batch['labels']
        
        tokens = model.tokenize(texts).squeeze(0).to(device)
        
        with torch.no_grad():
            text_features = model.model.encode_text(tokens)
            image_features = model.model.encode_image(images.to(device))
        
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
            logit_scale = model.model.logit_scale.exp()
            logits_per_image = logit_scale.to(image_features.device) * image_features @ text_features.t()
        
        pred = logits_per_image.argmax(dim=1)
        result = {
            "temperature": t,
            "Accuracy": accuracy_score(y_pred=pred, y_true=y_test),
            "Precision micro": precision_score(y_pred=pred, y_true=y_test, average='micro'),
            "Precision macro": precision_score(y_pred=pred, y_true=y_test, average='macro'),
            "Recall micro": recall_score(y_pred=pred, y_true=y_test, average='micro'),
            "Recall macro": recall_score(y_pred=pred, y_true=y_test, average='macro'),
            "F1 micro": f1_score(y_pred=pred, y_true=y_test, average='micro'),
            "F1 macro": f1_score(y_pred=pred, y_true=y_test, average='macro'),
            }
        results.append(result)

    json.dump(results, open('64_zs_classification_result.json', 'w'), indent=2)