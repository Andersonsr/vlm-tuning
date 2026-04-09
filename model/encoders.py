import torch
import clip
import os
from PIL import Image, ImageFile
import lightning as L
from torchmetrics.retrieval import RetrievalRecall
from torch.optim import AdamW, Adam
import loratorch 
from model.lora_utils import mark_only_lora_as_trainable
from model.adapter import residual_adapter
from LongCLIP.model import longclip
from lora_utils import mark_only_lora_as_trainable, load_lora, get_list_lora_layers, apply_lora
from loratorch_utils import apply_lora_attn_mlp
from dataset.datasets import GEO_INDICES
from model.GeoRSCLIPpreprocess import get_preprocess
import torch, open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_model(conf):
    split = conf.model.name.split(':')
    modelFam = split[0]
    modelName = split[1]

    if modelFam == 'CLIP':
        model, preprocess = clip.load(modelName, device='cpu')
        tokenize = clip.tokenize
    
    elif modelFam == 'LongCLIP':
        model, preprocess = longclip.load(f"/nethome/recpinfo/users/fibz/.cache/long-clip/{modelName}.pt", device='cpu')
        tokenize = longclip.tokenize
    
    elif modelFam == 'RemoteCLIP':      
        model, _, preprocess = open_clip.create_model_and_transforms(modelName)
        tokenize = open_clip.get_tokenizer(modelName)
        ckpt = torch.load(f"/nethome/recpinfo/users/fibz/.cache/remote-clip/RemoteCLIP-{modelName}.pt", map_location="cpu")
        model.load_state_dict(ckpt)
    
    elif modelFam == 'GeoRSCLIP':
        model, _, _ = open_clip.create_model_and_transforms(modelName, pretrained="openai")
        tokenize = open_clip.get_tokenizer(modelName)
        
        checkpoint = torch.load(f"/nethome/recpinfo/users/fibz/.cache/geors-clip/{modelName}.pt", map_location="cpu")
        msg = model.load_state_dict(checkpoint, strict=False)
        model = model.to("cpu")
        preprocess = get_preprocess(
                image_resolution=224,
        )
        
    elif modelFam == 'OpenCLIP':
        raise NotImplementedError()
    
    else:
        raise ValueError('{} not recognized'.format(modelFam))

    return model, preprocess, tokenize



class CLIP(L.LightningModule):
    def __init__(self, conf):
        super(CLIP, self).__init__()
        self.model, self.preprocess, self.tokenize = get_model(conf)

        self.multi_val = False

        if conf.dataset.name == 'geo':
            self.multi_val = True
            self.geo_indices_val = conf.dataset.geo_index_val

        self.model.logit_scale = torch.nn.Parameter(
            torch.log(torch.ones(1) * conf.model.temperature),
            requires_grad=conf.model.train_temperature
        )
        
        self.train_temperature = conf.model.train_temperature
        self.cooling = None
        self.lr = conf.train.learning_rate
        self.lora = conf.model.lora.lib if conf.model.lora.apply else 'none'
        # print(self.model)

        if conf.train.cooling.apply:
            self.cooling = conf.train.cooling.apply
            self.target_temperature = conf.train.cooling.final_temp
            self.cooling_steps = conf.train.cooling.iterations
            self.step = 0

        if conf.model.lora.apply:
            if conf.model.lora.lib == 'cliplora':
                print(conf.model.lora.params)
                apply_lora(conf.model.lora, self.model)
                mark_only_lora_as_trainable(self)

            elif conf.model.lora.lib == 'loratorch':
                self.model = apply_lora_attn_mlp(self.model, conf.model.lora)
                
        elif conf.model.residual_adapter.apply:
            for param in self.model.parameters():
                param.requires_grad = False

            residual_adapter(self.model, conf)

        self.save_hyperparameters(conf) 


    def prepareImages(self, images: list[str],) -> torch.Tensor:
        """
        :param images: list of paths to images
        :return: images embeddings
        """
        inputs = []
        for image in images:
            # print(image)
            input = self.preprocess(Image.open(image))
            inputs.append(input)

        return torch.stack(inputs)

    def encode_image(self, image):
        x = self.model.encode_image(image)
        if hasattr(self, 'vision_adapter'):
            x = self.vision_adapter(x)
        return x

    def encode_text(self, text):
        x = self.model.encode_text(text)
        if hasattr(self, 'text_adapter'):
            x = self.text_adapter(x)
        return x

    def update_temperature(self):
        if self.cooling == 'linear':
            cooling_rate = (100.0 - self.target_temperature) / self.cooling_steps
            temperature = max(100.0 - (self.step * cooling_rate), self.target_temperature)

        elif self.cooling == 'step':
            delta = 100.0 - self.target_temperature
            num_steps = self.cooling_steps // (delta // 5)
            cur_step = self.step // num_steps
            temperature = max(self.initial_temperature - (cur_step * 5), self.final_temperature)

        else:
            raise ValueError(f'Cooling rate {self.cooling} not recognized')

        new_temp = torch.nn.Parameter(torch.log(torch.ones(1) * temperature)) #, requires_grad=self.train_temperature)
        with torch.no_grad():
            self.model.logit_scale.copy_(new_temp)

        self.step += 1

    def forward(self, batch,):
        # isso nao é usado nunca mas ACHO que tem que é obrigado a dar override nessa funcao
        image_features = self.encode_image(batch['image'])
        text_features = self.encode_text(batch['text'])
        return image_features, text_features
    
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.train()
        
        if self.lora == 'cliplora':
            mark_only_lora_as_trainable(self.model)
            self.model.logit_scale.requires_grad = self.train_temperature
        
        elif self.lora == 'loratorch':
            loratorch.mark_only_lora_as_trainable(self.model)
            self.model.logit_scale.requires_grad = self.train_temperature

    def on_after_backward(self):
        if self.lora == 'loratorch':
            loratorch.register_model_param_after_backward(self.model)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return Adam(params, lr=self.lr)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        dataset = ''
        if self.multi_val:
            dataset = '{}_'.format(GEO_INDICES[self.geo_indices_val[dataloader_idx]])
        
        with torch.no_grad():
            image_features = self.model.encode_image(batch['image'])
            text_features = self.model.encode_text(batch['tokens'])
            world_size = self.trainer.world_size

            if world_size > 1:
                dim = image_features.shape[-1]
                gathered_image_features = self.all_gather(image_features) 
                gathered_text_features = self.all_gather(text_features) 
    
                image_features = gathered_image_features.view(-1, dim)
                text_features = gathered_text_features.view(-1, dim)
            
            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image_centroid = image_features.mean(dim=0)
            texts_centroid = text_features.mean(dim=0)

            centroid_distance = torch.linalg.norm(image_centroid - texts_centroid)
            pairwise_distance = torch.diagonal(torch.cdist(image_features, text_features, p=2)).mean()
            self.log(f'{dataset}centroid distance', centroid_distance, sync_dist=True, add_dataloader_idx=False)
            self.log(f'{dataset}pairwise distance', pairwise_distance, sync_dist=True, add_dataloader_idx=False)

            # adapters
            if hasattr(self, 'vision_adapter'):
                image_features = self.vision_adapter(image_features)

            if hasattr(self, 'text_adapter'):
                text_features = self.text_adapter(text_features)

            # cosine similarity as logits
            logit_scale = self.model.logit_scale.exp()
            logits_per_image = logit_scale.to(image_features.device) * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ce = torch.nn.CrossEntropyLoss()
            
            ground_truth = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=logits_per_image.device)
            self.log(f'{dataset}val_loss', (ce(logits_per_image, ground_truth) + ce(logits_per_text, ground_truth)) / 2, add_dataloader_idx=False, sync_dist=True)

            # similarity
            positive_mean = torch.diagonal(logits_per_image).mean()
            off_diagonal = logits_per_image * (1 - torch.eye(logits_per_image.shape[0]).to(logits_per_image.device))
            n = logits_per_image.shape[0]
            negative_mean = off_diagonal.sum() / (n ** 2 - n)
            self.log(f'{dataset}mean_positive_similarity', positive_mean, sync_dist=True, add_dataloader_idx=False)
            self.log(f'{dataset}mean_negative_similarity', negative_mean, sync_dist=True, add_dataloader_idx=False)

            #retrieval
            targets = torch.eye(logits_per_image.shape[0]).to(logits_per_image.device)
            indexes = torch.arange(targets.shape[0])
            indexes = indexes.repeat(targets.shape[0], 1).T

            targets = torch.eye(logits_per_image.shape[0]).to(logits_per_image.device)
            indexes = torch.arange(targets.shape[0])
            indexes = indexes.repeat(targets.shape[0], 1).T

            for k in [1, 5, 10]:
                rk = RetrievalRecall(top_k=k)
                self.log(f'{dataset}i2t r@{k}', rk(logits_per_image, targets, indexes), sync_dist=True, add_dataloader_idx=False)
                self.log(f'{dataset}t2i r@{k}', rk(logits_per_image.T, targets, indexes), sync_dist=True, add_dataloader_idx=False)

    def training_step(self, batch, batch_idx):
        if self.cooling is not None:
            self.update_temperature()

        image_features = self.encode_image(batch['image'])
        text_features = self.encode_text(batch['tokens'])
        world_size = self.trainer.world_size

        if world_size > 1:
            dim = image_features.shape[-1]
            gathered_image_features = self.all_gather(image_features , sync_grads=True)
            gathered_text_features = self.all_gather(text_features , sync_grads=True)
        
            # gathered_image_features[self.global_rank] = image_features
            # gathered_text_features[self.global_rank] = text_features

            image_features = gathered_image_features.view(-1, dim)
            text_features = gathered_text_features.view(-1, dim)
        
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # adapters
        if hasattr(self, 'vision_adapter'):
            image_features = self.vision_adapter(image_features)

        if hasattr(self, 'text_adapter'):
            text_features = self.text_adapter(text_features)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        self.log('temperature', logit_scale)

        logits_per_image = logit_scale.to(image_features.device) * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        ground_truth = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=logits_per_image.device)
        ce = torch.nn.CrossEntropyLoss()
        loss = (ce(logits_per_image, ground_truth) + ce(logits_per_text, ground_truth)) / 2   
        
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def learnable_parameters(self):
        learnable = 0
        total = 0
        for param in self.model.parameters():
            total += param.numel()
            if param.requires_grad:
                learnable += param.numel()

        print(f'total params: {total / 1e6:.2f}M,  learnable params: {learnable / 1e6:.2f}M')
        return total, learnable

