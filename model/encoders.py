from typing import Any
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
from model.GeoRSCLIPpreprocess import get_preprocess
import torch, open_clip
from peft import LoraConfig, get_peft_model
from adapter import ResidualProjection
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
import math


GEO_INDICES = {0: 'classification', 1: 'composition', 2: 'texture', 3: 'porosity', 4:'diagenesis'}


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize_transform(image, image_size: int = 224, patch_size: int = 16,) -> torch.Tensor:
    # image may be a PIL image or a [C, H, W] float tensor in [0, 1]
    is_tensor = isinstance(image, torch.Tensor)
    if is_tensor:
        h, w = image.shape[-2:]
    else:
        w, h = image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    image_resized = TF.resize(image, (h_patches * patch_size, w_patches * patch_size))
    if not is_tensor:
        image_resized = TF.to_tensor(image_resized)
    return TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)


# Augmentations adapted from https://github.com/rafaelrubo/lithofaciesclassification
# (Rubo et al., 2022, "Carbonate lithofacies classification in optical microscopy").
# Geometric: random shift of the crop window, horizontal flip and rotations restricted to multiples of 90
#            degrees, so no interpolation is needed (the repo's free rotation, zoom and shear are left out).
# Spectral:  ImageJ macro (NLM denoising, histogram equalization, contrast stretching, sharpen,
#            dichromacy, color casting, vignette) plus Keras brightness_range=[0.6, 1.0].

ROTATIONS = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270]


def geometric_augment(image: Image) -> Image:
    """Random horizontal flip and rotation by 0, 90, 180 or 270 degrees (lossless pixel transposes).

    Together they cover all 8 rotations/reflections of the square crop, vertical flips included.
    """
    # 50% de chance de flip horizontal
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    rotation = random.choice(ROTATIONS)
    return image if rotation is None else image.transpose(rotation)


def _stretch(x: torch.Tensor, saturated: float = 0.3) -> torch.Tensor:
    # ImageJ "Enhance Contrast" (saturated=0.3): saturate saturated% of pixels, split between both tails.
    # thresholds from the 8-bit histogram, like ImageJ (much faster than torch.quantile)
    cdf = torch.bincount((x * 255).round().long().flatten(), minlength=256).cumsum(0)
    q = saturated / 200.0 * cdf[-1]
    lo = (cdf > q).nonzero()[0, 0] / 255
    hi = (cdf >= cdf[-1] - q).nonzero()[0, 0] / 255
    return ((x - lo) / (hi - lo).clamp(min=1e-6)).clamp(0, 1)


def _equalize(x: torch.Tensor) -> torch.Tensor:
    return TF.equalize((x * 255).round().to(torch.uint8)).float() / 255


def _sharpen(x: torch.Tensor) -> torch.Tensor:
    # ImageJ Process > Sharpen kernel
    kernel = torch.tensor([[-1., -1., -1.], [-1., 12., -1.], [-1., -1., -1.]]) / 4
    kernel = kernel.to(x).expand(x.shape[0], 1, 3, 3)
    out = F.conv2d(F.pad(x.unsqueeze(0), (1, 1, 1, 1), mode='replicate'), kernel, groups=x.shape[0])
    return out.squeeze(0).clamp(0, 1)


# Machado et al. (2009) dichromacy simulation matrices (severity 1.0), applied in linear RGB
DEUTERANOPE = torch.tensor([
    [0.367322, 0.860646, -0.227968],
    [0.280085, 0.672501, 0.047413],
    [-0.011820, 0.042940, 0.968881],
])
TRITANOPE = torch.tensor([
    [1.255528, -0.076749, -0.178779],
    [-0.078411, 0.930809, 0.147602],
    [0.004733, 0.691367, 0.303900],
])


def _dichromacy(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    linear = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    linear = torch.einsum('ij,jhw->ihw', matrix.to(x), linear).clamp(0, 1)
    return torch.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


def _color_cast(x: torch.Tensor, keep: int) -> torch.Tensor:
    # stretch every channel except `keep`, so the image is cast towards that channel (0=R, 1=G, 2=B)
    out = x.clone()
    for c in range(3):
        if c != keep:
            out[c] = _stretch(x[c])
    return out


def _vignette(x: torch.Tensor) -> torch.Tensor:
    # radial shading, approximating the BaSiC flat-field profile used in the ImageJ macro
    _, h, w = x.shape
    strength = random.uniform(0.2, 0.5)
    cy = h / 2 + random.uniform(-0.1, 0.1) * h
    cx = w / 2 + random.uniform(-0.1, 0.1) * w
    ys = torch.arange(h, dtype=x.dtype).view(-1, 1)
    xs = torch.arange(w, dtype=x.dtype).view(1, -1)
    r2 = ((ys - cy) / (h / 2)) ** 2 + ((xs - cx) / (w / 2)) ** 2
    return (x * (1 - strength * r2 / r2.max())).clamp(0, 1)


def _brightness(x: torch.Tensor) -> torch.Tensor:
    return (x * random.uniform(0.6, 1.0)).clamp(0, 1)


_NLM_WARNED = False


def _nlm(x: torch.Tensor, sigma: float = 15.0, search_window: int = 11) -> torch.Tensor:
    # ImageJ "Non-local Means Denoising" (sigma=15); needs opencv, skipped if unavailable.
    # search_window=11 instead of opencv's default 21: ~3.5x faster, slightly weaker denoising
    global _NLM_WARNED
    try:
        import cv2
    except ImportError:
        if not _NLM_WARNED:
            print('WARNING: opencv not installed, skipping NLM denoising augmentation')
            _NLM_WARNED = True
        return x
    img = (x.permute(1, 2, 0).numpy() * 255).round().astype('uint8')
    img = cv2.fastNlMeansDenoisingColored(img, None, sigma, sigma, 7, search_window)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255


SPECTRAL_OPS = {
    'equalize': _equalize,
    'stretch': _stretch,
    'sharpen': _sharpen,
    'deuteranope': lambda x: _dichromacy(x, DEUTERANOPE),
    'tritanope': lambda x: _dichromacy(x, TRITANOPE),
    'red_cast': lambda x: _color_cast(x, 0),
    'green_cast': lambda x: _color_cast(x, 1),
    'blue_cast': lambda x: _color_cast(x, 2),
    'vignette': _vignette,
    'brightness': _brightness,
}


def spectral_augment(x: torch.Tensor, p: float = 0.8) -> torch.Tensor:
    """Applies one random spectral op with probability p.

    :param x: [3, H, W] float tensor in [0, 1]
    """
    if random.random() < p:
        x = random.choice(list(SPECTRAL_OPS.values()))(x)
    return x


def crop_transform(
    path: list,
    crop_size: int = 224,
    patch_size: int = 16,
    geometric: bool = False,
    spectral: bool = False,
    nlm: bool = False,
    max_shift: float = 0.2,
    p_nlm: float = 0.1,
    p_spectral: float = 0.8,
) -> torch.Tensor:
    """Center crop of crop_size; augmentations are only applied when requested (training).

    :param geometric: random shift of the crop window (up to max_shift * crop_size), horizontal flip and
        90 degree rotation
    :param spectral: one random spectral op with probability p_spectral, see spectral_augment
    :param nlm: NLM denoising with probability p_nlm, before the spectral op (slow, ~0.15 s per 512px crop)
    """
    image = Image.open(path[0])

    w, h = image.size

    if w < crop_size or h < crop_size:
        raise ValueError(
            f"Image size ({w}, {h}) is smaller than crop size ({crop_size}, {crop_size})"
        )

    center_x = w // 2
    center_y = h // 2

    if geometric:
        max_shift_x = min(
            int(crop_size * max_shift),
            center_x - crop_size // 2,
            w - (center_x + crop_size // 2),
        )

        max_shift_y = min(
            int(crop_size * max_shift),
            center_y - crop_size // 2,
            h - (center_y + crop_size // 2),
        )

        center_x += random.randint(-max_shift_x, max_shift_x)
        center_y += random.randint(-max_shift_y, max_shift_y)

    left = center_x - crop_size // 2
    top = center_y - crop_size // 2
    right = left + crop_size
    bottom = top + crop_size

    cropped_image = image.crop((left, top, right, bottom))

    if geometric:
        cropped_image = geometric_augment(cropped_image)

    if spectral or nlm:
        cropped_image = TF.to_tensor(cropped_image.convert('RGB'))

    if nlm and random.random() < p_nlm:
        cropped_image = _nlm(cropped_image)

    if spectral:
        cropped_image = spectral_augment(cropped_image, p=p_spectral)

    return resize_transform(
        cropped_image,
        image_size=crop_size,
        patch_size=patch_size,
    )



# def crop_transform(path: list, crop_size: int = 224, patch_size: int = 16) -> torch.Tensor:
#     image = Image.open(path[0])
#     center_x = image.width // 2
#     center_y = image.height // 2
#     left = center_x - crop_size // 2
#     top = center_y - crop_size // 2
#     right = center_x + crop_size // 2
#     bottom = center_y + crop_size // 2
#     cropped_image = image.crop((left, top, right, bottom))
#     return resize_transform(cropped_image, image_size=crop_size, patch_size=patch_size)

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
        # print(checkpoint.keys())
        
        msg = model.load_state_dict(checkpoint, strict=False)
        model = model.to("cpu")
        preprocess = get_preprocess(
                image_resolution=224,
        )
        
    elif modelFam == 'OpenCLIP':
        raise NotImplementedError()
    
    elif modelFam == 'DINOtxt':
        weights = '/nethome/recpinfo/users/fibz/cache/dinov3_vitl16_dinotxt.pth'
        repo = 'facebookresearch/dinov3'
        model, tokenizer = torch.hub.load(repo, 'dinov3_vitl16_dinotxt_tet1280d20h24l', source='github', weights=weights)
        model = DINOwrap(model)
        model.dim = 2048 if conf.model.average_local else 1024
        model.averaga_local = conf.model.average_local
        tokenize = tokenizer.tokenize
        preprocess =  lambda x: resize_transform(x, conf.model.image_size, 16)

    else:
        raise ValueError('{} not recognized'.format(modelFam))
    
    return model, preprocess, tokenize

class ExtraWrap(torch.nn.Module):
    # this is used to have all models with the same module names as CLIP 
    def __init__(self, model):
        super(ExtraWrap, self).__init__()
        self.transformer = model

class DINOwrap(torch.nn.Module):
    def __init__(self, model):
        super(DINOwrap, self).__init__()
        self.logit_scale = torch.nn.Parameter(torch.log(torch.ones(1) * 100.))
        self.transformer = model.text_model
        self.visual = ExtraWrap(model.visual_model)
        self.original_encode_text = model.encode_text
        self.original_encode_image = model.encode_image
        self.dim = 0
        self.averaga_local = None
        

    def encode_image(self, image):
        # cls_tokens, _, patch_tokens = self.visual.transformer.get_class_and_patch_tokens(image)
        x = self.original_encode_image(image)
        if not self.averaga_local:
            x = x[:, :1024]

        return x
    
    def encode_text(self, text):
        x = self.original_encode_text(text)
        if not self.averaga_local:
            x = x[:, :1024]
            
        return x
        
class CLIP(L.LightningModule):
    def __init__(self, conf):
        super(CLIP, self).__init__()
        self.model, self.preprocess, self.tokenize = get_model(conf)
        self.local_loss = conf.train.local_loss if hasattr(conf.train, 'local_loss') else True
        self.multi_val = False
        self.multi_positive = conf.train.multi_positive if hasattr(conf.train, 'multi_positive') else False
        self.loss_fn = torch.nn.CrossEntropyLoss()
        if conf.model.name.split(':')[0] != 'DINOtxt':
            self.dim = 512 if conf.model.name.split(':')[1] == 'ViT-B/32' else 768

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
        self.save_lora_only = bool(conf.model.lora.apply)

        if conf.train.cooling.apply:
            self.cooling = conf.train.cooling.apply
            self.initial_temperature = conf.model.temperature
            self.target_temperature = conf.train.cooling.final_temp
            self.cooling_steps = conf.train.cooling.iterations
            self.step = 0

        if hasattr(conf.model, 'vision_head_only') and conf.model.vision_head_only is True :
            # only works with dinotxt 
            for name, param in self.model.visual.named_parameters():
                if 'transformer.head.' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    # print('requires grad', name)
                    
            if conf.model.lora.apply:
                # lora will be applied only to the text tower  
                config = LoraConfig(
                    r=conf.model.lora.r, 
                    lora_alpha=conf.model.lora.alpha, 
                    target_modules=["qkv"], 
                    lora_dropout=conf.model.lora.dropout_rate, 
                    bias="none"
                )

                self.model.transformer = get_peft_model(self.model.transformer, config)                


        elif conf.model.lora.apply:
            if conf.model.lora.lib == 'cliplora':
                # print(conf.model.lora.params)
                apply_lora(conf.model.lora, self.model)
                mark_only_lora_as_trainable(self)
                print('LoRA applied!')

            if conf.model.lora.lib == 'peft':
                config = LoraConfig(
                    r=conf.model.lora.r, 
                    lora_alpha=conf.model.lora.alpha, 
                    target_modules=["qkv"], 
                    lora_dropout=conf.model.lora.dropout_rate, 
                    bias="none"
                )

                self.model = get_peft_model(self.model, config)
                # print(self.model)
            
            elif conf.model.lora.lib == 'loratorch':
                self.model = apply_lora_attn_mlp(self.model, conf.model.lora)

        elif conf.model.residual_adapter.apply:
            for param in self.model.parameters():
                param.requires_grad = False

            if conf.model.residual_adapter.target in ['both', 'vision']:
                self.vision_adapter = ResidualProjection(self.dim, conf.model.residual_adapter.bottleneck_reduction, conf.model.residual_adapter.alpha)

            if conf.model.residual_adapter.target in ['both', 'text']:
                self.text_adapter = ResidualProjection(self.dim, conf.model.residual_adapter.bottleneck_reduction, conf.model.residual_adapter.alpha)

            # print(self.model)

        self.save_hyperparameters(conf) 


    def prepareImages(self, images: list[Any],) -> torch.Tensor:
        """
        :param images: list of paths to images
        :return: images embeddings
        """
        inputs = []
        for image in images:
            if type(image) == str:
                image = Image.open(image)

            input = self.preprocess(image)
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
            cooling_rate = (self.initial_temperature - self.target_temperature) / self.cooling_steps
            temperature = max(self.initial_temperature - (self.step * cooling_rate), self.target_temperature)

        elif self.cooling == 'step':
            delta = self.initial_temperature - self.target_temperature
            num_steps = max(self.cooling_steps // max(delta // 5, 1), 1)
            cur_step = self.step // num_steps
            temperature = max(self.initial_temperature - (cur_step * 5), self.target_temperature)

        elif self.cooling == 'cosine':
            progress = min(self.step / self.cooling_steps, 1.0)
            temperature = self.target_temperature + 0.5 * (self.initial_temperature - self.target_temperature) * (1 + math.cos(math.pi * progress))

        else:
            raise ValueError(f'Cooling rate {self.cooling} not recognized')

        new_temp = torch.nn.Parameter(torch.log(torch.ones(1) * temperature)) #, requires_grad=self.train_temperature)
        with torch.no_grad():
            self.model.logit_scale.copy_(new_temp)

        self.step += 1

    def forward(self, batch,):
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

    def lora_state_dict(self, state_dict=None):
        # keeps only the LoRA weights, any other trainable params (e.g. vision head) and the temperature
        if state_dict is None:
            state_dict = self.state_dict()
        trainable = {name for name, param in self.named_parameters() if param.requires_grad}
        return {k: v for k, v in state_dict.items() if 'lora_' in k or k in trainable or k.endswith('logit_scale')}

    def on_save_checkpoint(self, checkpoint):
        if self.save_lora_only and 'state_dict' in checkpoint:
            checkpoint['state_dict'] = self.lora_state_dict(checkpoint['state_dict'])
            checkpoint['lora_only'] = True

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
            
            
            if hasattr(self, 'vision_adapter'):
                image_features = self.vision_adapter(image_features)

            if hasattr(self, 'text_adapter'):
                text_features = self.text_adapter(text_features)
            
            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            bs = text_features.shape[0]
            image_centroid = image_features.mean(dim=0)
            texts_centroid = text_features.mean(dim=0)

            centroid_distance = torch.linalg.norm(image_centroid - texts_centroid)
            pairwise_distance = torch.diagonal(torch.cdist(image_features, text_features, p=2)).mean()
            self.log(f'{dataset}centroid distance', centroid_distance, sync_dist=True, add_dataloader_idx=False, batch_size=bs)
            self.log(f'{dataset}pairwise distance', pairwise_distance, sync_dist=True, add_dataloader_idx=False, batch_size=bs)

            logit_scale = self.model.logit_scale.exp()
            logits_per_image = logit_scale.to(image_features.device) * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            # cosine similarity as logits
            if not self.multi_positive:
                labels = torch.arange(
                    image_features.shape[0],
                    device=image_features.device,
                    dtype=torch.long,
                )

                loss = (self.loss_fn(logits_per_image, labels) + self.loss_fn(logits_per_text, labels)) / 2
                self.log("val_loss", loss, sync_dist=True, batch_size=bs, )  

            else:
                labels = batch['class']
                loss_i2t = self.multi_positive_loss(
                    logits_per_image,
                    labels,
                    labels
                )

                loss_t2i = self.multi_positive_loss(
                    logits_per_text,
                    labels,
                    labels
                )

                # Symmetric image-text loss
                loss = (loss_i2t + loss_t2i) / 2
                self.log("val_loss", loss, sync_dist=True, batch_size=bs, )  

           
            #retrieval
            if not self.multi_positive:
                targets = torch.eye(logits_per_image.shape[0]).to(logits_per_image.device)
            
            else:
                labels = batch['class']
                targets = (
                    labels[:, None] == labels[None, :]
                ).float().to(logits_per_image.device)
                                

            n, m = logits_per_image.shape
            indexes = torch.repeat_interleave(
                torch.arange(n), 
                repeats=m
            ).to(logits_per_image.device)

            logits_per_image = logits_per_image.flatten()
            logits_per_text = logits_per_text.flatten()
            targets = targets.flatten()

            for k in [1, 5, 10]:
                rk = RetrievalRecall(top_k=k)
                self.log(f'{dataset}i2t r@{k}', rk(logits_per_image, targets, indexes), sync_dist=True, add_dataloader_idx=False, batch_size=bs)
                self.log(f'{dataset}t2i r@{k}', rk(logits_per_image.T, targets, indexes), sync_dist=True, add_dataloader_idx=False, batch_size=bs)

    def multi_positive_loss(self, logits, query_labels, key_labels):
        """
        Multi-positive contrastive loss.

        Every key with the same class as the query is considered positive.

        Args:
            logits:       [N_query, N_key]
            query_labels: [N_query]
            key_labels:   [N_key]
        """

        positive_mask = (
            query_labels[:, None] == key_labels[None, :]
        ).float()
        

        target = positive_mask / positive_mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1.0)

        log_probs = F.log_softmax(logits, dim=-1)

        loss = -(target * log_probs).sum(dim=-1).mean()

        return loss


    def training_step(self, batch, batch_idx):

        if self.cooling is not None:
            self.update_temperature()

        image_features = self.encode_image(batch['image'])
        text_features = self.encode_text(batch['tokens'])

        if hasattr(self, 'vision_adapter'):
            image_features = self.vision_adapter(image_features)

        if hasattr(self, 'text_adapter'):
            text_features = self.text_adapter(text_features)

        # normalize before gathering
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if self.multi_positive:
            class_labels = batch['class'].to(image_features.device)

        world_size = self.trainer.world_size
        local_bs = image_features.shape[0]
        # print('use local loss?', self.local_loss)

        # need to compute loss on all ranks for local loss, or on rank 0 for global loss
        if world_size > 1:
            # gather features from all ranks without syncing gradients
            gathered_image_features = self.all_gather(
                image_features, sync_grads=not self.local_loss
            )
            gathered_text_features = self.all_gather(
                text_features, sync_grads=not self.local_loss
            )

            if self.multi_positive:
                gathered_class_labels = self.all_gather(class_labels)

            # restore gradients for local features
            if self.local_loss:
                gathered_image_features[self.trainer.global_rank] = image_features
                gathered_text_features[self.trainer.global_rank] = text_features
                # print('inserted features requires grad', gathered_image_features[self.trainer.global_rank].requires_grad)
            
            dim = image_features.shape[-1]
            all_image_features = gathered_image_features.reshape(-1, dim)
            all_text_features = gathered_text_features.reshape(-1, dim)

            if self.multi_positive:
                all_class_labels = gathered_class_labels.reshape(-1)

        else:
            # single GPU case, no need to gather features
            all_image_features = image_features
            all_text_features = text_features
            if self.multi_positive:
                all_class_labels = class_labels


        if self.local_loss:
            query_image_features = image_features
            query_text_features = text_features
            if self.multi_positive:
                query_class_labels = class_labels

        else:
            query_image_features = all_image_features
            query_text_features = all_text_features
            if self.multi_positive:
                query_class_labels = all_class_labels

        logit_scale = self.model.logit_scale.exp()
        self.log("temperature", logit_scale, batch_size=local_bs)
        
        logits_per_image = logit_scale * query_image_features @ all_text_features.T
        logits_per_text = logit_scale * query_text_features @ all_image_features.T

        if not self.multi_positive:
            if self.local_loss:
                # in distributed training with local loss, create labels for the current rank's batch
                labels = torch.arange(
                    local_bs,
                    device=image_features.device,
                    dtype=torch.long,
                ) + self.trainer.global_rank * local_bs

            else:
                # logits are a square matrix, so labels are just the indices
                labels = torch.arange(
                    all_image_features.shape[0],
                    device=image_features.device,
                    dtype=torch.long,
                )

            loss = (self.loss_fn(logits_per_image, labels) + self.loss_fn(logits_per_text, labels)) / 2
            self.log("train_loss", loss, sync_dist=True, batch_size=local_bs, )  

            return loss

        else:
            loss_i2t = self.multi_positive_loss(
                logits_per_image,
                query_class_labels,
                all_class_labels
            )

            loss_t2i = self.multi_positive_loss(
                logits_per_text,
                query_class_labels,
                all_class_labels
            )

            # Symmetric image-text loss
            loss = (loss_i2t + loss_t2i) / 2
            self.log("train_loss", loss, sync_dist=True, batch_size=local_bs, )  
            # print(f'Muti positive loss {loss}')
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

