"""
Off-the-shelf vision encoders for image_retrieval.py, the baselines of the trained runs (encoder configs, examples in
encoders/), built with vlm-tuning's own loaders (DINOv3 code from torch hub, as model/dual_encoder.py):

    DINOv3     HuggingFace DINOv3 (encoder.weight_path: model id or folder)
    Fossil     DINOv3 SSL checkpoint of this project (encoder.weight_path, encoder.branch teacher or student) loaded
               into the hub's backbone (encoder.arch, default dinov3_vitl16), as the 'ssl' vision tower of DualEncoder
    DINOtxt    DINO-TXT's vision tower (encoder.dinotxt_weights), optionally with an SSL backbone (encoder.backbone_weights,
               encoder.branch); encoder.average_patch: CLS + mean patch (DINO-TXT's own 2048-d embedding) or CLS only
    CLIP       OpenAI CLIP's vision tower (encoder.weight_path: model name, e.g. ViT-L/14), with its own preprocessing

encoder.hub_repo / encoder.hub_source: torch hub of the DINOv3 code (default facebookresearch/dinov3 from github, or a
local clone with source local).

Every encoder is called as encoder(x, return_grid): the image vector [n, d] (return_grid False) or the grid tokens
[n, tokens, d] (patch tokens; CLIP has none).
"""
import torch
import torch.nn as nn

from model.dual_encoder import load_ssl_backbone

HUB_REPO, HUB_SOURCE = 'facebookresearch/dinov3', 'github'
DINOTXT_ARCH = 'dinov3_vitl16_dinotxt_tet1280d20h24l'


def hub(e):
    return e.get('hub_repo') or HUB_REPO, e.get('hub_source') or HUB_SOURCE


class Dinov3(nn.Module):
    def __init__(self, e):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(e.weight_path)
        self.num_registers = self.model.config.num_register_tokens

    def forward(self, x, return_grid):
        hidden = self.model(pixel_values=x).last_hidden_state
        return hidden[:, 1 + self.num_registers:] if return_grid else hidden[:, 0]


class Fossil(nn.Module):
    def __init__(self, e):
        super().__init__()
        if not e.get('weight_path'):
            raise ValueError('encoder.weight_path: the DINOv3 SSL checkpoint is required for Fossil')
        repo, source = hub(e)
        self.model = torch.hub.load(repo, e.get('arch') or 'dinov3_vitl16', source=source, pretrained=False)
        load_ssl_backbone(self.model, e.weight_path, e.get('branch') or 'teacher')

    def forward(self, x, return_grid):
        features = self.model.forward_features(x)
        return features['x_norm_patchtokens'] if return_grid else features['x_norm_clstoken']


class Dinotxt(nn.Module):
    def __init__(self, e):
        super().__init__()
        if not e.get('dinotxt_weights'):
            raise ValueError('encoder.dinotxt_weights: the DINO-TXT checkpoint is required for DINOtxt')
        repo, source = hub(e)
        # pretrained: DINO-TXT's backbone (LVD-1689M, downloaded by the hub) and its vision head
        model, _ = torch.hub.load(repo, DINOTXT_ARCH, source=source, pretrained=True, weights=e.dinotxt_weights)
        self.model = model.visual_model  # the text tower is not needed
        if e.get('backbone_weights'):
            load_ssl_backbone(self.model.backbone, e.backbone_weights, e.get('branch') or 'teacher')
        self.average_patch = e.get('average_patch', True)

    def forward(self, x, return_grid):
        features, patch_tokens, _ = self.model(x)
        if return_grid:
            return patch_tokens  # after the vision head, the ones averaged into the second half of the embedding
        return features if self.average_patch else features[:, :features.shape[1] // 2]


class CLIP(nn.Module):
    def __init__(self, e):
        super().__init__()
        import clip
        model, self._preprocess = clip.load(e.weight_path, device='cpu')
        self.model = model.visual.float()

    def forward(self, x, return_grid):
        if return_grid:
            raise ValueError('CLIP has no grid output, use --pooling global')
        return self.model(x)


ENCODERS = {'DINOv3': Dinov3, 'Fossil': Fossil, 'DINOtxt': Dinotxt, 'CLIP': CLIP}


def build_baseline(conf):
    name = conf.encoder.name
    if name not in ENCODERS:
        raise ValueError('unknown encoder {} ({}); trained models are evaluated with their run config.yaml'.format(
            name, ', '.join(ENCODERS)))
    return ENCODERS[name](conf.encoder)
