"""DualEncoder: a DINOv3 SSL vision backbone and a HuggingFace text encoder (e.g. multilingual-e5-large), pretrained
separately and aligned by trainable heads on top of them (LiT-like when the towers are frozen).

conf.model (see configs/dual_default.yaml):
    vision.repo/source/arch   torch hub entry of the backbone (dinov3_vitl16), built without pretrained weights
    vision.weights/branch     DINOv3 SSL checkpoint ('<branch>.backbone.' keys) loaded into it
    vision.tokens             qformer input: 'patch' or 'cls+patch'
    vision.pooled             mlp/linear input: 'cls' or 'cls+mean' (CLS concatenated with the mean patch token)
    text.model_id/pooling/prefix/max_length
    aligner.vision            'qformer', 'mlp' or 'linear'
    aligner.text              'linear', 'mlp' or 'none' (text embedding used as is: locked text space, LiT)
    aligner.embed_dim         shared space dimension (= text hidden size when aligner.text is 'none')
    lora.apply/encoder        LoRA on 'vision' (qkv), 'text' (lora.text_targets) or 'both' towers
Towers without LoRA are frozen, run without gradients and stay in eval mode (no dropout) during training.
"""
import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model
from transformers import AutoModel, AutoTokenizer


def load_ssl_backbone(backbone, path, branch='teacher'):
    """Loads the backbone of a DINOv3 SSL checkpoint (dinov3 repo training: state['model'] with '<branch>.backbone.'
    keys, branch 'teacher' or 'student') into backbone, a ViT of the same architecture. rope_embed.periods is computed
    from the RoPE settings, not learned, so the backbone's own copy is kept (the SSL checkpoint stores it in bf16)."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    state = state.get('model', state)
    prefix = f'{branch}.backbone.'
    weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix) and not k.endswith('rope_embed.periods')}
    if not weights:
        prefixes = sorted({'.'.join(k.split('.')[:2]) for k in state})[:10]
        raise RuntimeError(f'no {prefix}* weights in {path}; it has {prefixes}')
    missing, unexpected = backbone.load_state_dict(weights, strict=False)
    missing = [k for k in missing if not k.endswith('rope_embed.periods')]
    if missing or unexpected:
        raise RuntimeError(f'{path} ({branch}) does not match the backbone: missing {missing[:5]}, '
                           f'unexpected {unexpected[:5]}')
    print(f'loaded the {branch} SSL backbone ({len(weights)} tensors) from {path}')


class QFormerBlock(nn.Module):
    """Self-attention among the queries, cross-attention from the queries to the visual tokens and an MLP, each
    pre-norm with a residual connection."""

    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * mlp_ratio, dim))

    def forward(self, queries, visual):
        h = self.norm_self(queries)
        queries = queries + self.self_attention(h, h, h, need_weights=False)[0]
        queries = queries + self.cross_attention(self.norm_cross(queries), visual, visual, need_weights=False)[0]
        return queries + self.mlp(self.norm_mlp(queries))


class QFormer(nn.Module):
    """Q-Former-like pooling (BLIP-2, without the text branch): k learned queries read the visual tokens (b x n x
    input_dim) by cross-attention; the query outputs are pooled ('mean' or 'first') and projected to output_dim."""

    def __init__(self, input_dim, output_dim, k=32, layers=2, dim=768, heads=12, dropout=0.0, pooling='mean'):
        super().__init__()
        if pooling not in ('mean', 'first'):
            raise ValueError(f'unknown qformer pooling {pooling} (mean, first)')
        self.pooling = pooling
        self.visual_norm = nn.LayerNorm(input_dim)
        self.visual_proj = nn.Linear(input_dim, dim) if input_dim != dim else nn.Identity()
        self.queries = nn.Parameter(torch.randn(1, k, dim) * 0.02)
        self.blocks = nn.ModuleList(QFormerBlock(dim, heads, dropout=dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, output_dim)

    def forward(self, x):
        visual = self.visual_proj(self.visual_norm(x))
        queries = self.queries.expand(x.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, visual)
        queries = self.norm(queries)
        return self.proj(queries.mean(1) if self.pooling == 'mean' else queries[:, 0])


def build_head(kind, input_dim, output_dim, hidden=None):
    if kind == 'linear':
        return nn.Linear(input_dim, output_dim)
    if kind == 'mlp':
        hidden = hidden or max(input_dim, output_dim)
        return nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, output_dim))
    if kind == 'none':
        if input_dim != output_dim:
            raise ValueError(f"aligner 'none' needs embed_dim = encoder dim ({input_dim}), got {output_dim}")
        return nn.Identity()
    raise ValueError(f'unknown aligner head {kind} (linear, mlp, none)')


class TextTokenizer:
    """HF tokenizer with the interface of clip.tokenize: a string or list of strings -> a LongTensor [n, 2, max_length]
    with the input ids and the attention mask (padded to max_length, so the datasets can stack them; the encoder trims
    the batch to its longest text)."""

    def __init__(self, model_id, prefix='', max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.prefix, self.max_length = prefix, max_length

    def __call__(self, texts):
        texts = [texts] if isinstance(texts, str) else list(texts)
        enc = self.tokenizer([self.prefix + t for t in texts], padding='max_length', truncation=True,
                             max_length=self.max_length, return_tensors='pt')
        return torch.stack([enc['input_ids'], enc['attention_mask']], dim=1)


class DualEncoder(nn.Module):
    def __init__(self, conf):
        super().__init__()
        v, t, a, lora = conf.model.vision, conf.model.text, conf.model.aligner, conf.model.lora
        self.backbone = torch.hub.load(v.repo, v.arch, source=v.source, pretrained=False)
        if not v.get('weights'):
            raise ValueError('model.vision.weights: path to the DINOv3 SSL checkpoint is required')
        load_ssl_backbone(self.backbone, v.weights, v.get('branch', 'teacher'))
        self.text = AutoModel.from_pretrained(t.model_id)
        if t.get('grad_checkpointing', False):
            self.text.gradient_checkpointing_enable()
        self.text_pooling = t.get('pooling', 'mean')
        if self.text_pooling not in ('mean', 'cls', 'last'):
            raise ValueError(f'unknown text pooling {self.text_pooling} (mean, cls, last)')

        vision_dim, text_dim = self.backbone.embed_dim, self.text.config.hidden_size
        self.dim = a.embed_dim
        self.vision_kind = a.vision
        self.vision_tokens = v.get('tokens', 'cls+patch')
        self.vision_pooled = v.get('pooled', 'cls')
        if a.vision == 'qformer':
            self.vision_head = QFormer(vision_dim, self.dim, a.queries, a.layers, a.width, a.heads, a.dropout,
                                       a.get('pooling', 'mean'))
        else:
            input_dim = 2 * vision_dim if self.vision_pooled == 'cls+mean' else vision_dim
            self.vision_head = build_head(a.vision, input_dim, self.dim)
        self.text_head = build_head(a.text, text_dim, self.dim)

        for p in list(self.backbone.parameters()) + list(self.text.parameters()):
            p.requires_grad = False
        if lora.apply:
            towers = {'both': ('vision', 'text'), 'vision': ('vision',), 'text': ('text',)}[lora.encoder]
            if 'vision' in towers:
                self._add_lora(self.backbone, lora, list(lora.get('vision_targets', ['qkv'])))
            if 'text' in towers:
                self._add_lora(self.text, lora, list(lora.get('text_targets', ['query', 'key', 'value'])))
        self.vision_trainable = any(p.requires_grad for p in self.backbone.parameters())
        self.text_trainable = any(p.requires_grad for p in self.text.parameters())
        self.logit_scale = nn.Parameter(torch.log(torch.ones(1) * 100.))  # replaced by the CLIP module

    @staticmethod
    def _add_lora(module, lora, targets):
        config = LoraConfig(r=lora.r, lora_alpha=lora.alpha, target_modules=targets, lora_dropout=lora.dropout_rate,
                            bias='none')
        inject_adapter_in_model(config, module)
        for name, p in module.named_parameters():
            p.requires_grad = 'lora_' in name

    def train(self, mode=True):
        super().train(mode)
        # frozen towers: no dropout, they are fixed feature extractors
        for tower, trainable in ((self.backbone, self.vision_trainable), (self.text, self.text_trainable)):
            if not trainable:
                tower.eval()
        return self

    def encode_image(self, image):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.vision_trainable):
            out = self.backbone.forward_features(image)
        cls, patches = out['x_norm_clstoken'], out['x_norm_patchtokens']
        if self.vision_kind == 'qformer':
            tokens = torch.cat([cls[:, None], patches], 1) if self.vision_tokens == 'cls+patch' else patches
            return self.vision_head(tokens)
        features = torch.cat([cls, patches.mean(1)], -1) if self.vision_pooled == 'cls+mean' else cls
        return self.vision_head(features)

    def encode_text(self, tokens):
        ids, mask = tokens[:, 0], tokens[:, 1]
        length = int(mask.sum(1).max())
        ids, mask = ids[:, :length], mask[:, :length]
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.text_trainable):
            hidden = self.text(input_ids=ids, attention_mask=mask).last_hidden_state
        if self.text_pooling == 'mean':
            m = mask[..., None].to(hidden.dtype)
            pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
        elif self.text_pooling == 'cls':
            pooled = hidden[:, 0]
        else:  # last non-padding token (right padding)
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), mask.sum(1) - 1]
        return self.text_head(pooled)
