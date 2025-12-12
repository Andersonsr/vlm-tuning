import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from encoders import CLIP
from loratorch import mark_only_lora_as_trainable
from lora import apply_lora_attn_mlp
import torch


def createModel(conf):
    parts = conf.name.split(':')
    if parts[0] == 'CLIP':
        model = CLIP(conf)
    else:
        raise ValueError(f'Invalid model name {conf.name}')

    # print(model)
    if conf.apply_lora:
        print('Applying Lora')
        model.model = apply_lora_attn_mlp(model.model, 'visual', conf.lora.rank, conf.lora.alpha, False, True)
        model.model = apply_lora_attn_mlp(model.model, 'text', conf.lora.rank, conf.lora.alpha, False, True)
        mark_only_lora_as_trainable(model)
        # print('old logit scales ', model.model.logit_scale.requires_grad)
        model.model.logit_scale.requires_grad = conf.train_temperature

    if conf.adapter:
        raise NotImplementedError('Adapter not implemented')

    if conf.calibrate:
        raise NotImplementedError('calibration not implemented')

    if conf.load_weights:
        print('loading weights from {}'.format(conf.load_weights))
        weights = torch.load(conf.load_weights)
        model.load_state_dict(weights, strict=False)

        lora_path = os.path.join(os.path.dirname(conf.load_weights), 'lora.pt')
        if os.path.exists(lora_path):
            weights = torch.load(lora_path)
            model.load_state_dict(weights, strict=False)

    return model