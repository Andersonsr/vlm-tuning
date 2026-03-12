import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from encoders import CLIP
from lora_utils import mark_only_lora_as_trainable, load_lora, get_list_lora_layers, apply_lora
from loratorch_utils import apply_lora_attn_mlp
import torch
from model.adapter import residual_adapter


def createModel(conf,):
    parts = conf.model.name.split(':')
    if parts[0] == 'CLIP':
        model = CLIP(conf)
    else:
        raise ValueError(f'Invalid model name {conf.name}')

    # print(model)
    if conf.model.lora.apply:
        if conf.model.lora.lib == 'cliplora':
            apply_lora(conf.model.lora, model.model)
            mark_only_lora_as_trainable(model)

        elif conf.model.lora.lib == 'loratorch':
            model = apply_lora_attn_mlp(model, conf.model.lora)
            
    elif conf.model.residual_adapter.apply:
        for param in model.model.parameters():
            param.requires_grad = False

        residual_adapter(model, conf)

    if conf.model.calibrate:
        raise NotImplementedError('calibration not implemented')
    
    model.model.logit_scale.requires_grad = conf.model.train_temperature
    # TODO: load weights from a finetuned checkpoint
    
    return model