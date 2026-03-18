import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from encoders import CLIP, LongCLIP
from lora_utils import mark_only_lora_as_trainable, load_lora, get_list_lora_layers, apply_lora
from loratorch_utils import apply_lora_attn_mlp
import torch
from model.adapter import residual_adapter


def createModel(conf,):
    parts = conf.model.name.split(':')
    if parts[0] == 'CLIP':
        model = CLIP(conf)
    elif parts[0] == 'LongCLIP':
        model = LongCLIP(conf)
    else:
        raise ValueError(f'Invalid model name {conf.name}')
    
    return model