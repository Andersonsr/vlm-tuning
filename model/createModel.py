import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from encoders import CLIP, LongCLIP
from lora_utils import mark_only_lora_as_trainable, load_lora, get_list_lora_layers, apply_lora
from loratorch_utils import apply_lora_attn_mlp
import torch
from model.adapter import residual_adapter
import omegaconf



def createModel(conf,):
    torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
    parts = conf.model.name.split(':')
    if parts[0] == 'CLIP':
         model = CLIP(conf)

    elif parts[0] == 'LongCLIP':
        model = LongCLIP(conf)
        
    else:
        raise ValueError(f'Invalid model name {conf.name}')

    if conf.model.load_weights:
        
        path = os.path.join(conf.output_dir, 'pytorch_model', 'pytorch_model.bin') #'last.ckpt') 
        ckp = torch.load(path, weights_only=False)
        model.load_state_dict(ckp, strict=False)
        print('LOADING MODEL AT', path)

    return model

    #NWPU-CLIP-base32-cooling-batch1024