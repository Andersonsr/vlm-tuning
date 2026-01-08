import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from encoders import CLIP
from lora import apply_lora
from lora_utils import mark_only_lora_as_trainable, load_lora, get_list_lora_layers
import torch
from model.adapter import residual_adapter


def createModel(conf,):
    parts = conf.name.split(':')
    if parts[0] == 'CLIP':
        model = CLIP(conf)
    else:
        raise ValueError(f'Invalid model name {conf.name}')

    # print(model)
    if conf.lora.apply:
        print('Applying Lora')
        apply_lora(conf.lora, model.model)
        mark_only_lora_as_trainable(model)
        # print('old logit scales ', model.model.logit_scale.requires_grad)
        model.model.logit_scale.requires_grad = conf.train_temperature

    elif conf.residual_adapter.apply:
        for param in model.model.parameters():
            param.requires_grad = False

        model.model.logit_scale.requires_grad = conf.train_temperature
        residual_adapter(model, conf)

    if conf.calibrate:
        raise NotImplementedError('calibration not implemented')

    if conf.load_weights:
        print('loading weights from {}'.format(conf.load_weights))
        weights = torch.load(conf.load_weights)
        model.load_state_dict(weights, strict=False)

        lora_path = os.path.join(os.path.dirname(conf.load_weights), 'lora.pt')
        if os.path.exists(lora_path):
            load_lora(conf.lora, get_list_lora_layers(conf.model.lora, model.model))

    return model