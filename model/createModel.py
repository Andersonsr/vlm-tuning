import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from encoders import CLIP
import torch
import omegaconf


def find_weights(conf):
    # load_weights may be a path to a file or True to use the default locations inside output_dir
    if isinstance(conf.model.load_weights, str):
        return conf.model.load_weights

    candidates = [
        os.path.join(conf.output_dir, 'pytorch_model', 'pytorch_model.bin'),  # consolidated deepspeed checkpoint
        os.path.join(conf.output_dir, 'last.ckpt'),  # lightning checkpoint
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError('no weights found in {}'.format(candidates))


def read_state_dict(path):
    ckp = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = ckp['state_dict'] if 'state_dict' in ckp else ckp
    # remove wrappers added by lightning/deepspeed
    for prefix in ['_forward_module.', 'module.']:
        state_dict = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    return state_dict


def createModel(conf,):
    torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
    # LoRA layers are created here from conf, so the adapters saved in the checkpoint can be loaded on top of the base model
    model = CLIP(conf)
    # print("CONF", conf)

    if conf.model.load_weights:
        path = find_weights(conf)
        print('LOADING MODEL AT', path)
        state_dict = read_state_dict(path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if conf.model.lora.apply:
            # only the adapters are saved, the backbone comes from the pretrained weights
            missing_lora = [k for k in missing if 'lora_' in k]
            if len(missing_lora) > 0:
                raise RuntimeError('{} LoRA weights missing from {}, e.g. {}'.format(len(missing_lora), path, missing_lora[:5]))
            print('loaded {} tensors (LoRA adapters)'.format(len(state_dict) - len(unexpected)))
        elif len(missing) > 0:
            print('WARNING: {} weights missing from checkpoint, e.g. {}'.format(len(missing), missing[:5]))

        if len(unexpected) > 0:
            print('WARNING: {} unexpected weights in checkpoint, e.g. {}'.format(len(unexpected), unexpected[:5]))

    return model

    # NWPU-CLIP-base32-cooling-batch1024GeoRSCLIPpreprocess.py
