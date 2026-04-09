import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from encoders import CLIP
import torch
import omegaconf



def createModel(conf,):
    torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
    model = CLIP(conf)

    if conf.model.load_weights:
        
        path = os.path.join(conf.output_dir, 'pytorch_model', 'pytorch_model.bin') #'last.ckpt') 
        ckp = torch.load(path, weights_only=False)
        model.load_state_dict(ckp, strict=False)
        print('LOADING MODEL AT', path)

    return model

    # NWPU-CLIP-base32-cooling-batch1024