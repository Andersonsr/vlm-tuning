import torch
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
import os
import glob
import re
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', help='experiment dir', required=True)
    parser.add_argument('--last', help='load last model', action='store_true')
    parser.add_argument('--manual', help='manual save', action='store_true')
    parser.add_argument('--best', help='best epoch', action='store_true')
    args = parser.parse_args()

    # /nethome/recpinfo/users/fibz/data/checkpoint/vlm-finetuning/GEO-random-LongCLIP-large14-LoRA-batch1024-original-composition-r4
    input_dir = args.dir
    output_path = os.path.join(input_dir, 'pytorch_model')

    if args.best:
        checkpoints = glob.glob(os.path.join(input_dir, 'checkpoint-epoch=*.ckpt'))
        best = '0'
        for checkpoint in checkpoints:
            match = re.search(r"checkpoint-epoch=(\d+)", checkpoint)
            if match:
                epoch = match.group(1)
                if int(epoch) > int(best):
                    best = epoch

        input_dir = os.path.join(input_dir, f'checkpoint-epoch={best}.ckpt')
    
    elif args.manual:
        input_dir = os.path.join(input_dir, 'manual_save.ckpt')
        
    else:
        input_dir = os.path.join(input_dir, 'last.ckpt')
    
    convert_zero_checkpoint_to_fp32_state_dict(input_dir, output_path, exclude_frozen_parameters=False,)

    print(f"Converted checkpoint saved to {output_path}")

    # python consolidate.py --dir /nethome/recpinfo/users/fibz/data/checkpoint/vlm-finetuning/GEO-CLIP-large14-LoRA-batch512-original-composition/ --last