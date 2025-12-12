# code from LoRA-Torch repository
# https://github.com/Baijiong-Lin/LoRA-Torch/blob/main/examples/Finetune_open_clip_with_LoRA_Torch_on_CIFAR10.ipynb
import loratorch as lora


def apply_lora_attn_mlp(model, encoder_type='visual', rank=16, lora_alpha=32, mlp=True, attn=True):
    if encoder_type == 'visual':
        encoder = model.visual.transformer
    elif encoder_type == 'text':
        encoder = model.transformer
    else:
        raise ValueError("Invalid encoder_type. Choose 'visual' or 'text'.")

    enable_lora=['q', 'k', 'v', 'o']
    for i, resblock in enumerate(encoder.resblocks):
        if hasattr(resblock, 'attn') and attn:
            multihead = resblock.attn
            lora_multihead = lora.MultiheadAttention(r=rank,
                                    lora_alpha=lora_alpha,
                                    enable_lora=enable_lora,
                                    embed_dim=multihead.embed_dim,
                                    num_heads=multihead.num_heads,
                                    dropout=multihead.dropout,
                                    bias=True if hasattr(multihead, "in_proj_bias") else False,
                                    add_bias_kv=False if multihead.bias_k==None else True,
                                    add_zero_attn=multihead.add_zero_attn,
                                    kdim=multihead.kdim,
                                    vdim=multihead.vdim,
                                    batch_first=multihead.batch_first)
            lora_multihead.load_state_dict(multihead.state_dict(), strict=False)
            resblock.attn = lora_multihead

        if hasattr(resblock, 'mlp') and mlp:
            old_mlp_fc=resblock.mlp.c_fc
            old_mlp_proj=resblock.mlp.c_proj
            new_mlp_fc = lora.Linear(
                old_mlp_fc.in_features,
                old_mlp_fc.out_features,
                bias=True if hasattr(old_mlp_fc, "bias") else False,
                r=rank,
                lora_alpha=lora_alpha,
            )
            new_mlp_proj = lora.Linear(
                old_mlp_proj.in_features,
                old_mlp_proj.out_features,
                bias=True if hasattr(old_mlp_proj, "bias") else False,
                r=rank,
                lora_alpha=lora_alpha,
            )
            new_mlp_fc.load_state_dict(old_mlp_fc.state_dict(),strict=False)
            new_mlp_proj.load_state_dict(old_mlp_proj.state_dict(),strict=False)
            resblock.mlp.c_fc = new_mlp_fc
            resblock.mlp.c_proj = new_mlp_proj

    lora.mark_only_lora_as_trainable(model)
    return model