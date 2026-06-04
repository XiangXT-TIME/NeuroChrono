"""
Monkey-patch DeepSpeed to use torch.optim.AdamW instead of FusedAdam.
Import this BEFORE deepspeed is initialized.

Needed on clusters where glibc < 2.29 or C++17 is unavailable,
preventing DeepSpeed's CUDA extensions from compiling.
"""
import torch.optim


class PatchedAdam(torch.optim.AdamW):
    """AdamW wrapper that silently ignores FusedAdam-specific kwargs."""
    def __init__(self, params, **kwargs):
        # Remove FusedAdam-specific args that torch.optim.AdamW doesn't accept
        for k in ["adam_w_mode", "set_grad_none", "amsgrad_mode"]:
            kwargs.pop(k, None)
        super().__init__(params, **kwargs)


# Patch everywhere DeepSpeed imports FusedAdam
import deepspeed.ops.adam.fused_adam as _ds_fused
_ds_fused.FusedAdam = PatchedAdam

import deepspeed.ops.adam as _ds_adam
_ds_adam.FusedAdam = PatchedAdam

print("[patch] DeepSpeed FusedAdam → PatchedAdam (torch.optim.AdamW)")
