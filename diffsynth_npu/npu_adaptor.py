import os
import sys
import logging
import torch

from utils.device_utils import is_npu_available
from diffsynth_npu.utils.patch_utils import replace_npu_patch

if is_npu_available():
    from torch_npu.contrib import transfer_to_npu
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format=False

def _init_logging():
    rank = int(os.getenv('RANK', 0))
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[DiffSynth NPU] [%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

def patch_infer_features():
    from diffsynth_npu.patch import NPU_PATCH_MAP
    replace_npu_patch(NPU_PATCH_MAP)

def patch_train_features():
    from diffsynth_npu.wan_train import NPU_OPTIM_MAP
    optim_modules = ["npu_rope_apply", "npu_rms_norm", "WanModel", "SelfAttention", "CrossAttention", "flash_attention_sequence_parallelism"]
    replace_npu_patch(NPU_OPTIM_MAP, optim_modules)

_init_logging()
patch_infer_features()
patch_train_features()
