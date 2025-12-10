import os
import sys
import logging
import torch

from utils.device_utils import is_npu_available
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

def patch_features():
    _init_logging()

    from diffsynth_npu.patch.base_patch import (
        patch_xfusers_imports,
        patch_initialize_usp,
        patch_torch_ones,
        patch_torch_float64_to_float32,
        patch_tensor_double_to_float32
    )
    patch_xfusers_imports()
    patch_initialize_usp()
    patch_torch_ones()
    patch_torch_float64_to_float32()
    patch_tensor_double_to_float32()

patch_features()