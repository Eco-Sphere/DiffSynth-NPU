import os
import sys
import logging
import torch

from diffsynth_npu.utils.device_utils import is_npu_available
from diffsynth_npu.features_manager import DiffSynthFeaturesManager

if is_npu_available():
    from torch_npu.contrib import transfer_to_npu
    torch.npu.set_compile_mode(jit_compile=False)
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


def patch_features(mode: str = "infer"):
    """Unified entry for applying DiffSynth-NPU patches.

    Args:
        mode: ``\"all\"`` / ``\"infer\"`` / ``\"train\"``.
    """
    DiffSynthFeaturesManager.apply_features_patches(mode)

_init_logging()
patch_features()
