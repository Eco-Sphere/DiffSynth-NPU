from diffsynth_npu.features_manager.features import DiffSynthFeature
from diffsynth_npu.patch_manager import DiffSynthPatchesManager
from diffsynth_npu.utils.patch_utils import log_replace_info
from diffsynth_npu.patch.base_patch import patch_torch_float64_to_float32

class TorchFloat64To32Feature(DiffSynthFeature):
    """Patch torch.float64 to float32 on NPU."""

    def __init__(self) -> None:
        super().__init__("torch_float64_to_float32")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "infer")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_infer_modules(["patch_torch_float64_to_float32"])


def replace_patch_torch_float64_to_float32():
    patch_torch_float64_to_float32()
    log_replace_info("torch.float64", "replace_patch_torch_float64_to_float32")

