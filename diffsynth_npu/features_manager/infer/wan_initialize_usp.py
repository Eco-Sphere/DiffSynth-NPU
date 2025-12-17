from diffsynth_npu.features_manager.features import DiffSynthFeature
from diffsynth_npu.patch_manager import DiffSynthPatchesManager
from diffsynth_npu.utils.patch_utils import log_replace_info
from diffsynth_npu.patch.base_patch import patch_initialize_usp


class WanInitializeUSPFeature(DiffSynthFeature):
    """Patch WanVideoPipeline.initialize_usp to NPU-only implementation."""

    def __init__(self) -> None:
        super().__init__("wan_initialize_usp")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "infer")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_infer_modules(["patch_initialize_usp"])


def replace_patch_initialize_usp():
    patch_initialize_usp()
    log_replace_info("WanVideoPipeline.initialize_usp", "replace_patch_initialize_usp")

