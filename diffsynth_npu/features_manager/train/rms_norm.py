from diffsynth_npu.features_manager.features import DiffSynthFeature
from diffsynth_npu.patch_manager import DiffSynthPatchesManager


class RmsNormFeature(DiffSynthFeature):
    """Training feature for NPU-optimized RMSNorm."""

    def __init__(self) -> None:
        super().__init__("npu_rms_norm")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "train")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_train_modules(["npu_rms_norm"])



