from diffsynth_npu.features_manager.features import DiffSynthFeature
from diffsynth_npu.patch_manager import DiffSynthPatchesManager


class TorchOnesFeature(DiffSynthFeature):
    """Patch torch.ones to dtypes supported on NPU."""

    def __init__(self) -> None:
        super().__init__("torch_ones")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "infer")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_infer_modules(["patch_torch_ones"])



