from diffsynth_npu.features_manager.features import DiffSynthFeature
from diffsynth_npu.patch_manager import DiffSynthPatchesManager


class TensorDoubleToFloat32Feature(DiffSynthFeature):
    """Patch torch.Tensor.double() to use float32 on NPU."""

    def __init__(self) -> None:
        super().__init__("tensor_double_to_float32")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "infer")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_infer_modules(["patch_tensor_double_to_float32"])



