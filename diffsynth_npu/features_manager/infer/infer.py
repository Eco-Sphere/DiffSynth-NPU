from diffsynth_npu.features_manager.features import DiffSynthFeature
from diffsynth_npu.patch_manager import DiffSynthPatchesManager
from diffsynth_npu.patch import NPU_PATCH_MAP


class XfuserImportFeature(DiffSynthFeature):
    """Patch xfuser imports for NPU (xfuser is unavailable on NPU)."""

    def __init__(self) -> None:
        super().__init__("xfuser_imports")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "infer")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_infer_modules(["patch_xfusers_imports"])


class WanInitializeUSPFeature(DiffSynthFeature):
    """Patch WanVideoPipeline.initialize_usp to NPU-only implementation."""

    def __init__(self) -> None:
        super().__init__("wan_initialize_usp")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "infer")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_infer_modules(["patch_initialize_usp"])


class TorchOnesFeature(DiffSynthFeature):
    """Patch torch.ones to dtypes supported on NPU."""

    def __init__(self) -> None:
        super().__init__("torch_ones")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "infer")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_infer_modules(["patch_torch_ones"])


class TorchFloat64To32Feature(DiffSynthFeature):
    """Patch torch.float64 to float32 on NPU."""

    def __init__(self) -> None:
        super().__init__("torch_float64_to_float32")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "infer")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_infer_modules(["patch_torch_float64_to_float32"])


class TensorDoubleToFloat32Feature(DiffSynthFeature):
    """Patch torch.Tensor.double() to use float32 on NPU."""

    def __init__(self) -> None:
        super().__init__("tensor_double_to_float32")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "infer")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_infer_modules(["patch_tensor_double_to_float32"])


