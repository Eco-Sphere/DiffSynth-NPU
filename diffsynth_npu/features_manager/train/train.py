from diffsynth_npu.features_manager.features import DiffSynthFeature
from diffsynth_npu.patch_manager import DiffSynthPatchesManager


class RopeApplyFeature(DiffSynthFeature):
    """Training feature for NPU-optimized rope apply."""

    def __init__(self) -> None:
        super().__init__("npu_rope_apply")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "train")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_train_modules(["npu_rope_apply"])


class RmsNormFeature(DiffSynthFeature):
    """Training feature for NPU-optimized RMSNorm."""

    def __init__(self) -> None:
        super().__init__("npu_rms_norm")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "train")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_train_modules(["npu_rms_norm"])


class WanModelFeature(DiffSynthFeature):
    """Training feature for WanModel core module on NPU."""

    def __init__(self) -> None:
        super().__init__("wan_model")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "train")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_train_modules(["WanModel"])


class SelfAttentionFeature(DiffSynthFeature):
    """Training feature for Wan SelfAttention on NPU."""

    def __init__(self) -> None:
        super().__init__("self_attention")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "train")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_train_modules(["SelfAttention"])


class CrossAttentionFeature(DiffSynthFeature):
    """Training feature for Wan CrossAttention on NPU."""

    def __init__(self) -> None:
        super().__init__("cross_attention")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "train")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_train_modules(["CrossAttention"])


class FlashAttnSequenceParallelFeature(DiffSynthFeature):
    """Training feature for flash-attention sequence parallelism on NPU."""

    def __init__(self) -> None:
        super().__init__("flash_attention_sequence_parallelism")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "train")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_train_modules(["flash_attention_sequence_parallelism"])


