from typing import List

from diffsynth_npu.patch_manager import DiffSynthPatchesManager


class DiffSynthFeature:
    """Base class for DiffSynth-NPU features.

    Compared with MindSpeed's ``MindSpeedFeature``, this version is
    intentionally simplified: for now we only care about whether a
    feature should be enabled under a given ``mode`` (infer/train/all)
    and how it registers patches into ``DiffSynthPatchesManager``.
    """

    def __init__(self, feature_name: str) -> None:
        self.feature_name = feature_name.lower().strip().replace("-", "_")

    def is_need_apply(self, mode: str) -> bool:
        """Return ``True`` if this feature should be applied under ``mode``."""
        return True

    def pre_register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        """Hook for pre-patch registration (kept for symmetry with MindSpeed)."""
        return None

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        """Register runtime patches into the given patch manager."""
        raise NotImplementedError


class InferPatchFeature(DiffSynthFeature):
    """Feature that manages all inference-time patches."""

    def __init__(self) -> None:
        super().__init__("infer_patches")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "infer")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        # By default we let ``DiffSynthPatchesManager`` pick up all keys in
        # ``NPU_PATCH_MAP`` so there is nothing specific to register here.
        # The call is kept for future extensibility.
        from diffsynth_npu.patch import NPU_PATCH_MAP

        patch_manager.register_infer_modules(list(NPU_PATCH_MAP.keys()))


class TrainPatchFeature(DiffSynthFeature):
    """Feature that manages all training-time patches for WanVideo."""

    def __init__(self) -> None:
        super().__init__("train_patches")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "train")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        default_modules = [
            "npu_rope_apply",
            "npu_rms_norm",
            "WanModel",
            "SelfAttention",
            "CrossAttention",
            "flash_attention_sequence_parallelism",
        ]
        patch_manager.register_train_modules(default_modules)


class DiffSynthFeaturesManager:
    """Feature manager that orchestrates how patches are applied.

    It mirrors the high-level design of ``MindSpeedFeaturesManager``:
    each feature decides whether it should be active, and is responsible
    for registering its own patches into a shared patch manager.
    """

    FEATURES_LIST: List[DiffSynthFeature] = [
        InferPatchFeature(),
        TrainPatchFeature(),
    ]

    @classmethod
    def apply_features_pre_patches(cls, mode: str = "all") -> None:
        """Apply pre-patches of all features (currently a no-op hook)."""
        for feature in cls.FEATURES_LIST:
            if feature.is_need_apply(mode):
                feature.pre_register_patches(DiffSynthPatchesManager, mode)
        # For now there is nothing to apply here; this method is kept for
        # structural compatibility and future extension.

    @classmethod
    def apply_features_patches(cls, mode: str = "all") -> None:
        """Apply runtime patches of all features."""
        # Clear previous registrations before re-applying.
        DiffSynthPatchesManager.clear()
        for feature in cls.FEATURES_LIST:
            if feature.is_need_apply(mode):
                feature.register_patches(DiffSynthPatchesManager, mode)
        DiffSynthPatchesManager.apply_patches(mode)


