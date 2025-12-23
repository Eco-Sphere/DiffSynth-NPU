from typing import Any

from diffsynth_npu.patch_manager import DiffSynthPatchesManager


class DiffSynthFeature:
    """Base class for DiffSynth-NPU features.

    The design is inspired by MindSpeed's ``MindSpeedFeature`` but
    simplified to use ``mode`` (\"all\" / \"infer\" / \"train\") instead
    of a full argument namespace.
    """

    def __init__(self, feature_name: str) -> None:
        self.feature_name = feature_name.lower().strip().replace("-", "_")

    # NOTE: ``mode`` follows the same semantic as in ``patch_features``:
    #   - \"all\"   : apply both infer & train patches
    #   - \"infer\" : apply inference-only patches
    #   - \"train\" : apply training-only patches

    def is_need_apply(self, mode: str) -> bool:
        """Return True if this feature should be applied under ``mode``."""
        return True

    def pre_register_patches(
        self, patch_manager: DiffSynthPatchesManager, mode: str
    ) -> None:  # pragma: no cover - hook kept for symmetry
        """Hook for pre-patch registration (currently unused)."""
        return None

    def register_patches(
        self, patch_manager: DiffSynthPatchesManager, mode: str
    ) -> None:
        """Register runtime patches into the given patch manager."""
        raise NotImplementedError


