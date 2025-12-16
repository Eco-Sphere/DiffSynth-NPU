import logging
from typing import Dict, List, Optional

from diffsynth_npu.utils.patch_utils import replace_npu_patch

LOG = logging.getLogger(__name__)


class DiffSynthPatchesManager:
    """Lightweight patch manager for DiffSynth-NPU.

    This is inspired by MindSpeed's ``MindSpeedPatchesManager`` but is
    simplified to match DiffSynth-NPU's existing patch style
    (``NPU_PATCH_MAP`` / ``NPU_OPTIM_MAP`` + ``replace_npu_patch``).
    """

    # Names of inference / training modules that should be patched.
    _infer_modules: List[str] = []
    _train_modules: List[str] = []

    @classmethod
    def register_infer_modules(cls, modules: Optional[List[str]] = None) -> None:
        """Register inference-side modules to be patched.

        If ``modules`` is ``None``, all entries in ``NPU_PATCH_MAP`` will be
        used when applying patches.
        """
        if modules:
            cls._infer_modules.extend(modules)

    @classmethod
    def register_train_modules(cls, modules: Optional[List[str]] = None) -> None:
        """Register training-side modules to be patched."""
        if modules:
            cls._train_modules.extend(modules)

    @classmethod
    def clear(cls) -> None:
        """Clear all previously registered modules."""
        cls._infer_modules.clear()
        cls._train_modules.clear()

    @classmethod
    def apply_infer_patches(cls) -> None:
        """Apply inference-related patches via ``replace_npu_patch``."""
        from diffsynth_npu.patch import NPU_PATCH_MAP

        modules = cls._infer_modules or list(NPU_PATCH_MAP.keys())
        LOG.info("[DiffSynth NPU] applying inference patches: %s", modules)
        replace_npu_patch(NPU_PATCH_MAP, modules)

    @classmethod
    def apply_train_patches(cls) -> None:
        """Apply training-related patches via ``replace_npu_patch``."""
        from diffsynth_npu.wan_train import NPU_OPTIM_MAP

        # Default modules list is aligned with the original ``npu_adaptor``.
        default_modules = [
            "npu_rope_apply",
            "npu_rms_norm",
            "WanModel",
            "SelfAttention",
            "CrossAttention",
            "flash_attention_sequence_parallelism",
        ]
        modules = cls._train_modules or default_modules
        LOG.info("[DiffSynth NPU] applying training patches: %s", modules)
        replace_npu_patch(NPU_OPTIM_MAP, modules)

    @classmethod
    def apply_patches(cls, mode: str = "all") -> None:
        """Apply all registered patches according to ``mode``.

        Args:
            mode: ``\"all\"``, ``\"infer\"`` or ``\"train\"``.
        """
        if mode in ("all", "infer"):
            cls.apply_infer_patches()
        if mode in ("all", "train"):
            cls.apply_train_patches()


