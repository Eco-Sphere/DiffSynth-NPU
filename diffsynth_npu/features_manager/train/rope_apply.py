from diffsynth.models import wan_video_dit
from diffsynth_npu.features_manager.features import DiffSynthFeature
from diffsynth_npu.patch_manager import DiffSynthPatchesManager
from diffsynth_npu.utils.patch_utils import log_replace_info
from diffsynth_npu.wan_train.npu_rope_apply import npu_rope_apply


class RopeApplyFeature(DiffSynthFeature):
    """Training feature for NPU-optimized rope apply."""

    def __init__(self) -> None:
        super().__init__("npu_rope_apply")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "train")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_train_modules(["npu_rope_apply"])


def replace_npu_rope_apply():
    """Replace ``diffsynth.models.wan_video_dit.npu_rope_apply.replace_npu_rope_apply`` for locality."""

    wan_video_dit.rope_apply = npu_rope_apply
    log_replace_info("rope_apply of wan_video_dit", "replace_npu_rope_apply")

