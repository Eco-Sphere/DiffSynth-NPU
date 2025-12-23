from diffsynth.models import wan_video_dit
from diffsynth_npu.features_manager.features import DiffSynthFeature
from diffsynth_npu.patch_manager import DiffSynthPatchesManager
from diffsynth_npu.utils.patch_utils import log_replace_info
from diffsynth_npu.wan_train.npu_wan_video_dit_sp import _SelfAttentionNpu


class SelfAttentionFeature(DiffSynthFeature):
    """Training feature for Wan SelfAttention on NPU."""

    def __init__(self) -> None:
        super().__init__("self_attention")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "train")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_train_modules(["SelfAttention"])


def replace_npu_SelfAttention():
    """Replace ``diffsynth.models.wan_video_dit.npu_wan_video_dit_sp.replace_npu_SelfAttention`` for locality."""

    wan_video_dit.SelfAttention = _SelfAttentionNpu
    log_replace_info("SelfAttention", "SelfAttentionNpu")

