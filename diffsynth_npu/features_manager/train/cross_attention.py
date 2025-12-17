from diffsynth.models import wan_video_dit
from diffsynth_npu.features_manager.features import DiffSynthFeature
from diffsynth_npu.patch_manager import DiffSynthPatchesManager
from diffsynth_npu.utils.patch_utils import log_replace_info
from diffsynth_npu.wan_train.npu_wan_video_dit_sp import _CrossAttentionNpu


class CrossAttentionFeature(DiffSynthFeature):
    """Training feature for Wan CrossAttention on NPU."""

    def __init__(self) -> None:
        super().__init__("cross_attention")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "train")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_train_modules(["CrossAttention"])


def replace_npu_CrossAttention():
    """Replace ``diffsynth.models.wan_video_dit.npu_wan_video_dit_sp.replace_npu_CrossAttention`` for locality."""

    wan_video_dit.CrossAttention = _CrossAttentionNpu
    log_replace_info("CrossAttention", "CrossAttentionNpu")



