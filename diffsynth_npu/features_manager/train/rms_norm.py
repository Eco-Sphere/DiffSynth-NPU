from diffsynth.models import wan_video_dit
from diffsynth_npu.features_manager.features import DiffSynthFeature
from diffsynth_npu.patch_manager import DiffSynthPatchesManager
from diffsynth_npu.utils.patch_utils import log_replace_info
from diffsynth_npu.wan_train.npu_rsm_norm import NpuRMSNorm


class RmsNormFeature(DiffSynthFeature):
    """Training feature for NPU-optimized RMSNorm."""

    def __init__(self) -> None:
        super().__init__("npu_rms_norm")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "train")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_train_modules(["npu_rms_norm"])


def replace_npu_rms_norm():
    """Replace ``diffsynth.models.wan_video_dit.npu_rsm_norm.replace_npu_rms_norm`` for locality."""

    wan_video_dit.RMSNrom = NpuRMSNorm
    log_replace_info("RMSNorm of wan_video_dit", "replace_npu_rms_norm")

