from diffsynth.models import wan_video_dit
from diffsynth_npu.features_manager.features import DiffSynthFeature
from diffsynth_npu.patch_manager import DiffSynthPatchesManager
from diffsynth_npu.utils.patch_utils import log_replace_info
from diffsynth_npu.wan_train.npu_wan_video_dit_sp import _flash_attention_sequence_parallelism_Npu


class FlashAttnSequenceParallelFeature(DiffSynthFeature):
    """Training feature for flash-attention sequence parallelism on NPU."""

    def __init__(self) -> None:
        super().__init__("flash_attention_sequence_parallelism")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "train")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_train_modules(["flash_attention_sequence_parallelism"])


def replace_npu_flash_attention_sequence_parallelism():
    """Replace ``diffsynth.models.wan_video_dit.npu_wan_video_dit_sp.replace_npu_flash_attention_sequence_parallelism`` for locality."""

    wan_video_dit.flash_attention_sequence_parallelism = _flash_attention_sequence_parallelism_Npu
    log_replace_info("flash_attention_sequence_parallelism", "flash_attention_sequence_parallelism_Npu")



