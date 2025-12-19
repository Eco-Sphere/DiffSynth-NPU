from diffsynth.models import wan_video_dit
from diffsynth_npu.features_manager.features import DiffSynthFeature
from diffsynth_npu.patch_manager import DiffSynthPatchesManager
from diffsynth_npu.utils.patch_utils import log_replace_info
from diffsynth_npu.wan_train.npu_wan_video_dit_sp import (
    wanmodel__init__,
    _wanmodelforward,
    _wanmodelpatchify,
    _wanmodelunpatchify,
    _wanmodel_state_dict_converter,
)


class WanModelFeature(DiffSynthFeature):
    """Training feature for WanModel core module on NPU."""

    def __init__(self) -> None:
        super().__init__("wan_model")

    def is_need_apply(self, mode: str) -> bool:
        return mode in ("all", "train")

    def register_patches(self, patch_manager: DiffSynthPatchesManager, mode: str) -> None:  # type: ignore[name-defined]
        patch_manager.register_train_modules(["WanModel"])


def replace_npu_wanmodel():
    """Replace ``diffsynth.models.wan_video_dit.npu_wan_video_dit_sp.replace_npu_wanmodel`` for locality."""

    wan_video_dit.WanModel.__init__ = wanmodel__init__
    wan_video_dit.WanModel.forward = _wanmodelforward
    wan_video_dit.WanModel.patchify = _wanmodelpatchify
    wan_video_dit.WanModel.unpatchify = _wanmodelunpatchify
    wan_video_dit.WanModel.state_dict_converter = _wanmodel_state_dict_converter
    log_replace_info("WanModel", "WanModel_Npu")

