from diffsynth_npu.features_manager.features_manager import DiffSynthFeaturesManager

from diffsynth_npu.features_manager.train.rope_apply import replace_npu_rope_apply
from diffsynth_npu.features_manager.train.rms_norm import replace_npu_rms_norm
from diffsynth_npu.features_manager.train.cross_attention import replace_npu_CrossAttention
from diffsynth_npu.features_manager.train.self_attention import replace_npu_SelfAttention
from diffsynth_npu.features_manager.train.flash_attention_sequence_parallelism import replace_npu_flash_attention_sequence_parallelism
from diffsynth_npu.features_manager.train.wan_model import replace_npu_wanmodel

from diffsynth_npu.features_manager.infer.xfuser_imports import replace_patch_xfusers_imports
from diffsynth_npu.features_manager.infer.wan_initialize_usp import replace_patch_initialize_usp
from diffsynth_npu.features_manager.infer.torch_ones import replace_patch_torch_ones
from diffsynth_npu.features_manager.infer.torch_float64_to_float32 import replace_patch_torch_float64_to_float32
from diffsynth_npu.features_manager.infer.tensor_double_to_float32 import replace_patch_tensor_double_to_float32


NPU_TRAIN_PATCH_MAP = {
    "npu_rope_apply" : replace_npu_rope_apply,
    "npu_rms_norm" : replace_npu_rms_norm,
    "WanModel": replace_npu_wanmodel,
    "SelfAttention": replace_npu_SelfAttention,
    "CrossAttention": replace_npu_CrossAttention,
    "flash_attention_sequence_parallelism": replace_npu_flash_attention_sequence_parallelism,
}

NPU_INFER_PATCH_MAP = {
    "patch_xfusers_imports": replace_patch_xfusers_imports,
    "patch_initialize_usp": replace_patch_initialize_usp,
    "patch_torch_ones": replace_patch_torch_ones,
    "patch_torch_float64_to_float32": replace_patch_torch_float64_to_float32,
    "patch_tensor_double_to_float32": replace_patch_tensor_double_to_float32,
}
