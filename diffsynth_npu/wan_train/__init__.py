from .npu_rope_apply import replace_npu_rope_apply
from .NpuRMSNorm import replace_npu_rms_norm
from .npu_wan_video_sp import replace_npu_wanmodel, replace_npu_SelfAttention, replace_npu_CrossAttention, replace_npu_flash_attention_sequence_parallelism

NPU_OPTIM_MAP = {
    "npu_repo_apply" = replace_npu_rope_apply,
    "npu_rms_norm" = replace_npu_rms_norm,
    "WanModel": replace_npu_wanmodel,
    "SelfAttention": replace_npu_SelfAttention,
    "CrossAttention": replace_npu_CrossAttention,
    "flash_attention_sequence_parallelism": replace_npu_flash_attention_sequence_parallelism,
}

