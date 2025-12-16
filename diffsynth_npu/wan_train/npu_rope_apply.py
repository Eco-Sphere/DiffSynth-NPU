import torch
import torch_npu
from ..utils.patch_utils import log_replace_info
from einops import rearrange

def npu_rope_apply(x, freqs, num_heads):  # fp32
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)  # (1,32760,1536) => (1,32760,12,128)
    B, S, N, D = x.shape
    cos, sin = freqs
    def rotate_half(x):
        x1, x2 = torch.chunk(x.reshape((B, S, N, D // 2, 2)), 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1).reshape((B, S, N, D))
    res = x * cos + rotate_half(x) * sin  # 此处cos显示是fp64，x是bf16，出来的res也显示是fp64
    return res.flatten(2).to(x.dtype)

def replace_func():
    from diffsynth.models import wan_video_dit
    wan_video_dit.rope_apply = npu_rope_apply

def replace_npu_rope_apply():
    replace_func()
    log_replace_info("rope_apply of wan_video_dit", "replace_npu_rope_apply")
