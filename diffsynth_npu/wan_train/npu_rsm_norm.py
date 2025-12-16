import torch
import torch.nn as nn
import torch_npu
from ..utils.patch_utils import log_replace_info

class NpuRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]


def replace_func():
    from diffsynth.models import wan_video_dit
    wan_video_dit.RMSNrom  = NpuRMSNorm

def replace_npu_rms_norm():
    replace_func()
    log_replace_info("RMSNorm of wan_video_dit", "replace_npu_rms_norm" )
