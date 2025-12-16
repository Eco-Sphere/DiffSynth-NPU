import os
import logging
import torch
from torch import distributed as dist

from diffsynth_npu.utils.patch_utils import log_replace_info


def patch_xfusers_imports():
    """
    Monkey Patch xfuser.core.distributed and xfuser.core.long_ctx_attention
    """
    from diffsynth_npu.utils.distributed.parallel_mgr import (
        get_sequence_parallel_rank,
        get_sequence_parallel_world_size,
        get_sp_group,
    )
    from diffsynth_npu.utils.modules.attn_layer import xFuserLongContextAttention

    import sys
    from types import ModuleType

    # 创建 xfuser.core.distributed 模块 (NPU 无法安装 xfuser)
    if "xfuser" not in sys.modules:
        sys.modules["xfuser"] = ModuleType("xfuser")
    if "xfuser.core" not in sys.modules:
        sys.modules["xfuser.core"] = ModuleType("xfuser.core")

    distributed_module = ModuleType("distributed")
    distributed_module.get_sequence_parallel_rank = get_sequence_parallel_rank
    distributed_module.get_sequence_parallel_world_size = get_sequence_parallel_world_size
    distributed_module.get_sp_group = get_sp_group

    long_ctx_attention_module = ModuleType("long_ctx_attention")
    long_ctx_attention_module.xFuserLongContextAttention = xFuserLongContextAttention
    sys.modules["xfuser.core.long_ctx_attention"] = long_ctx_attention_module
    sys.modules["xfuser.core.distributed"] = distributed_module

    logging.info(
        "[NPU Patch] xfuser.core.distributed has been patched to redirect to NPU parallel functions"
    )
    logging.info(
        "[NPU Patch] xfuser.core.long_ctx_attention has been patched to redirect to NPU parallel functions"
    )


def patch_initialize_usp():
    """
    Monkey Patch WanVideoPipeline.initialize_usp
    """
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline
    from diffsynth_npu.utils.distributed.parallel_mgr import (
        init_parallel_env,
        ParallelConfig,
    )

    def patched_initialize_usp(self):
        dist.init_process_group(backend="hccl", init_method="env://")
        self.device = torch.device(f"npu:{os.getenv('RANK')}")
        torch.cuda.set_device(dist.get_rank())

        parallel_config = ParallelConfig(
            sp_degree=dist.get_world_size(),
            ulysses_degree=dist.get_world_size(),
            ring_degree=1,
            tp_degree=1,
            use_cfg_parallel=False,
            world_size=dist.get_world_size(),
        )
        init_parallel_env(parallel_config)
        torch.npu.set_device(dist.get_rank())

    # 应用 monkey patch
    WanVideoPipeline.initialize_usp = patched_initialize_usp
    logging.info(
        "[NPU Patch] WanVideoPipeline.initialize_usp has been patched to use NPU-only implementation"
    )


def patch_torch_ones():
    """
    Monkey patch torch.ones to use tensors supported by NPU
    """
    original_torch_ones = torch.ones

    def patched_torch_ones(*args, **kwargs):
        if "dtype" not in kwargs:
            return original_torch_ones(*args, **kwargs)
        else:
            cur_dtype = kwargs["dtype"]
            if cur_dtype in [
                torch.float64,
                torch.complex128,
                torch.complex64,
                torch.complex32,
            ]:
                kwargs["dtype"] = torch.float32
            return original_torch_ones(*args, **kwargs)

    torch.ones = patched_torch_ones
    logging.info("[NPU Patch] torch.ones has been patched to use tensors supported on NPU")


def patch_torch_float64_to_float32():
    """
    Monkey patch torch.float64 to float32
    """
    torch.float64 = torch.float32 if torch.npu.is_available() else torch.float64
    logging.info("[NPU Patch] torch.float64 has been patched to use float32 on NPU")


def patch_tensor_double_to_float32():
    """
    Monkey patch torch.Tensor.double() to float()
    """
    original_double = torch.Tensor.double

    def patched_double(self, memory_format=torch.preserve_format):
        # 在 NPU 环境下，使用 float32 替代 float64
        return self.float()

    torch.Tensor.double = patched_double
    logging.info("[NPU Patch] torch.Tensor.double() has been patched to use float32 on NPU")


# 统一对外的 replace_* 风格，便于通过 replace_npu_patch 批量调用
def replace_patch_xfusers_imports():
    patch_xfusers_imports()
    log_replace_info("xfuser imports", "replace_patch_xfusers_imports")


def replace_patch_initialize_usp():
    patch_initialize_usp()
    log_replace_info("WanVideoPipeline.initialize_usp", "replace_patch_initialize_usp")


def replace_patch_torch_ones():
    patch_torch_ones()
    log_replace_info("torch.ones", "replace_patch_torch_ones")


def replace_patch_torch_float64_to_float32():
    patch_torch_float64_to_float32()
    log_replace_info("torch.float64", "replace_patch_torch_float64_to_float32")


def replace_patch_tensor_double_to_float32():
    patch_tensor_double_to_float32()
    log_replace_info("torch.Tensor.double", "replace_patch_tensor_double_to_float32")
