# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

BLOCKS = 'blocks'
VACE_BLOCKS = 'vace_blocks'

def shard_model(
    model,
    device_id,
    # param_dtype=torch.float32,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
):
    targets = []
    if hasattr(model, 'blocks') and model.blocks is not None:
        targets.extend(model.blocks)
    if hasattr(model, 'vace_blocks') and model.vace_blocks is not None:
        targets.extend(model.vace_blocks)

    extra_targets = set()
    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, torch.nn.ModuleList):
            extra_targets.update(list(module))
        else:
            extra_targets.add(module)

    def lambda_fn(module: torch.nn.Module):
        if module in targets:
            return True
        if module in extra_targets:
            return True
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            return module.weight.numel() > 4_000_000
        return False

    lambda_policy = partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)

    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=lambda_policy,
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states,
        limit_all_gathers=True,
        use_orig_params=True)
    return model
