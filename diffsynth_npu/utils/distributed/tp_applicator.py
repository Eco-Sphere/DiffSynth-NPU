import torch
import torch.nn as nn
import torch_npu

from ..modules.model import WanSelfAttention, WanAttentionBlock, WanRMSNorm
from .parallel_mgr import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from .group_coordinator import GroupCoordinator


class TensorParallelApplicator:
    def __init__(self, tp_size, device_map="cpu", tp_group=None):
        self.tp_size = tp_size
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_group = tp_group or get_tp_group()
        self.device_map = device_map

    def apply_to_model(self, model):
        self._apply_tp_to_attention(model)
        self._apply_tp_to_ffn(model)

    def _apply_tp_to_attention(self, module):
        for name, child in module.named_children():
            if isinstance(child, WanSelfAttention):
                self._replace_self_attention(child)
            else:
                self._apply_tp_to_attention(child)

    def _replace_self_attention(self, child):
        child.dim = child.dim // self.tp_size
        child.num_heads = child.num_heads // self.tp_size
        orig_q = child.q
        orig_k = child.k
        orig_v = child.v
        orig_o = child.o
        orig_dtype = orig_q.weight.dtype

        column_out = orig_q.out_features // self.tp_size
        row_in = orig_o.in_features // self.tp_size

        child.q = ColumnParallelLinear(
            orig_q.in_features,
            column_out,
            bias=orig_q.bias is not None,
            gather_output=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            tp_group=self.tp_group
        ).to(dtype=orig_dtype).to(self.device_map)

        child.k = ColumnParallelLinear(
            orig_k.in_features,
            column_out,
            bias=orig_k.bias is not None,
            gather_output=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            tp_group=self.tp_group
        ).to(dtype=orig_dtype).to(self.device_map)

        child.v = ColumnParallelLinear(
            orig_v.in_features,
            column_out,
            bias=orig_v.bias is not None,
            gather_output=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            tp_group=self.tp_group
        ).to(dtype=orig_dtype).to(self.device_map)

        child.o = RowParallelLinear(
            row_in,
            orig_o.out_features,
            bias=orig_o.bias is not None,
            input_is_parallel=True,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            tp_group=self.tp_group
        ).to(dtype=orig_dtype).to(self.device_map)

        self._split_self_weights(child, orig_q, orig_k, orig_v, orig_o)

        if isinstance(child.norm_q, WanRMSNorm):
            ori_norm_q = child.norm_q
            child.norm_q = TensorParallelRMSNorm(
                dim=child.norm_q.dim,
                tp_size=self.tp_size,
                tp_group=self.tp_group
            )
            self._split_norm_weights(child.norm_q, ori_norm_q)

        if isinstance(child.norm_k, WanRMSNorm):
            ori_norm_k = child.norm_k
            child.norm_k = TensorParallelRMSNorm(
                dim=child.norm_k.dim,
                tp_size=self.tp_size,
                tp_group=self.tp_group
            )
            self._split_norm_weights(child.norm_k, ori_norm_k)
  

    def _split_self_weights(self, new_layer, orig_q, orig_k, orig_v, orig_o):
        q_chunk = torch.chunk(orig_q.weight.data, self.tp_size, dim=0)[self.tp_rank]
        new_layer.q.weight.data = q_chunk.contiguous()

        k_chunk = torch.chunk(orig_k.weight.data, self.tp_size, dim=0)[self.tp_rank]
        new_layer.k.weight.data = k_chunk.contiguous()

        v_chunk = torch.chunk(orig_v.weight.data, self.tp_size, dim=0)[self.tp_rank]
        new_layer.v.weight.data = v_chunk.contiguous()

        o_chunk = torch.chunk(orig_o.weight.data, self.tp_size, dim=1)[self.tp_rank]
        new_layer.o.weight.data = o_chunk.contiguous()

        if orig_q.bias is not None:
            bias_chunk = torch.chunk(orig_q.bias.data, self.tp_size, dim=0)[self.tp_rank]
            new_layer.q.bias.data = bias_chunk.contiguous()
        if orig_k.bias is not None:
            bias_chunk = torch.chunk(orig_k.bias.data, self.tp_size, dim=0)[self.tp_rank]
            new_layer.k.bias.data = bias_chunk.contiguous()
        if orig_v.bias is not None:
            bias_chunk = torch.chunk(orig_v.bias.data, self.tp_size, dim=0)[self.tp_rank]
            new_layer.v.bias.data = bias_chunk.contiguous()
        if orig_o.bias is not None:
            new_layer.o.bias.data = orig_o.bias.data.clone() / self.tp_size
    
    def _split_norm_weights(self, new_layer, norm):
        norm_chunk = torch.chunk(norm.weight.data, self.tp_size, dim=0)[self.tp_rank]
        new_layer.weight.data = norm_chunk.contiguous()

    def _replace_cross_attention(self, child):
        orig_wq = child.wq
        orig_wkv = child.wkv
        orig_wo = child.wo
        orig_dtype = orig_wq.weight.dtype

        column_out_wq = orig_wq.out_features // self.tp_size
        column_out_wkv = orig_wkv.out_features // self.tp_size
        row_in_wo = orig_wo.in_features // self.tp_size

        child.wq = ColumnParallelLinear(
            orig_wq.in_features,
            column_out_wq,
            bias=orig_wq.bias is not None,
            gather_output=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            tp_group=self.tp_group
        ).to(dtype=orig_dtype).to(self.device_map)

        child.wkv = ColumnParallelLinear(
            orig_wkv.in_features,
            column_out_wkv,
            bias=orig_wkv.bias is not None,
            gather_output=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            tp_group=self.tp_group
        ).to(dtype=orig_dtype).to(self.device_map)

        child.wo = RowParallelLinear(
            row_in_wo,
            orig_wo.out_features,
            bias=orig_wo.bias is not None,
            input_is_parallel=True,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            tp_group=self.tp_group
        ).to(dtype=orig_dtype).to(self.device_map)

        self._split_cross_attention_weights(child, orig_wq, orig_wkv, orig_wo)
        child.n_heads_per_tp = child.n_heads // self.tp_size

    def _split_cross_attention_weights(self, new_layer, orig_wq, orig_wkv, orig_wo):
        wq_chunk = torch.chunk(orig_wq.weight.data, self.tp_size, dim=0)[self.tp_rank]
        new_layer.wq.weight.data = wq_chunk.contiguous()
        if orig_wq.bias is not None:
            wq_bias_chunk = torch.chunk(orig_wq.bias.data, self.tp_size, dim=0)[self.tp_rank]
            new_layer.wq.bias.data = wq_bias_chunk.contiguous()

        wkv_chunk = torch.chunk(orig_wkv.weight.data, self.tp_size, dim=0)[self.tp_rank]
        new_layer.wkv.weight.data = wkv_chunk.contiguous()
        if orig_wkv.bias is not None:
            wkv_bias_chunk = torch.chunk(orig_wkv.bias.data, self.tp_size, dim=0)[self.tp_rank]
            new_layer.wkv.bias.data = wkv_bias_chunk.contiguous()

        wo_chunk = torch.chunk(orig_wo.weight.data, self.tp_size, dim=1)[self.tp_rank]
        new_layer.wo.weight.data = wo_chunk.contiguous()
        if orig_wo.bias is not None:
            new_layer.wo.bias.data = orig_wo.bias.data.clone() / self.tp_size

    def _apply_tp_to_ffn(self, module):
        for name, child in module.named_children():
            if isinstance(child, WanAttentionBlock):
                self._replace_ffn_layers(child)
            else:
                self._apply_tp_to_ffn(child)

    def _replace_ffn_layers(self, block):
        ff_layer = block.ffn
        orig_gelu_linear = ff_layer[0]
        inner_dim_per_tp = orig_gelu_linear.out_features // self.tp_size
        orig_dtype = orig_gelu_linear.weight.dtype

        ff_layer[0] = ColumnParallelLinear(
            in_features=orig_gelu_linear.in_features,
            out_features=inner_dim_per_tp,
            bias=orig_gelu_linear.bias is not None,
            gather_output=False,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            tp_group=self.tp_group
        ).to(dtype=orig_dtype).to(self.device_map)

        orig_output_linear = ff_layer[2]
        ff_layer[2] = RowParallelLinear(
            in_features=inner_dim_per_tp,
            out_features=orig_output_linear.out_features,
            bias=orig_output_linear.bias is not None,
            input_is_parallel=True,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            tp_group=self.tp_group
        ).to(dtype=orig_dtype).to(self.device_map)

        self._split_ffn_weights(ff_layer, orig_gelu_linear, orig_output_linear)

    def _split_ffn_weights(self, new_ffn, orig_first_linear, orig_second_linear):
        with torch.no_grad():
            first_weight_chunk = torch.chunk(orig_first_linear.weight.data, self.tp_size, dim=0)[self.tp_rank]
            new_ffn[0].weight.data.copy_(first_weight_chunk.contiguous())

            if orig_first_linear.bias is not None:
                first_bias_chunk = torch.chunk(orig_first_linear.bias.data, self.tp_size, dim=0)[self.tp_rank]
                new_ffn[0].bias.data.copy_(first_bias_chunk.contiguous())

            second_weight_chunk = torch.chunk(orig_second_linear.weight.data, self.tp_size, dim=1)[self.tp_rank]
            new_ffn[2].weight.data.copy_(second_weight_chunk.contiguous())

            if orig_second_linear.bias is not None:
                new_ffn[2].bias.data.copy_(orig_second_linear.bias.data.clone() / self.tp_size)


class ColumnParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, gather_output=True, tp_size=None, tp_rank=None, tp_group=None):
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.tp_rank = tp_rank or get_tensor_model_parallel_rank()
        self.tp_group = tp_group or get_tp_group()

        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        x = super().forward(x)
        return x


class RowParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, input_is_parallel=True, 
                 tp_size=None, tp_rank=None, tp_group=None, matmul_allreduce_type="torch"):
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.tp_rank = tp_rank or get_tensor_model_parallel_rank()
        self.tp_group = tp_group or get_tp_group()
        self.input_is_parallel = input_is_parallel

        if matmul_allreduce_type == "atb":
            try:
                from atb_ops.ops.matmul_allreduce import matmul_allreduce
                self.matmul_allreduce = matmul_allreduce
                self.matmul_allreduce_type = "atb"
            except Exception:
                self.matmul_allreduce = None
                self.matmul_allreduce_type = "torch"
        else:
            self.matmul_allreduce_type = matmul_allreduce_type

        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        if not self.input_is_parallel:
            x = torch.chunk(x, self.tp_size, dim=-1)[self.tp_rank]
        
        if self.matmul_allreduce_type == "atb":
            if x.dim() == 2:
                output = torch.empty((x.shape[0], self.weight.shape[0]), dtype=x.dtype, device=x.device)
            elif x.dim() == 3:
                b, s, hx = x.size()
                output = torch.empty((b, s, self.weight.shape[0]), dtype=x.dtype, device=x.device)
            self.matmul_allreduce(output, x, self.weight)
        elif self.matmul_allreduce_type == "torch_npu":
            if isinstance(self.tp_group, GroupCoordinator):
                tp_pg = self.tp_group.device_group
            else:
                tp_pg = self.tp_group
            hcom = tp_pg._get_backend(torch.device('npu')).get_hccl_comm_name
            output = torch_npu.npu_mm_all_reduce_base(x, self.weight, hcom)
        else:
            x = super().forward(x)
            # 执行All-Reduce聚合结果
            if isinstance(self.tp_group, GroupCoordinator):
                output = self.tp_group.all_reduce(x)
            else:
                torch.distributed.all_reduce(x, group=self.tp_group)
                output = x
        return output
    

class TensorParallelRMSNorm(nn.Module):
    def __init__(self, dim, tp_size, tp_group, eps=1e-6):
        super().__init__()
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(dim // self.tp_size))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        if isinstance(self.tp_group, GroupCoordinator):
            variance = self.tp_group.all_reduce(variance)
        else:
            torch.distributed.all_reduce(variance, group=self.tp_group)
        variance /= self.tp_size
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)