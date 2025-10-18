import logging
import torch
from torch import Tensor
import torch_npu
import torch.distributed as dist
import math
import os
from yunchang import LongContextAttention
try:
    from yunchang.kernels import AttnType
except ImportError:
    raise ImportError("Please install yunchang 0.6.0 or later")
from typing import Any

from mindiesd.layers.flash_attn.attention_forward import attention_forward

from ..distributed.parallel_mgr import get_sp_group
from ..distributed.comm import all_to_all_4D

logger = logging.getLogger(__name__)
MAX_TOKEN = 2147483647

class xFuserLongContextAttention(LongContextAttention):
    ring_impl_type_supported_kv_cache = ["basic"]

    def __init__(
        self,
        args: Any = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
        use_kv_cache: bool = False,
        attn_type: AttnType = AttnType.FA,
    ) -> None:
        """
        Arguments:
            scatter_idx: int = 2, the scatter dimension index for Ulysses All2All
            gather_idx: int = 1, the gather dimension index for Ulysses All2All
            ring_impl_type: str = "basic", the ring implementation type, currently only support "basic"
            use_pack_qkv: bool = False, whether to use pack qkv in the input
            use_kv_cache: bool = False, whether to use kv cache in the attention layer, which is applied in PipeFusion.
        """
        super().__init__(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            ring_impl_type=ring_impl_type,
            use_pack_qkv=use_pack_qkv,
            attn_type = attn_type,
        )
        self.use_kv_cache = use_kv_cache
        if (
            use_kv_cache
            and ring_impl_type not in self.ring_impl_type_supported_kv_cache
        ):
            raise RuntimeError(
                f"ring_impl_type: {ring_impl_type} do not support SP kv cache."
            )
        self.world_size = dist.get_world_size()
        self.args = args
        self.video_size = ['480*832', '832*480', '480*720', '720*480']

        self.algo = int(os.getenv('ALGO', 0))

        """
        if self.args.size in self.video_size:
            self.use_all_head = True
        else:
            self.use_all_head = False
        """
        
        self.ulysses_pg = get_sp_group().ulysses_group
        self.ring_pg = get_sp_group().ring_group

    def forward(
        self,
        attn,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        joint_tensor_query=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="none",
        scale=None
    ) -> Tensor:
        """forward

        Arguments:
            attn (Attention): the attention module
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args,
            joint_tensor_query: Tensor = None, a replicated tensor among processes appended to the front or rear of query, depends the joint_strategy  
            joint_tensor_key: Tensor = None, a replicated tensor among processes appended to the front or rear of key, depends the joint_strategy
            joint_tensor_value: Tensor = None, a replicated tensor among processes appended to the front or rear of value, depends the joint_strategy,
            *args: the args same as flash_attn_interface
            joint_strategy: str = "none", the joint strategy for joint attention, currently only support "front" and "rear"

        Returns:
            * output (Tensor): context output
        """

        query_layer = all_to_all_4D(input_=query, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
        key_layer = all_to_all_4D(input_=key, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
        value_layer = all_to_all_4D(input_=value, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)

        if get_sp_group().ring_world_size > 1:
            ring_size = get_sp_group().ring_world_size
            b, s, n, d = key_layer.shape
            k_full = torch.empty([ring_size, b, s, n, d], dtype=query_layer.dtype, device=query_layer.device)
            dist.all_gather_into_tensor(k_full, key_layer, group=self.ring_pg)
            key_layer = k_full.permute(1, 0, 2, 3, 4).reshape(b, -1, n, d)

            v_full = torch.empty([ring_size, b, s, n, d], dtype=query_layer.dtype, device=query_layer.device)
            dist.all_gather_into_tensor(v_full, value_layer, group=self.ring_pg)
            value_layer = v_full.permute(1, 0, 2, 3, 4).reshape(b, -1, n, d)


        # if self.use_all_head:
        try:
            if self.algo == 0:
                out = attention_forward(query_layer, key_layer, value_layer,
                                        opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
            elif self.algo == 1:
                out = attention_forward(query_layer, key_layer, value_layer,
                                        opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
            else:
                raise ValueError(f"select flash attention algorithm only support 0, 1, but got {self.algo}")
        # else:
        except:
            query_layer_list = query_layer.split(1, dim=2)
            key_layer_list = key_layer.split(1, dim=2)
            value_layer_list = value_layer.split(1, dim=2)
            output = []
            for_loop = query_layer.shape[2]
            for i in range(for_loop):
                if self.algo == 0:
                    out = attention_forward(query_layer_list[i], key_layer_list[i], value_layer_list[i],
                                        opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
                elif self.algo == 1:
                    out = attention_forward(query_layer_list[i], key_layer_list[i], value_layer_list[i],
                                        opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
                else:
                    raise ValueError(f"select flash attention algorithm only support 0, 1, but got f{self.algo}")

                output.append(out)
            out = torch.cat(output, dim=2)

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = all_to_all_4D(input_=context_layer, scatter_idx=1, gather_idx=2, group=self.ulysses_pg)

        return output

