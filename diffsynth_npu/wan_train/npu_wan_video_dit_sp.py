import torch
import torch.nn as nn
import torch.nn.functional as F
# SP 通信适配
from typing import Tuple, Optional, List
from diffsynth.models import wan_video_dit
from einops import rearrange

import is_npu_available
from deepspeed.sequence.layer import DistributedAttention
from .parallel_states import get_sequence_parallel_group, get_sequence_parallel_state, \
    get_sequence_parallel_size
import torch.distributed as dist
from ..patch_utils import log_replace_info
from diffsynth.models.utils import hash_state_dict_keys


def _flash_attention_sequence_parallelism_Npu(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                             compatibility_mode=False):
    """
    # q, k, v 的维度为 (batch_size, num_heads, seq_len, head_dim)

    并行[1, 40, 32760, 128]
    q:[1, 40, 32760, 128] -> [1, 80, 16380, 128] # b n s d
    k:torch.Size([1, 80, 16380, 128])
    v:torch.Size([1, 80, 16380, 128])
    """
    x = F.scaled_dot_product_attention(q, k, v)
    return x


class _SelfAttentionNpu(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = wan_video_dit.RMSNorm(dim, eps=eps)
        self.norm_k = wan_video_dit.RMSNorm(dim, eps=eps)

        # 使用 flash_attention 作为 local_attention
        self.dist_attn = DistributedAttention(
            # 传入 flash_attention
            local_attention=flash_attention_sequence_parallelism_Npu,
            # 获取 DeepSpeed 的序列并行组
            sequence_process_group=get_sequence_parallel_group(),
            gather_idx=2,  # 将输入时切分的序列gather聚合
            scatter_idx=1  # 将num_heads 分开
        )


    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        if get_sequence_parallel_state():
            """
            启用序列并行,使用deepspeed ulysses 分布式attention
            """
            # print(f"self attention: sp 分布式attention")
            # # 使用 DistributedAttention 计算注意力
            q = wan_video_dit.rope_apply(q, freqs, self.num_heads)
            k = wan_video_dit.rope_apply(k, freqs, self.num_heads)
            # 调整 q, k, v 的维度为 (batch_size, seq_len, num_heads, head_dim)
            q = rearrange(q, 'b s (n d) -> b n s d', n=self.num_heads)  # ([1, 32760, 5120]) -> [1, 40, 32760, 128] # [b n s d]
            k = rearrange(k, 'b s (n d) -> b n s d', n=self.num_heads)
            v = rearrange(v, 'b s (n d) -> b n s d', n=self.num_heads)  # [1, 32760, 5120] ->
            # import pdb;pdb.set_trace()
            x = self.dist_attn(q, k, v, batch_dim_idx=0)  # 使用 sequence parallel 计算 # [1, 32760, 5120] -> [1, 40, 32760, 128]
            x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
        else:
            # 原始
            x = wan_video_dit.flash_attention(
                q=wan_video_dit.rope_apply(q, freqs, self.num_heads),
                k=wan_video_dit.rope_apply(k, freqs, self.num_heads),
                v=v,
                num_heads=self.num_heads
            )
        return self.o(x)


class _CrossAttentionNpu(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = wan_video_dit.RMSNorm(dim, eps=eps)
        self.norm_k = wan_video_dit.RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input

        # 使用 flash_attention 作为 local_attention
        self.dist_attn = DistributedAttention(
            local_attention=flash_attention_sequence_parallelism_Npu,  # 传入 flash_attention
            sequence_process_group=get_sequence_parallel_group(),  # 获取 DeepSpeed 的序列并行组
            gather_idx=2,  # 将输入切分的序列gather
            scatter_idx=1  # 将num_heads切分
        )

        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = wan_video_dit.RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        if get_sequence_parallel_state():
            # 使用 DeepSpeed-Ulysses 计算 Cross-Attention
            # 调整维度为 (batch_size, seq_len, num_heads, head_dim)
            ##########################################################
            q = rearrange(q, 'b s (n d) -> b n s d', n=self.num_heads)  # [1, 40, 32760, 128]
            k = rearrange(k, 'b s (n d) -> b n s d', n=self.num_heads)  # [1, 40, 512, 128]
            v = rearrange(v, 'b s (n d) -> b n s d', n=self.num_heads)  # [1, 40, 512, 128]

            x = self.dist_attn(q, k, v, batch_dim_idx=0)
            x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)  # [1, 32760, 5120]
        else:
            x = wan_video_dit.flash_attention(q, k, v, num_heads=self.num_heads)

        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))  # [1, 257, 5120]
            v_img = self.v_img(img)  # [1, 257, 5120]

            if get_sequence_parallel_state():
                q = rearrange(q, "b n s d -> b s (n d)", n=self.num_heads)
            y = wan_video_dit.flash_attention(q, k_img, v_img, num_heads=self.num_heads)

            x = x + y
        return self.o(x)


##############################
# class WanModel_Npu(torch.nn.Module):
def wanmodel__init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
):
    torch.nn.Module.__init__(self)
    self.dim = dim
    self.freq_dim = freq_dim
    self.has_image_input = has_image_input
    self.patch_size = patch_size
    self.num_heads = num_heads

    self.patch_embedding = nn.Conv3d(
        in_dim, dim, kernel_size=patch_size, stride=patch_size)
    self.text_embedding = nn.Sequential(
        nn.Linear(text_dim, dim),
        nn.GELU(approximate='tanh'),
        nn.Linear(dim, dim)
    )
    self.time_embedding = nn.Sequential(
        nn.Linear(freq_dim, dim),
        nn.SiLU(),
        nn.Linear(dim, dim)
    )
    self.time_projection = nn.Sequential(
        nn.SiLU(), nn.Linear(dim, dim * 6))
    self.blocks = nn.ModuleList([
        wan_video_dit.DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
        for _ in range(num_layers)
    ])
    self.head = wan_video_dit.Head(dim, out_dim, patch_size, eps)
    head_dim = dim // num_heads
    self.freqs = wan_video_dit.precompute_freqs_cis_3d(head_dim)

    if has_image_input:
        self.img_emb = wan_video_dit.MLP(1280, dim)  # clip_feature_dim = 1280


def _wanmodelpatchify(self, x: torch.Tensor):
    x = self.patch_embedding(x)
    grid_size = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    return x, grid_size  # x, grid_size: (f, h, w)


def _wanmodelunpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
    return rearrange(
        x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
        f=grid_size[0], h=grid_size[1], w=grid_size[2],
        x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
    )


def _wanmodelforward(self,
                     x: torch.Tensor,
                     timestep: torch.Tensor,
                     context: torch.Tensor,
                     clip_feature: Optional[torch.Tensor] = None,
                     y: Optional[torch.Tensor] = None,
                     use_gradient_checkpointing: bool = False,
                     use_gradient_checkpointing_offload: bool = False,
                     **kwargs,
                     ):
    t = self.time_embedding(
        wan_video_dit.sinusoidal_embedding_1d(self.freq_dim, timestep))
    t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
    context = self.text_embedding(context)

    if self.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = self.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    x, (f, h, w) = self.patchify(x)

    freqs = torch.cat([
        self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    if get_sequence_parallel_state():
        # 序列并行切分 1-1
        sp_size = get_sequence_parallel_size()
        x = prepare_parallel_data_for_sp(x, dim=1, sp_size=sp_size)  # [1, 32760, 5120] -> [1, 16380, 5120]  # dim=1 对应 seq_len
        freqs = prepare_parallel_data_for_sp(freqs, dim=0, sp_size=sp_size)  # (32760,1,64)

    ######################
    # ROPE-新增性能优化方案
    ######################
    if is_npu_available():
        B, S, H = x.shape
        N = self.num_heads
        D = H // N

        cos, sin = torch.chunk(torch.view_as_real(freqs.to(torch.complex64)), 2, dim=-1)  # 通过放到首Block前，提前处理，减少每Block冗余计算, (32760,1,64) => [(32760,1,64,1), (32760,1,64,1)]
        cos = cos.unsqueeze(0).expand(-1, -1, -1, -1, 2).flatten(-2)  # 通过放到首Block前，提前处理，减少每Block冗余计算, (32760,1,64,1) ==> (1, 32760,1,64,1) ==> (1, 32760,12,64,2) ==> (1,32760,12,128)
        sin = sin.unsqueeze(0).expand(-1, -1, -1, -1, 2).flatten(-2)  # 通过放到首Block前，提前处理，减少每Block冗余计算

        freqs = (cos, sin)

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    for block in self.blocks:
        if self.training and use_gradient_checkpointing:
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                # 默认false，如果开启序列并行，SP>=2，打开可重入。
                use_reentrant = False
                if get_sequence_parallel_state():
                    use_reentrant = True

                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=use_reentrant,
                )
        else:
            x = block(x, context, t_mod, freqs)

    if get_sequence_parallel_state():
        x = recover_parallel_data_for_sp(x, dim=1, sp_size=sp_size)

    x = self.head(x, t)
    x = self.unpatchify(x, (f, h, w))
    return x


def _wanmodel_state_dict_converter():
    return WanModelStateDictConverter()


def get_sequence_parallel_num_groups():
    group = get_sequence_parallel_group()
    group_size = dist.get_world_size(group=group)  # 获取当前组的GPU数量
    world_size = dist.get_world_size()  # 全局GPU数量
    # print(f"-group_size: {group_size}, world_size: {world_size}, world_size // group_size{ world_size // group_size}---------")
    return


def prepare_parallel_data_for_sp(tensor, dim, sp_size=1):
    """
    将输入张量在指定维度上切分到多个GPU，支持序列并行。

    参数:
        tensor: 输入张量 [..., seq_len, ...]
        dim: 要切分的维度（如 seq_len 对应的维度索引）
        sp_size: 序列并行组的大小（GPU数量）
    """
    return PrepareParallelDataForSp.apply(tensor, get_sequence_parallel_group(), dim, sp_size)


class PrepareParallelDataForSp(torch.autograd.Function):
    """
    自定义自动微分函数，实现以下行为：
    - 前向传播：将输入张量按指定维度分片，仅保留当前rank对应的分片
    - 反向传播：从所有rank聚合梯度，并根据grad_scale模式进行梯度缩放

    典型应用场景：
    - 序列并行(Sequence Parallelism)中处理张量分片
    - 模型并行(Model Parallelism)中的梯度聚合
    """

    @staticmethod
    def forward(ctx, input_, process_group, dim, sp_size=None, grad_scale="down"):
        """
        前向传播逻辑：
        1. 保存反向传播所需的上下文信息
        2. 执行张量分片操作

        参数说明：
        ctx: 自动微分上下文对象，用于保存反向传播所需信息
        input_: 输入张量（需要分片的原始数据）
        process_group: 通信进程组
        dim: 分片操作的维度索引
        sp_size:
        grad_scale: 梯度缩放模式("up"/"down"/None)
        """

        # 保存反向传播所需的参数
        ctx.mode = process_group  # 通信组，用于反向传播时聚合梯度
        ctx.dim = dim  # 分片维度，反向传播时需要知道在哪个维度聚合
        ctx.grad_scale = grad_scale  # 梯度缩放模式

        # 保存分片大小信息（用于非均匀分片情况）List
        ctx.split_sizes = None

        seq_group = get_sequence_parallel_group()
        # 获取当前进程在序列并行组中的rank sp=2 时，只会返回0或者1
        sp_rank = torch.distributed.get_rank(seq_group)

        # 切分目标维度（seq_len）
        seq_len = input_.shape[dim]

        assert seq_len % sp_size == 0, f"seq_len ({seq_len}) 必须能被 sp_size ({sp_size}) 整除"

        chunk_size = seq_len // sp_size

        local_tensor = torch.split(input_, chunk_size, dim=dim)[sp_rank]

        return local_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播逻辑：
        1. 根据grad_scale模式对本地梯度进行缩放
        2. 从所有rank聚合梯度（执行all_gather操作）

        参数说明：
        grad_output: 前向传播输出的分片对应的梯度

        返回说明：
        返回的元组需要与forward的输入参数一一对应：
        - 第一个返回值是input_的梯度
        - 后续None对应process_group/dim/split_sizes/grad_scale的梯度（这些是非张量参数，不需要梯度）
        """

        # 梯度缩放处理
        if ctx.grad_scale == "up":
            # 梯度放大模式（通常用于数据并行）
            grad_output.mul_(dist.get_world_size(ctx.mode))
        elif ctx.grad_scale == "down":  # 显存inplace优化操作 - zh
            # 梯度缩小模式（防止梯度爆炸）
            grad_output.div_(dist.get_world_size(ctx.mode))  # 显存inplace优化操作 - zh

        # 0423 显存优化
        aggregated_grad = _gather(grad_output, ctx.mode, ctx.dim, ctx.split_sizes)

        # 返回梯度（None对应非张量参数的占位）
        return aggregated_grad, None, None, None, None


def _gather(input_: torch.Tensor,
            pg: dist.ProcessGroup,
            dim: int = -1,
            gather_sizes: Optional[List[int]] = None):
    """
    从进程组中所有进程收集张量，并沿指定维度拼接，支持对齐和非对齐数据。

    Args:
        input_: 要收集的输入张量（当前进程的本地数据）
        pg: 执行集合通信的进程组
        dim: 拼接维度（默认为最后一个维度）
        gather_sizes: 各进程张量在拼接维度上的大小列表（None表示所有张量大小相同）

    Returns:
        拼接后的完整张量（包含所有进程的数据）

    实现说明：
    1. 当gather_sizes=None时，使用常规all_gather（要求所有进程input_的shape相同）
    2. 当指定gather_sizes时，支持非均匀大小的张量收集（需要预先分配好各接收张量的shape）
    3. 最终通过torch.cat沿指定维度拼接结果

    典型应用场景：
    - 分布式训练中聚合各计算节点的部分结果
    - 序列并行中恢复完整的序列维度
    """
    # 参数校验：确保gather_sizes是列表（如果提供）
    assert gather_sizes is None or isinstance(gather_sizes, list)

    # 单进程情况直接返回（无需通信）
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return input_

    # 确保输入张量内存连续（提高通信效率）
    input_ = input_.contiguous()

    # print(f"_gather: {gather_sizes}")
    # 准备接收缓冲区列表
    if gather_sizes:
        # 非均匀大小模式：为每个rank创建指定大小的空张量
        tensor_list = []
        tensor_shape_base = list(input_.size())  # 基础shape模板
        for i in range(world_size):
            tensor_shape = tensor_shape_base.copy()
            tensor_shape[dim] = gather_sizes[i]  # 仅修改目标维度大小
            tensor_list.append(
                torch.empty(tensor_shape,
                            dtype=input_.dtype,
                            device=input_.device))
    else:
        # 均匀大小模式：创建与input_相同shape的空张量列表
        tensor_list = [
            torch.empty_like(input_) for _ in range(world_size)
        ]

    # 设备类型检查（仅支持CUDA/NPU设备）
    assert input_.device.type in ("cuda", "npu"), \
        f"Unsupported device type: {input_.device.type}"

    # 执行All-Gather集体通信
    # 效果：将每个进程的input_分发到所有进程的tensor_list对应位置
    torch.distributed.all_gather(tensor_list, input_, group=pg)

    # 沿目标维度拼接结果
    output = torch.cat(tensor_list, dim=dim)

    # 返回内存连续的结果
    return output.contiguous()


def recover_parallel_data_for_sp(tensor, dim, sp_size, grad_scale="up"):
    """
    在模型并行区域中从所有进程收集输入张量，并沿指定维度拼接它们。
    在反向传播时，分割梯度并根据梯度缩放模式进行缩放。
    该函数在收集和分散操作中处理对齐和未对齐的数据。

    参数:
        input_ (torch.Tensor): 要处理的输入张量
        process_group (dist.ProcessGroup): 执行操作的进程组
        dim (int): 拼接收集到的张量的维度
        gather_sizes (Optional[List[int]], optional): 要收集的每个张量部分的大小列表。
            如果未提供，则假定所有张量与输入张量形状相同。默认为None。
        grad_scale (str, optional): 梯度缩放模式。可选"up"(向上)、"down"(向下)或None。默认为"up"。

    返回:
        torch.Tensor: 收集和拼接后的结果张量
    """
    # 调用自定义自动微分函数的apply方法
    return RecoverParallelDataForSp.apply(tensor, get_sequence_parallel_group(), dim, sp_size, grad_scale)


class RecoverParallelDataForSp(torch.autograd.Function):
    """
    自定义的自动求导函数，用于在模型并行区域中从所有进程收集输入张量并拼接它们。
    在反向传播过程中，它会分割梯度并根据梯度缩放模式进行缩放。
    """

    @staticmethod
    def forward(ctx, tensor, process_group, dim, sp_size, grad_scale="up"):
        """
        前向传播：从指定进程组中的所有进程收集张量，并沿指定维度拼接它们。

        参数:
            ctx: 上下文对象，用于存储反向传播需要的信息
            input_: 要处理的输入张量
            process_group: 执行操作的进程组
            dim: 拼接的维度
            sp_sizes:
            grad_scale: 梯度缩放模式，可以是"up"（放大）、"down"（缩小）或None。默认为"up"

        返回:
            收集和拼接后的结果张量
        """
        # 存储反向传播需要的信息
        ctx.mode = process_group  # 进程组
        ctx.dim = dim  # 操作的维度
        ctx.grad_scale = grad_scale  # 梯度缩放模式
        ctx.gather_sizes = None  # 收集的大小列表
        ctx.sp_size = sp_size
        # print(f"recover grad_scale: {ctx.grad_scale}")

        if sp_size == 1:
            return tensor

        # 使用 all_gather 聚合所有GPU的切片
        tensor_list = [torch.empty_like(tensor) for _ in range(sp_size)]
        dist.all_gather(tensor_list, tensor, group=get_sequence_parallel_group())

        # 在目标维度拼接
        recovered_tensor = torch.cat(tensor_list, dim=dim)
        # print("recovered_tensor.requires_grad:", recovered_tensor.requires_grad)
        return recovered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：将梯度分发到输入张量，并根据梯度缩放模式进行缩放。

        参数:
            ctx: 上下文对象，包含前向传播存储的信息
            grad_output: 输出的梯度

        返回:
            输入相对于损失的梯度
        """
        if ctx.grad_scale == "up":
            grad_output.mul_(dist.get_world_size(ctx.mode))  # 显存inplace优化操作 - zh
        elif ctx.grad_scale == "down":
            grad_output.div_(dist.get_world_size(ctx.mode))  # 显存inplace优化操作 - zh

        # 调用分割操作将梯度分发回各个进程
        # 返回的元组中，None表示对应前向传播输入参数的梯度不需要计算
        return _split(grad_output, ctx.mode, ctx.dim, ctx.gather_sizes), None, None, None, None


def _split(
        input_: torch.Tensor,
        pg: dist.ProcessGroup,
        dim: int = -1,
        split_sizes: Optional[List[int]] = None
) -> torch.Tensor:
    """
    将张量沿指定维度分割，并返回当前进程组中当前rank对应的部分，支持对齐和未对齐的数据。

    参数:
        input_ (torch.Tensor): 要分割的输入张量
        pg (dist.ProcessGroup): 执行操作的进程组
        dim (int, 可选): 沿哪个维度分割张量。默认为-1(最后一个维度)
        split_sizes (Optional[List[int]], 可选): 张量分割后各部分大小的列表。
            如果未提供，张量将在进程间平均分割，余数部分分配给前几个进程。默认为None

    返回:
        torch.Tensor: 当前进程组中当前rank对应的张量部分
    """
    assert split_sizes is None or isinstance(split_sizes, list)

    world_size = dist.get_world_size(pg)

    if world_size == 1:
        return input_

    if split_sizes is None:
        dim_size = input_.size(dim)  # 获取指定维度的长度
        base_size = dim_size // world_size  # 计算每个进程的基础大小
        remainder = dim_size % world_size  # 计算余数

        split_sizes = [base_size + 1 if i < remainder else base_size for i in range(world_size)]

    tensor_list = torch.split(input_, split_sizes, dim=dim)

    rank = dist.get_rank(pg)  # 获取当前进程在进程组中的rank
    output = tensor_list[rank].contiguous()  # 确保返回的张量是连续的

    return output


####################通信并行相关method-end

class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        HASH_VALUES_MAP = {
            "t2v-large" : "cb104773c6c2cb6df4f9529ad5c60d0b",
            "t2v-large1": "9269f8db9040a9d860eaca435be61814",
            "t2v-large2": "aafcfd9672c3a2456dc46e1cb6e52c70",
            "t2v-large3": "6bfcfb3b342cb286ce886889d519a77e",

        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == HASH_VALUES_MAP["t2v-large"]:
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config

    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) ==  HASH_VALUES_MAP["t2v-large1"]:
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == HASH_VALUES_MAP["t2v-large2"]:
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == HASH_VALUES_MAP["t2v-large3"]:
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        else:
            config = {}
        return state_dict, config


def replace_npu_wanmodel():
    from diffsynth.models import wan_video_dit
    wan_video_dit.WanModel.__init__ = wanmodel__init__
    wan_video_dit.WanModel.forward = _wanmodelforward
    wan_video_dit.WanModel.patchify = _wanmodelpatchify
    wan_video_dit.WanModel.unpatchify = _wanmodelunpatchify
    wan_video_dit.WanModel.state_dict_converter = _wanmodel_state_dict_converter
    log_replace_info("WanModel", "WanModel_Npu")


def replace_npu_SelfAttention():
    from diffsynth.models import wan_video_dit
    wan_video_dit.SelfAttention = _SelfAttentionNpu
    log_replace_info("SelfAttention", "SelfAttentionNpu")


def replace_npu_CrossAttention():
    from diffsynth.models import wan_video_dit
    wan_video_dit.CrossAttention = _CrossAttentionNpu
    log_replace_info("CrossAttention", "CrossAttentionNpu")


def replace_npu_flash_attention_sequence_parallelism():
    from diffsynth.models import wan_video_dit
    wan_video_dit.flash_attention_sequence_parallelism = _flash_attention_sequence_parallelism_Npu
    log_replace_info("flash_attention_sequence_parallelism", "flash_attention_sequence_parallelism_Npu")