import torch
import torch_npu
import torch.distributed as dist
import os
try:
    from lcalib.functional import lcal_initialize
    enable_LCCL = True
except:
    lcal_initialize = None
    enable_LCCL = False
class COMM_INFO:
    def __init__(self):
        self.group = None
        self.world_size = 0
        self.rank = -1

lccl_info = COMM_INFO()
hccl_info = COMM_INFO()
_SEQUENCE_PARALLEL_STATE = False
_SEQUENCE_PARALLEL_GROUP = None # zh
_SEQUENCE_PARALLEL_SIZE = None # zh
_WORLD_RANK = None # zh
def initialize_sequence_parallel_state(sequence_parallel_size):
    global _SEQUENCE_PARALLEL_STATE
    if sequence_parallel_size > 1:
        _SEQUENCE_PARALLEL_STATE = True
        initialize_sequence_parallel_group(sequence_parallel_size)

def set_sequence_parallel_state(state):
    global _SEQUENCE_PARALLEL_STATE
    _SEQUENCE_PARALLEL_STATE = state

def set_sequence_parallel_size(sequence_parallel_size: int):
    global _SEQUENCE_PARALLEL_SIZE
    _SEQUENCE_PARALLEL_SIZE = sequence_parallel_size

def get_sequence_parallel_state():
    return _SEQUENCE_PARALLEL_STATE

def get_sequence_parallel_size():
    return _SEQUENCE_PARALLEL_SIZE


def initialize_sequence_parallel_group(sequence_parallel_size: int):
    """
    核心函数：初始化序列并行通信组
    将全局所有GPU划分为多个子组，每个子组处理不同的序列片段
    """
    global _SEQUENCE_PARALLEL_GROUP
    global _WORLD_RANK

    world_size = int(os.environ.get('WORLD_SIZE', 1))  # 参与训练的总GPU数
    rank = int(os.environ.get('LOCAL_RANK', 0))              # 当前GPU的全局rank
    _WORLD_RANK  = rank

    assert world_size % sequence_parallel_size == 0, \
        "sequence_parallel_size必须能整除world_size"

    num_groups = world_size // sequence_parallel_size
    print(f"将创建{num_groups}个序列并行组，每组{sequence_parallel_size}个GPU")

    for i in range(num_groups):
        start_rank = i * sequence_parallel_size
        end_rank = (i + 1) * sequence_parallel_size
        ranks = list(range(start_rank, end_rank))

        group = dist.new_group(ranks)

        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group
            break  

def get_sequence_parallel_group():
    return _SEQUENCE_PARALLEL_GROUP

def destroy_sequence_parallel_group():
    """Destroy the sequence parallel group."""
    dist.destroy_process_group()