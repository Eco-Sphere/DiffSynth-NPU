export ALGO=1
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2

export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export ASCEND_LAUNCH_BLOCKING=0
export PROFILING_ENABLE=0
export PROFILING_DIR=./prof_wan2.1_vace_14b_ascend_$(date +%Y%m%d-%H%M%S)
torchrun --master_port=20313 --nproc_per_node=8 Wan2.1-VACE-14B-Ascend.py

if [ $PROFILING_ENABLE -eq 1 ]; then
    tar -czvf ${PROFILING_DIR}.tar.gz ${PROFILING_DIR}
fi
