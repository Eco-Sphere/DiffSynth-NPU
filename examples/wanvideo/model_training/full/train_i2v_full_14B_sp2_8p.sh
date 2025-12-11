# Source CANN
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# Set Logs
mkdir -p ./logs
logfile=$(date +%Y%m%d)_$(date +%H%M%S)
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0

# Set Perf
export TASK_QUEUE_ENABLE=1 # 是否开启TASK多线程下发，绝大多数情况下，打开该功能会进一步提升整网训练性能
export COMBINED_ENABLE=1 # 显存上，性能下。非连续转连续二级推导优化，开启设置为1。当模型中有大量AsStrided高耗时算子被调用时，可以尝试开启此优化以获得潜在的device执行效率的提升。但是Host下发性能存在下降风险。
export ACLNN_CACHE_LIMIT=100000

# Set Mem
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1200
export CLOSE_MATMUL_K_SHIFT=1

# Set Training Configs
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CP_SP_SIZE=2          # 序列并行
GPUS_PER_NODE=8
ACC=1                 # 梯度累积batch
MASTER_ADDR=localhost
MASTER_PORT=6006
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS examples/wanvideo/model_training/full/Wan2.1-I2V-14B.py \
  --sequence_context_parallelism_size $CP_SP_SIZE \
  --task train \
  --train_architecture full \
  --dataset_path ./path_to_your_data \
  --metadata_name metadata.json \
  --dit_path "[
        \"/home/local_data/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors\",
        \"/home/local_data/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors\",
        \"/home/local_data/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors\",
        \"/home/local_data/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors\",
        \"/home/local_data/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors\",
        \"/home/local_data/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors\",
        \"/home/local_data/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors\"
    ]"  \
  --steps_per_epoch 1233 \
  --max_epochs 1 \
  --learning_rate 2e-5 \
  --accumulate_grad_batches $ACC \
  --dataloader_num_worker 0 \
  --use_gradient_checkpointing \
  --training_strategy "deepspeed_stage_3"
# chmod -R 777 logs/train_${logfile}.log
set +x