# Source CANN
export ASCEND_RT_VISIBLE_DEVICES=0
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set Logs
mkdir -p ./logs
logfile=$(date +%Y%m%d)_$(date +%H%M%S)

# Set Perf
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export ACLNN_CACHE_LIMIT=100000

# Set Mem
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

python examples/wanvideo/model_training/full/Wan2.1-I2V-14B.py \
    --metadata_name metadata.json \
    --task data_process \
    --dataset_path ./path_to_your_data \
    --output_path ./output \
    --text_encoder_path "./Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth" \
    --vae_path "./Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth" \
    --image_encoder_path "./Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --dataloader_num_workers=16 \
    --tiled \
    --num_frames 81 \
    --height 832 \
    --width  480 \
    --tile_size_height 52 \
    --tile_size_width 30 \
    --tile_stride_height 26 \
    --tile_stride_width 15| tee logs/data_process_${logfile}.log 2>&1
chmod 440 logs/data_process_${logfile}.log
set +x