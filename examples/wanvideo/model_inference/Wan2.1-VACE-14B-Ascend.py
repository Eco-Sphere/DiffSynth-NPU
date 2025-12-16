import os
import time
from datetime import datetime
import torch
import diffsynth_npu.npu_adaptor

from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

import torch.distributed as dist

model_path = "./models/"
wan2p1_vace_14b_path = "./models/Wan-AI/Wan2.1-VACE-14B"
HEIGHT=432
WIDTH=768

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="npu",
    model_configs=[
        ModelConfig(
            model_id="Wan-AI/Wan2.1-VACE-14B",
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
            local_model_path=model_path,
            skip_download=True,
        ),
        ModelConfig(
            path=[os.path.join(wan2p1_vace_14b_path, "models_t5_umt5-xxl-enc-bf16.pth")],
            model_id="Wan-AI/Wan2.1-VACE-14B",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
            skip_download=True
        ),
        ModelConfig(
            path=[os.path.join(wan2p1_vace_14b_path, "Wan2.1_VAE.pth")],
            model_id="Wan-AI/Wan2.1-VACE-14B",
            origin_file_pattern="Wan2.1_VAE.pth",
            skip_download=True
        ),
    ],
    tokenizer_config=ModelConfig(
        model_id="Wan-AI/Wan2.1-VACE-14B",
        origin_file_pattern="google/*",
        local_model_path=model_path,
        skip_download=True
    ),
    redirect_common_files=False,
    use_usp=True,
)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=["data/examples/wan/depth_video.mp4", "data/examples/wan/cat_fightning.jpg"]
)

torch.npu.synchronize()

CONTROL_VIDEO = VideoData("data/examples/wan/depth_video.mp4", height=HEIGHT, width=WIDTH)
VACE_IMAGE = Image.open("data/examples/wan/cat_fightning.jpg").resize((WIDTH, HEIGHT))
PROMPT = "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"
NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

def generate_iv2v(prompt, negative_prompt, vace_video, vace_reference_image, height, width):
    if dist.get_rank() == 0:
        print("========== [WARM UP] REFERENCE IMAGE + REFERENCE VIDEO -> VIDEO TEST ============")

    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        vace_video=vace_video,
        seed=0, tiled=False,
        vace_reference_image=vace_reference_image,
        num_inference_steps=2,
        num_frames=81,
        sigma_shift=16.0,
        cfg_merge=True,
        height=height,
        width=width
    )

    if dist.get_rank() == 0:
        print("========== [GENERATE] REFERENCE IMAGE + REFERENCE VIDEO -> VIDEO TEST ============")

    e2e_time = time.time()
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        vace_video=vace_video,
        seed=0, tiled=False,
        vace_reference_image=vace_reference_image,
        num_inference_steps=20,
        num_frames=81,
        sigma_shift=16.0,
        cfg_merge=True,
        height=HEIGHT,
        width=WIDTH
    )

    if dist.get_rank() == 0:
        save_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_video(video, f"video1_14b_{timestamp}.mp4", fps=15, quality=5)
        save_time = time.time() - save_time
        print(f"MP4 Save time: {save_time}")
        print(f"E2E time: {time.time() - e2e_time}")

def generate_v2v(prompt, negative_prompt, vace_video, height, width):
    if dist.get_rank() == 0:
        print("========== [WARM UP] REFERENCE VIDEO -> VIDEO TEST ============")
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        vace_video=vace_video,
        seed=0, tiled=False,
        num_inference_steps=2,
        num_frames=81,
        sigma_shift=16.0,
        cfg_merge=True,
        height=height,
        width=width
    )

    if dist.get_rank() == 0:
        print("========== [GENERATE] REFERENCE VIDEO -> VIDEO TEST ============")

    e2e_time = time.time()
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        vace_video=vace_video,
        seed=0, tiled=False,
        num_inference_steps=20,
        num_frames=81,
        sigma_shift=16.0,
        cfg_merge=True,
        height=height,
        width=width
    )

    if dist.get_rank() == 0:
        save_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_video(video, f"video1_14b_{timestamp}.mp4", fps=15, quality=5)
        save_time = time.time() - save_time
        print(f"MP4 Save time: {save_time}")
        print(f"E2E time: {time.time() - e2e_time}")


def generate_i2v(prompt, negative_prompt, vace_reference_image, height, width):
    if dist.get_rank() == 0:
        print("========== [WARM UP] REFERENCE IMAGE -> VIDEO TEST ============")

    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=0, tiled=False,
        vace_reference_image=vace_reference_image,
        num_inference_steps=2,
        num_frames=61,
        sigma_shift=16.0,
        cfg_merge=True,
        height=height,
        width=width
    )

    if dist.get_rank() == 0:
        print("========== [GENERATE] REFERENCE IMAGE -> VIDEO TEST ============")

    e2e_time = time.time()
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=0, tiled=False,
        vace_reference_image=vace_reference_image,
        num_inference_steps=20,
        num_frames=61,
        sigma_shift=16.0,
        cfg_merge=True,
        height=height,
        width=width
    )

    if dist.get_rank() == 0:
        save_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_video(video, f"video1_14b_{timestamp}.mp4", fps=15, quality=5)
        save_time = time.time() - save_time
        print(f"MP4 Save time: {save_time}")
        print(f"E2E time: {time.time() - e2e_time}")

if __name__ == "__main__":
    generate_iv2v(PROMPT, NEGATIVE_PROMPT, CONTROL_VIDEO, VACE_IMAGE, HEIGHT, WIDTH)
    generate_v2v(PROMPT, NEGATIVE_PROMPT, CONTROL_VIDEO, HEIGHT, WIDTH)
    generate_i2v(PROMPT, NEGATIVE_PROMPT, VACE_IMAGE, HEIGHT, WIDTH)
