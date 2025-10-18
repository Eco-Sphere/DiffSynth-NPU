# 通义万相（Wan）

Wan 是由阿里巴巴通义实验室开源的一系列视频生成模型。

## 安装

在使用本系列模型之前，请确保已安装diffsyth-studio与diffsynth-npu

## 快速开始

通过运行以下代码可以快速加载 [Wan-AI/Wan2.1-VACE-14B](https://www.modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B) 模型并进行推理

```python
import torch
import diffsynth
import diffsynth_npu
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="npu",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="Wan2.1_VAE.pth"),
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
pipe.enable_vram_management()

video = pipe(
    prompt="纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=0, tiled=True,
)
save_video(video, "video1.mp4", fps=15, quality=5)
```

或是运行以下脚本进行推理：
1. 示例代码：
    - [Wan2.1-VACE-14B-Ascend.py](../Wan2.1-VACE-14B-Ascend.py)
    - [start_test.sh](../start_test.sh)
2. 运行方式：
    ```shell
    bash start_test.sh
    ```

## 模型推理

以下部分将会帮助您理解我们的功能并编写推理代码。

<details>

<summary>加载模型</summary>

模型通过 `from_pretrained` 加载：

```python
import torch
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="npu",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
)
```

其中 `torch_dtype` 和 `device` 是计算精度和计算设备。`model_configs` 可通过多种方式配置模型路径：

* 从[魔搭社区](https://modelscope.cn/)下载模型并加载。此时需要填写 `model_id` 和 `origin_file_pattern`，例如

```python
ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
```

* 从本地文件路径加载模型。此时需要填写 `path`，例如

```python
ModelConfig(path="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
```

对于从多个文件加载的单一模型，使用列表即可，例如

```python
ModelConfig(path=[
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors",
])
```

`ModelConfig` 提供了额外的参数用于控制模型加载时的行为：

* `local_model_path`: 用于保存下载模型的路径，默认值为 `"./models"`。
* `skip_download`: 是否跳过下载，默认值为 `False`。当您的网络无法访问[魔搭社区](https://modelscope.cn/)时，请手动下载必要的文件，并将其设置为 `True`。

`from_pretrained` 提供了额外的参数用于控制模型加载时的行为：

* `tokenizer_config`: Wan 模型的 tokenizer 路径，默认值为 `ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*")`。
* `redirect_common_files`: 是否重定向重复模型文件，默认值为 `True`。由于 Wan 系列模型包括多个基础模型，每个基础模型的 text encoder 等模块都是相同的，为避免重复下载，我们会对模型路径进行重定向。
* `use_usp`: 是否启用 Unified Sequence Parallel，默认值为 `False`。用于多 GPU 并行推理。

</details>


<details>

<summary>显存管理</summary>

DiffSynth-Studio 为 Wan 模型提供了细粒度的显存管理，让模型能够在低显存设备上进行推理，可通过以下代码开启 offload 功能，在显存有限的设备上将部分模块 offload 到内存中。

```python
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()
```
</details>

<details>

<summary>输入参数</summary>

Pipeline 在推理阶段能够接收以下输入参数：

* `prompt`: 提示词，描述画面中出现的内容。
* `negative_prompt`: 负向提示词，描述画面中不应该出现的内容，默认值为 `""`。
* `input_image`: 输入图片，适用于图生视频模型，例如 [`Wan-AI/Wan2.1-I2V-14B-480P`](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P)、[`PAI/Wan2.1-Fun-1.3B-InP`](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-InP)，以及首尾帧模型，例如 [`Wan-AI/Wan2.1-FLF2V-14B-720P`](Wan-AI/Wan2.1-FLF2V-14B-720P)。
* `end_image`: 结尾帧，适用于首尾帧模型，例如 [`Wan-AI/Wan2.1-FLF2V-14B-720P`](Wan-AI/Wan2.1-FLF2V-14B-720P)。
* `input_video`: 输入视频，用于视频生视频，适用于任意 Wan 系列模型，需与参数 `denoising_strength` 配合使用。
* `denoising_strength`: 去噪强度，范围为 [0, 1]。数值越小，生成的视频越接近 `input_video`。
* `control_video`: 控制视频，适用于带控制能力的 Wan 系列模型，例如 [`PAI/Wan2.1-Fun-1.3B-Control`](https://modelscope.cn/models/PAI/Wan2.1-Fun-1.3B-Control)。
* `reference_image`: 参考图片，适用于带参考图能力的 Wan 系列模型，例如 [`PAI/Wan2.1-Fun-V1.1-1.3B-Control`](https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control)。
* `camera_control_direction`: 镜头控制方向，可选 "Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown" 之一，适用于 Camera-Control 模型，例如 [PAI/Wan2.1-Fun-V1.1-14B-Control-Camera](https://www.modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control-Camera)。
* `camera_control_speed`: 镜头控制速度，适用于 Camera-Control 模型，例如 [PAI/Wan2.1-Fun-V1.1-14B-Control-Camera](https://www.modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control-Camera)。
* `camera_control_origin`: 镜头控制序列的原点坐标，请参考[原论文](https://arxiv.org/pdf/2404.02101)进行设置，适用于 Camera-Control 模型，例如 [PAI/Wan2.1-Fun-V1.1-14B-Control-Camera](https://www.modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-14B-Control-Camera)。
* `vace_video`: VACE 模型的输入视频，适用于 VACE 系列模型，例如 [`iic/VACE-Wan2.1-1.3B-Preview`](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview)。
* `vace_video_mask`: VACE 模型的 mask 视频，适用于 VACE 系列模型，例如 [`iic/VACE-Wan2.1-1.3B-Preview`](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview)。
* `vace_reference_image`: VACE 模型的参考图片，适用于 VACE 系列模型，例如 [`iic/VACE-Wan2.1-1.3B-Preview`](https://modelscope.cn/models/iic/VACE-Wan2.1-1.3B-Preview)。
* `vace_scale`: VACE 模型对基础模型的影响程度，默认为1。数值越大，控制强度越高，但画面崩坏概率越大。
* `seed`: 随机种子。默认为 `None`，即完全随机。
* `rand_device`: 生成随机高斯噪声矩阵的计算设备，默认为 `"cpu"`。当设置为 `cuda` 时，在不同 GPU 上会导致不同的生成结果。
* `height`: 帧高度，默认为 480。需设置为 16 的倍数，不满足时向上取整。
* `width`: 帧宽度，默认为 832。需设置为 16 的倍数，不满足时向上取整。
* `num_frames`: 帧数，默认为 81。需设置为 4 的倍数 + 1，不满足时向上取整，最小值为 1。
* `cfg_scale`: Classifier-free guidance 机制的数值，默认为 5。数值越大，提示词的控制效果越强，但画面崩坏的概率越大。
* `cfg_merge`: 是否合并 Classifier-free guidance 的两侧进行统一推理，默认为 `False`。该参数目前仅在基础的文生视频和图生视频模型上生效。
* `switch_DiT_boundary`: 切换 DiT 模型的时间点，默认值为 0.875，仅对多 DiT 的混合模型生效，例如 [Wan-AI/Wan2.2-I2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B)。
* `num_inference_steps`: 推理次数，默认值为 50。
* `sigma_shift`: Rectified Flow 理论中的参数，默认为 5。数值越大，模型在去噪的开始阶段停留的步骤数越多，可适当调大这个参数来提高画面质量，但会因生成过程与训练过程不一致导致生成的视频内容与训练数据存在差异。
* `motion_bucket_id`: 运动幅度，范围为 [0, 100]。适用于速度控制模块，例如 [`DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1`](https://modelscope.cn/models/DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1)，数值越大，运动幅度越大。
* `tiled`: 是否启用 VAE 分块推理，默认为 `False`。设置为 `True` 时可显著减少 VAE 编解码阶段的显存占用，会产生少许误差，以及少量推理时间延长。
* `tile_size`: VAE 编解码阶段的分块大小，默认为 (30, 52)，仅在 `tiled=True` 时生效。
* `tile_stride`: VAE 编解码阶段的分块步长，默认为 (15, 26)，仅在 `tiled=True` 时生效，需保证其数值小于或等于 `tile_size`。
* `sliding_window_size`: DiT 部分的滑动窗口大小。实验性功能，效果不稳定。
* `sliding_window_stride`: DiT 部分的滑动窗口步长。实验性功能，效果不稳定。
* `tea_cache_l1_thresh`: TeaCache 的阈值，数值越大，速度越快，画面质量越差。请注意，开启 TeaCache 后推理速度并非均匀，因此进度条上显示的剩余时间将会变得不准确。
* `tea_cache_model_id`: TeaCache 的参数模板，可选 `"Wan2.1-T2V-1.3B"`、`Wan2.1-T2V-14B`、`Wan2.1-I2V-14B-480P`、`Wan2.1-I2V-14B-720P` 之一。
* `progress_bar_cmd`: 进度条，默认为 `tqdm.tqdm`。可通过设置为 `lambda x:x` 来屏蔽进度条。

</details>