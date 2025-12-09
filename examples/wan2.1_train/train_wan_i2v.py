import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
import json
from diffsynth_npu.utils.device_utils import is_npu_available
from diffsynth_npu.utils.wan_utils.wan_utils import npu_optimize

if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

    torch.npu.config.allow_internal_format = False
        ### SP 通信适配
    from diffsynth_npu.wan_train.parallel_states import initialize_sequence_parallel_state, \
        destroy_sequence_parallel_group, get_sequence_parallel_state, set_sequence_parallel_state, \
        set_sequence_parallel_size, get_sequence_parallel_group, get_sequence_parallel_size
    import torch.distributed as dist

class I2VDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480,
                 width=832):

        if metadata_path.rsplit(".")[-1] == "json":
            metadata = pd.read_json(metadata_path)
            self.path = [os.path.join(base_path, file_path) for file_path in metadata["path"]]
            self.text = metadata["cap"].to_list()
        elif metadata_path.rsplit(".")[-1] == "csv":
            metadata = pd.read_csv(metadata_path)
            self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
            self.text = metadata["text"].to_list()
        else:
            raise ValueError("Only support metadata in json or csv format")

        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width

        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        width, height = image.size

        return image

    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (
                num_frames - 1) * interval:
            reader.close()
            return None

        frames = []
        first_frame_image = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            if first_frame_image is None:
                first_frame_image = frame
            frame = self.crop_and_resize(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        return frames, first_frame_image

    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames, first_frame_image = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id,
                                                                   self.frame_interval, self.num_frames,
                                                                   self.frame_process)
        return frames, first_frame_image

    def load_text_video_raw_data(self, data_id):
        text = self.path[data_id]
        video = self.load_video(self.path[data_id])
        data = {"text": text, "video": video}
        return data

    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        video, first_frame_image = self.load_video(path)
        data = {"text": text, "video": video, "first_frame_image": np.array(first_frame_image), "path": path}
        return data

    def __len__(self):
        return len(self.path)


class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, image_encoder_path, vae_path, num_frames, height, width, tiled=False,
                 tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models([text_encoder_path, image_encoder_path, vae_path])
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        self.num_frames = num_frames
        self.height = height
        self.width = width

    def test_step(self, batch, batch_idx):
        text, video, first_frame_image_tensor, path = batch["text"][0], batch["video"], batch["first_frame_image"][0], \
        batch["path"][0]
        self.pipe.device = self.device
        if video is not None:
            prompt_emb = self.pipe.encode_prompt(text)
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            first_frame_image = Image.fromarray(np.array(first_frame_image_tensor.cpu()))
            cond_data_dict = self.pipe.encode_image(first_frame_image, num_frames=self.num_frames, height=self.height,
                                                    width=self.width)
            data = {"latents": latents, "prompt_emb": prompt_emb, "clip_fea": cond_data_dict["clip_feature"][0],
                    "y": cond_data_dict["y"][0]}
            torch.save(data, path + ".tensors.pth")


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch):
        if metadata_path.rsplit(".")[-1] == "json":
            metadata = pd.read_json(metadata_path)
            self.path = [os.path.join(base_path, file_name) for file_name in metadata["path"]]
        elif metadata_path.rsplit(".")[-1] == "csv":
            metadata = pd.read_csv(metadata_path)
            self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        else:
            raise ValueError("Only support metadata in json or csv format")

        print(len(self.path), "videos in metadata.")
        self.path = [i + ".tensors.pth" for i in self.path if os.path.exists(i + ".tensors.pth")]
        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0

        self.steps_per_epoch = steps_per_epoch

    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0] # 精度对齐
        data_id = (data_id + index) % len(self.path)  # For fixed seed.
        path = self.path[index % len(self.path)]
        data = torch.load(path, weights_only=True, map_location="cpu")
        return data

    def __len__(self):
        return self.steps_per_epoch


class LightningModelForTrain(pl.LightningModule):
    def __init__(self, dit_path, learning_rate=1e-5, lora_rank=4, lora_alpha=4, train_architecture="lora",
                 lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
                 use_gradient_checkpointing=True):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            # 将 dit_path 从字符串解析为 Python 列表
            dit_path = json.loads(dit_path)
            model_manager.load_models([dit_path])

        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)

        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()

    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2",
                          init_lora_weights="kaiming"):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

    def training_step(self, batch, batch_idx):
        # 初始化序列并行子通信组
        if ARGS.sequence_context_parallelism_size > 1 and get_sequence_parallel_group() is None:
            # print(f"---------------SP序列并行策略: 开启, SP_SIZE: {ARGS.sequence_context_parallelism_size}------------------")
            sp_size = ARGS.sequence_context_parallelism_size # 假设你想要的 SP size 是 2（根据你硬件配置决定）
            initialize_sequence_parallel_state(sp_size)
            set_sequence_parallel_state(True)
            set_sequence_parallel_size(sp_size)

        # Data
        latents = batch["latents"].to(self.device)
        prompt_emb = batch["prompt_emb"]
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        clip_feature = batch["clip_fea"].to(self.device)
        y = batch["y"].to(self.device)

        # Loss
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(self.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self.device).type):
            noise_pred = self.pipe.denoising_model()(
                noisy_latents, timestep=timestep, **prompt_emb, **extra_input,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                clip_feature=clip_feature, y=y
            )
            loss = torch.nn.functional.mse_loss(noise_pred[..., 1:].float(), training_target[..., 1:].float())
            loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(
            filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--metadata_name",
        type=str,
        default="metadata.json",
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="The number of gpu.",
    )
    parser.add_argument(
        "--sequence_context_parallelism_size",
        type=int,
        default=1,
        help="sequence_context_parallelism_size",
    )

    args = parser.parse_args()
    return args


def data_process(args):
    dataset = I2VDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, args.metadata_name),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width)
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)


from torch.utils.data.distributed import DistributedSampler

def init_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')

class DistributedSequenceParallelismSampler(DistributedSampler):
    def __init__(self, dataset,  cp_size=2, shuffle=False):

        self.dataset = dataset
        # print("SP序列并行暂不支持数据shuffle")
        self.shuffle = shuffle
        self.epoch = 0
        
        # 获取并行信息 # 右侧数据为预设 4卡 CP2 情况
        self.world_size = dist.get_world_size()  # 总GPU数 (4)
        self.rank = dist.get_rank()             # 当前GPU rank (0-3)
        self.cp_size = cp_size                        # CP组大小 (2)
        self.cp_group = self.rank // self.cp_size  # CP组号 (0或1)
        self.tot_cp_groups_num = self.world_size // self.cp_size # CP 组的数量
        
        # 计算总批次数和分配方案
        self.total_batches = len(dataset)
        self.batches_per_group = (self.total_batches + 1) // 2  # 向上取整


        if cp_size > 1:
            """开启SP序列并行"""
            # ========== 断言检查区块 ==========
            assert torch.distributed.is_initialized(), "必须首先初始化分布式环境"
            assert cp_size > 0, f"CP组大小必须为正数，当前为{cp_size}"
            assert cp_size > 0 and shuffle==False, f"开启SP并行的场景下,暂不支持开shuffle"
            assert self.world_size % cp_size == 0, (
                f"GPU总数{self.world_size}必须能被cp_size={cp_size}整除，"
                f"当前余数为{self.world_size % cp_size}"
            )
            assert len(dataset) >= self.world_size // cp_size, (
                f"数据集大小{len(dataset)}不能小于CP组数量{self.world_size//cp_size}，"
                "否则会导致部分GPU无数据"
            )
            # ================================

    def __iter__(self):
        # 生成交替分配的批次索引
        indices = []

        if self.cp_size == 1: 
            """不开启CP情况的处理"""
            for global_idx in range(len(self.dataset)):
                if global_idx % self.world_size == self.rank:
                    indices.append(global_idx)
        else:
            """开启CP情况的处理"""
            # 情况1: 总CP组数=1 (数据并行模式)
            if self.tot_cp_groups_num == 1:
                indices = list(range(len(self.dataset)))
            
            # 情况2: 多CP组交替分配
            else:
                for global_idx in range(len(self.dataset)):
                    assigned_group = global_idx % self.tot_cp_groups_num
                    if assigned_group == (self.rank // self.cp_size):
                        indices.append(global_idx)
            
        return iter(indices)

    def __len__(self):
        return self.batches_per_group

    def set_epoch(self, epoch):
        self.epoch = epoch 


def train(args):
    init_distributed()
    if args.sequence_context_parallelism_size <= 1 \
            and get_sequence_parallel_group() is None: 
        print("---------------SP序列并行策略: 未开启------------------")

    dataset = TensorDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, args.metadata_name),
        steps_per_epoch=args.steps_per_epoch,
    )
    cp_sp_parallelism_sampler = DistributedSequenceParallelismSampler(dataset, \
                                cp_size=args.sequence_context_parallelism_size, shuffle=False) # 支持cp_sp的sampler，暂不支持shuffle

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        sampler=cp_sp_parallelism_sampler
    )
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy, 
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)], # trainer里原配置
    )
    trainer.fit(model, dataloader)

ARGS = []
if __name__ == '__main__':
    npu_optimize(["npu_rope_apply", "npu_rms_norm", "WanModel", "SelfAttention", "CrossAttention", "flash_attention_sequence_parallelism"])
    args = parse_args()
    ARGS = args
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
