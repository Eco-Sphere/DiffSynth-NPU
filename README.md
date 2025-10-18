# Ascend Extension for DiffSynth-Studio

## 1. 简介

本项目开发了名为diffsynth_npu的Ascend Extension for DiffSynth插件，使昇腾NPU可以适配DiffSynth框架，为使用DiffSynth框架的开发者提供昇腾AI处理器的超强算力。

昇腾为基于华为昇腾处理器和软件的行业应用及服务提供全栈AI计算基础设施。您可以通过访问[昇腾社区](https://www.hiascend.com/zh)，了解关于昇腾的更多信息。

## 2. 版本说明
| DiffSynth 版本 | DiffSynth-NPU 版本|
| - | - |
| 1.18.0 | 1.18.0 |

## 3. 当前版本支持的模型
| 模型名称 | 验证状态 | 运行样例 |
| - | - | - |
| Wan2.1-VACE-14B | 验证通过✅ | [doc](./examples/wanvideo/model_inference/docs/wan2.1_vace.md) |
| Wan2.1-VACE-1.3B | 未验证⏳ | / |
| Wan2.1-T2V-1.3B | 未验证⏳ | / |
| Wan2.1-T2V-14B | 未验证⏳ | / |
| Wan2.1-I2V-14B-480P | 未验证⏳ | / |
| Wan2.1-I2V-14B-720P | 未验证⏳ | / |
| Wan2.2-T2V-A14B | 未验证⏳ | / |
| Wan2.2-I2V-A14B | 未验证⏳ | / |
| Wan2.2-TI2V-5B | 未验证⏳ | / |
| Wan2.2-Animate-14B | 未验证⏳ | / |
| Wan2.2-S2V-14B | 未验证⏳ | / |

## 4. 未来支持/适配
### 4.1. 基础功能适配 - [基础功能 Patch](./diffsynth_npu/patch/base_patch.py)
| 功能名称 | 状态 | 路径 |
| - | - | - |
| xDiT 并行 | 已适配✅ | [distributed](./diffsynth_npu/utils/distributed) |
| dtype 适配 | 已适配✅ | [base_patch.py](./diffsynth_npu/patch/base_patch.py) |

### 4.2. 亲和算子适配
| 功能名称 | 状态 | 路径 |
| - | - | - |
| NPU Laser Attention | 已适配✅ | [attn_layer.py](./diffsynth_npu/utils/modules/attn_layer.py) |
| NPU Layer Norm | 计划中🕐 | \ |
| NPU RmsNorm | 计划中🕐 | \ |
| NPU Fast Gelu | 计划中🕐 | \ |

### 4.3. 深度优化适配
| 功能名称 | 状态 | 路径 |
| - | - | - |
| NPU Cache | 计划中🕐 | \ |
| PAD FREQS 优化 | 计划中🕐 | \ |
| Rope 前置计算 | 计划中🕐 | \ |
| VACE 并行 | 计划中🕐 | \ |
| VAE 并行 | 计划中🕐 | \ |
| CFG 并行 | 计划中🕐 | \ |
| FSDP 分片 | 计划中🕐 | \ |

## 5. 快速开始

### 5.1. 环境准备
#### 5.1.1 准备运行环境

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.11.10 | - |
  | torch | 2.1.0 | - |

#### 5.1.2 获取CANN&MindIE安装包&环境准备
- 设备支持
Atlas 800I/800T A2(8*64G)推理设备：支持的卡数最小为1
- [Atlas 800I/800T A2(8*64G)](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/softwareinst/instg/instg_0001.html)

#### 5.1.3 CANN安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构，{soc}表示昇腾AI处理器的版本。
chmod +x ./Ascend-cann-toolkit_{version}_linux-{arch}.run
chmod +x ./Ascend-cann-kernels-{soc}_{version}_linux.run
# 校验软件包安装文件的一致性和完整性
./Ascend-cann-toolkit_{version}_linux-{arch}.run --check
./Ascend-cann-kernels-{soc}_{version}_linux.run --check
# 安装
./Ascend-cann-toolkit_{version}_linux-{arch}.run --install
./Ascend-cann-kernels-{soc}_{version}_linux.run --install

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

#### 5.1.4 环境依赖安装
```shell
pip3 install -r requirements.txt
```

#### 5.1.5 MindIE安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x ./Ascend-mindie_${version}_linux-${arch}.run
./Ascend-mindie_${version}_linux-${arch}.run --check

# 方式一：默认路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install
# 设置环境变量
cd /usr/local/Ascend/mindie && source set_env.sh

# 方式二：指定路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install-path=${AieInstallPath}
# 设置环境变量
cd ${AieInstallPath}/mindie && source set_env.sh
```

#### 5.1.6 Torch_npu安装
下载 pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
```shell
tar -xzvf pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
# 解压后，会有whl包
pip install torch_npu-{pytorchversion}.xxxx.{arch}.whl
```

#### 5.1.7 gcc、g++安装
```shell
# 若环境镜像中没有gcc、g++，请用户自行安装
yum install gcc
yum install g++

# 导入头文件路径
export CPLUS_INCLUDE_PATH=/usr/include/c++/12/:/usr/include/c++/12/aarch64-openEuler-linux/:$CPLUS_INCLUDE_PATH
```
注：若使用openeuler镜像，需要配置gcc、g++环境，否则会导致`fatal error: 'stdio.h' file not found`

### 5.2 编译安装 DiffSynth-Studio
> 参考[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)

### 5.3 编译安装 DiffSynth-NPU
```shell
pip install -e ./
```

### 5.3 使用方式
```shell
import diffsynth
import diffsynth_npu

...
```