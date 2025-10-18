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
### 4.1. 基础功能适配
| 功能名称 | 状态 |
| - | - |
| xDiT 并行 | 已适配✅ |
| dtype 适配 | 已适配✅ |

### 4.2. 亲和算子适配
| 功能名称 | 状态 |
| - | - |
| NPU Laser Attention | 已适配✅ |
| NPU Layer Norm | 计划中🕐 |
| NPU RmsNorm | 计划中🕐 |
| NPU Fast Gelu | 计划中🕐 |

### 4.3. 深度优化适配
| 功能名称 | 状态 |
| - | - |
| NPU Cache | 计划中🕐 |
| Rope 前置计算 | 计划中🕐 |
| VACE 并行 | 计划中🕐 |
| VAE 并行 | 计划中🕐 |
| CFG 并行 | 计划中🕐 |
| FSDP 分片 | 计划中🕐 |

## 5. 快速开始

### 5.1 编译安装
```shell
pip install diffsynth_npu
```

### 5.2 使用方式
```shell
import diffsynth
import diffsynth_npu

...
```