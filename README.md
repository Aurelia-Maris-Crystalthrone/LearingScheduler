# LearningScheduler
> 面向深度学习场景的多语言学习率调度器（LR Scheduler）实现库，重点适配大模型「无限训练（InfiniTrain）」场景，支持多种调度策略验证与对比。

## 目录
- [项目介绍](#项目介绍)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
  - [环境要求](#环境要求)
  - [编译与运行（C++版本）](#编译与运行c版本)
  - [快速验证（Python版本）](#快速验证python版本)
- [模块说明](#模块说明)
  - [demo 模块](#demo-模块)
  - [infinitrain 模块](#infinitrain-模块)
- [调度策略说明](#调度策略说明)
- [使用示例](#使用示例)
  - [C++ 基础使用](#c-基础使用)
  - [Python 无限训练适配](#python-无限训练适配)
- [日志说明](#日志说明)
- [文档参考](#文档参考)
- [常见问题](#常见问题)

## 项目介绍
LearningScheduler 是一套专注于**深度学习训练中学习率调度**的工具库，核心解决以下问题：
- 提供 C++（高性能）和 Python（易适配框架）双版本学习率调度器实现；
- 适配大模型「无限训练（InfiniTrain）」场景，优化长时训练的学习率衰减逻辑；
- 内置 StepLR、ChainedLR 等经典调度策略，支持效果验证与日志对比；
- 配套完整的设计文档和使用说明，保证策略实现的可复现性。

本仓库既适用于研究者快速验证不同调度策略对模型训练的影响，也适用于工程人员直接集成高性能调度模块到训练框架中。

## 核心特性
✅ **多语言支持**：C++ 底层实现（高性能）+ Python 上层适配（兼容 PyTorch/TensorFlow）  
✅ **场景适配**：专门优化「无限训练（InfiniTrain）」场景的调度逻辑  
✅ **多策略验证**：内置 StepLR、ChainedLR 等经典策略，支持 GPT2/Llama 模型验证  
✅ **可复现性**：提供完整训练日志和对比数据，保证策略效果可复现  
✅ **文档齐全**：配套设计文档、项目文档，清晰说明实现思路和使用方式  

## 快速开始
### 环境要求
| 环境类型       | 要求                                                                 |
|----------------|----------------------------------------------------------------------|
| 操作系统       | Linux (Ubuntu 18.04+) / macOS 10.15+ / Windows 10+ (WSL2 推荐)      |
| C++ 编译环境   | CMake 3.16+、GCC 7.5+ / Clang 10+                                    |
| Python 环境    | Python 3.8+、PyTorch 1.10+（可选，用于无限训练验证）                 |
| 依赖库（Python）| torch、transformers、numpy（`pip install -r requirements.txt`）       |

### 编译与运行（C++版本）
```bash
# 克隆仓库
git clone https://xxx.xxx.xxx/LearningScheduler.git
cd LearningScheduler

# 进入demo目录，创建编译目录
cd demo && mkdir build && cd build

# 编译代码
cmake ..
make -j4

# 运行StepLR示例（生成调度日志）
./step_lr_demo
```

### 快速验证（Python版本）
```bash
# 进入infinitrain目录
cd ../infinitrain

# 安装依赖
pip install -r requirements.txt

# 运行GPT2模型StepLR无限训练验证
python train.py --model gpt2 --scheduler step_lr --infini_train True

# 运行Llama模型ChainedLR无限训练验证
python train.py --model llama --scheduler chained_lr --infini_train True
```

## 模块说明
### demo 模块
核心为**基础调度器实现与验证**，面向通用深度学习场景，目录结构：
```
demo/
├── build/               # 编译产物目录（自动生成）
├── examples/            # C++调度器使用示例代码
├── lr_scheduler.h       # C++核心头文件（定义调度器类/接口）
├── align.py             # Python辅助脚本（多语言结果对齐/可视化）
├── CMakeLists.txt       # C++编译配置文件
├── steplr.log           # StepLR策略运行日志
├── 学习调度器-设计文档.md  # 核心设计思路与接口规范
└── 学习调度器-项目文档.md  # 完整使用说明与实现细节
```

### infinitrain 模块
核心为**无限训练场景适配与验证**，面向大模型长时训练，目录结构：
```
infinitrain/
├── lr_scheduler.py      # Python调度器实现（适配无限训练）
├── train.py             # 大模型训练脚本（集成调度器）
├── requirements.txt     # Python依赖清单
├── gpt2_chained_infini.log  # GPT2+ChainedLR无限训练日志
├── gpt2_steplr_infini.log    # GPT2+StepLR无限训练日志
├── llama_chained_infini.log  # Llama+ChainedLR无限训练日志
└── llama_steplr_infini.log    # Llama+StepLR无限训练日志
```

## 调度策略说明
| 策略名称       | 核心逻辑                                                                 | 适用场景                     |
|----------------|--------------------------------------------------------------------------|------------------------------|
| StepLR（阶梯LR）| 每经过指定步数，学习率乘以衰减因子（如每1000步×0.1）                     | 通用场景、短周期训练         |
| ChainedLR（链式LR）| 组合多个调度策略，按顺序执行（如先StepLR后CosineAnnealing）             | 大模型长时训练、多阶段调度   |
| CosineAnnealingLR（余弦退火） | 学习率按余弦函数周期性衰减                                               | 无限训练、避免学习率过早停滞 |

## 使用示例
### C++ 基础使用
```cpp
#include "lr_scheduler.h"
#include <iostream>

int main() {
    // 初始化StepLR调度器：初始学习率0.01，步长100，衰减因子0.1
    StepLRScheduler scheduler(0.01, 100, 0.1);
    
    // 模拟训练步数，输出学习率变化
    for (int step = 0; step < 300; step++) {
        float lr = scheduler.get_lr(step);
        if (step % 100 == 0) {
            std::cout << "Step: " << step << ", LR: " << lr << std::endl;
        }
    }
    return 0;
}
```
**输出示例**：
```
Step: 0, LR: 0.01
Step: 100, LR: 0.001
Step: 200, LR: 0.0001
```

### Python 无限训练适配
```python
from lr_scheduler import InfiniStepLRScheduler
import torch
import torch.nn as nn

# 初始化模型（以GPT2为例）
from transformers import GPT2Model
model = GPT2Model.from_pretrained("gpt2")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 初始化无限训练StepLR调度器：步长1000，衰减因子0.9，无最小学习率限制
scheduler = InfiniStepLRScheduler(optimizer, step_size=1000, gamma=0.9)

# 模拟无限训练流程
step = 0
while True:  # 无限训练循环
    # 训练步骤（省略前向/反向传播）
    loss = torch.tensor(1.0, requires_grad=True)
    loss.backward()
    optimizer.step()
    
    # 更新学习率
    scheduler.step(step)
    if step % 1000 == 0:
        print(f"Step: {step}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    step += 1
    if step > 5000:  # 示例终止条件
        break
```

## 日志说明
仓库中已包含的日志文件记录了不同模型+调度策略的训练细节，核心字段说明：
| 字段          | 含义                                                                 |
|---------------|----------------------------------------------------------------------|
| step          | 训练步数                                                             |
| lr            | 当前学习率                                                           |
| loss          | 训练损失值                                                           |
| infini_flag   | 无限训练标记（True表示启用无限训练调度逻辑）                         |
| eta           | 剩余训练时间（无限训练场景下为预估值）                               |

可通过 `demo/align.py` 脚本解析日志并可视化学习率变化：
```bash
python demo/align.py --log_path infinitrain/gpt2_steplr_infini.log --save_plot lr_curve.png
```

## 文档参考
- 《学习调度器-设计文档.md》：核心调度器的设计思路、类结构、接口定义；
- 《学习调度器-项目文档.md》：完整的项目背景、实现细节、测试用例；
- `examples/` 目录：包含更多C++/Python使用示例，覆盖不同调度策略。

## 常见问题
| 问题现象                          | 解决方案                                                                 |
|-----------------------------------|--------------------------------------------------------------------------|
| C++编译报错「找不到头文件」       | 检查CMakeLists.txt中头文件路径配置，确保`lr_scheduler.h`在指定目录下     |
| Python运行提示「模型加载失败」    | 确保transformers库版本≥4.20.0，或手动下载GPT2/Llama权重到本地目录       |
| 无限训练日志中学习率无变化        | 检查调度器`step_size`参数是否设置合理，或确认`step`参数是否正确传入`scheduler.step()` |
| 多语言调度器结果不一致            | 使用`demo/align.py`对比日志，检查浮点数精度或调度器参数是否完全对齐       |

---

### 总结
1. LearningScheduler 是多语言学习率调度器库，核心适配大模型无限训练场景，支持StepLR、ChainedLR等经典策略；
2. 仓库分为demo（基础实现）和infinitrain（无限训练验证）两大模块，提供C++/Python双版本实现，兼顾性能与易用性；
3. 配套完整的日志、文档和示例代码，可快速验证不同调度策略对GPT2/Llama等大模型训练的影响。
