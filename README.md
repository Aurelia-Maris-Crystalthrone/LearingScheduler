# InfiniTrain 学习率调度器模块

## 概述

本模块为 InfiniTrain 训练框架提供独立、可扩展、状态可恢复的学习率调度器（LR Scheduler）实现。完全对齐 PyTorch `torch.optim.lr_scheduler` 的数值行为，支持基础调度策略（ConstantLR、StepLR、LinearLR、LambdaLR）和组合调度策略（SequentialLR、ChainedScheduler）。通过清晰的接口设计，调度器与优化器解耦，仅通过 `SetLearningRate()` 方法交互，满足职责单一、最小耦合的原则。

## 特性

- **与 PyTorch 数值对齐**：每个调度器的学习率变化曲线与 PyTorch 同名类完全一致。
- **状态可恢复**：所有调度器实现 `State()` / `LoadState()` 接口，支持训练中断后恢复。
- **易于扩展**：新增调度策略只需继承 `LRScheduler` 并实现 `ComputeLR()`，无需修改现有代码。
- **组合能力**：支持顺序组合（SequentialLR）和链式组合（ChainedScheduler），可任意嵌套。
- **零侵入**：不修改优化器内部逻辑，仅通过公开接口设置学习率。

## 快速开始

### 依赖

- C++11 或更高版本
- [gflags](https://github.com/gflags/gflags)（用于命令行解析，示例需要）

### 编译

将 `lr_scheduler.h` 放置于项目 `include/` 目录，并在需要使用的地方包含。示例程序 `main.cc` 可如下编译：

```bash
g++ -std=c++11 -Iinclude -lgflags -lpthread examples/main.cc -o lr_scheduler_demo