# InfiniTrain 学习率调度器模块

## 概述
本模块为 InfiniTrain 训练框架提供独立、可扩展、状态可恢复的学习率调度器实现，与 PyTorch 数值对齐。

## 特性
- 与 PyTorch 数值对齐
- 状态可恢复（Checkpoint/Resume）
- 易于扩展（继承 `LRScheduler` 即可）
- 支持组合策略（SequentialLR, ChainedScheduler）
- 零侵入优化器

## 快速开始（Windows 11 x64）

### 1. 安装依赖
- **Visual Studio 2019/2022**：安装时选择“使用 C++的桌面开发”工作负载。
- **CMake**（≥3.15）：[下载地址](https://cmake.org/download/)
- **vcpkg**（用于安装 gflags）：
  ```powershell
  git clone https://github.com/Microsoft/vcpkg.git
  cd vcpkg
  .\bootstrap-vcpkg.bat
  .\vcpkg integrate install
  .\vcpkg install gflags:x64-windows