"""
PyTorch 对齐验证脚本
使用方法：
    python align.py --model gpt2 --scheduler steplr --steps 10
需要安装：torch, transformers
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    StepLR, LinearLR, SequentialLR, ChainedScheduler, ConstantLR, LambdaLR
)
import argparse
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model(model_name):
    if model_name == 'gpt2':
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config(
            n_layer=2,
            n_head=2,
            n_embd=64,
            vocab_size=1000
        )
        model = GPT2LMHeadModel(config)
    elif model_name == 'llama':
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=64,
            intermediate_size=128,
            vocab_size=1000
        )
        model = LlamaForCausalLM(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

def get_scheduler(name, optimizer, base_lr, warmup_steps, step_size, gamma, milestones=None, scheduler_list=None):
    if name == 'constant':
        return ConstantLR(optimizer, factor=1.0, total_iters=0)  # 实际 constant 需自定义，这里用 factor=1.0 替代
        # 更准确的 constant 可以用 LambdaLR 实现
        return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    elif name == 'steplr':
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == 'linear':
        # PyTorch LinearLR 是线性从 factor 到 1.0，这里我们需要自定义 warmup
        # 可以用 LambdaLR 模拟线性从0到1
        return LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0)
    elif name == 'sequential':
        # 解析 scheduler_list
        types = scheduler_list.split(',')
        milestones = [int(m) for m in milestones.split(',')]
        schedulers = []
        for t in types:
            if t == 'linear':
                schedulers.append(LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / warmup_steps)))
            elif t == 'steplr':
                schedulers.append(StepLR(optimizer, step_size=step_size, gamma=gamma))
            else:
                schedulers.append(ConstantLR(optimizer, factor=1.0))
        return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
    elif name == 'chained':
        types = scheduler_list.split(',')
        schedulers = []
        for t in types:
            if t == 'linear':
                schedulers.append(LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / warmup_steps)))
            elif t == 'steplr':
                schedulers.append(StepLR(optimizer, step_size=step_size, gamma=gamma))
            else:
                schedulers.append(ConstantLR(optimizer, factor=1.0))
        return ChainedScheduler(schedulers)
    else:
        raise ValueError(f"Unknown scheduler: {name}")

def train_step(model, optimizer, scheduler, input_ids):
    model.train()
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2', choices=['gpt2', 'llama'])
    parser.add_argument('--scheduler', type=str, default='steplr', 
                        choices=['constant', 'steplr', 'linear', 'sequential', 'chained'])
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--step_size', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--milestones', type=str, default='5')
    parser.add_argument('--scheduler_list', type=str, default='linear,steplr')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(args.model).to(device)
    optimizer = AdamW(model.parameters(), lr=args.base_lr)
    scheduler = get_scheduler(
        args.scheduler, optimizer, args.base_lr, args.warmup_steps,
        args.step_size, args.gamma, args.milestones, args.scheduler_list
    )

    # 生成随机输入
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

    print(f"Model: {args.model}, Scheduler: {args.scheduler}, Steps: {args.steps}")
    print("Step\tLoss\tLR")
    for step in range(1, args.steps+1):
        loss = train_step(model, optimizer, scheduler, input_ids)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{step}\t{loss:.6f}\t{current_lr:.6f}")
