# -*- coding: utf-8 -*-
import json
import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(params, tc):
    """AdamW + 初始线性 warmup（此后学习率保持不变）"""
    opt = torch.optim.AdamW([p for p in params if p.requires_grad], lr=tc.lr, weight_decay=tc.weight_decay)
    w = max(1, tc.warmup_epochs)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda e: min(1.0, (e + 1) / w))


def make_generator(device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def freeze(module: torch.nn.Module) -> torch.nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    return module.eval()


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class BestKeeper:
    """按验证指标保存最优 checkpoint。"""

    def __init__(self, path: str, mode: str = "max"):
        self.path, self.mode = path, mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.epoch = 0

    def update(self, value: float, epoch: int, payload: dict) -> bool:
        if (value > self.best) if self.mode == "max" else (value < self.best):
            self.best, self.epoch = value, epoch
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            torch.save(payload, self.path)
            return True
        return False
