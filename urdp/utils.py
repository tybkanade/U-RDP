# -*- coding: utf-8 -*-
import contextlib
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


def autocast(device: str, amp: str = "bf16"):
    """混合精度上下文；CPU 或 amp="none" 时不做任何事。"""
    if not str(device).startswith("cuda") or amp == "none":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


class EMA:
    """参数的指数滑动平均。apply() 期间模块使用 EMA 权重，退出时恢复训练权重。"""

    def __init__(self, module: torch.nn.Module, decay: float):
        self.module, self.decay = module, decay
        self.shadow = {k: p.detach().clone() for k, p in module.named_parameters()}

    @torch.no_grad()
    def update(self):
        for k, p in self.module.named_parameters():
            self.shadow[k].lerp_(p.detach(), 1 - self.decay)

    @contextlib.contextmanager
    def apply(self):
        backup = {k: p.detach().clone() for k, p in self.module.named_parameters()}
        with torch.no_grad():
            for k, p in self.module.named_parameters():
                p.copy_(self.shadow[k])
        try:
            yield
        finally:
            with torch.no_grad():
                for k, p in self.module.named_parameters():
                    p.copy_(backup[k])


def append_jsonl(obj, path: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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
    """
    按验证指标只保留最优的 top_k 个 checkpoint（默认 2 个），其余 epoch 的权重不落盘。

        第 1 名：path（如 stage5_cmaa.pth），后续阶段与推理都读取这个文件
        第 2 名：path 加后缀 _top2（如 stage5_cmaa_top2.pth），仅作备份
        <path>.topk.json 记录各名次对应的 epoch 与指标

    mode="max" 用于验证 acc（阶段 1、5）；mode="min" 用于验证损失（阶段 2-4，这些阶段没有分类输出）。
    """

    def __init__(self, path: str, mode: str = "max", top_k: int = 2):
        self.path, self.mode, self.top_k = path, mode, max(1, top_k)
        root, ext = os.path.splitext(path)
        self.paths = [path] + [f"{root}_top{i}{ext}" for i in range(2, self.top_k + 1)]
        self.records = []          # [(score, epoch)]，按名次排序，与 self.paths 一一对应
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        for p in self.paths:       # 清理同名旧文件，避免与本次训练混淆
            if os.path.isfile(p):
                os.remove(p)

    @property
    def best(self) -> float:
        return self.records[0][0] if self.records else (-float("inf") if self.mode == "max" else float("inf"))

    @property
    def epoch(self) -> int:
        return self.records[0][1] if self.records else 0

    def _better(self, a: float, b: float) -> bool:
        return a > b if self.mode == "max" else a < b

    def update(self, value: float, epoch: int, payload: dict) -> bool:
        """返回 True 表示当前 epoch 成为新的第 1 名。"""
        rank = next((i for i, (s, _) in enumerate(self.records) if self._better(value, s)), len(self.records))
        if rank >= self.top_k:
            return False
        # 名次 rank 及之后的文件依次后移一位（挤出最后一名），再把当前权重写到 rank 位置
        for i in range(min(len(self.records), self.top_k - 1), rank, -1):
            os.replace(self.paths[i - 1], self.paths[i])
        torch.save(payload, self.paths[rank])
        self.records.insert(rank, (float(value), int(epoch)))
        del self.records[self.top_k:]
        save_json([{"rank": i + 1, "epoch": e, "score": s, "file": os.path.basename(p)}
                   for i, ((s, e), p) in enumerate(zip(self.records, self.paths))], self.path + ".topk.json")
        return rank == 0
