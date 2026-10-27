# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


def align_loss(u: torch.Tensor, v: torch.Tensor, lam: float, eps_num: float = 1e-8,
               sq_norm_reduction: str = "mean", reduction: str = "mean") -> torch.Tensor:
    """
    论文 Eq.2：L_align(u,v;λ) = λ‖u-v‖² + (1-λ)(1 - <u,v>/(‖u‖‖v‖ + ε_num))
    sq_norm_reduction="sum" 为字面平方 L2 范数；"mean" 除以维度，使两项量级相当。
    """
    u, v = u.float(), v.float()
    sq = (u - v).pow(2)
    sq = sq.sum(-1) if sq_norm_reduction == "sum" else sq.mean(-1)
    cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1) + eps_num)
    loss = lam * sq + (1 - lam) * (1 - cos)
    return loss.mean() if reduction == "mean" else loss


class FocalLoss(nn.Module):
    """Focal loss（γ=2.0，无类别特定 α）"""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits.float(), target, reduction="none")
        return ((1 - torch.exp(-ce)) ** self.gamma * ce).mean()
