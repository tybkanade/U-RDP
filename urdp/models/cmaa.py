# -*- coding: utf-8 -*-
"""
Cross-Modal Attention Aggregation Module（论文 Section III-C, Eq.10-12）

    y_m^(0) = W_m z_m^in + b_m,  M = {ct, fs, tab, post}
    第 ℓ 层：q_m = y_m^(ℓ-1)，C_m^(ℓ-1) 为其余三个模态
        Q = W_Q q, K = W_K C, V = W_V C, R = W_R C           (每头 d_h = d/H = 96)
        A = softmax(Q K^T / √d_h)                          (在 3 个上下文模态上 softmax)
        o = (σ(R) ⊙ V) A^T                                 (σ(R) 对上下文值逐维门控)
        y_m^(ℓ) = LN(q_m + W_O Concat(o^1..o^H))           (W_O ∈ R^{768×768})
    L 层后：a = softmax([ψ(y_m^(L))]_m)，f = Σ_m a_m y_m^(L)，ψ 为隐层 256 的两层 MLP
"""
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from urdp.models.encoders import MLPHead

MODALITIES = ("ct", "fs", "tab", "post")


class GatedCrossModalAttention(nn.Module):
    def __init__(self, d: int = 768, heads: int = 8):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_R = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.dh).transpose(1, 2)   # [B,H,L,dh]

    def forward(self, q: torch.Tensor, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """q: [B,d], ctx: [B,3,d] -> y: [B,d], A: [B,H,3]"""
        B = q.size(0)
        Q = self._split(self.W_Q(q).unsqueeze(1))              # [B,H,1,dh]
        K = self._split(self.W_K(ctx))                         # [B,H,3,dh]
        V = self._split(self.W_V(ctx))
        R = self._split(self.W_R(ctx))
        A = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.dh), dim=-1)  # [B,H,1,3]
        o = A @ (torch.sigmoid(R) * V)                          # [B,H,1,dh]
        o = o.transpose(1, 2).reshape(B, -1)                    # Concat heads -> [B,d]
        return self.ln(q + self.W_O(o)), A.squeeze(2)


class CMAALayer(nn.Module):
    def __init__(self, d: int = 768, heads: int = 8):
        super().__init__()
        self.attn = GatedCrossModalAttention(d, heads)

    def forward(self, Y: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Y: [B,4,d]，四个模态同时以上一层输出为输入"""
        outs, attns = [], []
        for i in range(Y.size(1)):
            ctx = torch.cat([Y[:, :i], Y[:, i + 1:]], dim=1)   # 其余三个模态
            y, A = self.attn(Y[:, i], ctx)
            outs.append(y)
            attns.append(A)
        return torch.stack(outs, dim=1), attns


class CMAA(nn.Module):
    def __init__(self, d: int = 768, heads: int = 8, num_layers: int = 4,
                 score_hidden: int = 256, num_classes: int = 3,
                 cls_hidden: int = 256, cls_dropout: float = 0.2):
        super().__init__()
        self.in_proj = nn.ModuleDict({m: nn.Linear(d, d) for m in MODALITIES})
        self.layers = nn.ModuleList([CMAALayer(d, heads) for _ in range(num_layers)])
        self.psi = nn.Sequential(nn.Linear(d, score_hidden), nn.ReLU(), nn.Linear(score_hidden, 1))
        self.head = MLPHead(d, cls_hidden, num_classes, cls_dropout)

    def forward(self, z_ct, z_fs, z_tab, z_post) -> Dict[str, torch.Tensor]:
        inputs = {"ct": z_ct, "fs": z_fs, "tab": z_tab, "post": z_post}
        Y = torch.stack([self.in_proj[m](inputs[m].float()) for m in MODALITIES], dim=1)  # [B,4,d]
        layer_attns = []
        for layer in self.layers:
            Y, attns = layer(Y)
            layer_attns.append(attns)
        a = torch.softmax(self.psi(Y).squeeze(-1), dim=1)       # [B,4]
        f = (a.unsqueeze(-1) * Y).sum(dim=1)                    # [B,d]
        return {
            "logits": self.head(f),
            "fusion_feature": f,
            "modal_weights": a,                                  # 顺序 [ct, fs, tab, post]
            "layer_attns": layer_attns,
        }

