# -*- coding: utf-8 -*-
"""
模态编码器（论文 Section III-A）
    CT      : 3D ResNet-34 + 线性投影
    FS      : 冻结 TITAN（预提取 slide embedding）+ 可训练投影 φ_fs
    Tabular : TabTransformer + 线性投影
    术后 WSI: 冻结 TITAN + 可训练投影 φ_post + PAM（Eq.1）
"""
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    """Linear -> ReLU -> Dropout -> Linear"""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x.float())


# ======================================================================
# 3D ResNet-34（BasicBlock × [3,4,6,3]；GroupNorm 代替 BatchNorm 以适应小 batch）
# ======================================================================
def _norm(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(16, c)


class BasicBlock3d(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = _norm(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = _norm(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), _norm(cout))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = F.relu(self.n1(self.conv1(x)), inplace=True)
        return F.relu(self.n2(self.conv2(out)) + idt, inplace=True)


class ResNet3D34(nn.Module):
    def __init__(self, in_channels: int = 1, out_dim: int = 768, layers=(3, 4, 6, 3)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, 7, stride=(1, 2, 2), padding=3, bias=False),
            _norm(64), nn.ReLU(inplace=True), nn.MaxPool3d(3, 2, 1),
        )
        stages, cin = [], 64
        for i, (c, n) in enumerate(zip((64, 128, 256, 512), layers)):
            blocks = [BasicBlock3d(cin, c, 1 if i == 0 else 2)] + [BasicBlock3d(c, c) for _ in range(n - 1)]
            stages.append(nn.Sequential(*blocks))
            cin = c
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(512, out_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.proj(torch.flatten(self.pool(self.stages(self.stem(x))), 1))


# ======================================================================
# TabTransformer：类别列 token + Transformer（CLS），连续列标准化 + 投影，拼接后 MLP
# ======================================================================
class _ResMLP(nn.Module):
    def __init__(self, d: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, d))

    def forward(self, x):
        return x + self.net(x)


class TabTransformer(nn.Module):
    def __init__(self, n_num: int, cat_cardinalities: Sequence[int], out_dim: int = 768,
                 emb_dim: int = 32, heads: int = 4, layers: int = 4, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.n_num, self.n_cat = n_num, len(cat_cardinalities)
        fused = 0
        if self.n_cat:
            self.embs = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cat_cardinalities])
            self.col_emb = nn.Parameter(torch.zeros(self.n_cat, emb_dim))
            self.cls = nn.Parameter(torch.zeros(1, 1, emb_dim))
            layer = nn.TransformerEncoderLayer(emb_dim, heads, max(256, 2 * emb_dim), dropout,
                                               activation="relu", batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.cat_norm = nn.LayerNorm(emb_dim)
            fused += emb_dim
        if n_num:
            # 可学习标准化 (x - μ) / |σ| · γ + β
            self.num_mu = nn.Parameter(torch.zeros(n_num))
            self.num_sigma = nn.Parameter(torch.ones(n_num))
            self.num_gamma = nn.Parameter(torch.ones(n_num))
            self.num_beta = nn.Parameter(torch.zeros(n_num))
            self.num_proj = nn.Sequential(nn.Linear(n_num, max(32, hidden // 2)), nn.ReLU())
            fused += max(32, hidden // 2)
        if fused == 0:
            raise ValueError("表格既无连续列也无类别列")
        self.stem = nn.Sequential(nn.Linear(fused, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.backbone = nn.Sequential(*[_ResMLP(hidden, max(hidden, 4 * hidden // 3), dropout) for _ in range(2)])
        self.head_norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, out_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        if self.n_cat:
            nn.init.normal_(self.cls, std=0.02)

    def forward(self, x_num, x_cat, num_miss=None):
        parts = []
        if self.n_cat:
            tok = torch.stack([e(x_cat[:, i]) for i, e in enumerate(self.embs)], 1) + self.col_emb
            tok = torch.cat([self.cls.expand(tok.size(0), -1, -1), tok], 1)
            parts.append(self.cat_norm(self.transformer(tok))[:, 0])
        if self.n_num:
            x = (x_num - self.num_mu) / (self.num_sigma.abs() + 1e-6) * self.num_gamma + self.num_beta
            if num_miss is not None:
                x = x * (~num_miss).float()
            parts.append(self.num_proj(x))
        h = self.backbone(self.stem(torch.cat(parts, -1)))
        return self.proj(self.head_norm(h))


# ======================================================================
# PAM (Eq.1)
# ======================================================================
class PAM(nn.Module):
    """
    m = max_n h_n,  s_n = w_a^T ReLU(W_h h_n + W_m m),  a = softmax(s),  z = m + Σ a_n h_n
    """

    def __init__(self, d: int = 768, hidden: int = 256):
        super().__init__()
        self.W_h = nn.Linear(d, hidden, bias=False)
        self.W_m = nn.Linear(d, hidden, bias=False)
        self.w_a = nn.Linear(hidden, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [N,d] -> [1,d]"""
        m = h.max(0, keepdim=True)[0]
        a = torch.softmax(self.w_a(F.relu(self.W_h(h) + self.W_m(m))), 0)
        return m + (a * h).sum(0, keepdim=True)


# ======================================================================
# 组合编码器
# ======================================================================
class IntraEncoder(nn.Module):
    """术中三模态 -> z_ct, z_fs, z_tab"""

    def __init__(self, mc, n_num: int, cat_cardinalities: Sequence[int]):
        super().__init__()
        self.ct = ResNet3D34(1, mc.d)
        self.fs_proj = nn.Linear(mc.titan_dim, mc.d)
        self.tab = TabTransformer(n_num, cat_cardinalities, mc.d, mc.tab_emb_dim, mc.tab_heads,
                                  mc.tab_layers, mc.tab_hidden, mc.tab_dropout)

    def forward(self, batch: Dict, device) -> Dict[str, torch.Tensor]:
        return {
            "z_ct": self.ct(batch["CT"].to(device)),
            "z_fs": self.fs_proj(batch["fs"].to(device)),
            "z_tab": self.tab(batch["x_num"].to(device), batch["x_cat"].to(device), batch["num_miss"].to(device)),
        }


class PostEncoder(nn.Module):
    """术后 WSI 列表 -> z_post（仅训练期使用）"""

    def __init__(self, mc):
        super().__init__()
        self.post_proj = nn.Linear(mc.titan_dim, mc.d)
        self.pam = PAM(mc.d, mc.pam_hidden)

    def forward(self, batch: Dict, device) -> torch.Tensor:
        return torch.cat([self.pam(self.post_proj(torch.stack(slides).to(device)))
                          for slides in batch["post"]], 0)
