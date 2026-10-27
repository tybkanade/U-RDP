# -*- coding: utf-8 -*-
"""
U-RDP Pathology Feature Generation Module（论文 Section III-B）

    AnchorPredictor   : Eq.3   z_anchor = z_fs + W2 D0.2[ReLU(W1 LN(z_fs))]
    CPD               : 8 层 decoder-only Transformer（hidden 1024, 16 heads, FFN 4096, dropout 0.2）
    ResidualDiffusion : Eq.4-6 残差 r = z_post - z_anchor 上的 v-parameterized 扩散（T=100, cosine）
                        + 64 步 Analytic-DPM 逆过程
    GUGate            : Eq.7-9 K 次采样的均值/方差 -> g = σ(b_g - softplus(a_g) ⊙ ν)
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from urdp.losses import align_loss


class SinusoidalPosEmb(nn.Module):
    """t: [B] -> [B, dim]"""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        half = dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(base) * torch.arange(half) / max(half - 1, 1)),
                             persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t.float().view(-1, 1) * self.freqs.view(1, -1)
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ======================================================================
# Anchor Predictor (Eq.3)
# ======================================================================
class AnchorPredictor(nn.Module):
    def __init__(self, d: int = 768, dropout: float = 0.2):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.W1 = nn.Linear(d, d)
        self.W2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, z_fs: torch.Tensor) -> torch.Tensor:
        return z_fs + self.W2(self.drop(F.relu(self.W1(self.ln(z_fs)))))


# ======================================================================
# Conditional Pathology Decoder (CPD)
# ======================================================================
class CPD(nn.Module):
    """
    v_θ = f_θ(r_t, t, c)
    decoder 序列为 [time token, r_t token]（因果自注意力），
    条件 c = [z_ct, z_fs, z_tab] 作为 cross-attention memory；读取最后一个 token 输出 v。
    """

    def __init__(self, d: int = 768, hidden: int = 1024, layers: int = 8, heads: int = 16,
                 ffn: int = 4096, dropout: float = 0.2, time_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.r_in = nn.Linear(d, hidden)
        self.cond_in = nn.Linear(d, hidden)
        self.pos = nn.Parameter(torch.zeros(1, 2, hidden))       # decoder 序列位置
        self.cond_pos = nn.Parameter(torch.zeros(1, 3, hidden))  # 三个条件模态的类型嵌入
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.cond_pos, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=ffn, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, d)
        self.register_buffer(
            "causal_mask", torch.triu(torch.full((2, 2), float("-inf")), diagonal=1), persistent=False
        )

    def forward(self, r_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """r_t: [B,d], t: [B] (1..T), cond: [B,3,d] -> v_θ: [B,d]"""
        tok = torch.stack([self.time_mlp(t), self.r_in(r_t)], dim=1) + self.pos   # [B,2,H]
        mem = self.cond_in(cond) + self.cond_pos                                  # [B,3,H]
        h = self.decoder(tok, mem, tgt_mask=self.causal_mask)
        return self.out(self.out_norm(h[:, -1]))


# ======================================================================
# Residual diffusion (Eq.4-6) + Analytic-DPM sampler
# ======================================================================
def cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    """返回 ᾱ_0..ᾱ_T（长度 T+1，ᾱ_0 = 1）"""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    abar = f / f[0]
    betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
    abar = torch.cat([torch.ones(1, dtype=torch.float64), torch.cumprod(1 - betas, dim=0)])
    return abar


class ResidualDiffusion(nn.Module):
    def __init__(self, cpd: CPD, T: int = 100, sample_steps: int = 64,
                 lambda_b: float = 0.4, lambda_c: float = 0.2,
                 eps_num: float = 1e-8, sq_norm_reduction: str = "mean"):
        super().__init__()
        self.cpd = cpd
        self.T = T
        self.sample_steps = sample_steps
        self.lambda_b = lambda_b
        self.lambda_c = lambda_c
        self.eps_num = eps_num
        self.sq_norm_reduction = sq_norm_reduction

        abar = cosine_alpha_bar(T)
        self.register_buffer("abar", abar.float())           # [T+1]
        # Analytic-DPM：Γ_n = E‖ε_θ(x_n,n)‖² / d，训练后由 estimate_gamma 估计
        self.register_buffer("gamma", torch.ones(T + 1))
        self.register_buffer("gamma_ready", torch.tensor(False))

    # ---------------- 工具 ----------------
    def _coef(self, t: torch.Tensor):
        a = self.abar[t].unsqueeze(-1)
        return a.sqrt(), (1 - a).sqrt()

    # ---------------- 训练 ----------------
    def losses(self, r: torch.Tensor, cond: torch.Tensor):
        """
        r: [B,d] 真实残差，cond: [B,3,d]
        L_diff = ‖v - v_θ‖²；r̂_0 = √ᾱ r_t - √(1-ᾱ) v_θ；L_res = L_align(r̂_0, r; λb)
        L_CPD = λc L_diff + (1-λc) L_res
        """
        B = r.size(0)
        t = torch.randint(1, self.T + 1, (B,), device=r.device)
        eps = torch.randn_like(r)
        sa, s1a = self._coef(t)
        r_t = sa * r + s1a * eps                     # Eq.4
        v = sa * eps - s1a * r                       # Eq.5
        v_hat = self.cpd(r_t, t, cond)
        sq = (v_hat.float() - v).pow(2)
        l_diff = (sq.sum(-1) if self.sq_norm_reduction == "sum" else sq.mean(-1)).mean()
        r0_hat = sa * r_t - s1a * v_hat              # Eq.6
        l_res = align_loss(r0_hat, r, self.lambda_b, self.eps_num, self.sq_norm_reduction)
        l_cpd = self.lambda_c * l_diff + (1 - self.lambda_c) * l_res
        return {"loss": l_cpd, "diff_loss": l_diff, "res_loss": l_res, "r0_hat": r0_hat}

    # ---------------- Analytic-DPM ----------------
    @torch.no_grad()
    def estimate_gamma(self, cond_all: torch.Tensor, r_all: torch.Tensor,
                       batch_size: int = 256, n_rep: int = 1):
        """
        在训练集 (r, c) 上估计 Γ_n = E_q ‖ε_θ(x_n, n)‖² / d（Bao et al., 2022, Analytic-DPM）。
        v-param 下 ε_θ = √(1-ᾱ) x_t + √ᾱ v_θ。
        """
        was_training = self.training
        self.eval()
        device = self.abar.device
        d = r_all.size(-1)
        gamma = torch.ones(self.T + 1, device=device)
        for n in range(1, self.T + 1):
            acc, cnt = 0.0, 0
            for _ in range(n_rep):
                for i in range(0, r_all.size(0), batch_size):
                    r = r_all[i:i + batch_size].to(device)
                    c = cond_all[i:i + batch_size].to(device)
                    t = torch.full((r.size(0),), n, device=device, dtype=torch.long)
                    sa, s1a = self._coef(t)
                    x_t = sa * r + s1a * torch.randn_like(r)
                    eps_hat = s1a * x_t + sa * self.cpd(x_t, t, c).float()
                    acc += float(eps_hat.pow(2).sum())
                    cnt += r.size(0)
            gamma[n] = acc / (cnt * d)
        self.gamma.copy_(gamma)
        self.gamma_ready.fill_(True)
        self.train(was_training)

    def _timestep_seq(self, steps: int) -> torch.Tensor:
        """均匀子序列 τ: T = τ_S > ... > τ_1 > τ_0 = 0"""
        seq = torch.linspace(0, self.T, steps + 1).round().long()
        return torch.unique(seq)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps: Optional[int] = None,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Analytic-DPM（默认求解器配置：最优方差、无方差裁剪）逆过程，从 N(0,I) 生成 r̂。
        cond: [B,3,d] -> r̂: [B,d]
        """
        steps = steps or self.sample_steps
        B, d = cond.size(0), cond.size(-1)
        device = cond.device
        x = torch.randn(B, d, device=device, generator=generator)
        seq = self._timestep_seq(steps).to(device)
        for i in range(len(seq) - 1, 0, -1):
            n, s = int(seq[i]), int(seq[i - 1])
            a_n, a_s = self.abar[n], self.abar[s]
            t = torch.full((B,), n, device=device, dtype=torch.long)
            v_hat = self.cpd(x, t, cond).float()
            x0_hat = a_n.sqrt() * x - (1 - a_n).sqrt() * v_hat
            eps_hat = (1 - a_n).sqrt() * x + a_n.sqrt() * v_hat
            if s == 0:
                x = x0_hat
                break
            # DDPM 后验方差 λ² 与 Analytic-DPM 最优方差 σ*²
            lam2 = (1 - a_s) / (1 - a_n) * (1 - a_n / a_s)
            coef = (1 - a_s - lam2).clamp_min(0).sqrt()
            mean = a_s.sqrt() * x0_hat + coef * eps_hat
            if bool(self.gamma_ready):
                corr = (coef - ((1 - a_n) * a_s / a_n).sqrt()) ** 2
                sigma2 = lam2 + corr * (1 - (1 - a_n) * self.gamma[n])
                sigma2 = sigma2.clamp_min(0)
            else:
                sigma2 = lam2
            noise = torch.randn(B, d, device=device, generator=generator)
            x = mean + sigma2.sqrt() * noise
        return x

    @torch.no_grad()
    def sample_k(self, cond: torch.Tensor, K: int, steps: Optional[int] = None,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """K 条独立逆扩散轨迹（同一条件、独立初始噪声），并行计算。返回 [K,B,d]"""
        B = cond.size(0)
        cond_rep = cond.unsqueeze(0).expand(K, *cond.shape).reshape(K * B, *cond.shape[1:])
        r = self.sample(cond_rep, steps=steps, generator=generator)
        return r.view(K, B, -1)


# ======================================================================
# Generation Uncertainty Gate (Eq.7-9)
# ======================================================================
class GUGate(nn.Module):
    def __init__(self, d: int = 768):
        super().__init__()
        self.a_g = nn.Parameter(torch.zeros(d))
        self.b_g = nn.Parameter(torch.zeros(d))

    @staticmethod
    def stats(cands: torch.Tensor):
        """cands: [K,B,d] -> r̄, ν（Eq.7，有偏方差 1/K）"""
        r_bar = cands.mean(dim=0)
        nu = (cands - r_bar.unsqueeze(0)).pow(2).mean(dim=0)
        return r_bar, nu

    def forward(self, cands: torch.Tensor):
        r_bar, nu = self.stats(cands)
        g = torch.sigmoid(self.b_g - F.softplus(self.a_g) * nu)   # Eq.8
        return {"r_tilde": g * r_bar, "g": g, "r_bar": r_bar, "nu": nu}
