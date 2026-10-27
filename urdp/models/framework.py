# -*- coding: utf-8 -*-
"""
U-RDP 生成 + 融合部分（论文 Fig.3），在冻结编码器输出的 z_ct / z_fs / z_tab（及训练期 z_post）上运行：

    z_anchor = AnchorPredictor(z_fs)                         Eq.3
    r        = z_post - z_anchor                             （训练目标）
    c        = [z_ct, z_fs, z_tab]
    r̂^(1..K) = CPD 的 K 条 Analytic-DPM 逆过程               Eq.4-6
    r̃        = GU-Gate(r̂^(1..K))                            Eq.7-8
    z̃_post   = z_anchor + r̃                                 Eq.9
    logits   = CMAA(z_ct, z_fs, z_tab, z̃_post)               Eq.10-12
"""
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from urdp.losses import align_loss
from urdp.models.cmaa import CMAA
from urdp.models.generation import CPD, AnchorPredictor, GUGate, ResidualDiffusion

# 训练阶段 -> 该阶段训练的模块 / checkpoint 文件
STAGE_MODULES = {2: "anchor", 3: "cpd", 4: "gu_gate", 5: "cmaa"}
CKPT_NAMES = {2: "stage2_anchor.pth", 3: "stage3_cpd.pth", 4: "stage4_gu_gate.pth", 5: "stage5_cmaa.pth"}


class URDP(nn.Module):
    def __init__(self, mc):
        super().__init__()
        self.mc = mc
        self.anchor = AnchorPredictor(mc.d, mc.anchor_dropout)
        self.diffusion = ResidualDiffusion(
            CPD(mc.d, mc.cpd_hidden, mc.cpd_layers, mc.cpd_heads, mc.cpd_ffn, mc.cpd_dropout, mc.cpd_time_dim),
            T=mc.timesteps, sample_steps=mc.sample_steps, lambda_b=mc.lambda_b, lambda_c=mc.lambda_c,
            eps_num=mc.eps_num, sq_norm_reduction=mc.sq_norm_reduction,
        )
        self.gu_gate = GUGate(mc.d)
        self.cmaa = CMAA(mc.d, mc.cmaa_heads, mc.cmaa_layers, mc.cmaa_score_hidden,
                         mc.num_classes, mc.cls_hidden, mc.cls_dropout)

    def module(self, name: str) -> nn.Module:
        return self.diffusion.cpd if name == "cpd" else getattr(self, name)

    @staticmethod
    def condition(z_ct, z_fs, z_tab) -> torch.Tensor:
        return torch.stack([z_ct, z_fs, z_tab], dim=1)

    def _align(self, u, v, lam):
        return align_loss(u, v, lam, self.mc.eps_num, self.mc.sq_norm_reduction)

    # ---------------- 各阶段损失 ----------------
    def anchor_loss(self, z_fs, z_post):
        return self._align(self.anchor(z_fs), z_post, self.mc.lambda_a)

    def cpd_loss(self, z_ct, z_fs, z_tab, z_post) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            r = z_post - self.anchor(z_fs)
        return self.diffusion.losses(r, self.condition(z_ct, z_fs, z_tab))

    def gu_gate_loss(self, gen: Dict[str, torch.Tensor], z_post):
        return self._align(gen["r_tilde"], z_post - gen["z_anchor"], self.mc.lambda_d)

    # ---------------- 生成与分类 ----------------
    def generate(self, z_ct, z_fs, z_tab, K: Optional[int] = None, generator: Optional[torch.Generator] = None):
        """Anchor 与 CPD 不求梯度；GU-Gate 保留梯度。"""
        with torch.no_grad():
            z_anchor = self.anchor(z_fs)
            cands = self.diffusion.sample_k(self.condition(z_ct, z_fs, z_tab), K or self.mc.K, generator=generator)
        gate = self.gu_gate(cands)
        return {"z_anchor": z_anchor, "cands": cands, "z_post_tilde": z_anchor + gate["r_tilde"], **gate}

    def classify(self, z_ct, z_fs, z_tab, z_post_tilde):
        return self.cmaa(z_ct, z_fs, z_tab, z_post_tilde)

    @torch.no_grad()
    def predict(self, z_ct, z_fs, z_tab, K: Optional[int] = None, generator: Optional[torch.Generator] = None):
        """推理只用术中三模态。uncertainty 为候选逐维方差的平均（病例级不确定度）。"""
        gen = self.generate(z_ct, z_fs, z_tab, K=K, generator=generator)
        out = self.classify(z_ct, z_fs, z_tab, gen["z_post_tilde"])
        return {"probs": out["logits"].float().softmax(-1), "uncertainty": gen["nu"].mean(-1),
                "modal_weights": out["modal_weights"], **gen}

    # ---------------- 冻结与加载 ----------------
    def set_trainable(self, stage: int):
        """只训练当前阶段的模块，其余全部冻结并置于 eval。"""
        target = self.module(STAGE_MODULES[stage])
        self.requires_grad_(False).eval()
        target.requires_grad_(True).train()
        return list(target.parameters())

    def state_for(self, stage: int) -> Dict:
        state = {"module": self.module(STAGE_MODULES[stage]).state_dict()}
        if stage == 3:
            state["gamma"] = self.diffusion.gamma.cpu()
            state["gamma_ready"] = bool(self.diffusion.gamma_ready)
        return state

    def load_stage(self, stage: int, fold_dir: str, map_location="cpu"):
        ck = torch.load(os.path.join(fold_dir, CKPT_NAMES[stage]), map_location=map_location)
        self.module(STAGE_MODULES[stage]).load_state_dict(ck["module"])
        if stage == 3 and ck.get("gamma_ready"):
            self.diffusion.gamma.copy_(ck["gamma"])
            self.diffusion.gamma_ready.fill_(True)
        return ck

    @classmethod
    def from_fold(cls, mc, fold_dir: str, upto: int = 5, device="cpu") -> "URDP":
        model = cls(mc).to(device)
        for s in range(2, upto + 1):
            model.load_stage(s, fold_dir, map_location=device)
        return model.eval()
