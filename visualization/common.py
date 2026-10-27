# -*- coding: utf-8 -*-
"""可视化共用工具：读取训练目录配置、作图样式、MMD。"""
import os
import sys
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from urdp.config import load_config  # noqa: E402

COLORS = {"real": "#2E86AB", "generated": "#E84855", "fs": "#7FB069", "anchor": "#F4A259",
          "full": "#9B5DE5", "residual": "#00BBF9", "sampling": "#00A6A6", "gugate": "#E84855",
          "realreal": "#555555"}


def load_run_config(log_dir: str, overrides=()):
    """优先读取训练时保存的 <log_dir>/config.yaml"""
    path = os.path.join(log_dir, "config.yaml")
    cfg = load_config(path if os.path.isfile(path) else None, overrides)
    cfg.train.log_dir = log_dir
    return cfg


def fold_ids(log_dir: str, folds=None) -> List[int]:
    if folds:
        return list(folds)
    ids = [int(d.split("_")[1]) for d in os.listdir(log_dir) if d.startswith("fold_") and d.split("_")[1].isdigit()]
    return sorted(ids)


def setup_style():
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def savefig(fig, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"[saved] {os.path.join(out_dir, name)}.png / .pdf")


def mmd2_rbf(x, y, max_n: int = 2000, seed: int = 0) -> float:
    """无偏 MMD²，RBF 核，带宽取合并样本两两距离平方的中位数（median heuristic）。"""
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    if x.size(0) > max_n:
        x = x[torch.randperm(x.size(0), generator=g)[:max_n]]
    if y.size(0) > max_n:
        y = y[torch.randperm(y.size(0), generator=g)[:max_n]]
    z = torch.cat([x, y])
    d2 = torch.cdist(z, z).pow(2)
    sigma2 = d2[d2 > 0].median().clamp_min(1e-12)
    k = torch.exp(-d2 / sigma2)
    n, m = x.size(0), y.size(0)
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    mmd = ((kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
           + (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1)) - 2 * kxy.mean())
    return float(mmd)


def mean_std_band(ax, x, ys: np.ndarray, label: str, color: str, **kw):
    """ys: [n_folds, n_x]，画均值曲线与 ±std 阴影"""
    m, s = np.nanmean(ys, 0), np.nanstd(ys, 0)
    ax.plot(x, m, label=label, color=color, lw=1.6, **kw)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, lw=0)


# ----------------------------------------------------------------------
# 训练时逐 epoch 保存的测试折嵌入（train.track_embeddings=true）
# ----------------------------------------------------------------------
def emb_dir(log_dir: str, fold: int) -> str:
    return os.path.join(log_dir, f"fold_{fold}", "embeddings")


def tracked_epochs(log_dir: str, fold: int, stage: int) -> List[int]:
    d = emb_dir(log_dir, fold)
    if not os.path.isdir(d):
        return []
    pre = f"stage{stage}_ep"
    return sorted(int(f[len(pre):-3]) for f in os.listdir(d) if f.startswith(pre) and f.endswith(".pt"))


def load_reference(log_dir: str, fold: int):
    return torch.load(os.path.join(emb_dir(log_dir, fold), "reference.pt"), map_location="cpu")


def load_gu_gate(log_dir: str, fold: int, d: int = 768):
    """最终 GU-Gate（阶段 4）；不存在时返回 None"""
    from urdp.models.generation import GUGate
    path = os.path.join(log_dir, f"fold_{fold}", "stage4_gu_gate.pth")
    if not os.path.isfile(path):
        return None
    gate = GUGate(d)
    gate.load_state_dict(torch.load(path, map_location="cpu")["module"])
    return gate.eval()


@torch.no_grad()
def generated_variants(log_dir: str, fold: int, stage3_epoch: int, gate=None):
    """
    由阶段 3 某 epoch 的候选构造各生成变体（均为 z_post 空间）：
        residual  : z_anchor + r̂^(1)               （单次采样）
        sampling  : z_anchor + mean_k r̂^(k)
        gugate    : z_anchor + GU-Gate(r̂^(1..K))   （使用最终训练的 GU-Gate）
    """
    e = torch.load(os.path.join(emb_dir(log_dir, fold), f"stage3_ep{stage3_epoch:03d}.pt"), map_location="cpu")
    za, c = e["z_anchor"].float(), e["cands"].float()
    out = {"residual": za + c[0], "sampling": za + c.mean(0)}
    if gate is not None:
        out["gugate"] = za + gate(c)["r_tilde"]
    return out
