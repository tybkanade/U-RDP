# -*- coding: utf-8 -*-
"""
论文 Fig.4：测试折上真实术后病理嵌入、U-RDP 生成嵌入、FS 嵌入的 UMAP（epoch 1 / 10 / 30）。
每个 epoch 单独联合拟合一次 UMAP（相同超参数与随机种子），不同面板坐标不可直接比较。

前置：训练时开启 train.track_embeddings=true（阶段 3 每个 epoch 保存测试折候选）。
  python visualization/fig4_umap.py --log_dir runs/urdp --fold 1 --epochs 1 10 30
"""
import argparse
import os

import numpy as np
import torch

from common import (COLORS, generated_variants, load_gu_gate, load_reference, savefig, setup_style,
                    tracked_epochs)
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="runs/urdp")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--epochs", type=int, nargs="+", default=[1, 10, 30])
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="默认 <log_dir>/figures")
    args = p.parse_args()
    import umap

    setup_style()
    ref = load_reference(args.log_dir, args.fold)
    avail = tracked_epochs(args.log_dir, args.fold, 3)
    if not avail:
        raise FileNotFoundError("未找到阶段 3 逐 epoch 嵌入，请以 --set train.track_embeddings=true 训练阶段 3")
    epochs = [e for e in args.epochs if e in avail] or [avail[0], avail[len(avail) // 2], avail[-1]]
    gate = load_gu_gate(args.log_dir, args.fold)

    fig, axes = plt.subplots(1, len(epochs), figsize=(4.2 * len(epochs), 4))
    axes = np.atleast_1d(axes)
    for ax, ep, tag in zip(axes, epochs, "abcdefgh"):
        var = generated_variants(args.log_dir, args.fold, ep, gate)
        gen = var.get("gugate", var["sampling"])
        groups = [("Real postoperative", ref["z_post"].float(), COLORS["real"]),
                  ("U-RDP generated", gen, COLORS["generated"]),
                  ("Frozen section", ref["z_fs"].float(), COLORS["fs"])]
        X = torch.cat([g[1] for g in groups]).numpy()
        emb = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=args.seed).fit_transform(X)
        start = 0
        for name, z, color in groups:
            sl = emb[start:start + len(z)]
            ax.scatter(sl[:, 0], sl[:, 1], s=8, alpha=0.7, c=color, label=name, linewidths=0)
            start += len(z)
        ax.set_title(f"({tag}) Epoch {ep}")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(loc="best", frameon=False, markerscale=2)
    savefig(fig, args.out or os.path.join(args.log_dir, "figures"), f"fig4_umap_fold{args.fold}")


if __name__ == "__main__":
    main()
