# -*- coding: utf-8 -*-
"""
论文 Fig.6：生成表示与真实术后病理表示（测试折）之间的 MMD 随训练 epoch 的变化，阴影为五折标准差。

曲线：
  Anchor                         z_anchor（阶段 2 各 epoch）
  Full-feature                   直接生成 z_post 的消融（--full_log_dir，可选）
  Residual                       z_anchor + r̂^(1)（阶段 3 各 epoch）
  Residual + Sampling            z_anchor + mean_k r̂^(k)
  Residual + Sampling + GU-Gate  z_anchor + GU-Gate(r̂^(1..K))（使用最终训练的 GU-Gate）
  Real-Real                      真实表示随机二分后两半之间的 MMD（真实分布的自然波动）

前置：
  python train.py --set train.track_embeddings=true                                  # 主实验
  python train.py --stages 3 --set train.track_embeddings=true model.generation_target=full \\
      train.log_dir=runs/urdp_full train.features_dir=runs/urdp                       # Full-feature（可选）
  python visualization/fig6_mmd.py --log_dir runs/urdp --full_log_dir runs/urdp_full
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (COLORS, emb_dir, fold_ids, generated_variants, load_gu_gate, load_reference, mean_std_band,
                    mmd2_rbf, savefig, setup_style, tracked_epochs)
from urdp.utils import save_json


def curve(per_fold: dict):
    """per_fold: {fold: {epoch: value}} -> (epochs, [n_folds, n_epochs])，只取所有折共有的 epoch"""
    if not per_fold:
        return None, None
    common = sorted(set.intersection(*[set(v) for v in per_fold.values()]))
    return np.array(common), np.array([[v[e] for e in common] for v in per_fold.values()])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="runs/urdp")
    p.add_argument("--full_log_dir", default=None, help="generation_target=full 的消融运行目录")
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    setup_style()

    res = {k: {} for k in ("anchor", "full", "residual", "sampling", "gugate")}
    realreal = []
    for fold in fold_ids(args.log_dir, args.folds):
        if not os.path.isdir(emb_dir(args.log_dir, fold)):
            continue
        ref = load_reference(args.log_dir, fold)
        real = ref["z_post"].float()
        perm = torch.randperm(len(real), generator=torch.Generator().manual_seed(fold))
        realreal.append(mmd2_rbf(real[perm[: len(real) // 2]], real[perm[len(real) // 2:]]))

        for ep in tracked_epochs(args.log_dir, fold, 2):
            za = torch.load(os.path.join(emb_dir(args.log_dir, fold), f"stage2_ep{ep:03d}.pt"))["z_anchor"]
            res["anchor"].setdefault(fold, {})[ep] = mmd2_rbf(za.float(), real)
        gate = load_gu_gate(args.log_dir, fold)
        for ep in tracked_epochs(args.log_dir, fold, 3):
            for k, z in generated_variants(args.log_dir, fold, ep, gate).items():
                res[k].setdefault(fold, {})[ep] = mmd2_rbf(z, real)
        if args.full_log_dir and os.path.isdir(emb_dir(args.full_log_dir, fold)):
            for ep in tracked_epochs(args.full_log_dir, fold, 3):
                z = generated_variants(args.full_log_dir, fold, ep)["residual"]   # 基底为 0，即直接生成 z_post
                res["full"].setdefault(fold, {})[ep] = mmd2_rbf(z, real)
    if not realreal:
        raise FileNotFoundError("未找到逐 epoch 嵌入，请以 --set train.track_embeddings=true 训练")

    labels = {"anchor": "Anchor", "full": "Full-feature", "residual": "Residual",
              "sampling": "Residual + Sampling", "gugate": "Residual + Sampling + GU-Gate"}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    dump = {}
    for k, name in labels.items():
        ep, ys = curve(res[k])
        if ep is None or not len(ep):
            continue
        mean_std_band(ax, ep, ys, name, COLORS[k])
        dump[k] = {"epochs": ep.tolist(), "mean": ys.mean(0).tolist(), "std": ys.std(0).tolist()}
    rr = np.array(realreal)
    ax.axhline(rr.mean(), color=COLORS["realreal"], ls="--", lw=1.2, label="Real-Real")
    ax.axhspan(rr.mean() - rr.std(), rr.mean() + rr.std(), color=COLORS["realreal"], alpha=0.1, lw=0)
    dump["realreal"] = {"mean": float(rr.mean()), "std": float(rr.std())}
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MMD$^2$ (generated vs. real)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    out = args.out or os.path.join(args.log_dir, "figures")
    savefig(fig, out, "fig6_mmd")
    save_json(dump, os.path.join(out, "fig6_mmd.json"))


if __name__ == "__main__":
    main()
