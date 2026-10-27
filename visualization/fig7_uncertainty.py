# -*- coding: utf-8 -*-
"""
论文 Fig.7 + Table IV：不确定度分析（测试折，事后分析，不用于选型）
  (a) 候选数 K = 1..8 时的 AUROC，以及病例级不确定度与生成误差的 Spearman 相关（五折 mean ± std）
  (b) 代表性折、K=6 时病例级不确定度与生成误差的散点及线性趋势
  Table IV：各 K 下每例模型推理时间（仅特征之后的生成 + 融合部分，不含 WSI 预处理与特征提取）

  病例级不确定度 = K 个候选逐维方差 ν 的平均
  生成误差       = L_res = L_align(r̄, r; λb)，r = z_post - z_anchor

  python visualization/fig7_uncertainty.py --log_dir runs/urdp --fold_b 1
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import fold_ids, load_run_config, savefig, setup_style
from urdp.losses import align_loss
from urdp.models.framework import URDP
from urdp.utils import make_generator, save_json


@torch.no_grad()
def run_fold(cfg, fold: int, ks, device, bs: int = 64):
    fd = os.path.join(cfg.train.log_dir, f"fold_{fold}")
    f = torch.load(os.path.join(cfg.train.features_dir or cfg.train.log_dir, f"fold_{fold}", "features.pt"),
                   map_location=device)
    idx = f["test_idx"]
    z = {k: f[k][idx] for k in ("z_ct", "z_fs", "z_tab", "z_post", "y")}
    model = URDP.from_fold(cfg.model, fd, upto=5, device=device, tag=cfg.train.run_tag)
    out = {}
    for K in ks:
        gen = make_generator(device, 0)
        probs, unc, err = [], [], []
        for i in range(0, len(idx), bs):
            b = {k: v[i:i + bs] for k, v in z.items()}
            o = model.predict(b["z_ct"], b["z_fs"], b["z_tab"], K=K, generator=gen)
            probs.append(o["probs"].cpu())
            unc.append(o["uncertainty"].float().cpu())
            r = b["z_post"] - o["z_anchor"]
            err.append(align_loss(o["r_bar"], r, cfg.model.lambda_b, cfg.model.eps_num,
                                  cfg.model.sq_norm_reduction, reduction="none").cpu())
        probs, unc, err = torch.cat(probs).numpy(), torch.cat(unc).numpy(), torch.cat(err).numpy()
        y = z["y"].cpu().numpy()
        auroc = roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=np.arange(cfg.model.num_classes))
        rho, pval = spearmanr(unc, err) if K > 1 else (np.nan, np.nan)
        out[K] = {"auroc": float(auroc), "rho": float(rho), "p": float(pval), "unc": unc, "err": err}
    return out


@torch.no_grad()
def time_per_case(cfg, fold: int, ks, device, n_cases: int = 20):
    fd = os.path.join(cfg.train.log_dir, f"fold_{fold}")
    model = URDP.from_fold(cfg.model, fd, upto=5, device=device, tag=cfg.train.run_tag)
    z = [torch.randn(1, cfg.model.d, device=device) for _ in range(3)]
    res = {}
    for K in ks:
        model.predict(*z, K=K)                                # warmup
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_cases):
            model.predict(*z, K=K)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        res[K] = (time.perf_counter() - t0) / n_cases
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="runs/urdp")
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--fold_b", type=int, default=None, help="(b) 使用的代表性折，默认第一个")
    p.add_argument("--k_max", type=int, default=8)
    p.add_argument("--k_b", type=int, default=6)
    p.add_argument("--out", default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    setup_style()
    cfg = load_run_config(args.log_dir, args.set)
    device = cfg.device
    ks = list(range(1, args.k_max + 1))
    folds = fold_ids(args.log_dir, args.folds)
    folds = [f for f in folds if os.path.isfile(os.path.join(args.log_dir, f"fold_{f}", "stage5_cmaa.pth"))]
    if not folds:
        raise FileNotFoundError("没有已完成阶段 5 的折")

    per = {f: run_fold(cfg, f, ks, device) for f in folds}
    au = np.array([[per[f][k]["auroc"] for k in ks] for f in folds])
    rho = np.array([[per[f][k]["rho"] for k in ks] for f in folds])

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"wspace": 0.5})
    ax1.errorbar(ks, au.mean(0), au.std(0), marker="o", color="#2E86AB", capsize=3, label="AUROC")
    ax1.set_xlabel("Number of candidates K")
    ax1.set_ylabel("Macro AUROC", color="#2E86AB")
    ax2 = ax1.twinx()
    ax1.set_xticks(ks)
    ax2.errorbar(ks[1:], np.nanmean(rho, 0)[1:], np.nanstd(rho, 0)[1:], marker="s", color="#E84855",
                 capsize=3, label="Spearman ρ")
    ax2.set_ylabel("Uncertainty–error Spearman ρ", color="#E84855")
    ax2.spines["right"].set_visible(True)
    ax1.set_title("(a) Sensitivity to K")

    fb = args.fold_b if args.fold_b in per else folds[0]
    r = per[fb][args.k_b]
    ax3.scatter(r["unc"], r["err"], s=10, alpha=0.6, color="#2E86AB", linewidths=0)
    coef = np.polyfit(r["unc"], r["err"], 1)
    xs = np.linspace(r["unc"].min(), r["unc"].max(), 50)
    ax3.plot(xs, np.polyval(coef, xs), "--", color="#E84855")
    ax3.set_xlabel("Case-level uncertainty")
    ax3.set_ylabel("Generation error ($\\mathcal{L}_{res}$)")
    pstr = "p < 0.001" if r["p"] < 1e-3 else f"p = {r['p']:.3f}"
    ax3.set_title(f"(b) Fold {fb}, K = {args.k_b}: ρ = {r['rho']:.3f}, {pstr}")
    out = args.out or os.path.join(args.log_dir, "figures")
    savefig(fig, out, "fig7_uncertainty")

    timing = time_per_case(cfg, folds[0], ks, device)
    summary = {"K": ks, "auroc_mean": au.mean(0).tolist(), "auroc_std": au.std(0).tolist(),
               "rho_mean": np.nanmean(rho, 0).tolist(), "rho_std": np.nanstd(rho, 0).tolist(),
               "fold_b": fb, "rho_b": r["rho"], "p_b": r["p"],
               "model_time_s_per_case": [timing[k] for k in ks]}
    save_json(summary, os.path.join(out, "fig7_uncertainty.json"))
    print("\n K | AUROC          | Spearman ρ     | model time (s/case)")
    for i, k in enumerate(ks):
        print(f"{k:2d} | {au.mean(0)[i]:.4f}±{au.std(0)[i]:.4f} | "
              f"{np.nanmean(rho, 0)[i]:.4f}±{np.nanstd(rho, 0)[i]:.4f} | {timing[k]:.3f}")


if __name__ == "__main__":
    main()
