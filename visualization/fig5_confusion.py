# -*- coding: utf-8 -*-
"""
论文 Fig.5：混淆矩阵
  (a) 内部队列：五个测试折的 pooled out-of-fold 预测（internal_summary.json，train.py 阶段 5 生成）
  (b) 外部队列：五个 fold 模型概率平均后的患者级集成（external_summary.json，evaluate.py 生成）

  python visualization/fig5_confusion.py --log_dir runs/urdp
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from common import load_run_config, savefig, setup_style


def plot_cm(ax, cm, names, title):
    cm = np.asarray(cm)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    acc = np.trace(cm) / cm.sum()
    ax.set_title(f"{title}\nAcc = {100 * acc:.2f}% (n={cm.sum()})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="runs/urdp")
    p.add_argument("--internal", default=None, help="默认 <log_dir>/internal_summary.json")
    p.add_argument("--external", default=None, help="默认 <log_dir>/external_summary.json")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    setup_style()
    names = load_run_config(args.log_dir).data.class_names
    panels = []
    for path, key, title in ((args.internal or os.path.join(args.log_dir, "internal_summary.json"), "pooled_oof",
                              "(a) Internal cohort (pooled OOF)"),
                             (args.external or os.path.join(args.log_dir, "external_summary.json"), "ensemble",
                              "(b) External cohort (ensemble)")):
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                panels.append((json.load(f)[key]["cm"], title))
        else:
            print(f"[skip] {path} 不存在")
    if not panels:
        raise FileNotFoundError("没有可用的 summary 文件")
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.2), layout="constrained")
    for ax, (cm, title) in zip(np.atleast_1d(axes), panels):
        plot_cm(ax, cm, names, title)
    savefig(fig, args.out or os.path.join(args.log_dir, "figures"), "fig5_confusion")


if __name__ == "__main__":
    main()
