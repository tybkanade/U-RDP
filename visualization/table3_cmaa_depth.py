# -*- coding: utf-8 -*-
"""
论文 Table III：CMAA 深度 L = 1..8 的事后敏感性分析（测试折）。
复用主实验阶段 1-4 的结果，只为每个 L 重新训练阶段 5（输出带 run_tag=L{n} 后缀，不覆盖主实验）。

  python visualization/table3_cmaa_depth.py --log_dir runs/urdp                   # 训练 + 汇总
  python visualization/table3_cmaa_depth.py --log_dir runs/urdp --no_train        # 只汇总已有结果
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from common import fold_ids, load_run_config, savefig, setup_style
from urdp.metrics import KEYS
from urdp.trainer import Trainer
from urdp.utils import load_json, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="runs/urdp")
    p.add_argument("--depths", type=int, nargs="+", default=list(range(1, 9)))
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--no_train", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    setup_style()
    folds = fold_ids(args.log_dir, args.folds)

    rows = {}
    for L in args.depths:
        cfg = load_run_config(args.log_dir, list(args.set) + [f"model.cmaa_layers={L}", f"train.run_tag=L{L}",
                                                               "train.track_embeddings=false"])
        path = os.path.join(args.log_dir, f"internal_summary_L{L}.json")
        if not args.no_train:
            Trainer(cfg).run([5], folds)
        if os.path.isfile(path):
            rows[L] = load_json(path)
        else:
            print(f"[skip] L={L}: {path} 不存在")
    if not rows:
        raise FileNotFoundError("没有可汇总的结果")

    Ls = sorted(rows)
    print("\n  L | " + " | ".join(f"{k:>13}" for k in KEYS))
    for L in Ls:
        m, s = rows[L]["mean"], rows[L]["std"]
        print(f"{L:3d} | " + " | ".join(f"{100 * m[k]:6.2f}±{100 * s[k]:5.2f}" for k in KEYS))
    out = args.out or os.path.join(args.log_dir, "figures")
    os.makedirs(out, exist_ok=True)
    save_json({L: {"mean": rows[L]["mean"], "std": rows[L]["std"]} for L in Ls},
              os.path.join(out, "table3_cmaa_depth.json"))

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for k, c in (("acc", "#2E86AB"), ("auroc", "#E84855")):
        m = np.array([rows[L]["mean"][k] for L in Ls]) * 100
        s = np.array([rows[L]["std"][k] for L in Ls]) * 100
        ax.errorbar(Ls, m, s, marker="o", capsize=3, color=c, label=k.upper() if k == "auroc" else "Accuracy")
    ax.set_xticks(Ls)
    ax.set_xlabel("CMAA layers L")
    ax.set_ylabel("%")
    ax.legend(frameon=False)
    savefig(fig, out, "table3_cmaa_depth")


if __name__ == "__main__":
    main()
