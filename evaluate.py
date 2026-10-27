# -*- coding: utf-8 -*-
"""
外部队列评估（论文 Table I 外部部分 / Fig.5(b)）：
五个 fold-specific 模型分别评估 -> mean ± std；概率平均集成 -> 混淆矩阵。

  python evaluate.py --config configs/default.yaml --excel external.xlsx --fs_dir /path/to/external_fs
"""
import argparse
import os

from urdp import URDPDataset, load_config, predict_cohort
from urdp.metrics import classification_metrics, fmt, format_cm, mean_std, print_mean_std
from urdp.utils import save_json


def main():
    p = argparse.ArgumentParser(description="Evaluate U-RDP on a labeled cohort")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--excel", required=True)
    p.add_argument("--fs_dir", default=None, help="默认使用配置中的 data.fs_dir")
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--out", default=None, help="默认 <log_dir>/external_summary.json")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)

    ds = URDPDataset(cfg.data, excel=args.excel, fs_dir=args.fs_dir, with_labels=True)
    res = predict_cohort(cfg, ds, args.folds)
    y, n_cls = ds.y.numpy()[res["index"]], cfg.model.num_classes

    per_fold = []
    for fold, r in res["per_fold"].items():
        m = classification_metrics(y, r["probs"], n_cls)
        print(f"[fold {fold}] {fmt(m)}")
        per_fold.append({"fold": fold, **m})
    summary = {"per_fold": per_fold, **mean_std(per_fold), "ensemble": classification_metrics(y, res["probs"], n_cls)}
    save_json(summary, args.out or os.path.join(cfg.train.log_dir, "external_summary.json"))
    print_mean_std("External cohort (fold-specific models, mean ± std)", summary)
    print(f"Ensemble: {fmt(summary['ensemble'])}")
    print(format_cm(summary["ensemble"]["cm"], cfg.data.class_names))


if __name__ == "__main__":
    main()
