# -*- coding: utf-8 -*-
"""
U-RDP 训练入口

  python train.py --config configs/default.yaml                     # 全部 5 折、阶段 1-5
  python train.py --config configs/default.yaml --stages 2 3 --folds 1
  python train.py --config configs/default.yaml --set train.epochs=2 train.batch_size=16
"""
import argparse

from urdp import Trainer, load_config


def main():
    p = argparse.ArgumentParser(description="Train U-RDP")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4, 5], choices=[1, 2, 3, 4, 5])
    p.add_argument("--folds", type=int, nargs="+", default=None, help="1-based，默认全部")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="覆盖配置，如 train.epochs=2")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    Trainer(cfg).run(args.stages, args.folds or range(1, cfg.train.n_folds + 1))


if __name__ == "__main__":
    main()
