# -*- coding: utf-8 -*-
"""
对新患者推理（无需标签与术后病理），输出 CSV：五个 fold 模型集成后的类别概率、预测类别、生成不确定度。

  python predict.py --config configs/default.yaml --excel new_patients.xlsx --out predictions.csv
"""
import argparse

import pandas as pd

from urdp import URDPDataset, load_config, predict_cohort


def main():
    p = argparse.ArgumentParser(description="Predict with U-RDP")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--excel", required=True)
    p.add_argument("--fs_dir", default=None)
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)

    ds = URDPDataset(cfg.data, excel=args.excel, fs_dir=args.fs_dir, with_labels=False)
    res = predict_cohort(cfg, ds, args.folds)
    names = cfg.data.class_names
    df = pd.DataFrame({cfg.data.fs_id_col: ds.df[cfg.data.fs_id_col].values[res["index"]]})
    for i, n in enumerate(names):
        df[f"prob_{n}"] = res["probs"][:, i]
    df["pred"] = [names[i] for i in res["probs"].argmax(1)]
    df["uncertainty"] = res["uncertainty"]
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[predict] {len(df)} patients -> {args.out}")


if __name__ == "__main__":
    main()
