# -*- coding: utf-8 -*-
"""
推理：只需术前 CT、术中 FS、表格。每个 fold 模型使用该折训练集的 CT 统计量与冻结编码器，
五个 fold 模型的类别概率取平均得到患者级集成预测（论文 Fig.5(b)）。
"""
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from urdp.config import Config
from urdp.data import URDPDataset, collate
from urdp.models.framework import URDP
from urdp.trainer import build_intra_encoder
from urdp.utils import freeze, load_json, make_generator


class FoldPredictor:
    """单个 fold-specific 模型：冻结的 IntraEncoder + Anchor/CPD/GU-Gate/CMAA。"""

    def __init__(self, cfg: Config, fold_dir: str, device: str):
        self.device = device
        self.ct_stats = load_json(os.path.join(fold_dir, "ct_stats.json"))
        enc = build_intra_encoder(cfg)
        enc.load_state_dict(torch.load(os.path.join(fold_dir, "stage1_intra.pth"), map_location="cpu")["encoder"])
        self.encoder = freeze(enc.to(device))
        self.model = URDP.from_fold(cfg.model, fold_dir, upto=5, device=device)

    @torch.no_grad()
    def predict(self, ds: URDPDataset, batch_size: int = 8, num_workers: int = 0, seed: int = 0) -> Dict:
        ds.set_ct_stats(self.ct_stats["mean"], self.ct_stats["std"])
        ds.load_ct, ds.load_post = True, False
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
        gen = make_generator(self.device, seed)
        probs, unc, idx = [], [], []
        for batch in tqdm(loader, leave=False):
            z = self.encoder(batch, self.device)
            out = self.model.predict(z["z_ct"], z["z_fs"], z["z_tab"], generator=gen)
            probs.append(out["probs"].cpu())
            unc.append(out["uncertainty"].float().cpu())
            idx.append(batch["index"])
        return {"probs": torch.cat(probs).numpy(), "uncertainty": torch.cat(unc).numpy(),
                "index": torch.cat(idx).numpy()}


def predict_cohort(cfg: Config, ds: URDPDataset, folds: Optional[List[int]] = None, batch_size: int = 8) -> Dict:
    """返回每个 fold 模型的预测，以及五模型概率平均的集成结果。"""
    folds = folds or list(range(1, cfg.train.n_folds + 1))
    per_fold = {}
    for fold in folds:
        fd = os.path.join(cfg.train.log_dir, f"fold_{fold}")
        print(f"[predict] fold {fold}: {fd}")
        per_fold[fold] = FoldPredictor(cfg, fd, cfg.device).predict(ds, batch_size, cfg.train.num_workers)
    probs = np.stack([r["probs"] for r in per_fold.values()])
    unc = np.stack([r["uncertainty"] for r in per_fold.values()])
    return {"per_fold": per_fold, "probs": probs.mean(0), "uncertainty": unc.mean(0),
            "index": next(iter(per_fold.values()))["index"]}
