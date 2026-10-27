# -*- coding: utf-8 -*-
"""
U-RDP 数据：患者表 -> CT ROI / FS TITAN 特征 / 表格特征 / 术后 WSI TITAN 特征列表。
包括 3/1/1 患者级五折划分与当前折训练集 CT 统计量。
"""
import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from urdp.config import DataConfig


def _read_h5_features(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(f["features"][()], dtype=np.float32).reshape(-1)


def cat_cardinalities(dc: DataConfig) -> List[int]:
    """与 URDPDataset.x_cat 的列顺序一致（最后一个取值表示缺失）。"""
    card = [3] * len(dc.binary_cols)
    if dc.density_col:
        card.append(4)
    if dc.pleura_col:
        card.append(6)
    if dc.gender_col:
        card.append(3)
    return card


class URDPDataset(Dataset):
    """
    require_post=True（训练队列）时只保留术后 WSI 特征存在的患者，保证阶段 1-5 使用同一批患者、同一份划分；
    推理/外部评估不需要术后病理。各阶段读取的模态不同，用 view() 取得只加载所需模态的视图。

    每个样本：
        CT        [1, T, H, W]  病灶中心 ROI，用 set_ct_stats 设定的训练集统计量标准化
        fs        [titan_dim]   术中冰冻切片 TITAN slide embedding
        x_num     [n_num]       连续变量（缺失置 0）
        num_miss  [n_num]       连续变量缺失标记
        x_cat     [n_cat]       二元/类别变量
        post      List[[titan_dim]]  术后 WSI（load_post=True 时）
        y         int           类别索引；无标签时为 -1
        index     int           行号
    """

    def __init__(self, dc: DataConfig, excel: Optional[str] = None, fs_dir: Optional[str] = None,
                 post_dir: Optional[str] = None, with_labels: bool = True,
                 require_post: bool = False, verbose: bool = True):
        self.dc = dc
        self.fs_dir = fs_dir or dc.fs_dir
        self.post_dir = post_dir or dc.post_dir
        self.with_labels = with_labels
        self.roi = tuple(dc.ct_roi)
        self.ct_stats: Optional[Tuple[float, float]] = None
        self.class_names = list(dc.class_names)

        df = pd.read_excel(excel or dc.excel)
        need = [dc.ct_col, dc.fs_id_col] + ([dc.label_col] if with_labels else [])
        need += [c for c in dc.num_cols + dc.binary_cols + [dc.density_col, dc.pleura_col, dc.gender_col] if c]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"患者表缺少列：{missing}")

        df[dc.fs_id_col] = df[dc.fs_id_col].astype(str).str.strip()
        df["__ct__"] = df[dc.ct_col].astype(str).str.strip()
        df["__fs__"] = df[dc.fs_id_col].map(lambda s: os.path.join(self.fs_dir, f"{s}.h5"))
        keep = df["__ct__"].map(os.path.isfile) & df["__fs__"].map(os.path.isfile)
        n_ct, n_fs = int((~df["__ct__"].map(os.path.isfile)).sum()), int((~df["__fs__"].map(os.path.isfile)).sum())
        n_label = n_post = 0
        if with_labels:
            df[dc.label_col] = df[dc.label_col].astype(str).str.strip()
            ok = df[dc.label_col].isin(self.class_names)
            n_label = int((~ok).sum())
            keep &= ok
        if require_post:
            if dc.post_prefix_col not in df.columns:
                raise ValueError(f"患者表缺少列：{dc.post_prefix_col}")
            ok = df[dc.post_prefix_col].map(lambda p: bool(glob.glob(
                os.path.join(self.post_dir, str(p).strip(), "*.h5"))))
            n_post = int((~ok).sum())
            keep &= ok
        if verbose:
            print(f"[URDPDataset] {len(df)} 行 -> 保留 {int(keep.sum())} 行 | 缺 CT {n_ct}，缺 FS {n_fs}，"
                  f"非法标签 {n_label}" + (f"，缺术后 WSI {n_post}" if require_post else ""))
        self.df = df[keep].reset_index(drop=True)
        self._encode_table()

        if with_labels:
            mp = {n: i for i, n in enumerate(self.class_names)}
            self.y = torch.tensor(self.df[dc.label_col].map(mp).values, dtype=torch.long)
        else:
            self.y = torch.full((len(self.df),), -1, dtype=torch.long)

    # ------------------------------------------------------------------
    def _encode_table(self):
        dc, df = self.dc, self.df
        num = df[dc.num_cols].apply(pd.to_numeric, errors="coerce") if dc.num_cols else pd.DataFrame(index=df.index)
        miss = num.isna() | (num == -1)
        self.x_num = torch.tensor(num.mask(miss, 0.0).values, dtype=torch.float32)
        self.num_miss = torch.tensor(miss.values, dtype=torch.bool)

        cols = []
        for c in dc.binary_cols:
            v = pd.to_numeric(df[c], errors="coerce")
            cols.append(v.where(v.isin([0, 1]), 2).astype(int).values)
        if dc.density_col:
            mp = {"磨玻璃": 0, "部分实性": 1, "实性": 2}

            def density(v):
                s = str(v).strip()
                if s in mp:
                    return mp[s]
                try:
                    i = int(float(s))
                    return i if i in (0, 1, 2) else 3
                except ValueError:
                    return 3
            cols.append(df[dc.density_col].map(density).astype(int).values)
        if dc.pleura_col:
            v = pd.to_numeric(df[dc.pleura_col], errors="coerce")
            cols.append(v.fillna(-1).clip(-1, 4).astype(int).replace(-1, 5).values)
        if dc.gender_col:
            def gender(v):
                s = str(v).strip().lower()
                if s in ("男", "male", "m", "1", "1.0"):
                    return 1
                if s in ("女", "female", "f", "0", "0.0"):
                    return 0
                return 2
            cols.append(df[dc.gender_col].map(gender).astype(int).values)
        self.x_cat = (torch.tensor(np.stack(cols, 1), dtype=torch.long) if cols
                      else torch.zeros((len(df), 0), dtype=torch.long))

    # ------------------------------------------------------------------
    def set_ct_stats(self, mean: float, std: float):
        """当前折训练集 CT 统计量，原样用于验证/测试/外部数据。"""
        self.ct_stats = (float(mean), float(std))

    def load_ct_raw(self, idx: int) -> np.ndarray:
        """裁剪后、未标准化的 CT (T, H, W)。"""
        vol = np.load(self.df.at[idx, "__ct__"]).astype(np.float32)
        if self.dc.ct_layout == "HWT":           # (H, W, T) -> (T, H, W)
            vol = np.moveaxis(vol, -1, 0)
        sl = []
        for n, r in zip(vol.shape, self.roi):
            if n < r:
                raise ValueError(f"CT 尺寸 {vol.shape} 小于 ROI {self.roi}")
            sl.append(slice((n - r) // 2, (n - r) // 2 + r))
        return vol[tuple(sl)]

    def _post_list(self, idx: int) -> List[torch.Tensor]:
        prefix = str(self.df.at[idx, self.dc.post_prefix_col]).strip()
        files = sorted(glob.glob(os.path.join(self.post_dir, prefix, "*.h5")))
        if not files:
            raise FileNotFoundError(f"未找到术后 WSI 特征：{os.path.join(self.post_dir, prefix)}")
        return [torch.from_numpy(_read_h5_features(p)) for p in files]

    def __len__(self):
        return len(self.df)

    def view(self, load_ct: bool = True, load_post: bool = False) -> "DatasetView":
        return DatasetView(self, load_ct, load_post)

    def __getitem__(self, idx: int) -> Dict:
        return self.get(idx, load_ct=True, load_post=False)

    def get(self, idx: int, load_ct: bool, load_post: bool) -> Dict:
        item = {
            "fs": torch.from_numpy(_read_h5_features(self.df.at[idx, "__fs__"])),
            "x_num": self.x_num[idx],
            "num_miss": self.num_miss[idx],
            "x_cat": self.x_cat[idx],
            "y": int(self.y[idx]),
            "index": idx,
        }
        if load_ct:
            if self.ct_stats is None:
                raise RuntimeError("请先调用 set_ct_stats(mean, std)")
            m, s = self.ct_stats
            ct = (self.load_ct_raw(idx) - m) / (s + 1e-6)
            item["CT"] = torch.from_numpy(np.ascontiguousarray(ct))[None]
        if load_post:
            item["post"] = self._post_list(idx)
        return item


class DatasetView(Dataset):
    """
    同一患者集合上只加载指定模态的视图：
        阶段 1 术中分支 view(load_ct=True, load_post=False)
        阶段 1 术后分支 view(load_ct=False, load_post=True)
        特征缓存       view(load_ct=True, load_post=True)
        推理           view(load_ct=True, load_post=False)
    """

    def __init__(self, ds: URDPDataset, load_ct: bool, load_post: bool, indices: Optional[Sequence[int]] = None):
        self.ds, self.load_ct, self.load_post = ds, load_ct, load_post
        self.indices = list(range(len(ds))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict:
        return self.ds.get(self.indices[i], self.load_ct, self.load_post)


def collate(batch: List[Dict]) -> Dict:
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if k == "post":
            out[k] = vals
        elif k in ("y", "index"):
            out[k] = torch.tensor(vals, dtype=torch.long)
        else:
            out[k] = torch.stack(vals, 0)
    return out


# ----------------------------------------------------------------------
# 划分与统计量
# ----------------------------------------------------------------------
def kfold_train_val_test(labels: Sequence[int], n_splits: int = 5, seed: int = 42):
    """
    患者级分层划分为 n_splits 个互不相交分区 P_0..P_{K-1}；
    第 k 折：test = P_k，val = P_{(k+1)%K}，train = 其余（3 训练 / 1 验证 / 1 测试）。
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(n_splits)]
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        for k, chunk in enumerate(np.array_split(idx, n_splits)):
            parts[k].extend(int(i) for i in chunk)
    folds = []
    for k in range(n_splits):
        v = (k + 1) % n_splits
        train = sorted(i for j in range(n_splits) if j not in (k, v) for i in parts[j])
        folds.append({"train": train, "val": sorted(parts[v]), "test": sorted(parts[k])})
    return folds


def load_or_create_splits(ds: URDPDataset, n_splits: int, seed: int, path: str):
    """固定划分写入 splits.json，所有阶段复用。"""
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if saved["n_samples"] != len(ds):
            raise ValueError(f"{path} 记录 {saved['n_samples']} 例，当前数据 {len(ds)} 例，请换 log_dir")
        return saved["folds"]
    folds = kfold_train_val_test(ds.y.tolist(), n_splits, seed)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n_samples": len(ds), "seed": seed, "folds": folds}, f)
    for k, fd in enumerate(folds, 1):
        cnt = lambda ids: np.bincount(ds.y[ids].numpy(), minlength=len(ds.class_names)).tolist()
        print(f"[split] fold {k}: train {len(fd['train'])} {cnt(fd['train'])} | "
              f"val {len(fd['val'])} {cnt(fd['val'])} | test {len(fd['test'])} {cnt(fd['test'])}")
    return folds


def compute_ct_stats(ds: URDPDataset, idx: Sequence[int]) -> Tuple[float, float]:
    """体素级均值/标准差（ROI 裁剪后），只用训练分区。"""
    s = s2 = n = 0.0
    for i in idx:
        v = ds.load_ct_raw(i).astype(np.float64)
        s += v.sum()
        s2 += np.square(v).sum()
        n += v.size
    mean = s / n
    return float(mean), float(np.sqrt(max(s2 / n - mean ** 2, 1e-12)))
