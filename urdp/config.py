# -*- coding: utf-8 -*-
"""U-RDP 配置：数据（路径/列名）、模型（论文超参数）、训练。可由 YAML 与命令行覆盖。"""
from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional, Sequence

import torch
import yaml


@dataclass
class DataConfig:
    excel: str = ""                       # 患者表（每行一位患者）
    fs_dir: str = ""                      # 术中冰冻切片 TITAN 特征目录：<fs_dir>/<冰冻切片号>.h5
    post_dir: str = ""                    # 术后 WSI TITAN 特征根目录：<post_dir>/<术后病理前缀>/*.h5
    ct_col: str = "__npy_path__"          # 病灶中心 CT 体积 .npy 路径列，形状 (H, W, D)
    fs_id_col: str = "冰冻切片号"
    post_prefix_col: str = "术后病理前缀"
    label_col: str = "术后病理诊断类型"
    class_names: List[str] = field(default_factory=lambda: ["AAH/AIS", "MIA", "AC"])
    # 表格：4 个连续变量 + 15 个二元 CT 描述符 + 3 个类别变量
    num_cols: List[str] = field(default_factory=lambda: ["年龄", "长径cm", "短径cm", "实性成分长径"])
    binary_cols: List[str] = field(default_factory=lambda: [
        "圆形、类圆形", "不规则", "分叶", "结节征", "空泡", "空洞", "钙化", "支气管征",
        "清楚", "光滑锐利", "毛刺", "尖角、桃尖", "索条", "血管集束征", "胸膜凹陷",
    ])
    density_col: Optional[str] = "密度"
    pleura_col: Optional[str] = "与胸膜的关系"
    gender_col: Optional[str] = "性别"
    ct_roi: List[int] = field(default_factory=lambda: [33, 51, 51])   # (T, H, W)


@dataclass
class ModelConfig:
    d: int = 768                    # 共享嵌入维度
    titan_dim: int = 768
    num_classes: int = 3
    # 表征预训练临时分类头
    temp_head_hidden: int = 512
    temp_head_dropout: float = 0.5
    # TabTransformer
    tab_emb_dim: int = 32
    tab_heads: int = 4
    tab_layers: int = 4
    tab_hidden: int = 256
    tab_dropout: float = 0.1
    # PAM (Eq.1)，d_a 论文未给出
    pam_hidden: int = 256
    # Anchor (Eq.3)
    anchor_dropout: float = 0.2
    # L_align (Eq.2) 权重
    lambda_a: float = 0.3
    lambda_b: float = 0.4
    lambda_c: float = 0.2
    lambda_d: float = 0.3
    eps_num: float = 1e-8
    sq_norm_reduction: str = "mean"   # "mean"：‖·‖² 按维度平均；"sum"：字面平方 L2 范数
    # 残差扩散 (Eq.4-6)
    timesteps: int = 100
    sample_steps: int = 64
    # CPD
    cpd_layers: int = 8
    cpd_hidden: int = 1024
    cpd_heads: int = 16
    cpd_ffn: int = 4096
    cpd_dropout: float = 0.2
    cpd_time_dim: int = 256
    # Multi-sampling + GU-Gate
    K: int = 6
    # CMAA (Eq.10-12)
    cmaa_layers: int = 4
    cmaa_heads: int = 8
    cmaa_score_hidden: int = 256
    cls_hidden: int = 256
    cls_dropout: float = 0.2
    focal_gamma: float = 2.0


@dataclass
class TrainConfig:
    log_dir: str = "runs/urdp"
    n_folds: int = 5
    seed: int = 42
    device: str = "auto"
    num_workers: int = 8
    lr: float = 1e-5
    weight_decay: float = 5e-5
    epochs: int = 60
    warmup_epochs: int = 5          # 论文仅写 "initial warmup"
    batch_size_pretrain: int = 8    # 阶段一（原始影像）
    batch_size: int = 32            # 阶段二至五（缓存特征）
    grad_clip: float = 1.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def device(self) -> str:
        if self.train.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.train.device

    def to_dict(self):
        return asdict(self)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)


def _update(section, values: dict, name: str):
    valid = {f.name for f in fields(section)}
    for k, v in (values or {}).items():
        if k not in valid:
            raise KeyError(f"未知配置项 {name}.{k}")
        setattr(section, k, v)


def load_config(path: Optional[str] = None, overrides: Sequence[str] = ()) -> Config:
    """
    path      : YAML 文件，顶层为 data / model / train
    overrides : 形如 "train.epochs=2" 的字符串，值按 YAML 解析
    """
    cfg = Config()
    if path:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        for sec, vals in raw.items():
            if not hasattr(cfg, sec):
                raise KeyError(f"未知配置段 {sec}")
            _update(getattr(cfg, sec), vals, sec)
    for item in overrides:
        key, _, val = item.partition("=")
        sec, _, name = key.strip().partition(".")
        if not hasattr(cfg, sec) or not name:
            raise KeyError(f"覆盖项格式应为 section.key=value，得到 {item}")
        _update(getattr(cfg, sec), {name: yaml.safe_load(val)}, sec)
    return cfg
