# U-RDP

Uncertainty-Aware Residual Diffusion for Intraoperative Invasiveness Prediction of Lung Adenocarcinoma Spectrum Nodules.

根据术前 CT、术中冰冻切片（FS）和结构化表格数据，预测肺腺癌谱系结节的三分类浸润程度（AAH/AIS、MIA、IAC）。术后病理 WSI 只在训练期作为 privileged information 使用，推理时不需要。

```text
CT ── 3D ResNet-34 ─────────> z_ct  ┐
FS ── TITAN + φ_fs ─────────> z_fs  ├─> c = [z_ct, z_fs, z_tab]
Tabular ── TabTransformer ──> z_tab ┘          │
z_fs ── Anchor Predictor ──> z_anchor          │
                CPD（残差扩散）× K 次采样 ──> 均值 / 方差 ──> GU-Gate ──> r̃
                       z̃_post = z_anchor + r̃
      [z_ct, z_fs, z_tab, z̃_post] ──> CMAA ──> 分类头
```

## 目录

```text
configs/default.yaml     数据路径、列名、模型与训练超参数
train.py                 训练（阶段 1-5，五折）
evaluate.py              有标签队列评估（外部验证）
predict.py               新患者推理，输出 CSV
urdp/
  config.py              配置 dataclass 与 YAML / 命令行覆盖
  data.py                数据集、3/1/1 五折划分、CT 训练集统计量
  models/
    encoders.py          3D ResNet-34、TabTransformer、PAM、术中/术后编码器
    generation.py        Anchor Predictor、CPD、残差扩散 + Analytic-DPM、GU-Gate
    cmaa.py              Cross-Modal Attention Aggregation
    framework.py         U-RDP 生成 + 融合整体模型
  trainer.py             分阶段训练
  inference.py           fold 模型加载与五模型集成推理
  losses.py              L_align、Focal loss
  metrics.py             Accuracy / 宏平均 Precision、Recall、F1、AUROC
```

## 安装

```bash
pip install -r requirements.txt
```

## 数据准备

患者表（Excel）每行一位患者，列名在 `configs/default.yaml` 的 `data` 段配置：

| 内容 | 默认列 | 说明 |
|---|---|---|
| CT | `__npy_path__` | 病灶中心 CT 体积 `.npy`，形状 `(H, W, D)`，1.25 mm 轴向间距，中心裁剪为 `33×51×51` |
| 冰冻切片 | `冰冻切片号` | `<fs_dir>/<冰冻切片号>.h5`，键 `features` 为 768 维 TITAN slide embedding |
| 术后病理 | `术后病理前缀` | `<post_dir>/<前缀>/*.h5`，每张 WSI 一个 TITAN embedding（仅训练需要） |
| 标签 | `术后病理诊断类型` | `AAH/AIS`、`MIA`、`AC` |
| 表格 | `num_cols` / `binary_cols` / 密度 / 胸膜关系 / 性别 | 4 个连续变量、15 个二元 CT 描述符、3 个类别变量；连续变量缺失填 `-1` 或留空 |

WSI 特征使用 TRIDENT（HEST 组织分割，20× 下 512×512 patch，CONCH v1.5）+ 冻结的 TITAN 提取。

## 训练

```bash
python train.py --config configs/default.yaml
```

每折依次执行以下阶段，后一阶段冻结前面所有模块，结果写入 `<log_dir>/fold_k/`：

| 阶段 | 训练模块 | 损失 | 选模型依据 | 输出 |
|---|---|---|---|---|
| 1 | 术中编码器 + 临时头；术后 φ_post + PAM + 临时头 | Focal | 验证 acc | `stage1_intra.pth`、`stage1_post.pth`、`ct_stats.json`、`features.pt` |
| 2 | Anchor Predictor | L_anchor | 验证损失 | `stage2_anchor.pth` |
| 3 | CPD；结束后估计 Analytic-DPM Γ | L_CPD | 验证损失 | `stage3_cpd.pth` |
| 4 | GU-Gate（K=6 次采样） | L_GU-Gate | 验证损失 | `stage4_gu_gate.pth` |
| 5 | CMAA + 分类头；在测试折上评估 | Focal | 验证 acc | `stage5_cmaa.pth`、`test_metrics.json` |

划分为患者级五折：每折 3 份训练、1 份验证、1 份测试，保存在 `<log_dir>/splits.json` 供所有阶段复用。阶段 5 结束后输出五折测试指标的均值±标准差和 pooled OOF 混淆矩阵（`internal_summary.json`）。

常用参数：

```bash
python train.py --stages 2 3 --folds 1                         # 只跑第 1 折的阶段 2、3
python train.py --set train.epochs=30 train.batch_size=16      # 覆盖任意配置项
```

## 评估与推理

```bash
# 外部队列：五个 fold 模型各自的指标 mean ± std，以及概率平均集成的混淆矩阵
python evaluate.py --excel external.xlsx --fs_dir /path/to/external_fs

# 新患者：不需要标签和术后病理
python predict.py --excel new_patients.xlsx --out predictions.csv
```

`predictions.csv` 包含各类别概率、预测类别和 `uncertainty`（K 个残差候选逐维方差的均值，越大表示生成越不稳定）。

Python 调用：

```python
from urdp import URDPDataset, load_config, predict_cohort

cfg = load_config("configs/default.yaml")
ds = URDPDataset(cfg.data, excel="new_patients.xlsx", with_labels=False)
res = predict_cohort(cfg, ds)          # res["probs"]: [N, 3]，res["uncertainty"]: [N]
```

## 实现说明

论文未给出、由本实现设定的细节：

- PAM 注意力隐层 `d_a = 256`；warmup 5 个 epoch，之后学习率不变；batch size 阶段一 8、其余 32。
- Eq.2 中的 `‖u−v‖²` 默认按维度取平均，使其与余弦项量级相当；设 `model.sq_norm_reduction=sum` 可改为字面平方 L2 范数。
- GU-Gate 参数 `a_g、b_g` 初始化为 0（初始门控约 0.5）。
- 3D ResNet-34 随机初始化，使用 GroupNorm 适应小 batch。
- CPD 的 decoder 序列为 `[time token, r_t token]`（因果自注意力），条件 `c` 作为 cross-attention memory。
- 验证、测试和推理的采样使用固定随机种子，结果可复现。
