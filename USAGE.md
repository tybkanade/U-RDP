# U-RDP 使用说明

## 1. 环境

```bash
conda activate torch                 # 服务器 10.1.115.106 上已有环境，依赖齐全
pip install -r requirements.txt      # 新环境时安装
```

选空闲 GPU：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=1        # 改成空闲的卡号
```

## 2. 数据准备

患者表（Excel）每行一位患者。列名可在配置 `data` 段修改。

| 内容 | 默认列名 | 文件要求 |
|---|---|---|
| CT | `__npy_path__` | `.npy`，轴顺序 `(层, 高, 宽)`，1.25 mm 轴向间距；以体积中心裁剪为 `33×51×51`（层数更多时取中间 33 层） |
| 冰冻切片 | `冰冻切片号` | `<fs_dir>/<冰冻切片号>.h5`，键 `features` 为 768 维 TITAN slide embedding |
| 术后 WSI | `术后病理前缀` | `<post_dir>/<前缀>/*.h5`，每张 WSI 一个 768 维 TITAN embedding；**只在训练时需要** |
| 标签 | `术后病理诊断类型` | `AAH/AIS`、`MIA`、`AC` |
| 连续变量 | `年龄, 长径cm, 短径cm, 实性成分长径` | `-1` 或空表示缺失 |
| 二元 CT 描述符 | 15 列（圆形、类圆形 … 胸膜凹陷） | `0/1`，其他值视为缺失 |
| 类别变量 | `密度`（磨玻璃/部分实性/实性）、`与胸膜的关系`（0–4，-1 缺失）、`性别`（男/女） | |

训练时只保留 CT、FS、术后 WSI 文件都存在且标签合法的患者，日志会打印每类缺失的数量。阶段 1–5 共用同一批患者和同一份划分（`<log_dir>/splits.json`）。

## 3. 配置

`configs/default.yaml` 是完整的默认配置；`configs/server.yaml` 只写了服务器上的数据路径，其余用默认值。任何配置项都可以在命令行用 `--set 段.键=值` 覆盖。

常用项：

| 配置项 | 默认 | 说明 |
|---|---|---|
| `data.excel` / `data.fs_dir` / `data.post_dir` | — | 患者表、FS 特征目录、术后 WSI 特征目录 |
| `data.ct_layout` | `THW` | `.npy` 轴顺序；旧数据若是 `(高,宽,层)` 改为 `HWT` |
| `data.ct_roi` | `[33, 51, 51]` | CT 中心裁剪大小 |
| `train.log_dir` | `runs/urdp` | 所有输出目录 |
| `train.epochs` | `60` | 每个阶段的最大 epoch |
| `train.lr` / `train.weight_decay` | `1e-5` / `5e-5` | AdamW |
| `train.warmup_epochs` | `5` | 线性 warmup 后学习率不变 |
| `train.batch_size_pretrain` | `8` | 阶段 1（读原始 CT） |
| `train.batch_size` | `32` | 阶段 2–5（读缓存特征） |
| `train.num_workers` | `8` | DataLoader 进程数 |
| `train.seed` | `42` | 划分与训练随机种子 |
| `train.device` | `auto` | `auto` / `cuda` / `cpu` |
| `train.track_embeddings` | `false` | 阶段 2/3 每个 epoch 保存测试折生成嵌入（画 Fig.4、Fig.6 需要） |
| `train.features_dir` | 空 | 从另一个运行目录读取阶段 1 特征（消融复用） |
| `train.run_tag` | 空 | 阶段 5 输出文件后缀（CMAA 深度实验用） |
| `model.K` | `6` | 残差候选数 |
| `model.sample_steps` | `64` | Analytic-DPM 逆扩散步数 |
| `model.timesteps` | `100` | 前向扩散步数 |
| `model.cmaa_layers` | `4` | CMAA 层数 L |
| `model.lambda_a/b/c/d` | `0.3/0.4/0.2/0.3` | 各对齐损失权重 |
| `model.sq_norm_reduction` | `mean` | `L_align` 中平方距离按维度平均；`sum` 为字面平方 L2 |
| `model.generation_target` | `residual` | `full` 为直接生成 z_post 的消融（不用 Anchor） |
| `train.keep_topk` | `2` | 每个模块只保留验证指标最好的 2 个 checkpoint：第 1 名为原文件名，第 2 名为 `*_top2.pth` |
| `train.amp` | `bf16` | 混合精度（`bf16`/`fp16`/`none`），只影响速度 |

从旧版生成器沿用的可选项，默认关闭以保持与论文一致：

| 配置项 | 开启示例 | 作用 |
|---|---|---|
| `model.residual_norm` | `true` | 用训练分区残差的逐维均值/标准差标准化后再扩散，采样后还原；`L_res` 仍在原始尺度上计算 |
| `model.min_snr_gamma` | `4` | `L_diff` 使用 min-SNR-γ 加权，减少高噪声时间步对训练的主导 |
| `model.ema_decay` | `0.999` | 阶段 3 对 CPD 维护 EMA 权重，验证、保存和推理都用 EMA 权重 |

```bash
python train.py --config configs/server.yaml \
    --set model.residual_norm=true model.min_snr_gamma=4 model.ema_decay=0.999
```

每个阶段的逐 epoch 训练损失、学习率和验证指标写入 `fold_k/history_stage{2..5}.jsonl`。

## 4. 训练

```bash
python train.py --config configs/server.yaml
```

默认对 5 折依次跑阶段 1–5：

| 阶段 | 训练 | 冻结 | 选模型 | 输出（`<log_dir>/fold_k/`） |
|---|---|---|---|---|
| 1 | 术中编码器（ResNet-34 + TabTransformer + φ_fs）+ 临时头；术后 φ_post + PAM + 临时头 | TITAN（特征已预提取） | 验证 acc | `stage1_intra.pth`、`stage1_post.pth`、`ct_stats.json`、`features.pt` |
| 2 | Anchor Predictor | 编码器 | 验证 L_anchor | `stage2_anchor.pth` |
| 3 | CPD；结束后估计 Analytic-DPM Γ | 编码器、Anchor | 验证 L_CPD | `stage3_cpd.pth` |
| 4 | GU-Gate（K 次采样） | 以上全部 | 验证 L_GU | `stage4_gu_gate.pth` |
| 5 | CMAA + 分类头，最后在测试折评估 | 以上全部 | 验证 acc | `stage5_cmaa.pth`、`test_metrics.json`、`test_predictions.pt` |

阶段 5 结束后在 `<log_dir>/internal_summary.json` 汇总五折测试指标均值±标准差和 pooled OOF 混淆矩阵。

常用命令：

```bash
# 只跑某些折 / 阶段（后面的阶段依赖前面阶段的输出）
python train.py --config configs/server.yaml --folds 1 2 --stages 1
python train.py --config configs/server.yaml --stages 3 4 5

# 先用 1 个 epoch 检查能否跑通
python train.py --config configs/server.yaml --folds 1 --set train.log_dir=runs/smoke train.epochs=1

# 需要画 Fig.4 / Fig.6 时开启嵌入记录
python train.py --config configs/server.yaml --set train.track_embeddings=true

# 后台运行
nohup python -u train.py --config configs/server.yaml --set train.track_embeddings=true > runs/urdp.log 2>&1 &
tail -f runs/urdp.log
```

## 5. 评估与推理

```bash
# 有标签的外部队列：5 个 fold 模型各自指标 mean ± std，以及概率平均集成的混淆矩阵
python evaluate.py --config configs/server.yaml --excel external.xlsx --fs_dir /path/to/external_fs
#   -> <log_dir>/external_summary.json

# 新患者（不需要标签和术后病理）
python predict.py --config configs/server.yaml --excel new_patients.xlsx --out predictions.csv
#   -> 每类概率 prob_*、预测类别 pred、生成不确定度 uncertainty
```

两者都会使用各折训练集的 CT 统计量和冻结编码器，`--folds` 可只用部分折模型。

Python 调用：

```python
from urdp import URDPDataset, load_config, predict_cohort

cfg = load_config("configs/server.yaml")
ds = URDPDataset(cfg.data, excel="new_patients.xlsx", with_labels=False)
res = predict_cohort(cfg, ds)          # res["probs"]: [N, 3]，res["uncertainty"]: [N]
```

## 6. 可视化（`visualization/`）

图片和数据输出到 `<log_dir>/figures/`（png + pdf + json）。

| 命令 | 对应 | 前置条件 |
|---|---|---|
| `python visualization/fig4_umap.py --log_dir runs/urdp --fold 1 --epochs 1 10 30` | Fig.4 UMAP | 训练时 `track_embeddings=true` |
| `python visualization/fig5_confusion.py --log_dir runs/urdp` | Fig.5 混淆矩阵 | 阶段 5；外部部分需先跑 `evaluate.py` |
| `python visualization/fig6_mmd.py --log_dir runs/urdp --full_log_dir runs/urdp_full` | Fig.6 MMD 曲线 | `track_embeddings=true`；Full-feature 曲线需下面的消融 |
| `python visualization/fig7_uncertainty.py --log_dir runs/urdp --fold_b 1` | Fig.7 + Table IV | 阶段 5 |
| `python visualization/table3_cmaa_depth.py --log_dir runs/urdp` | Table III | 阶段 1–4；会为 L=1..8 各重训一次阶段 5 |

Fig.6 的 Full-feature 消融（复用主实验的阶段 1 特征，只跑阶段 3）：

```bash
python train.py --config configs/server.yaml --stages 3 \
    --set model.generation_target=full train.track_embeddings=true \
          train.log_dir=runs/urdp_full train.features_dir=runs/urdp
```

`table3_cmaa_depth.py --no_train` 只汇总已有结果；`--depths 2 4 6` 可只跑部分深度。

## 7. 输出目录

```text
<log_dir>/
  config.yaml              本次训练使用的完整配置
  splits.json              五折划分（患者行号）
  internal_summary.json    五折测试汇总
  external_summary.json    evaluate.py 输出
  figures/                 可视化
  fold_k/
    ct_stats.json  features.pt  stage1_*.pth ... stage5_cmaa.pth
    test_metrics.json  test_predictions.pt
    embeddings/            track_embeddings=true 时的逐 epoch 嵌入
```

换数据后请换一个 `log_dir`：`splits.json` 与患者数不一致时会报错，避免误用旧划分。
