# -*- coding: utf-8 -*-
"""
U-RDP 分阶段训练（每折独立，后一阶段冻结前面所有模块）

    阶段 1  表征预训练：术中分支（ResNet-34 + TabTransformer + 投影）与术后分支（φ_post + PAM），
            各接临时分类头用 focal loss 训练，按验证 acc 选模型；随后冻结并缓存所有患者特征
    阶段 2  Anchor Predictor       L_anchor   = L_align(z_anchor, z_post; λa)
    阶段 3  CPD 残差扩散           L_CPD      = λc L_diff + (1-λc) L_res；训练后估计 Analytic-DPM Γ
    阶段 4  GU-Gate               L_GU-Gate  = L_align(r̃, r; λd)（K 次采样，CPD 冻结）
    阶段 5  CMAA + 分类头          Focal loss；最终在 held-out 测试折上评估
"""
import contextlib
import os
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from urdp.config import Config
from urdp.data import DatasetView, URDPDataset, cat_cardinalities, collate, compute_ct_stats, load_or_create_splits
from urdp.losses import FocalLoss
from urdp.metrics import KEYS, classification_metrics, fmt, format_cm, mean_std, print_mean_std
from urdp.models.encoders import IntraEncoder, MLPHead, PostEncoder
from urdp.models.framework import STAGE_MODULES, URDP, ckpt_name
from urdp.utils import EMA, BestKeeper, append_jsonl, autocast, freeze, load_json, make_generator, make_optimizer, save_json, set_seed

SPLITS = ("train", "val", "test")


class PretrainModel(nn.Module):
    """编码器 + 临时分类头（阶段一结束后丢弃临时头）"""

    def __init__(self, cfg: Config, branch: str):
        super().__init__()
        mc, dc = cfg.model, cfg.data
        self.branch = branch
        if branch == "intra":
            self.encoder = IntraEncoder(mc, len(dc.num_cols), cat_cardinalities(dc))
            in_dim = 3 * mc.d
        else:
            self.encoder = PostEncoder(mc)
            in_dim = mc.d
        self.temp_head = MLPHead(in_dim, mc.temp_head_hidden, mc.num_classes, mc.temp_head_dropout)

    def forward(self, batch, device):
        z = self.encoder(batch, device)
        if self.branch == "intra":
            z = torch.cat([z["z_ct"], z["z_fs"], z["z_tab"]], -1)
        return self.temp_head(z)


def build_intra_encoder(cfg: Config) -> IntraEncoder:
    return IntraEncoder(cfg.model, len(cfg.data.num_cols), cat_cardinalities(cfg.data))


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device
        self.log_dir = cfg.train.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self._ds = None
        self._folds = None
        self._focal = FocalLoss(cfg.model.focal_gamma)

    def _amp(self):
        """混合精度上下文（train.amp），只加速、不改变方法。"""
        return autocast(self.device, self.cfg.train.amp)

    # ------------------------------------------------------------------
    @property
    def dataset(self) -> URDPDataset:
        if self._ds is None:
            # 训练队列要求四模态齐全：阶段 1-5 共用同一批患者与同一份划分
            self._ds = URDPDataset(self.cfg.data, with_labels=True, require_post=True)
        return self._ds

    @property
    def folds(self):
        if self._folds is None:
            self._folds = load_or_create_splits(self.dataset, self.cfg.train.n_folds, self.cfg.train.seed,
                                                os.path.join(self.log_dir, "splits.json"))
        return self._folds

    def fold_dir(self, fold: int) -> str:
        d = os.path.join(self.log_dir, f"fold_{fold}")
        os.makedirs(d, exist_ok=True)
        return d

    def run(self, stages: Iterable[int], folds: Iterable[int]):
        self.cfg.save(os.path.join(self.log_dir, f"config{self._sfx}.yaml"))
        stages = sorted(set(stages))
        if 2 in stages and not URDP.stages_needed(self.cfg.model, 2):
            print("[info] generation_target=full：跳过阶段 2（不使用 Anchor）")
            stages.remove(2)
        for fold in folds:
            for stage in stages:
                print(f"\n========== fold {fold} | stage {stage} ==========")
                set_seed(self.cfg.train.seed + fold)
                if stage == 1:
                    self.pretrain(fold)
                else:
                    self.fit_stage(stage, fold)
        if 5 in stages:
            self.summarize_internal()

    # ==================================================================
    # 阶段一
    # ==================================================================
    def _loader(self, idx, shuffle: bool, load_ct: bool, load_post: bool, bs: int = None) -> DataLoader:
        return DataLoader(DatasetView(self.dataset, load_ct, load_post, idx), batch_size=bs or self.cfg.train.batch_size_pretrain, shuffle=shuffle,
                          num_workers=self.cfg.train.num_workers, pin_memory=True, collate_fn=collate)

    def _run_epoch(self, model, loader, crit, opt=None) -> Dict:
        model.train(opt is not None)
        probs, ys, total = [], [], 0.0
        with torch.set_grad_enabled(opt is not None):
            for batch in tqdm(loader, leave=False):
                y = batch["y"].to(self.device)
                with self._amp():
                    logits = model(batch, self.device)
                loss = crit(logits.float(), y)
                if opt is not None:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), self.cfg.train.grad_clip)
                    opt.step()
                total += float(loss) * y.size(0)
                probs.append(logits.detach().float().softmax(-1).cpu())
                ys.append(y.cpu())
        m = classification_metrics(torch.cat(ys).numpy(), torch.cat(probs).numpy(), self.cfg.model.num_classes)
        m["loss"] = total / len(loader.dataset)
        return m

    def pretrain(self, fold: int):
        tc, fd = self.cfg.train, self.folds[fold - 1]
        out = self.fold_dir(fold)
        mean, std = compute_ct_stats(self.dataset, fd["train"])
        self.dataset.set_ct_stats(mean, std)
        save_json({"mean": mean, "std": std}, os.path.join(out, "ct_stats.json"))
        print(f"[CT stats] mean={mean:.4f} std={std:.4f}")

        crit = FocalLoss(self.cfg.model.focal_gamma)
        for branch, (load_ct, load_post) in (("intra", (True, False)), ("post", (False, True))):
            model = PretrainModel(self.cfg, branch).to(self.device)
            opt, sched = make_optimizer(model.parameters(), tc)
            tl = self._loader(fd["train"], True, load_ct, load_post)
            vl = self._loader(fd["val"], False, load_ct, load_post)
            keeper = BestKeeper(os.path.join(out, f"stage1_{branch}.pth"), "max", tc.keep_topk)
            for ep in range(1, tc.epochs + 1):
                tr = self._run_epoch(model, tl, crit, opt)
                va = self._run_epoch(model, vl, crit)
                sched.step()
                keeper.update(va["acc"], ep, {"epoch": ep, "val": va, "encoder": model.encoder.state_dict()})
                print(f"[{branch}] ep{ep:02d} train {fmt(tr)} loss={tr['loss']:.4f} | val {fmt(va)}")
            print(f"[{branch}] best val acc={keeper.best:.4f} @ epoch {keeper.epoch}")
        self.cache_features(fold)

    @torch.no_grad()
    def cache_features(self, fold: int):
        out = self.fold_dir(fold)
        intra = build_intra_encoder(self.cfg)
        intra.load_state_dict(torch.load(os.path.join(out, "stage1_intra.pth"), map_location="cpu")["encoder"])
        post = PostEncoder(self.cfg.model)
        post.load_state_dict(torch.load(os.path.join(out, "stage1_post.pth"), map_location="cpu")["encoder"])
        intra, post = freeze(intra.to(self.device)), freeze(post.to(self.device))

        loader = self._loader(None, False, True, True)
        feats = {k: [] for k in ("z_ct", "z_fs", "z_tab", "z_post", "y")}
        for batch in tqdm(loader, desc="cache features"):
            with self._amp():
                z_intra, z_post = intra(batch, self.device), post(batch, self.device)
            for k, v in z_intra.items():
                feats[k].append(v.float().cpu())
            feats["z_post"].append(z_post.float().cpu())
            feats["y"].append(batch["y"])
        feats = {k: torch.cat(v) for k, v in feats.items()}
        fd = self.folds[fold - 1]
        feats.update({f"{s}_idx": torch.tensor(fd[s]) for s in SPLITS})
        torch.save(feats, os.path.join(out, "features.pt"))
        print(f"[cache] {feats['y'].numel()} patients -> {os.path.join(out, 'features.pt')}")
        # 各阶段使用的数据：
        #   阶段 2 Anchor   : 输入 z_fs，监督 z_post
        #   阶段 3 CPD      : 条件 [z_ct, z_fs, z_tab]，目标 r = z_post - z_anchor
        #   阶段 4 GU-Gate  : 条件 [z_ct, z_fs, z_tab]，目标 r
        #   阶段 5 CMAA     : [z_ct, z_fs, z_tab, 生成的 z_post_tilde]（不使用 z_post）

    # ==================================================================
    # 阶段二至五（缓存特征）
    # ==================================================================
    def features_file(self, fold: int) -> str:
        """阶段 1 缓存特征。train.features_dir 可指向另一次运行（消融实验复用同一套冻结编码器）"""
        root = self.cfg.train.features_dir or self.log_dir
        return os.path.join(root, f"fold_{fold}", "features.pt")

    def load_features(self, fold: int) -> Dict[str, Dict[str, torch.Tensor]]:
        path = self.features_file(fold)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} 不存在，请先运行阶段 1")
        f = torch.load(path, map_location=self.device)
        out = {}
        for s in SPLITS:
            idx = f[f"{s}_idx"]
            out[s] = {k: f[k][idx] for k in ("z_ct", "z_fs", "z_tab", "z_post", "y")}
            out[s]["idx"] = idx
        return out

    @staticmethod
    def _batches(feats: Dict, bs: int, shuffle: bool):
        n = feats["y"].numel()
        order = torch.randperm(n, device=feats["y"].device) if shuffle else torch.arange(n, device=feats["y"].device)
        for i in range(0, n, bs):
            b = order[i:i + bs]
            yield {k: v[b] for k, v in feats.items()}

    def _stage_loss(self, model: URDP, stage: int, b: Dict, generator=None) -> Dict:
        if stage == 2:
            return {"loss": model.anchor_loss(b["z_fs"], b["z_post"])}
        if stage == 3:
            return model.cpd_loss(b["z_ct"], b["z_fs"], b["z_tab"], b["z_post"])
        gen = model.generate(b["z_ct"], b["z_fs"], b["z_tab"], generator=generator)
        if stage == 4:
            return {"loss": model.gu_gate_loss(gen, b["z_post"])}
        logits = model.classify(b["z_ct"], b["z_fs"], b["z_tab"], gen["z_post_tilde"].detach())["logits"]
        return {"loss": self._focal(logits, b["y"]), "probs": logits.detach().float().softmax(-1)}

    @torch.no_grad()
    def _evaluate(self, model: URDP, stage: int, feats: Dict, seed: int = 0) -> Dict:
        """验证/测试：固定随机种子，使不同 epoch 可比。"""
        model.eval()
        gen = make_generator(self.device, seed)
        devices = [torch.device(self.device)] if self.device.startswith("cuda") else []
        tot, n, probs = {}, 0, []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for b in self._batches(feats, self.cfg.train.batch_size, False):
                with self._amp():
                    out = self._stage_loss(model, stage, b, generator=gen)
                for k in ("loss", "diff_loss", "res_loss"):
                    if k in out:
                        tot[k] = tot.get(k, 0.0) + float(out[k]) * b["y"].numel()
                if "probs" in out:
                    probs.append(out["probs"])
                n += b["y"].numel()
        res = {k: v / n for k, v in tot.items()}
        if probs:
            p = torch.cat(probs)
            res.update(classification_metrics(feats["y"].cpu().numpy(), p.cpu().numpy(), self.cfg.model.num_classes))
            res["probs"] = p
        return res

    def fit_stage(self, stage: int, fold: int):
        tc, out = self.cfg.train, self.fold_dir(fold)
        feats = self.load_features(fold)
        tag = tc.run_tag
        model = URDP.from_fold(self.cfg.model, out, upto=stage - 1, device=self.device)
        opt, sched = make_optimizer(model.set_trainable(stage), tc)
        target = model.module(STAGE_MODULES[stage])
        if stage == 3:
            # 可选的残差标准化：统计量只用训练分区（model.residual_norm=False 时为恒等变换）
            tr = feats["train"]
            with torch.no_grad():
                model.diffusion.set_residual_stats(tr["z_post"] - model.base(tr["z_fs"]))
        # 可选的 EMA：仅阶段 3 的 CPD
        ema = EMA(target, self.cfg.model.ema_decay) if (stage == 3 and self.cfg.model.ema_decay > 0) else None
        mode = "max" if stage == 5 else "min"
        keeper = BestKeeper(os.path.join(out, ckpt_name(stage, tag)), mode, tc.keep_topk)
        history = os.path.join(out, f"history_stage{stage}{self._sfx if stage == 5 else ''}.jsonl")
        if os.path.isfile(history):
            os.remove(history)
        track = tc.track_embeddings and stage in (2, 3)
        if track:
            self._save_reference(fold, feats["test"])

        for ep in range(1, tc.epochs + 1):
            target.train()
            tot, n, skipped = 0.0, 0, 0
            for b in self._batches(feats["train"], tc.batch_size, True):
                with self._amp():
                    loss = self._stage_loss(model, stage, b)["loss"]
                if not torch.isfinite(loss):
                    skipped += 1          # 数值异常的 batch 直接跳过（旧版 NaNGuard）
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(target.parameters(), tc.grad_clip)
                opt.step()
                if ema is not None:
                    ema.update()
                tot += float(loss) * b["y"].numel()
                n += b["y"].numel()
            sched.step()
            # 验证与保存：启用 EMA 时用 EMA 权重
            with (ema.apply() if ema is not None else contextlib.nullcontext()):
                val = self._evaluate(model, stage, feats["val"])
                score = val["acc"] if stage == 5 else val["loss"]
                keeper.update(score, ep, {"epoch": ep, "val_score": score, **model.state_for(stage)})
                if track:
                    self._track_epoch(model, stage, fold, ep, feats["test"])
            extra = (f" L_diff={val['diff_loss']:.4f} L_res={val['res_loss']:.4f}" if stage == 3
                     else f" | {fmt(val)}" if stage == 5 else "")
            warn = f" (跳过 {skipped} 个非有限 batch)" if skipped else ""
            print(f"[stage {stage}] ep{ep:02d} train loss={tot / max(n, 1):.4f} | val loss={val['loss']:.4f}{extra}{warn}")
            append_jsonl({"epoch": ep, "train_loss": tot / max(n, 1), "lr": opt.param_groups[0]["lr"],
                          **{k: v for k, v in val.items() if k not in ("probs", "cm")}}, history)
        print(f"[stage {stage}] best val {'acc' if stage == 5 else 'loss'}={keeper.best:.4f} @ epoch {keeper.epoch}")

        model.load_stage(stage, out, map_location=self.device, tag=tag)
        if stage == 3:
            # Analytic-DPM：在训练分区上估计 Γ_n，写回 checkpoint
            tr = feats["train"]
            with torch.no_grad():
                r = tr["z_post"] - model.base(tr["z_fs"])
            with self._amp():
                model.diffusion.estimate_gamma(model.condition(tr["z_ct"], tr["z_fs"], tr["z_tab"]), r)
            ck = torch.load(keeper.path, map_location="cpu")
            ck.update(model.state_for(3))
            torch.save(ck, keeper.path)
            print("[stage 3] Analytic-DPM Γ estimated")
        if stage == 5:
            self.test(model, fold, feats["test"])

    # ------------------------------------------------------------------
    # 可视化用的逐 epoch 嵌入（测试折，固定种子）
    # ------------------------------------------------------------------
    def _emb_dir(self, fold: int) -> str:
        d = os.path.join(self.fold_dir(fold), "embeddings")
        os.makedirs(d, exist_ok=True)
        return d

    def _save_reference(self, fold: int, feats: Dict):
        torch.save({k: feats[k].cpu() for k in ("z_ct", "z_fs", "z_tab", "z_post", "y", "idx")},
                   os.path.join(self._emb_dir(fold), "reference.pt"))

    @torch.no_grad()
    def _track_epoch(self, model: URDP, stage: int, fold: int, ep: int, feats: Dict, seed: int = 0):
        """阶段 2 保存 z_anchor；阶段 3 另存 K 个残差候选 [K, N, d]"""
        model.eval()
        gen = make_generator(self.device, seed)
        anchors, cands = [], []
        for b in self._batches(feats, self.cfg.train.batch_size, False):
            anchors.append(model.base(b["z_fs"]).cpu())
            if stage == 3:
                with self._amp():
                    c = model.diffusion.sample_k(model.condition(b["z_ct"], b["z_fs"], b["z_tab"]),
                                                 self.cfg.model.K, generator=gen)
                cands.append(c.float().cpu())
        out = {"z_anchor": torch.cat(anchors).half()}
        if stage == 3:
            out["cands"] = torch.cat(cands, dim=1).half()
        torch.save(out, os.path.join(self._emb_dir(fold), f"stage{stage}_ep{ep:03d}.pt"))

    def test(self, model: URDP, fold: int, feats: Dict):
        """held-out 测试折，只用最终选定的 checkpoint 评估一次。"""
        res = self._evaluate(model, 5, feats)
        names = self.cfg.data.class_names
        print(f"[fold {fold}] TEST {fmt(res)}\n{format_cm(res['cm'], names)}")
        out, sfx = self.fold_dir(fold), self._sfx
        torch.save({"idx": feats["idx"].cpu(), "y": feats["y"].cpu(), "probs": res.pop("probs").cpu()},
                   os.path.join(out, f"test_predictions{sfx}.pt"))
        save_json({k: res[k] for k in KEYS + ("cm",)}, os.path.join(out, f"test_metrics{sfx}.json"))

    @property
    def _sfx(self) -> str:
        return f"_{self.cfg.train.run_tag}" if self.cfg.train.run_tag else ""

    def summarize_internal(self):
        """五折测试指标 mean±std（Table I）+ pooled OOF 混淆矩阵（Fig.5(a)）"""
        per_fold, ys, ps, sfx = [], [], [], self._sfx
        for fold in range(1, self.cfg.train.n_folds + 1):
            d = os.path.join(self.log_dir, f"fold_{fold}")
            if not os.path.isfile(os.path.join(d, f"test_metrics{sfx}.json")):
                continue
            per_fold.append({"fold": fold, **load_json(os.path.join(d, f"test_metrics{sfx}.json"))})
            pred = torch.load(os.path.join(d, f"test_predictions{sfx}.pt"))
            ys.append(pred["y"].numpy())
            ps.append(pred["probs"].numpy())
        if not per_fold:
            return
        summary = {"per_fold": per_fold, **mean_std(per_fold)}
        summary["pooled_oof"] = classification_metrics(np.concatenate(ys), np.concatenate(ps),
                                                       self.cfg.model.num_classes)
        save_json(summary, os.path.join(self.log_dir, f"internal_summary{sfx}.json"))
        print_mean_std(f"Internal test ({len(per_fold)} folds, mean ± std)", summary)
        print("Pooled OOF confusion matrix:")
        print(format_cm(summary["pooled_oof"]["cm"], self.cfg.data.class_names))
        return summary
