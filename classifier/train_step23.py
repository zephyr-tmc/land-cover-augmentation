import os
import json
import time
import random
import argparse
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from dataset import make_dataloaders
from model import build_model

# 复用原训练实现（避免重复/不影响你正在跑的 train.py）
import train as base_train

# 复用仓库内 evaluate.py 的指标实现
# 添加父目录到 sys.path 以便导入 evaluate.py
import sys
_eval_py_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _eval_py_dir not in sys.path:
    sys.path.insert(0, _eval_py_dir)
try:
    from evaluate import (
        _validate_inputs,
        compute_basic_metrics,
        compute_roc_prc_metrics,
        macro_f1_argmax,
        save_metrics as _save_metrics_json,
        plot_cm as _plot_cm,
        plot_roc_ovr as _plot_roc,
        plot_prc_ovr as _plot_prc,
    )
except Exception as e:
    _validate_inputs = None
    compute_basic_metrics = None
    compute_roc_prc_metrics = None
    macro_f1_argmax = None
    _save_metrics_json = None
    _plot_cm = None
    _plot_roc = None
    _plot_prc = None
    print(f"[Warning] 导入 evaluate.py 失败: {e}")

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


class FocalLoss(nn.Module):
    """多类 focal loss（可选 class weight）。"""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # label smoothing：转为 soft target
        if self.label_smoothing > 0:
            num_classes = logits.size(1)
            with torch.no_grad():
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.label_smoothing / (num_classes - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
            logp = torch.log_softmax(logits, dim=1)
            p = torch.exp(logp)
            focal = (1.0 - p).pow(self.gamma)
            loss = -(true_dist * focal * logp).sum(dim=1)
        else:
            logp = torch.log_softmax(logits, dim=1)
            p = torch.exp(logp)
            pt = p[torch.arange(p.size(0), device=p.device), target]
            logpt = logp[torch.arange(logp.size(0), device=logp.device), target]
            focal = (1.0 - pt).pow(self.gamma)
            loss = -focal * logpt

        if self.weight is not None:
            w = self.weight[target]
            loss = loss * w
        return loss.mean()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_probs_and_labels(model, dl, device):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in dl:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    y_prob = np.concatenate(all_probs, axis=0) if len(all_probs) else np.zeros((0, 0), dtype=np.float32)
    y_true = np.concatenate(all_labels, axis=0) if len(all_labels) else np.zeros((0,), dtype=np.int64)
    return y_true, y_prob


def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]) -> dict:
    if _validate_inputs is None or compute_basic_metrics is None or compute_roc_prc_metrics is None or macro_f1_argmax is None:
        raise RuntimeError("evaluate.py 不可用：请确认 /root/autodl-tmp/data/templete/train/evaluate.py 可被导入，且依赖 sklearn 已安装。")
    y_true2, y_prob2 = _validate_inputs(y_true, y_prob, class_names)
    basic = compute_basic_metrics(y_true2, y_prob2, class_names)
    curves = compute_roc_prc_metrics(y_true2, y_prob2, class_names)
    mf1 = float(macro_f1_argmax(y_true2, y_prob2))
    metrics = {
        "num_samples": int(len(y_true2)),
        "num_classes": int(len(class_names)),
        "class_names": class_names,
        **basic,
        **curves,
        "macro_f1": {"argmax": mf1, "val_fixed_thresholds": None, "per_class_val_thresholds": None},
    }
    return metrics


def flatten_metrics_for_wandb(prefix: str, metrics: dict) -> dict:
    out = {}
    if not metrics:
        return out
    p = prefix.rstrip("/") + "/"
    out[p + "accuracy"] = float(metrics.get("accuracy", 0.0))
    macro = metrics.get("macro", {}) or {}
    out[p + "macro/precision"] = float(macro.get("precision", 0.0))
    out[p + "macro/recall"] = float(macro.get("recall", 0.0))
    mf1 = (metrics.get("macro_f1", {}) or {}).get("argmax", None)
    if mf1 is not None:
        out[p + "macro/f1"] = float(mf1)
    pr_ap = metrics.get("pr_ap", {}) or {}
    roc_auc = metrics.get("roc_auc", {}) or {}
    out[p + "pr_ap/micro"] = float(pr_ap.get("micro", 0.0) or 0.0)
    out[p + "pr_ap/macro"] = float(pr_ap.get("macro", 0.0) or 0.0)
    out[p + "roc_auc/micro"] = float(roc_auc.get("micro", 0.0) or 0.0)
    out[p + "roc_auc/macro"] = float(roc_auc.get("macro", 0.0) or 0.0)
    for row in (metrics.get("per_class", []) or []):
        cls = row.get("class", "unknown")
        out[p + f"per_class/{cls}/precision"] = float(row.get("precision", 0.0))
        out[p + f"per_class/{cls}/recall"] = float(row.get("recall", 0.0))
        out[p + f"per_class/{cls}/f1"] = float(row.get("f1", 0.0))
        out[p + f"per_class/{cls}/support"] = int(row.get("support", 0))
    ap_pc = (metrics.get("pr_ap", {}) or {}).get("per_class", {}) or {}
    for cls, ap in ap_pc.items():
        try:
            out[p + f"per_class/{cls}/ap"] = float(ap) if ap is not None else 0.0
        except Exception:
            out[p + f"per_class/{cls}/ap"] = 0.0
    return out


def save_split_outputs(out_dir: str, split: str, y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], save_plots: bool):
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    np.save(os.path.join(split_dir, "y_prob.npy"), y_prob)
    np.save(os.path.join(split_dir, "y_true.npy"), y_true)
    with open(os.path.join(split_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, ensure_ascii=False)
    metrics = compute_all_metrics(y_true, y_prob, class_names)
    if _save_metrics_json is not None:
        _save_metrics_json(split_dir, metrics, filename="metrics.json")
    else:
        with open(os.path.join(split_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    if save_plots and _plot_cm is not None and _plot_roc is not None and _plot_prc is not None:
        try:
            _plot_cm(np.asarray(metrics["confusion_matrix"]), class_names, os.path.join(split_dir, "cm.png"))
            _plot_roc(y_true, y_prob, class_names, os.path.join(split_dir, "roc.png"))
            _plot_prc(y_true, y_prob, class_names, os.path.join(split_dir, "prc.png"))
        except Exception as e:
            print(f"[Warning] plot failed for split={split}: {e}")
    return split_dir, metrics


def resolve_baseline1218(
    baseline: str,
    bamboo_add: int,
    extra_train_dirs: Optional[List[str]],
) -> Tuple[str, Optional[List[str]], Optional[Dict[str, int]], Optional[Dict[str, int]]]:
    """
    返回：
    - rebalance_strategy
    - extra_train_dirs（可能被置空）
    - target_hist（本轮不用，返回 None）
    - add_hist（只对 bamboo）
    """
    # 历史别名
    if baseline == "Diffusion-raw":
        baseline = "lora-raw"
    if baseline == "Diffusion-align":
        baseline = "lora-align"

    if baseline == "Real-only":
        return "none", None, None, None
    if baseline == "B1dup":
        return "dup_original_add_k", None, None, {"bamboo": int(bamboo_add)}
    if baseline in ["B2trad", "GAN", "lora-raw", "lora-align"]:
        if not extra_train_dirs:
            print(f"[Warning] baseline={baseline} 但未提供 --extra-train-dirs，将退化为 Real-only。")
            return "none", None, None, None
        return "fill_with_extra_add_k", extra_train_dirs, None, {"bamboo": int(bamboo_add)}

    raise ValueError(f"Unknown baseline1218: {baseline}")


def compute_class_weights(counts: List[int], mode: str) -> Optional[torch.Tensor]:
    """
    mode:
      - none
      - inv: 1/n 形式（归一化到 mean=1）
      - sqrt_inv: 1/sqrt(n) 形式（归一化到 mean=1）
    """
    mode = (mode or "none").lower()
    if mode == "none":
        return None
    counts_t = torch.tensor(counts, dtype=torch.float32).clamp(min=1)
    K = len(counts_t)
    # 以均匀类分布的期望为基准，做 mean=1 归一化（更稳定）
    if mode == "inv":
        w = (counts_t.sum() / K) / counts_t
    elif mode == "sqrt_inv":
        w = (counts_t.sum() / K) / counts_t.sqrt()
    else:
        raise ValueError(f"Unknown class weight mode: {mode}")
    w = w * (K / w.sum())
    return w


def main():
    parser = argparse.ArgumentParser(description="Baseline_1218 Step2/3 trainer (new entry, does not touch train.py)")

    # data
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/data/dataset_preprocessed")
    parser.add_argument("--layout", type=str, default="split", choices=["auto", "split", "subdir"])
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--test-dir", type=str, default=None)
    parser.add_argument("--train-subdir", type=str, default="org286")
    parser.add_argument("--val-subdir", type=str, default="val286")
    parser.add_argument("--test-subdir", type=str, default=None)

    # exp
    parser.add_argument("--baseline1218", type=str, required=True,
                        choices=["Real-only", "B1dup", "B2trad", "GAN", "lora-raw", "lora-align", "Diffusion-raw", "Diffusion-align"])
    parser.add_argument("--bamboo-add", type=int, default=224)
    parser.add_argument("--extra-train-dirs", type=str, default=None, help="逗号分隔；允许只含 bamboo")

    # model/train
    parser.add_argument("--backbone", type=str, default="smallcnnpp", choices=["smallcnn", "smallcnnpp", "resnet18"])
    parser.add_argument("--img-size", type=int, default=286)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aug-loader", type=str, default="traditional", choices=["preprocess_only", "traditional"])
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--clip-grad-norm", type=float, default=0.0)

    # loss
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--class-weight", type=str, default="none", choices=["none", "inv", "sqrt_inv"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    # two-stage
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-lr-mult", type=float, default=0.5)

    # output
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--eval-test", type=str, default="end", choices=["none", "end", "epoch"],
                        help="是否在 test 上评估：none=不评估；end=仅训练结束对best评估；epoch=每个epoch都评估并记录")
    parser.add_argument("--save-plots", action="store_true", help="保存 cm/roc/prc 图到输出目录（val/test）")
    parser.add_argument("--best-metric", type=str, default="val_loss", choices=["val_loss", "val_macro_f1", "val_accuracy"],
                        help="选择 best 模型的指标：val_loss=验证集loss（越小越好）；val_macro_f1=验证集macro-F1（越大越好）；val_accuracy=验证集准确率（越大越好）")

    # wandb
    parser.add_argument("--wandb", action="store_true", help="启用 Weights & Biases 监控")
    parser.add_argument("--wandb-project", type=str, default="Baseline_0106", help="wandb project")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity/team（可选）")
    parser.add_argument("--wandb-name", type=str, default=None, help="wandb run name（可选）")
    parser.add_argument("--wandb-tags", type=str, default=None, help="逗号分隔 tags（可选）")
    parser.add_argument("--wandb-offline", action="store_true", help="wandb 离线模式（不上传）")
    parser.add_argument("--wandb-dir", type=str, default=None, help="wandb 本地缓存目录（默认 out_dir/wandb）")

    args = parser.parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # layout resolve
    data_root = os.path.abspath(args.data_root)
    layout = args.layout
    if layout == "auto":
        layout = "split" if os.path.isdir(os.path.join(data_root, "train")) else "subdir"

    if layout == "split":
        train_dir = args.train_dir or os.path.join(data_root, "train")
        val_dir = args.val_dir or os.path.join(data_root, "val")
        test_dir = args.test_dir or (os.path.join(data_root, "test") if os.path.isdir(os.path.join(data_root, "test")) else None)
        train_only_with_subdir = None
        val_only_with_subdir = None
        test_only_with_subdir = None
    else:
        train_dir = data_root
        val_dir = data_root
        test_dir = data_root if args.test_subdir else None
        train_only_with_subdir = args.train_subdir
        val_only_with_subdir = args.val_subdir
        test_only_with_subdir = args.test_subdir

    extra_train_dirs = None
    if args.extra_train_dirs:
        extra_train_dirs = [p.strip() for p in args.extra_train_dirs.split(",") if p.strip()]

    # 构造 stage2 的训练集策略（baseline1218）
    rebalance_strategy2, extra2, target_hist2, add_hist2 = resolve_baseline1218(
        args.baseline1218, args.bamboo_add, extra_train_dirs
    )

    # stage1：Real-only
    stage1_epochs = args.stage1_epochs if args.stage1_epochs is not None else max(1, args.epochs // 2)
    if not args.two_stage:
        stage1_epochs = 0

    # build loaders（stage2）
    dl_train2, dl_val, dl_test, num_classes, class_names, _, counts2 = make_dataloaders(
        root_train=train_dir,
        root_val=val_dir,
        root_test=test_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=args.aug_loader,
        train_only_with_subdir=train_only_with_subdir,
        val_only_with_subdir=val_only_with_subdir,
        test_only_with_subdir=test_only_with_subdir,
        extra_train_dirs=extra2,
        use_weighted_sampler=False,
        rebalance_strategy=rebalance_strategy2,
        target_hist=target_hist2,
        add_hist=add_hist2,
    )

    dl_train1 = None
    counts1 = None
    if args.two_stage:
        dl_train1, _, _, nc1, cn1, _, counts1 = make_dataloaders(
            root_train=train_dir,
            root_val=val_dir,
            root_test=test_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mode=args.aug_loader,
            train_only_with_subdir=train_only_with_subdir,
            val_only_with_subdir=val_only_with_subdir,
            test_only_with_subdir=test_only_with_subdir,
            extra_train_dirs=None,
            use_weighted_sampler=False,
            rebalance_strategy="none",
            target_hist=None,
            add_hist=None,
        )
        assert nc1 == num_classes and cn1 == class_names, "two-stage: 类别映射不一致"

    print("[Class names]:", class_names)
    if counts1 is not None:
        print("[Two-stage] stage1 counts:", counts1)
    print("[Stage2] counts:", counts2)

    # loss
    cw = compute_class_weights(counts2 if not args.two_stage else (counts1 if stage1_epochs > 0 else counts2), args.class_weight)
    if cw is not None:
        print(f"[Class weight mode={args.class_weight}] (mean=1):", [round(v, 4) for v in cw.tolist()])

    if args.loss == "ce":
        criterion_tr = nn.CrossEntropyLoss(weight=cw.to(device) if cw is not None else None, label_smoothing=args.label_smoothing)
        criterion_va = nn.CrossEntropyLoss(weight=None, label_smoothing=args.label_smoothing)
    else:
        criterion_tr = FocalLoss(gamma=args.focal_gamma, weight=cw.to(device) if cw is not None else None, label_smoothing=args.label_smoothing)
        criterion_va = nn.CrossEntropyLoss(weight=None, label_smoothing=args.label_smoothing)

    # model/opt
    model = build_model(num_classes, backbone=args.backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    early = base_train.EarlyStopper(patience=80, min_delta=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    ema = base_train.ModelEma(model, decay=0.999) if args.ema else None

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 同时跟踪三个指标的最佳模型
    best_metrics = {
        "val_loss": {"value": float("inf"), "path": os.path.join(out_dir, f"best_val_loss_{args.backbone}_{args.aug_loader}.pt")},
        "val_macro_f1": {"value": float("-inf"), "path": os.path.join(out_dir, f"best_val_macro_f1_{args.backbone}_{args.aug_loader}.pt")},
        "val_accuracy": {"value": float("-inf"), "path": os.path.join(out_dir, f"best_val_accuracy_{args.backbone}_{args.aug_loader}.pt")},
    }
    log = []

    # wandb init
    wb_run = None
    if args.wandb:
        if wandb is None:
            raise RuntimeError("已开启 --wandb，但当前环境未安装 wandb。请先 `pip install wandb`。")
        wandb_dir = args.wandb_dir or os.path.join(out_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        tags = [t.strip() for t in (args.wandb_tags or "").split(",") if t.strip()] or None
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or f"{args.baseline1218}_{args.backbone}",
            dir=wandb_dir,
            tags=tags,
            config=vars(args),
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")

    for ep in range(1, args.epochs + 1):
        if args.two_stage and stage1_epochs > 0 and ep == stage1_epochs + 1:
            mult = float(args.stage2_lr_mult)
            for pg in optimizer.param_groups:
                pg["lr"] = pg.get("lr", args.lr) * mult
            print(f"[Two-stage] Enter stage2 at epoch={ep}, lr *= {mult}")

        dl_train_used = dl_train2
        if args.two_stage and stage1_epochs > 0 and dl_train1 is not None:
            dl_train_used = dl_train1 if ep <= stage1_epochs else dl_train2

        tr_loss, tr_acc, _, _ = base_train.train_one_epoch(
            model, dl_train_used, device, criterion_tr, optimizer,
            args.max_steps, scaler=scaler, clip_grad_norm=args.clip_grad_norm, ema=ema
        )
        model_for_eval = ema.ema if ema is not None else model
        val_loss, val_acc, _, _ = base_train.evaluate(model_for_eval, dl_val, device, criterion_va)
        scheduler.step(val_loss)

        log.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": val_loss, "val_acc": val_acc})
        print(f"[{ep:03d}] train {tr_loss:.4f}/{tr_acc:.4f}  val {val_loss:.4f}/{val_acc:.4f}")

        # wandb log（含 val 的完整指标）
        val_metrics_epoch = None
        if wb_run is not None:
            lr_now = float(optimizer.param_groups[0].get("lr", args.lr))
            payload = {
                "epoch": ep,
                "train/loss": float(tr_loss),
                "train/accuracy": float(tr_acc),
                "train/lr": lr_now,
                "val/loss": float(val_loss),
                "val/accuracy": float(val_acc),
            }
            try:
                y_true_val, y_prob_val = predict_probs_and_labels(model_for_eval, dl_val, device)
                val_metrics_epoch = compute_all_metrics(y_true_val, y_prob_val, class_names)
                payload.update(flatten_metrics_for_wandb("val", val_metrics_epoch))
            except Exception as e:
                print(f"[Warning] compute val metrics failed: {e}")

            if args.eval_test == "epoch" and dl_test is not None:
                try:
                    y_true_te, y_prob_te = predict_probs_and_labels(model_for_eval, dl_test, device)
                    te_metrics = compute_all_metrics(y_true_te, y_prob_te, class_names)
                    payload.update(flatten_metrics_for_wandb("test", te_metrics))
                except Exception as e:
                    print(f"[Warning] compute test metrics (per-epoch) failed: {e}")

            wandb.log(payload)

        # 同时跟踪三个指标的最佳模型
        # 1. val_loss (越小越好)
        if float(val_loss) < best_metrics["val_loss"]["value"]:
            best_metrics["val_loss"]["value"] = float(val_loss)
            torch.save({
                "model": model_for_eval.state_dict(),
                "class_names": class_names,
                "args": vars(args),
                "metric": "val_loss",
                "metric_value": float(val_loss),
                "epoch": ep,
            }, best_metrics["val_loss"]["path"])
            print(f"  [Best val_loss] {val_loss:.6f} (saved)")

        # 2. val_accuracy (越大越好)
        if float(val_acc) > best_metrics["val_accuracy"]["value"]:
            best_metrics["val_accuracy"]["value"] = float(val_acc)
            torch.save({
                "model": model_for_eval.state_dict(),
                "class_names": class_names,
                "args": vars(args),
                "metric": "val_accuracy",
                "metric_value": float(val_acc),
                "epoch": ep,
            }, best_metrics["val_accuracy"]["path"])
            print(f"  [Best val_accuracy] {val_acc:.6f} (saved)")

        # 3. val_macro_f1 (越大越好)
        if val_metrics_epoch is None:
            try:
                y_true_val, y_prob_val = predict_probs_and_labels(model_for_eval, dl_val, device)
                val_metrics_epoch = compute_all_metrics(y_true_val, y_prob_val, class_names)
            except Exception as e:
                print(f"[Warning] compute val metrics for macro_f1 tracking failed: {e}")
        if val_metrics_epoch is not None:
            current_f1 = float(val_metrics_epoch.get("macro_f1", {}).get("argmax", 0.0))
            if current_f1 > best_metrics["val_macro_f1"]["value"]:
                best_metrics["val_macro_f1"]["value"] = current_f1
                torch.save({
                    "model": model_for_eval.state_dict(),
                    "class_names": class_names,
                    "args": vars(args),
                    "metric": "val_macro_f1",
                    "metric_value": current_f1,
                    "epoch": ep,
                }, best_metrics["val_macro_f1"]["path"])
                print(f"  [Best val_macro_f1] {current_f1:.6f} (saved)")

        if early.step(val_loss):
            print("Early stopping.")
            break

    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    # 分别用三个 best 模型在 val 和 test 上评估
    summary = {
        "out_dir": out_dir,
        "best_models": {},
        "splits": {},
    }

    # 为每个 best 模型创建评估结果
    for metric_name, best_info in best_metrics.items():
        if not os.path.exists(best_info["path"]):
            print(f"[Warning] Best model for {metric_name} not found: {best_info['path']}")
            continue

        print(f"\n[Evaluating best model by {metric_name}]")
        ckpt = torch.load(best_info["path"], map_location=device)
        (ema.ema if ema is not None else model).load_state_dict(ckpt["model"])
        model_for_eval = ema.ema if ema is not None else model

        summary["best_models"][metric_name] = {
            "path": best_info["path"],
            "metric_value": best_info["value"],
            "epoch": ckpt.get("epoch", "unknown"),
        }

        # 在 val 上评估
        try:
            y_true_val, y_prob_val = predict_probs_and_labels(model_for_eval, dl_val, device)
            val_dir, val_metrics = save_split_outputs(out_dir, f"val_best_{metric_name}", y_true_val, y_prob_val, class_names, save_plots=args.save_plots)
            if metric_name not in summary["splits"]:
                summary["splits"][metric_name] = {}
            summary["splits"][metric_name]["val"] = {"dir": val_dir, "metrics": val_metrics}
            print(f"  Val metrics: macro-F1={val_metrics.get('macro_f1', {}).get('argmax', 0.0):.6f}, accuracy={val_metrics.get('accuracy', 0.0):.6f}")
        except Exception as e:
            print(f"[Warning] export val for {metric_name} failed: {e}")

        # 在 test 上评估（如果存在）
        if dl_test is not None and args.eval_test in ["end", "epoch"]:
            try:
                y_true_te, y_prob_te = predict_probs_and_labels(model_for_eval, dl_test, device)
                test_dir, te_metrics = save_split_outputs(out_dir, f"test_best_{metric_name}", y_true_te, y_prob_te, class_names, save_plots=args.save_plots)
                if metric_name not in summary["splits"]:
                    summary["splits"][metric_name] = {}
                summary["splits"][metric_name]["test"] = {"dir": test_dir, "metrics": te_metrics}
                print(f"  Test metrics: macro-F1={te_metrics.get('macro_f1', {}).get('argmax', 0.0):.6f}, accuracy={te_metrics.get('accuracy', 0.0):.6f}")
            except Exception as e:
                print(f"[Warning] export test for {metric_name} failed: {e}")

    # 兼容：根目录复制一份（默认用 val_macro_f1 的结果，如果存在）
    default_metric = "val_macro_f1" if "val_macro_f1" in summary["splits"] else list(summary["splits"].keys())[0] if summary["splits"] else None
    if default_metric and "test" in summary["splits"].get(default_metric, {}):
        src_dir = summary["splits"][default_metric]["test"]["dir"]
        try:
            import shutil
            for fn in ["y_prob.npy", "y_true.npy", "class_names.json", "metrics.json", "cm.png", "roc.png", "prc.png"]:
                sp = os.path.join(src_dir, fn)
                if os.path.exists(sp):
                    shutil.copy2(sp, os.path.join(out_dir, fn))
        except Exception as e:
            print(f"[Warning] copy export to root failed: {e}")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # wandb finalize：记录三个 best 模型的结果
    if wb_run is not None:
        try:
            final_payload = {"epoch": ep}
            for metric_name, best_info in best_metrics.items():
                if metric_name in summary["splits"]:
                    if "val" in summary["splits"][metric_name]:
                        final_payload.update(flatten_metrics_for_wandb(f"val_best_{metric_name}", summary["splits"][metric_name]["val"].get("metrics", {})))
                    if "test" in summary["splits"][metric_name]:
                        final_payload.update(flatten_metrics_for_wandb(f"test_best_{metric_name}", summary["splits"][metric_name]["test"].get("metrics", {})))
            wandb.log(final_payload)
            # 保存三个 best 模型为 artifact
            for metric_name, best_info in best_metrics.items():
                if os.path.exists(best_info["path"]):
                    art = wandb.Artifact(name=f"{wb_run.name}-best_{metric_name}", type="model")
                    art.add_file(best_info["path"])
                    wb_run.log_artifact(art)
        except Exception as e:
            print(f"[Warning] wandb finalize failed: {e}")
        wb_run.finish()

    print("\n[Best models saved]")
    for metric_name, best_info in best_metrics.items():
        if os.path.exists(best_info["path"]):
            print(f"  {metric_name}: {best_info['path']} (value={best_info['value']:.6f})")
    print(f"\nSaved exports under: {out_dir}/")
    print("  - val_best_val_loss/, val_best_val_macro_f1/, val_best_val_accuracy/")
    if dl_test is not None:
        print("  - test_best_val_loss/, test_best_val_macro_f1/, test_best_val_accuracy/")


if __name__ == "__main__":
    main()


