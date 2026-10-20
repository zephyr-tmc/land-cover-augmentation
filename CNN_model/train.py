# train.py
import os, json, time, math, random, argparse, numpy as np
from typing import Optional, List
from collections import defaultdict
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import make_dataloaders
from model import build_model

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class ModelEma:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        import copy
        self.decay = decay
        self.ema = copy.deepcopy(model).to(next(model.parameters()).device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for (n, p), (_, p_ema) in zip(model.state_dict().items(), self.ema.state_dict().items()):
            if p.dtype.is_floating_point:
                p_ema.mul_(d).add_(p.detach(), alpha=1.0 - d)
            else:
                p_ema.copy_(p)

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience; self.min_delta = min_delta
        self.best = float('inf'); self.count = 0
    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss; self.count = 0; return False
        self.count += 1
        return self.count > self.patience

# 旧实现包含 aug_mode/alpha，这里整体替换为“无在线增强”的瘦身版本
def train_one_epoch(
    model,
    dl,
    device,
    criterion,
    optimizer,
    max_steps: Optional[int],
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    clip_grad_norm: float = 0.0,
    ema: Optional[ModelEma] = None,
):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    # 按类累计（便于观测每类）
    per_class_loss_sum = None  # numpy later
    per_class_count = None
    steps = 0
    for x, y in dl:
        x = x.to(device); y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(x)
            loss = criterion(logits, y)
            preds = logits.argmax(1)
            correct = (preds == y).sum().item()

        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_grad_norm and clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad_norm and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        bsz = x.size(0)
        total_loss += loss.item() * bsz
        total_correct += correct
        total += bsz

        # 逐类 loss/计数（按样本平均）
        with torch.no_grad():
            num_classes = logits.size(1)
            if per_class_loss_sum is None:
                import numpy as _np
                per_class_loss_sum = _np.zeros((num_classes,), dtype=_np.float64)
                per_class_count = _np.zeros((num_classes,), dtype=_np.int64)
            # 按样本逐个计算其交叉熵损失并归入对应真实类
            # 为避免重复前向，使用已算的 logits：取真实标签的负 log softmax
            log_probs = torch.log_softmax(logits.detach(), dim=1)
            sample_loss = -log_probs[torch.arange(bsz, device=log_probs.device), y]
            import numpy as _np
            for cls in y.unique().tolist():
                mask = (y == cls)
                cls_loss = sample_loss[mask].mean().item() if mask.any() else 0.0
                per_class_loss_sum[cls] += cls_loss * int(mask.sum().item())
                per_class_count[cls] += int(mask.sum().item())
        steps += 1
        if max_steps and steps >= max_steps:
            break

    # 将逐类统计随返回一起给出（均值）
    per_class_loss_avg = None
    if per_class_loss_sum is not None and per_class_count is not None:
        import numpy as _np
        with _np.errstate(divide='ignore', invalid='ignore'):
            per_class_loss_avg = _np.divide(per_class_loss_sum, _np.maximum(per_class_count, 1)).tolist()

    return total_loss / max(total, 1), total_correct / max(total, 1), per_class_loss_avg, per_class_count.tolist() if per_class_count is not None else None

@torch.no_grad()
def evaluate(model, dl, device, criterion):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    per_class_loss_sum = None
    per_class_count = None
    for x, y in dl:
        x = x.to(device); y = y.to(device)
        with torch.cuda.amp.autocast(enabled=False):
            logits = model(x)
            loss = criterion(logits, y)
        preds = logits.argmax(1)
        bsz = x.size(0)
        total_loss += loss.item() * bsz
        total_correct += (preds == y).sum().item()
        total += bsz
        # 逐类 loss/计数
        with torch.no_grad():
            num_classes = logits.size(1)
            if per_class_loss_sum is None:
                import numpy as _np
                per_class_loss_sum = _np.zeros((num_classes,), dtype=_np.float64)
                per_class_count = _np.zeros((num_classes,), dtype=_np.int64)
            log_probs = torch.log_softmax(logits.detach(), dim=1)
            sample_loss = -log_probs[torch.arange(bsz, device=log_probs.device), y]
            import numpy as _np
            for cls in y.unique().tolist():
                mask = (y == cls)
                cls_loss = sample_loss[mask].mean().item() if mask.any() else 0.0
                per_class_loss_sum[cls] += cls_loss * int(mask.sum().item())
                per_class_count[cls] += int(mask.sum().item())
    per_class_loss_avg = None
    if per_class_loss_sum is not None and per_class_count is not None:
        import numpy as _np
        with _np.errstate(divide='ignore', invalid='ignore'):
            per_class_loss_avg = _np.divide(per_class_loss_sum, _np.maximum(per_class_count, 1)).tolist()
    return total_loss / max(total, 1), total_correct / max(total, 1), per_class_loss_avg, per_class_count.tolist() if per_class_count is not None else None

def main():
    parser = argparse.ArgumentParser(description="Train CNN land-cover classifier")
    default_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # 尝试默认指向 dataset_cls（若存在），否则 images
    default_data_root = os.path.join(default_project_dir, "dataset_cls")
    if not os.path.isdir(default_data_root):
        default_data_root = os.path.join(default_project_dir, "images")

    parser.add_argument("--data-root", type=str, default=default_data_root,
                        help="数据根目录。支持两种布局：1) split: root/{train,val,test}/{class} 2) subdir: root/{class}/{org286,val286}")
    parser.add_argument("--layout", type=str, default="auto", choices=["auto", "split", "subdir"],
                        help="数据布局。auto 会自动检测是否存在 train/val/test 目录")
    parser.add_argument("--train-subdir", type=str, default="org286",
                        help="当 layout=subdir 时用于训练的子目录名（如 org286）")
    parser.add_argument("--val-subdir", type=str, default="val286",
                        help="当 layout=subdir 时用于验证的子目录名（如 val286）")
    parser.add_argument("--test-subdir", type=str, default=None,
                        help="当 layout=subdir 时用于测试的子目录名（可选）")
    parser.add_argument("--train-dir", type=str, default=None, help="当 layout=split 时，显式指定 train 目录")
    parser.add_argument("--val-dir", type=str, default=None, help="当 layout=split 时，显式指定 val 目录")
    parser.add_argument("--test-dir", type=str, default=None, help="当 layout=split 时，显式指定 test 目录（可选）")
    parser.add_argument("--img-size", type=int, default=286)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="smallcnn", choices=["smallcnn", "resnet18"])
    parser.add_argument("--aug-loader", type=str, default="preprocess_only", choices=["preprocess_only", "traditional"],
                        help="Loader 端图像增强策略（训练：preprocess_only/traditional；验证/测试固定 preprocess_only）")
    parser.add_argument("--max-steps", type=int, default=None, help="每 epoch 最大步数（等预算可用）")
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--out-dir", type=str, default=os.path.join(default_project_dir, "outputs"))
    parser.add_argument("--use-amp", action="store_true", help="启用自动混合精度")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="交叉熵标签平滑")
    parser.add_argument("--clip-grad-norm", type=float, default=0.0, help="梯度裁剪上限，0 表示不裁剪")
    parser.add_argument("--ema", action="store_true", help="启用模型 EMA")
    parser.add_argument("--weighted-sampler", action="store_true", help="训练集使用加权随机采样以平衡类别")
    parser.add_argument("--extra-train-dirs", type=str, default=None,
                        help="以逗号分隔的额外训练目录（如 GAN/DDPM 生成数据的 train 根目录）")

    args = parser.parse_args()

    set_seed(42)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    # ====== 路径 & 基本参数 ======
    data_root = os.path.abspath(args.data_root)
    layout = args.layout
    if layout == "auto":
        if os.path.isdir(os.path.join(data_root, "train")) and os.path.isdir(os.path.join(data_root, "val")):
            layout = "split"
        else:
            layout = "subdir"

    if layout == "split":
        train_dir = args.train_dir or os.path.join(data_root, "train")
        val_dir   = args.val_dir   or os.path.join(data_root, "val")
        test_dir  = args.test_dir  or (os.path.join(data_root, "test") if os.path.isdir(os.path.join(data_root, "test")) else None)
        train_only_with_subdir = None
        val_only_with_subdir   = None
        test_only_with_subdir  = None
    else:  # subdir
        train_dir = data_root
        val_dir   = data_root
        test_dir  = data_root if args.test_subdir else None
        train_only_with_subdir = args.train_subdir
        val_only_with_subdir   = args.val_subdir
        test_only_with_subdir  = args.test_subdir

    img_size   = args.img_size
    batch_size = args.batch_size
    epochs     = args.epochs
    lr         = args.lr
    backbone   = args.backbone
    aug_loader_mode = args.aug_loader
    max_steps_per_epoch = args.max_steps
    num_workers = args.num_workers
    use_amp = args.use_amp
    label_smoothing = args.label_smoothing
    clip_grad_norm = args.clip_grad_norm
    use_ema = args.ema
    use_weighted_sampler = args.weighted_sampler
    extra_train_dirs = None
    if args.extra_train_dirs:
        extra_train_dirs = [p.strip() for p in args.extra_train_dirs.split(",") if p.strip()]

    # ====== DataLoader ======
    dl_train, dl_val, dl_test, num_classes, class_names, class_weights, counts = make_dataloaders(
        root_train=train_dir,
        root_val=val_dir,
        root_test=test_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        mode=aug_loader_mode,
        train_only_with_subdir=train_only_with_subdir,
        val_only_with_subdir=val_only_with_subdir,
        test_only_with_subdir=test_only_with_subdir,
        extra_train_dirs=extra_train_dirs,
        use_weighted_sampler=use_weighted_sampler,
    )

    # ====== 类别权重（Weighted CE + √反比 + 均值=1）======
    # counts 由 make_dataloaders 返回，是按 class index 的训练集计数列表
    counts_t = torch.tensor(counts, dtype=torch.float32).clamp(min=1)
    K = len(counts_t)
    # √反比
    w = (counts_t.sum() / K) / counts_t.sqrt()
    # 归一化到“均值=1”，避免整体损失尺度改变导致学习率不稳
    w = w * (K / w.sum())

    print("[Class counts]:", counts)
    print("[Weighted CE sqrt_inv weights (mean=1)]:", [round(v, 4) for v in w.tolist()])

    # ====== 模型/损失/优化 ======
    model = build_model(num_classes, backbone=backbone).to(device)
    # 训练集：加权；验证/测试：不加权（更可比）
    criterion_tr = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=label_smoothing)
    criterion_va = nn.CrossEntropyLoss(weight=None,         label_smoothing=label_smoothing)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early = EarlyStopper(patience=10, min_delta=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = ModelEma(model, decay=0.999) if use_ema else None


    # ====== 训练 ======
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    log = []

    best_val = float('inf')
    best_path = os.path.join(out_dir, f"best_{backbone}_{aug_loader_mode}.pt")

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, tr_cls_loss, tr_cls_cnt = train_one_epoch(
            model, dl_train, device, criterion_tr, optimizer,
            max_steps_per_epoch,
            scaler=scaler, clip_grad_norm=clip_grad_norm, ema=ema
        )
        model_for_eval = ema.ema if ema is not None else model
        val_loss, val_acc, va_cls_loss, va_cls_cnt = evaluate(model_for_eval, dl_val, device, criterion_va)
        scheduler.step(val_loss)

        log.append({
            "epoch": ep,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": val_loss,   "val_acc": val_acc,
            "train_class_loss": tr_cls_loss,
            "train_class_count": tr_cls_cnt,
            "val_class_loss": va_cls_loss,
            "val_class_count": va_cls_cnt,
            "class_names": class_names,
        })
        print(f"[{ep:03d}] train {tr_loss:.4f}/{tr_acc:.4f}  val {val_loss:.4f}/{val_acc:.4f}")
        # 可选：打印每类 loss（前几类预览）
        if tr_cls_loss is not None:
            preview = ", ".join([f"{class_names[i]}:{tr_cls_loss[i]:.3f}" for i in range(min(len(class_names), 4))])
            print("  per-class train loss (head):", preview)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": (model_for_eval.state_dict() if ema is not None else model.state_dict()),
                "class_names": class_names,
                "args": vars(args),
            }, best_path)

        if early.step(val_loss):
            print("Early stopping.")
            break

    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("Best model saved to:", best_path)

    # ====== 概率导出（优先测试集，如无则导出验证集） ======
    export_name = None
    export_loader = None
    if dl_test is not None:
        export_name = "test"
        export_loader = dl_test
    else:
        export_name = "val"
        export_loader = dl_val

    if export_loader is not None:
        ckpt = torch.load(best_path, map_location=device)
        (ema.ema if ema is not None else model).load_state_dict(ckpt["model"])
        model_for_eval = ema.ema if ema is not None else model
        model_for_eval.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x, y in export_loader:
                x = x.to(device)
                logits = model_for_eval(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(y.numpy())
        import numpy as np
        y_prob = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_labels, axis=0)
        np.save(os.path.join(out_dir, "y_prob.npy"), y_prob)
        np.save(os.path.join(out_dir, "y_true.npy"), y_true)
        with open(os.path.join(out_dir, "class_names.json"), "w") as f:
            json.dump(class_names, f, ensure_ascii=False)
        print(f"Saved {export_name} predictions for ROC/PRC.")
        
if __name__ == "__main__":
    main()
