import os, json, argparse, random, numpy as np
from typing import Optional
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import make_dataloaders
from model import build_model
from augment import cutmix_batch


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dl, device, criterion, optimizer, alpha: float, max_steps: Optional[int], scaler, clip_grad_norm: float):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    steps = 0
    for x, y in dl:
        x = x.to(device); y = y.to(device)
        x, y_a, y_b, lam = cutmix_batch(x, y, alpha)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(x)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            preds = logits.argmax(1)
            major = y_a if lam >= 0.5 else y_b
            correct = (preds == major).sum().item()
        if scaler is not None:
            scaler.scale(loss).backward()
            if clip_grad_norm and clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        bsz = x.size(0)
        total_loss += loss.item() * bsz; total_correct += correct; total += bsz
        steps += 1
        if max_steps and steps >= max_steps: break
    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def evaluate(model, dl, device, criterion):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in dl:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(1)
        bsz = x.size(0)
        total_loss += loss.item() * bsz
        total_correct += (preds == y).sum().item()
        total += bsz
    return total_loss / max(total, 1), total_correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train with CutMix")
    default_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_data_root = os.path.join(default_project_dir, "dataset_cls")
    if not os.path.isdir(default_data_root):
        default_data_root = os.path.join(default_project_dir, "images")

    parser.add_argument("--data-root", type=str, default=default_data_root)
    parser.add_argument("--layout", type=str, default="auto", choices=["auto", "split", "subdir"])
    parser.add_argument("--train-subdir", type=str, default="org286")
    parser.add_argument("--val-subdir", type=str, default="val286")
    parser.add_argument("--test-subdir", type=str, default=None)
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--test-dir", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=286)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="smallcnn", choices=["smallcnn", "resnet18"])
    parser.add_argument("--aug-loader", type=str, default="preprocess_only", choices=["preprocess_only", "traditional"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--out-dir", type=str, default=os.path.join(default_project_dir, "outputs_cutmix"))
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--clip-grad-norm", type=float, default=0.0)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--extra-train-dirs", type=str, default=None)

    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    else:
        train_dir = data_root; val_dir = data_root
        test_dir  = data_root if args.test_subdir else None
        train_only_with_subdir = args.train_subdir
        val_only_with_subdir   = args.val_subdir
        test_only_with_subdir  = args.test_subdir

    extra_train_dirs = None
    if args.extra_train_dirs:
        extra_train_dirs = [p.strip() for p in args.extra_train_dirs.split(",") if p.strip()]

    dl_train, dl_val, dl_test, num_classes, class_names, class_weights, counts = make_dataloaders(
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
        extra_train_dirs=extra_train_dirs,
        use_weighted_sampler=args.weighted_sampler,
    )

    model = build_model(num_classes, backbone=args.backbone).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=args.label_smoothing)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    out_dir = os.path.abspath(args.out_dir); os.makedirs(out_dir, exist_ok=True)
    best_val = float('inf')
    best_path = os.path.join(out_dir, f"best_{args.backbone}_{args.aug_loader}_cutmix.pt")

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, device, criterion, optimizer, args.alpha, args.max_steps, scaler, args.clip_grad_norm)
        val_loss, val_acc = evaluate(model, dl_val, device, criterion)
        scheduler.step(val_loss)
        print(f"[{ep:03d}] train {tr_loss:.4f}/{tr_acc:.4f}  val {val_loss:.4f}/{val_acc:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "class_names": class_names, "args": vars(args)}, best_path)

    export_loader = dl_test if dl_test is not None else dl_val
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in export_loader:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
            all_probs.append(probs); all_labels.append(y.numpy())
    import numpy as np
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    np.save(os.path.join(out_dir, "y_prob.npy"), y_prob)
    np.save(os.path.join(out_dir, "y_true.npy"), y_true)
    with open(os.path.join(out_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, ensure_ascii=False)
    print("Saved predictions for ROC/PRC.")


if __name__ == "__main__":
    main()


