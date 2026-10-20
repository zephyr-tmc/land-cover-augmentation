# augment.py
import math, torch
import numpy as np

def mixup_batch(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mix, y_a, y_b, lam

def cutmix_batch(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = x.size()
    idx = torch.randperm(B, device=x.device)
    bbx1, bby1, bbx2, bby2 = _rand_bbox(W, H, lam)
    x2 = x[idx]
    x[:, :, bby1:bby2, bbx1:bbx2] = x2[:, :, bby1:bby2, bbx1:bbx2]
    # 重算 lam（按面积）
    lam_adj = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[idx]
    return x, y_a, y_b, lam_adj

def _rand_bbox(W, H, lam):
    cut_w = int(W * math.sqrt(1 - lam))
    cut_h = int(H * math.sqrt(1 - lam))
    cx = np.random.randint(W); cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2
