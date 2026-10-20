import os
import math
import argparse
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils


def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_is_valid_file(only_with_subdir: Optional[str]):
    if not only_with_subdir:
        return None
    allowed_exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")
    sub = f"/{only_with_subdir.strip('/')}/"

    def _predicate(path: str) -> bool:
        p = path.replace("\\", "/")
        return p.lower().endswith(allowed_exts) and (sub in p)

    return _predicate


class SimpleUNet(nn.Module):
    def __init__(self, num_classes: int, base: int = 64, img_size: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        ch = base
        # 时间嵌入 + 类别嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, ch), nn.SiLU(), nn.Linear(ch, ch)
        )
        self.label_emb = nn.Embedding(num_classes, ch)

        self.down1 = nn.Sequential(nn.Conv2d(3, ch, 3, 1, 1), nn.SiLU(), nn.Conv2d(ch, ch, 3, 1, 1), nn.SiLU())
        self.down2 = nn.Sequential(nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.SiLU(), nn.Conv2d(ch * 2, ch * 2, 3, 1, 1), nn.SiLU())
        self.down3 = nn.Sequential(nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.SiLU(), nn.Conv2d(ch * 4, ch * 4, 3, 1, 1), nn.SiLU())
        self.mid   = nn.Sequential(nn.Conv2d(ch * 4, ch * 4, 3, 1, 1), nn.SiLU(), nn.Conv2d(ch * 4, ch * 4, 3, 1, 1), nn.SiLU())
        self.up3   = nn.Sequential(nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1), nn.SiLU(), nn.Conv2d(ch * 2, ch * 2, 3, 1, 1), nn.SiLU())
        self.up2   = nn.Sequential(nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.SiLU(), nn.Conv2d(ch, ch, 3, 1, 1), nn.SiLU())
        self.out   = nn.Conv2d(ch, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # t 归一化到 [0,1]，作为时间嵌入
        t = t.float().view(-1, 1) / 1000.0
        h_t = self.time_mlp(t)  # (B, ch)
        h_y = self.label_emb(y)  # (B, ch)
        cond = (h_t + h_y).unsqueeze(-1).unsqueeze(-1)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        m  = self.mid(d3 + cond)
        u3 = self.up3(m)
        u2 = self.up2(u3 + d2)
        out = self.out(u2 + d1)
        return out


class DDPM:
    def __init__(self, model: nn.Module, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device=None):
        self.model = model
        self.timesteps = timesteps
        self.device = device or next(model.parameters()).device
        betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = betas * (1.0 - torch.roll(self.alphas_cumprod, 1)) / (1.0 - self.alphas_cumprod)
        self.posterior_variance[0] = betas[0]

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x0 + sqrt_om * noise

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        betas_t = torch.linspace(1e-4, 0.02, self.timesteps, device=x.device)[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_cumprod_t = (1.0 / self.sqrt_alphas_cumprod[t]).view(-1, 1, 1, 1)
        model_out = self.model(x, t, y)
        x0_pred = (x - sqrt_one_minus_alphas_cumprod_t * model_out) * sqrt_recip_alphas_cumprod_t
        if (t == 0).all():
            return x0_pred
        var = self.posterior_variance[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        return x0_pred + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, shape: tuple[int, int, int, int], y: torch.Tensor) -> torch.Tensor:
        x = torch.randn(shape, device=self.device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, dtype=torch.long, device=self.device)
            x = self.p_sample(x, t, y)
        return x


def build_dataloader(data_root: str, layout: str, img_size: int, batch_size: int, num_workers: int,
                     train_subdir: str) -> tuple[DataLoader, list[str]]:
    if layout == "split":
        root = os.path.join(data_root, "train")
        is_valid = None
    else:
        root = data_root
        is_valid = _build_is_valid_file(train_subdir)

    tfm = transforms.Compose([
        transforms.Resize(img_size, antialias=True),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1,1]
    ])
    ds = datasets.ImageFolder(root=root, transform=tfm, is_valid_file=is_valid)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dl, ds.classes


def train(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = os.path.abspath(args.data_root)
    layout = args.layout
    if layout == "auto":
        if os.path.isdir(os.path.join(data_root, "train")) and os.path.isdir(os.path.join(data_root, "val")):
            layout = "split"
        else:
            layout = "subdir"

    dl, class_names = build_dataloader(
        data_root=data_root,
        layout=layout,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_subdir=args.train_subdir,
    )

    model = SimpleUNet(num_classes=len(class_names), base=args.base, img_size=args.img_size).to(device)
    ddpm = DDPM(model, timesteps=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    mse = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in dl:
            x = x.to(device); y = y.to(device)
            bsz = x.size(0)
            t = torch.randint(0, ddpm.timesteps, (bsz,), device=device)
            noise = torch.randn_like(x)
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                x_noisy = ddpm.q_sample(x, t, noise)
                pred = model(x_noisy, t, y)
                loss = mse(pred, noise)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # 预览
        model.eval()
        with torch.no_grad():
            grid_list = []
            for cid in range(min(len(class_names), 4)):
                yb = torch.full((8,), cid, dtype=torch.long, device=device)
                imgs = ddpm.sample((8, 3, args.img_size, args.img_size), yb)
                imgs = (imgs.clamp(-1, 1) + 1) / 2.0
                grid_list.append(imgs)
            if grid_list:
                preview = torch.cat(grid_list, dim=0)
                os.makedirs(args.out_dir, exist_ok=True)
                vutils.save_image(preview, os.path.join(args.out_dir, f"preview_epoch_{epoch:03d}.png"), nrow=8)

    # 保存
    torch.save({
        "model": model.state_dict(),
        "class_names": class_names,
        "args": vars(args),
    }, os.path.join(args.out_dir, "ddpm.pt"))

    # 采样到 ImageFolder 结构
    if args.samples_per_class > 0:
        sample_only(args)


@torch.no_grad()
def sample_only(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.resume, map_location=device)
    class_names = ckpt["class_names"]
    model = SimpleUNet(num_classes=len(class_names), base=args.base, img_size=args.img_size).to(device)
    model.load_state_dict(ckpt["model"])
    ddpm = DDPM(model, timesteps=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end, device=device)
    model.eval()

    save_root = os.path.join(args.out_dir, "samples")
    os.makedirs(save_root, exist_ok=True)
    for cid, cname in enumerate(class_names):
        save_dir = os.path.join(save_root, cname)
        os.makedirs(save_dir, exist_ok=True)
        remaining = args.samples_per_class
        idx = 0
        while remaining > 0:
            cur = min(32, remaining)
            yb = torch.full((cur,), cid, dtype=torch.long, device=device)
            imgs = ddpm.sample((cur, 3, args.img_size, args.img_size), yb)
            imgs = (imgs.clamp(-1, 1) + 1) / 2.0
            for i in range(cur):
                vutils.save_image(imgs[i], os.path.join(save_dir, f"{idx:06d}.png"))
                idx += 1
            remaining -= cur


def main():
    parser = argparse.ArgumentParser(description="Class-conditional DDPM (lightweight)")
    default_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_data_root = os.path.join(default_project_dir, "dataset_cls")
    if not os.path.isdir(default_data_root):
        default_data_root = os.path.join(default_project_dir, "images")

    parser.add_argument("--data-root", type=str, default=default_data_root)
    parser.add_argument("--layout", type=str, default="auto", choices=["auto", "split", "subdir"])
    parser.add_argument("--train-subdir", type=str, default="org286")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--out-dir", type=str, default=os.path.join(default_project_dir, "generated_ddpm"))
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None, help="若提供，将从此 ckpt 进行采样")

    args = parser.parse_args()

    if args.resume:
        sample_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()


