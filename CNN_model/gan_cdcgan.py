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


class Generator(nn.Module):
    def __init__(self, nz: int, num_classes: int, ngf: int = 64, img_size: int = 128):
        super().__init__()
        self.nz = nz
        self.num_classes = num_classes
        self.img_size = img_size
        self.label_emb = nn.Embedding(num_classes, nz)

        start_size = img_size // 16  # 4 for 64, 8 for 128
        assert img_size % 16 == 0, "img_size must be divisible by 16"

        self.net = nn.Sequential(
            # input Z cat label: (nz*2) x 1 x 1
            nn.ConvTranspose2d(nz * 2, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # z: (B, nz, 1, 1) ; y: (B,)
        y_emb = self.label_emb(y)  # (B, nz)
        y_emb = y_emb.unsqueeze(2).unsqueeze(3)  # (B, nz, 1, 1)
        h = torch.cat([z, y_emb], dim=1)
        return self.net(h)


class Discriminator(nn.Module):
    def __init__(self, num_classes: int, ndf: int = 64, img_size: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        # 将标签嵌入为 1xHxW 的条件面，拼在图像通道上
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)

        self.net = nn.Sequential(
            # input is (3+1) x H x W
            nn.Conv2d(4, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        label_map = self.label_emb(y).view(b, 1, self.img_size, self.img_size)
        if h != self.img_size or w != self.img_size:
            label_map = torch.nn.functional.interpolate(label_map, size=(h, w), mode="nearest")
        x_in = torch.cat([x, label_map], dim=1)
        out = self.net(x_in).view(-1)
        return out


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


def save_samples(G: Generator, device: torch.device, class_names: list[str], nz: int, out_dir: str,
                 samples_per_class: int, img_size: int):
    G.eval()
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for cid, cname in enumerate(class_names):
            save_dir = os.path.join(out_dir, cname)
            os.makedirs(save_dir, exist_ok=True)
            num = samples_per_class
            bs = 64
            saved = 0
            while saved < num:
                cur = min(bs, num - saved)
                z = torch.randn(cur, nz, 1, 1, device=device)
                y = torch.full((cur,), cid, dtype=torch.long, device=device)
                fake = G(z, y)
                # 反归一化到 [0,1]
                imgs = (fake.clamp(-1, 1) + 1) / 2.0
                for i in range(cur):
                    idx = saved + i
                    path = os.path.join(save_dir, f"{idx:06d}.png")
                    vutils.save_image(imgs[i], path)
                saved += cur


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
    num_classes = len(class_names)

    G = Generator(nz=args.nz, num_classes=num_classes, ngf=args.ngf, img_size=args.img_size).to(device)
    D = Discriminator(num_classes=num_classes, ndf=args.ndf, img_size=args.img_size).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    real_label = 0.9
    fake_label = 0.1

    os.makedirs(args.out_dir, exist_ok=True)
    sample_dir = os.path.join(args.out_dir, "preview")
    os.makedirs(sample_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        for i, (x, y) in enumerate(dl, start=1):
            x = x.to(device); y = y.to(device)
            bsz = x.size(0)

            # 训练 D
            optD.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                out_real = D(x, y)
                label_real = torch.full((bsz,), real_label, device=device)
                loss_real = criterion(out_real, label_real)

                z = torch.randn(bsz, args.nz, 1, 1, device=device)
                with torch.no_grad():
                    fake = G(z, y)
                out_fake = D(fake, y)
                label_fake = torch.full((bsz,), fake_label, device=device)
                loss_fake = criterion(out_fake, label_fake)
                loss_D = loss_real + loss_fake

            scaler.scale(loss_D).backward()
            scaler.step(optD)

            # 训练 G
            optG.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                z = torch.randn(bsz, args.nz, 1, 1, device=device)
                fake = G(z, y)
                out = D(fake, y)
                label_g = torch.full((bsz,), real_label, device=device)
                loss_G = criterion(out, label_g)
            scaler.scale(loss_G).backward()
            scaler.step(optG)
            scaler.update()

        # 每个 epoch 存一点预览
        G.eval()
        with torch.no_grad():
            grid_list = []
            for cid in range(min(num_classes, 8)):
                z = torch.randn(8, args.nz, 1, 1, device=device)
                yb = torch.full((8,), cid, dtype=torch.long, device=device)
                fake = G(z, yb)
                imgs = (fake.clamp(-1, 1) + 1) / 2.0
                grid_list.append(imgs)
            if grid_list:
                preview = torch.cat(grid_list, dim=0)
                vutils.save_image(preview, os.path.join(sample_dir, f"epoch_{epoch:03d}.png"), nrow=8)

    # 训练完成保存权重
    torch.save({
        "G": G.state_dict(),
        "class_names": class_names,
        "args": vars(args),
    }, os.path.join(args.out_dir, "generator.pt"))

    # 采样到 ImageFolder 结构：out_dir/samples/<class>/*.png
    if args.samples_per_class > 0:
        save_root = os.path.join(args.out_dir, "samples")
        save_samples(G, device, class_names, args.nz, save_root, args.samples_per_class, args.img_size)


def sample_only(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.resume, map_location=device)
    class_names = ckpt["class_names"]
    num_classes = len(class_names)

    G = Generator(nz=args.nz, num_classes=num_classes, ngf=args.ngf, img_size=args.img_size).to(device)
    G.load_state_dict(ckpt["G"])
    save_root = os.path.join(args.out_dir, "samples")
    save_samples(G, device, class_names, args.nz, save_root, args.samples_per_class, args.img_size)


def main():
    parser = argparse.ArgumentParser(description="Conditional DCGAN for class-conditional generation")
    default_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_data_root = os.path.join(default_project_dir, "dataset_cls")
    if not os.path.isdir(default_data_root):
        default_data_root = os.path.join(default_project_dir, "images")

    parser.add_argument("--data-root", type=str, default=default_data_root)
    parser.add_argument("--layout", type=str, default="auto", choices=["auto", "split", "subdir"])
    parser.add_argument("--train-subdir", type=str, default="org286")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default=os.path.join(default_project_dir, "generated_gan"))
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="若提供，将直接从此生成器权重进行采样")

    args = parser.parse_args()

    if args.resume:
        sample_only(args)
    else:
        train(args)


if __name__ == "__main__":
    main()


