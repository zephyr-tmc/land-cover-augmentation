#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, random, math, sys
from pathlib import Path
from hashlib import md5
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS

def resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w == 0 or h == 0:
        return img
    if w < h:
        new_w = size
        new_h = int(round(h * size / w))
    else:
        new_h = size
        new_w = int(round(w * size / h))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return img

def augment_one(
    img: Image.Image,
    hflip: bool, vflip: bool,
    height_shift: float, width_shift: float,
    rotation_deg: float, shear_deg: float,
    zoom_min: float, zoom_max: float,
    rng: random.Random,
) -> Image.Image:
    # 1) flips（按 TRUE/TRUE 的语义，这里各自 p=0.5，若你希望总是翻转可改为 if hflip: img = ImageOps.mirror(img) 等）
    if hflip and rng.random() < 0.5:
        img = ImageOps.mirror(img)   # 水平翻转
    if vflip and rng.random() < 0.5:
        img = ImageOps.flip(img)     # 垂直翻转

    W, H = img.size

    # 2) 仿射参数（一次性完成：平移+缩放+旋切；旋转按你的配置为 0）
    #   translate 接受像素，shift 范围按比例给出
    tx = int(round((rng.uniform(-width_shift,  width_shift))  * W))
    ty = int(round((rng.uniform(-height_shift, height_shift)) * H))

    # 旋转
    angle = rng.uniform(-rotation_deg, rotation_deg) if rotation_deg > 0 else 0.0

    # 缩放（zoom）：默认解释为 [zoom_min, zoom_max]，推荐 [0.9, 1.0]（最多缩小10%）
    scale = rng.uniform(zoom_min, zoom_max)
    # torchvision 的 affine 里用的是 1/scale 还是 scale？这里直接传 scale（=缩放因子）

    # shear：同时对 x/y 轴各取一个角度（可能很激进，按你的 180 默认）
    if shear_deg > 0:
        shear_x = rng.uniform(-shear_deg, shear_deg)
        shear_y = rng.uniform(-shear_deg, shear_deg)
        shear = [shear_x, shear_y]
    else:
        shear = [0.0, 0.0]

    # 3) 应用 affine（采用双三次插值，超出画面区域以常数0填充）
    img = TF.affine(
        img, angle=angle, translate=[tx, ty], scale=scale, shear=shear,
        interpolation=TF.InterpolationMode.BICUBIC, fill=0
    )
    return img

def process_one(
    src_path: Path, dst_root: Path, in_root: Path, out_ext: str,
    img_size: int, per_image: int,
    hflip: bool, vflip: bool,
    height_shift: float, width_shift: float,
    rotation_deg: float, shear_deg: float,
    zoom_min: float, zoom_max: float,
    seed: int
) -> int:
    rel = src_path.relative_to(in_root)
    stem = rel.stem
    dst_dir = (dst_root / rel.parent)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 以文件名+全局seed派生一个稳定 RNG，保证可复现
    h = int(md5((str(rel) + f"#{seed}").encode("utf-8")).hexdigest(), 16) % (2**31-1)
    rng = random.Random(h)

    saved = 0
    try:
        with Image.open(src_path) as im0:
            im0 = im0.convert("RGB")
            for i in range(per_image):
                im = im0.copy()
                im = augment_one(
                    im, hflip, vflip,
                    height_shift, width_shift,
                    rotation_deg, shear_deg,
                    zoom_min, zoom_max, rng
                )
                # 保证与主数据同尺寸
                im = resize_center_crop(im, img_size)
                out_name = f"{stem}_aug{i+1:02d}{out_ext}"
                im.save(dst_dir / out_name, quality=95)
                saved += 1
    except Exception as e:
        print(f"[ERR] {src_path}: {e}", file=sys.stderr)

    return saved

def main():
    ap = argparse.ArgumentParser(description="Offline traditional augmentation and save to disk.")
    ap.add_argument("--in-root",    type=Path, required=True, help="输入根目录（建议=训练集根目录，如 dataset_cls/train 或 dataset_cls/<class>/org286 的上级）")
    ap.add_argument("--out-root",   type=Path, required=True, help="输出根目录（将镜像类别结构）")
    ap.add_argument("--img-size",   type=int, default=286, help="最终写盘前统一到的尺寸（默认286）")
    ap.add_argument("--per-image",  type=int, default=1, help="每张原图生成多少张增强图（100% 扩充=1）")
    ap.add_argument("--ext",        type=str, default=".png", help="输出图片扩展名（.png/.jpg 等）")
    ap.add_argument("--workers",    type=int, default=8, help="并发线程数")
    ap.add_argument("--seed",       type=int, default=20240819, help="全局随机种子（保证可复现）")

    # 传统增强参数（给出与你的表格一致的默认值）
    ap.add_argument("--hflip", action="store_true", help="允许水平翻转（TRUE）")
    ap.add_argument("--vflip", action="store_true", help="允许垂直翻转（TRUE）")
    ap.add_argument("--height-shift", type=float, default=0.3, help="相对高度平移范围（0.3=±30%）")
    ap.add_argument("--width-shift",  type=float, default=0.0, help="相对宽度平移范围（0=禁用）")
    ap.add_argument("--rotation",     type=float, default=0.0, help="旋转范围（度），0 表示禁用")
    ap.add_argument("--shear",        type=float, default=180.0, help="shear 范围（度），默认 180 很激进，谨慎使用")
    ap.add_argument("--zoom-min",     type=float, default=0.9, help="缩放下界（例如 0.9）")
    ap.add_argument("--zoom-max",     type=float, default=1.0, help="缩放上界（例如 1.0）")

    args = ap.parse_args()

    src_files = [p for p in args.in_root.rglob("*") if p.is_file() and is_image(p)]
    print(f"[Info] Found {len(src_files)} images under {args.in_root}")
    if not src_files:
        return

    done, saved_total = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                process_one, p, args.out_root, args.in_root, args.ext,
                args.img_size, args.per_image,
                args.hflip, args.vflip,
                args.height_shift, args.width_shift,
                args.rotation, args.shear,
                args.zoom_min, args.zoom_max,
                args.seed
            )
            for p in src_files
        ]
        for f in as_completed(futs):
            saved_total += f.result()
            done += 1
            if done % 200 == 0:
                print(f"[Progress] {done}/{len(src_files)}")

    print(f"[Done] Wrote {saved_total} augmented images to {args.out_root}")

if __name__ == "__main__":
    main()
