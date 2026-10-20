#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS

def resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    # 等价于 torchvision.transforms.Resize(size) + CenterCrop(size) 对方形图 512→286
    # Resize：若传入 int，按“短边= size”等比缩放；512×512→286×286
    # 方图下 CenterCrop(size) 无效果，但为了一致性保留
    w, h = img.size
    if w == 0 or h == 0:
        return img
    # 等比缩放，让短边=size
    if w < h:
        new_w = size
        new_h = int(round(h * size / w))
    else:
        new_h = size
        new_w = int(round(w * size / h))
    img = img.resize((new_w, new_h), Image.BICUBIC)
    # 中心裁剪到 size×size
    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return img

def process_one(src_path: Path, dst_root: Path, in_root: Path, out_ext: str, size: int) -> bool:
    rel = src_path.relative_to(in_root)
    dst_path = (dst_root / rel).with_suffix(out_ext)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            im = resize_center_crop(im, size)
            im.save(dst_path, quality=95)
        return True
    except Exception as e:
        print(f"[ERR] {src_path}: {e}", file=sys.stderr)
        return False

def main():
    ap = argparse.ArgumentParser(description="Offline preprocess: resize+center-crop to target size.")
    ap.add_argument("--in-root",  type=Path, required=True, help="输入根目录（按类别子目录组织）")
    ap.add_argument("--out-root", type=Path, required=True, help="输出根目录（将镜像类别结构）")
    ap.add_argument("--img-size", type=int, default=286, help="目标边长（默认286）")
    ap.add_argument("--ext",      type=str, default=".png", help="输出图片扩展名（.png/.jpg 等）")
    ap.add_argument("--workers",  type=int, default=8, help="并发线程数")
    args = ap.parse_args()

    src_files = [p for p in args.in_root.rglob("*") if p.is_file() and is_image(p)]
    print(f"[Info] Found {len(src_files)} images under {args.in_root}")
    if not src_files:
        return

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, p, args.out_root, args.in_root, args.ext, args.img_size) for p in src_files]
        for f in as_completed(futs):
            done += 1
            if done % 200 == 0:
                print(f"[Progress] {done}/{len(src_files)}")

    print(f"[Done] Wrote preprocessed images to {args.out_root}")

if __name__ == "__main__":
    main()
