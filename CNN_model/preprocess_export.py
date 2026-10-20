import os
import argparse
from typing import List, Optional, Tuple

from PIL import Image
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pil_resize(img: Image.Image, size: int) -> Image.Image:
    # 兼容 Pillow 版本差异
    try:
        resample = Image.Resampling.LANCZOS  # Pillow >= 9.1
    except Exception:
        resample = Image.LANCZOS
    return img.resize((size, size), resample=resample)


def center_crop_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


## 去黑边功能已按需求移除


def deterministic_preprocess(
    img: Image.Image,
    img_size: int,
) -> Image.Image:
    # 1) 统一色彩空间为 RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    # 2) 中心裁成正方形，再缩放到 img_size（确定性）
    img = center_crop_to_square(img)
    img = _pil_resize(img, img_size)
    return img


def iter_split_structure(src_root: str) -> List[Tuple[str, str, str]]:
    """遍历 split 布局：返回 (split, class_name, src_path) 列表"""
    out = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(src_root, split)
        if not os.path.isdir(split_dir):
            continue
        for cname in sorted(os.listdir(split_dir)):
            cdir = os.path.join(split_dir, cname)
            if not os.path.isdir(cdir):
                continue
            for fname in sorted(os.listdir(cdir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
                    out.append((split, cname, os.path.join(cdir, fname)))
    return out


def iter_subdir_structure(src_root: str, subdirs: List[str]) -> List[Tuple[str, str, str]]:
    """遍历 subdir 布局：返回 (subdir, class_name, src_path) 列表"""
    out = []
    for cname in sorted(os.listdir(src_root)):
        cdir = os.path.join(src_root, cname)
        if not os.path.isdir(cdir):
            continue
        for sub in subdirs:
            sdir = os.path.join(cdir, sub)
            if not os.path.isdir(sdir):
                continue
            for fname in sorted(os.listdir(sdir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
                    out.append((sub, cname, os.path.join(sdir, fname)))
    return out


def main():
    parser = argparse.ArgumentParser(description="Export deterministic preprocessed images to a new ImageFolder directory")
    default_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_src_root = os.path.join(default_project_dir, "dataset_cls")
    default_dst_root = os.path.join(default_project_dir, "dataset_preprocessed")

    parser.add_argument("--src-root", type=str, default=default_src_root, help="源数据根目录")
    parser.add_argument("--dst-root", type=str, default=default_dst_root, help="输出数据根目录（将创建镜像结构）")
    parser.add_argument("--layout", type=str, default="auto", choices=["auto", "split", "subdir"], help="目录布局")
    parser.add_argument("--subdirs", type=str, default="org286,val286", help="layout=subdir 时要导出的子目录，逗号分隔")
    parser.add_argument("--img-size", type=int, default=286)
    parser.add_argument("--ext", type=str, default=".png", choices=[".png", ".jpg", ".jpeg"], help="输出格式")
    parser.add_argument("--skip-existing", action="store_true", help="若目标文件已存在则跳过")

    args = parser.parse_args()

    src_root = os.path.abspath(args.src_root)
    dst_root = os.path.abspath(args.dst_root)
    ensure_dir(dst_root)

    layout = args.layout
    if layout == "auto":
        if os.path.isdir(os.path.join(src_root, "train")) and os.path.isdir(os.path.join(src_root, "val")):
            layout = "split"
        else:
            layout = "subdir"

    if layout == "split":
        items = iter_split_structure(src_root)
    else:
        subdirs = [s.strip() for s in args.subdirs.split(",") if s.strip()]
        items = iter_subdir_structure(src_root, subdirs)

    total = len(items)
    print(f"Found {total} images to process. layout={layout}")

    num_ok, num_fail, num_skip = 0, 0, 0
    for idx, (first_level, cname, src_path) in enumerate(items, start=1):
        try:
            with Image.open(src_path) as im:
                im = deterministic_preprocess(im, args.img_size)

            # 目标路径
            if layout == "split":
                dst_dir = os.path.join(dst_root, first_level, cname)
            else:
                # subdir 布局：保持 <dst_root>/<class>/<subdir>/
                dst_dir = os.path.join(dst_root, cname, first_level)
            ensure_dir(dst_dir)

            base = os.path.splitext(os.path.basename(src_path))[0]
            dst_path = os.path.join(dst_dir, base + args.ext)
            if args.skip_existing and os.path.isfile(dst_path):
                num_skip += 1
                if idx % 500 == 0:
                    print(f"[{idx}/{total}] skipped")
                continue

            # PNG/JPEG 默认不带 EXIF（PNG 无 EXIF），此处不写入任何元信息
            save_params = {}
            if args.ext.lower() in (".jpg", ".jpeg"):
                save_params["quality"] = 95
                save_params["optimize"] = True
            im.save(dst_path, **save_params)
            num_ok += 1
        except Exception as e:
            num_fail += 1
            if idx % 200 == 0:
                print(f"[{idx}/{total}] error: {src_path} -> {e}")

        if idx % 1000 == 0:
            print(f"Progress {idx}/{total} | ok={num_ok} skip={num_skip} fail={num_fail}")

    print(f"Done. ok={num_ok}, skip={num_skip}, fail={num_fail}")


if __name__ == "__main__":
    main()


