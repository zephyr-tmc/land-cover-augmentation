#!/usr/bin/env python3
"""
LoRA Pipeline - Stage 4

Based on the previously generated lora_train.txt, call `accelerate launch train_network.py` to perform LoRA training,
and output a .ckpt format LoRA weight file, which is convenient for directly loading in the Stable Diffusion inference code. 
Note:
- This script is merely a "thin wrapper", the actual training logic still lies in `train_network.py` (kohya-ss / sd-scripts).
- You need to have the kohya environment installed beforehand and be able to run `accelerate launch train_network.py ...` in the command line. `。
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="LoRA Pipeline Stage4: train LoRA via kohya train_network.py")
    parser.add_argument(
        "--lora_train_file",
        type=str,
        default="/root/autodl-tmp/data/templete/outputs/pipeline_lora_outputs/stage3/lora_train.txt",
        help="lora_train.txt 文件路径（Stage3 生成）",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="/root/autodl-tmp/models/checkpoint/stable-diffusion-v1-5.ckpt",
        help="Stable Diffusion v1.5 基础模型 .ckpt 路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/data/templete/outputs/pipeline_lora_outputs/stage4",
        help="LoRA 训练输出目录（将包含 .ckpt）",
    )
    parser.add_argument(
        "--train_network_script",
        type=str,
        default="train_network.py",
        help="kohya 的 train_network.py 脚本路径（默认假定在当前工作目录 / PATH 中可直接调用）",
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="可选：本地 tokenizer 缓存目录，用于离线加载 openai/clip-vit-large-patch14",
    )
    # accelerate options (important for 512 training)
    parser.add_argument(
        "--accelerate_mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="accelerate mixed precision (default: fp16)",
    )
    parser.add_argument(
        "--accelerate_num_processes",
        type=int,
        default=1,
        help="accelerate --num_processes (default: 1)",
    )
    # 常用超参数
    parser.add_argument("--network_dim", type=int, default=8, help="LoRA rank (network_dim)")
    parser.add_argument("--network_alpha", type=int, default=8, help="LoRA alpha (network_alpha)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--resolution", type=str, default="512,512", help="训练分辨率")
    parser.add_argument("--train_batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--max_train_steps", type=int, default=6000, help="最大训练步数")
    # bucket / resolution control (important for texture fidelity)
    parser.add_argument(
        "--enable_bucket",
        action="store_true",
        help="enable kohya bucket (default: off in this wrapper; turn on only when needed)",
    )
    parser.add_argument(
        "--min_bucket_reso",
        type=int,
        default=None,
        help="optional: set kohya --min_bucket_reso (e.g. 288 to avoid being snapped down to 256)",
    )
    parser.add_argument(
        "--max_bucket_reso",
        type=int,
        default=None,
        help="optional: set kohya --max_bucket_reso (e.g. 288 to lock buckets to 288)",
    )
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=None,
        help="optional: set kohya --bucket_reso_steps (e.g. 32 so 288 becomes an exact bucket)",
    )
    parser.add_argument(
        "--bucket_no_upscale",
        action="store_true",
        help="optional: pass kohya --bucket_no_upscale",
    )
    # train_network options that affect texture / speed / VRAM
    parser.add_argument("--gradient_checkpointing", action="store_true", help="enable gradient checkpointing")
    parser.add_argument("--cache_latents", action="store_true", help="cache latents")
    parser.add_argument("--cache_latents_to_disk", action="store_true", help="cache latents to disk")
    parser.add_argument("--vae_batch_size", type=int, default=None, help="vae batch size when caching latents")
    parser.add_argument("--save_every_n_steps", type=int, default=None, help="save checkpoint every N steps")
    parser.add_argument(
        "--output_name",
        type=str,
        default="landstyle_lora",
        help="输出 LoRA 文件名（不带扩展名），会生成 <output_name>.ckpt",
    )

    args = parser.parse_args()

    lora_train_file = Path(args.lora_train_file).expanduser().resolve()
    pretrained_model = Path(args.pretrained_model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.add(output_dir / "stage4_train_lora.log", rotation="10 MB")
    logger.info("=== LoRA Stage4: Start training via kohya train_network.py ===")
    logger.info(f"lora_train_file: {lora_train_file}")
    logger.info(f"pretrained_model: {pretrained_model}")
    logger.info(f"output_dir: {output_dir}")

    if not lora_train_file.exists():
        raise FileNotFoundError(f"lora_train.txt not found: {lora_train_file}")
    if not pretrained_model.exists():
        raise FileNotFoundError(f"pretrained model not found: {pretrained_model}")

    # 将 lora_train.txt 转换为 kohya 可直接使用的 caption .txt 文件（每张图像一个同名 .txt）
    logger.info("Preparing caption .txt files and metadata JSON from lora_train.txt for kohya train_network.py ...")
    image_paths = []
    num_captions = 0
    metadata = {}

    with lora_train_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            img_path_str, caption = line.split("|", 1)
            img_path = Path(img_path_str)
            if not img_path.exists():
                logger.warning(f"Image not found for caption entry, skip: {img_path}")
                continue

            caption_clean = caption.strip()

            # 为每张图片生成同名 .txt caption（兼容直接用 DreamBooth 布局的场景）
            caption_path = img_path.with_suffix(".txt")
            caption_path.write_text(caption_clean, encoding="utf-8")

            abs_img_path = str(img_path.resolve())
            image_paths.append(abs_img_path)
            num_captions += 1

            # 同时构建 metadata.json（Fine-tuning 模式），键为绝对路径
            metadata[abs_img_path] = {"caption": caption_clean}

    if not image_paths:
        raise RuntimeError(
            f"No valid entries parsed from lora_train.txt: {lora_train_file}. "
            f"Please check its content."
        )

    # 计算所有图片的公共根目录，作为 --train_data_dir 传给 kohya（主要用于默认逻辑；
    # 实际上我们在 metadata.json 里使用的是绝对路径）
    train_data_dir = Path(os.path.commonpath(image_paths))
    if train_data_dir.is_file():
        train_data_dir = train_data_dir.parent

    logger.info(f"Prepared {num_captions} caption files. Using train_data_dir={train_data_dir}")

    # 将 metadata 保存到输出目录，供 --in_json 使用（Fine-tuning / captions 模式）
    metadata_path = output_dir / "lora_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metadata JSON for kohya: {metadata_path}")

    # 组装命令（基本等价于你在配置里写的 accelerate 命令）
    cmd = [
        "accelerate",
        "launch",
        f"--num_processes={args.accelerate_num_processes}",
        f"--mixed_precision={args.accelerate_mixed_precision}",
        args.train_network_script,
        f"--pretrained_model_name_or_path={pretrained_model}",
        f"--train_data_dir={train_data_dir}",
        f"--in_json={metadata_path}",
        "--network_module=networks.lora",
        f"--network_dim={args.network_dim}",
        f"--network_alpha={args.network_alpha}",
        f"--learning_rate={args.learning_rate}",
        f"--resolution={args.resolution}",
        f"--output_dir={output_dir}",
        f"--train_batch_size={args.train_batch_size}",
        f"--max_train_steps={args.max_train_steps}",
        # 关键：要求直接存成 .ckpt
        "--save_model_as=ckpt",
        f"--output_name={args.output_name}",
    ]

    # Bucket-related options (only if enabled)
    if args.enable_bucket:
        cmd.append("--enable_bucket")
        if args.min_bucket_reso is not None:
            cmd.append(f"--min_bucket_reso={args.min_bucket_reso}")
        if args.max_bucket_reso is not None:
            cmd.append(f"--max_bucket_reso={args.max_bucket_reso}")
        if args.bucket_reso_steps is not None:
            cmd.append(f"--bucket_reso_steps={args.bucket_reso_steps}")
        if args.bucket_no_upscale:
            cmd.append("--bucket_no_upscale")

    if args.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if args.cache_latents:
        cmd.append("--cache_latents")
    if args.cache_latents_to_disk:
        cmd.append("--cache_latents_to_disk")
    if args.vae_batch_size is not None:
        cmd.append(f"--vae_batch_size={args.vae_batch_size}")
    if args.save_every_n_steps is not None:
        cmd.append(f"--save_every_n_steps={args.save_every_n_steps}")

    # 如果提供了 tokenizer_cache_dir，则传给 train_network.py，避免每次都尝试从网络下载
    if args.tokenizer_cache_dir is not None:
        cmd.append(f"--tokenizer_cache_dir={args.tokenizer_cache_dir}")

    logger.info("Running command:\n" + " ".join(str(c) for c in cmd))

    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("LoRA training failed, please check logs above.")
        raise SystemExit(result.returncode)

    # 训练完成后，kohya 会在 output_dir 里生成 <output_name>.ckpt
    final_ckpt = output_dir / f"{args.output_name}.ckpt"
    if final_ckpt.exists():
        logger.info(f"LoRA training finished. ckpt saved to: {final_ckpt}")
    else:
        logger.warning(
            f"Training finished but {final_ckpt} not found. "
            f"Please check output_dir: {output_dir} for actual file name."
        )


if __name__ == "__main__":
    main()


