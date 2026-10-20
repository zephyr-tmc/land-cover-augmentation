#!/usr/bin/env python3
import os, base64, json, time
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

API_URL = "http://localhost:6006/sdapi/v1/img2img"
IN_DIR  = os.environ.get("LAND_COVER_INPUT", "data/train/pine")  # 输入图片文件夹
OUT_DIR = os.environ.get("LAND_COVER_OUTPUT", "generated/diffusion/pine")  # 输出目录
os.makedirs(OUT_DIR, exist_ok=True)

# 你自己的默认参数（可直接替换成题面里的字段）
BASE_PAYLOAD = {
    "prompt": "nadir, overhead, aerial, orthophoto",   # 你的正向提示
    "negative_prompt": "blurry, artifacts, overexposure",
    "sampler_name": "DPM++ 2M",                            # 或者 "DPM++ 2M Karras" 等
    "scheduler": "Karras",
    "steps": 30,
    "cfg_scale": 6.0,
    "denoising_strength": 0.45,                         # 你讨论过的 0.35/0.40/0.45 也可在这里改
    "width": 512,                                       # 通常以 init 图尺寸为准；可按需覆盖
    "height": 512,
    "batch_size": 1,
    "n_iter": 1,
    "tiling": False,
    "restore_faces": False,
    "send_images": True,
    "save_images": False,
    # 下面这些是 img2img 相关可选项，按需打开
    "resize_mode": 0,             # 0=Just resize, 1=Crop and resize, 2=Resize and fill
    "inpaint_full_res": True,
    "inpainting_mask_invert": 0,
    # ——这里才是关键：覆盖模型 / VAE——
    "override_settings": {
        "sd_model_checkpoint": "stable-diffusion-v1-5.safetensors",
        "sd_vae": "vae-ft-mse-840000-ema-pruned.safetensors"
    },
    # 设为 False 表示 *本次调用后不恢复*，反之 True 会恢复 GUI 原来的模型
    "override_settings_restore_afterwards": False
}

def load_image_as_base64(path):
    with Image.open(path) as img:
        # 建议统一到 RGB；某些灰度/带透明通道图，服务端可能有不同处理
        img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8"), img.size

def save_b64_image(b64_str, save_path):
    image_bytes = base64.b64decode(b64_str)
    with open(save_path, "wb") as f:
        f.write(image_bytes)

def is_image_file(name):
    name = name.lower()
    return any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"])

def main():
    files = [f for f in os.listdir(IN_DIR) if is_image_file(f)]
    if not files:
        print("No images found in:", IN_DIR)
        return

    for fname in tqdm(files, desc="img2img batching"):
        in_path = os.path.join(IN_DIR, fname)
        try:
            b64, (w, h) = load_image_as_base64(in_path)

            payload = dict(BASE_PAYLOAD)  # 复制基准参数
            payload["init_images"] = [b64]


            resp = requests.post(API_URL, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            # 返回格式：{"images": ["<base64>", ...], "parameters": {...}, "info": "string"}
            images = data.get("images", [])
            if not images:
                print(f"[WARN] No images in response for {fname}")
                continue

            # 若 batch_size/n_iter>1，可能会返回多张；统一编号保存
            stem, _ = os.path.splitext(fname)
            for i, img_b64 in enumerate(images):
                out_name = f"{stem}_out_{i:02d}.png"
                out_path = os.path.join(OUT_DIR, out_name)
                save_b64_image(img_b64, out_path)

            # 可选：把完整 info 存到同名 json 以便复现实验
            """
            info_txt = data.get("info", "")
            params   = data.get("parameters", {})
            with open(os.path.join(OUT_DIR, f"{stem}_meta.json"), "w") as f:
                json.dump({"request": payload, "parameters": params, "info": info_txt}, f, ensure_ascii=False, indent=2)
            """

            # 轻限速，避免 GUI 后端阻塞（自行酌情）
            time.sleep(0.05)

        except requests.exceptions.RequestException as e:
            print(f"[ERR] HTTP error for {fname}: {e}")
        except Exception as e:
            print(f"[ERR] Failed on {fname}: {e}")

if __name__ == "__main__":
    main()
