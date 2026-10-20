# 実行 / 运行 / Running

[日本語](../README.md) · [简体中文](../README.zh-CN.md) · [English](../README.en.md)

The commands below describe the curated source layout. They do not reproduce the historical results table without the original data, exact settings, split manifests and checkpoints.

## 1. Environment and data

Use Python 3.10–3.12 and install `requirements.txt`. Select a matched PyTorch/torchvision build for your hardware. The classification environment and the external sd-scripts training environment may need different dependency versions; keep them separate.

The classifier accepts this ImageFolder layout:

```text
data/
  train/{bamboo,pine,rice,tea}/*.png
  val/{bamboo,pine,rice,tea}/*.png
  test/{bamboo,pine,rice,tea}/*.png
```

Keep validation and test images separate from all augmentation inputs. Prepare real train/val/test splits before generating training samples. The dataset itself is not included.

## 2. Classification baseline

Run from the repository root:

```bash
python classifier/train_step23.py \
  --data-root ./data --layout split \
  --baseline1218 Real-only --backbone smallcnnpp \
  --epochs 100 --img-size 286 --batch-size 32 \
  --lr 0.0001 --label-smoothing 0.05 \
  --num-workers 4 --eval-test end --save-plots \
  --out-dir ./outputs/real-only
```

`--eval-test end` keeps test evaluation at the end. Select checkpoints using validation metrics, not test metrics. See `--help` for class weighting, focal loss, EMA and staged training options.

For augmentation comparisons, use the same split and classifier configuration. The runner accepts `B1dup`, `B2trad`, `GAN`, `lora-raw` and `lora-align`. For generated data, supply an additional directory containing `bamboo/` images:

```bash
python classifier/train_step23.py \
  --data-root ./data --layout split \
  --baseline1218 lora-align --bamboo-add 224 \
  --extra-train-dirs ./generated/lora-align \
  --backbone smallcnnpp --epochs 100 \
  --eval-test end --out-dir ./outputs/lora-align
```

## 3. Attribute captions

Use separate reference folders with known observation scales:

```bash
python lora/tag_bamboo_zoom_color_and_make_lora_train.py \
  --dataset-dir ./data/train/bamboo \
  --ref zoom_18_16=./references/zl_18.16 \
  --ref zoom_18_52=./references/zl_18.52 \
  --output-file ./outputs/tags/lora_train.txt
```

The script writes captions plus a JSON report containing attribute assignments. Use the actual emitted report path in the generation command below.

## 4. LoRA training and generation

Prepare [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) separately using its own dependency instructions. The publication does not vendor sd-scripts or model checkpoints. These wrappers target the Stable Diffusion 1.5 workflow used in the research; compatibility with every future sd-scripts revision is not claimed.

```bash
python lora/lora_stage4_train_lora.py \
  --lora_train_file ./outputs/tags/lora_train.txt \
  --pretrained_model /path/to/stable-diffusion-v1-5.ckpt \
  --train_network_script /path/to/sd-scripts/train_network.py \
  --output_dir ./outputs/lora
```

```bash
python lora/bamboo_constrained_randomization.py \
  --report-json /path/to/tagging-report.json \
  --dataset-dir ./data/train/bamboo \
  --sd-scripts-dir /path/to/sd-scripts \
  --sd-ckpt /path/to/stable-diffusion-v1-5.ckpt \
  --lora-ckpt /path/to/trained-lora.ckpt \
  --out-root ./generated/lora-align \
  --n-images 400 --max-keep 224 --resize286
```

Paths in `/path/to/` examples are inputs you must supply. Original server-specific defaults in research wrappers are not required when explicit arguments are supplied.

## 5. Earlier experiments

`CNN_model/` preserves earlier work separately from the later classifier. It includes conditional GAN/DDPM experiments, MixUp/CutMix runners, traditional offline augmentation and a local Stable Diffusion img2img client. Those settings are not interchangeable with the thesis comparison.

`CNN_model/batch_img2img.py` uses `LAND_COVER_INPUT` and `LAND_COVER_OUTPUT` to choose its local folders. It requires a separately running local img2img API at the address configured in that file.

## 6. Saved predictions

`evaluate.py` accepts a directory containing `y_true.npy`, `y_prob.npy` and `class_names.json`:

```bash
python evaluate.py ./outputs/predictions
```

The later runner also exports validation/test metrics and optionally plots. Keep the full run configuration, split identifiers, random seeds, software versions and checkpoint selection rule with any new reported result.
