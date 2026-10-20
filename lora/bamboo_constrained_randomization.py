#!/usr/bin/env python3
"""
Constrained randomization generation for bamboo with a *fixed* LoRA (no retraining).

Workflow
--------
1) Learn real attribute distributions from the existing report.json (tags per real image)
2) Sample tokens under constraints:
   - zoom sampled from a user-defined target distribution (dominant mode constraint)
   - other attrs sampled from conditional joint P(color,light,shadow,sharp | zoom) with Laplace smoothing
   - optional token dropout (zoom usually not dropped)
3) Generate with sd-scripts/gen_img.py using --from_file, grouped by (steps, scale)
4) Filter generated images by matching simple statistics to the real dataset (quantile band)
5) Optionally resize accepted images to 286 for Baseline_1203

This script is designed for your setup:
- SD v1.5 ckpt
- LoRA weights produced by pipeline/lora_stage4_train_lora.py
- tokens: zoom_18_52 / zoom_18_16, color_*, light_*, shadow_*, sharp_*
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


# ----------------------------
# Feature extraction (cheap)
# ----------------------------


def _lap_var(gray: np.ndarray) -> float:
    lap = (
        -4 * gray
        + np.roll(gray, 1, 0)
        + np.roll(gray, -1, 0)
        + np.roll(gray, 1, 1)
        + np.roll(gray, -1, 1)
    )
    return float(np.var(lap))


def _edge_density(gray: np.ndarray, thr: float = 0.12) -> float:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(mag > thr))


def _highfreq_energy_ratio(gray: np.ndarray) -> float:
    f = np.fft.fftshift(np.fft.fft2(gray))
    p = (np.abs(f) ** 2).astype(np.float64)
    h, w = gray.shape
    yy, xx = np.mgrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = rr.max()
    high = float(p[rr >= 0.35 * rmax].mean())
    allm = float(p.mean())
    return float(high / (allm + 1e-12))


def image_stats(img: Image.Image) -> Dict[str, float]:
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    gray = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)
    return {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "lap_var": _lap_var(gray),
        "edge": _edge_density(gray),
        "hf_ratio": _highfreq_energy_ratio(gray),
        "shadow_ratio": float(np.mean(gray < 0.12)),
    }


def compute_quantile_band(stats_list: List[Dict[str, float]], q_lo: float, q_hi: float) -> Dict[str, Tuple[float, float]]:
    keys = sorted(stats_list[0].keys())
    arr = {k: np.array([s[k] for s in stats_list], dtype=np.float32) for k in keys}
    band = {k: (float(np.quantile(arr[k], q_lo)), float(np.quantile(arr[k], q_hi))) for k in keys}
    return band


def in_band(s: Dict[str, float], band: Dict[str, Tuple[float, float]]) -> bool:
    for k, (lo, hi) in band.items():
        v = float(s[k])
        if v < lo or v > hi:
            return False
    return True


# ----------------------------
# Distribution sampling
# ----------------------------


@dataclass(frozen=True)
class TokenCombo:
    zoom: str
    color: str
    light: str
    shadow: str
    sharp: str

    def as_tuple(self) -> Tuple[str, str, str, str, str]:
        return (self.zoom, self.color, self.light, self.shadow, self.sharp)


def parse_target_zoom(s: str) -> Dict[str, float]:
    """
    Format: zoom_18_52:0.3,zoom_18_16:0.7
    """
    out: Dict[str, float] = {}
    for part in (p.strip() for p in s.split(",") if p.strip()):
        if ":" not in part:
            raise ValueError(f"bad --target-zoom item: {part}")
        k, v = part.split(":", 1)
        out[k.strip()] = float(v)
    z = sum(out.values())
    if z <= 0:
        raise ValueError("target zoom probabilities sum to 0")
    return {k: v / z for k, v in out.items()}


def weighted_choice(rng: random.Random, items: Sequence[str], probs: Sequence[float]) -> str:
    r = rng.random()
    acc = 0.0
    for it, p in zip(items, probs):
        acc += p
        if r <= acc:
            return it
    return items[-1]


def build_conditional_joint(
    tags: Iterable[Dict[str, str]], alpha: float
) -> Tuple[List[str], List[float], Dict[str, Tuple[List[TokenCombo], List[float]]]]:
    """
    Returns:
    - zoom_items, zoom_probs (empirical, smoothed)  [not used if user provides target zoom]
    - cond_map: zoom -> (combos, probs) for P(color,light,shadow,sharp | zoom) with smoothing
    """
    by_zoom = defaultdict(list)
    for t in tags:
        z = t.get("zoom_token", "zoom_unknown")
        c = t.get("color_token", "color_unknown")
        l = t.get("light_token", "light_unknown")
        sh = t.get("shadow_token", "shadow_unknown")
        sp = t.get("sharp_token", "sharp_unknown")
        by_zoom[z].append(TokenCombo(z, c, l, sh, sp))

    # zoom marginal (smoothed)
    zoom_items = sorted(by_zoom.keys())
    zoom_counts = np.array([len(by_zoom[z]) for z in zoom_items], dtype=np.float64)
    zoom_probs = (zoom_counts + alpha) / (zoom_counts.sum() + alpha * len(zoom_items))

    cond_map: Dict[str, Tuple[List[TokenCombo], List[float]]] = {}
    for z in zoom_items:
        combos = by_zoom[z]
        counts = Counter([c.as_tuple() for c in combos])
        combo_items = sorted(counts.keys())
        # Laplace smoothing over observed support (practical)
        vals = np.array([counts[k] for k in combo_items], dtype=np.float64)
        probs = (vals + alpha) / (vals.sum() + alpha * len(combo_items))
        cond_map[z] = ([TokenCombo(*k) for k in combo_items], probs.tolist())

    return zoom_items, zoom_probs.tolist(), cond_map


def maybe_drop(token: str, rng: random.Random, p_drop: float) -> str:
    if p_drop <= 0:
        return token
    return "" if rng.random() < p_drop else token


def make_prompt(
    combo: TokenCombo,
    *,
    fixed_text: str,
    negative: str,
    drop_attr: float,
    drop_zoom: float,
    rng: random.Random,
) -> str:
    z = maybe_drop(combo.zoom, rng, drop_zoom)
    c = maybe_drop(combo.color, rng, drop_attr)
    l = maybe_drop(combo.light, rng, drop_attr)
    sh = maybe_drop(combo.shadow, rng, drop_attr)
    sp = maybe_drop(combo.sharp, rng, drop_attr)
    tokens = [t for t in [z, c, l, sh, sp] if t]
    # keep prompt short to avoid SD aesthetic bias
    prompt = ", ".join(tokens + [fixed_text])
    if negative.strip():
        prompt = f"{prompt} --n {negative.strip()}"
    return prompt


# ----------------------------
# Generation
# ----------------------------


def run_gen_img_batch(
    *,
    sd_scripts_dir: Path,
    ckpt: Path,
    lora: Path,
    tokenizer_cache_dir: Path | None,
    outdir: Path,
    prompt_file: Path,
    seed: int,
    steps: int,
    scale: float,
    W: int,
    H: int,
    batch_size: int,
) -> None:
    cmd = [
        "python",
        str(sd_scripts_dir / "gen_img.py"),
        "--ckpt",
        str(ckpt),
        "--network_module",
        "networks.lora",
        "--network_weights",
        str(lora),
        "--network_mul",
        "1.0",
        "--outdir",
        str(outdir),
        "--W",
        str(W),
        "--H",
        str(H),
        "--images_per_prompt",
        "1",
        "--n_iter",
        "1",
        "--batch_size",
        str(batch_size),
        "--steps",
        str(steps),
        "--scale",
        str(scale),
        "--seed",
        str(seed),
        "--from_file",
        str(prompt_file),
        "--sequential_file_name",
    ]
    if tokenizer_cache_dir is not None:
        cmd += ["--tokenizer_cache_dir", str(tokenizer_cache_dir)]
    subprocess.run(cmd, check=True)


def list_images(dir_path: Path) -> List[Path]:
    out: List[Path] = []
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        out.extend(sorted(dir_path.glob(f"*{ext}")))
    return out


def resize_dir(in_dir: Path, out_dir: Path, size: int = 286) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in list_images(in_dir):
        with Image.open(p) as im:
            im = im.convert("RGB").resize((size, size), resample=Image.LANCZOS)
            im.save(out_dir / p.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-json", type=str, required=True, help="report json from tag_bamboo_zoom_color_and_make_lora_train.py")
    ap.add_argument("--dataset-dir", type=str, required=True, help="real dataset dir for filtering stats, e.g. .../train/bamboo")
    ap.add_argument("--sd-scripts-dir", type=str, default="/root/autodl-tmp/sd-scripts", help="sd-scripts directory")
    ap.add_argument("--sd-ckpt", type=str, required=True)
    ap.add_argument("--lora-ckpt", type=str, required=True)
    ap.add_argument("--tokenizer-cache-dir", type=str, default="", help="optional tokenizer cache dir")
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--n-images", type=int, default=400)
    ap.add_argument("--target-zoom", type=str, default="zoom_18_52:0.3,zoom_18_16:0.7")
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing alpha")
    ap.add_argument("--drop-attr", type=float, default=0.25, help="dropout prob for non-zoom tokens")
    ap.add_argument("--drop-zoom", type=float, default=0.0, help="dropout prob for zoom token (usually 0)")
    ap.add_argument("--fixed-text", type=str, default="aerial orthophoto, nadir top-down, bamboo canopy, landcover")
    ap.add_argument(
        "--negative",
        type=str,
        default="low quality, blurry, jpeg artifacts, watercolor, painterly, buildings, roads, cars, mixed green and yellow patches",
    )
    ap.add_argument("--scales", type=str, default="4.5,5.5,6.5")
    ap.add_argument("--steps", type=str, default="20,25,30")
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--filter-qlo", type=float, default=0.05)
    ap.add_argument("--filter-qhi", type=float, default=0.95)
    ap.add_argument("--max-keep", type=int, default=400, help="max accepted images to keep")
    ap.add_argument("--resize286", action="store_true", help="also write accepted images resized to 286x286")
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    np.random.seed(int(args.seed))

    report = json.loads(Path(args.report_json).read_text(encoding="utf-8"))
    tags = report.get("tags", {})
    # tags contains a special _thresholds key we might have added; filter it out
    per_image_tags: List[Dict[str, str]] = []
    for k, v in tags.items():
        if k == "_thresholds":
            continue
        if isinstance(v, dict):
            per_image_tags.append(v)
    if not per_image_tags:
        raise RuntimeError("no per-image tags found in report.json")

    _, _, cond_map = build_conditional_joint(per_image_tags, alpha=float(args.alpha))
    target_zoom = parse_target_zoom(args.target_zoom)

    zoom_items = list(target_zoom.keys())
    zoom_probs = [target_zoom[z] for z in zoom_items]
    # ensure all zooms exist in cond_map
    for z in zoom_items:
        if z not in cond_map:
            raise RuntimeError(f"zoom {z} not found in report distribution. available={list(cond_map.keys())}")

    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    steps_list = [int(x) for x in args.steps.split(",") if x.strip()]
    if not scales or not steps_list:
        raise ValueError("empty --scales or --steps")

    out_root = Path(args.out_root).expanduser().resolve()
    gen_dir = out_root / "generated_raw"
    keep_dir = out_root / "accepted"
    keep286_dir = out_root / "accepted_286"
    meta_dir = out_root / "meta"
    gen_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Sample prompts with constrained token combos
    manifest: List[Dict[str, object]] = []
    prompts_by_group: Dict[Tuple[int, float], List[str]] = defaultdict(list)

    for i in range(int(args.n_images)):
        z = weighted_choice(rng, zoom_items, zoom_probs)
        combos, probs = cond_map[z]
        # choose conditional combo
        idx = int(np.searchsorted(np.cumsum(probs), rng.random(), side="right"))
        if idx >= len(combos):
            idx = len(combos) - 1
        combo = combos[idx]

        st = rng.choice(steps_list)
        sc = rng.choice(scales)
        prompt = make_prompt(
            combo,
            fixed_text=str(args.fixed_text),
            negative=str(args.negative),
            drop_attr=float(args.drop_attr),
            drop_zoom=float(args.drop_zoom),
            rng=rng,
        )
        prompts_by_group[(st, sc)].append(prompt)
        manifest.append(
            {
                "i": i,
                "zoom": combo.zoom,
                "color": combo.color,
                "light": combo.light,
                "shadow": combo.shadow,
                "sharp": combo.sharp,
                "steps": st,
                "scale": sc,
                "prompt": prompt,
            }
        )

    (meta_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Generate grouped by (steps, scale)
    sd_scripts_dir = Path(args.sd_scripts_dir).expanduser().resolve()
    ckpt = Path(args.sd_ckpt).expanduser().resolve()
    lora = Path(args.lora_ckpt).expanduser().resolve()
    tokenizer_cache_dir = Path(args.tokenizer_cache_dir).expanduser().resolve() if str(args.tokenizer_cache_dir).strip() else None

    for gi, ((st, sc), prompts) in enumerate(sorted(prompts_by_group.items(), key=lambda x: (x[0][0], x[0][1]))):
        prompt_file = meta_dir / f"prompts_steps{st}_scale{sc}.txt"
        prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        outdir = gen_dir / f"steps{st}_scale{sc}"
        outdir.mkdir(parents=True, exist_ok=True)
        run_gen_img_batch(
            sd_scripts_dir=sd_scripts_dir,
            ckpt=ckpt,
            lora=lora,
            tokenizer_cache_dir=tokenizer_cache_dir,
            outdir=outdir,
            prompt_file=prompt_file,
            seed=int(args.seed) + gi * 1000,
            steps=int(st),
            scale=float(sc),
            W=int(args.W),
            H=int(args.H),
            batch_size=int(args.batch_size),
        )

    # Build real dataset stats band
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    real_paths = list_images(dataset_dir)
    if not real_paths:
        raise RuntimeError(f"no images in dataset dir: {dataset_dir}")
    real_stats: List[Dict[str, float]] = []
    for p in real_paths:
        with Image.open(p) as im:
            real_stats.append(image_stats(im))
    band = compute_quantile_band(real_stats, float(args.filter_qlo), float(args.filter_qhi))
    (meta_dir / "filter_band.json").write_text(json.dumps(band, ensure_ascii=False, indent=2), encoding="utf-8")

    # Filter generated images into accepted/
    keep_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    total = 0
    for sub in sorted(gen_dir.glob("*")):
        if not sub.is_dir():
            continue
        for p in list_images(sub):
            total += 1
            with Image.open(p) as im:
                st = image_stats(im)
            if in_band(st, band):
                shutil.copy2(p, keep_dir / p.name)
                kept += 1
                if kept >= int(args.max_keep):
                    break
        if kept >= int(args.max_keep):
            break

    print(f"[constrained_randomization] generated_total={total} accepted={kept} band_q=({args.filter_qlo},{args.filter_qhi}) out={keep_dir}")

    if args.resize286:
        resize_dir(keep_dir, keep286_dir, size=286)
        print(f"[constrained_randomization] resized accepted -> {keep286_dir}")


if __name__ == "__main__":
    main()


