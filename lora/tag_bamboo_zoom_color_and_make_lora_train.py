#!/usr/bin/env python3
"""
Tag bamboo images with:
  1) discrete zoom bin tokens (e.g. zoom_18_16 / zoom_18_52)
  2) discrete color/season tokens (e.g. color_green / color_yellow / color_mixed)
and generate a kohya-compatible lora_train.txt:
  /abs/path/to/image.png|caption ...

Rationale
---------
Your earlier issue ("a single generated image mixes green+yellow") is a typical symptom of
over-ambiguous captions/prompts. This script makes captions more specific by adding
explicit tokens, and lets you sample tokens according to the real dataset's histogram.

Zoom bin inference
-----------------
If you don't have original zoom metadata for dataset_preprocessed images, we can learn a
simple classifier from reference data that DOES have known zoom (e.g. map_tools outputs):
  -ref zoom_18_16=/path/to/zl_18.16
  -ref zoom_18_52=/path/to/zl_18.52
Then predict the zoom bin for each dataset image using simple texture features.

Additional bins (recommended)
-----------------------------
Besides color/season, the dataset shows strong multi-modal distributions in:
- illumination / exposure (bright vs dark)
- shadow amount (shadowy vs non-shadowy)
- sharpness / texture (high-frequency vs smooth)

This script can add extra discrete tokens to reduce mode-mixing in generation.
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def _rgb_to_hsv_np(rgb01: np.ndarray) -> np.ndarray:
    """
    rgb01: HxWx3 in [0,1]
    returns: HxWx3 (h,s,v) in [0,1]
    """
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn

    # Hue
    h = np.zeros_like(mx)
    mask = diff > 1e-8
    # avoid division by zero
    diff_safe = np.where(mask, diff, 1.0)

    # When max is r
    mr = (mx == r) & mask
    h[mr] = ((g[mr] - b[mr]) / diff_safe[mr]) % 6.0
    # When max is g
    mg = (mx == g) & mask
    h[mg] = ((b[mg] - r[mg]) / diff_safe[mg]) + 2.0
    # When max is b
    mb = (mx == b) & mask
    h[mb] = ((r[mb] - g[mb]) / diff_safe[mb]) + 4.0
    h = (h / 6.0) % 1.0

    # Saturation
    s = np.zeros_like(mx)
    s[mx > 1e-8] = diff[mx > 1e-8] / mx[mx > 1e-8]

    v = mx
    return np.stack([h, s, v], axis=-1)


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


def extract_zoom_features(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    gray = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)
    feats = np.array(
        [
            _lap_var(gray),
            _edge_density(gray),
            _highfreq_energy_ratio(gray),
            float(gray.mean()),
            float(gray.std()),
        ],
        dtype=np.float32,
    )
    return feats


def infer_color_token(img: Image.Image) -> str:
    """
    Heuristic:
    - compute hue distribution over sufficiently saturated pixels
    - decide green/yellow/mixed
    """
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    hsv = _rgb_to_hsv_np(a)
    h = hsv[..., 0]  # 0..1
    s = hsv[..., 1]
    v = hsv[..., 2]

    mask = (s > 0.18) & (v > 0.15)
    if float(mask.mean()) < 0.05:
        return "color_low_sat"

    h_deg = (h[mask] * 360.0).astype(np.float32)

    # green roughly: 70..160 deg, yellow roughly: 35..70 deg (remote sensing can shift)
    green = np.mean((h_deg >= 70.0) & (h_deg <= 160.0))
    yellow = np.mean((h_deg >= 35.0) & (h_deg < 70.0))

    # mixed if both are present substantially
    if green > 0.25 and yellow > 0.18:
        return "color_mixed"
    if green >= yellow:
        return "color_green"
    return "color_yellow"


def compute_light_shadow_sharp_metrics(img: Image.Image) -> Tuple[float, float, float]:
    """
    Returns (v_mean, shadow_ratio, sharpness_lap_var).
    """
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    hsv = _rgb_to_hsv_np(a)
    v = hsv[..., 2].astype(np.float32)
    v_mean = float(v.mean())

    gray = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)
    shadow_ratio = float(np.mean(gray < 0.12))
    sharp = _lap_var(gray)
    return v_mean, shadow_ratio, sharp


def _bin_by_quantiles(x: np.ndarray, q1: float, q2: float) -> Tuple[float, float]:
    """
    Return (t1, t2) where t1=quantile(q1), t2=quantile(q2).
    """
    t1 = float(np.quantile(x, q1))
    t2 = float(np.quantile(x, q2))
    if t2 < t1:
        t1, t2 = t2, t1
    return t1, t2


def load_image_paths(dir_path: Path, exts: Tuple[str, ...]) -> List[Path]:
    paths: List[Path] = []
    for ext in exts:
        paths.extend(Path(p).resolve() for p in glob.glob(str(dir_path / f"*{ext}")))
    return sorted(list({p for p in paths}))


@dataclass
class RefSet:
    token: str
    paths: List[Path]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=str, required=True, help="e.g. /root/autodl-tmp/data/dataset_preprocessed/train/bamboo")
    ap.add_argument(
        "--ref",
        action="append",
        default=[],
        help="reference zoom dirs in form token=/path/to/dir, e.g. zoom_18_16=/.../zl_18.16",
    )
    ap.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="output lora_train.txt",
    )
    ap.add_argument(
        "--caption-template",
        type=str,
        default=(
            "aerial orthophoto, remote sensing top-down view of dense bamboo forest canopy texture, "
            "{zoom_token}, {color_token}, {light_token}, {shadow_token}, {sharp_token}"
        ),
        help="must include {zoom_token} and {color_token}",
    )
    ap.add_argument(
        "--exts",
        type=str,
        default=".png,.jpg,.jpeg",
        help="comma-separated extensions to include",
    )
    ap.add_argument(
        "--save-report",
        type=str,
        default="",
        help="optional json report path to save predicted histograms and per-image tags",
    )
    ap.add_argument(
        "--swap-zoom-labels",
        action="store_true",
        help="swap zoom token names after prediction (useful if you know the semantic mapping is reversed)",
    )
    ap.add_argument(
        "--enforce-zoom-majority",
        type=str,
        default="",
        help="if set to a token name (e.g. zoom_18_52), and it is NOT the majority after prediction, "
        "automatically swap the two zoom labels (only works when exactly 2 zoom classes exist).",
    )
    ap.add_argument(
        "--no-extra-tokens",
        action="store_true",
        help="do not add light/shadow/sharp tokens (only zoom+color).",
    )
    args = ap.parse_args()

    if "{zoom_token}" not in args.caption_template or "{color_token}" not in args.caption_template:
        raise ValueError("--caption-template must include {zoom_token} and {color_token}")
    if not args.no_extra_tokens:
        for k in ["{light_token}", "{shadow_token}", "{sharp_token}"]:
            if k not in args.caption_template:
                raise ValueError(
                    f"--caption-template must include {k} unless --no-extra-tokens is set."
                )

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    exts = tuple(e.strip() for e in args.exts.split(",") if e.strip())
    ds_paths = load_image_paths(dataset_dir, exts)
    if not ds_paths:
        raise RuntimeError(f"no images in dataset-dir: {dataset_dir}")

    # parse refs
    ref_sets: List[RefSet] = []
    for item in args.ref:
        if "=" not in item:
            raise ValueError(f"bad --ref: {item} (expected token=/path)")
        token, path = item.split("=", 1)
        token = token.strip()
        path = Path(path.strip()).expanduser().resolve()
        if not token:
            raise ValueError(f"empty token in --ref: {item}")
        if not path.exists():
            raise FileNotFoundError(f"ref dir not found: {path}")
        ref_paths = load_image_paths(path, (".png", ".jpg", ".jpeg"))
        if not ref_paths:
            raise RuntimeError(f"no images in ref dir: {path}")
        ref_sets.append(RefSet(token=token, paths=ref_paths))

    # Train zoom classifier if refs provided; else fall back to a single zoom token.
    zoom_token_default = "zoom_unknown"
    zoom_predict = None

    if ref_sets:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except Exception as e:
            raise RuntimeError("scikit-learn is required for zoom inference. Please install scikit-learn.") from e

        Xs: List[np.ndarray] = []
        ys: List[int] = []
        id_to_token: Dict[int, str] = {}
        for ci, rs in enumerate(ref_sets):
            id_to_token[ci] = rs.token
            for p in rs.paths:
                with Image.open(p) as im:
                    Xs.append(extract_zoom_features(im))
                ys.append(ci)

        X = np.stack(Xs, axis=0)
        y = np.array(ys, dtype=np.int64)

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X, y)

        def _pred_token(img: Image.Image) -> str:
            f = extract_zoom_features(img)[None, :]
            ci = int(clf.predict(f)[0])
            return id_to_token.get(ci, zoom_token_default)

        zoom_predict = _pred_token
    else:
        def _pred_token_no_ref(_img: Image.Image) -> str:
            return zoom_token_default

        zoom_predict = _pred_token_no_ref

    # Tag dataset and write lora_train
    lines: List[str] = []
    tags: Dict[str, Dict[str, str]] = {}
    zoom_hist: Dict[str, int] = {}
    color_hist: Dict[str, int] = {}
    light_hist: Dict[str, int] = {}
    shadow_hist: Dict[str, int] = {}
    sharp_hist: Dict[str, int] = {}

    # First pass: predict tokens
    preds: List[Tuple[Path, str, str, float, float, float]] = []
    for p in ds_paths:
        with Image.open(p) as im:
            zt = zoom_predict(im) if zoom_predict is not None else zoom_token_default
            ct = infer_color_token(im)
            v_mean, shadow_ratio, sharp = compute_light_shadow_sharp_metrics(im)
        preds.append((p, zt, ct, v_mean, shadow_ratio, sharp))

    # Optional post-processing: swap zoom labels
    if args.swap_zoom_labels or args.enforce_zoom_majority:
        zset = sorted({zt for _, zt, _, _, _, _ in preds})
        if len(zset) == 2:
            a, b = zset[0], zset[1]
            zh: Dict[str, int] = {a: 0, b: 0}
            for _, zt, _, _, _, _ in preds:
                if zt in zh:
                    zh[zt] += 1
            should_swap = bool(args.swap_zoom_labels)
            if args.enforce_zoom_majority.strip():
                target = args.enforce_zoom_majority.strip()
                if target in zh:
                    maj = a if zh[a] >= zh[b] else b
                    if maj != target:
                        should_swap = True
            if should_swap:
                mapping = {a: b, b: a}
                preds = [(p, mapping.get(zt, zt), ct, v, sh, sp) for (p, zt, ct, v, sh, sp) in preds]

    # Second pass: finalize histograms and captions
    if args.no_extra_tokens:
        # Only zoom+color tokens
        for p, zt, ct, _v, _sh, _sp in preds:
            zoom_hist[zt] = zoom_hist.get(zt, 0) + 1
            color_hist[ct] = color_hist.get(ct, 0) + 1
            caption = args.caption_template.format(zoom_token=zt, color_token=ct).strip()
            lines.append(f"{str(p.resolve())}|{caption}")
            tags[str(p.resolve())] = {"zoom_token": zt, "color_token": ct}
    else:
        v_all = np.array([v for _, _, _, v, _, _ in preds], dtype=np.float32)
        sh_all = np.array([sh for _, _, _, _, sh, _ in preds], dtype=np.float32)
        sp_all = np.array([sp for _, _, _, _, _, sp in preds], dtype=np.float32)

        v_t1, v_t2 = _bin_by_quantiles(v_all, 0.33, 0.66)
        sh_t = float(np.quantile(sh_all, 0.80))  # top 20% as shadow_high
        sp_t1, sp_t2 = _bin_by_quantiles(sp_all, 0.33, 0.66)

        def light_token(v: float) -> str:
            if v < v_t1:
                return "light_dark"
            if v > v_t2:
                return "light_bright"
            return "light_normal"

        def shadow_token(sh: float) -> str:
            return "shadow_high" if sh >= sh_t else "shadow_low"

        def sharp_token(sp: float) -> str:
            if sp < sp_t1:
                return "sharp_low"
            if sp > sp_t2:
                return "sharp_high"
            return "sharp_normal"

        for p, zt, ct, v, sh, sp in preds:
            lt = light_token(v)
            sht = shadow_token(sh)
            spt = sharp_token(sp)

            zoom_hist[zt] = zoom_hist.get(zt, 0) + 1
            color_hist[ct] = color_hist.get(ct, 0) + 1
            light_hist[lt] = light_hist.get(lt, 0) + 1
            shadow_hist[sht] = shadow_hist.get(sht, 0) + 1
            sharp_hist[spt] = sharp_hist.get(spt, 0) + 1

            caption = args.caption_template.format(
                zoom_token=zt,
                color_token=ct,
                light_token=lt,
                shadow_token=sht,
                sharp_token=spt,
            ).strip()
            lines.append(f"{str(p.resolve())}|{caption}")
            tags[str(p.resolve())] = {
                "zoom_token": zt,
                "color_token": ct,
                "light_token": lt,
                "shadow_token": sht,
                "sharp_token": spt,
            }

        # Put thresholds into tags for reproducibility
        tags["_thresholds"] = {
            "light_v_mean_q33": v_t1,
            "light_v_mean_q66": v_t2,
            "shadow_ratio_q80": sh_t,
            "sharp_lapvar_q33": sp_t1,
            "sharp_lapvar_q66": sp_t2,
        }

    lines.sort()
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[tag_bamboo_zoom_color_and_make_lora_train] wrote {len(lines)} lines -> {output_file}")
    print("[zoom_hist]", json.dumps(zoom_hist, ensure_ascii=False))
    print("[color_hist]", json.dumps(color_hist, ensure_ascii=False))
    if not args.no_extra_tokens:
        print("[light_hist]", json.dumps(light_hist, ensure_ascii=False))
        print("[shadow_hist]", json.dumps(shadow_hist, ensure_ascii=False))
        print("[sharp_hist]", json.dumps(sharp_hist, ensure_ascii=False))

    if args.save_report:
        report_path = Path(args.save_report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "dataset_dir": str(dataset_dir),
            "n_images": len(ds_paths),
            "zoom_hist": zoom_hist,
            "color_hist": color_hist,
            "light_hist": light_hist,
            "shadow_hist": shadow_hist,
            "sharp_hist": sharp_hist,
            "tags": tags,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[report] saved -> {report_path}")


if __name__ == "__main__":
    main()


