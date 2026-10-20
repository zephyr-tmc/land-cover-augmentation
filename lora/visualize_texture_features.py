#!/usr/bin/env python3
"""
テクスチャ特徴量（ラプラシアン分散，エッジ密度，FFT高周波成分比，平均輝度，コントラスト）
の抽出過程を可視化するスクリプト．

発表用のイメージ図を生成し，原画像が各処理を経てどのように変換されるかを直感的に理解できるようにする．
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _to_gray(rgb01: np.ndarray) -> np.ndarray:
    """RGB [0,1] -> グレースケール"""
    return (0.299 * rgb01[..., 0] + 0.587 * rgb01[..., 1] + 0.114 * rgb01[..., 2]).astype(np.float32)


def _lap_image(gray: np.ndarray) -> np.ndarray:
    """ラプラシアン画像（可視化用）"""
    lap = (
        -4 * gray
        + np.roll(gray, 1, 0)
        + np.roll(gray, -1, 0)
        + np.roll(gray, 1, 1)
        + np.roll(gray, -1, 1)
    )
    return lap


def _lap_var(gray: np.ndarray) -> float:
    lap = _lap_image(gray)
    return float(np.var(lap))


def _edge_map(gray: np.ndarray) -> np.ndarray:
    """エッジ強度マップ（勾配の大きさ）"""
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def _edge_density(gray: np.ndarray, thr: float = 0.12) -> float:
    mag = _edge_map(gray)
    return float(np.mean(mag > thr))


def _fft_spectrum(gray: np.ndarray) -> np.ndarray:
    """FFT パワースペクトル（対数スケール，可視化用）"""
    f = np.fft.fftshift(np.fft.fft2(gray))
    p = np.abs(f) ** 2
    return np.log1p(p).astype(np.float32)


def _highfreq_mask(gray: np.ndarray, ratio: float = 0.35) -> np.ndarray:
    """高周波成分マスク（外側35%を高周波とする）"""
    h, w = gray.shape
    yy, xx = np.mgrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = rr.max()
    return (rr >= ratio * rmax).astype(np.float32)


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


def extract_all_visualizations(img: Image.Image) -> dict:
    """各特徴量の可視化用データを抽出"""
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    gray = _to_gray(a)

    lap_img = _lap_image(gray)
    edge_img = _edge_map(gray)
    fft_spec = _fft_spectrum(gray)
    hf_mask = _highfreq_mask(gray)

    return {
        "original": a,
        "gray": gray,
        "laplacian": lap_img,
        "edge_map": edge_img,
        "fft_spectrum": fft_spec,
        "highfreq_mask": hf_mask,
        "lap_var": _lap_var(gray),
        "edge_density": _edge_density(gray),
        "highfreq_ratio": _highfreq_energy_ratio(gray),
        "mean_luminance": float(gray.mean()),
        "contrast": float(gray.std()),
    }


def create_sample_image(size: int = 256) -> np.ndarray:
    """竹テクスチャ風のサンプル画像を生成（画像が指定されていない場合用）"""
    h, w = size, size
    np.random.seed(42)
    # ベース: ノイズ＋周期的パターンでテクスチャ感を出す
    x = np.linspace(0, 4 * np.pi, w)
    y = np.linspace(0, 4 * np.pi, h)
    xx, yy = np.meshgrid(x, y)
    base = 0.5 + 0.2 * np.sin(xx) * np.cos(yy) + 0.15 * np.random.randn(h, w).astype(np.float32)
    base = np.clip(base, 0, 1)
    # 緑がかった色
    r = base * 0.4 + 0.3
    g = base * 0.5 + 0.45
    b = base * 0.3 + 0.25
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def normalize_for_display(arr: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """表示用に [0,1] に正規化"""
    a = arr.astype(np.float64)
    if vmin is None:
        vmin = a.min()
    if vmax is None:
        vmax = a.max()
    if vmax - vmin < 1e-10:
        return np.zeros_like(a)
    return np.clip((a - vmin) / (vmax - vmin), 0, 1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="テクスチャ特徴量抽出の可視化図を生成")
    ap.add_argument(
        "--input",
        "-i",
        type=str,
        default="",
        help="入力画像パス（未指定時はサンプル画像を使用）",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=str,
        default="texture_feature_visualization.png",
        help="出力画像パス",
    )
    ap.add_argument(
        "--size",
        type=int,
        default=256,
        help="サンプル画像使用時のサイズ",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="出力解像度",
    )
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib が必要です: pip install matplotlib")

    # 入力画像の読み込み
    if args.input and Path(args.input).exists():
        with Image.open(args.input) as im:
            im = im.convert("RGB")
        a = np.asarray(im, dtype=np.float32) / 255.0
        # 中央を切り出して正方形に（大きすぎる場合）
        h, w = a.shape[:2]
        if h != w:
            s = min(h, w)
            top = (h - s) // 2
            left = (w - s) // 2
            a = a[top : top + s, left : left + s]
        if a.shape[0] > 512:
            pil = Image.fromarray((a * 255).astype(np.uint8))
            pil = pil.resize((256, 256), Image.Resampling.LANCZOS)
            a = np.asarray(pil, dtype=np.float32) / 255.0
        img = Image.fromarray((a * 255).astype(np.uint8))
    else:
        sample = create_sample_image(args.size)
        img = Image.fromarray((np.clip(sample, 0, 1) * 255).astype(np.uint8))

    data = extract_all_visualizations(img)

    # 図の作成: 2行3列
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), dpi=args.dpi)

    titles = [
        ("Original", data["original"], None, True),
        ("Grayscale", data["gray"], "gray", False),
        ("Laplacian", data["laplacian"], "gray", False),
        ("Edge density", data["edge_map"], "hot", False),
        ("FFT spectrum", data["fft_spectrum"], "magma", False),
        ("High-freq mask", data["highfreq_mask"], "viridis", False),
    ]

    for ax, (title, arr, cmap, is_rgb) in zip(axes.flat, titles):
        if is_rgb:
            ax.imshow(arr)
        else:
            disp = normalize_for_display(arr)
            ax.imshow(disp, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # 特徴量ベクトルの表示
    feat_text = (
        f"φ(x) = [\n"
        f"  Laplacian var: {data['lap_var']:.4f}, "
        f"Edge density: {data['edge_density']:.4f},\n"
        f"  FFT high-freq ratio: {data['highfreq_ratio']:.4f}, "
        f"Mean luminance: {data['mean_luminance']:.4f}, "
        f"Contrast: {data['contrast']:.4f}\n"
        f"]"
    )
    fig.text(0.5, 0.02, feat_text, ha="center", fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("Texture feature φ(x) extraction pipeline", fontsize=12, y=1.02)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=args.dpi)
    plt.close()
    print(f"保存しました: {out_path}")


if __name__ == "__main__":
    main()
