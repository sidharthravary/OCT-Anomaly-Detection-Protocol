"""04_preprocess.py - retina ROI crop, resize and cache every B-scan.

Per image:
  1. grayscale uint8; the burned-in scale bar (bottom-left) is blacked out
  2. retina detection: Gaussian blur -> Otsu threshold -> binary opening; per
     column, the first and last retina pixel; the retina spans the 2nd
     percentile of tops to the 98th percentile of bottoms
  3. crop a fixed-height window (roi_height rows, full width) centred on the
     retina, clipped to the frame; centre crop if detection fails
  4. resize to image_size x image_size, store as uint8

Every image is cached, including the excluded layout outliers (they are scored
separately later). Normalisation happens at load time (common.OCTDataset).

Outputs: outputs/cache/images.npy (N x S x S uint8), outputs/cache/index.csv,
outputs/figures/preprocess_examples.png
"""
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_otsu
from tqdm import tqdm

from common import CLASSES, DataSource, load_config, load_manifest, set_seed, setup_logging

SCALE_BAR = (slice(455, 492), slice(0, 86))   # rows, cols; EDA: colour confined to rows 463-483, cols 13-63
MIN_COVERAGE = 0.3                             # share of columns that must contain retina
MIN_SPAN = 60                                  # rows


def detect_retina(gray):
    """Return (top, bottom, coverage) of the retinal band, or None if detection fails."""
    s = ndimage.gaussian_filter(gray.astype(np.float32), 4)
    try:
        t = threshold_otsu(s)
    except ValueError:  # constant image
        return None
    mask = ndimage.binary_opening(s > t, iterations=2)
    cols = mask.any(axis=0)
    coverage = cols.mean()
    if coverage < MIN_COVERAGE:
        return None
    tops = mask.argmax(axis=0)[cols]
    bottoms = (mask.shape[0] - 1 - mask[::-1].argmax(axis=0))[cols]
    top, bottom = np.percentile(tops, 2), np.percentile(bottoms, 98)
    if bottom - top < MIN_SPAN:
        return None
    return float(top), float(bottom), float(coverage)


def preprocess(gray, roi_height, size):
    """Crop + resize one grayscale B-scan. Returns (image, info dict)."""
    gray = gray.copy()
    gray[SCALE_BAR] = 0
    h = gray.shape[0]
    roi_height = min(roi_height, h)
    det = detect_retina(gray)
    if det is None:
        crop_top, info = (h - roi_height) // 2, dict(retina_top=np.nan, retina_bottom=np.nan, coverage=np.nan,
                                                   fallback=True)
    else:
        top, bottom, cov = det
        crop_top = int(round((top + bottom) / 2 - roi_height / 2))
        crop_top = min(max(crop_top, 0), h - roi_height)
        info = dict(retina_top=top, retina_bottom=bottom, coverage=cov, fallback=False)
    crop = gray[crop_top:crop_top + roi_height]
    out = np.asarray(Image.fromarray(crop).resize((size, size), Image.BILINEAR))
    info.update(crop_top=crop_top, crop_bottom=crop_top + roi_height)
    return out, info


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    log = setup_logging("04_preprocess", cfg)
    out = Path(cfg["output_dir"])
    cache, fig_dir = out / "cache", out / "figures"
    cache.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    size, roi_height = cfg["image_size"], cfg["roi_height"]

    m = load_manifest(cfg)
    src = DataSource(cfg["data_root"])
    missing = set(m.image_path) - set(src.image_paths)
    if missing:
        sys.exit(f"ERROR: {len(missing)} manifest images not found in data_root, e.g. {sorted(missing)[:3]}")
    n = len(m)
    row_of = {p: i for i, p in enumerate(m.image_path)}

    # examples for the before/after grid, chosen up front (the pass is sequential)
    rng = np.random.default_rng(cfg["seed"])
    main_pool = m[m.excluded.fillna("") == ""]
    ex = [p for c in CLASSES for p in rng.choice(main_pool.image_path[main_pool.class_folder == c].values, 5,
                                                   replace=False)]
    outliers = m.image_path[m.excluded.fillna("") != ""].values
    if len(outliers):
        ex.append(rng.choice(outliers))
    ex_raw = {}

    images = np.lib.format.open_memmap(cache / "images.npy.tmp", mode="w+", dtype=np.uint8, shape=(n, size, size))
    infos = [None] * n
    for rel, data in tqdm(src.iter_bytes(m.image_path), total=n, desc="preprocess", file=sys.stdout, mininterval=10):
        with Image.open(io.BytesIO(data)) as img:
            gray = np.asarray(img.convert("L"))
        i = row_of[rel]
        images[i], infos[i] = preprocess(gray, roi_height, size)
        if rel in ex:
            ex_raw[rel] = gray
    images.flush()
    del images
    (cache / "images.npy").unlink(missing_ok=True)
    (cache / "images.npy.tmp").rename(cache / "images.npy")

    idx = pd.concat([pd.DataFrame({"row": np.arange(n), "image_path": m.image_path}), pd.DataFrame(infos)], axis=1)
    idx.to_csv(cache / "index.csv", index=False)

    # ------------------------------------------------------------ report
    j = idx.merge(m[["image_path", "class_folder", "excluded"]], on="image_path")
    j["main"] = j.excluded.fillna("") == ""
    j["span"] = j.retina_bottom - j.retina_top
    j["clipped"] = (j.retina_top < j.crop_top) | (j.retina_bottom > j.crop_bottom)
    log.info(f"cached {n} images -> {cache / 'images.npy'} ({n * size * size / 1e6:.0f} MB), "
             f"crop {roi_height} x full width -> {size} x {size}")
    log.info(f"fallback (centre crop) rate: {j.fallback.mean():.2%} overall, "
             f"{j[j.main].fallback.mean():.2%} main experiment, {j[~j.main].fallback.mean():.2%} excluded")
    log.info("per class (main experiment only):")
    log.info(j[j.main].groupby("class_folder").agg(images=("row", "size"), fallback=("fallback", "mean"),
                                                   retina_clipped=("clipped", "mean"), span_median=("span", "median"),
                                                   crop_top_median=("crop_top", "median"))
             .reindex(CLASSES).to_string(float_format=lambda v: f"{v:.3f}"))
    log.info("retina_clipped = detected retina extends beyond the crop window (crop is centred, so both edges lose "
             "a little).")

    # ------------------------------------------------- before/after grid
    cached = np.load(cache / "images.npy", mmap_mode="r")
    ncols = 4
    nrows = int(np.ceil(len(ex) / ncols))
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=(ncols * 3.2, nrows * 2 * 2.1))
    for k, rel in enumerate(ex):
        r, c = divmod(k, ncols)
        info = j.loc[row_of[rel]]
        a, b = axes[2 * r, c], axes[2 * r + 1, c]
        a.imshow(ex_raw[rel], cmap="gray", vmin=0, vmax=255, aspect="auto")
        for y in (info.crop_top, info.crop_bottom):
            a.axhline(y, color="#eb6834", linewidth=1.2)
        if not info.fallback:
            for y in (info.retina_top, info.retina_bottom):
                a.axhline(y, color="#2a78d6", linewidth=0.8, linestyle="--")
        tag = " [excluded]" if not info.main else (" [fallback]" if info.fallback else "")
        a.set_title(rel + tag, fontsize=6.5, color="#52514e")
        b.imshow(cached[row_of[rel]], cmap="gray", vmin=0, vmax=255)
        for ax in (a, b):
            ax.set_xticks([])
            ax.set_yticks([])
    for k in range(len(ex), nrows * ncols):
        r, c = divmod(k, ncols)
        axes[2 * r, c].axis("off")
        axes[2 * r + 1, c].axis("off")
    fig.suptitle("Before (orange = crop window, blue dashed = detected retina) and after (224 x 224)",
                 fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(fig_dir / "preprocess_examples.png", dpi=200, facecolor="white")
    plt.close(fig)
    log.info(f"example grid -> {fig_dir / 'preprocess_examples.png'}")


if __name__ == "__main__":
    main()
