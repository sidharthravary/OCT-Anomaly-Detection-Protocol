"""01b_eda.py - exploratory data analysis of the NEH retinal OCT dataset.

Runs after 01_dataset_check.py and before common.py / 02 (it imports nothing
from common.py). One sequential pass over every image computes per-image
statistics; everything else is derived from those.

Questions it answers:
  * composition: folder class vs. specialist B-scan label, per volume and per slice position
  * image format / channels / intensity / noise / sharpness, and whether any of
    them differ by class (a shortcut a model could learn instead of pathology)
  * where the retina sits in the frame (input for the ROI crop in 04)
  * duplicate and near-duplicate images across patient folders (leakage)
  * how separable the classes look in a simple linear view (PCA + patient-grouped probe)

Outputs (outputs/eda/): image_stats.csv, duplicate_case_pairs.csv, eda_report.md,
figures/*.png. Large arrays go to outputs/cache/ (git-ignored).
"""
import hashlib
import io
import logging
import random
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

CLASSES = ["NORMAL", "DRUSEN", "CNV"]
LABELS = ["normal", "drusen", "cnv"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
THUMB = 64
# categorical slots 1-3 of the validated reference palette (all-pairs safe)
COLOR = {"NORMAL": "#2a78d6", "DRUSEN": "#eb6834", "CNV": "#1baf7a",
         "normal": "#2a78d6", "drusen": "#eb6834", "cnv": "#1baf7a"}
EXT_COLOR = {".tif": "#4a3aa7", ".jpg": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------- plumbing
def load_config(path="config.yaml"):
    if not Path(path).is_file():
        sys.exit(f"ERROR: {path} not found. Run from the project root.")
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(output_dir):
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("01b")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(log_dir / "01b_eda.log", mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def open_source(data_root):
    """Return (entries, read_in_order(entries) -> iterator of (entry, bytes), info_csv_bytes)."""
    root = Path(data_root)
    if root.is_dir():
        entries = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        info = next((p.read_bytes() for p in root.rglob("data_information.csv")), None)
        return entries, lambda es: ((e, (root / e).read_bytes()) for e in es), info
    if root.suffix.lower() != ".zip":
        sys.exit(f"ERROR: data_root '{root}' is neither a folder nor a .zip file.")
    outer = zipfile.ZipFile(root)
    inner = [n for n in outer.namelist() if n.lower().endswith(".zip")]
    zf = zipfile.ZipFile(outer.open(inner[0])) if inner else outer
    info = next((outer.read(n) for n in outer.namelist() if n.endswith("data_information.csv")), None)
    off = {i.filename: i.header_offset for i in zf.infolist()}
    entries = [n for n in off if PurePosixPath(n).suffix.lower() in IMAGE_EXTS]
    return entries, lambda es: ((e, zf.read(e)) for e in sorted(es, key=off.__getitem__)), info


def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def md_table(df, floatfmt="{:.3f}"):
    cols = [str(c) for c in df.columns]
    rows = [[floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r] for r in df.values]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])


# ------------------------------------------------------- per-image stats
IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def image_stats(gray):
    """Scalar descriptors of one uint8 grayscale B-scan."""
    g = gray.astype(np.float32)
    h, w = g.shape
    conv = ndimage.convolve(g, IMMERKAER, mode="reflect")
    noise = np.sqrt(np.pi / 2) * np.abs(conv[1:-1, 1:-1]).sum() / (6 * (w - 2) * (h - 2))
    rows = g.mean(axis=1)
    smooth = ndimage.gaussian_filter1d(rows, 5)
    lo, hi = np.percentile(smooth, 5), smooth.max()
    above = np.where(smooth > lo + 0.5 * (hi - lo))[0]
    cols = g.mean(axis=0)
    return dict(
        mean=g.mean(), std=g.std(), p01=np.percentile(g, 1), p99=np.percentile(g, 99),
        frac_dark=(gray < 10).mean(), frac_sat=(gray >= 250).mean(),
        noise_sigma=noise, lap_var=ndimage.laplace(g).var(),
        band_center=int(smooth.argmax()), band_top=int(above.min()), band_bottom=int(above.max()),
        dark_cols=int((cols < 5).sum()), dark_rows=int((rows < 5).sum()),
    ), rows


def dhash(thumb):
    """64-bit difference hash from a 64x64 thumbnail."""
    small = np.asarray(Image.fromarray(thumb).resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


# ------------------------------------------------------------------ main
def main():
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    out = Path(cfg["output_dir"])
    eda, fig_dir, cache = out / "eda", out / "eda" / "figures", out / "cache"
    for d in (eda, fig_dir, cache):
        d.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out)
    rep = ["# EDA report - NEH retinal OCT dataset", "",
           "Generated by `01b_eda.py`. All numbers are computed from the files on disk.", ""]

    # ---------------------------------------------------------- index
    entries, read_in_order, info_bytes = open_source(cfg["data_root"])
    recs = []
    for e in entries:
        parts = PurePosixPath(e).parts
        i = next(k for k, p in enumerate(parts) if p.upper() in CLASSES)
        rel = parts[i:]
        cls, case = rel[0].upper(), int(rel[1])
        eye_dir = rel[2].upper() if len(rel) == 4 else "unspecified"
        stem = PurePosixPath(e).stem
        idx, lab = re.match(r"^(\d+)_([A-Za-z]+)$", stem).groups()
        recs.append(dict(entry=e, image_path="/".join(rel), class_folder=cls, case_number=case,
                         patient_id=f"{cls}-{case:03d}", eye_dir=eye_dir, bscan_index=int(idx),
                         bscan_label=lab.lower(), ext=PurePosixPath(e).suffix.lower()))
    df = pd.DataFrame(recs)
    if info_bytes is not None:  # authoritative eye from data_information.csv
        info = pd.read_csv(io.BytesIO(info_bytes)).drop_duplicates("Directory").set_index("Directory")
        df["eye"] = df["image_path"].map(info["Eye"])
        assert df["eye"].notna().all(), "some files have no row in data_information.csv"
        assert (df["bscan_label"] == df["image_path"].map(info["Label"]).str.lower()).all()
    else:
        df["eye"] = df["eye_dir"]
    df["volume_id"] = df["patient_id"] + "-" + df["eye"]
    df = df.sort_values(["class_folder", "case_number", "eye", "bscan_index"]).reset_index(drop=True)
    log.info(f"indexed {len(df)} images")

    # choose full-resolution examples before the pass (reading out of order is slow)
    rng = random.Random(seed)
    combos = [(c, l) for c in CLASSES for l in LABELS if ((df.class_folder == c) & (df.bscan_label == l)).any()]
    examples = {}
    for c, l in combos:
        sub = df[(df.class_folder == c) & (df.bscan_label == l)]
        pats = rng.sample(sorted(sub.patient_id.unique()), min(5, sub.patient_id.nunique()))
        for p in pats:
            examples[sub[sub.patient_id == p].sample(1, random_state=seed).entry.iloc[0]] = (c, l)
    while len(examples) < len(combos) * 5:  # pad combos with fewer than 5 patients
        e = df.sample(1, random_state=len(examples)).entry.iloc[0]
        examples.setdefault(e, (df.set_index("entry").at[e, "class_folder"], df.set_index("entry").at[e, "bscan_label"]))

    # ------------------------------------------------------ one pass
    pos = {e: i for i, e in enumerate(df.entry)}
    n = len(df)
    thumbs = np.zeros((n, THUMB, THUMB), np.uint8)
    row_prof = np.zeros((n, 496), np.float16)
    stats, full = [None] * n, {}
    for e, data in tqdm(read_in_order(list(df.entry)), total=n, desc="EDA pass", file=sys.stdout, mininterval=10):
        i = pos[e]
        with Image.open(io.BytesIO(data)) as img:
            arr = np.asarray(img)
            mode, (w, h) = img.mode, img.size
        colour_box = (-1, -1, -1, -1)
        if arr.ndim == 3:
            diff = np.maximum(*(np.abs(arr[..., 0].astype(np.int16) - arr[..., k]) for k in (1, 2)))
            chan_diff = int(diff.max())
            ys, xs = np.nonzero(diff > 10)
            if len(ys):
                colour_box = (ys.min(), ys.max(), xs.min(), xs.max())
            gray = arr[..., 0] if chan_diff == 0 else np.asarray(Image.fromarray(arr).convert("L"))
        else:
            chan_diff, gray = 0, arr
        s, rows = image_stats(gray)
        s.update(md5=hashlib.md5(data).hexdigest(), mode=mode, width=w, height=h, chan_diff=chan_diff,
                 colour_y0=colour_box[0], colour_y1=colour_box[1], colour_x0=colour_box[2], colour_x1=colour_box[3],
                 file_kb=len(data) / 1024)
        stats[i] = s
        if h == row_prof.shape[1]:
            row_prof[i] = rows
        thumbs[i] = np.asarray(Image.fromarray(gray).resize((THUMB, THUMB), Image.BOX))
        if e in examples:
            full[e] = gray
    df = pd.concat([df, pd.DataFrame(stats)], axis=1)
    df["dhash"] = [dhash(t) for t in thumbs]
    np.save(cache / "eda_thumbs64.npy", thumbs)
    np.save(cache / "eda_row_profiles.npy", row_prof)
    df.drop(columns=["entry"]).to_csv(eda / "image_stats.csv", index=False)
    log.info("pass done; per-image stats -> outputs/eda/image_stats.csv")

    abn = df.bscan_label != "normal"
    clean = (df.class_folder == "NORMAL") & (df.bscan_label == "normal")

    # ================================================================ 1
    rep += ["## 1. Composition", ""]
    ct = pd.crosstab(df.class_folder, df.bscan_label).reindex(index=CLASSES, columns=LABELS, fill_value=0)
    ct["total"] = ct.sum(axis=1)
    rep += ["B-scan label (specialist, per image) vs. class folder (patient diagnosis):", "",
            md_table(ct.reset_index()), ""]
    vol = df.groupby(["class_folder", "patient_id"]).agg(n=("image_path", "size"), eyes=("eye", "nunique"),
                                                          abn_frac=("bscan_label", lambda s: (s != "normal").mean()))
    vs = vol.groupby("class_folder").agg(patients=("n", "size"), scans_min=("n", "min"), scans_median=("n", "median"),
                                         scans_max=("n", "max"), two_eyes=("eyes", lambda s: (s == 2).sum()),
                                         abnormal_frac_median=("abn_frac", "median")).reindex(CLASSES)
    rep += ["Per patient (case folder):", "", md_table(vs.reset_index(), "{:.2f}"), "",
            f"Clean-normal pool (NORMAL folder and normal label): **{clean.sum()} images** from "
            f"{df[clean].patient_id.nunique()} patients. Excluded from it: {((df.class_folder == 'NORMAL') & ~clean).sum()} "
            f"drusen-labelled slices in {df[(df.class_folder == 'NORMAL') & abn].patient_id.nunique()} NORMAL patients.", ""]
    zero_abn = vol[(vol.index.get_level_values(0) != "NORMAL") & (vol.abn_frac == 0)]
    rep += [f"DRUSEN/CNV patients with **no** abnormal-labelled slice: {len(zero_abn)} "
            f"{list(zero_abn.index.get_level_values(1))[:10]}", ""]

    fig, ax = plt.subplots(figsize=(7, 2.6))
    left = np.zeros(3)
    for l in LABELS:
        v = ct.loc[CLASSES, l].values
        ax.barh(CLASSES, v, left=left, color=COLOR[l], edgecolor="white", linewidth=2, label=f"{l} label", height=0.6)
        for y, (x0, x) in enumerate(zip(left, v)):
            if x > 400:
                ax.text(x0 + x / 2, y, f"{x:,}", ha="center", va="center", color="white", fontsize=8)
        left += v
    ax.invert_yaxis()
    ax.set_xlabel("B-scans")
    ax.set_title("Specialist B-scan label within each class folder", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.02))
    style(ax, "x")
    save(fig, fig_dir / "01_composition.png")

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.6), sharey=True)
    bins = np.arange(0, 66, 3)
    for ax, c in zip(axes, CLASSES):
        ax.hist(vol.loc[c, "n"], bins=bins, color=COLOR[c], edgecolor="white", linewidth=1)
        ax.set_title(f"{c} ({len(vol.loc[c])} patients)", fontsize=9, loc="left")
        ax.set_xlabel("B-scans per patient")
        style(ax)
    axes[0].set_ylabel("patients")
    save(fig, fig_dir / "02_scans_per_patient.png")

    # slice position: where in the volume are abnormal slices?
    df["rel_pos"] = df.groupby("volume_id").bscan_index.transform(lambda s: (s - s.min()) / max(s.max() - s.min(), 1))
    df["pos_bin"] = np.minimum((df.rel_pos * 10).astype(int), 9)
    fig, ax = plt.subplots(figsize=(6, 3))
    for c in CLASSES:
        f = df[df.class_folder == c].groupby("pos_bin").bscan_label.apply(lambda s: (s != "normal").mean())
        ax.plot((f.index + 0.5) / 10, f.values, color=COLOR[c], linewidth=2, marker="o", markersize=5, label=c)
    ax.set_xlabel("relative slice position in volume (0 = first, 1 = last)")
    ax.set_ylabel("share of slices labelled abnormal")
    ax.set_ylim(0, 1)
    ax.set_title("Abnormal slices concentrate in the middle of the volume", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    save(fig, fig_dir / "03_abnormal_by_slice_position.png")
    mid = df[df.class_folder != "NORMAL"]
    rep += ["Share of abnormal-labelled slices in DRUSEN/CNV volumes by relative position: "
            f"outer 20% = {(mid[(mid.rel_pos < .2) | (mid.rel_pos > .8)].bscan_label != 'normal').mean():.2f}, "
            f"central 20% = {(mid[(mid.rel_pos > .4) & (mid.rel_pos < .6)].bscan_label != 'normal').mean():.2f} "
            "(pathology is foveal; peripheral slices of diseased eyes often look normal).", ""]

    # ================================================================ 2
    rep += ["## 2. Image format, channels and size", ""]
    fmt = df.groupby(["class_folder", "ext", "mode", "width", "height"]).size().rename("images").reset_index()
    has_col = df.colour_y0 >= 0
    in_bar = has_col & (df.colour_y0 >= 455) & (df.colour_x1 <= 80)
    bar = df[in_bar]
    other = df[has_col & ~in_bar]
    rep += [md_table(fmt), "",
            f"Channel check: {has_col.sum()} of {n} images contain coloured pixels (|R-G| or |R-B| > 10). "
            f"In {in_bar.sum()} the colour is confined to rows {bar.colour_y0.min()}-{bar.colour_y1.max()}, columns "
            f"{bar.colour_x0.min()}-{bar.colour_x1.max()}: the burned-in '200 um' scale bar (bottom-left). "
            f"The other {len(other)} are from patients { {k: int(v) for k, v in other.patient_id.value_counts().items()} }: the layout outliers "
            "of section 4 plus a few scans with a thin coloured marker line drawn by the device software. "
            "Pixels are otherwise gray stored as RGB -> convert to 1 channel; crop or mask the scale bar in 04.", "",
            f"All images {df.width.iloc[0]}x{df.height.iloc[0]} (WxH) = aspect {df.width.iloc[0] / df.height.iloc[0]:.2f}; "
            "a plain resize to 224x224 squashes the retina horizontally, which is why 04 crops the retinal band first.", ""]
    ext_share = pd.crosstab(df.class_folder, df.ext, normalize="index").reindex(CLASSES)
    rep += ["Share of JPG per class folder: " + ", ".join(f"{c} {ext_share.at[c, '.jpg']:.1%}" for c in CLASSES)
            + ". Format is constant within a patient, so it is a *patient-level* property that correlates with class.", ""]
    ext_auc = roc_auc_score(df.class_folder == "NORMAL", df.ext == ".jpg")
    rep += [f"AUC of 'is JPG' for predicting the NORMAL folder: {ext_auc:.3f} (0.5 = no information).", ""]

    # ================================================================ 3
    rep += ["## 3. Intensity, noise and sharpness", ""]
    q = df.groupby(["class_folder", "ext"])[["mean", "std", "p99", "frac_dark", "noise_sigma", "lap_var", "file_kb"]] \
        .median().reindex(CLASSES, level=0).reset_index()
    rep += ["Medians per class folder and file format:", "", md_table(q, "{:.2f}"), ""]
    ql = df.groupby("bscan_label")[["mean", "std", "noise_sigma", "lap_var"]].median().reindex(LABELS).reset_index()
    rep += ["Medians per B-scan label:", "", md_table(ql, "{:.2f}"), ""]
    cn = df[clean]
    for col in ["mean", "noise_sigma", "lap_var"]:
        a = roc_auc_score(cn.ext == ".jpg", cn[col])
        rep += [f"Within clean-normal images, AUC of `{col}` for telling JPG from TIF: {a:.3f}"]
    rep += [""]

    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
    for ax, col, xl in zip(axes, ["mean", "noise_sigma", "lap_var"],
                           ["mean intensity (0-255)", "noise sigma (Immerkaer)", "Laplacian variance (sharpness)"]):
        hi = df[col].quantile(0.995)
        b = np.linspace(df[col].min(), hi, 50)
        for l in LABELS:
            ax.hist(df.loc[df.bscan_label == l, col].clip(upper=hi), bins=b, density=True, histtype="step",
                    linewidth=2, color=COLOR[l], label=f"{l} label")
        ax.set_xlabel(xl)
        style(ax)
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Pixel statistics by B-scan label", x=0.07, ha="left", fontsize=10, color=INK)
    save(fig, fig_dir / "04_intensity_by_label.png")

    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
    for ax, col, xl in zip(axes, ["mean", "noise_sigma", "lap_var"],
                           ["mean intensity (0-255)", "noise sigma (Immerkaer)", "Laplacian variance (sharpness)"]):
        hi = cn[col].quantile(0.995)
        b = np.linspace(cn[col].min(), hi, 50)
        for e in [".tif", ".jpg"]:
            ax.hist(cn.loc[cn.ext == e, col].clip(upper=hi), bins=b, density=True, histtype="step", linewidth=2,
                    color=EXT_COLOR[e], label=e)
        ax.set_xlabel(xl)
        style(ax)
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Clean-normal images only: JPG vs TIF (format-shortcut check)", x=0.07, ha="left", fontsize=10, color=INK)
    save(fig, fig_dir / "05_format_check_clean_normal.png")

    # ================================================================ 4
    rep += ["## 4. Where the retina sits (input for the ROI crop in 04)", ""]
    height = df.height.iloc[0]
    df["band_height"] = df.band_bottom - df.band_top
    bq = df[["band_center", "band_top", "band_bottom", "band_height"]].describe(percentiles=[.01, .05, .5, .95, .99]).T
    rep += [md_table(bq.reset_index(), "{:.1f}"), "",
            f"The half-max band is at most {int(df.band_height.quantile(.99))} rows tall for 99% of images "
            f"(frame height {height}). Band centre ranges {int(df.band_center.quantile(.01))}-"
            f"{int(df.band_center.quantile(.99))} (1st-99th pct) -> the retina moves vertically between scans, "
            "so a per-image crop centred on the band keeps it in view.", ""]
    rep += [f"Images with >50 fully dark columns (<5 mean; black wedges from tilted / motion-registered scans): "
            + ", ".join(f"{c} {(df.dark_cols[df.class_folder == c] > 50).mean():.1%}" for c in CLASSES) + ".", ""]

    # layout outliers: scan pasted small into the top-left of a black canvas
    pmed = df.groupby("patient_id").dark_cols.median()
    odd = sorted(pmed[pmed > 200].index)
    df["layout_outlier"] = df.patient_id.isin(odd)
    rep += ["### Layout outliers (acquisition shortcut)", "",
            f"Patients whose median image has >200 fully dark columns: **{len(odd)}** -> {odd} "
            f"({df.layout_outlier.sum()} images; next-highest patient median = {int(pmed[pmed <= 200].max())} columns). "
            "In these the B-scan is a small picture in the top-left of a black canvas with a text box in the middle "
            "(different export). None of them is NORMAL, so any model can separate them by layout alone; an "
            "anomaly detector trained on normals would score them high for the wrong reason. The band detector "
            "also fails on them (median band centre "
            f"{df[df.layout_outlier].band_center.median():.0f} vs {df[~df.layout_outlier].band_center.median():.0f}).", ""]
    ids = [np.where((df.patient_id == p).values)[0][len(df[df.patient_id == p]) // 2] for p in odd]
    ids += list(np.random.default_rng(seed).choice(np.where(~df.layout_outlier.values)[0], 7 * 2 - len(ids), replace=False))
    fig, axes = plt.subplots(2, 7, figsize=(14, 4.2))
    for ax, i in zip(axes.flat, ids):
        ax.imshow(thumbs[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(df.image_path[i], fontsize=6, color=INK2 if not df.layout_outlier[i] else COLOR["drusen"])
        ax.axis("off")
    fig.suptitle("Top: one image from each layout-outlier patient. Bottom: random regular images (64x64 thumbnails)",
                 x=0.12, ha="left", fontsize=10, color=INK)
    save(fig, fig_dir / "10_layout_outliers.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3), gridspec_kw=dict(width_ratios=[1.3, 1]))
    y = np.arange(height)
    for c in CLASSES:
        m = row_prof[(df.class_folder == c).values].astype(np.float32).mean(axis=0)
        axes[0].plot(y, m, color=COLOR[c], linewidth=2, label=c)
    axes[0].set_xlabel("image row (0 = top)")
    axes[0].set_ylabel("mean row intensity")
    axes[0].set_title("Average row-intensity profile", fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    style(axes[0])
    for c in CLASSES:
        axes[1].hist(df.loc[df.class_folder == c, "band_center"], bins=np.arange(0, height + 10, 10), histtype="step",
                     linewidth=2, color=COLOR[c], label=c, density=True)
    axes[1].set_xlabel("row of brightest band (retina centre)")
    axes[1].set_title("Retina vertical position per image", fontsize=10, loc="left")
    style(axes[1])
    save(fig, fig_dir / "06_retina_position.png")

    # ================================================================ 5
    ex = sorted(full, key=lambda e: (combos.index(examples[e]), e))
    ncol = 5
    fig, axes = plt.subplots(len(combos), ncol, figsize=(ncol * 2.4, len(combos) * 1.7))
    for r, (c, l) in enumerate(combos):
        row = [e for e in ex if examples[e] == (c, l)][:ncol]
        for k in range(ncol):
            ax = axes[r, k]
            ax.axis("off")
            if k < len(row):
                ax.imshow(full[row[k]], cmap="gray", vmin=0, vmax=255, aspect="auto")
                ax.set_title(row[k].split("NEH_UT_2021RetinalOCTDataset/")[-1], fontsize=5, color=INK2)
        axes[r, 0].text(-0.08, 0.5, f"{c} folder\n{l} label", transform=axes[r, 0].transAxes, ha="right",
                        va="center", fontsize=8, color=COLOR[l])
    save(fig, fig_dir / "07_examples_by_folder_and_label.png")

    t = thumbs.astype(np.float32)
    fig, axes = plt.subplots(2, 3, figsize=(8, 5.5))
    for k, l in enumerate(LABELS):
        sel = (df.bscan_label == l).values
        axes[0, k].imshow(t[sel].mean(0), cmap="gray", vmin=0, vmax=255)
        axes[0, k].set_title(f"mean image, {l} (n={sel.sum():,})", fontsize=8, color=COLOR[l])
        axes[1, k].imshow(t[sel].std(0), cmap="magma")
        axes[1, k].set_title(f"pixel std, {l}", fontsize=8, color=COLOR[l])
    for ax in axes.flat:
        ax.axis("off")
    save(fig, fig_dir / "08_mean_std_images.png")

    # ================================================================ 6
    rep += ["## 5. Duplicates across patient folders (leakage)", ""]
    g = df.groupby("md5")
    dup = df[g.image_path.transform("size") > 1]
    pairs = defaultdict(lambda: dict(n=0, same_label=0))
    for _, grp in dup.groupby("md5"):
        rows = grp.to_dict("records")
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                if rows[a]["patient_id"] != rows[b]["patient_id"]:
                    key = tuple(sorted([rows[a]["patient_id"], rows[b]["patient_id"]]))
                    pairs[key]["n"] += 1
                    pairs[key]["same_label"] += rows[a]["bscan_label"] == rows[b]["bscan_label"]
    # near-duplicates: identical 64-bit dHash and near-identical thumbnails across patients
    near = defaultdict(int)
    for _, grp in df.groupby("dhash"):
        if grp.patient_id.nunique() < 2:
            continue
        idx = grp.index.values
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                pa, pb = df.at[idx[a], "patient_id"], df.at[idx[b], "patient_id"]
                if pa != pb and df.at[idx[a], "md5"] != df.at[idx[b], "md5"] \
                        and np.abs(t[idx[a]] - t[idx[b]]).mean() < 3:
                    near[tuple(sorted([pa, pb]))] += 1
    size = df.groupby("patient_id").size()
    dp = pd.DataFrame([dict(patient_a=a, patient_b=b, identical_images=v["n"], same_label=v["same_label"],
                            near_duplicates=near.pop((a, b), 0), scans_a=size[a], scans_b=size[b])
                       for (a, b), v in sorted(pairs.items())]
                      + [dict(patient_a=a, patient_b=b, identical_images=0, same_label=0, near_duplicates=v,
                              scans_a=size[a], scans_b=size[b]) for (a, b), v in sorted(near.items())])
    dp.to_csv(eda / "duplicate_case_pairs.csv", index=False)
    if len(dp):
        dp["cross_class"] = dp.patient_a.str.split("-").str[0] != dp.patient_b.str.split("-").str[0]
        rep += [f"Byte-identical images shared between patient folders: {int(dp.identical_images.sum())} image pairs "
                f"linking **{(dp.identical_images > 0).sum()} pairs of patient folders** "
                f"({int(dp[dp.identical_images > 0].cross_class.sum())} of them across class folders). "
                f"Near-duplicates (same dHash, thumbnail MAE < 3, different bytes): {int(dp.near_duplicates.sum())}.", "",
                md_table(dp), ""]
    else:
        rep += ["No duplicates across patient folders.", ""]

    # ================================================================ 7
    rep += ["## 6. Linear view of the images (PCA on 64x64 thumbnails)", ""]
    X = t.reshape(n, -1) / 255.0
    pca = PCA(n_components=50, random_state=seed).fit(X)
    Z = pca.transform(X)
    rep += [f"50 PCs explain {pca.explained_variance_ratio_.sum():.1%} of pixel variance "
            f"(PC1 {pca.explained_variance_ratio_[0]:.1%}, PC2 {pca.explained_variance_ratio_[1]:.1%}).", ""]
    groups = df.patient_id.values
    gkf = GroupKFold(n_splits=5)
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    tasks = {"B-scan abnormal (label != normal)": abn.values,
             "file is JPG (format visible in pixels?)": (df.ext == ".jpg").values,
             "JPG vs TIF within clean-normal only": None}
    for name, yv in tasks.items():
        if yv is None:
            m = clean.values
            p = cross_val_predict(probe, Z[m], (df.ext == ".jpg").values[m], groups=groups[m], cv=gkf,
                                  method="predict_proba")[:, 1]
            auc = roc_auc_score((df.ext == ".jpg").values[m], p)
        else:
            p = cross_val_predict(probe, Z, yv, groups=groups, cv=gkf, method="predict_proba")[:, 1]
            auc = roc_auc_score(yv, p)
        rep += [f"- Logistic probe on 50 PCs, 5-fold grouped by patient -> **{name}**: AUC {auc:.3f}"]
        log.info(f"probe {name}: AUC {auc:.3f}")
    rep += ["", "The abnormal probe is a supervised *sanity check* of how much signal a linear model finds in raw "
            "pixels; it is not one of the project's methods (those never see abnormal images in training).", ""]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    order = np.random.default_rng(seed).permutation(n)
    for l in LABELS:
        s = order[(df.bscan_label.values[order] == l)]
        axes[0].scatter(Z[s, 0], Z[s, 1], s=3, alpha=0.35, color=COLOR[l], label=f"{l} label", linewidths=0)
    for e in [".tif", ".jpg"]:
        s = order[(df.ext.values[order] == e)]
        axes[1].scatter(Z[s, 0], Z[s, 1], s=3, alpha=0.35, color=EXT_COLOR[e], label=e, linewidths=0)
    for ax, ttl in zip(axes, ["coloured by B-scan label", "coloured by file format"]):
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(ttl, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, markerscale=4)
        style(ax, None)
    save(fig, fig_dir / "09_pca.png")

    # ================================================================ 8
    rep += ["## 7. Normalisation reference (clean-normal pool)", "",
            f"Pixel mean / std over clean-normal images, [0,1] scale: {cn['mean'].mean() / 255:.4f} / "
            f"{np.sqrt((cn['std'] ** 2 + (cn['mean'] - cn['mean'].mean()) ** 2).mean()) / 255:.4f} "
            "(full frame, before ROI crop; for reference only - pretrained encoders use ImageNet stats per the PRD).", ""]

    rep += ["## 8. Implications for the pipeline", "",
            "- **Patient id** must be `CLASS-case` (case numbers restart per class); **eye** comes from "
            "data_information.csv, which covers flat folders too.",
            "- **Labels**: filename suffix == CSV label for every file; normalise case (`NOrmal`, `DRusen`, ...).",
            "- **Channels**: convert to single-channel gray; crop/mask the bottom-left scale bar.",
            "- **ROI**: retina band moves vertically by ~240 rows across images; crop per image around the band, "
            "then resize (the 768x496 frame is 1.55:1).",
            f"- **Layout outliers** ({len(odd)} abnormal-only patients) and **duplicate folders** "
            f"({int((dp.identical_images > 0).sum()) if len(dp) else 0} pairs) need a decision before splitting "
            "(see sections 4 and 5).",
            "- **Format**: JPG share differs by class, but a patient-grouped linear probe cannot tell JPG from TIF "
            "from the pixels, so format is not an obvious pixel shortcut; still report results stratified by format.",
            ""]
    (eda / "eda_report.md").write_text("\n".join(rep), encoding="utf-8")
    df.drop(columns=["entry", "pos_bin"]).to_csv(eda / "image_stats.csv", index=False)  # now with derived columns
    log.info("\n".join(rep))
    log.info(f"\nfigures -> {fig_dir}")


if __name__ == "__main__":
    main()
