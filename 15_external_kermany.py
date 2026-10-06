"""15_external_kermany.py - external validation on the Kermany OCT test set (Aresta et al. 2025 comparison).

Not part of the main protocol (changes.md item 27). Nothing here retrains a model or touches the NEH splits,
thresholds or main tables; outputs go to outputs/external/.

Stages (run all by default):
  download   fetch only CellData/OCT/test/* (1,000 JPEGs, ~85 MB) from the 8.4 GB Mendeley archive via HTTP
             range requests (remote_zip.py) -> data_external/kermany_test/; the train file names are listed
             (not downloaded) to check patient overlap
  eda        image size / channels / borders / patients -> outputs/external/kermany_eda.md
  preprocess the unchanged 04_preprocess.preprocess (scale-bar mask, retina crop, 224 x 224)
  score      all saved NEH models, plus a "map-max" score per model (Aresta et al.: max of the smoothed
             anomaly map instead of a whole-image average); NEH val/test are re-scored with map-max too
  report     ROC-AUC per class vs Aresta's Kermany numbers; NEH-val thresholds applied unchanged

Score choice is fixed in advance: for each model the main score and map-max are compared on NEH *validation*
only; the winner is named in the report. Test results for both are reported either way.
"""
import argparse
import importlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from common import (IMAGENET_MEAN, IMAGENET_STD, OCTDataset, get_device, load_config, scoring_frame, set_seed,
                    setup_logging)

RAW = Path("data_external/kermany_test")
OUT = Path("outputs/external")
CLASSES = ["NORMAL", "DRUSEN", "CNV", "DME"]
ARESTA = {"overall": 0.87, "DRUSEN": 0.70, "CNV": 0.95, "DME": 0.95}   # Aresta et al. 2025, Fig. 6 (Kermany test)
MODELS = ["ae", "svdd_frozen", "svdd_ft", "vae", "mkd"]
MAPMAX = ["ae", "svdd_frozen", "vae", "mkd"]                         # models with a spatial anomaly map
SIGMA = 4                                                              # Gaussian smoothing of maps (pixels, 224 grid)


# ------------------------------------------------------------------ download
def download(log):
    from remote_zip import open_remote_zip
    z = open_remote_zip()
    names = z.namelist()
    test = [n for n in names if re.match(r"CellData/OCT/test/[A-Z]+/.+\.jpe?g$", n)]
    train = [n for n in names if re.match(r"CellData/OCT/train/[A-Z]+/.+\.jpe?g$", n)]
    pd.Series(train, name="path").to_csv(OUT / "kermany_train_listing.csv", index=False)
    log.info(f"archive: {len(names)} entries; OCT test {len(test)} images, train {len(train)} (listed only)")
    n_new = 0
    for n in test:
        dst = RAW / Path(n).relative_to("CellData/OCT/test")
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(z.read(n))
        n_new += 1
    log.info(f"downloaded {n_new} new test images -> {RAW}")


def test_frame():
    rows = []
    for p in sorted(RAW.glob("*/*.jp*g")):
        m = re.match(r"([A-Z]+)-(\d+)-(\d+)", p.stem)
        rows.append(dict(path=str(p), class_folder=p.parent.name, patient_id=f"K-{m.group(2)}", image_no=m.group(3)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ eda
def eda(df, log):
    sizes, modes, same_ch, white, black = [], [], [], [], []
    for p in df.path:
        im = Image.open(p)
        sizes.append(im.size)
        modes.append(im.mode)
        a = np.asarray(im)
        same_ch.append(a.ndim == 2 or bool((a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all()))
        g = a if a.ndim == 2 else a[..., 0]
        white.append(float((g >= 250).mean()))
        black.append(float((g <= 5).mean()))
    df = df.assign(width=[s[0] for s in sizes], height=[s[1] for s in sizes], mode=modes, grey=same_ch,
                   white_frac=white, black_frac=black)
    train = pd.read_csv(OUT / "kermany_train_listing.csv").path
    train_pat = set("K-" + train.str.extract(r"[A-Z]+-(\d+)-")[0])
    overlap = sorted(set(df.patient_id) & train_pat)
    lines = ["# Kermany OCT test set: EDA (vs NEH)", "",
             f"- Images: {len(df)}; per class: {df.class_folder.value_counts().to_dict()}",
             f"- Patients: {df.patient_id.nunique()} (images per patient: median "
             f"{df.groupby('patient_id').size().median():.0f}, max {df.groupby('patient_id').size().max()})",
             f"- Test patients that also appear in Kermany *train*: {len(overlap)} "
             f"({len(overlap) / df.patient_id.nunique():.0%})",
             f"- File format: {Path(df.path[0]).suffix} for all; PIL modes {df['mode'].value_counts().to_dict()}; "
             f"identical channels / single channel: {np.mean(same_ch):.0%}",
             "- Image sizes (width x height): "
             + ", ".join(f"{w}x{h}: {n}" for (w, h), n in df.groupby(["width", "height"]).size().items()),
             f"- Near-white pixels (>= 250): mean {np.mean(white):.1%}; images with > 5% near-white: "
             f"{(np.array(white) > 0.05).mean():.0%} (NEH: black background, no white areas)",
             f"- Near-black pixels (<= 5): mean {np.mean(black):.1%}",
             "", "NEH for comparison: 768 x 496 for every image, TIF 79% / JPG 21%, greyscale stored as RGB, "
             "burned-in scale bar (masked), full volumes with per-slice labels."]
    (OUT / "kermany_eda.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines[2:9]:
        log.info(line[2:])
    return df


# ------------------------------------------------------------------ preprocess
def remove_white_padding(g, level=250, min_px=200):
    """Kermany images often have white wedges at the edges (padding from registration/rotation); NEH has a black
    background. Near-white components that touch the image border and are larger than min_px are set to 0."""
    from scipy import ndimage
    lab, n = ndimage.label(g >= level)
    if n == 0:
        return g, 0.0
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(n + 1))
    kill = [b for b in border if b and sizes[b] >= min_px]
    m = np.isin(lab, kill)
    g = g.copy()
    g[m] = 0
    return g, float(m.mean())


def preprocess_all(df, cfg, log, harmonise):
    pre = importlib.import_module("04_preprocess")
    imgs, info = [], []
    for p in df.path:
        g = np.asarray(Image.open(p).convert("L"))
        removed = 0.0
        if harmonise:
            g, removed = remove_white_padding(g)
        im, inf = pre.preprocess(g, cfg["roi_height"], cfg["image_size"])
        inf["white_removed"] = removed
        imgs.append(im)
        info.append(inf)
    info = pd.DataFrame(info)
    log.info(f"preprocessed {len(imgs)} images (harmonise={harmonise}); retina-detection fallback "
             f"{info.fallback.mean():.1%}; white padding removed: mean {info.white_removed.mean():.1%} of pixels")
    return np.stack(imgs), info


# ------------------------------------------------------------------ scoring
def smooth_max(m):
    """Max of a Gaussian-smoothed B x H x W map (Aresta et al.: anomaly score = max of the anomaly map)."""
    from torchvision.transforms.functional import gaussian_blur
    k = int(4 * SIGMA) | 1
    return gaussian_blur(m[:, None], [k, k], [SIGMA, SIGMA])[:, 0].flatten(1).amax(1)


class Scorer:
    """Loads every saved NEH model once; score(x_unit) -> {column: B scores} for a batch in [0, 1]."""

    def __init__(self, device):
        from nets import ConvAutoencoder, ConvVAE, DeepSVDD, ResNetFeatures, VGGStudent, VGGTeacher
        self.d = device
        load = lambda p: torch.load(p, map_location=device)
        self.ae = ConvAutoencoder().to(device).eval()
        self.ae.load_state_dict(load("models/ae.pt")["state_dict"])
        self.vae = ConvVAE().to(device).eval()
        self.vae.load_state_dict(load("models/vae.pt")["state_dict"])
        self.res = ResNetFeatures().to(device).eval()
        c = load("models/svdd_centers.pt")
        self.c_frozen, self.c_ft, self.l3_mean = c["frozen_c"], c["ft_c"], c["layer3_mean"]
        ck = load("models/svdd_ft.pt")
        self.ft = DeepSVDD(ck["hp"]["rep_dim"]).to(device).eval()
        self.ft.load_state_dict(ck["state_dict"])
        self.student = VGGStudent().to(device).eval()
        self.student.load_state_dict(load("models/mkd.pt")["state_dict"])
        self.teacher = VGGTeacher().to(device).eval()
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def score(self, x):
        from nets import frozen_embedding, mkd_loss
        out = {}
        e = (self.ae(x) - x) ** 2
        out["ae"], out["ae_mapmax"] = e.flatten(1).mean(1), smooth_max(e[:, 0])
        x_hat, _, _ = self.vae(x)
        e = (x_hat - x) ** 2
        out["vae"], out["vae_mapmax"] = e.flatten(1).mean(1), smooth_max(e[:, 0])
        xi = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        l3, l4 = self.res(xi)
        out["svdd_frozen"] = ((frozen_embedding(l3, l4) - self.c_frozen) ** 2).sum(1)
        dmap = ((l3 - self.l3_mean[None]) ** 2).sum(1, keepdim=True)
        out["svdd_frozen_mapmax"] = smooth_max(F.interpolate(dmap, size=x.shape[-2:], mode="bilinear",
                                                             align_corners=False)[:, 0])
        out["svdd_ft"] = ((self.ft(xi) - self.c_ft) ** 2).sum(1)
        with torch.autocast(self.d.type, dtype=torch.bfloat16, enabled=self.d.type == "cuda"):
            fs, ft = self.student(xi), self.teacher(xi)
        out["mkd"] = mkd_loss(fs, ft, 0.01)
        total = 0                                   # Aresta eq. 1 per location (lambda = 1), mean over each level
        for s, t in zip(fs, ft):
            s, t = s.float(), t.float()
            d = ((s - t) ** 2).mean(1) + (1 - F.cosine_similarity(s, t, dim=1))
            total = total + F.interpolate(d[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        out["mkd_mapmax"] = smooth_max(total)
        return {k: v.float().cpu().numpy() for k, v in out.items()}


def run(scorer, loader, n):
    cols = {}
    for x, i in loader:
        for k, v in scorer.score(x.to(scorer.d)).items():
            cols.setdefault(k, np.zeros(n))[i.numpy()] = v
    return pd.DataFrame(cols)


# ------------------------------------------------------------------ report
def auc(y, s):
    return roc_auc_score(y, s) if 0 < np.mean(y) < 1 else np.nan


def patient_boot_ci(df, col, rng, n_boot=1000):
    """95% CI of the overall ROC-AUC, resampling patients within class (as script 10 does for NEH)."""
    pats = {c: df[df.class_folder == c].patient_id.unique() for c in df.class_folder.unique()}
    groups = {p: g for p, g in df.groupby("patient_id")}
    vals = []
    for _ in range(n_boot):
        pick = [p for c, ps in pats.items() for p in rng.choice(ps, len(ps), replace=True)]
        b = pd.concat([groups[p] for p in pick])
        vals.append(auc(b.class_folder != "NORMAL", b[col]))
    return np.nanpercentile(vals, [2.5, 97.5])


def report(k_all, neh, cfg, log):
    rng = np.random.default_rng(cfg["seed"])
    k = k_all[k_all.variant == "harmonised"].reset_index(drop=True)
    k_raw = k_all[k_all.variant == "raw"].reset_index(drop=True)
    cols = MODELS + [f"{m}_mapmax" for m in MAPMAX]
    v, t = neh[neh.split == "val"], neh[neh.split == "test"]
    rows, choice = [], {}
    for c in cols:
        vn = v.bscan_label == "normal"
        thr = np.percentile(v.loc[(v.class_folder == "NORMAL") & vn, c], cfg["threshold_percentile"])
        y = k.class_folder != "NORMAL"
        r = dict(score=c,
                 neh_val_auc=auc(~vn, v[c]),
                 neh_test_auc=auc(t.bscan_label != "normal", t[c]),
                 neh_test_drusen=auc(t[t.bscan_label != "cnv"].bscan_label == "drusen", t[t.bscan_label != "cnv"][c]),
                 neh_test_cnv=auc(t[t.bscan_label != "drusen"].bscan_label == "cnv", t[t.bscan_label != "drusen"][c]),
                 k_auc=auc(y, k[c]),
                 k_auc_raw=auc(k_raw.class_folder != "NORMAL", k_raw[c]))
        r["k_auc_ci"] = "[{:.3f}, {:.3f}]".format(*patient_boot_ci(k, c, rng))
        for cls in ("DRUSEN", "CNV", "DME"):
            m = k.class_folder.isin(["NORMAL", cls])
            r[f"k_{cls.lower()}"] = auc(k[m].class_folder == cls, k[m][c])
        r["k_pr_auc"] = average_precision_score(y, k[c])
        r["k_spec_at_neh_thr"] = float((k.loc[~y, c] <= thr).mean())
        r["k_sens_at_neh_thr"] = float((k.loc[y, c] > thr).mean())
        rows.append(r)
    res = pd.DataFrame(rows)
    for m in MAPMAX:                                 # pre-registered choice: NEH validation only
        a, b = res.set_index("score").neh_val_auc[[m, f"{m}_mapmax"]]
        choice[m] = f"{m}_mapmax" if b > a else m
    choice["svdd_ft"] = "svdd_ft"
    res.to_csv(OUT / "kermany_results.csv", index=False)

    f = lambda x: f"{x:.3f}"
    lines = ["# External validation on the Kermany OCT test set", "",
             "Models trained on NEH only (no retraining, no tuning on Kermany). Kermany test: 1,000 central B-scans, "
             "250 each NORMAL / DRUSEN / CNV / DME (DME never seen by any model). B-scan ROC-AUC; CI = 1000 patient "
             "bootstrap resamples within class. `_mapmax` = max of the Gaussian-smoothed anomaly map (Aresta et al. "
             "2025) instead of the whole-image score. Thresholds = NEH val p95, applied unchanged. Main columns use "
             "harmonised images (edge-connected white padding set to black, as NEH's background); the last column "
             "uses the raw images.", "",
             "## Comparison with Aresta et al. 2025 (Kermany test, B-scan ROC-AUC)", "",
             "| method | all abnormal | drusen | CNV | DME | all abnormal, raw images |", "|---|---|---|---|---|---|",
             f"| **Aresta 2025 (reverse distillation, trained on 176+ in-house normal volumes)** | {ARESTA['overall']:.2f} | "
             f"{ARESTA['DRUSEN']:.2f} | {ARESTA['CNV']:.2f} | {ARESTA['DME']:.2f} | |"]
    for c in cols:
        r = res.set_index("score").loc[c]
        tag = " (val-chosen)" if c in choice.values() else ""
        lines.append(f"| {c}{tag} | {f(r.k_auc)} {r.k_auc_ci} | {f(r.k_drusen)} | {f(r.k_cnv)} | {f(r.k_dme)} | {f(r.k_auc_raw)} |")
    lines += ["", "## Same scores on NEH (B-scan ROC-AUC) and threshold transfer to Kermany", "",
              "| score | NEH val | NEH test | NEH test drusen | NEH test CNV | Kermany PR-AUC (base 0.75) | "
              "Kermany specificity @ NEH thr | Kermany sensitivity @ NEH thr |", "|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        lines.append(f"| {r.score} | {f(r.neh_val_auc)} | {f(r.neh_test_auc)} | {f(r.neh_test_drusen)} | "
                     f"{f(r.neh_test_cnv)} | {f(r.k_pr_auc)} | {f(r.k_spec_at_neh_thr)} | {f(r.k_sens_at_neh_thr)} |")
    lines += ["", "Validation choice (main score vs map-max, by NEH val B-scan ROC-AUC): "
              + ", ".join(f"{m} -> {c}" for m, c in choice.items())]
    lines += ["", "## NEH test, case level (max over each patient's B-scans)", "",
              "| score | case ROC-AUC [95% CI] |", "|---|---|"]
    case = t.groupby("patient_id").agg(class_folder=("class_folder", "first"), **{c: (c, "max") for c in cols})
    case = case.reset_index()
    for c in cols:
        lo, hi = patient_boot_ci(case, c, np.random.default_rng(cfg["seed"]))
        lines.append(f"| {c} | {f(auc(case.class_folder != 'NORMAL', case[c]))} [{lo:.3f}, {hi:.3f}] |")
    (OUT / "kermany_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="download,eda,score,report")
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    log = setup_logging("15_external_kermany", cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(",")
    if "download" in stages:
        download(log)
    k = test_frame()
    if "eda" in stages:
        k = eda(k, log)
    if "score" in stages:
        device = get_device(cfg["device"])
        scorer = Scorer(device)
        parts = []
        for variant, harm in (("harmonised", True), ("raw", False)):
            imgs, info = preprocess_all(k, cfg, log, harm)
            if harm:
                np.save(OUT / "kermany_preprocessed_harmonised.npy", imgs[::25])   # 40-image sample for figures
            x = torch.from_numpy(imgs.astype(np.float32) / 255.0)[:, None]
            kl = DataLoader(TensorDataset(x, torch.arange(len(x))), batch_size=16)
            parts.append(pd.concat([k[["path", "class_folder", "patient_id"]].assign(variant=variant),
                                    info[["fallback", "white_removed"]], run(scorer, kl, len(k))], axis=1))
        pd.concat(parts, ignore_index=True).to_csv(OUT / "kermany_scores.csv", index=False)
        log.info(f"Kermany scored -> {OUT / 'kermany_scores.csv'}")
        neh = scoring_frame(cfg)
        neh = neh[neh.split.isin(["val", "test"])].reset_index(drop=True)
        nl = DataLoader(OCTDataset(cfg, neh, norm="unit"), batch_size=16, num_workers=0)
        neh = pd.concat([neh, run(scorer, nl, len(neh))], axis=1)
        neh.to_csv(OUT / "neh_scores_with_mapmax.csv", index=False)
        log.info(f"NEH val/test re-scored (incl. map-max) -> {OUT / 'neh_scores_with_mapmax.csv'}")
        torch.cuda.empty_cache()
    if "report" in stages:
        report(pd.read_csv(OUT / "kermany_scores.csv"), pd.read_csv(OUT / "neh_scores_with_mapmax.csv"), cfg, log)


if __name__ == "__main__":
    main()
