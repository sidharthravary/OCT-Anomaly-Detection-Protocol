"""16_patchcore.py - M5: patch-level nearest-neighbour anomaly detection (PatchCore-style) on frozen ResNet18.

One-class, no training: a memory bank of patch features is built from the clean-normal B-scans of the TRAIN
patients only (the same 3,998 images as M1-M4). A B-scan's anomaly score is the distance of its most unusual
patch to the nearest healthy patch (max over the Gaussian-smoothed patch-distance map). Not part of the
original protocol (changes.md item 28); outputs go to outputs/improvements/.

Pre-registered variants (the choice is made on NEH validation B-scan ROC-AUC only):
  r18_l23  layer2 (128 x 28 x 28) + layer3 upsampled (256) -> 384-d patches, 28 x 28 grid
  r18_l12  layer1 (64 x 56 x 56, pooled to 28 x 28) + layer2 (128) -> 192-d patches, 28 x 28 grid (finer detail)
Each patch feature is averaged over its 3 x 3 neighbourhood (local context, as in PatchCore). The memory bank
is a greedy k-centre coreset of BANK patches drawn from a random candidate pool (both seeded).

Scores: NEH val + test (all B-scans) -> outputs/improvements/neh_patchcore_scores.csv;
        Kermany test (harmonised, as in 15) -> outputs/improvements/kermany_patchcore_scores.csv
"""
import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from common import (IMAGENET_MEAN, IMAGENET_STD, OCTDataset, get_device, get_split_frame, load_config, scoring_frame,
                    set_seed, setup_logging)
from nets import ResNetFeatures

OUT = Path("outputs/improvements")
VARIANTS = ("r18_l23", "r18_l12")
CANDIDATES = 400_000          # random candidate patches for the coreset
BANK = 20_000                 # coreset size
PROJ = 128                    # random-projection dimension used only to pick the coreset
SIGMA = 1.0                   # Gaussian smoothing of the 28 x 28 patch map (in patch units)


def patch_features(net, x, variant):
    """B x 3 x 224 x 224 (ImageNet-normalised) -> B x D x 28 x 28 locally averaged patch features."""
    s = net.stem(x)
    l1 = net.layer1(s)                 # 64 x 56 x 56
    l2 = net.layer2(l1)                # 128 x 28 x 28
    if variant == "r18_l23":
        l3 = net.layer3(l2)            # 256 x 14 x 14
        f = torch.cat([l2, F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)], 1)
    else:
        f = torch.cat([F.avg_pool2d(l1, 2), l2], 1)
    return F.avg_pool2d(f, 3, stride=1, padding=1)


def greedy_coreset(feats, k, device, rng):
    """Greedy k-centre selection on a random projection (PatchCore). feats: N x D (CPU float32)."""
    proj = torch.from_numpy(rng.standard_normal((feats.shape[1], PROJ)).astype(np.float32)).to(device)
    z = feats.to(device) @ proj
    sel = [int(rng.integers(len(z)))]
    dmin = torch.cdist(z, z[sel[-1]][None]).squeeze(1)
    for _ in range(k - 1):
        i = int(torch.argmax(dmin))
        sel.append(i)
        dmin = torch.minimum(dmin, torch.cdist(z, z[i][None]).squeeze(1))
    return feats[torch.tensor(sel)]


@torch.no_grad()
def build_bank(net, cfg, variant, device, log, rng):
    train = get_split_frame(cfg, "train")
    dl = DataLoader(OCTDataset(cfg, train, norm="imagenet"), batch_size=32, num_workers=0)
    per_img = max(1, CANDIDATES // len(train))
    cand = []
    for x, _ in dl:
        f = patch_features(net, x.to(device), variant)                    # B x D x 28 x 28
        f = f.permute(0, 2, 3, 1).reshape(f.shape[0], -1, f.shape[1])     # B x 784 x D
        for row in f:
            cand.append(row[torch.from_numpy(rng.choice(row.shape[0], per_img, replace=False))].cpu())
    cand = torch.cat(cand)
    t0 = time.time()
    bank = greedy_coreset(cand, BANK, device, rng).to(device)
    log.info(f"{variant}: {len(train)} train clean-normal B-scans -> {len(cand):,} candidate patches -> "
             f"coreset {len(bank):,} x {bank.shape[1]} ({time.time() - t0:.0f}s)")
    return bank


@torch.no_grad()
def score_loader(net, bank, loader, variant, device, n):
    from torchvision.transforms.functional import gaussian_blur
    out = np.zeros(n)
    bank_sq = (bank ** 2).sum(1)
    for x, i in loader:
        f = patch_features(net, x.to(device), variant)
        b, d, h, w = f.shape
        q = f.permute(0, 2, 3, 1).reshape(-1, d)
        # squared distance to every bank patch, in chunks; keep the nearest
        nn_d = torch.empty(q.shape[0], device=device)
        for s in range(0, q.shape[0], 8192):
            qs = q[s:s + 8192]
            dist = (qs ** 2).sum(1, keepdim=True) - 2 * qs @ bank.T + bank_sq[None]
            nn_d[s:s + 8192] = dist.clamp_min(0).min(1).values.sqrt()
        m = gaussian_blur(nn_d.view(b, 1, h, w), [5, 5], [SIGMA, SIGMA])[:, 0]
        out[i.numpy()] = m.flatten(1).amax(1).cpu().numpy()
    return out


def kermany_tensor(cfg, log):
    """Harmonised Kermany test images, preprocessed exactly as in 15_external_kermany.py, ImageNet-normalised."""
    ext = importlib.import_module("15_external_kermany")
    k = ext.test_frame()
    imgs, _ = ext.preprocess_all(k, cfg, log, harmonise=True)
    x = torch.from_numpy(imgs.astype(np.float32) / 255.0)[:, None].expand(-1, 3, -1, -1)
    x = (x - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return k[["path", "class_folder", "patient_id"]], x


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    log = setup_logging("16_patchcore", cfg)
    device = get_device(cfg["device"])
    OUT.mkdir(parents=True, exist_ok=True)
    net = ResNetFeatures().to(device).eval()

    neh = scoring_frame(cfg)
    neh = neh[neh.split.isin(["val", "test"])].reset_index(drop=True)
    nl = DataLoader(OCTDataset(cfg, neh, norm="imagenet"), batch_size=16, num_workers=0)
    kf, kx = kermany_tensor(cfg, log)
    kl = DataLoader(TensorDataset(kx, torch.arange(len(kx))), batch_size=16)

    for variant in VARIANTS:
        rng = np.random.default_rng(cfg["seed"])
        bank = build_bank(net, cfg, variant, device, log, rng)
        t0 = time.time()
        neh[f"pc_{variant}"] = score_loader(net, bank, nl, variant, device, len(neh))
        kf[f"pc_{variant}"] = score_loader(net, bank, kl, variant, device, len(kf))
        log.info(f"{variant}: scored {len(neh)} NEH + {len(kf)} Kermany B-scans ({time.time() - t0:.0f}s)")
        del bank
        torch.cuda.empty_cache()
    neh.to_csv(OUT / "neh_patchcore_scores.csv", index=False)
    kf.to_csv(OUT / "kermany_patchcore_scores.csv", index=False)
    log.info(f"-> {OUT / 'neh_patchcore_scores.csv'}, {OUT / 'kermany_patchcore_scores.csv'}")


if __name__ == "__main__":
    main()
