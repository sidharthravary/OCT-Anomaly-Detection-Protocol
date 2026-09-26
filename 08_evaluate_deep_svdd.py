"""08_evaluate_deep_svdd.py - score val / test B-scans with both Deep SVDD variants (M2).

svdd_frozen: ||z - c||^2 on the frozen 768-d ResNet18 embedding
svdd_ft:     ||z - c||^2 on the fine-tuned 128-d embedding
Heatmaps: frozen layer3 map (256 x 14 x 14); per location, squared distance to the mean train feature at
that location; upsampled to 224 and overlaid.
Excluded layout-outlier patients are scored too (split "excluded"). Only validation metrics are printed.

Outputs: outputs/scores/svdd_frozen_scores.csv, outputs/scores/svdd_ft_scores.csv,
outputs/figures/svdd_heatmaps/<CLASS>.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (CLASSES, IMAGENET_MEAN, IMAGENET_STD, OCTDataset, get_device, load_config, require,
                    scoring_frame, set_seed, setup_logging, val_summary)
from nets import DeepSVDD, ResNetFeatures, frozen_embedding

N_HEATMAPS = 8


def location_distance(l3, layer3_mean):
    """B x 14 x 14 squared distance of each layer3 feature vector to the train mean at that location."""
    return ((l3 - layer3_mean[None]) ** 2).sum(1)


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    log = setup_logging("08_evaluate_deep_svdd", cfg)
    device = get_device(cfg["device"])
    out = Path(cfg["output_dir"])
    (out / "scores").mkdir(parents=True, exist_ok=True)
    heat_dir = out / "figures" / "svdd_heatmaps"
    heat_dir.mkdir(parents=True, exist_ok=True)

    centers = torch.load(require(Path("models") / "svdd_centers.pt", "Run 07_train_deep_svdd.py first."),
                         map_location=device)
    ckpt = torch.load(require(Path("models") / "svdd_ft.pt", "Run 07_train_deep_svdd.py first."), map_location=device)
    frozen = ResNetFeatures().to(device).eval()
    ft = DeepSVDD(ckpt["hp"]["rep_dim"]).to(device)
    ft.load_state_dict(ckpt["state_dict"])
    ft.eval()
    c_frozen, c_ft, l3_mean = centers["frozen_c"], centers["ft_c"], centers["layer3_mean"]
    log.info(f"loaded models/svdd_ft.pt (epoch {ckpt['epoch']}, val mean distance {ckpt['val_dist']:.6f})")

    df = scoring_frame(cfg)
    dl = DataLoader(OCTDataset(cfg, df, norm="imagenet"), batch_size=64, shuffle=False, num_workers=0)
    s_frozen, s_ft = np.zeros(len(df)), np.zeros(len(df))
    with torch.no_grad():
        for x, i in dl:
            x = x.to(device)
            l3, l4 = frozen(x)
            s_frozen[i.numpy()] = ((frozen_embedding(l3, l4) - c_frozen) ** 2).sum(1).cpu().numpy()
            s_ft[i.numpy()] = ((ft(x) - c_ft) ** 2).sum(1).cpu().numpy()

    for name, s in (("svdd_frozen", s_frozen), ("svdd_ft", s_ft)):
        d = df.assign(score=s)
        d.to_csv(out / "scores" / f"{name}_scores.csv", index=False)
        log.info(f"{name}: scored {len(d)} B-scans -> {out / 'scores' / f'{name}_scores.csv'}")
        val_summary(d, cfg, log, "score")

    # ------------------------------------------------ heatmaps (test, per class)
    rng = np.random.default_rng(cfg["seed"])
    df["score"] = s_frozen
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    maps_all = {}
    for cls in CLASSES:
        pool = df[(df.split == "test") & (df.class_folder == cls)]
        pool = pool[(pool.bscan_label != "normal") if cls != "NORMAL" else (pool.bscan_label == "normal")]
        pick = pool.iloc[rng.choice(len(pool), N_HEATMAPS, replace=False)]
        ds = OCTDataset(cfg, pick, norm="imagenet")
        x = torch.stack([ds[k][0] for k in range(len(ds))]).to(device)
        with torch.no_grad():
            l3, _ = frozen(x)
            dmap = F.interpolate(location_distance(l3, l3_mean)[:, None], size=x.shape[-2:], mode="bilinear",
                                 align_corners=False)[:, 0].cpu().numpy()
        maps_all[cls] = (pick, (x.cpu() * std + mean)[:, 0].numpy(), dmap)
    vmax = np.percentile(np.concatenate([m[2].ravel() for m in maps_all.values()]), 99.5)
    vmin = np.percentile(maps_all["NORMAL"][2], 5)
    for cls, (pick, imgs, dmap) in maps_all.items():
        fig, axes = plt.subplots(2, N_HEATMAPS, figsize=(N_HEATMAPS * 1.9, 4.2))
        for k in range(N_HEATMAPS):
            axes[0, k].imshow(imgs[k], cmap="gray", vmin=0, vmax=1)
            axes[0, k].set_title(f"{pick.bscan_label.iloc[k]}  s={pick.score.iloc[k]:.3f}", fontsize=6)
            axes[1, k].imshow(imgs[k], cmap="gray", vmin=0, vmax=1)
            axes[1, k].imshow(dmap[k], cmap="inferno", alpha=0.5, vmin=vmin, vmax=vmax)
            axes[0, k].axis("off")
            axes[1, k].axis("off")
        fig.suptitle(f"svdd_frozen layer3 distance map (shared colour scale): test B-scans, {cls} folder",
                     fontsize=10, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(heat_dir / f"{cls}.png", dpi=200, facecolor="white")
        plt.close(fig)
    log.info(f"heatmaps -> {heat_dir}")


if __name__ == "__main__":
    main()
