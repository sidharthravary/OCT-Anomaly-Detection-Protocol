"""06_evaluate_autoencoder.py - score val / test B-scans with the trained autoencoder (M1).

score      = mean squared error per pixel
score_ssim = 1 - SSIM
Excluded layout-outlier patients are scored too (split "excluded").
Only validation metrics are printed; test metrics are computed in script 10.

  --loss mse | mse_ssim   which checkpoint to score (models/ae.pt or models/ae_mse_ssim.pt)

Outputs: outputs/scores/<tag>_scores.csv, outputs/figures/<tag>_heatmaps/<CLASS>.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

from common import (CLASSES, OCTDataset, get_device, load_config, require, scoring_frame, set_seed, setup_logging,
                    val_summary)
from nets import ConvAutoencoder, ssim

N_HEATMAPS = 8


@torch.no_grad()
def reconstruct(model, loader, device):
    """Yield (indices, x, x_hat) batches."""
    for x, i in loader:
        x = x.to(device)
        yield i.numpy(), x, model(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["mse", "mse_ssim"], default="mse")
    args = ap.parse_args()
    tag = "ae" if args.loss == "mse" else f"ae_{args.loss}"

    cfg = load_config()
    set_seed(cfg["seed"])
    log = setup_logging(f"06_evaluate_{tag}", cfg)
    device = get_device(cfg["device"])
    out = Path(cfg["output_dir"])
    (out / "scores").mkdir(parents=True, exist_ok=True)
    heat_dir = out / "figures" / f"{tag}_heatmaps"
    heat_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(require(Path("models") / f"{tag}.pt", "Run 05_train_autoencoder.py first."), map_location=device)
    model = ConvAutoencoder().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log.info(f"loaded models/{tag}.pt (epoch {ckpt['epoch']}, val loss {ckpt['val_loss']:.6f})")

    df = scoring_frame(cfg)
    dl = DataLoader(OCTDataset(cfg, df, norm="unit"), batch_size=64, shuffle=False, num_workers=0)
    mse, one_minus_ssim = np.zeros(len(df)), np.zeros(len(df))
    for i, x, x_hat in reconstruct(model, dl, device):
        mse[i] = ((x_hat - x) ** 2).flatten(1).mean(1).cpu().numpy()
        one_minus_ssim[i] = (1 - ssim(x_hat, x)).cpu().numpy()
    df["score"], df["score_ssim"] = mse, one_minus_ssim
    df.to_csv(out / "scores" / f"{tag}_scores.csv", index=False)
    log.info(f"scored {len(df)} B-scans -> {out / 'scores' / f'{tag}_scores.csv'}  "
             f"({df.split.value_counts().to_dict()})")

    val_summary(df, cfg, log, "score")
    val_summary(df, cfg, log, "score_ssim")

    # ---------------------------------------- residual heatmaps (test, per class)
    rng = np.random.default_rng(cfg["seed"])
    for cls in CLASSES:
        pool = df[(df.split == "test") & (df.class_folder == cls)]
        # for diseased classes show abnormal-labelled slices; NORMAL shows normal ones
        pool = pool[(pool.bscan_label != "normal") if cls != "NORMAL" else (pool.bscan_label == "normal")]
        pick = pool.iloc[rng.choice(len(pool), N_HEATMAPS, replace=False)]
        ds = OCTDataset(cfg, pick, norm="unit")
        x = torch.stack([ds[k][0] for k in range(len(ds))]).to(device)
        with torch.no_grad():
            x_hat = model(x)
        res = ((x - x_hat) ** 2).squeeze(1).cpu().numpy()
        res = np.stack([ndimage.gaussian_filter(r, 2) for r in res])
        vmax = np.percentile(res, 99.5)
        fig, axes = plt.subplots(3, N_HEATMAPS, figsize=(N_HEATMAPS * 1.9, 6))
        for k in range(N_HEATMAPS):
            img = x[k, 0].cpu().numpy()
            axes[0, k].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[0, k].set_title(f"{pick.bscan_label.iloc[k]}  s={pick.score.iloc[k]:.4f}", fontsize=6)
            axes[1, k].imshow(x_hat[k, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[2, k].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[2, k].imshow(res[k], cmap="inferno", alpha=0.55, vmin=0, vmax=vmax)
            for r in range(3):
                axes[r, k].axis("off")
        for r, name in enumerate(["input", "reconstruction", "squared residual"]):
            axes[r, 0].text(-0.08, 0.5, name, transform=axes[r, 0].transAxes, rotation=90, ha="right", va="center",
                            fontsize=8)
        fig.suptitle(f"{tag}: test B-scans, {cls} folder", fontsize=10, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(heat_dir / f"{cls}.png", dpi=200, facecolor="white")
        plt.close(fig)
    log.info(f"heatmaps -> {heat_dir}")


if __name__ == "__main__":
    main()
