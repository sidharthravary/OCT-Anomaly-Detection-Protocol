"""07_train_deep_svdd.py - M2, Deep SVDD on an ImageNet ResNet18 (primary method).

1. svdd_frozen: no training. GAP(layer3) ++ GAP(layer4) of the frozen ResNet18 (768-d, L2-normalised);
   centre c = mean train embedding; score = ||z - c||^2 (computed in 08).
   Also stores the per-location mean of the train layer3 maps (256 x 14 x 14) for the 08 heatmaps.
2. svdd_ft: ResNet18 + bias-free linear head to 128-d, frozen up to layer2. c from an initial pass over
   train, |c_i| < 0.01 set to +-0.01 (Ruff et al. 2018). Loss = mean ||z - c||^2. Adam lr 1e-4,
   weight decay 1e-6, up to 50 epochs, early stopping (patience 10) on the val-normal mean distance.

Train = clean-normal B-scans of train patients (augmented for svdd_ft only); val = clean-normal of val patients.

  --smoke   2 epochs x 5 batches, writes models/*_smoke.pt

Outputs: models/svdd_ft.pt, models/svdd_centers.pt, outputs/metrics/svdd_ft_history.csv,
outputs/figures/svdd_ft_loss.png
"""
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import OCTDataset, get_device, get_split_frame, load_config, set_seed, setup_logging
from nets import DeepSVDD, ResNetFeatures, frozen_embedding

HP = dict(lr=1e-4, weight_decay=1e-6, batch_size=32, max_epochs=50, patience=10, rep_dim=128, eps=0.01)


@torch.no_grad()
def frozen_pass(loader, device, max_batches=None):
    """Mean frozen embedding (centre c) and per-location mean of the layer3 maps."""
    net = ResNetFeatures().to(device).eval()
    z_sum, l3_sum, n = 0, 0, 0
    for b, (x, _) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        l3, l4 = net(x.to(device))
        z_sum = z_sum + frozen_embedding(l3, l4).sum(0)
        l3_sum = l3_sum + l3.sum(0)
        n += len(x)
    return z_sum / n, l3_sum / n


@torch.no_grad()
def mean_distance(model, loader, c, device, max_batches=None):
    model.eval()
    total, n = 0.0, 0
    for b, (x, _) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        z = model(x.to(device))
        total += ((z - c) ** 2).sum(1).sum().item()
        n += len(x)
    return total / n


@torch.no_grad()
def init_center(model, loader, device, eps, max_batches=None):
    model.eval()
    s, n = 0, 0
    for b, (x, _) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        s = s + model(x.to(device)).sum(0)
        n += len(x)
    c = s / n
    small = c.abs() < eps
    c[small & (c < 0)] = -eps
    c[small & (c >= 0)] = eps
    return c, int(small.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sfx = "_smoke" if args.smoke else ""

    cfg = load_config()
    set_seed(cfg["seed"])
    log = setup_logging(f"07_train_deep_svdd{sfx}", cfg)
    device = get_device(cfg["device"])
    out = Path(cfg["output_dir"])
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    mb = 5 if args.smoke else None

    train_df, val_df = get_split_frame(cfg, "train"), get_split_frame(cfg, "val")
    plain_train = DataLoader(OCTDataset(cfg, train_df, norm="imagenet"), batch_size=64, shuffle=False, num_workers=0)
    val_dl = DataLoader(OCTDataset(cfg, val_df, norm="imagenet"), batch_size=64, shuffle=False, num_workers=0)
    log.info(f"device {device} | train {len(train_df)} clean-normal B-scans | val {len(val_df)} | {HP}")

    # ------------------------------------------------------------ 1. frozen
    t0 = time.time()
    c_frozen, layer3_mean = frozen_pass(plain_train, device, mb)
    log.info(f"svdd_frozen: centre from {len(train_df) if mb is None else mb * 64} train embeddings (768-d), "
             f"layer3 mean map {tuple(layer3_mean.shape)}  [{time.time() - t0:.0f}s]")

    # --------------------------------------------------------- 2. fine-tuned
    model = DeepSVDD(HP["rep_dim"]).to(device)
    c_ft, n_small = init_center(model, plain_train, device, HP["eps"], mb)
    log.info(f"svdd_ft: initial centre, {n_small} of {HP['rep_dim']} coordinates clamped to +-{HP['eps']}; "
             f"trainable parameters {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    g = torch.Generator().manual_seed(cfg["seed"])
    train_dl = DataLoader(OCTDataset(cfg, train_df, norm="imagenet", augment=True), batch_size=HP["batch_size"],
                          shuffle=True, generator=g, num_workers=0)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=HP["lr"],
                           weight_decay=HP["weight_decay"])
    best = mean_distance(model, val_dl, c_ft, device, mb)
    log.info(f"epoch   0  val mean distance {best:.6f}")
    best_epoch, wait, hist = 0, 0, []
    ckpt = Path("models") / f"svdd_ft{sfx}.pt"
    torch.save(dict(state_dict=model.state_dict(), epoch=0, val_dist=best, hp=HP), ckpt)
    for epoch in range(1, (2 if args.smoke else HP["max_epochs"]) + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for b, (x, _) in enumerate(train_dl):
            if mb is not None and b >= mb:
                break
            z = model(x.to(device))
            loss = ((z - c_ft) ** 2).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
            n += len(x)
        va = mean_distance(model, val_dl, c_ft, device, mb)
        hist.append(dict(epoch=epoch, train_dist=total / n, val_dist=va, seconds=time.time() - t0))
        improved = va < best
        if improved:
            best, best_epoch, wait = va, epoch, 0
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch, val_dist=va, hp=HP), ckpt)
        else:
            wait += 1
        log.info(f"epoch {epoch:3d}  train {total / n:.6f}  val {va:.6f}  {time.time() - t0:5.1f}s"
                 + ("  *" if improved else ""))
        if wait >= HP["patience"]:
            log.info(f"early stop: no val improvement for {HP['patience']} epochs")
            break

    torch.save(dict(frozen_c=c_frozen.cpu(), layer3_mean=layer3_mean.cpu(), ft_c=c_ft.cpu()),
               Path("models") / f"svdd_centers{sfx}.pt")
    h = pd.DataFrame(hist)
    h.to_csv(out / "metrics" / f"svdd_ft{sfx}_history.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(h.epoch, h.train_dist, color="#2a78d6", linewidth=2, label="train (augmented)")
    ax.plot(h.epoch, h.val_dist, color="#eb6834", linewidth=2, label="val normal")
    ax.axvline(best_epoch, color="#52514e", linewidth=1, linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean ||z - c||^2")
    ax.set_title("Deep SVDD (fine-tuned) training curve", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    fig.savefig(out / "figures" / f"svdd_ft{sfx}_loss.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"best val mean distance {best:.6f} at epoch {best_epoch} -> {ckpt}, models/svdd_centers{sfx}.pt")
    if hist:
        log.info(f"total training time {h.seconds.sum() / 60:.1f} min")


if __name__ == "__main__":
    main()
