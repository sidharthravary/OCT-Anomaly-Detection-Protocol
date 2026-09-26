"""09_train_vae.py - M3, convolutional VAE: train on clean-normal B-scans, then score val / test.

Same encoder / decoder as M1, 256-d latent (fully connected from the 7 x 7 map).
Loss per sample = sum over pixels of squared error + beta * KL, beta = 1 with a linear warm-up over the
first 10 epochs. (The squared error is summed, not averaged, so that it is on the same scale as the KL
summed over latent dimensions; see outputs/logs/changes.md.)
Adam lr 1e-3, batch 32, up to 100 epochs, early stopping (patience 10) on the val-normal loss at beta = 1,
counted only after the warm-up.

Scores (posterior mean, no sampling):
  score    = mean squared reconstruction error per pixel (comparable to M1)
  score_kl = summed squared error + KL (negative-ELBO-style)
Excluded layout-outlier patients are scored too (split "excluded"). Only validation metrics are printed.

  --smoke   2 epochs x 5 batches, writes models/vae_smoke.pt, no scoring

Outputs: models/vae.pt, outputs/scores/vae_scores.csv, outputs/metrics/vae_history.csv,
outputs/figures/vae_loss.png, outputs/figures/vae_reconstructions.png
"""
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import (CLASSES, OCTDataset, get_device, get_split_frame, load_config, scoring_frame, set_seed,
                    setup_logging, val_summary)
from nets import ConvVAE, kl_divergence

HP = dict(lr=1e-3, batch_size=32, max_epochs=100, patience=10, latent=256, beta=1.0, warmup_epochs=10)


def vae_terms(model, x):
    x_hat, mu, logvar = model(x)
    sse = ((x_hat - x) ** 2).flatten(1).sum(1)
    return x_hat, sse, kl_divergence(mu, logvar)


def run_epoch(model, loader, device, beta, opt=None, max_batches=None):
    model.train(opt is not None)
    tot = dict(loss=0.0, sse=0.0, kl=0.0)
    n = 0
    with torch.set_grad_enabled(opt is not None):
        for b, (x, _) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            x = x.to(device)
            _, sse, kl = vae_terms(model, x)
            loss = (sse + beta * kl).mean()
            if opt is not None:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            tot["loss"] += loss.item() * len(x)
            tot["sse"] += sse.sum().item()
            tot["kl"] += kl.sum().item()
            n += len(x)
    return {k: v / n for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = "vae_smoke" if args.smoke else "vae"

    cfg = load_config()
    set_seed(cfg["seed"])
    log = setup_logging(f"09_train_{tag}", cfg)
    device = get_device(cfg["device"])
    out = Path(cfg["output_dir"])
    for d in ("metrics", "figures", "scores"):
        (out / d).mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    mb = 5 if args.smoke else None

    train_df, val_df = get_split_frame(cfg, "train"), get_split_frame(cfg, "val")
    g = torch.Generator().manual_seed(cfg["seed"])
    train_dl = DataLoader(OCTDataset(cfg, train_df, norm="unit", augment=True), batch_size=HP["batch_size"],
                          shuffle=True, generator=g, num_workers=0)
    val_dl = DataLoader(OCTDataset(cfg, val_df, norm="unit"), batch_size=HP["batch_size"], shuffle=False,
                        num_workers=0)
    log.info(f"device {device} | train {len(train_df)} | val {len(val_df)} | {HP}")

    model = ConvVAE(HP["latent"]).to(device)
    log.info(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"])
    ckpt = Path("models") / f"{tag}.pt"
    best, best_epoch, wait, hist = float("inf"), 0, 0, []
    for epoch in range(1, (2 if args.smoke else HP["max_epochs"]) + 1):
        t0 = time.time()
        beta = HP["beta"] * min(1.0, epoch / HP["warmup_epochs"])
        tr = run_epoch(model, train_dl, device, beta, opt, mb)
        va = run_epoch(model, val_dl, device, HP["beta"], None, mb)
        hist.append(dict(epoch=epoch, beta=beta, train_loss=tr["loss"], train_sse=tr["sse"], train_kl=tr["kl"],
                         val_loss=va["loss"], val_sse=va["sse"], val_kl=va["kl"], seconds=time.time() - t0))
        warm = epoch >= HP["warmup_epochs"] or args.smoke
        improved = warm and va["loss"] < best
        if improved:
            best, best_epoch, wait = va["loss"], epoch, 0
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch, val_loss=va["loss"], hp=HP), ckpt)
        elif warm:
            wait += 1
        log.info(f"epoch {epoch:3d}  beta {beta:.2f}  train {tr['loss']:.2f} (sse {tr['sse']:.2f}, kl {tr['kl']:.2f})"
                 f"  val {va['loss']:.2f} (sse {va['sse']:.2f}, kl {va['kl']:.2f})  {time.time() - t0:5.1f}s"
                 + ("  *" if improved else ""))
        if wait >= HP["patience"]:
            log.info(f"early stop: no val improvement for {HP['patience']} epochs after warm-up")
            break

    h = pd.DataFrame(hist)
    h.to_csv(out / "metrics" / f"{tag}_history.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, col, ttl in ((axes[0], "sse", "summed squared error"), (axes[1], "kl", "KL divergence")):
        ax.plot(h.epoch, h[f"train_{col}"], color="#2a78d6", linewidth=2, label="train (augmented)")
        ax.plot(h.epoch, h[f"val_{col}"], color="#eb6834", linewidth=2, label="val normal")
        ax.axvline(best_epoch, color="#52514e", linewidth=1, linestyle="--")
        ax.set_xlabel("epoch")
        ax.set_title(ttl, fontsize=10, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(out / "figures" / f"{tag}_loss.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"best val loss {best:.2f} at epoch {best_epoch} -> {ckpt}; training {h.seconds.sum() / 60:.1f} min")
    if args.smoke:
        return

    # ------------------------------------------------------------- scoring
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    model.eval()
    df = scoring_frame(cfg)
    dl = DataLoader(OCTDataset(cfg, df, norm="unit"), batch_size=64, shuffle=False, num_workers=0)
    s_mse, s_kl = np.zeros(len(df)), np.zeros(len(df))
    with torch.no_grad():
        for x, i in dl:
            x = x.to(device)
            _, sse, kl = vae_terms(model, x)
            s_mse[i.numpy()] = (sse / x[0].numel()).cpu().numpy()
            s_kl[i.numpy()] = (sse + kl).cpu().numpy()
    df["score"], df["score_kl"] = s_mse, s_kl
    df.to_csv(out / "scores" / "vae_scores.csv", index=False)
    log.info(f"scored {len(df)} B-scans -> {out / 'scores' / 'vae_scores.csv'}")
    val_summary(df, cfg, log, "score")
    val_summary(df, cfg, log, "score_kl")

    # reconstructions: 4 val images per class folder
    rng = np.random.default_rng(cfg["seed"])
    pick = pd.concat([df[(df.split == "val") & (df.class_folder == c)].sample(4, random_state=int(rng.integers(1e6)))
                      for c in CLASSES])
    ds = OCTDataset(cfg, pick, norm="unit")
    x = torch.stack([ds[k][0] for k in range(len(ds))]).to(device)
    with torch.no_grad():
        x_hat, _, _ = model(x)
    fig, axes = plt.subplots(2, len(pick), figsize=(len(pick) * 1.5, 3.4))
    for k in range(len(pick)):
        axes[0, k].imshow(x[k, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[0, k].set_title(f"{pick.class_folder.iloc[k]}/{pick.bscan_label.iloc[k]}", fontsize=6)
        axes[1, k].imshow(x_hat[k, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[0, k].axis("off")
        axes[1, k].axis("off")
    fig.suptitle("VAE: val inputs (top) and reconstructions from the posterior mean (bottom)", fontsize=10,
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "figures" / "vae_reconstructions.png", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
