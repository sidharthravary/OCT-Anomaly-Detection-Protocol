"""05_train_autoencoder.py - M1, convolutional autoencoder trained on clean-normal B-scans.

Train: clean-normal B-scans of train patients, with augmentation.
Early stopping: reconstruction loss on clean-normal B-scans of val patients (no augmentation).
Adam lr 1e-3, batch 32, up to 100 epochs, patience 10 (PRD section 7).

  --loss mse       (default)  MSE
  --loss mse_ssim             0.5 * MSE + 0.5 * (1 - SSIM)   (PRD optional variant)
  --smoke                     2 epochs x 5 batches, writes models/ae_smoke.pt (pipeline check)

Outputs: models/ae.pt (or ae_<loss>.pt for the variant), outputs/metrics/ae_history.csv,
outputs/figures/ae_loss.png
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
from nets import ConvAutoencoder, ssim

HP = dict(lr=1e-3, batch_size=32, max_epochs=100, patience=10)


def recon_loss(x_hat, x, kind):
    mse = ((x_hat - x) ** 2).flatten(1).mean(dim=1)
    if kind == "mse":
        return mse
    return 0.5 * mse + 0.5 * (1 - ssim(x_hat, x))


def run_epoch(model, loader, device, kind, opt=None, max_batches=None):
    model.train(opt is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for b, (x, _) in enumerate(loader):
            if max_batches is not None and b >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            loss = recon_loss(model(x), x, kind).mean()
            if opt is not None:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            total += loss.item() * len(x)
            n += len(x)
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["mse", "mse_ssim"], default="mse")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    tag = "ae" if args.loss == "mse" else f"ae_{args.loss}"
    if args.smoke:
        tag += "_smoke"
    log = setup_logging(f"05_train_{tag}", cfg)
    device = get_device(cfg["device"])
    out = Path(cfg["output_dir"])
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    train_df, val_df = get_split_frame(cfg, "train"), get_split_frame(cfg, "val")
    g = torch.Generator().manual_seed(cfg["seed"])
    train_dl = DataLoader(OCTDataset(cfg, train_df, norm="unit", augment=True), batch_size=HP["batch_size"],
                          shuffle=True, generator=g, num_workers=0, pin_memory=device.type == "cuda")
    val_dl = DataLoader(OCTDataset(cfg, val_df, norm="unit"), batch_size=HP["batch_size"], shuffle=False,
                        num_workers=0)
    log.info(f"device {device} | train {len(train_df)} clean-normal B-scans ({train_df.patient_id.nunique()} patients)"
             f" | val {len(val_df)} ({val_df.patient_id.nunique()} patients) | loss {args.loss} | {HP}")

    model = ConvAutoencoder().to(device)
    log.info(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"])

    max_epochs, max_batches = (2, 5) if args.smoke else (HP["max_epochs"], None)
    best, best_epoch, wait, hist = float("inf"), 0, 0, []
    ckpt = Path("models") / f"{tag}.pt"
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_dl, device, args.loss, opt, max_batches)
        va = run_epoch(model, val_dl, device, args.loss, None, max_batches)
        hist.append(dict(epoch=epoch, train_loss=tr, val_loss=va, seconds=time.time() - t0))
        improved = va < best
        if improved:
            best, best_epoch, wait = va, epoch, 0
            torch.save(dict(state_dict=model.state_dict(), epoch=epoch, val_loss=va, loss=args.loss, hp=HP), ckpt)
        else:
            wait += 1
        log.info(f"epoch {epoch:3d}  train {tr:.6f}  val {va:.6f}  {time.time() - t0:5.1f}s" + ("  *" if improved else ""))
        if wait >= HP["patience"]:
            log.info(f"early stop: no val improvement for {HP['patience']} epochs")
            break

    h = pd.DataFrame(hist)
    h.to_csv(out / "metrics" / f"{tag}_history.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(h.epoch, h.train_loss, color="#2a78d6", linewidth=2, label="train (augmented)")
    ax.plot(h.epoch, h.val_loss, color="#eb6834", linewidth=2, label="val normal")
    ax.axvline(best_epoch, color="#52514e", linewidth=1, linestyle="--")
    ax.text(best_epoch, ax.get_ylim()[1], f" best epoch {best_epoch}", va="top", fontsize=8, color="#52514e")
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"reconstruction loss ({args.loss})")
    ax.set_title(f"Autoencoder ({args.loss}) training curve", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    fig.savefig(out / "figures" / f"{tag}_loss.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"best val loss {best:.6f} at epoch {best_epoch} -> {ckpt}")
    log.info(f"total training time {h.seconds.sum() / 60:.1f} min")


if __name__ == "__main__":
    main()
