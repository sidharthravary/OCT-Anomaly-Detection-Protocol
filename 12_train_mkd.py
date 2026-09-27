"""12_train_mkd.py - M4 (stretch), multiresolution knowledge distillation (Salehi et al., CVPR 2021).

Teacher: ImageNet VGG-16, frozen. Student: smaller VGG-style network trained on clean-normal B-scans to
reproduce the teacher's activations at 4 critical layers (relu2_2, relu3_3, relu4_3, relu5_3).
Loss = L_val (squared distance) + lambda * L_dir (1 - cosine), lambda = 0.01. The anomaly score is the
same loss at test time. Re-implemented from the paper (nets.py); same splits, cache and score format,
so 10 / 11 pick it up automatically.

Adam lr 1e-3, batch 32, up to 100 epochs, early stopping (patience 10) on the val-normal loss.
Forward passes use bfloat16 autocast on CUDA (4 GB GPU); the loss is computed in float32.

  --smoke   2 epochs x 5 batches, writes models/mkd_smoke.pt, no scoring

Outputs: models/mkd.pt, outputs/scores/mkd_scores.csv, outputs/metrics/mkd_history.csv,
outputs/figures/mkd_loss.png, outputs/figures/mkd_heatmaps/<CLASS>.png
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

from common import (CLASSES, IMAGENET_MEAN, IMAGENET_STD, OCTDataset, get_device, get_split_frame, load_config,
                    scoring_frame, set_seed, setup_logging, val_summary)
from nets import VGGStudent, VGGTeacher, mkd_loss, mkd_map

HP = dict(lr=1e-3, batch_size=32, max_epochs=100, patience=10, lam=0.01)


def amp(device):
    """bfloat16 autocast on CUDA: keeps VGG-16 teacher + student within a 4 GB GPU at batch 32."""
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def run_epoch(student, teacher, loader, device, opt=None, max_batches=None):
    student.train(opt is not None)
    total, n = 0.0, 0
    for b, (x, _) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x = x.to(device)
        with amp(device):
            with torch.no_grad():
                t = teacher(x)
            with torch.set_grad_enabled(opt is not None):
                feats = student(x)
        loss = mkd_loss(feats, t, HP["lam"]).mean()
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        total += loss.item() * len(x)
        n += len(x)
    return total / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = "mkd_smoke" if args.smoke else "mkd"

    cfg = load_config()
    set_seed(cfg["seed"])
    log = setup_logging(f"12_train_{tag}", cfg)
    device = get_device(cfg["device"])
    out = Path(cfg["output_dir"])
    for d in ("metrics", "figures", "scores"):
        (out / d).mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    mb = 5 if args.smoke else None

    train_df, val_df = get_split_frame(cfg, "train"), get_split_frame(cfg, "val")
    g = torch.Generator().manual_seed(cfg["seed"])
    train_dl = DataLoader(OCTDataset(cfg, train_df, norm="imagenet", augment=True), batch_size=HP["batch_size"],
                          shuffle=True, generator=g, num_workers=0)
    val_dl = DataLoader(OCTDataset(cfg, val_df, norm="imagenet"), batch_size=HP["batch_size"], num_workers=0)
    teacher, student = VGGTeacher().to(device), VGGStudent().to(device)
    log.info(f"device {device} | train {len(train_df)} | val {len(val_df)} | student params "
             f"{sum(p.numel() for p in student.parameters()):,} | {HP}")
    opt = torch.optim.Adam(student.parameters(), lr=HP["lr"])

    ckpt = Path("models") / f"{tag}.pt"
    best, best_epoch, wait, hist = float("inf"), 0, 0, []
    for epoch in range(1, (2 if args.smoke else HP["max_epochs"]) + 1):
        t0 = time.time()
        tr = run_epoch(student, teacher, train_dl, device, opt, mb)
        va = run_epoch(student, teacher, val_dl, device, None, mb)
        hist.append(dict(epoch=epoch, train_loss=tr, val_loss=va, seconds=time.time() - t0))
        improved = va < best
        if improved:
            best, best_epoch, wait = va, epoch, 0
            torch.save(dict(state_dict=student.state_dict(), epoch=epoch, val_loss=va, hp=HP), ckpt)
        else:
            wait += 1
        log.info(f"epoch {epoch:3d}  train {tr:.5f}  val {va:.5f}  {time.time() - t0:5.1f}s" + ("  *" if improved else ""))
        if wait >= HP["patience"]:
            log.info(f"early stop: no val improvement for {HP['patience']} epochs")
            break

    h = pd.DataFrame(hist)
    h.to_csv(out / "metrics" / f"{tag}_history.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(h.epoch, h.train_loss, color="#2a78d6", linewidth=2, label="train (augmented)")
    ax.plot(h.epoch, h.val_loss, color="#eb6834", linewidth=2, label="val normal")
    ax.axvline(best_epoch, color="#52514e", linewidth=1, linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("L_val + 0.01 L_dir")
    ax.set_title("MKD student training curve", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    fig.savefig(out / "figures" / f"{tag}_loss.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"best val loss {best:.5f} at epoch {best_epoch} -> {ckpt}; training {h.seconds.sum() / 60:.1f} min")
    if args.smoke:
        return

    # ------------------------------------------------------------- scoring
    student.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    student.eval()
    df = scoring_frame(cfg)
    s = np.zeros(len(df))
    with torch.no_grad():
        for x, i in DataLoader(OCTDataset(cfg, df, norm="imagenet"), batch_size=64, num_workers=0):
            x = x.to(device)
            with amp(device):
                fs, ft = student(x), teacher(x)
            s[i.numpy()] = mkd_loss(fs, ft, HP["lam"]).cpu().numpy()
    df["score"] = s
    df.to_csv(out / "scores" / "mkd_scores.csv", index=False)
    log.info(f"scored {len(df)} B-scans -> {out / 'scores' / 'mkd_scores.csv'}")
    val_summary(df, cfg, log, "score")

    # ------------------------------------------------ heatmaps (test, per class)
    heat_dir = out / "figures" / "mkd_heatmaps"
    heat_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    mean, std = torch.tensor(IMAGENET_MEAN).view(3, 1, 1), torch.tensor(IMAGENET_STD).view(3, 1, 1)
    maps = {}
    for cls in CLASSES:
        pool = df[(df.split == "test") & (df.class_folder == cls)]
        pool = pool[(pool.bscan_label != "normal") if cls != "NORMAL" else (pool.bscan_label == "normal")]
        pick = pool.iloc[rng.choice(len(pool), 8, replace=False)]
        ds = OCTDataset(cfg, pick, norm="imagenet")
        x = torch.stack([ds[k][0] for k in range(len(ds))]).to(device)
        with torch.no_grad(), amp(device):
            fs, ft = student(x), teacher(x)
        m = mkd_map(fs, ft, x.shape[-2:]).cpu().numpy()
        maps[cls] = (pick, (x.cpu() * std + mean)[:, 0].numpy(), m)
    vmin, vmax = np.percentile(np.concatenate([v[2].ravel() for v in maps.values()]), [5, 99.5])
    for cls, (pick, imgs, m) in maps.items():
        fig, axes = plt.subplots(2, 8, figsize=(15.2, 4.2))
        for k in range(8):
            axes[0, k].imshow(imgs[k], cmap="gray", vmin=0, vmax=1)
            axes[0, k].set_title(f"{pick.bscan_label.iloc[k]}  s={pick.score.iloc[k]:.3f}", fontsize=6)
            axes[1, k].imshow(imgs[k], cmap="gray", vmin=0, vmax=1)
            axes[1, k].imshow(m[k], cmap="inferno", alpha=0.5, vmin=vmin, vmax=vmax)
            axes[0, k].axis("off")
            axes[1, k].axis("off")
        fig.suptitle(f"MKD teacher-student discrepancy (shared colour scale): test B-scans, {cls} folder",
                     fontsize=10, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(heat_dir / f"{cls}.png", dpi=200, facecolor="white")
        plt.close(fig)
    log.info(f"heatmaps -> {heat_dir}")


if __name__ == "__main__":
    main()
