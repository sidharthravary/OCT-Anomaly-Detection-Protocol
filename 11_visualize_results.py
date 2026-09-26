"""11_visualize_results.py - report figures from the scores and the comparison table (runs after 10).

  roc_pr_bscan.png / roc_pr_case.png   ROC and PR curves, every method overlaid, one figure per level
  score_distributions.png              test score histograms for normal / drusen / cnv B-scans, per method
  confusion_matrices.png               confusion matrices at the primary threshold, per method and level
  best_method_heatmaps.png             3 NORMAL, 3 DRUSEN, 3 CNV test B-scans with the best method's anomaly map

All figures: outputs/figures/, PNG, 200 dpi. "Best" = highest B-scan PR-AUC in comparison_table.csv.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.metrics import precision_recall_curve, roc_curve

from common import (CLASSES, IMAGENET_MEAN, IMAGENET_STD, OCTDataset, aggregate_case_scores, get_device,
                    load_config, require, set_seed, setup_logging)

# categorical slots 1-5 of the validated reference palette, fixed order
METHOD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
LABEL_COLORS = {"normal": "#2a78d6", "drusen": "#eb6834", "cnv": "#1baf7a"}
INK2, GRID = "#52514e", "#e4e3df"
NAMES = {"ae": "Autoencoder (M1)", "ae_mse_ssim": "AE, MSE+SSIM", "svdd_frozen": "Deep SVDD frozen (M2)",
         "svdd_ft": "Deep SVDD fine-tuned (M2)", "vae": "VAE (M3)", "mkd": "MKD (M4)"}


def style(ax, grid="both"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def anomaly_maps(method, pick, cfg, device):
    """Return (images in [0,1], anomaly maps) for the picked B-scans, using the method's own mechanism."""
    if method.startswith("ae") or method == "vae":
        from nets import ConvAutoencoder, ConvVAE
        tag = method if method != "vae" else "vae"
        model = ConvVAE() if method == "vae" else ConvAutoencoder()
        model.load_state_dict(torch.load(require(Path("models") / f"{tag}.pt", "train it first"),
                                         map_location=device)["state_dict"])
        model.to(device).eval()
        ds = OCTDataset(cfg, pick, norm="unit")
        x = torch.stack([ds[k][0] for k in range(len(ds))]).to(device)
        with torch.no_grad():
            out = model(x)
        x_hat = out[0] if isinstance(out, tuple) else out
        res = ((x - x_hat) ** 2)[:, 0].cpu().numpy()
        return x[:, 0].cpu().numpy(), np.stack([ndimage.gaussian_filter(r, 2) for r in res])
    # svdd_*: layer3 distance map of the frozen encoder (as in 08)
    from nets import ResNetFeatures
    centers = torch.load(require(Path("models") / "svdd_centers.pt", "run 07 first"), map_location=device)
    net = ResNetFeatures().to(device).eval()
    ds = OCTDataset(cfg, pick, norm="imagenet")
    x = torch.stack([ds[k][0] for k in range(len(ds))]).to(device)
    with torch.no_grad():
        l3, _ = net(x)
        d = ((l3 - centers["layer3_mean"][None]) ** 2).sum(1)[:, None]
        d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    mean, std = torch.tensor(IMAGENET_MEAN).view(3, 1, 1), torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x.cpu() * std + mean)[:, 0].numpy(), d.cpu().numpy()


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    log = setup_logging("11_visualize_results", cfg)
    out = Path(cfg["output_dir"])
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab = pd.read_csv(require(out / "metrics" / "comparison_table.csv", "Run 10_compare_models.py first."))
    methods = list(dict.fromkeys(tab.method))
    color = {m: METHOD_COLORS[k % len(METHOD_COLORS)] for k, m in enumerate(methods)}
    scores = {m: pd.read_csv(out / "scores" / f"{m}_scores.csv") for m in methods}
    tests = {m: s[s.split == "test"] for m, s in scores.items()}

    # ------------------------------------------------------ ROC + PR per level
    for level in ("bscan", "case"):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
        for m in methods:
            t = tests[m]
            if level == "bscan":
                y, s = (t.bscan_label != "normal").values, t.score.values
            else:
                c = aggregate_case_scores(t, cfg["case_aggregation"])
                y, s = (c.class_folder != "NORMAL").values, c.score.values
            r = tab[(tab.method == m) & (tab.level == level)].iloc[0]
            fpr, tpr, _ = roc_curve(y, s)
            prec, rec, _ = precision_recall_curve(y, s)
            a1.plot(fpr, tpr, color=color[m], linewidth=2, label=f"{NAMES.get(m, m)}  {r.roc_auc:.3f}")
            a2.plot(rec, prec, color=color[m], linewidth=2, label=f"{NAMES.get(m, m)}  {r.pr_auc:.3f}")
        a1.plot([0, 1], [0, 1], color=GRID, linewidth=1, linestyle="--")
        a2.axhline(y.mean(), color=GRID, linewidth=1, linestyle="--")
        a1.set(xlabel="false positive rate (1 - specificity)", ylabel="true positive rate (sensitivity)")
        a2.set(xlabel="recall (sensitivity)", ylabel="precision", ylim=(0, 1.02))
        a1.set_title("ROC (legend: ROC-AUC)", fontsize=10, loc="left")
        a2.set_title("Precision-recall (legend: PR-AUC)", fontsize=10, loc="left")
        for a in (a1, a2):
            style(a)
            a.legend(frameon=False, fontsize=7.5, loc="lower right" if a is a1 else "lower left")
        fig.suptitle(f"Test set, {'B-scan' if level == 'bscan' else 'case (patient)'} level", x=0.02, ha="left",
                     fontsize=11)
        fig.tight_layout()
        save(fig, fig_dir / f"roc_pr_{level}.png")

    # ------------------------------------------------ score distributions
    fig, axes = plt.subplots(1, len(methods), figsize=(3.4 * len(methods), 3), squeeze=False)
    for ax, m in zip(axes[0], methods):
        t = tests[m]
        lo, hi = np.percentile(t.score, [0.5, 99.5])
        bins = np.linspace(lo, hi, 50)
        for lab in ("normal", "drusen", "cnv"):
            ax.hist(t.score[t.bscan_label == lab].clip(lo, hi), bins=bins, density=True, histtype="step", linewidth=2,
                    color=LABEL_COLORS[lab], label=f"{lab} (n={int((t.bscan_label == lab).sum()):,})")
        thr = tab[(tab.method == m) & (tab.level == "bscan")].threshold_primary.iloc[0]
        ax.axvline(thr, color=INK2, linewidth=1, linestyle="--")
        ax.set_title(NAMES.get(m, m), fontsize=9, loc="left")
        ax.set_xlabel("anomaly score")
        style(ax, "y")
    axes[0, 0].set_ylabel("density")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Test B-scan scores by specialist label (dashed: primary threshold)", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, fig_dir / "score_distributions.png")

    # --------------------------------------------------- confusion matrices
    fig, axes = plt.subplots(2, len(methods), figsize=(2.6 * len(methods), 5.2), squeeze=False)
    for j, m in enumerate(methods):
        for i, level in enumerate(("bscan", "case")):
            cm = pd.read_csv(out / "metrics" / f"{m}_confusion_{level}.csv").set_index("threshold").loc["primary"]
            mat = np.array([[cm.TN, cm.FP], [cm.FN, cm.TP]], dtype=float)
            ax = axes[i, j]
            ax.imshow(mat / mat.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
            for (r, c), v in np.ndenumerate(mat):
                share = v / mat[r].sum()
                ax.text(c, r, f"{int(v):,}\n{share:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if share > 0.6 else "#0b0b0b")
            ax.set_xticks([0, 1], ["pred normal", "pred abnormal"], fontsize=7)
            ax.set_yticks([0, 1], ["normal", "abnormal"], fontsize=7)
            ax.tick_params(length=0)
            ax.set_title(f"{NAMES.get(m, m)}\n{'B-scan' if level == 'bscan' else 'case'}", fontsize=8, loc="left")
    fig.suptitle("Confusion matrices at the primary threshold (row-normalised shading)", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, fig_dir / "confusion_matrices.png")

    # ------------------------------------------------ best-method heatmaps
    best = tab[tab.level == "bscan"].sort_values("pr_auc", ascending=False).method.iloc[0]
    device = get_device(cfg["device"])
    t = tests[best]
    rng = np.random.default_rng(cfg["seed"])
    pick = pd.concat([t[(t.class_folder == c) & ((t.bscan_label == "normal") if c == "NORMAL" else
                                                 (t.bscan_label != "normal"))].sample(3, random_state=int(rng.integers(1e6)))
                      for c in CLASSES])
    imgs, maps = anomaly_maps(best, pick, cfg, device)
    vmin, vmax = np.percentile(maps, [5, 99.5])
    fig, axes = plt.subplots(2, 9, figsize=(17, 4.2))
    for k in range(9):
        axes[0, k].imshow(imgs[k], cmap="gray", vmin=0, vmax=1)
        axes[1, k].imshow(imgs[k], cmap="gray", vmin=0, vmax=1)
        axes[1, k].imshow(maps[k], cmap="inferno", alpha=0.5, vmin=vmin, vmax=vmax)
        axes[0, k].set_title(f"{pick.class_folder.iloc[k]} / {pick.bscan_label.iloc[k]}\nscore {pick.score.iloc[k]:.4g}",
                             fontsize=7)
        axes[0, k].axis("off")
        axes[1, k].axis("off")
    fig.suptitle(f"Best method by B-scan PR-AUC: {NAMES.get(best, best)} - test B-scans and anomaly maps",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, fig_dir / "best_method_heatmaps.png")
    log.info(f"figures -> {fig_dir} (best method: {best})")


if __name__ == "__main__":
    main()
