"""mid_review_val_summary.py - validation-only metrics and ROC figure for the mid-review (PRD section 9).

Uses only split == "val" rows of the score files; the test set is not read. Threshold = 95th percentile of
val clean-normal scores (the protocol's primary threshold), so accuracy / precision / recall / F1 are at a
fixed ~95% specificity on clean-normal B-scans.

Outputs: outputs/mid_review/val_metrics.{csv,md}, outputs/mid_review/val_roc.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from common import aggregate_case_scores, load_config

METHODS = {"ae": "M1 Autoencoder", "svdd_frozen": "M2 Deep SVDD (frozen)", "svdd_ft": "M2 Deep SVDD (fine-tuned)"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def metrics(y, s, thr):
    p = s >= thr
    tp, fp, fn, tn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum(), (~p & ~y).sum()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s), accuracy=(tp + tn) / len(y),
                precision=prec, recall=rec, specificity=tn / max(tn + fp, 1),
                f1=2 * prec * rec / max(prec + rec, 1e-12))


def main():
    cfg = load_config()
    out = Path(cfg["output_dir"]) / "mid_review"
    out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for m, name in METHODS.items():
        d = pd.read_csv(Path(cfg["output_dir"]) / "scores" / f"{m}_scores.csv")
        v = d[d.split == "val"]
        clean = v[(v.class_folder == "NORMAL") & (v.bscan_label == "normal")]
        thr_b = np.percentile(clean.score, cfg["threshold_percentile"])
        yb = (v.bscan_label != "normal").values
        rows.append(dict(method=name, level="B-scan", n=len(v), **metrics(yb, v.score.values, thr_b)))
        c = aggregate_case_scores(v, cfg["case_aggregation"])
        yc = (c.class_folder != "NORMAL").values
        thr_c = np.percentile(c.score[~yc], cfg["threshold_percentile"])
        rows.append(dict(method=name, level="patient", n=len(c), **metrics(yc, c.score.values, thr_c)))
        curves[name] = roc_curve(yb, v.score.values)[:2]
    t = pd.DataFrame(rows)
    t.to_csv(out / "val_metrics.csv", index=False)
    cols = list(t.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + " |" for r in t.itertuples(index=False)]
    (out / "val_metrics.md").write_text(
        "# Validation metrics (mid-review, test set untouched)\n\nThreshold: 95th percentile of val clean-normal "
        "scores (B-scan) / val NORMAL patients' scores (patient).\n\n" + "\n".join(md), encoding="utf-8")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    fig, ax = plt.subplots(figsize=(5, 4.4))
    for (name, (fpr, tpr)), col in zip(curves.items(), COLORS):
        auc = t[(t.method == name) & (t.level == "B-scan")].roc_auc.iloc[0]
        ax.plot(fpr, tpr, color=col, linewidth=2, label=f"{name}  AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#c3c2b7", linewidth=1, linestyle="--")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate (recall)")
    ax.set_title("Validation ROC, B-scan level (n = 3,674)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#e4e3df", linewidth=0.6)
    fig.savefig(out / "val_roc.png", dpi=200, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
